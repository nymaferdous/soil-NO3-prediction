"""
app.py — FastAPI inference server for soil NO3 prediction.
Usage: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertModel, DistilBertTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "./model")

app = FastAPI(title="Soil NO3 Prediction API", version="1.0.0")

# ── Global state loaded at startup ───────────────────────────────────────────
rf      = None
scaler  = None
pca     = None
config  = None
dummy_columns   = None
tokenizer       = None
bert_model      = None


@app.on_event("startup")
def load_artifacts():
    global rf, scaler, pca, config, dummy_columns, tokenizer, bert_model

    rf     = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    pca    = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))

    with open(os.path.join(MODEL_DIR, "dummy_columns.json")) as f:
        dummy_columns = json.load(f)

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        config = json.load(f)

    tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    bert_model.eval()


# ── Request / Response schemas ───────────────────────────────────────────────
class PredictionRequest(BaseModel):
    SRadiation: float
    Prec:       float
    WD:         float
    WS:         float
    AT:         float
    WC:         float
    ST:         float
    RH:         float   # maps to "RH " column
    pH:         float
    Crop:       str
    Treatment:  str
    site_name:  str     # maps to "Site Name"
    notes:      str = ""


class PredictionResponse(BaseModel):
    predicted_NO3: float
    wc_flag: bool        # True if WC is above physics threshold (high-WC regime)
    wc_threshold: float


# ── Helper: build features from a single request ─────────────────────────────
def build_features(req: PredictionRequest) -> np.ndarray:
    num_vals = [
        req.SRadiation, req.Prec, req.WD, req.WS,
        req.AT, req.WC, req.ST, req.RH, req.pH,
    ]

    # One-hot encode categorical features
    row = {"Crop": req.Crop, "Treatment": req.Treatment, "Site Name": req.site_name}
    df_row = pd.DataFrame([row])
    df_dummies = pd.get_dummies(df_row, columns=["Crop", "Treatment", "Site Name"])

    # Align to training dummy columns
    for col in dummy_columns:
        if col not in df_dummies.columns:
            df_dummies[col] = 0
    df_dummies = df_dummies[dummy_columns]

    # BERT embedding on notes text
    inputs = tokenizer(
        req.notes, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        out = bert_model(**inputs)
    raw_emb  = out.last_hidden_state[:, 0, :].numpy()
    text_emb = pca.transform(raw_emb)   # shape (1, 2)

    X = np.hstack([
        np.array(num_vals).reshape(1, -1),
        df_dummies.values,
        text_emb,
    ])
    return scaler.transform(X)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        X = build_features(req)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    no3_pred     = float(rf.predict(X)[0])
    wc_threshold = config.get("wc_threshold", 0.0)
    wc_flag      = req.WC > wc_threshold

    return PredictionResponse(
        predicted_NO3=round(no3_pred, 4),
        wc_flag=wc_flag,
        wc_threshold=round(wc_threshold, 4),
    )
