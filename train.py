"""
train.py — Train RandomForest + DistilBERT pipeline for soil NO3 prediction.
Saves all artifacts needed for inference to ./model/

Usage:
    python train.py --data new_NO3Field.csv --output ./model
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import DistilBertModel, DistilBertTokenizer

NUM_COLS = ["SRadiation", "Prec", "WD", "WS", "AT", "WC", "ST", "RH ", "pH"]
CAT_COLS = ["Crop", "Treatment", "Site Name"]
TEXT_COL = "Notes"
TARGET   = "NO3"


# ── BERT embeddings ───────────────────────────────────────────────────────────
def get_bert_embeddings(text_list, tokenizer, bert_model, batch_size=32):
    embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch  = text_list[i : i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            out = bert_model(**inputs)
            embeddings.append(out.last_hidden_state[:, 0, :].numpy())
    return np.vstack(embeddings)


def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load & clean ──────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=NUM_COLS + [TARGET]).copy()

    # ── 2. BERT embeddings on Notes text ────────────────────────────────────
    print("Loading DistilBERT...")
    tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

    df["combined_text"] = df[[TEXT_COL]].astype(str).agg(" ".join, axis=1)
    print("Generating BERT embeddings...")
    raw_embeddings = get_bert_embeddings(
        df["combined_text"].tolist(), tokenizer, bert_model
    )

    pca = PCA(n_components=2)
    text_embeddings = pca.fit_transform(raw_embeddings)
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    # ── 3. One-hot encode categorical columns ────────────────────────────────
    df_dummies = pd.get_dummies(df[CAT_COLS], drop_first=False)
    dummy_columns = list(df_dummies.columns)

    X = np.hstack([df[NUM_COLS].values, df_dummies.values, text_embeddings])
    y = df[TARGET].values

    # ── 4. Scale & split ─────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    # ── 5. Train Random Forest ───────────────────────────────────────────────
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(
        f"R²={r2_score(y_test, y_pred):.4f}  "
        f"RMSE={np.sqrt(mean_squared_error(y_test, y_pred)):.4f}  "
        f"MAE={mean_absolute_error(y_test, y_pred):.4f}"
    )

    # ── 6. Save artifacts ────────────────────────────────────────────────────
    joblib.dump(rf,     os.path.join(output_dir, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(pca,    os.path.join(output_dir, "pca.joblib"))

    with open(os.path.join(output_dir, "dummy_columns.json"), "w") as f:
        json.dump(dummy_columns, f)

    # Save WC physics-correction threshold (75th pct of training WC)
    wc_idx       = NUM_COLS.index("WC")
    wc_train     = X_train[:, wc_idx]            # already scaled; store raw
    wc_raw_train = df[NUM_COLS].values[: len(X_train), wc_idx]
    wc_threshold = float(np.nanquantile(wc_raw_train, 0.75))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(
            {
                "num_cols"     : NUM_COLS,
                "cat_cols"     : CAT_COLS,
                "text_col"     : TEXT_COL,
                "target"       : TARGET,
                "wc_threshold" : wc_threshold,
                "pca_components": 2,
            },
            f,
            indent=2,
        )

    print(f"All artifacts saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="new_NO3Field.csv")
    parser.add_argument("--output", default="./model")
    args = parser.parse_args()
    main(args.data, args.output)
