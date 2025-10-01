# src/model/batch_predict.py
import argparse, os, joblib, pandas as pd, numpy as np

EXCLUDE_COLS = [
    "Row_ID","Grand_Total","Grand_Total_Tab","Grand_Total_Quantity","Grand_Total_Fused",
    "Cost_per_Sqft_Tab","Total_Cost_per_Sqft_Q","Cost_per_Sqft_Fused","Grand_Total_Target"
]

def series_or_default(df, col, default):
    return df[col] if col in df.columns else pd.Series([default]*len(df), index=df.index)

def main(model_path, features_path, out_path, preview_csv="", preview_n=200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if preview_csv:
        os.makedirs(os.path.dirname(preview_csv), exist_ok=True)

    pipe = joblib.load(model_path)
    df = pd.read_parquet(features_path).copy()

    features = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[features].copy()

    # Predict Grand_Total
    y_pred = pipe.predict(X)
    pred = df[["Row_ID"]].copy()
    pred["Grand_Total_ML"] = y_pred

    # Per-sqft with robust area fallback
    area = pd.to_numeric(series_or_default(df, "Area_Sqft", 0), errors="coerce").fillna(0)
    floor_area = pd.to_numeric(series_or_default(df, "flooring_area_sqft", 0), errors="coerce").fillna(0)
    area = area.mask(area <= 0, floor_area)

    with np.errstate(divide="ignore", invalid="ignore"):
        pred["Cost_per_Sqft_ML"] = np.where(area > 0, pred["Grand_Total_ML"] / area, np.nan)

    pred.to_parquet(out_path, index=False)
    if preview_csv:
        pred.head(int(preview_n)).to_csv(preview_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)   # usually data/processed/cost_breakdown.parquet
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview-csv", default="")
    ap.add_argument("--preview-n", type=int, default=200)
    args = ap.parse_args()
    main(args.model, args.features, args.out, args.preview_csv, args.preview_n)
