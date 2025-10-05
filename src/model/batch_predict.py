# src/model/batch_predict.py
import argparse
import os
import joblib
import numpy as np
import pandas as pd

def _expected_columns_from_preprocessor(preprocessor):
    """
    Inspect a fitted ColumnTransformer and return (num_cols, cat_cols) it expects.
    Works with our pipeline that names transformers ('num', 'cat').
    """
    num_cols = []
    cat_cols = []
    # fitted attr is transformers_
    for name, transformer, cols in getattr(preprocessor, "transformers_", []):
        # 'cols' can be a list of column names
        if cols is None:
            continue
        if name == "num":
            num_cols.extend(list(cols))
        elif name == "cat":
            cat_cols.extend(list(cols))
        else:
            # Fallback: try to infer by transformer type
            tname = transformer.__class__.__name__.lower()
            if "standardscaler" in tname:
                num_cols.extend(list(cols))
            elif "onehotencoder" in tname:
                cat_cols.extend(list(cols))
    # Preserve order and uniqueness
    num_cols = list(dict.fromkeys(num_cols))
    cat_cols = list(dict.fromkeys(cat_cols))
    return num_cols, cat_cols


def _build_model_input(df: pd.DataFrame, num_cols, cat_cols):
    """
    Build an input DataFrame containing exactly the columns the preprocessor expects.
    - For missing numeric columns: fill with 0.0
    - For missing categorical columns: fill with empty string ""
    """
    X = pd.DataFrame(index=df.index)
    for c in num_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        else:
            X[c] = 0.0
    for c in cat_cols:
        if c in df.columns:
            X[c] = df[c].astype("object")
        else:
            X[c] = ""
    return X


def main(model_path: str, features_path: str, out_path: str, preview_csv: str = "", preview_n: int = 200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if preview_csv:
        os.makedirs(os.path.dirname(preview_csv), exist_ok=True)

    # 1) Load fitted pipeline (preprocessor + regressor)
    pipe = joblib.load(model_path)
    preprocessor = pipe.named_steps["preprocessor"]

    # 2) Read features parquet produced by M3 (cost mapping)
    df = pd.read_parquet(features_path)

    # 3) Determine expected raw input columns (as used during training)
    num_cols, cat_cols = _expected_columns_from_preprocessor(preprocessor)
    expected_cols = num_cols + cat_cols

    # 4) Construct the exact input table the model expects
    X = _build_model_input(df, num_cols, cat_cols)

    # 5) Predict total (Grand_Total_ML)
    y_pred = pipe.predict(X)
    df_out = df.copy()
    df_out["Grand_Total_ML"] = y_pred

    # Optional: compute ML cost per sqft if area is available
    if "Area_Sqft" in df_out.columns:
        area = pd.to_numeric(df_out["Area_Sqft"], errors="coerce").replace(0, np.nan)
        df_out["Cost_per_Sqft_ML"] = df_out["Grand_Total_ML"] / area

    # 6) Save outputs
    df_out.to_parquet(out_path, index=False)

    if preview_csv:
        n = int(preview_n)
        cols_preview = ["Row_ID", "City", "Area_Sqft", "Grand_Total_ML"]
        cols_preview += ["Cost_per_Sqft_ML"] if "Cost_per_Sqft_ML" in df_out.columns else []
        exist = [c for c in cols_preview if c in df_out.columns]
        df_out[exist].head(n).to_csv(preview_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained pipeline (joblib)")
    ap.add_argument("--features", required=True, help="Path to parquet with features (M3 output)")
    ap.add_argument("--out", required=True, help="Where to save predictions parquet")
    ap.add_argument("--preview-csv", default="", help="Optional small CSV preview")
    ap.add_argument("--preview-n", type=int, default=200)
    args = ap.parse_args()
    main(args.model, args.features, args.out, args.preview_csv, args.preview_n)
