import argparse
import json
import os
import yaml
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(costs_path, processed_path, model_out_path, metrics_out_path):
    P = load_params()
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)

    print("Loading and merging data...")
    # M3 features (engineered *_Est, quantities, etc.)
    features_df = pd.read_parquet(costs_path)

    # Ground-truth comes from the processed (tab) table
    target_source_df = pd.read_parquet(processed_path)
    target_df = target_source_df[["Row_ID", "Grand_Total"]].copy()
    target_df.rename(columns={"Grand_Total": "Grand_Total_Target"}, inplace=True)
    TARGET = "Grand_Total_Target"

    # Join on Row_ID
    df = pd.merge(features_df, target_df, on="Row_ID", how="inner")
    df.dropna(subset=[TARGET], inplace=True)
    print(f"Data prepared with {len(df)} rows.")

    # ------------------------------------------------------------------
    # Prevent LABEL LEAKAGE:
    #  - Remove all ground-truth *_Cost columns from the raw table
    #  - Remove true totals and per-sqft labels
    #  - Keep engineered *_Est columns from M3 (safe to learn from)
    # ------------------------------------------------------------------
    BASE_EXCLUDE = [
        "Row_ID", TARGET,
        "Grand_Total", "Grand_Total_Tab", "Grand_Total_Quantity", "Grand_Total_Fused",
        "Cost_per_Sqft_Tab", "Total_Cost_per_Sqft_Q", "Cost_per_Sqft_Fused",
    ]

    LEAK_EXACT = [
        "Painting_Material_Cost","Painting_Labor_Cost",
        "Flooring_Material_Cost","Flooring_Labor_Cost",
        "Ceiling_Material_Cost","Ceiling_Labor_Cost",
        "Electrical_Material_Cost","Electrical_Labor_Cost",
        "Kitchen_Package_Cost","Bathroom_Package_Cost",
        "Plumbing_Cost","Furniture_Cost",
        "Wastage_Sundries_Cost","Contractor_Overhead_Cost",
        "GST_Amount","Total_Cost_per_Sqft",
    ]

    all_cols = list(df.columns)
    # Any *_Cost that is NOT an engineered *_Est should be excluded
    leak_cols_auto = [c for c in all_cols if c.endswith("_Cost") and not c.endswith("_Est")]

    EXCLUDE_COLS = list(dict.fromkeys(BASE_EXCLUDE + LEAK_EXACT + leak_cols_auto))

    features = [c for c in all_cols if c not in EXCLUDE_COLS]
    X = df[features]
    y = df[TARGET]

    # Split numeric vs categorical (treat bool as categorical to be safe)
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    print(f"Training with {len(features)} features "
          f"({len(numerical_features)} numeric, {len(categorical_features)} categorical).")

    # Build pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numerical_features)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model_params = P.get("train", {}).get("model", {})
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", lgb.LGBMRegressor(
            random_state=P["train"]["split"]["random_state"],
            **model_params
        )),
    ])

    split_params = P["train"]["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_params["test_size"], random_state=split_params["random_state"]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    print("Saving new model and metrics...")
    joblib.dump(pipeline, model_out_path)
    metrics = {"rmse": rmse, "mae": mae, "r2_score": r2, "n_rows": int(len(df))}
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--costs", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    main(args.costs, args.processed, args.model_out, args.metrics_out)
