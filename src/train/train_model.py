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
    features_df = pd.read_parquet(costs_path)
    
    target_source_df = pd.read_parquet(processed_path)
    target_df = target_source_df[["Row_ID", "Grand_Total"]].copy()
    
    target_df.rename(columns={"Grand_Total": "Grand_Total_Target"}, inplace=True)
    TARGET = "Grand_Total_Target"

    df = pd.merge(features_df, target_df, on="Row_ID", how="inner")
    df.dropna(subset=[TARGET], inplace=True)
    print(f"Data prepared with {len(df)} rows.")

    EXCLUDE_COLS = [
        "Row_ID", TARGET, "Grand_Total", "Grand_Total_Tab", "Grand_Total_Quantity", "Grand_Total_Fused",
        "Cost_per_Sqft_Tab", "Total_Cost_per_Sqft_Q", "Cost_per_Sqft_Fused"
    ]
    features = [col for col in df.columns if col not in EXCLUDE_COLS]
    X = df[features]
    y = df[TARGET]

    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    print(f"Training with {len(features)} clean features.")

    print("Building model pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model_params = P.get("train", {}).get("model", {})
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=P["train"]["split"]["random_state"], **model_params))
    ])

    split_params = P["train"]["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_params["test_size"], random_state=split_params["random_state"]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    print("Saving new model and metrics...")
    joblib.dump(pipeline, model_out_path)
    metrics = { "rmse": rmse, "mae": mae, "r2_score": r2 }
    with open(metrics_out_path, "w") as f:
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