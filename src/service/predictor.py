# src/service/predictor.py
"""
CostEstimator service wrapper
- Loads your trained sklearn Pipeline (preprocessor + LightGBM)
- Uses your existing create_feature_dataframe() (params.yaml-backed)
- Aligns the input schema to the model's expected columns, coerces dtypes,
  and handles *_x / *_y suffixes to avoid missing-column & isnan errors.
"""

import os
import sys
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set

# -------------------------------------------------------------------
# Ensure project root in sys.path (…/src/service -> …/)
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------
# Import your feature builder (you said it's at src/inference/predict.py)
# -------------------------------------------------------------------
try:
    from src.inference.predict import create_feature_dataframe
except ModuleNotFoundError:
    try:
        from src.predict import create_feature_dataframe
    except ModuleNotFoundError:
        try:
            from predict import create_feature_dataframe
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Could not import create_feature_dataframe. "
                "Expected at one of: "
                "src/inference/predict.py (src.inference.predict), "
                "src/predict.py (src.predict), or repo root predict.py (predict)."
            ) from e


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def load_params(params_path: str = "params.yaml") -> Dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def exclude_leak_columns(cols) -> set:
    """
    Exclude the same columns as training/inference to prevent label leakage.
    - removes *_Cost (but keeps *_Est engineered features)
    - removes true totals/psf labels
    """
    EXCLUDE_BASE = {
        "Row_ID",
        "Grand_Total", "Grand_Total_Tab", "Grand_Total_Quantity", "Grand_Total_Fused",
        "Cost_per_Sqft_Tab", "Total_Cost_per_Sqft_Q", "Cost_per_Sqft_Fused",
        "Grand_Total_Target",
    }
    LEAK_EXACT = {
        "Painting_Material_Cost","Painting_Labor_Cost",
        "Flooring_Material_Cost","Flooring_Labor_Cost",
        "Ceiling_Material_Cost","Ceiling_Labor_Cost",
        "Electrical_Material_Cost","Electrical_Labor_Cost",
        "Kitchen_Package_Cost","Bathroom_Package_Cost",
        "Plumbing_Cost","Furniture_Cost",
        "Wastage_Sundries_Cost","Contractor_Overhead_Cost",
        "GST_Amount","Total_Cost_per_Sqft",
    }
    leak_auto = {c for c in cols if c.endswith("_Cost") and not c.endswith("_Est")}
    return set(EXCLUDE_BASE) | set(LEAK_EXACT) | leak_auto


def series_or_default(df: pd.DataFrame, col: str, default):
    return df[col] if col in df.columns else pd.Series([default] * len(df), index=df.index)


def _expected_columns_from_preprocessor(pipe) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Introspect the trained pipeline to find which input columns it expects.
    Returns: (all_expected, numeric_expected, categorical_expected)
    """
    all_expected, num_expected, cat_expected = set(), set(), set()
    pre = getattr(pipe, "named_steps", {}).get("preprocessor", None)

    if pre is not None and hasattr(pre, "transformers_"):
        for name, trans, cols in pre.transformers_:
            if cols is None:
                continue
            cols_list = list(cols) if isinstance(cols, (list, tuple, np.ndarray, pd.Index)) else [cols]
            all_expected.update(cols_list)
            if name == "num":
                num_expected.update(cols_list)
            elif name == "cat":
                cat_expected.update(cols_list)

    # Fallback: if nothing captured (unlikely), try feature_names_in_
    if not all_expected and hasattr(pipe, "feature_names_in_"):
        all_expected = set(pipe.feature_names_in_)

    return all_expected, num_expected, cat_expected


def _align_columns_for_model(X: pd.DataFrame, pipe) -> pd.DataFrame:
    """
    Ensure X has exactly the columns the trained model expects:
    - If model expects *_x / *_y columns and only base exists, copy base → expected.
    - If model expects base and we only have *_x or *_y, copy that to base.
    - Coerce all expected numeric columns to float (errors→NaN→0.0).
    - Fill expected categorical NaNs with "".
    - Drop any extra columns not expected by the preprocessor (prevents passthrough surprises).
    """
    exp_all, exp_num, exp_cat = _expected_columns_from_preprocessor(pipe)

    # If we can't infer, return as-is (but this should almost never happen)
    if not exp_all:
        return X

    X = X.copy()

    # 1) Create missing expected columns from base or suffixed variants where possible
    for c in list(exp_all):
        if c in X.columns:
            continue

        # expected *_x / *_y → copy base if present
        if c.endswith("_x") or c.endswith("_y"):
            base = c[:-2]
            if base in X.columns:
                X[c] = X[base]
                continue

        # expected base → copy *_x or *_y if present
        if f"{c}_x" in X.columns:
            X[c] = X[f"{c}_x"]
            continue
        if f"{c}_y" in X.columns:
            X[c] = X[f"{c}_y"]
            continue

        # otherwise, init by type
        if c in exp_num:
            X[c] = 0.0
        elif c in exp_cat:
            X[c] = ""
        else:
            X[c] = ""

    # 2) Keep ONLY what the preprocessor expects (avoid odd passthrough objects)
    X = X[[c for c in X.columns if c in exp_all]]

    # 3) Coerce numeric columns to float (prevents np.isnan errors inside StandardScaler)
    for c in exp_num:
        # If absent (shouldn't be now), create and fill 0.0
        if c not in X.columns:
            X[c] = 0.0
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 4) Categorical: ensure no NaNs; leave bools as bools; otherwise fill empty strings
    for c in exp_cat:
        if c not in X.columns:
            X[c] = ""
        # preserve bool dtype if already bool; else fillna("")
        if X[c].dtype != bool:
            X[c] = X[c].fillna("").astype(object)

    # 5) Final numeric NaNs → 0.0 (after coercion)
    if exp_num:
        X[list(exp_num)] = X[list(exp_num)].fillna(0.0)

    return X


# -------------------------------------------------------------------
# Main service class
# -------------------------------------------------------------------
class CostEstimator:
    """
    Wraps your trained sklearn Pipeline (preprocessor + LightGBM)
    and provides single/batch prediction using the same feature policy
    used in training (no leakage). Also aligns columns/dtypes to the
    model’s expected schema (handles _x/_y & numeric coercion).
    """

    def __init__(self, model_path: str, params_path: str = "params.yaml"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.pipe = joblib.load(model_path)
        self.P = load_params(params_path)

    def _prepare_features(self, df_features: pd.DataFrame) -> pd.DataFrame:
        # 1) Drop leak columns (same policy as training)
        exclude = exclude_leak_columns(df_features.columns)
        X = df_features[[c for c in df_features.columns if c not in exclude]].copy()
        # 2) Align to model’s expected schema & coerce dtypes
        X = _align_columns_for_model(X, self.pipe)
        return X

    def predict_from_raw_rows(self, rows: List[Dict]) -> pd.DataFrame:
        """
        rows: list of raw input dicts with keys like:
          City, Room_Type, Area_Sqft, Paint_Quality, Floor_Type, Floor_Quality,
          Ceiling_Type, Ceiling_Quality, Has_Electrical, Kitchen_Package, Bathroom_Package,
          (optional) Material_Price_Index, City_Multiplier, City_Tier, etc.

        Returns a DataFrame with engineered features + predictions.
        """
        feats_list = []
        for r in rows:
            df = create_feature_dataframe(r)  # uses params.yaml; builds quantities + *_Est
            feats_list.append(df)
        feats = pd.concat(feats_list, ignore_index=True)

        X = self._prepare_features(feats)
        yhat = self.pipe.predict(X)

        out = feats.copy()
        out["Grand_Total_ML"] = yhat

        area = pd.to_numeric(series_or_default(out, "Area_Sqft", 0), errors="coerce").fillna(0)
        out["Cost_per_Sqft_ML"] = np.where(area > 0, out["Grand_Total_ML"] / area, np.nan)
        return out

    def predict_from_dataframe(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        rows = df_raw.fillna("").to_dict(orient="records")
        return self.predict_from_raw_rows(rows)


# -------------------------------------------------------------------
# Optional: quick manual test (run this file directly)
# -------------------------------------------------------------------
if __name__ == "__main__":
    MODEL = "models/lgbm.pkl" if os.path.exists("models/lgbm.pkl") else "models/model.joblib"
    est = CostEstimator(model_path=MODEL, params_path="params.yaml")
    sample = [{
        "City": "Bangalore",
        "City_Tier": "Tier1",
        "City_Multiplier": 1.0,
        "Material_Price_Index": 1.0,
        "Room_Type": "Bedroom",
        "Area_Sqft": 120.0,
        "Paint_Quality": "Economy",
        "Floor_Type": "Ceramic_Tile",
        "Floor_Quality": "Economy",
        "Ceiling_Type": "POP",
        "Ceiling_Quality": "Economy",
        "Has_Electrical": True,
        "Kitchen_Package": "None",
        "Bathroom_Package": "None",
    }]
    df_out = est.predict_from_raw_rows(sample)
    print(json.dumps(df_out[["Grand_Total_ML", "Cost_per_Sqft_ML"]].iloc[0].to_dict(), indent=2))
