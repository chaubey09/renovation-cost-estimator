# src/service/ml_utils.py
from __future__ import annotations
from typing import List, Iterable
import numpy as np, pandas as pd

def to_number(x, default=0.0) -> float:
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0]) if not x.empty else float(default)
        if isinstance(x, (np.ndarray, list, tuple, pd.Index)):
            arr = np.asarray(x).ravel()
            return float(arr[0]) if arr.size else float(default)
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        try:
            v = pd.to_numeric(x, errors="coerce")
            if isinstance(v, pd.Series):
                v = v.dropna()
                return float(v.iloc[0]) if len(v) else float(default)
            return float(v) if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

def rupees(x) -> str:
    v = to_number(x, default=np.nan)
    return "—" if not (isinstance(v, float) and np.isfinite(v)) else f"₹{v:,.0f}"

def dedupe_preserve_order(seq: Iterable):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def get_expected_input_cols(pipe):
    cols = getattr(pipe, "feature_names_in_", None)
    if cols is not None: return list(cols)
    try:
        pre = pipe.named_steps.get("preprocessor")
        cols = getattr(pre, "feature_names_in_", None)
        if cols is not None: return list(cols)
    except Exception:
        pass
    return None

_SUFFIX_BASES = ["Area_Sqft","Paint_Quality","Floor_Type","Floor_Quality","Ceiling_Type","Ceiling_Quality","Has_Electrical"]

def mirror_suffix_columns(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    if not expected_cols: return df
    df = df.copy()
    for base in _SUFFIX_BASES:
        if f"{base}_x" in expected_cols and f"{base}_x" not in df.columns and base in df.columns:
            df[f"{base}_x"] = df[base]
        if f"{base}_y" in expected_cols and f"{base}_y" not in df.columns and base in df.columns:
            df[f"{base}_y"] = df[base]
    return df

def align_to_model(Xcand: pd.DataFrame, expected_cols: List[str] | None) -> pd.DataFrame:
    if expected_cols is None:
        return Xcand.loc[:, ~Xcand.columns.duplicated()].copy()
    exp = dedupe_preserve_order([str(c) for c in expected_cols])
    X = Xcand.loc[:, ~Xcand.columns.duplicated()].copy()
    for c in exp:
        if c not in X.columns: X[c] = 0
    return X.reindex(columns=exp)
