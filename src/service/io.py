# src/service/io.py
from __future__ import annotations
import os, yaml, joblib, pandas as pd
from typing import List, Optional

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model(path: str = "models/lgbm.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def load_cost_indices(path: str = "data/external/cost_indices.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["City","City_Multiplier","Material_Price_Index","City_Tier"])
    df = pd.read_csv(path)
    rename = {"city":"City","city_multiplier":"City_Multiplier","material_price_index":"Material_Price_Index","city_tier":"City_Tier"}
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns})
    for c in ("City_Multiplier","Material_Price_Index"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_all_cities(processed_pq="data/processed/processed.parquet",
                    raw_csv="data/raw/renovation_cost_dataset_india_v2.csv",
                    cost_idx_csv="data/external/cost_indices.csv") -> List[str]:
    cities, seen = [], set()
    def add(series: Optional[pd.Series]):
        if series is None: return
        for c in series.dropna().astype(str).str.strip():
            if c and c not in seen:
                seen.add(c); cities.append(c)
    try:
        if os.path.exists(processed_pq):
            add(pd.read_parquet(processed_pq, columns=["City"])["City"])
    except Exception: pass
    try:
        if os.path.exists(raw_csv):
            add(pd.read_csv(raw_csv, usecols=["City"])["City"])
    except Exception: pass
    try:
        if os.path.exists(cost_idx_csv):
            ci = pd.read_csv(cost_idx_csv)
            ci = ci.rename(columns={"city":"City"})
            if "City" in ci.columns: add(ci["City"])
    except Exception: pass
    return sorted(cities)
