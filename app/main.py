from __future__ import annotations
import os, math, logging
from typing import List, Optional, Dict, Any

import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.service.io import load_params, load_model, load_cost_indices, load_all_cities
from src.service.ml_utils import to_number, rupees, get_expected_input_cols, mirror_suffix_columns, align_to_model
from src.service.features import compute_engineered_features, lookup_city_index
from src.service.calc import calculate_realistic_quantity_cost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("renovation-api")

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm.pkl")
PARAMS_PATH       = os.getenv("PARAMS_PATH", "params.yaml")
COST_INDEX_PATH   = os.getenv("COST_INDEX_PATH", "data/external/cost_indices.csv")
PROCESSED_PARQUET = os.getenv("PROCESSED_PARQUET", "data/processed/processed.parquet")
RAW_CSV           = os.getenv("RAW_CSV", "data/raw/renovation_cost_dataset_india_v2.csv")
MODEL_TARGET      = os.getenv("MODEL_TARGET", "total").lower()  # "total" | "per_sqft"

SERVICE_NAME = os.getenv("SERVICE_NAME", "renovation-estimator")
SERVICE_STATUS = {"service": SERVICE_NAME, "status": "ok", "docs": "/docs", "openapi": "/openapi.json"}

def dedupe_preserve_order(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

def load_state():
    P = load_params(PARAMS_PATH)
    model = load_model(DEFAULT_MODEL_PATH)
    expected = get_expected_input_cols(model)
    ci_df = load_cost_indices(COST_INDEX_PATH)
    cities = load_all_cities(PROCESSED_PARQUET, RAW_CSV, COST_INDEX_PATH)
    return {"params": P, "model": model, "expected": expected, "cost_idx": ci_df, "cities": cities}

app_state: Dict[str, Any] = load_state()
app = FastAPI(title="Renovation Cost Estimator API",
              description="LightGBM cost estimates with config/lookup support.",
              version="1.0.0")

class EstimateIn(BaseModel):
    area_sqft: float = Field(..., gt=0)
    city: str
    renovation_level: str
    bedrooms: Optional[int] = Field(3, ge=0)
    bathrooms: Optional[int] = Field(2, ge=0)
    property_type: Optional[str] = "apartment"
    has_electrical: Optional[bool] = True
    ceiling_type: Optional[str] = "None"
    room_type: Optional[str] = "Bedroom"
    floor_type: Optional[str] = None
    kitchen_package: Optional[str] = None
    bathroom_package: Optional[str] = None
    @field_validator("renovation_level")
    @classmethod
    def norm_level(cls, v: str) -> str: return v.capitalize()

@app.get("/", tags=["meta"])
def root(): return SERVICE_STATUS

@app.get("/healthz", tags=["meta"])
def healthz(): return {"status": "healthy"}

@app.get("/meta/cities", tags=["meta"])
def meta_cities(): return {"cities": app_state.get("cities", [])}

def snake_to_model(d: EstimateIn) -> dict:
    j = d.model_dump()
    out = {
        "City": j["city"],
        "Area_Sqft": j["area_sqft"],
        "Renovation_Level": j["renovation_level"],
        "Room_Type": j.get("room_type") or "Bedroom",
        "Has_Electrical": j.get("has_electrical") if j.get("has_electrical") is not None else True,
        "Ceiling_Type": j.get("ceiling_type") or "None",
    }
    if j.get("floor_type"): out["Floor_Type"] = j["floor_type"]
    if j.get("kitchen_package"): out["Kitchen_Package"] = j["kitchen_package"]
    if j.get("bathroom_package"): out["Bathroom_Package"] = j["bathroom_package"]
    return out

@app.post("/estimate", tags=["estimate"])
def estimate(payload: EstimateIn):
    try:
        P      = app_state["params"]
        model  = app_state["model"]
        expect = app_state.get("expected")
        ci_df  = app_state["cost_idx"]

        mini = snake_to_model(payload)
        feats = compute_engineered_features(mini, P, ci_df)
        X = align_to_model(mirror_suffix_columns(feats, expect or []), expect)

        pred = model.predict(X)
        ml_total = to_number(pred, 0.0)
        if MODEL_TARGET == "per_sqft":
            ml_total *= to_number(mini["Area_Sqft"], 0.0)

        base_cost = to_number(feats["Subtotal"].iloc[0], 0.0)
        qty_break = calculate_realistic_quantity_cost(base_cost)
        lo, hi = ml_total * 0.90, ml_total * 1.10
        city_idx = lookup_city_index(mini["City"], ci_df, P)

        return JSONResponse({
            "estimate": ml_total, "currency": "INR", "display": rupees(ml_total),
            "range_low": lo, "range_high": hi,
            "meta": {
                "city": mini["City"], "renovation_level": mini["Renovation_Level"],
                "applied_city_multiplier": city_idx["City_Multiplier"],
                "applied_material_price_index": city_idx["Material_Price_Index"],
                "city_tier": city_idx.get("City_Tier",""),
                "qty_baseline_final": qty_break["final_total_qty"],
            }
        })
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing key in input or config: {ke!s}") from ke
    except Exception as e:
        logger.exception("Estimation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Estimation failed: {e!s}")
