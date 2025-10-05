# app/main.py
# FastAPI microservice exposing LightGBM renovation cost estimates.
# - Loads model/params/lookups once on startup
# - Minimal input schema -> engineered features aligned to training
# - Compatible with a model predicting either "total" or "per_sqft"
#   via env: MODEL_TARGET = "total" | "per_sqft"

from __future__ import annotations

import os
import json
import math
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ------------------------------- logging -------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("renovation-api")

# ----------------------------- configuration ---------------------------

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm.pkl")
PARAMS_PATH = os.getenv("PARAMS_PATH", "params.yaml")
COST_INDEX_PATH = os.getenv("COST_INDEX_PATH", "data/external/cost_indices.csv")
PROCESSED_PARQUET = os.getenv("PROCESSED_PARQUET", "data/processed/processed.parquet")
RAW_CSV = os.getenv("RAW_CSV", "data/raw/renovation_cost_dataset_india_v2.csv")

# IMPORTANT: Set to "per_sqft" if your model outputs price-per-sqft.
MODEL_TARGET = os.getenv("MODEL_TARGET", "total").lower()  # "total" | "per_sqft"

SERVICE_NAME = os.getenv("SERVICE_NAME", "renovation-estimator")
SERVICE_STATUS = {"service": SERVICE_NAME, "status": "ok", "docs": "/docs", "openapi": "/openapi.json"}

# ------------------------------- utilities -----------------------------

def to_number(x, default=0.0) -> float:
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0]) if not x.empty else float(default)
        if isinstance(x, (np.ndarray, list, tuple, pd.Index)):
            return float(np.asarray(x).ravel()[0]) if len(x) else float(default)
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        try:
            v = pd.to_numeric(x, errors="coerce")
            if isinstance(v, pd.Series):
                v = v.dropna()
                return float(v.iloc[0]) if len(v) else float(default)
            if pd.isna(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)

def rupees(x: float) -> str:
    v = to_number(x, default=np.nan)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "—"
    return f"₹{v:,.0f}"

def dedupe_preserve_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ------------------------------ data loading ---------------------------

def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_model(path: str):
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found at {path}")
    logger.info("Loading model from %s", path)
    return joblib.load(path)

def load_cost_indices(path: str) -> pd.DataFrame:
    """
    Expected columns: City, City_Multiplier, Material_Price_Index (optional: City_Tier)
    """
    if not os.path.exists(path):
        logger.warning("Cost index file missing at %s; using empty frame", path)
        return pd.DataFrame(columns=["City", "City_Multiplier", "Material_Price_Index", "City_Tier"])
    df = pd.read_csv(path)
    rename_map = {
        "city": "City",
        "city_multiplier": "City_Multiplier",
        "material_price_index": "Material_Price_Index",
        "city_tier": "City_Tier",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    if "City_Multiplier" in df.columns:
        df["City_Multiplier"] = pd.to_numeric(df["City_Multiplier"], errors="coerce")
    if "Material_Price_Index" in df.columns:
        df["Material_Price_Index"] = pd.to_numeric(df["Material_Price_Index"], errors="coerce")
    return df

def load_all_cities(processed_pq: str, raw_csv: str, cost_idx_csv: str) -> List[str]:
    cities: List[str] = []
    seen = set()

    def add(series: Optional[pd.Series]):
        if series is None:
            return
        for c in series.dropna().astype(str).str.strip():
            if c and c not in seen:
                seen.add(c); cities.append(c)

    # processed parquet
    try:
        if os.path.exists(processed_pq):
            dfp = pd.read_parquet(processed_pq, columns=["City"])
            if "City" in dfp.columns:
                add(dfp["City"])
    except Exception:
        pass

    # raw csv
    try:
        if os.path.exists(raw_csv):
            dfr = pd.read_csv(raw_csv, usecols=["City"])
            if "City" in dfr.columns:
                add(dfr["City"])
    except Exception:
        pass

    # cost index
    try:
        if os.path.exists(cost_idx_csv):
            ci = pd.read_csv(cost_idx_csv)
            ci = ci.rename(columns={
                "city": "City",
                "city_multiplier": "City_Multiplier",
                "material_price_index": "Material_Price_Index",
                "city_tier": "City_Tier",
            })
            if "City" in ci.columns:
                add(ci["City"])
    except Exception:
        pass

    return sorted(cities)

# ----------------------- feature engineering helpers -------------------

def get_expected_input_cols(pipe) -> Optional[List[str]]:
    cols = getattr(pipe, "feature_names_in_", None)
    if cols is not None:
        return list(cols)
    try:
        pre = pipe.named_steps.get("preprocessor")
        cols = getattr(pre, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    except Exception:
        pass
    return None

_SUFFIX_BASES = [
    "Area_Sqft","Paint_Quality","Floor_Type","Floor_Quality",
    "Ceiling_Type","Ceiling_Quality","Has_Electrical"
]

def mirror_suffix_columns(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    if not expected_cols:
        return df
    df = df.copy()
    for base in _SUFFIX_BASES:
        if f"{base}_x" in expected_cols and f"{base}_x" not in df.columns and base in df.columns:
            df[f"{base}_x"] = df[base]
        if f"{base}_y" in expected_cols and f"{base}_y" not in df.columns and base in df.columns:
            df[f"{base}_y"] = df[base]
    return df

def align_to_model(Xcand: pd.DataFrame, expected_cols: Optional[List[str]]) -> pd.DataFrame:
    if expected_cols is None:
        return Xcand.loc[:, ~Xcand.columns.duplicated()].copy()
    exp = dedupe_preserve_order([str(c) for c in expected_cols])
    Xcand = Xcand.loc[:, ~Xcand.columns.duplicated()].copy()
    for c in exp:
        if c not in Xcand.columns:
            Xcand[c] = 0
    return Xcand.reindex(columns=exp)

def spec_to_quality(spec: str) -> str:
    s = str(spec).lower()
    if s in {"economy","basic"}: return "Economy"
    if s in {"premium","luxury"}: return "Premium"
    return "Standard"

def lookup_city_index(city: str, ci_df: pd.DataFrame, P: dict) -> dict:
    base_mpi = float(P.get("rates", {}).get("base_mpi", 1.0))
    out = {"Material_Price_Index": base_mpi, "City_Multiplier": 1.0, "City_Tier": ""}
    if ci_df is None or ci_df.empty or not city:
        return out
    hit = ci_df.loc[ci_df["City"].astype(str).str.lower() == str(city).lower()]
    if len(hit):
        if "Material_Price_Index" in hit.columns and not pd.isna(hit["Material_Price_Index"].iloc[0]):
            out["Material_Price_Index"] = float(hit["Material_Price_Index"].iloc[0])
        if "City_Multiplier" in hit.columns and not pd.isna(hit["City_Multiplier"].iloc[0]):
            out["City_Multiplier"] = float(hit["City_Multiplier"].iloc[0])
        if "City_Tier" in hit.columns and not pd.isna(hit["City_Tier"].iloc[0]):
            out["City_Tier"] = str(hit["City_Tier"].iloc[0])
    return out

def calculate_realistic_quantity_cost(base_cost: float) -> dict:
    base_cost = to_number(base_cost, 0.0)
    contractor_margin = 0.15 * base_cost
    wastage_sundries  = 0.05 * base_cost
    contingency       = 0.07 * base_cost
    pre_gst_total = base_cost + contractor_margin + wastage_sundries + contingency
    gst           = 0.18 * pre_gst_total
    final_total   = pre_gst_total + gst
    return {
        "base_cost": float(base_cost),
        "contractor_margin": float(contractor_margin),
        "wastage_sundries": float(wastage_sundries),
        "contingency": float(contingency),
        "pre_gst_total": float(pre_gst_total),
        "gst": float(gst),
        "final_total_qty": float(final_total),
    }

def compute_engineered_features(mini: dict, P: dict, ci_df: pd.DataFrame) -> pd.DataFrame:
    # Parse inputs
    city  = str(mini.get("City", ""))
    area  = to_number(mini.get("Area_Sqft", 0.0), 0.0)
    level = str(mini.get("Renovation_Level", "Economy"))
    room_type = str(mini.get("Room_Type", "Bedroom"))
    has_elec = bool(mini.get("Has_Electrical", True))
    ceil_type = str(mini.get("Ceiling_Type", "None"))
    # Qualities derived from Spec Level (override allowed via mini)
    qual = spec_to_quality(level)
    paint_quality = str(mini.get("Paint_Quality", qual))
    floor_quality = str(mini.get("Floor_Quality", qual))
    ceil_quality  = str(mini.get("Ceiling_Quality", qual))
    floor_type = str(mini.get("Floor_Type", "Ceramic_Tile"))
    kitchen_pkg  = str(mini.get("Kitchen_Package", "None"))
    bathroom_pkg = str(mini.get("Bathroom_Package", "None"))

    # City lookup → MPI, Multiplier, Tier
    city_idx = lookup_city_index(city, ci_df, P)
    city_mult = float(city_idx["City_Multiplier"])
    mpi       = float(city_idx["Material_Price_Index"])
    city_tier = str(city_idx.get("City_Tier", ""))

    # Rules / rates from params
    paint_rules = dict(P["quant_rules"]["paint_wall_multiplier_by_room"])
    ceil_rules  = dict(P["quant_rules"]["false_ceiling_ratio_by_type"])
    elec_base   = dict(P["quant_rules"]["electrical_points_base"])
    elec_scale  = float(P["quant_rules"]["electrical_area_scale"])

    paint_mat_tbl = dict(P["rates"]["painting"]["material_per_sqft_by_quality"])
    floor_tbl     = dict(P["rates"]["flooring"]["material_per_sqft_by_type_quality"])
    ceil_tbl      = dict(P["rates"]["ceiling"]["material_per_sqft_by_type_quality"])
    paint_lab_psf = float(P["rates"]["painting"]["labor_per_sqft"])
    floor_lab_psf = float(P["rates"]["flooring"]["labor_per_sqft"])
    ceil_lab_psf  = float(P["rates"]["ceiling"]["labor_per_sqft"])
    elec_mat_pp   = float(P["rates"]["electrical"]["per_point_material"])
    elec_lab_pp   = float(P["rates"]["electrical"]["per_point_labor"])
    pkg_k         = dict(P["rates"]["packages"]["kitchen"])
    pkg_b         = dict(P["rates"]["packages"]["bathroom"])

    # Quantities
    wall_mult  = float(paint_rules.get(room_type, paint_rules.get("default", 3.2)))
    ceil_ratio = float(ceil_rules.get(ceil_type, ceil_rules.get("default", 0.0)))
    elec_base_pts = float(elec_base.get(room_type, elec_base.get("default", 2.0)))

    paintable_area_sqft      = area * wall_mult
    flooring_area_sqft       = area
    false_ceiling_area_sqft  = area * ceil_ratio
    electrical_points        = int(round((elec_base_pts + area * elec_scale))) if has_elec else 0

    # Cost mapping (materials + labor) → Subtotal
    paint_mat_rate = float(paint_mat_tbl.get(paint_quality, paint_mat_tbl.get("Standard", 0.0)))
    base_floor = floor_tbl.get(floor_type, next(iter(floor_tbl.values())))
    floor_mat_rate = float(base_floor.get(floor_quality, base_floor.get("Standard", 0.0)))
    if ceil_type == "None":
        ceil_mat_rate = 0.0
    else:
        base_ceil = ceil_tbl.get(ceil_type, next(iter(ceil_tbl.values())))
        ceil_mat_rate = float(base_ceil.get(ceil_quality, base_ceil.get("Standard", 0.0)))

    Painting_Material_Est = paintable_area_sqft * paint_mat_rate * mpi * city_mult
    Painting_Labor_Est    = paintable_area_sqft * paint_lab_psf * city_mult
    Flooring_Material_Est = flooring_area_sqft * floor_mat_rate * mpi * city_mult
    Flooring_Labor_Est    = flooring_area_sqft * floor_lab_psf * city_mult
    Ceiling_Material_Est  = false_ceiling_area_sqft * ceil_mat_rate * mpi * city_mult
    Ceiling_Labor_Est     = false_ceiling_area_sqft * ceil_lab_psf * city_mult
    Electrical_Material_Est = electrical_points * elec_mat_pp * mpi * city_mult
    Electrical_Labor_Est    = electrical_points * elec_lab_pp * city_mult
    Kitchen_Package_Cost_Est  = float(pkg_k.get(kitchen_pkg, 0.0))
    Bathroom_Package_Cost_Est = float(pkg_b.get(bathroom_pkg, 0.0))

    Subtotal = (
        Painting_Material_Est + Painting_Labor_Est +
        Flooring_Material_Est + Flooring_Labor_Est +
        Ceiling_Material_Est  + Ceiling_Labor_Est  +
        Electrical_Material_Est + Electrical_Labor_Est +
        Kitchen_Package_Cost_Est + Bathroom_Package_Cost_Est
    )

    row: Dict[str, Any] = {
        'Material_Price_Index': mpi, 'City': city, 'City_Tier': city_tier,
        'City_Multiplier': city_mult, 'Labor_Day_Rate_Min': 0.0, 'Labor_Day_Rate_Max': 0.0,
        'Room_Type': room_type, 'Area_Sqft': area, 'Renovation_Level': level,
        'Paint_Quality': paint_quality, 'Floor_Type': floor_type, 'Floor_Quality': floor_quality,
        'Ceiling_Type': ceil_type, 'Ceiling_Quality': ceil_quality, 'Has_Electrical': has_elec,
        'Furniture_Level': 'Basic', 'Kitchen_Package': kitchen_pkg, 'Bathroom_Package': bathroom_pkg,
        # zeros for legacy/intermediate columns:
        'Painting_Material_Cost': 0, 'Painting_Labor_Cost': 0, 'Flooring_Material_Cost': 0,
        'Flooring_Labor_Cost': 0, 'Ceiling_Material_Cost': 0, 'Ceiling_Labor_Cost': 0,
        'Electrical_Material_Cost': 0, 'Electrical_Labor_Cost': 0, 'Kitchen_Package_Cost': 0,
        'Bathroom_Package_Cost': 0, 'Plumbing_Cost': 0, 'Furniture_Cost': 0,
        'Wastage_Sundries_Cost': 0, 'Contractor_Overhead_Cost': 0, 'GST_Amount': 0, 'Total_Cost_per_Sqft': 0,
        # engineered cols:
        'paintable_area_sqft': paintable_area_sqft, 'flooring_area_sqft': flooring_area_sqft,
        'false_ceiling_area_sqft': false_ceiling_area_sqft, 'electrical_points': electrical_points,
        'paint_mat_rate': paint_mat_rate, 'paint_lab_rate': paint_lab_psf,
        'Painting_Material_Est': Painting_Material_Est, 'Painting_Labor_Est': Painting_Labor_Est,
        'floor_mat_rate': floor_mat_rate, 'floor_lab_rate': floor_lab_psf,
        'Flooring_Material_Est': Flooring_Material_Est, 'Flooring_Labor_Est': Flooring_Labor_Est,
        'ceil_mat_rate': ceil_mat_rate, 'ceil_lab_rate': ceil_lab_psf,
        'Ceiling_Material_Est': Ceiling_Material_Est, 'Ceiling_Labor_Est': Ceiling_Labor_Est,
        'Electrical_Material_Est': Electrical_Material_Est, 'Electrical_Labor_Est': Electrical_Labor_Est,
        'Kitchen_Package_Cost_Est': Kitchen_Package_Cost_Est, 'Bathroom_Package_Cost_Est': Bathroom_Package_Cost_Est,
        'Subtotal': Subtotal,
        'Wastage_Sundries_Est': 0, 'Overhead_Est': 0, 'PreGST_Total_Est': 0, 'GST_Est': 0,
    }

    # ensure all features model might expect:
    all_model_features = [
        'Material_Price_Index','City','City_Tier','City_Multiplier','Labor_Day_Rate_Min',
        'Labor_Day_Rate_Max','Room_Type','Area_Sqft','Renovation_Level','Paint_Quality',
        'Floor_Type','Floor_Quality','Ceiling_Type','Ceiling_Quality','Has_Electrical',
        'Furniture_Level','Kitchen_Package','Bathroom_Package','Painting_Material_Cost',
        'Painting_Labor_Cost','Flooring_Material_Cost','Flooring_Labor_Cost','Ceiling_Material_Cost',
        'Ceiling_Labor_Cost','Electrical_Material_Cost','Electrical_Labor_Cost','Kitchen_Package_Cost',
        'Bathroom_Package_Cost','Plumbing_Cost','Furniture_Cost','Wastage_Sundries_Cost',
        'Contractor_Overhead_Cost','GST_Amount','Total_Cost_per_Sqft','paintable_area_sqft',
        'flooring_area_sqft','false_ceiling_area_sqft','electrical_points','paint_mat_rate',
        'paint_lab_rate','Painting_Material_Est','Painting_Labor_Est','floor_mat_rate','floor_lab_rate',
        'Flooring_Material_Est','Flooring_Labor_Est','ceil_mat_rate','ceil_lab_rate',
        'Ceiling_Material_Est','Ceiling_Labor_Est','Electrical_Material_Est','Electrical_Labor_Est',
        'Kitchen_Package_Cost_Est','Bathroom_Package_Cost_Est','Subtotal','Wastage_Sundries_Est',
        'Overhead_Est','PreGST_Total_Est','GST_Est'
    ]
    for c in all_model_features:
        row.setdefault(c, 0)

    return pd.DataFrame([row], columns=all_model_features + ["Subtotal"])

# ------------------------------- API models ----------------------------

class EstimateIn(BaseModel):
    area_sqft: float = Field(..., gt=0)
    city: str
    renovation_level: str
    bedrooms: Optional[int] = Field(3, ge=0)
    bathrooms: Optional[int] = Field(2, ge=0)
    property_type: Optional[str] = "apartment"
    # optional overrides (advanced)
    has_electrical: Optional[bool] = True
    ceiling_type: Optional[str] = "None"
    room_type: Optional[str] = "Bedroom"
    floor_type: Optional[str] = None
    kitchen_package: Optional[str] = None
    bathroom_package: Optional[str] = None

    @field_validator("renovation_level")
    @classmethod
    def norm_level(cls, v: str) -> str:
        return v.capitalize()

# ---------------------------- app lifespan -----------------------------

app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model, config, and lookups...")
    P = load_params(PARAMS_PATH)
    model = load_model(DEFAULT_MODEL_PATH)
    expected = get_expected_input_cols(model)
    ci_df = load_cost_indices(COST_INDEX_PATH)
    cities = load_all_cities(PROCESSED_PARQUET, RAW_CSV, COST_INDEX_PATH)

    app_state["params"] = P
    app_state["model"] = model
    app_state["expected"] = expected
    app_state["cost_idx"] = ci_df
    app_state["cities"] = cities
    logger.info("Startup complete. MODEL_TARGET=%s", MODEL_TARGET)
    yield
    # no teardown required

app = FastAPI(
    title="Renovation Cost Estimator API",
    description="FastAPI microservice exposing LightGBM cost estimates with config/lookup support.",
    version="1.0.0",
    lifespan=lifespan,
)

# -------------------------------- routes ------------------------------

@app.get("/", tags=["meta"])
def root():
    return SERVICE_STATUS

@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "healthy"}

@app.get("/meta/cities", tags=["meta"])
def meta_cities():
    return {"cities": app_state.get("cities", [])}

def snake_to_model(mini_in: EstimateIn) -> dict:
    """
    Map incoming snake_case to model/training column names.
    Includes minimal inputs; advanced fields are optional.
    """
    d = mini_in.model_dump()
    # core
    out = {
        "City": d["city"],
        "Area_Sqft": d["area_sqft"],
        "Renovation_Level": d["renovation_level"],
        # default choices for transparent quantities:
        "Room_Type": d.get("room_type") or "Bedroom",
        "Has_Electrical": d.get("has_electrical") if d.get("has_electrical") is not None else True,
        "Ceiling_Type": d.get("ceiling_type") or "None",
    }
    # optional overrides (let model use more specific info if desired)
    if d.get("floor_type"):        out["Floor_Type"] = d["floor_type"]
    if d.get("kitchen_package"):   out["Kitchen_Package"] = d["kitchen_package"]
    if d.get("bathroom_package"):  out["Bathroom_Package"] = d["bathroom_package"]
    return out

@app.post("/estimate", tags=["estimate"])
def estimate(payload: EstimateIn):
    try:
        P = app_state["params"]
        model = app_state["model"]
        expected = app_state.get("expected")
        ci_df = app_state["cost_idx"]

        # 1) Map inputs
        mini = snake_to_model(payload)

        # 2) Build engineered features
        feats = compute_engineered_features(mini, P, ci_df)

        # 3) Mirror suffixes & align to training schema
        feats = mirror_suffix_columns(feats, expected or [])
        X = align_to_model(feats, expected)

        # 4) Predict
        pred = model.predict(X)
        ml_total = to_number(pred, 0.0)

        # If model outputs per-sq-ft, convert to total
        if MODEL_TARGET == "per_sqft":
            ml_total *= to_number(mini["Area_Sqft"], 0.0)

        # 5) Quantity baseline for range (transparent)
        base_cost = to_number(feats["Subtotal"].iloc[0], 0.0)
        qty_break = calculate_realistic_quantity_cost(base_cost)

        # 6) 10% range around ML estimate
        lo, hi = ml_total * 0.90, ml_total * 1.10

        # 7) City meta included for clarity
        city_idx = lookup_city_index(mini["City"], ci_df, P)

        return JSONResponse({
            "estimate": ml_total,
            "currency": "INR",
            "display": rupees(ml_total),
            "range_low": lo,
            "range_high": hi,
            "meta": {
                "city": mini["City"],
                "renovation_level": mini["Renovation_Level"],
                "applied_city_multiplier": city_idx["City_Multiplier"],
                "applied_material_price_index": city_idx["Material_Price_Index"],
                "city_tier": city_idx.get("City_Tier", ""),
                "qty_baseline_final": qty_break["final_total_qty"],
            }
        })
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing key in input or config: {ke!s}") from ke
    except Exception as e:
        # Include a small hint in logs, return user-friendly message
        logger.exception("Estimation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Estimation failed: {e!s}")

# ------------------------------- uvicorn -------------------------------

# Start with:
# uvicorn app.main:app --host 0.0.0.0 --port 8000
# (Render supplies $PORT automatically in its Start Command)
