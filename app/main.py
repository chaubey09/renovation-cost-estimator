# app/main.py
from __future__ import annotations

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("renovation-api")

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm.pkl")
DEFAULT_PARAMS_PATH = os.getenv("PARAMS_PATH", "params.yaml")
DEFAULT_COST_INDEX_CSV = os.getenv("COST_INDEX_CSV", "data/external/cost_indices.csv")

# ---------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------
class EstimateIn(BaseModel):
    area_sqft: float = Field(..., ge=10, description="Carpet/built-up area in square feet")
    city: str = Field(..., description="City name; should appear in /meta/cities")
    renovation_level: str = Field(..., description="Economy | Standard | Premium (case-insensitive)")
    bedrooms: Optional[int] = Field(None, ge=0, description="Optional, not used by current FE")
    bathrooms: Optional[int] = Field(None, ge=0, description="Optional, not used by current FE")
    property_type: Optional[str] = Field("apartment", description="Optional, not used by current FE")

    # Advanced flags — optional; we keep defaults identical to your Streamlit UI
    has_electrical: Optional[bool] = True
    false_ceiling: Optional[bool] = False
    room_type: Optional[str] = "Bedroom"  # Bedroom | Living_Room | Kitchen | Bathroom | Other

    # Optional explicit choices; when omitted we derive from renovation_level
    floor_type: Optional[str] = None
    kitchen_package: Optional[str] = None
    bathroom_package: Optional[str] = None


class EstimateOut(BaseModel):
    estimate: float
    currency: str = "INR"
    display: str
    range_low: float
    range_high: float
    meta: Dict[str, Any]


# ---------------------------------------------------------------------
# Streamlit-equivalent helpers (ported verbatim in spirit)
# ---------------------------------------------------------------------
def to_number(x, default=0.0) -> float:
    try:
        if isinstance(x, pd.Series):
            return float(x.iloc[0]) if not x.empty else float(default)
        if isinstance(x, (np.ndarray, list, tuple, pd.Index)):
            return float(np.asarray(x).ravel()[0]) if len(x) else float(default)
        return float(x)
    except Exception:
        try:
            v = pd.to_numeric(x, errors="coerce")
            if isinstance(v, pd.Series):
                v = v.dropna()
                return float(v.iloc[0]) if len(v) else float(default)
            if np.isnan(v):
                return float(default)
            return float(v)
        except Exception:
            return float(default)


def spec_to_quality(spec: str) -> str:
    s = str(spec).lower()
    if s in {"economy", "basic"}:
        return "Economy"
    if s in {"premium", "luxury"}:
        return "Premium"
    return "Standard"


def dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _rename_ci_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "city": "City",
        "city_multiplier": "City_Multiplier",
        "material_price_index": "Material_Price_Index",
        "city_tier": "City_Tier",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})


def load_cost_indices(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning("Cost index CSV not found at %s; using empty frame.", path)
        return pd.DataFrame(columns=["City", "City_Multiplier", "Material_Price_Index", "City_Tier"])
    df = pd.read_csv(path)
    df = _rename_ci_columns(df)
    if "City_Multiplier" in df.columns:
        df["City_Multiplier"] = pd.to_numeric(df["City_Multiplier"], errors="coerce")
    if "Material_Price_Index" in df.columns:
        df["Material_Price_Index"] = pd.to_numeric(df["Material_Price_Index"], errors="coerce")
    return df


def load_params(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"params.yaml not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    "Area_Sqft",
    "Paint_Quality",
    "Floor_Type",
    "Floor_Quality",
    "Ceiling_Type",
    "Ceiling_Quality",
    "Has_Electrical",
]


def mirror_suffix_columns(df: pd.DataFrame, expected_cols: Optional[List[str]]) -> pd.DataFrame:
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


def compute_engineered_features(mini: dict, P: dict, ci_df: pd.DataFrame) -> pd.DataFrame:
    # Parse minimal inputs
    city = str(mini.get("City", ""))
    area = to_number(mini.get("Area_Sqft", 0.0), 0.0)
    level = str(mini.get("Renovation_Level", "Economy"))
    room_type = str(mini.get("Room_Type", "Bedroom"))
    has_elec = bool(mini.get("Has_Electrical", True))
    ceil_type = str(mini.get("Ceiling_Type", "None"))

    # Derive qualities from spec level unless overridden
    qual = spec_to_quality(level)
    paint_quality = str(mini.get("Paint_Quality", qual))
    floor_quality = str(mini.get("Floor_Quality", qual))
    ceil_quality = str(mini.get("Ceiling_Quality", qual))
    floor_type = str(mini.get("Floor_Type", "Ceramic_Tile"))
    kitchen_pkg = str(mini.get("Kitchen_Package", "None"))
    bathroom_pkg = str(mini.get("Bathroom_Package", "None"))

    # City lookup
    city_idx = lookup_city_index(city, ci_df, P)
    city_mult = float(city_idx["City_Multiplier"])
    mpi = float(city_idx["Material_Price_Index"])
    city_tier = str(city_idx.get("City_Tier", ""))

    # Rates / rules from params.yaml
    paint_rules = dict(P["quant_rules"]["paint_wall_multiplier_by_room"])
    ceil_rules = dict(P["quant_rules"]["false_ceiling_ratio_by_type"])
    elec_base = dict(P["quant_rules"]["electrical_points_base"])
    elec_scale = float(P["quant_rules"]["electrical_area_scale"])

    paint_mat_tbl = dict(P["rates"]["painting"]["material_per_sqft_by_quality"])
    floor_tbl = dict(P["rates"]["flooring"]["material_per_sqft_by_type_quality"])
    ceil_tbl = dict(P["rates"]["ceiling"]["material_per_sqft_by_type_quality"])
    paint_lab_psf = float(P["rates"]["painting"]["labor_per_sqft"])
    floor_lab_psf = float(P["rates"]["flooring"]["labor_per_sqft"])
    ceil_lab_psf = float(P["rates"]["ceiling"]["labor_per_sqft"])
    elec_mat_pp = float(P["rates"]["electrical"]["per_point_material"])
    elec_lab_pp = float(P["rates"]["electrical"]["per_point_labor"])
    pkg_k = dict(P["rates"]["packages"]["kitchen"])
    pkg_b = dict(P["rates"]["packages"]["bathroom"])

    # Quantities
    wall_mult = float(paint_rules.get(room_type, paint_rules.get("default", 3.2)))
    ceil_ratio = float(ceil_rules.get(ceil_type, ceil_rules.get("default", 0.0)))
    elec_base_pts = float(elec_base.get(room_type, elec_base.get("default", 2.0)))

    paintable_area_sqft = area * wall_mult
    flooring_area_sqft = area
    false_ceiling_area_sqft = area * ceil_ratio
    electrical_points = int(round((elec_base_pts + area * elec_scale))) if has_elec else 0

    # Rates selection
    paint_mat_rate = float(paint_mat_tbl.get(paint_quality, paint_mat_tbl.get("Standard", 0.0)))
    base_floor = floor_tbl.get(floor_type, next(iter(floor_tbl.values())))
    floor_mat_rate = float(base_floor.get(floor_quality, base_floor.get("Standard", 0.0)))
    if ceil_type == "None":
        ceil_mat_rate = 0.0
    else:
        base_ceil = ceil_tbl.get(ceil_type, next(iter(ceil_tbl.values())))
        ceil_mat_rate = float(base_ceil.get(ceil_quality, base_ceil.get("Standard", 0.0)))

    # Cost estimates
    Painting_Material_Est = paintable_area_sqft * paint_mat_rate * mpi * city_mult
    Painting_Labor_Est = paintable_area_sqft * paint_lab_psf * city_mult
    Flooring_Material_Est = flooring_area_sqft * floor_mat_rate * mpi * city_mult
    Flooring_Labor_Est = flooring_area_sqft * floor_lab_psf * city_mult
    Ceiling_Material_Est = false_ceiling_area_sqft * ceil_mat_rate * mpi * city_mult
    Ceiling_Labor_Est = false_ceiling_area_sqft * ceil_lab_psf * city_mult
    Electrical_Material_Est = electrical_points * elec_mat_pp * mpi * city_mult
    Electrical_Labor_Est = electrical_points * elec_lab_pp * city_mult
    Kitchen_Package_Cost_Est = float(pkg_k.get(kitchen_pkg, 0.0))
    Bathroom_Package_Cost_Est = float(pkg_b.get(bathroom_pkg, 0.0))

    Subtotal = (
        Painting_Material_Est
        + Painting_Labor_Est
        + Flooring_Material_Est
        + Flooring_Labor_Est
        + Ceiling_Material_Est
        + Ceiling_Labor_Est
        + Electrical_Material_Est
        + Electrical_Labor_Est
        + Kitchen_Package_Cost_Est
        + Bathroom_Package_Cost_Est
    )

    row = {
        "Material_Price_Index": mpi,
        "City": city,
        "City_Tier": city_tier,
        "City_Multiplier": city_mult,
        "Labor_Day_Rate_Min": 0.0,
        "Labor_Day_Rate_Max": 0.0,
        "Room_Type": room_type,
        "Area_Sqft": area,
        "Renovation_Level": level,
        "Paint_Quality": paint_quality,
        "Floor_Type": floor_type,
        "Floor_Quality": floor_quality,
        "Ceiling_Type": ceil_type,
        "Ceiling_Quality": ceil_quality,
        "Has_Electrical": has_elec,
        "Furniture_Level": "Basic",
        "Kitchen_Package": kitchen_pkg,
        "Bathroom_Package": bathroom_pkg,
        # legacy cost columns (zeros)
        "Painting_Material_Cost": 0,
        "Painting_Labor_Cost": 0,
        "Flooring_Material_Cost": 0,
        "Flooring_Labor_Cost": 0,
        "Ceiling_Material_Cost": 0,
        "Ceiling_Labor_Cost": 0,
        "Electrical_Material_Cost": 0,
        "Electrical_Labor_Cost": 0,
        "Kitchen_Package_Cost": 0,
        "Bathroom_Package_Cost": 0,
        "Plumbing_Cost": 0,
        "Furniture_Cost": 0,
        "Wastage_Sundries_Cost": 0,
        "Contractor_Overhead_Cost": 0,
        "GST_Amount": 0,
        "Total_Cost_per_Sqft": 0,
        # engineered columns
        "paintable_area_sqft": paintable_area_sqft,
        "flooring_area_sqft": flooring_area_sqft,
        "false_ceiling_area_sqft": false_ceiling_area_sqft,
        "electrical_points": electrical_points,
        "paint_mat_rate": paint_mat_rate,
        "paint_lab_rate": paint_lab_psf,
        "Painting_Material_Est": Painting_Material_Est,
        "Painting_Labor_Est": Painting_Labor_Est,
        "floor_mat_rate": floor_mat_rate,
        "floor_lab_rate": floor_lab_psf,
        "Flooring_Material_Est": Flooring_Material_Est,
        "Flooring_Labor_Est": Flooring_Labor_Est,
        "ceil_mat_rate": ceil_mat_rate,
        "ceil_lab_rate": ceil_lab_psf,
        "Ceiling_Material_Est": Ceiling_Material_Est,
        "Ceiling_Labor_Est": Ceiling_Labor_Est,
        "Electrical_Material_Est": Electrical_Material_Est,
        "Electrical_Labor_Est": Electrical_Labor_Est,
        "Kitchen_Package_Cost_Est": Kitchen_Package_Cost_Est,
        "Bathroom_Package_Cost_Est": Bathroom_Package_Cost_Est,
        "Subtotal": Subtotal,
        "Wastage_Sundries_Est": 0,
        "Overhead_Est": 0,
        "PreGST_Total_Est": 0,
        "GST_Est": 0,
    }

    df = pd.DataFrame([row])
    # Ensure all expected engineered names exist (zero-fill safety)
    engineered_names = [
        "Material_Price_Index",
        "City",
        "City_Tier",
        "City_Multiplier",
        "Labor_Day_Rate_Min",
        "Labor_Day_Rate_Max",
        "Room_Type",
        "Area_Sqft",
        "Renovation_Level",
        "Paint_Quality",
        "Floor_Type",
        "Floor_Quality",
        "Ceiling_Type",
        "Ceiling_Quality",
        "Has_Electrical",
        "Furniture_Level",
        "Kitchen_Package",
        "Bathroom_Package",
        "Painting_Material_Cost",
        "Painting_Labor_Cost",
        "Flooring_Material_Cost",
        "Flooring_Labor_Cost",
        "Ceiling_Material_Cost",
        "Ceiling_Labor_Cost",
        "Electrical_Material_Cost",
        "Electrical_Labor_Cost",
        "Kitchen_Package_Cost",
        "Bathroom_Package_Cost",
        "Plumbing_Cost",
        "Furniture_Cost",
        "Wastage_Sundries_Cost",
        "Contractor_Overhead_Cost",
        "GST_Amount",
        "Total_Cost_per_Sqft",
        "paintable_area_sqft",
        "flooring_area_sqft",
        "false_ceiling_area_sqft",
        "electrical_points",
        "paint_mat_rate",
        "paint_lab_rate",
        "Painting_Material_Est",
        "Painting_Labor_Est",
        "floor_mat_rate",
        "floor_lab_rate",
        "Flooring_Material_Est",
        "Flooring_Labor_Est",
        "ceil_mat_rate",
        "ceil_lab_rate",
        "Ceiling_Material_Est",
        "Ceiling_Labor_Est",
        "Electrical_Material_Est",
        "Electrical_Labor_Est",
        "Kitchen_Package_Cost_Est",
        "Bathroom_Package_Cost_Est",
        "Subtotal",
        "Wastage_Sundries_Est",
        "Overhead_Est",
        "PreGST_Total_Est",
        "GST_Est",
    ]
    for c in engineered_names:
        if c not in df.columns:
            df[c] = 0
    return df


def calculate_realistic_quantity_cost(base_cost: float) -> dict:
    base_cost = to_number(base_cost, 0.0)
    contractor_margin = 0.15 * base_cost
    wastage_sundries = 0.05 * base_cost
    contingency = 0.07 * base_cost
    pre_gst_total = base_cost + contractor_margin + wastage_sundries + contingency
    gst = 0.18 * pre_gst_total
    final_total = pre_gst_total + gst
    return {
        "base_cost": float(base_cost),
        "contractor_margin": float(contractor_margin),
        "wastage_sundries": float(wastage_sundries),
        "contingency": float(contingency),
        "pre_gst_total": float(pre_gst_total),
        "gst": float(gst),
        "final_total_qty": float(final_total),
    }


# ---------------------------------------------------------------------
# Lifespan: load model, params, cost indices once
# ---------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model, config, and lookups...")
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise RuntimeError(f"Model file not found at {DEFAULT_MODEL_PATH}")
    model = joblib.load(DEFAULT_MODEL_PATH)
    params = load_params(DEFAULT_PARAMS_PATH)
    cost_idx = load_cost_indices(DEFAULT_COST_INDEX_CSV)
    expected_cols = get_expected_input_cols(model)
    if expected_cols:
        logger.info("Model exposes %d expected columns.", len(expected_cols))
    else:
        logger.info("Model did not expose feature_names_in_ / feature_name_; will align best-effort.")
    app.state.model = model
    app.state.params = params
    app.state.cost_idx = cost_idx
    app.state.expected_cols = expected_cols
    # cities for meta endpoint
    app.state.cities = sorted(set(cost_idx["City"].dropna().astype(str))) if not cost_idx.empty else []
    yield
    # (nothing to clean up)


app = FastAPI(
    title="Renovation Cost Estimator API",
    description="FastAPI microservice exposing LightGBM cost estimates with config/lookup support.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root():
    return {"service": "renovation-estimator", "status": "ok", "docs": "/docs", "openapi": "/openapi.json"}


@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "healthy"}


@app.get("/meta/cities", tags=["meta"])
def meta_cities():
    return {"cities": app.state.cities}


def _map_request_to_training_names(inp: EstimateIn, P: dict) -> dict:
    # Derive qualities from level if user did not override via optional fields
    level = spec_to_quality(inp.renovation_level)
    # Defaults identical to Streamlit
    floor_type = inp.floor_type or next(iter(P["rates"]["flooring"]["material_per_sqft_by_type_quality"].keys()))
    kitchen_pkg = inp.kitchen_package or next(iter(P["rates"]["packages"]["kitchen"].keys()))
    bathroom_pkg = inp.bathroom_package or next(iter(P["rates"]["packages"]["bathroom"].keys()))
    ceil_type = "POP" if (inp.false_ceiling is True) else "None"

    return {
        "City": inp.city,
        "Area_Sqft": inp.area_sqft,
        "Renovation_Level": inp.renovation_level,
        "Room_Type": inp.room_type or "Bedroom",
        "Has_Electrical": bool(inp.has_electrical) if inp.has_electrical is not None else True,
        "Ceiling_Type": ceil_type,
        "Ceiling_Quality": level,
        "Paint_Quality": level,
        "Floor_Type": floor_type,
        "Floor_Quality": level,
        "Kitchen_Package": kitchen_pkg,
        "Bathroom_Package": bathroom_pkg,
    }


@app.post("/estimate", response_model=EstimateOut, tags=["estimate"])
def estimate(req: EstimateIn):
    try:
        P = app.state.params
        ci = app.state.cost_idx
        model = app.state.model
        expected = app.state.expected_cols

        # Map API payload → training names dict (same as Streamlit mini)
        mini = _map_request_to_training_names(req, P)

        # Build engineered features (Streamlit-equivalent)
        feats = compute_engineered_features(mini, P, ci)

        # Mirror *_x / *_y columns if model expects them
        feats = mirror_suffix_columns(feats, expected)

        # Align to the model schema (add missing cols as 0, reorder)
        X = align_to_model(feats, expected)

        # Predict
        pred = model.predict(X)
        ml_total = to_number(pred, 0.0)

        # Quantity baseline (for range)
        base_cost = to_number(feats["Subtotal"].iloc[0], 0.0)
        qty_break = calculate_realistic_quantity_cost(base_cost)
        # Present a simple ±10% band around ML total (same as Streamlit ML-only)
        lo, hi = ml_total * 0.90, ml_total * 1.10

        # City meta used
        found = lookup_city_index(mini["City"], ci, P)

        return EstimateOut(
            estimate=float(ml_total),
            display=f"₹{ml_total:,.0f}",
            range_low=float(lo),
            range_high=float(hi),
            meta={
                "city": mini["City"],
                "renovation_level": req.renovation_level,
                "applied_city_multiplier": found["City_Multiplier"],
                "applied_material_price_index": found["Material_Price_Index"],
                "city_tier": found.get("City_Tier", ""),
                "qty_baseline_final": qty_break["final_total_qty"],
            },
        )
    except HTTPException:
        raise
    except KeyError as ke:
        msg = f"Missing configuration key: {ke}. Check params.yaml structure."
        logger.exception(msg)
        raise HTTPException(status_code=500, detail=msg)
    except Exception as e:
        logger.exception("Estimation failed")
        raise HTTPException(status_code=500, detail=f"Estimation failed: {e}")


# ---------------------------------------------------------------------
# Local dev entrypoint (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, reload=False)
