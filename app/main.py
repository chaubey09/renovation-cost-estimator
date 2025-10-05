# app/main.py
from __future__ import annotations

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional, Dict, Any

import joblib
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# ---------------------------
# Optional: import your logic
# ---------------------------
# Expecting you have something like:
#   src/features.py -> build_features(payload: dict, config: dict, lookups: dict) -> pd.DataFrame
#   src/inference.py -> predict_cost(model, features_df: pd.DataFrame, config: dict) -> float
# If your function names/paths differ, just tweak these imports.

BUILD_FEATURES = None
PREDICT_COST = None

try:
    # Adjust these to your real modules if needed
    from src.features import build_features as _build_features
    from src.inference import predict_cost as _predict_cost
    BUILD_FEATURES = _build_features
    PREDICT_COST = _predict_cost
except Exception:
    # Fallback stub if you haven't split logic yet.
    # Replace this with your real FE and inference calls.
    def _build_features(payload: Dict[str, Any], config: Dict[str, Any], lookups: Dict[str, Any]) -> pd.DataFrame:
        # Minimal example: echo inputs as a one-row DataFrame.
        # In production, call your real feature engineering.
        df = pd.DataFrame([payload])
        # Example: enrich with city multipliers if present
        if "city" in payload and "cost_indices" in lookups:
            ci = lookups["cost_indices"]
            row = ci.loc[ci["city"].str.lower() == str(payload["city"]).lower()]
            if not row.empty and "city_multiplier" in row:
                df["city_multiplier"] = float(row.iloc[0]["city_multiplier"])
        return df

    def _predict_cost(model, features_df: pd.DataFrame, config: Dict[str, Any]) -> float:
        # Your LightGBM model likely expects a specific column set here.
        # Map/rename features_df columns to what your model expects.
        yhat = model.predict(features_df)[0]
        # Optionally apply any business rules from config
        return float(yhat)

    BUILD_FEATURES = _build_features
    PREDICT_COST = _predict_cost


# ---------------------------
# Configuration via ENV (optional)
# ---------------------------
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm.pkl")
DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH", "params.yaml")
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "data/external")

# ---------------------------
# Pydantic Schemas
# ---------------------------
RenovationLevel = Literal["basic", "standard", "premium"]

class EstimateRequest(BaseModel):
    area_sqft: float = Field(..., gt=0, description="Total built-up area in square feet")
    city: str = Field(..., min_length=1, description="City name for price index/multipliers")
    renovation_level: RenovationLevel = Field(..., description="Scope level affecting per-sqft rates")
    bedrooms: Optional[int] = Field(default=None, ge=0)
    bathrooms: Optional[int] = Field(default=None, ge=0)
    property_type: Optional[Literal["apartment", "villa", "independent_floor", "office"]] = "apartment"
    # Add any other inputs your FE requires (material choices, age, etc.)
    # e.g., material_grade: Optional[Literal["economy","mid","luxury"]] = "mid"

    @field_validator("city")
    @classmethod
    def city_strip(cls, v: str) -> str:
        return v.strip()

class EstimateResponse(BaseModel):
    estimate: float
    currency: Literal["INR", "USD", "EUR"] = "INR"
    inputs_echo: Dict[str, Any]
    meta: Dict[str, Any] = {}

# ---------------------------
# Lifespan: load heavy assets once
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger("uvicorn.error")
    logger.info("Loading model, config, and lookups...")

    # Load model
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise RuntimeError(f"Model file not found at {DEFAULT_MODEL_PATH}")
    model = joblib.load(DEFAULT_MODEL_PATH)

    # Load YAML config
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        raise RuntimeError(f"Config file not found at {DEFAULT_CONFIG_PATH}")
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Load lookups (example: cost_indices.csv). Add more as needed.
    lookups: Dict[str, Any] = {}
    cost_indices_csv = os.path.join(DEFAULT_DATA_DIR, "cost_indices.csv")
    if os.path.exists(cost_indices_csv):
        lookups["cost_indices"] = pd.read_csv(cost_indices_csv)

    # Store in app.state
    app.state.model = model
    app.state.config = config
    app.state.lookups = lookups

    logger.info("Assets loaded. Ready to serve requests.")
    yield
    # Teardown (if needed)


app = FastAPI(
    title="Renovation Cost Estimator API",
    version="1.0.0",
    description="FastAPI microservice exposing LightGBM cost estimates with config/lookup support.",
    lifespan=lifespan,
)

# CORS (adjust origins to your frontend domains if any)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Health & Root
# ---------------------------
@app.get("/", tags=["meta"])
def root():
    return {
        "service": "renovation-estimator",
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }

@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "healthy"}

# ---------------------------
# Main endpoint
# ---------------------------
@app.post("/estimate", response_model=EstimateResponse, tags=["inference"])
def estimate(req: EstimateRequest):
    try:
        # Access assets
        model = app.state.model
        config = app.state.config
        lookups = app.state.lookups

        # Optional: validate city is supported (if you want strict checking)
        if "cost_indices" in lookups:
            cities = set(lookups["cost_indices"]["city"].str.lower())
            if req.city.lower() not in cities:
                # Soft-fail: either raise or proceed without city multiplier
                # Here, we return a helpful 400
                raise HTTPException(
                    status_code=400,
                    detail=f"City '{req.city}' not supported. Supported cities include e.g. {sorted(list(cities))[:8]} ...",
                )

        # Convert to dict and run your FE + inference
        payload = req.model_dump()
        features_df = BUILD_FEATURES(payload, config, lookups)
        estimate_value = PREDICT_COST(model, features_df, config)

        return EstimateResponse(
            estimate=round(estimate_value, 2),
            currency=config.get("currency", "INR"),
            inputs_echo=payload,
            meta={
                "model_path": DEFAULT_MODEL_PATH,
                "config_path": DEFAULT_CONFIG_PATH,
                "features_shape": list(features_df.shape),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        # Log the error for observability
        logging.getLogger("uvicorn.error").exception("Estimation error")
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")