# app/main.py
from __future__ import annotations

import os
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

import joblib
import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

# -----------------------------------------------------------------------------
# Config / Paths (can be overridden via Render env vars)
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm.pkl")
DEFAULT_CONFIG_PATH = os.getenv("CONFIG_PATH", "params.yaml")
DEFAULT_DATA_DIR = os.getenv("DATA_DIR", "data/external")
DEFAULT_COST_INDICES_FILE = os.getenv("COST_INDICES_FILE", "cost_indices.csv")

# -----------------------------------------------------------------------------
# FastAPI setup & logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")

description = (
    "FastAPI microservice exposing LightGBM cost estimates "
    "with config/lookup support."
)

# -----------------------------------------------------------------------------
# Pydantic request/response models
#  - Expose snake_case keys on the API
#  - Map to training-time column names internally
# -----------------------------------------------------------------------------
class EstimateRequest(BaseModel):
    area_sqft: float
    city: str
    renovation_level: str
    bedrooms: int
    bathrooms: int
    property_type: str

    @field_validator("area_sqft")
    @classmethod
    def area_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("area_sqft must be > 0")
        return v

    @field_validator("bedrooms", "bathrooms")
    @classmethod
    def non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError("bedrooms/bathrooms must be >= 0")
        return v


class EstimateResponse(BaseModel):
    estimate: float
    currency: str = "INR"
    inputs: Dict[str, Any]


# Map API fields (snake_case) -> training columns (exact names used during training)
# Adjust the right-hand side to match the columns your pipeline expects.
API_TO_TRAINING = {
    "area_sqft": "Area_Sqft",
    "city": "City",
    "renovation_level": "Renovation_Level",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "property_type": "Property_Type",
}

# If your pipeline expects extra engineered columns at predict-time,
# put their default values here, e.g., {"Some_Flag": 0}
DEFAULTS: Dict[str, Any] = {}


def build_model_df(
    payload: Dict[str, Any],
    allowed_cities: Optional[set[str]] = None
) -> pd.DataFrame:
    """
    Convert snake_case API payload to a single-row DataFrame with training column names.
    Optionally validate city against provided list.
    """
    # normalize keys to snake_case
    snake = {k.lower(): v for k, v in payload.items()}

    row: Dict[str, Any] = {}
    missing: List[str] = []
    for api_key, train_col in API_TO_TRAINING.items():
        if api_key not in snake:
            missing.append(api_key)
            continue
        row[train_col] = snake[api_key]

    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields: {', '.join(missing)}"
        )

    # City validation (friendly 400 instead of 500)
    if "City" in row and isinstance(row["City"], str) and allowed_cities is not None:
        if row["City"].strip().lower() not in {c.lower() for c in allowed_cities}:
            # Show only first few to keep message short
            sample = ", ".join(sorted(list(allowed_cities))[:10])
            raise HTTPException(
                status_code=400,
                detail=f"City '{row['City']}' not supported. Examples: {sample} ..."
            )

    # Apply defaults for any extra features your model expects
    for k, v in DEFAULTS.items():
        row.setdefault(k, v)

    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# Lifespan: load model/config/lookups once
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model, config, and lookups...")

    # model (optional soft-fail: service starts, /estimate returns 503 if missing)
    model = None
    if os.path.exists(DEFAULT_MODEL_PATH):
        try:
            model = joblib.load(DEFAULT_MODEL_PATH)
            logger.info("Model loaded from %s", DEFAULT_MODEL_PATH)
        except Exception as e:
            logger.warning("Model load failed from %s: %s", DEFAULT_MODEL_PATH, e)
    else:
        logger.warning("Model file not found at %s", DEFAULT_MODEL_PATH)

    # params.yaml (optional)
    config = None
    if os.path.exists(DEFAULT_CONFIG_PATH):
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info("Config loaded from %s", DEFAULT_CONFIG_PATH)
        except Exception as e:
            logger.warning("Config load failed from %s: %s", DEFAULT_CONFIG_PATH, e)
    else:
        logger.info("Config file not present at %s (optional)", DEFAULT_CONFIG_PATH)

    # cost indices lookup (optional)
    cost_indices = None
    lookup_path = os.path.join(DEFAULT_DATA_DIR, DEFAULT_COST_INDICES_FILE)
    if os.path.exists(lookup_path):
        try:
            cost_indices = pd.read_csv(lookup_path)
            logger.info("Cost indices loaded from %s", lookup_path)
        except Exception as e:
            logger.warning("Failed to read cost indices at %s: %s", lookup_path, e)
    else:
        logger.info("Lookup CSV not present at %s (optional)", lookup_path)

    app.state.model = model
    app.state.config = config
    app.state.cost_indices = cost_indices

    yield

    # (Optional) place for teardown if needed
    logger.info("Shutting down service")


app = FastAPI(
    title="Renovation Cost Estimator API",
    version="1.0.0",
    description=description,
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Utility: get allowed cities from lookup (case-insensitive)
# -----------------------------------------------------------------------------
def get_allowed_cities(app: FastAPI) -> Optional[set[str]]:
    ci = getattr(app.state, "cost_indices", None)
    if ci is None or not isinstance(ci, pd.DataFrame):
        return None
    # be resilient to column case
    city_col = next((c for c in ci.columns if c.lower() == "city"), None)
    if not city_col:
        return None
    # ensure strings
    return set(ci[city_col].dropna().astype(str).unique().tolist())


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {
        "service": "renovation-estimator",
        "status": "ok",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/healthz", tags=["meta"])
def healthz() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/meta/cities", tags=["meta"])
def meta_cities() -> Dict[str, List[str]]:
    cities = sorted(list(get_allowed_cities(app) or []))
    return {"cities": cities}


@app.post("/estimate", response_model=EstimateResponse, tags=["inference"])
def estimate(req: EstimateRequest, request: Request):
    # Require model to be loaded
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Expected at '{DEFAULT_MODEL_PATH}'."
        )

    # Optional request id for easier log correlation
    rid = uuid.uuid4().hex[:8]

    try:
        payload = req.model_dump()
        allowed_cities = get_allowed_cities(app)

        X = build_model_df(payload, allowed_cities=allowed_cities)

        # If your persisted object is a Pipeline with preprocessing + model,
        # .predict(X) will handle encoders/scalers internally.
        preds = app.state.model.predict(X)
        estimate_value = float(preds[0])

        logger.info("[rid=%s] OK /estimate inputs=%s estimate=%.2f", rid, payload, estimate_value)

        return EstimateResponse(estimate=estimate_value, inputs=payload)

    except HTTPException as e:
        logger.warning("[rid=%s] HTTP %s /estimate: %s", rid, e.status_code, e.detail)
        raise

    except KeyError as e:
        # Typical cause: unexpected key/value (e.g., city not in lookup)
        msg = f"Unknown field or value: {e.args[0]}"
        logger.warning("[rid=%s] 400 KeyError: %s", rid, msg)
        raise HTTPException(status_code=400, detail=msg)

    except FileNotFoundError as e:
        msg = f"Server missing file: {getattr(e, 'filename', str(e))}"
        logger.error("[rid=%s] 500 FileNotFoundError: %s", rid, msg)
        raise HTTPException(status_code=500, detail=msg)

    except Exception as e:
        # Last-resort guard with a stable message
        logger.exception("[rid=%s] 500 Estimation failed", rid)
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")


# -----------------------------------------------------------------------------
# Exception handler (optional: consistent JSON for unexpected errors)
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Let HTTPException pass through (already handled above)
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    logger.exception("Unhandled server error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
