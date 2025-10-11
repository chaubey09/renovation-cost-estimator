# app/main.py
from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Basic in-app params (lightweight, works even if a trained model isn't present)
# --------------------------------------------------------------------------------------

QUALITY_BASE_RATE = {
    "Economy": 1200.0,
    "Standard": 1600.0,
    "Luxury": 2200.0,
}

CITY_MULTIPLIER = {
    "Delhi": 1.00,
    "Mumbai": 1.15,
    "Bengaluru": 1.10,
    "Hyderabad": 1.05,
    "Chennai": 1.05,
    "Kolkata": 0.95,
    "Pune": 1.08,
    "Ahmedabad": 0.92,
    "Jaipur": 0.90,
    "Kanpur": 0.85,
}

DEFAULT_CITY_TIER = {
    "Delhi": "Tier-1",
    "Mumbai": "Tier-1",
    "Bengaluru": "Tier-1",
    "Hyderabad": "Tier-1",
    "Chennai": "Tier-1",
    "Pune": "Tier-1",
    "Kolkata": "Tier-2",
    "Ahmedabad": "Tier-2",
    "Jaipur": "Tier-2",
    "Kanpur": "Tier-3",
}

CURRENCY = "INR"


def _city_multiplier(city: str) -> tuple[float, str, bool]:
    """
    Returns (multiplier, tier, used_default_flag)
    - If city missing, treat as 'Other' with Tier-3 default multiplier 0.90
    """
    c = (city or "").strip()
    if c in CITY_MULTIPLIER:
        return CITY_MULTIPLIER[c], DEFAULT_CITY_TIER.get(c, "Tier-2"), False
    # Default for unknown cities
    return 0.90, "Tier-3", True


def _normalize_quality(q: Optional[str]) -> str:
    s = (q or "").strip().lower()
    if s in {"economy", "basic"}:
        return "Economy"
    if s in {"luxury", "premium"}:
        return "Luxury"
    return "Standard"


# --------------------------------------------------------------------------------------
# Pydantic models (v2)
# --------------------------------------------------------------------------------------

# Full /predict (kept for backward compatibility with your current clients)
class PredictIn(BaseModel):
    area_sqft: float = Field(..., gt=0)
    city: str
    renovation_level: Optional[str] = "Standard"  # Economy | Standard | Luxury
    bedrooms: Optional[int] = 0
    bathrooms: Optional[int] = 0
    property_type: Optional[str] = "apartment"

    # Optional room hints / flags (ignored by simple baseline)
    room_type: Optional[
        Literal["Living Room", "Bedroom", "Kids Room", "Kitchen", "Bathroom", "Dining"]
    ] = None
    has_electrical: Optional[bool] = True
    ceiling_type: Optional[str] = "None"
    feature_flags: Optional[Dict[str, bool]] = None
    feature_quantities: Optional[Dict[str, float]] = None


# Minimal /estimate input (Streamlit/Wix style)
class EasyEstimateIn(BaseModel):
    city: str
    area_sqft: float = Field(..., gt=0)
    desired_quality: Literal["Economy", "Standard", "Luxury"] = "Standard"
    room_type: Literal[
        "Living Room", "Bedroom", "Kids Room", "Kitchen", "Bathroom", "Dining"
    ] = "Bedroom"


# Back-compat variant that also accepts "renovation_level"
class EasyEstimateCompat(EasyEstimateIn):
    renovation_level: Optional[str] = None


# --------------------------------------------------------------------------------------
# Core estimation (simple baseline; replace/augment with your trained model if desired)
# --------------------------------------------------------------------------------------
def predict_core(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core cost computation used by both /predict and /estimate.
    - If you have a trained model, you can load it in startup and use it here.
    - This baseline is transparent, deterministic, and good enough for integration.
    """
    area = float(payload.get("area_sqft") or 0.0)
    city = str(payload.get("city") or "").strip()
    level_raw = payload.get("renovation_level") or payload.get("desired_quality")
    quality = _normalize_quality(level_raw)

    base_rate = QUALITY_BASE_RATE.get(quality, QUALITY_BASE_RATE["Standard"])
    city_mult, tier, used_default = _city_multiplier(city)

    bedrooms = int(payload.get("bedrooms") or 0)
    bathrooms = int(payload.get("bathrooms") or 0)

    # Small per-room bumps (kept modest)
    bedrooms_bump = max(0, bedrooms) * 0.01  # +1% per bedroom
    bathrooms_bump = max(0, bathrooms) * 0.02  # +2% per bathroom

    # Final rate
    rate = base_rate * city_mult * (1.0 + bedrooms_bump + bathrooms_bump)

    estimate = max(0.0, rate * area)
    lo = estimate * 0.90
    hi = estimate * 1.10

    meta = {
        "city": city or "Other",
        "city_tier": tier,
        "applied_city_multiplier": city_mult,
        "applied_material_price_index": 1.0,  # placeholder for future material index
        "used_default_for_city": used_default,
        "renovation_level": quality,
        "qty_baseline_final": {
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
        },
    }

    # A friendly display string
    display = f"₹{round(estimate):,}"

    return {
        "estimate": estimate,
        "display": display,
        "range_low": lo,
        "range_high": hi,
        "currency": CURRENCY,
        "meta": meta,
    }


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------

app = FastAPI(title="Renovation Cost Estimator API", version="0.1.0")

# CORS (open for Wix/Streamlit usage)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ---------------------------------- Health -------------------------------------------
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok", "version": app.version}


@app.get("/health")
def health_alias() -> Dict[str, Any]:
    return {"ok": True, "message": "health alias", "version": app.version}


# ---------------------------------- Cities -------------------------------------------
@app.get("/cities")
def get_cities() -> Dict[str, Any]:
    cities = sorted(set(CITY_MULTIPLIER.keys()) | {"Other"})
    return {"cities": cities}


# ---------------------------------- Predict (full) -----------------------------------
@app.post("/predict")
def predict(req: PredictIn) -> Dict[str, Any]:
    try:
        payload = req.model_dump()
        # normalize quality name internally
        payload["renovation_level"] = _normalize_quality(payload.get("renovation_level"))
        return predict_core(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e


# ---------------------------- Minimal Estimate endpoints -----------------------------
@app.post("/estimate")
def estimate_minimal(req: EasyEstimateCompat) -> Dict[str, Any]:
    """
    Minimal inputs (Streamlit/Wix-friendly):
      - area_sqft, city, desired_quality (or renovation_level), room_type (optional)
    """
    try:
        desired = req.desired_quality or _normalize_quality(req.renovation_level)
        payload = {
            "city": req.city,
            "area_sqft": float(req.area_sqft),
            "desired_quality": desired,
            "renovation_level": desired,  # route to same key used by predict_core
            "room_type": req.room_type,
            # Defaults below keep computation stable:
            "bedrooms": 0,
            "bathrooms": 0,
            "property_type": "apartment",
            "has_electrical": True,
            "ceiling_type": "None",
            "feature_flags": {},
            "feature_quantities": {},
        }
        return predict_core(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Estimate failed: {e}") from e


@app.post("/estimate-single-room")
def estimate_single_room_alias(req: EasyEstimateCompat) -> Dict[str, Any]:
    # Behaviour identical to /estimate; just an alias
    return estimate_minimal(req)
