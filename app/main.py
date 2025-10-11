# app/main.py
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Your existing loader; now made tolerant by our wrapper below
from src.service.io import load_model

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Renovation Cost Estimator API")

# Allow your Streamlit frontends / local dev to talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Model loading (non-fatal)
# -----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm.pkl")

# Global app state (model lives here)
app_state: Dict[str, Any] = {"model": None, "model_path": DEFAULT_MODEL_PATH}

def _try_load_model(path: str) -> Optional[Any]:
    """
    Try to load the model. Never raise at startup.
    Returns the model object or None.
    """
    try:
        return load_model(path)  # may return None if file missing (see io.py)
    except Exception as e:
        # Log and carry on; API should still start
        print(f"[WARN] Failed to load model at '{path}': {e}")
        return None

@app.on_event("startup")
async def _on_startup():
    app_state["model"] = _try_load_model(DEFAULT_MODEL_PATH)

# -----------------------------------------------------------------------------
# Health endpoints (both /healthz and /health for compatibility)
# -----------------------------------------------------------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "has_model": bool(app_state.get("model")),
        "model_path": app_state.get("model_path"),
    }

@app.get("/health")
def health_alias():
    # Streamlit app sometimes probes /health first
    return healthz()

# -----------------------------------------------------------------------------
# Example guarded endpoint that needs the model
# (Your real predict/estimate code can call app_state["model"])
# -----------------------------------------------------------------------------
@app.post("/predict")
def predict(payload: dict):
    model = app_state.get("model")
    if model is None:
        # Clear error instead of a 500
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    # TODO: Use your real preprocessing + model inference here
    # pred = model.predict(...)
    return {"ok": True, "prediction": 123.0, "note": "stub response"}

# -----------------------------------------------------------------------------
# Keep your existing routers if they exist (estimate endpoints, etc.)
# This safely no-ops if those modules are not present.
# -----------------------------------------------------------------------------
def _include_optional_routers():
    # Example: if you have app.api:router or app.routes:router, include them
    for modpath in ("app.api", "app.routes"):
        try:
            module = __import__(modpath, fromlist=["router"])
            router = getattr(module, "router", None)
            if router is not None:
                app.include_router(router)
                print(f"[INFO] Included router from '{modpath}'")
        except Exception:
            # Silently ignore if not present
            pass

_include_optional_routers()
