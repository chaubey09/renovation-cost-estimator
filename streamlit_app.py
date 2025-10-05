# app.py
# Streamlit app with minimal inputs, auto City → (MPI, Multiplier) lookup, ML-first estimate
# and ALL cities loaded from processed parquet + raw CSV + cost index.

import os
from typing import List
import joblib
import yaml
import numpy as np
import pandas as pd
import streamlit as st

# ============================ UI helpers ============================
def to_number(x, default=0.0) -> float:
    """Coerce Series/array/scalar into a float scalar robustly."""
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

def rupees(x) -> str:
    v = to_number(x, default=np.nan)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "—"
    return f"₹{v:,.0f}"

st.set_page_config(page_title="Renovation Cost Estimator", page_icon="🛠️", layout="wide")
st.title("🛠️ Renovation Cost Estimator")
st.caption("Pick a city and basics — we auto-apply city price factors & material index. ML drives the estimate by default.")

# =================== config/model/index loading ====================
@st.cache_resource(show_spinner=False)
def load_params(path="params.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@st.cache_resource(show_spinner=True)
def load_model():
    for p in ["models/lgbm.pkl", "models/model.joblib", "models/model.pkl"]:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError("No model found. Expected models/lgbm.pkl or models/model.joblib")

@st.cache_resource(show_spinner=False)
def load_cost_indices(path="data/external/cost_indices.csv") -> pd.DataFrame:
    """
    Expected columns (as used by your pipeline):
      City, City_Multiplier, Material_Price_Index
    Optional: City_Tier
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        # normalize column names to exactly what we use downstream
        rename_map = {
            "city": "City",
            "city_multiplier": "City_Multiplier",
            "material_price_index": "Material_Price_Index",
            "city_tier": "City_Tier",
        }
        df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
        # ensure types
        if "City_Multiplier" in df.columns:
            df["City_Multiplier"] = pd.to_numeric(df["City_Multiplier"], errors="coerce")
        if "Material_Price_Index" in df.columns:
            df["Material_Price_Index"] = pd.to_numeric(df["Material_Price_Index"], errors="coerce")
        return df
    # safe empty frame if file missing
    return pd.DataFrame(columns=["City", "City_Multiplier", "Material_Price_Index", "City_Tier"])

# -------- NEW: load all cities from multiple sources (union) ----------
@st.cache_resource(show_spinner=False)
def load_all_cities(
    processed_pq: str = "data/processed/processed.parquet",
    raw_csv: str = "data/raw/renovation_cost_dataset_india_v2.csv",
    cost_idx_csv: str = "data/external/cost_indices.csv",
) -> List[str]:
    """
    Returns a sorted, de-duplicated list of City names by UNION of:
      - processed parquet (preferred),
      - raw CSV (fallback),
      - cost index CSV (ensures calibration cities are included).
    """
    cities: List[str] = []
    seen = set()

    def add(series: pd.Series | None):
        if series is None:
            return
        for c in series.dropna().astype(str).str.strip():
            if c and c not in seen:
                seen.add(c)
                cities.append(c)

    # 1) processed parquet
    try:
        if os.path.exists(processed_pq):
            dfp = pd.read_parquet(processed_pq, columns=["City"])
            if "City" in dfp.columns:
                add(dfp["City"])
    except Exception:
        pass

    # 2) raw csv
    try:
        if os.path.exists(raw_csv):
            dfr = pd.read_csv(raw_csv, usecols=["City"])
            if "City" in dfr.columns:
                add(dfr["City"])
    except Exception:
        pass

    # 3) cost index (so we never lose those)
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

def get_expected_input_cols(pipe):
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

def dedupe_preserve_order(seq):
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

# ======================= city → (MPI, Multiplier) =======================
def lookup_city_index(city: str, ci_df: pd.DataFrame, P: dict) -> dict:
    """
    Return Material_Price_Index, City_Multiplier, City_Tier from cost_indices.csv.
    Fallbacks: MPI -> params.rates.base_mpi, Multiplier -> 1.0, Tier -> ''.
    """
    base_mpi = float(P["rates"].get("base_mpi", 1.0))
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

# =================== quantity calculator (transparent) ===================
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

# =================== ML guardrails / (optional) blend ===================
def calibrate_ml_against_qty(ml_estimate: float, qty_estimate: float, level: str) -> float:
    level = str(level).lower()
    if level in {"economy", "basic"}:
        lo_factor, hi_factor = 0.75, 1.15
    elif level in {"premium", "luxury"}:
        lo_factor, hi_factor = 0.65, 1.35
    else:
        lo_factor, hi_factor = 0.70, 1.25
    qty = float(qty_estimate); ml = float(ml_estimate)
    lo = lo_factor * qty; hi = hi_factor * qty
    return float(max(lo, min(ml, hi)))

def blended_recommendation(ml_estimate: float, realistic_qty_estimate: float, level: str) -> dict:
    level = str(level).lower()
    if level in {"economy", "basic"}:
        w_qty, w_ml = 0.70, 0.30
    elif level in {"premium", "luxury"}:
        w_qty, w_ml = 0.40, 0.60
    else:
        w_qty, w_ml = 0.50, 0.50
    qty = float(realistic_qty_estimate); ml = float(ml_estimate)
    recommended = w_qty * qty + w_ml * ml
    return {
        "recommended_price": float(recommended),
        "price_range": (float(recommended * 0.90), float(recommended * 1.10)),
        "weights": {"qty": w_qty, "ml": w_ml}
    }

# ===================== feature engineering (minimal UI) =====================
_SUFFIX_BASES = ["Area_Sqft","Paint_Quality","Floor_Type","Floor_Quality",
                 "Ceiling_Type","Ceiling_Quality","Has_Electrical"]

def mirror_suffix_columns(df: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
    if not expected_cols:
        return df
    df = df.copy()
    for base in _SUFFIX_BASES:
        if f"{base}_x" in expected_cols and f"{base}_x" not in df.columns and base in df.columns:
            df[f"{base}_x"] = df[base]
        if f"{base}_y" in expected_cols and f"{base}_y" not in df.columns and base in df.columns:
            df[f"{base}_y"] = df[base]
    return df

def align_to_model(Xcand: pd.DataFrame, expected_cols: list) -> pd.DataFrame:
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
    if ceil_type == "None": ceil_mat_rate = 0.0
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

    row = {
        'Material_Price_Index': mpi, 'City': city, 'City_Tier': city_tier,
        'City_Multiplier': city_mult, 'Labor_Day_Rate_Min': 0.0, 'Labor_Day_Rate_Max': 0.0,
        'Room_Type': room_type, 'Area_Sqft': area, 'Renovation_Level': level,
        'Paint_Quality': paint_quality, 'Floor_Type': floor_type, 'Floor_Quality': floor_quality,
        'Ceiling_Type': ceil_type, 'Ceiling_Quality': ceil_quality, 'Has_Electrical': has_elec,
        'Furniture_Level': 'Basic', 'Kitchen_Package': kitchen_pkg, 'Bathroom_Package': bathroom_pkg,
        # legacy/cost columns (zeros at inference)
        'Painting_Material_Cost': 0, 'Painting_Labor_Cost': 0, 'Flooring_Material_Cost': 0,
        'Flooring_Labor_Cost': 0, 'Ceiling_Material_Cost': 0, 'Ceiling_Labor_Cost': 0,
        'Electrical_Material_Cost': 0, 'Electrical_Labor_Cost': 0, 'Kitchen_Package_Cost': 0,
        'Bathroom_Package_Cost': 0, 'Plumbing_Cost': 0, 'Furniture_Cost': 0,
        'Wastage_Sundries_Cost': 0, 'Contractor_Overhead_Cost': 0, 'GST_Amount': 0, 'Total_Cost_per_Sqft': 0,
        # engineered columns used by the model
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

# ================================ UI =================================
P = load_params()
pipe = load_model()
expected = get_expected_input_cols(pipe)
ci_df = load_cost_indices()           # used for city → (MPI, multiplier, tier)
city_options = load_all_cities()      # <-- NEW: ALL cities from data + cost index

if not city_options:
    st.warning(
        "No cities found in processed/raw data or cost index file. "
        "We’ll still use defaults (MPI from params, City_Multiplier=1.0), but please check your data paths."
    )

with st.form("inputs"):
    st.subheader("🔢 Minimal inputs")
    c1, c2, c3 = st.columns(3)
    with c1:
        # City is a selectbox from known cities (searchable by typing)
        city = st.selectbox(
            "City",
            options=city_options or ["Bangalore"],
            index=0,
            help="Start typing to search — options come from your dataset + cost index."
        )
        area = st.number_input("Area (sqft)", min_value=10.0, value=120.0, step=10.0)
    with c2:
        spec_level = st.selectbox("Spec / Renovation Level", ["Economy","Standard","Premium"], index=0)
        has_elec = st.toggle("Has Electrical?", value=True)
    with c3:
        use_false_ceiling = st.toggle("False Ceiling?", value=False)
        room_type = st.selectbox("Room Type", ["Bedroom","Living_Room","Kitchen","Bathroom","Other"], index=0)

    st.write("")
    mode = st.selectbox(
        "Estimator mode",
        ["ML only (recommended)", "ML + guardrails", "Blended (Qty/ML)"],
        help="ML only: show the model prediction.\nML + guardrails: clamp extreme ML near Qty.\nBlended: weighted average of Qty and ML."
    )

    with st.expander("Advanced (optional)"):
        colA, colB = st.columns(2)
        with colA:
            floor_type = st.selectbox("Floor Type", list(P["rates"]["flooring"]["material_per_sqft_by_type_quality"].keys()), index=0)
            kitchen_pkg = st.selectbox("Kitchen Package", list(P["rates"]["packages"]["kitchen"].keys()), index=0)
        with colB:
            bathroom_pkg = st.selectbox("Bathroom Package", list(P["rates"]["packages"]["bathroom"].keys()), index=0)
        # Qualities map from Spec Level (Economy/Standard/Premium)
        quality_map = {"Economy":"Economy","Standard":"Standard","Premium":"Premium"}
        paint_quality = quality_map[spec_level]
        floor_quality = quality_map[spec_level]
        ceil_quality  = quality_map[spec_level]
        ceil_type = "POP" if use_false_ceiling else "None"

    submitted = st.form_submit_button("Estimate", type="primary")

if submitted:
    try:
        # Minimal inputs → full feature row (city factors are looked up automatically)
        mini = {
            "City": city,
            "Area_Sqft": area,
            "Renovation_Level": spec_level,
            "Room_Type": room_type,
            "Has_Electrical": has_elec,
            "Ceiling_Type": "POP" if use_false_ceiling else "None",
            "Ceiling_Quality": ceil_quality,
            "Paint_Quality": paint_quality,
            "Floor_Type": floor_type,
            "Floor_Quality": floor_quality,
            "Kitchen_Package": kitchen_pkg,
            "Bathroom_Package": bathroom_pkg,
        }

        feats = compute_engineered_features(mini, P, ci_df)
        # show user what factors were auto-applied
        found = lookup_city_index(city, ci_df, P)

        feats = mirror_suffix_columns(feats, expected or [])
        X = align_to_model(feats, expected)

        # ML prediction (primary)
        ml_pred = pipe.predict(X)
        ml_total = to_number(ml_pred, 0.0)

        # Qty breakdown for transparency or optional modes
        base_cost = to_number(feats["Subtotal"].iloc[0], 0.0)
        qty_break = calculate_realistic_quantity_cost(base_cost)
        qty_total = to_number(qty_break["final_total_qty"], 0.0)

        # Decide output
        if mode.startswith("ML only"):
            recommended = ml_total
            lo, hi = recommended * 0.90, recommended * 1.10
            details_title = "Model prediction (primary)"
            details = {
                "ML_total": rupees(ml_total),
                "City_Multiplier_applied": found["City_Multiplier"],
                "Material_Price_Index_applied": found["Material_Price_Index"],
                "City_Tier": found.get("City_Tier", ""),
            }
        elif mode.startswith("ML + guardrails"):
            ml_clamped = calibrate_ml_against_qty(ml_total, qty_total, spec_level)
            recommended = ml_clamped
            lo, hi = recommended * 0.90, recommended * 1.10
            details_title = "ML with guardrails"
            details = {
                "ML_raw": rupees(ml_total),
                "ML_clamped": rupees(ml_clamped),
                "Qty_reference": rupees(qty_total),
                "City_Multiplier_applied": found["City_Multiplier"],
                "Material_Price_Index_applied": found["Material_Price_Index"],
                "City_Tier": found.get("City_Tier", ""),
            }
        else:  # Blended
            blend = blended_recommendation(ml_total, qty_total, spec_level)
            recommended = blend["recommended_price"]; lo, hi = blend["price_range"]
            details_title = "Blended weights"
            details = {
                "weights": blend["weights"],
                "ML_total": rupees(ml_total),
                "Qty_total": rupees(qty_total),
                "City_Multiplier_applied": found["City_Multiplier"],
                "Material_Price_Index_applied": found["Material_Price_Index"],
                "City_Tier": found.get("City_Tier", ""),
            }

        st.success("Estimate generated")
        st.markdown(f"### 💰 Estimated Project Cost: **{rupees(recommended)}**")
        st.caption(f"Budget range: {rupees(lo)} — {rupees(hi)}")
        st.caption("City factor & material index were applied automatically from cost indices.")

        with st.expander(details_title):
            st.json(details)

        with st.expander("Quantity-based breakdown (for transparency)"):
            st.json({k: rupees(v) for k, v in qty_break.items()})

        with st.expander("Debug: ML input row & expected schema"):
            st.write("Auto-applied City Index:", found)
            st.dataframe(feats.T.rename(columns={0: "value"}))
            st.write("Model expected columns (if available):")
            st.write(dedupe_preserve_order(expected) if expected is not None else "(model did not expose)")

    except Exception as e:
        st.error(f"Failed to estimate: {e}")
