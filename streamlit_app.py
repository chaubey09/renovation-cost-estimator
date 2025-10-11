# streamlit_app.py
# Renovation Cost Estimator — with Single-Room & Full-House flows,
# image upload → color palette, optional auto-suggest, and robust API fallbacks.
#
# Notes:
# - All widgets use unique `key=` values to avoid DuplicateWidgetID.
# - Works with either:
#     * Extended API: /health, /cities, /estimate-single-room, /estimate-full-house
#     * Base API (your earlier main.py): /healthz, /meta/cities, /estimate
# - If the extended endpoints are missing, the app falls back to the base /estimate endpoint.

import os
import io
import json
import requests
import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime
from typing import List, Optional, Tuple, Dict

# === Images & palette ===
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import colorsys

# === PDF / Report ===
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing, Rect

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

ROOM_TYPES = ["Living Room", "Bedroom", "Kids Room", "Kitchen", "Bathroom", "Dining"]
QUALITIES = ["Economy", "Standard", "Luxury"]  # "Luxury" maps to "Premium" internally if needed

# ---------------------------------------------------------------------
# Helpers: API
# ---------------------------------------------------------------------

def _try_get(url: str, timeout: int = 10):
    """GET -> parsed JSON or None (never raises)."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            # Some endpoints (rarely) may return text; try to coerce JSON if possible
            return json.loads(r.text)
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_cities(api_base: str) -> List[str]:
    """
    Prefer extended `/cities`, then fallback to base `/meta/cities`.
    Ensures 'Other' exists; returns sensible defaults if nothing found.
    """
    cities: List[str] = []

    # Try extended
    data = _try_get(f"{api_base}/cities")
    if isinstance(data, dict) and "cities" in data and data["cities"]:
        cities = list(data["cities"])

    # Fallback to base
    if not cities:
        data = _try_get(f"{api_base}/meta/cities")
        if isinstance(data, dict) and "cities" in data and data["cities"]:
            cities = list(data["cities"])

    # Reasonable defaults if still empty
    if not cities:
        cities = [
            "Delhi","Mumbai","Bengaluru","Hyderabad","Chennai",
            "Kolkata","Pune","Ahmedabad","Jaipur","Kanpur"
        ]

    # Always include Other
    if "Other" not in cities:
        cities.append("Other")

    # Sort with Delhi first (nice UX), then alpha
    return sorted(set(cities), key=lambda c: (c != "Delhi", c.lower()))

def health_status(api_base: str) -> str:
    """
    Accept either `/health` (extended) or `/healthz` (base).
    Returns a short, human-readable status string.
    """
    data = _try_get(f"{api_base}/health")
    if isinstance(data, dict):
        return f"API OK • {data}"

    data = _try_get(f"{api_base}/healthz")
    if isinstance(data, dict):
        return f"API OK • {data}"

    return "Health check failed"

def _normalize_quality_for_base_api(q: str) -> str:
    s = (q or "").strip().lower()
    if s in {"economy", "basic"}:
        return "Economy"
    if s in {"luxury", "premium"}:
        return "Premium"
    return "Standard"

def _fallback_single_room_payload(city: str, room_type: str, area_sqft: float, quality: str) -> dict:
    """
    Map UI inputs to the base `/estimate` payload used by your existing FastAPI.
    """
    return {
        "city": (city or "").strip() or "Other",
        "area_sqft": float(area_sqft or 0.0),
        "renovation_level": _normalize_quality_for_base_api(quality),
        "room_type": room_type or "Bedroom",
        # Optional extras available in the base API:
        "has_electrical": True,
        "ceiling_type": "None",
        # floor_type / kitchen_package / bathroom_package can be added later if surfaced in UI
    }

def call_single_room(api_base: str, payload_ext: dict) -> dict:
    """
    Try extended `/estimate-single-room`.
    If unavailable, fallback to base `/estimate` and wrap the response in a common format.
    """
    # Try extended first
    try:
        r = requests.post(f"{api_base}/estimate-single-room", json=payload_ext, timeout=30)
        if r.status_code == 200:
            return {"ok": True, "data": r.json(), "mode": "extended"}
    except Exception:
        pass

    # Fallback → base `/estimate`
    base_payload = _fallback_single_room_payload(
        city=payload_ext.get("city") or "",
        room_type=payload_ext.get("room_type") or "",
        area_sqft=payload_ext.get("area_sqft") or 0.0,
        quality=payload_ext.get("desired_quality") or "Standard",
    )
    try:
        r = requests.post(f"{api_base}/estimate", json=base_payload, timeout=30)
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else json.loads(r.text)
        meta = data.get("meta", {}) if isinstance(data, dict) else {}
        est  = float((data or {}).get("estimate", 0.0) or 0.0)
        area = float(base_payload.get("area_sqft") or 1.0)

        # Minimal compatible breakdown for UI
        bd = {
            "Grand Total": est,
            "Total Cost per Sqft": est / max(area, 1.0),
        }
        wrapped = {
            "city_used": meta.get("city", base_payload["city"]),
            "city_tier_used": meta.get("city_tier", ""),
            "city_multiplier_used": meta.get("applied_city_multiplier", 1.0),
            "inputs_mapped": {"desired_quality": base_payload["renovation_level"]},
            "breakdown": bd,
            "applied_rules": [],
        }
        return {"ok": True, "data": wrapped, "mode": "fallback-base"}
    except Exception as e:
        return {"ok": False, "error": str(e), "mode": "failed"}

def call_full_house(api_base: str, rooms: List[dict]) -> dict:
    """
    Try extended `/estimate-full-house`.
    Otherwise call base `/estimate` per room and aggregate.
    """
    # Extended pathway
    try:
        r = requests.post(f"{api_base}/estimate-full-house", json={"rooms": rooms}, timeout=60)
        if r.status_code == 200:
            return {"ok": True, "data": r.json(), "mode": "extended"}
    except Exception:
        pass

    # Fallback aggregation
    per_room = []
    total_gt = 0.0
    total_gst = 0.0   # not exposed by base endpoint
    total_oh  = 0.0   # not exposed by base endpoint

    for rinfo in rooms:
        p = _fallback_single_room_payload(
            city=rinfo.get("city", ""),
            room_type=rinfo.get("room_type", "Bedroom"),
            area_sqft=float(rinfo.get("area_sqft") or 0.0),
            quality=rinfo.get("desired_quality", "Standard"),
        )
        try:
            resp = requests.post(f"{api_base}/estimate", json=p, timeout=20)
            resp.raise_for_status()
            j = resp.json() if resp.headers.get("content-type","").startswith("application/json") else json.loads(resp.text)
            meta = (j or {}).get("meta", {})
            est  = float((j or {}).get("estimate", 0.0) or 0.0)
            area = float(p.get("area_sqft") or 1.0)
            bd = {
                "Grand Total": est,
                "Total Cost per Sqft": est / max(area, 1.0),
            }
            per_room.append({
                "breakdown": bd,
                "city_used": meta.get("city", p["city"]),
                "city_tier_used": meta.get("city_tier", ""),
                "city_multiplier_used": meta.get("applied_city_multiplier", 1.0),
                "inputs_mapped": {"desired_quality": p["renovation_level"]},
            })
            total_gt += est
        except Exception:
            # Skip problematic row but continue others
            continue

    return {
        "ok": True,
        "data": {
            "per_room": per_room,
            "total_grand_total": total_gt,
            "total_gst": total_gst,
            "total_contractor_overhead": total_oh,
        },
        "mode": "fallback-base",
    }

def money(n):
    try:
        return f"₹{round(float(n)):,.0f}"
    except Exception:
        return "—"

def clamp_round_breakdown(bd: dict) -> dict:
    """
    Normalize key names and clamp/round numerics.
    Accepts either 'Grand Total' / 'Grand_Total' and
    'Total Cost per Sqft' / 'Total_Cost_per_Sqft'.
    """
    bd = dict(bd or {})

    # Normalize keys if needed
    if "Grand_Total" in bd and "Grand Total" not in bd:
        bd["Grand Total"] = bd.pop("Grand_Total")
    if "Total_Cost_per_Sqft" in bd and "Total Cost per Sqft" not in bd:
        bd["Total Cost per Sqft"] = bd.pop("Total_Cost_per_Sqft")

    # Clamp & round all except psf (handled after)
    for k, v in list(bd.items()):
        if k != "Total Cost per Sqft":
            try:
                bd[k] = round(max(0.0, float(v)), 2)
            except Exception:
                pass

    # Compute total if missing
    comps = [c for c in bd if c not in ("Grand Total", "Total Cost per Sqft")]
    if "Grand Total" not in bd:
        try:
            bd["Grand Total"] = round(sum(float(bd[c]) for c in comps), 2)
        except Exception:
            bd["Grand Total"] = 0.0

    # Round psf if present
    if "Total Cost per Sqft" in bd:
        try:
            bd["Total Cost per Sqft"] = round(float(bd["Total Cost per Sqft"]), 2)
        except Exception:
            pass

    return bd

# ---------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------
def extract_palette(img: Image.Image, k: int = 5) -> List[Tuple[int,int,int]]:
    img_small = img.convert("RGB").resize((200, 200))
    arr = np.array(img_small).reshape(-1, 3)
    km = KMeans(n_clusters=min(k, len(arr)), n_init="auto", random_state=42).fit(arr)
    labels, counts = np.unique(km.labels_, return_counts=True)
    order = counts.argsort()[::-1]
    centers = km.cluster_centers_.astype(int)
    return [tuple(map(int, centers[i])) for i in labels[order]]

def swatch_hex(rgb: Tuple[int,int,int]) -> str:
    return '#%02x%02x%02x' % rgb

def aggregate_master_palette(per_image_palettes: List[List[Tuple[int,int,int]]], k: int = 5):
    flat = [rgb for pal in per_image_palettes for rgb in pal]
    if not flat:
        return []
    km = KMeans(n_clusters=min(k, len(flat)), n_init="auto", random_state=42).fit(np.array(flat))
    centers = km.cluster_centers_.astype(int)
    return [tuple(map(int, c)) for c in centers]

# ---------------------------------------------------------------------
# Style recommendations (coarse)
# ---------------------------------------------------------------------
def style_from_palette(palette):
    if not palette:
        return "Soft Neutral"
    hues = [colorsys.rgb_to_hsv(r/255, g/255, b/255)[0] for r, g, b in palette]
    avg_h = float(np.mean(hues))
    if avg_h < 0.08 or avg_h > 0.92:
        return "Warm Minimal"
    if 0.30 < avg_h < 0.60:
        return "Earthy Contemporary"
    if 0.60 <= avg_h <= 0.80:
        return "Cool Modern"
    return "Soft Neutral"

def recommend_finishes(palette):
    if not palette:
        return {"base_paint":"Soft Grey","accent_paint":"Charcoal","ceiling":"Off-White (lighter than walls)","floor_tone":"Neutral Medium (matte)"}
    hsvs = [colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in palette]
    avg_h = float(np.mean([h for h, s, v in hsvs])); avg_s = float(np.mean([s for h, s, v in hsvs])); avg_v = float(np.mean([v for h, s, v in hsvs]))
    if   (avg_h < 0.08 or avg_h > 0.92): base, accent, floor = "Warm White / Ivory", "Terracotta / Burnt Sienna", "Medium Warm (Oak/Teak look)"
    elif (0.60 <= avg_h <= 0.80):        base, accent, floor = "Snow White / Cool Grey", "Teal / Navy", "Light Cool (Ash/White Oak look)"
    elif (0.30 <  avg_h < 0.60):         base, accent, floor = "Greige / Alabaster", "Sage / Olive", "Neutral Medium (matte)"
    else:                                base, accent, floor = "Soft Grey / Linen", "Charcoal / Dusty Blue", "Neutral Medium (matte)"
    if avg_v < 0.45:
        base = base.split("/")[0].strip() + " (lighter)"
    if avg_s > 0.55:
        base = "Neutral Off-White"
    return {"base_paint": base, "accent_paint": accent, "ceiling": "Lighter than walls (same hue family)", "floor_tone": floor}

# ---------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------
def _fmt_money(n):
    try:
        return f"₹{round(float(n)):,.0f}"
    except Exception:
        return "—"

def _df_to_table_data(df, amount_col="Amount (₹)"):
    rows = [["Cost Head", amount_col]]
    for _, r in df.iterrows():
        rows.append([str(r["Cost Head"]), _fmt_money(r[amount_col])])
    return rows

def _table_style(header_bold=True):
    return TableStyle([
        ("FONT", (0,0), (-1,0), "Helvetica-Bold" if header_bold else "Helvetica", 10),
        ("FONT", (0,1), (-1,-1), "Helvetica", 9),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
    ])

def palette_row(palette, width=540, height=20):
    d = Drawing(width, height)
    n = max(1, len(palette)); w = width / n
    for i, (r, g, b) in enumerate(palette):
        d.add(Rect(i*w, 0, w, height, fillColor=colors.Color(r/255, g/255, b/255),
                   strokeColor=colors.black, strokeWidth=0.1))
    return d

def _two_col_table(rows, colw=(200, 320)):
    t = Table(rows, hAlign="LEFT", colWidths=list(colw))
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
    ]))
    return t

def build_single_room_pdf(meta: dict, bd_df: pd.DataFrame, area_sqft: float,
                          images_for_pdf: Optional[List[bytes]] = None,
                          palette_for_pdf: Optional[List[Tuple[int,int,int]]] = None,
                          recommendations: Optional[dict] = None) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet(); story = []
    story += [Paragraph("Renovation Cost Estimate — Single Room", styles["Title"]), Spacer(1,6)]
    story += [Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]), Spacer(1,12)]
    city = meta.get("city_used","—"); tier = meta.get("city_tier_used","—")
    mult = meta.get("city_multiplier_used",1.0); dq = meta.get("inputs_mapped",{}).get("desired_quality","—")
    used_default = "Yes" if meta.get("used_default_for_city") else "No"
    meta_tbl = Table([["City",city,"City Tier",tier],
                      ["City Multiplier",f"{float(mult):.2f}","Used Default Tier-3?",used_default],
                      ["Desired Quality",dq,"Area (sq ft)",f"{area_sqft:.0f}"]],
                     hAlign="LEFT", colWidths=[105,165,105,165])
    meta_tbl.setStyle(_table_style()); story += [meta_tbl, Spacer(1,12)]
    try:
        grand_total = float(bd_df.loc[bd_df["Cost Head"]=="Grand Total","Amount (₹)"].values[0])
    except Exception:
        grand_total = 0.0
    try:
        cost_sqft = float(bd_df.loc[bd_df["Cost Head"]=="Total Cost per Sqft","Amount (₹)"].values[0])
    except Exception:
        cost_sqft = grand_total / max(area_sqft, 1.0)
    kpi_tbl = Table([["Grand Total", _fmt_money(grand_total), "₹ / sq ft", f"{cost_sqft:,.2f}"]],
                    hAlign="LEFT", colWidths=[105,165,105,165])
    kpi_tbl.setStyle(TableStyle([
        ("FONT",(0,0),(-1,-1),"Helvetica-Bold",11),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("BOX",(0,0),(-1,-1),0.5,colors.grey),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),8),
    ]))
    story += [kpi_tbl, Spacer(1,12)]
    if images_for_pdf:
        story.append(Paragraph("Inspiration Images", styles["Heading2"]))
        row = []
        for b in images_for_pdf[:4]:
            buf = io.BytesIO(b)
            row.append(RLImage(buf, width=240, height=160))
            if len(row) == 2:
                story += row + [Spacer(1,6)]; row = []
        if row:
            story += row + [Spacer(1,6)]
    if palette_for_pdf:
        story.append(Paragraph("Suggested Palette", styles["Heading2"]))
        story.append(palette_row(palette_for_pdf)); story.append(Spacer(1,10))
    if recommendations:
        story.append(Paragraph("Recommended Finishes", styles["Heading2"]))
        rows = [["Item","Suggestion"],
                ["Base Paint", recommendations.get("base_paint","—")],
                ["Accent Wall", recommendations.get("accent_paint","—")],
                ["Ceiling", recommendations.get("ceiling","—")],
                ["Floor Tone", recommendations.get("floor_tone","—")]]
        story.append(_two_col_table(rows)); story.append(Spacer(1,10))
    table_df = bd_df.copy()
    table_df["_order"] = table_df["Cost Head"].apply(lambda x: 999 if "Total" in x else 0)
    table_df = table_df.sort_values(["_order","Cost Head"]).drop(columns=["_order"])
    tbl = Table(_df_to_table_data(table_df), hAlign="LEFT", colWidths=[320,220])
    tbl.setStyle(_table_style(header_bold=True))
    story += [Paragraph("Cost Breakdown", styles["Heading2"]), tbl]
    doc.build(story); buffer.seek(0); return buffer.getvalue()

def build_full_house_pdf(summary: dict) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet(); story=[]
    story += [Paragraph("Renovation Cost Estimate — Full House", styles["Title"]), Spacer(1,6)]
    story += [Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", styles["Normal"]), Spacer(1,12)]
    total = float(summary.get("total_grand_total",0.0)); gst = float(summary.get("total_gst",0.0)); oh = float(summary.get("total_contractor_overhead",0.0))
    kpi_tbl = Table([["Grand Total", _fmt_money(total), "Total GST", _fmt_money(gst)],
                     ["Contractor Overhead", _fmt_money(oh), "", ""]],
                     hAlign="LEFT", colWidths=[150,150,150,150])
    kpi_tbl.setStyle(TableStyle([
        ("FONT",(0,0),(-1,-1),"Helvetica-Bold",11),
        ("BACKGROUND",(0,0),(-1,0),colors.whitesmoke),
        ("BOX",(0,0),(-1,-1),0.5,colors.grey),
        ("INNERGRID",(0,0),(-1,-1),0.25,colors.grey),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("TOPPADDING",(0,0),(-1,-1),8),
    ]))
    story += [kpi_tbl, Spacer(1,12)]
    for idx, room in enumerate(summary.get("per_room", []), start=1):
        story += [Paragraph(f"Room {idx}", styles["Heading2"])]
        city = room.get("city_used","—"); tier = room.get("city_tier_used","—"); mult = room.get("city_multiplier_used",1.0)
        dq = room.get("inputs_mapped",{}).get("desired_quality","—")
        meta_tbl = Table([["City",city,"City Tier",tier],
                          ["City Multiplier",f"{float(mult):.2f}","Used Default Tier-3?","Yes" if room.get("used_default_for_city") else "No"],
                          ["Desired Quality",dq,"",""]],
                         hAlign="LEFT", colWidths=[105,165,105,165])
        meta_tbl.setStyle(_table_style()); story += [meta_tbl, Spacer(1,6)]
        bd = room.get("breakdown",{})
        df_room = pd.DataFrame([{"Cost Head": k.replace("_"," ") if isinstance(k,str) else str(k), "Amount (₹)": float(v)} for k,v in bd.items()])
        df_room["_order"] = df_room["Cost Head"].apply(lambda x: 999 if "Total" in x else 0)
        df_room = df_room.sort_values(by=["_order","Cost Head"]).drop(columns=["_order"])
        tbl = Table(_df_to_table_data(df_room), hAlign="LEFT", colWidths=[320,220])
        tbl.setStyle(_table_style(header_bold=True)); story += [tbl, Spacer(1,10)]
    doc.build(story); buffer.seek(0); return buffer.getvalue()

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Renovation Cost Estimator", page_icon="🧮", layout="wide")
st.title("🧮 Renovation Cost Estimator")

with st.sidebar:
    st.subheader("Backend")
    api_input = st.text_input("API URL", value=API_URL, key="sb_api_url")
    if api_input != API_URL:
        API_URL = api_input
        fetch_cities.clear()  # refresh cache on URL change
    st.divider()
    msg = health_status(API_URL)
    if msg.startswith("API OK"):
        st.success(msg)
    else:
        st.error(msg)

tabs = st.tabs(["Single Room", "Full House", "About"])

# ---------------------------------------------------------------------
# Single Room Tab
# ---------------------------------------------------------------------
with tabs[0]:
    st.subheader("Estimate for a Single Room")
    cities = fetch_cities(API_URL)
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        city_sel = st.selectbox(
            "City",
            options=cities,
            index=(cities.index("Delhi") if "Delhi" in cities else 0),
            key="sr_city",
        )
    with c2:
        room = st.selectbox("Room Type", options=ROOM_TYPES, index=0, key="sr_room_type")
    with c3:
        area = st.number_input("Room Size (sq ft)", min_value=20.0, value=200.0, step=10.0, key="sr_area")
    with c4:
        quality = st.selectbox("Desired Quality", options=QUALITIES, index=1, key="sr_quality")

    # Inspiration / image upload
    st.subheader("🖼️ Upload reference image(s) (optional)")
    uploads = st.file_uploader("Upload 1–6 images", type=["jpg","jpeg","png"], accept_multiple_files=True, key="sr_uploads")
    uploads = uploads[:6] if uploads else []
    per_img_cols = st.columns(3)
    per_image_palettes: List[List[Tuple[int,int,int]]] = []
    first_image = None
    for i, f in enumerate(uploads):
        with per_img_cols[i % 3]:
            img = Image.open(f).convert("RGB")
            if first_image is None:
                first_image = img.copy()
            st.image(img, use_container_width=True, caption=f"Image {i+1}")
            pal = extract_palette(img, k=5)
            per_image_palettes.append(pal)
            wcols = st.columns(len(pal))
            for j, rgb in enumerate(pal):
                wcols[j].markdown(
                    f"<div style='height:28px;background:{swatch_hex(rgb)};border-radius:4px'></div>",
                    unsafe_allow_html=True,
                )

    master_palette = aggregate_master_palette(per_image_palettes, k=5)
    if master_palette:
        st.markdown("**Overall Palette**")
        scols = st.columns(len(master_palette))
        for j, rgb in enumerate(master_palette):
            scols[j].markdown(f"<div style='height:32px;background:{swatch_hex(rgb)};border-radius:4px'></div>", unsafe_allow_html=True)
        st.info(f"Style vibe: **{style_from_palette(master_palette)}**")
        recs = recommend_finishes(master_palette)
        st.caption(f"Suggested finishes → Base: {recs['base_paint']} • Accent: {recs['accent_paint']} • Ceiling: {recs['ceiling']} • Floor: {recs['floor_tone']}")
    else:
        recs = None

    st.subheader("Auto-Suggest features")
    autos = {}
    if first_image and st.button("🔎 Suggest from image", key="sr_suggest_from_image"):
        with st.spinner("Analyzing image…"):
            # Lazy import best-effort (won't crash if transformers/torch not installed)
            try:
                import torch  # noqa
                from transformers import CLIPProcessor, CLIPModel  # noqa
                # The heavy auto-suggest from your prior app can be plugged here
                # For now, keep disabled to avoid runtime download on end-user machines
                autos = {"flags": {}, "qty": {}}  # replace with real inference if you enable CLIP
                st.info("Auto-suggest is currently a placeholder to avoid heavy model downloads.")
            except Exception:
                st.warning("Transformers/torch not installed. Skipping auto-suggest.")
                autos = {"flags": {}, "qty": {}}

    st.subheader("Select your features")
    st.caption("Tick what you want and set quantities. The estimate will reflect these components explicitly.")
    # Feature flags
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        flag_false_ceiling = st.checkbox("False ceiling + cove LEDs", value=bool(autos.get("flags",{}).get("false_ceiling", True)), key="sr_false_ceiling")
        flag_wall_panel    = st.checkbox("Wall panel / art feature", value=bool(autos.get("flags",{}).get("wall_panel_art", False)), key="sr_wall_panel")
    with f2:
        flag_headboard     = st.checkbox("Upholstered headboard", value=bool(autos.get("flags",{}).get("upholstered_headboard", room=="Bedroom")), key="sr_headboard")
        flag_study         = st.checkbox("Study desk", value=bool(autos.get("flags",{}).get("study_desk", False)), key="sr_study")
    with f3:
        flag_bed_q         = st.checkbox("Queen bed", value=bool(autos.get("flags",{}).get("bed_queen", False)), key="sr_bed_q")
        flag_bed_k         = st.checkbox("King bed", value=bool(autos.get("flags",{}).get("bed_king", room=="Bedroom")), key="sr_bed_k")
    with f4:
        flag_tv            = st.checkbox("TV unit", value=bool(autos.get("flags",{}).get("tv_unit", False)), key="sr_tv_unit")
        flag_curtains      = st.checkbox("Premium curtains", value=bool(autos.get("qty",{}).get("curtains_width_ft", 0) > 0), key="sr_curtains")

    st.markdown("**Quantities**")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        wardrobe_rft   = st.number_input("Wardrobe (running ft)", min_value=0.0, value=float(autos.get("qty",{}).get("wardrobe_rft", 0.0)), step=1.0, key="sr_qty_wardrobe")
    with q2:
        wallpaper_sqft = st.number_input("Wallpaper (sq ft)", min_value=0.0, value=float(autos.get("qty",{}).get("wallpaper_sqft", 0.0)), step=10.0, key="sr_qty_wallpaper")
    with q3:
        curtains_width = st.number_input("Curtains width (ft)", min_value=0.0, value=float(autos.get("qty",{}).get("curtains_width_ft", 0.0)), step=1.0, key="sr_qty_curtains")
    with q4:
        mirror_count   = st.number_input("Decor mirrors (count)", min_value=0, value=int(autos.get("qty",{}).get("mirror_count", 0)), step=1, key="sr_qty_mirrors")

    feature_flags = {
        "false_ceiling": flag_false_ceiling,
        "wall_panel_art": flag_wall_panel,
        "upholstered_headboard": flag_headboard,
        "study_desk": flag_study,
        "tv_unit": flag_tv,
        "bed_queen": flag_bed_q and not flag_bed_k,
        "bed_king": flag_bed_k,
    }
    feature_quantities = {
        "wardrobe_rft": wardrobe_rft,
        "wallpaper_sqft": wallpaper_sqft,
        "curtains_width_ft": curtains_width,
        "mirror_count": float(mirror_count),
    }

    # City override
    if city_sel == "Other":
        city_to_send = st.text_input("Enter City Name (Tier-3 default):", value="", placeholder="e.g., Nashik", key="sr_city_other").strip() or "Other"
    else:
        city_to_send = city_sel

    if st.button("Estimate", type="primary", key="sr_estimate_btn"):
        with st.spinner("Estimating..."):
            # Extended payload (if backend supports it)
            payload = {
                "city": city_to_send,
                "room_type": room,
                "area_sqft": float(area),
                "desired_quality": quality,
                "feature_flags": feature_flags,
                "feature_quantities": feature_quantities,
            }
            result = call_single_room(API_URL, payload)
            if not result.get("ok"):
                st.error(f"Request failed: {result.get('error','Unknown error')}")
            else:
                mode = result.get("mode")
                resp = result["data"]
                st.success(f"Estimate ready ({mode}).")
                meta = st.columns(4)
                meta[0].metric("City Used", resp.get("city_used","—"))
                meta[1].metric("City Tier", resp.get("city_tier_used","—"))
                meta[2].metric("City Multiplier", f"{resp.get('city_multiplier_used',0):.2f}")
                meta[3].metric("₹ / sq ft", f"{resp.get('breakdown',{}).get('Total Cost per Sqft',0):,.2f}")

                # Feature adds (extended only)
                if resp.get("applied_rules"):
                    st.markdown("**Feature-wise additions (this estimate)**")
                    for rule in resp["applied_rules"]:
                        label = rule.get("desc","")
                        if "qty" in rule:
                            label += f" — qty: {rule['qty']}"
                        add_sum = sum(rule.get("amounts",{}).values())
                        st.write(f"- {label}: ₹{add_sum:,.0f}")

                bd = clamp_round_breakdown(resp.get("breakdown", {}))
                df = pd.DataFrame([{"Cost Head": k.replace("_"," ") if isinstance(k,str) else str(k), "Amount (₹)": float(v)} for k,v in bd.items()])
                df["_order"] = df["Cost Head"].apply(lambda x: 999 if "Total" in x else 0)
                df = df.sort_values(by=["_order","Cost Head"]).drop(columns=["_order"])
                st.dataframe(df, use_container_width=True)

                chart_df = df[~df["Cost Head"].isin(["Grand Total", "Total Cost per Sqft"])].set_index("Cost Head")
                st.bar_chart(chart_df["Amount (₹)"])

                # Build PDF (embed images/palette if present)
                images_for_pdf = []
                for up in uploads[:4]:
                    try:
                        images_for_pdf.append(up.getvalue())
                    except Exception:
                        pass
                pdf_bytes = build_single_room_pdf(
                    resp, df, area_sqft=area,
                    images_for_pdf=images_for_pdf if images_for_pdf else None,
                    palette_for_pdf=master_palette if master_palette else None,
                    recommendations=recs if master_palette else None,
                )
                st.download_button(
                    "📄 Download PDF",
                    data=pdf_bytes,
                    file_name=f"estimate_single_{resp.get('city_used','city')}_{room.replace(' ','_')}.pdf",
                    mime="application/pdf",
                    key="sr_pdf_dl",
                )

# ---------------------------------------------------------------------
# Full House Tab
# ---------------------------------------------------------------------
with tabs[1]:
    st.subheader("Estimate for a Full House")
    st.caption("Add rooms below and click Estimate.")

    if "rooms" not in st.session_state:
        st.session_state.rooms = [
            {"city":"Delhi","room_type":"Living Room","area_sqft":200.0,"desired_quality":"Standard","design_premium_pct":10.0},
            {"city":"Kanpur","room_type":"Kitchen","area_sqft":120.0,"desired_quality":"Economy","design_premium_pct":5.0},
        ]

    def add_room():
        st.session_state.rooms.append({"city":"Delhi","room_type":"Bedroom","area_sqft":150.0,"desired_quality":"Standard","design_premium_pct":10.0})
    def clear_rooms():
        st.session_state.rooms = []

    c_add, c_clear = st.columns([1, 1])
    with c_add:
        st.button("➕ Add Room", on_click=add_room, key="fh_add_btn")
    with c_clear:
        st.button("🧹 Clear", on_click=clear_rooms, key="fh_clear_btn")

    editable = st.data_editor(
        pd.DataFrame(st.session_state.rooms),
        key="fh_rooms_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "city": st.column_config.TextColumn(),
            "room_type": st.column_config.SelectboxColumn(options=ROOM_TYPES),
            "area_sqft": st.column_config.NumberColumn(min_value=20, step=5),
            "desired_quality": st.column_config.SelectboxColumn(options=QUALITIES),
            "design_premium_pct": st.column_config.NumberColumn(min_value=0, max_value=50, step=1),
        },
    )

    if st.button("Estimate Full House", type="primary", key="fh_estimate_btn"):
        rooms_payload = editable.to_dict(orient="records")
        for r in rooms_payload:
            r["area_sqft"] = float(r.get("area_sqft",0) or 0)
            r["design_premium_pct"] = float(r.get("design_premium_pct",0) or 0)
            if r["area_sqft"] <= 0:
                st.error("All rooms must have area > 0.")
                st.stop()
        with st.spinner("Estimating full house..."):
            result = call_full_house(API_URL, rooms_payload)
            if not result.get("ok"):
                st.error(f"Request failed: {result.get('error','Unknown error')}")
            else:
                mode = result.get("mode")
                resp = result["data"]
                total = resp.get("total_grand_total",0.0); gst = resp.get("total_gst",0.0); oh = resp.get("total_contractor_overhead",0.0)
                c1, c2, c3 = st.columns(3)
                c1.metric("Full House Grand Total", money(total))
                c2.metric("Total GST", money(gst))
                c3.metric("Contractor Overhead", money(oh))
                st.success(f"Estimate ready ({mode}).")
                st.markdown("### Per-room breakdown")
                for i, rr in enumerate(resp.get("per_room", []), start=1):
                    st.markdown(f"**Room {i}: {rr.get('city_used','')} — {rr.get('inputs_mapped',{}).get('desired_quality','')}**")
                    meta = st.columns(3)
                    meta[0].caption(f"Tier: {rr.get('city_tier_used','—')}")
                    meta[1].caption(f"Multiplier: {rr.get('city_multiplier_used',0):.2f}")
                    meta[2].caption(f"₹/sqft: {rr.get('breakdown',{}).get('Total Cost per Sqft',0):,.2f}")
                    df_room = pd.DataFrame([{"Cost Head": k.replace("_"," ") if isinstance(k,str) else str(k), "Amount (₹)": float(v)} for k,v in rr.get("breakdown",{}).items()])
                    df_room["_order"] = df_room["Cost Head"].apply(lambda x: 999 if "Total" in x else 0)
                    df_room = df_room.sort_values(by=["_order","Cost Head"]).drop(columns=["_order"])
                    st.dataframe(df_room, use_container_width=True)

# ---------------------------------------------------------------------
# About Tab
# ---------------------------------------------------------------------
with tabs[2]:
    st.subheader("About")
    st.markdown("""
- Upload images to get a color palette and optional **Auto-Suggest (beta)** of features (disabled by default to avoid heavy downloads).
- Toggle **features & quantities** to see transparent price changes (wardrobe rft, wallpaper sqft, curtains width, etc.).
- Backend endpoints supported:
  - **Preferred**: `GET /health`, `GET /cities`, `POST /estimate-single-room`, `POST /estimate-full-house`
  - **Fallback**: `GET /healthz`, `GET /meta/cities`, `POST /estimate`
""")
