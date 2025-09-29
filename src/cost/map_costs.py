# src/cost/map_costs.py
import argparse
import os
import json
import pandas as pd
import numpy as np
import yaml

# ------------------ helpers ------------------

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_optional_csv(path):
    return pd.read_csv(path) if (path and os.path.exists(path)) else None

def series_or_default(df: pd.DataFrame, col: str, default):
    """Return df[col] if present else a default-filled Series of same length."""
    if col in df.columns:
        try:
            return df[col].fillna(default)
        except Exception:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)

def get_pkg_cost(row, packages_df, defaults, kind):
    """
    kind: 'kitchen' or 'bathroom'
    expects on row: 'Kitchen_Package' / 'Bathroom_Package'
    packages_df (optional) columns: kind, code, price
    """
    key = f"{kind}_Package"
    val = str(row.get(key, "None"))
    if packages_df is not None and key in row.index:
        hits = packages_df[
            (packages_df["kind"].str.lower() == kind) & (packages_df["code"] == val)
        ]
        if len(hits):
            return float(hits["price"].iloc[0])
    return float(defaults.get(val, 0))


# ------------------ main ------------------

def main(tab_pq, qty_pq, cost_index_csv, packages_csv, outp, metrics_path, plots_dir,
         preview_csv="", preview_n=200):
    P = load_params()
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    if preview_csv:
        os.makedirs(os.path.dirname(preview_csv), exist_ok=True)

    # Load inputs
    tab = pd.read_parquet(tab_pq)
    qty = pd.read_parquet(qty_pq)

    # Merge tabular + quantities
    on_cols = [c for c in ["Row_ID", "Room_Type"] if (c in tab.columns) and (c in qty.columns)]
    if not on_cols:
        if "Row_ID" in tab.columns and "Row_ID" in qty.columns:
            on_cols = ["Row_ID"]
        else:
            on_cols = list(set(tab.columns).intersection(qty.columns))
    df = tab.merge(qty, on=on_cols, how="left")

    # Optional city/material index table (columns: City, City_Multiplier, Material_Price_Index)
    ci = read_optional_csv(cost_index_csv)
    if ci is not None and ("City" in df.columns) and ("City" in ci.columns):
        df = df.drop(columns=["City_Multiplier", "Material_Price_Index"], errors="ignore") \
               .merge(ci, on="City", how="left")

    # Base indices with fallbacks
    df["Material_Price_Index"] = pd.to_numeric(
        series_or_default(df, "Material_Price_Index", P["rates"]["base_mpi"]), errors="coerce"
    ).fillna(P["rates"]["base_mpi"])
    df["City_Multiplier"] = pd.to_numeric(
        series_or_default(df, "City_Multiplier", 1.0), errors="coerce"
    ).fillna(1.0)

    # Robust category inputs
    pq = series_or_default(df, "Paint_Quality", "Standard").astype(str)
    ft = series_or_default(df, "Floor_Type", "Ceramic_Tile").astype(str)
    fq = series_or_default(df, "Floor_Quality", "Standard").astype(str)
    ct = series_or_default(df, "Ceiling_Type", "None").astype(str)
    cq = series_or_default(df, "Ceiling_Quality", "Standard").astype(str)

    # Quantities -> numeric
    df["paintable_area_sqft"] = pd.to_numeric(series_or_default(df, "paintable_area_sqft", 0), errors="coerce").fillna(0)
    df["flooring_area_sqft"] = pd.to_numeric(series_or_default(df, "flooring_area_sqft", 0), errors="coerce").fillna(0)
    df["false_ceiling_area_sqft"] = pd.to_numeric(series_or_default(df, "false_ceiling_area_sqft", 0), errors="coerce").fillna(0)
    df["electrical_points"] = pd.to_numeric(series_or_default(df, "electrical_points", 0), errors="coerce").fillna(0)

    # ---------------- Painting ----------------
    paint_mat_tbl = P["rates"]["painting"]["material_per_sqft_by_quality"]
    df["paint_mat_rate"] = pq.map(paint_mat_tbl).fillna(paint_mat_tbl["Standard"])
    df["paint_lab_rate"] = P["rates"]["painting"]["labor_per_sqft"]

    df["Painting_Material_Est"] = (
        df["paintable_area_sqft"] * df["paint_mat_rate"] * df["Material_Price_Index"] * df["City_Multiplier"]
    )
    df["Painting_Labor_Est"] = (
        df["paintable_area_sqft"] * df["paint_lab_rate"] * df["City_Multiplier"]
    )

    # ---------------- Flooring ----------------
    floor_tbl = P["rates"]["flooring"]["material_per_sqft_by_type_quality"]

    def floor_rate(t, q):
        base = floor_tbl.get(t, floor_tbl["Ceramic_Tile"])
        return base.get(q, base["Standard"])

    df["floor_mat_rate"] = [floor_rate(t, q) for t, q in zip(ft, fq)]
    df["floor_lab_rate"] = P["rates"]["flooring"]["labor_per_sqft"]

    df["Flooring_Material_Est"] = (
        df["flooring_area_sqft"] * df["floor_mat_rate"] * df["Material_Price_Index"] * df["City_Multiplier"]
    )
    df["Flooring_Labor_Est"] = (
        df["flooring_area_sqft"] * df["floor_lab_rate"] * df["City_Multiplier"]
    )

    # ---------------- Ceiling -----------------
    ceil_tbl = P["rates"]["ceiling"]["material_per_sqft_by_type_quality"]

    def ceil_rate(t, q):
        if t == "None":
            return 0.0
        base = ceil_tbl.get(t, ceil_tbl["POP"])
        return base.get(q, base["Standard"])

    df["ceil_mat_rate"] = [ceil_rate(t, q) for t, q in zip(ct, cq)]
    df["ceil_lab_rate"] = P["rates"]["ceiling"]["labor_per_sqft"]

    df["Ceiling_Material_Est"] = (
        df["false_ceiling_area_sqft"] * df["ceil_mat_rate"] * df["Material_Price_Index"] * df["City_Multiplier"]
    )
    df["Ceiling_Labor_Est"] = (
        df["false_ceiling_area_sqft"] * df["ceil_lab_rate"] * df["City_Multiplier"]
    )

    # --------------- Electrical --------------
    df["Electrical_Material_Est"] = (
        df["electrical_points"] * P["rates"]["electrical"]["per_point_material"] * df["Material_Price_Index"] * df["City_Multiplier"]
    )
    df["Electrical_Labor_Est"] = (
        df["electrical_points"] * P["rates"]["electrical"]["per_point_labor"] * df["City_Multiplier"]
    )

    # ---------------- Packages ----------------
    packages_df = read_optional_csv(packages_csv)
    df["Kitchen_Package_Cost_Est"] = df.apply(
        lambda r: get_pkg_cost(r, packages_df, P["rates"]["packages"]["kitchen"], "kitchen"), axis=1
    )
    df["Bathroom_Package_Cost_Est"] = df.apply(
        lambda r: get_pkg_cost(r, packages_df, P["rates"]["packages"]["bathroom"], "bathroom"), axis=1
    )

    # ---------------- Totals ------------------
    cols = [
        "Painting_Material_Est", "Painting_Labor_Est",
        "Flooring_Material_Est", "Flooring_Labor_Est",
        "Ceiling_Material_Est", "Ceiling_Labor_Est",
        "Electrical_Material_Est", "Electrical_Labor_Est",
        "Kitchen_Package_Cost_Est", "Bathroom_Package_Cost_Est",
    ]
    df["Subtotal"] = df[cols].sum(axis=1)
    df["Wastage_Sundries_Est"] = df["Subtotal"] * float(P["cost"]["wastage_pct"])
    df["Overhead_Est"] = (df["Subtotal"] + df["Wastage_Sundries_Est"]) * float(P["cost"]["contractor_overhead_pct"])
    df["PreGST_Total_Est"] = df["Subtotal"] + df["Wastage_Sundries_Est"] + df["Overhead_Est"]
    df["GST_Est"] = df["PreGST_Total_Est"] * float(P["cost"]["gst_rate"])
    df["Grand_Total_Quantity"] = df["PreGST_Total_Est"] + df["GST_Est"]

    # ---- Per-sqft with robust area estimation (keep Series, avoid np.where) ----
    # 1) Start with raw area
    area_est = pd.to_numeric(series_or_default(df, "Area_Sqft", 0), errors="coerce").fillna(0)

    # 2) Fallback: use flooring area where area <= 0
    floor_area = pd.to_numeric(df["flooring_area_sqft"], errors="coerce").fillna(0)
    area_est = area_est.mask(area_est <= 0, floor_area)

    # 3) Fallback: back-calc from paintable area using room-specific multiplier
    needs_back = area_est <= 0
    if needs_back.any():
        mult_map = P["quant_rules"]["paint_wall_multiplier_by_room"]
        room = df["Room_Type"] if "Room_Type" in df.columns else pd.Series([""] * len(df), index=df.index)
        mults = room.astype(str).map(lambda rt: mult_map.get(rt, mult_map.get("default", 3.2))).replace(0, np.nan)
        back = (df["paintable_area_sqft"] / mults).replace([np.inf, -np.inf], np.nan).fillna(0)
        area_est = area_est.mask(needs_back, back)

    area = pd.to_numeric(area_est, errors="coerce").fillna(0)
    df["Total_Cost_per_Sqft_Q"] = np.where(area > 0, df["Grand_Total_Quantity"] / area, np.nan)

    # Save breakdown
    df.to_parquet(outp, index=False)

    # ---------------- Metrics (robust) ----------------
    vals = pd.to_numeric(df["Total_Cost_per_Sqft_Q"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if vals.isna().all():
        # Fallback recompute
        total = pd.to_numeric(df.get("Grand_Total_Quantity", np.nan), errors="coerce")
        raw_area = pd.to_numeric(df.get("Area_Sqft", 0), errors="coerce").fillna(0)
        psf = total.div(raw_area).replace([np.inf, -np.inf], np.nan)
        vals = psf

    mean_val = float(np.nanmean(vals)) if len(vals) else float("nan")

    if "City" in df.columns:
        by_city_series = df.groupby("City")["Total_Cost_per_Sqft_Q"].mean(numeric_only=True)
        by_city = by_city_series.replace([np.inf, -np.inf], np.nan).dropna().round(2).to_dict()
    else:
        by_city = {}

    metrics = {
        "n_rows": int(len(df)),
        "mean_cost_per_sqft_Q": mean_val,
        "by_city_mean_cost_per_sqft_Q": by_city,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # ---------------- Plots -------------------
    if "City" in df.columns:
        by_city_df = df.groupby("City", as_index=False)["Total_Cost_per_Sqft_Q"].mean().round(2)
        by_city_df.to_csv(os.path.join(plots_dir, "cost_per_sqft_by_city.csv"), index=False)
    else:
        pd.DataFrame({"City": [], "Total_Cost_per_Sqft_Q": []}).to_csv(
            os.path.join(plots_dir, "cost_per_sqft_by_city.csv"), index=False
        )

    # ---------------- Preview CSV ----------------
    if preview_csv:
        cols_preview = [
            "Row_ID","City","Room_Type","Area_Sqft",
            "Painting_Material_Est","Painting_Labor_Est",
            "Flooring_Material_Est","Flooring_Labor_Est",
            "Ceiling_Material_Est","Ceiling_Labor_Est",
            "Electrical_Material_Est","Electrical_Labor_Est",
            "Kitchen_Package_Cost_Est","Bathroom_Package_Cost_Est",
            "Grand_Total_Quantity","Total_Cost_per_Sqft_Q"
        ]
        existing = [c for c in cols_preview if c in df.columns]
        df[existing].head(int(preview_n)).to_csv(preview_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--qty", required=True)
    ap.add_argument("--cost-index", default="")
    ap.add_argument("--packages", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--plots", required=True)
    ap.add_argument("--preview-csv", default="")
    ap.add_argument("--preview-n", type=int, default=200)
    args = ap.parse_args()
    main(args.tab, args.qty, args.cost_index, args.packages, args.out, args.metrics, args.plots,
         args.preview_csv, args.preview_n)
