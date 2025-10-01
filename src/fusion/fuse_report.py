# src/fusion/fuse_report.py
import argparse
import json
import os
import pandas as pd
import numpy as np
import yaml


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_parent_dir(path: str):
    d = os.path.dirname(path or "")
    if d:
        os.makedirs(d, exist_ok=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def main(
    tab_pq: str,
    cost_pq: str,
    outp: str,
    metrics_path: str,
    preview_csv: str = "",
    preview_n: int = 200,
    ml_pq: str = "",
):
    _ensure_parent_dir(outp)
    if preview_csv:
        _ensure_parent_dir(preview_csv)
    if metrics_path:
        _ensure_parent_dir(metrics_path)

    # ---- config ----
    P = load_params()
    F = P.get("fusion", {})
    w_tab = float(F.get("w_tabular", 0.5))
    w_qty = float(F.get("w_quantity", 0.5))
    w_ml = float(F.get("w_ml", 0.0))  # new weight for ML
    calib = float(F.get("calibration_factor", 1.0))

    # ---- load inputs ----
    cost = pd.read_parquet(cost_pq)  # M3 output
    df = cost.copy()

    # Bring in tab baselines if present
    tab = pd.read_parquet(tab_pq)
    if "Row_ID" in tab.columns and "Row_ID" in df.columns:
        base_cols = []
        if "Grand_Total" in tab.columns:
            base_cols.append("Grand_Total")
        if "Total_Cost_per_Sqft" in tab.columns:
            base_cols.append("Total_Cost_per_Sqft")
        if base_cols:
            base = tab[["Row_ID"] + base_cols].copy()
            if "Grand_Total" in base.columns:
                base.rename(columns={"Grand_Total": "Grand_Total_Tab"}, inplace=True)
            if "Total_Cost_per_Sqft" in base.columns:
                base.rename(columns={"Total_Cost_per_Sqft": "Cost_per_Sqft_Tab"}, inplace=True)
            df = df.merge(base, on="Row_ID", how="left")

    # Optionally merge ML predictions: expects Row_ID, Grand_Total_ML, Cost_per_Sqft_ML
    if ml_pq and os.path.exists(ml_pq):
        ml = pd.read_parquet(ml_pq)
        if "Row_ID" in ml.columns:
            df = df.merge(ml, on="Row_ID", how="left")

    for col in [
        "Grand_Total_Tab", "Cost_per_Sqft_Tab",
        "Grand_Total_Quantity", "Total_Cost_per_Sqft_Q",
        "Grand_Total_ML", "Cost_per_Sqft_ML",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # ---- fuse ----
    gt_tab = _num(df["Grand_Total_Tab"])
    gt_qty = _num(df["Grand_Total_Quantity"])
    gt_ml  = _num(df["Grand_Total_ML"])

    psf_tab = _num(df["Cost_per_Sqft_Tab"])
    psf_qty = _num(df["Total_Cost_per_Sqft_Q"])
    psf_ml  = _num(df["Cost_per_Sqft_ML"])

    df["Grand_Total_Fused"] = calib * (
        np.nan_to_num(w_tab * gt_tab) + np.nan_to_num(w_qty * gt_qty) + np.nan_to_num(w_ml * gt_ml)
    )
    df["Cost_per_Sqft_Fused"] = calib * (
        np.nan_to_num(w_tab * psf_tab) + np.nan_to_num(w_qty * psf_qty) + np.nan_to_num(w_ml * psf_ml)
    )

    # ---- select & save ----
    keep = [
        "Row_ID","City","City_Tier","Room_Type","Area_Sqft",
        "Grand_Total_Tab","Cost_per_Sqft_Tab",
        "Grand_Total_Quantity","Total_Cost_per_Sqft_Q",
        "Grand_Total_ML","Cost_per_Sqft_ML",
        "Grand_Total_Fused","Cost_per_Sqft_Fused",
    ]
    final = df[[c for c in keep if c in df.columns]].copy()
    final.to_parquet(outp, index=False)

    # ---- metrics (merge) ----
    fused_psf   = _num(final.get("Cost_per_Sqft_Fused", pd.Series(dtype=float))).replace([np.inf, -np.inf], np.nan)
    gt_tab_clean = _num(final.get("Grand_Total_Tab", pd.Series(dtype=float)))
    gt_qty_clean = _num(final.get("Grand_Total_Quantity", pd.Series(dtype=float)))
    gt_fused     = _num(final.get("Grand_Total_Fused", pd.Series(dtype=float)))

    m = {
        "mean_fused_cost_per_sqft": float(fused_psf.mean(skipna=True)),
        "weights": {"w_tabular": w_tab, "w_quantity": w_qty, "w_ml": w_ml, "calibration_factor": calib},
    }
    if "Grand_Total_Tab" in final.columns:
        m["mean_abs_diff_vs_tab"] = float((gt_fused - gt_tab_clean).abs().mean(skipna=True))
    if "Grand_Total_Quantity" in final.columns:
        m["mean_abs_diff_vs_qty"] = float((gt_fused - gt_qty_clean).abs().mean(skipna=True))
    if "Grand_Total_ML" in final.columns:
        m["ml_coverage_ratio"] = float(_num(final["Grand_Total_ML"]).notna().mean())

    prev = {}
    if metrics_path and os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    prev["fusion"] = m
    if metrics_path:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(prev, f, indent=2)

    # ---- preview CSV ----
    if preview_csv:
        cols_preview = [
            "Row_ID","City","Area_Sqft",
            "Grand_Total_Tab","Cost_per_Sqft_Tab",
            "Grand_Total_Quantity","Total_Cost_per_Sqft_Q",
            "Grand_Total_ML","Cost_per_Sqft_ML",
            "Grand_Total_Fused","Cost_per_Sqft_Fused",
        ]
        exist = [c for c in cols_preview if c in final.columns]
        final[exist].head(int(preview_n)).to_csv(preview_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True, help="Parquet with tabular baselines")
    ap.add_argument("--cost", required=True, help="Parquet from M3 (cost breakdown)")
    ap.add_argument("--out", required=True, help="Output parquet with fused estimates")
    ap.add_argument("--metrics", required=True, help="Path to JSON metrics (merged)")
    ap.add_argument("--preview-csv", default="", help="Optional CSV preview path")
    ap.add_argument("--preview-n", type=int, default=200, help="Rows for preview CSV")
    ap.add_argument("--ml", default="", help="Optional parquet with ML predictions")
    args = ap.parse_args()

    main(
        tab_pq=args.tab,
        cost_pq=args.cost,
        outp=args.out,
        metrics_path=args.metrics,
        preview_csv=args.preview_csv,
        preview_n=args.preview_n,
        ml_pq=args.ml,
    )
