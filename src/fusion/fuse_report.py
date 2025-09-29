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

def main(tab_pq, cost_pq, outp, metrics_path, preview_csv="", preview_n=200):
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    if preview_csv:
        os.makedirs(os.path.dirname(preview_csv), exist_ok=True)

    # ---- config ----
    P = load_params()
    F = P.get("fusion", {})
    w_tab = float(F.get("w_tabular", 0.5))
    w_qty = float(F.get("w_quantity", 0.5))
    calib = float(F.get("calibration_factor", 1.0))

    # ---- load inputs ----
    cost = pd.read_parquet(cost_pq)  # output from M3
    df = cost.copy()

    # bring in tab baselines if present
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

    for col in ["Grand_Total_Tab", "Cost_per_Sqft_Tab", "Grand_Total_Quantity", "Total_Cost_per_Sqft_Q"]:
        if col not in df.columns:
            df[col] = np.nan

    # ---- fuse ----
    gt_tab = pd.to_numeric(df["Grand_Total_Tab"], errors="coerce")
    gt_qty = pd.to_numeric(df["Grand_Total_Quantity"], errors="coerce")
    psf_tab = pd.to_numeric(df["Cost_per_Sqft_Tab"], errors="coerce")
    psf_qty = pd.to_numeric(df["Total_Cost_per_Sqft_Q"], errors="coerce")

    df["Grand_Total_Fused"] = calib * (np.nan_to_num(w_tab * gt_tab) + np.nan_to_num(w_qty * gt_qty))
    df["Cost_per_Sqft_Fused"] = calib * (np.nan_to_num(w_tab * psf_tab) + np.nan_to_num(w_qty * psf_qty))

    # ---- select & save ----
    keep = [
        "Row_ID", "City", "City_Tier", "Room_Type", "Area_Sqft",
        "Grand_Total_Tab", "Cost_per_Sqft_Tab",
        "Grand_Total_Quantity", "Total_Cost_per_Sqft_Q",
        "Grand_Total_Fused", "Cost_per_Sqft_Fused"
    ]
    final = df[[c for c in keep if c in df.columns]].copy()
    final.to_parquet(outp, index=False)

    # ---- metrics (merge into existing) ----
    fused_psf = pd.to_numeric(final.get("Cost_per_Sqft_Fused", pd.Series(dtype=float)), errors="coerce") \
        .replace([np.inf, -np.inf], np.nan)
    gt_tab_clean = pd.to_numeric(final.get("Grand_Total_Tab", pd.Series(dtype=float)), errors="coerce")
    gt_fused_clean = pd.to_numeric(final.get("Grand_Total_Fused", pd.Series(dtype=float)), errors="coerce")

    m = {}
    if "Grand_Total_Tab" in final.columns:
        diff = (gt_fused_clean - gt_tab_clean).abs()
        m["mean_abs_diff_vs_tab"] = float(diff.mean(skipna=True))
    m["mean_fused_cost_per_sqft"] = float(fused_psf.mean(skipna=True))

    prev = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
        except Exception:
            prev = {}
    prev["fusion"] = m
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2)

    # ---- preview CSV for faculty ----
    if preview_csv:
        cols_preview = [
            "Row_ID", "City", "Area_Sqft",
            "Grand_Total_Tab", "Cost_per_Sqft_Tab",
            "Grand_Total_Quantity", "Total_Cost_per_Sqft_Q",
            "Grand_Total_Fused", "Cost_per_Sqft_Fused"
        ]
        exist = [c for c in cols_preview if c in final.columns]
        final[exist].head(int(preview_n)).to_csv(preview_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--cost", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--preview-csv", default="")
    ap.add_argument("--preview-n", type=int, default=200)
    args = ap.parse_args()
    main(args.tab, args.cost, args.out, args.metrics, args.preview_csv, args.preview_n)
