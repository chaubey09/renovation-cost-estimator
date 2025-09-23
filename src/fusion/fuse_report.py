# src/fusion/fuse_report.py
import argparse, pandas as pd, numpy as np, yaml, json, os

def load_params():
    with open("params.yaml","r") as f:
        return yaml.safe_load(f)

def main(tab_pq, cost_pq, outp, metrics_path):
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    P = load_params()
    tab = pd.read_parquet(tab_pq)
    cost = pd.read_parquet(cost_pq)

    df = cost.copy()

    w_tab = float(P["fusion"]["w_tabular"])
    w_qty = float(P["fusion"]["w_quantity"])
    calib = float(P["fusion"]["calibration_factor"])

    base_col = "Grand_Total"
    base_psf = "Total_Cost_per_Sqft"

    if base_col in tab.columns:
        base_total = tab[["Row_ID", base_col]].rename(columns={base_col:"Grand_Total_Tab"})
        df = df.merge(base_total, on="Row_ID", how="left")
    else:
        df["Grand_Total_Tab"] = np.nan

    if base_psf in tab.columns:
        base_psf_df = tab[["Row_ID", base_psf]].rename(columns={base_psf:"Cost_per_Sqft_Tab"})
        df = df.merge(base_psf_df, on="Row_ID", how="left")
    else:
        df["Cost_per_Sqft_Tab"] = np.nan

    q_total = df["Grand_Total_Quantity"]
    q_psf   = df["Total_Cost_per_Sqft_Q"]

    df["Grand_Total_Fused"] = calib * (
        np.nan_to_num(w_tab * df["Grand_Total_Tab"]) + np.nan_to_num(w_qty * q_total)
    )
    df["Cost_per_Sqft_Fused"] = calib * (
        np.nan_to_num(w_tab * df["Cost_per_Sqft_Tab"]) + np.nan_to_num(w_qty * q_psf)
    )

    keep = [
      "Row_ID","City","City_Tier","Room_Type","Area_Sqft",
      "Grand_Total_Tab","Cost_per_Sqft_Tab",
      "Grand_Total_Quantity","Total_Cost_per_Sqft_Q",
      "Grand_Total_Fused","Cost_per_Sqft_Fused"
    ]
    existing = [c for c in keep if c in df.columns]
    final = df[existing].copy()

    final.to_parquet(outp, index=False)



    m = {}
    if "Grand_Total_Tab" in final.columns:
        diff = final["Grand_Total_Fused"] - final["Grand_Total_Tab"]
        m["mean_abs_diff_vs_tab"] = float(np.abs(diff).mean())
    m["mean_fused_cost_per_sqft"] = float(final["Cost_per_Sqft_Fused"].mean())

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            prev = json.load(f)
    else:
        prev = {}
    prev.update({"fusion": m})

    with open(metrics_path, "w") as f:
        json.dump(prev, f, indent=2)
    
    if args.preview_csv:
        cols = ["Row_ID","City","Area_Sqft",
                "Grand_Total_Tab","Cost_per_Sqft_Tab",
                "Grand_Total_Quantity","Total_Cost_per_Sqft_Q",
                "Grand_Total_Fused","Cost_per_Sqft_Fused"]
        existing = [c for c in cols if c in final.columns]
        final[existing].head(args.preview_n).to_csv(args.preview_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--cost", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args()
    main(args.tab, args.cost, args.out, args.metrics)
    ap.add_argument("--preview-csv", default="")
    ap.add_argument("--preview-n", type=int, default=200)
