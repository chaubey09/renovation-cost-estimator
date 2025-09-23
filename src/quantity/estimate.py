# src/quantity/estimate.py
import argparse, pandas as pd, numpy as np, yaml, json, os

def load_params():
    with open("params.yaml","r") as f:
        return yaml.safe_load(f)

def paint_multiplier(room, rules):
    m = rules["paint_wall_multiplier_by_room"]
    return m.get(room, m["default"])

def false_ceiling_ratio(ctype, rules):
    r = rules["false_ceiling_ratio_by_type"]
    return r.get(ctype, r["default"])

def base_elec_points(room, rules):
    base = rules["electrical_points_base"].get(room, rules["electrical_points_base"]["default"])
    return base

def main(tab_pq, detections_jsonl, outp):
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    P = load_params()
    df = pd.read_parquet(tab_pq)

    qty = df[[
        "Row_ID","Room_Type","Area_Sqft","Ceiling_Type","Ceiling_Quality",
        "Paint_Quality","Floor_Type","Floor_Quality","Has_Electrical"
    ]].copy()

    qty["paintable_area_sqft"] = qty.apply(
        lambda r: (r["Area_Sqft"] or 0) * paint_multiplier(str(r["Room_Type"]), P["quant_rules"]), axis=1
    )

    qty["flooring_area_sqft"] = qty["Area_Sqft"].fillna(0)

    qty["false_ceiling_area_sqft"] = qty.apply(
        lambda r: (r["Area_Sqft"] or 0) * false_ceiling_ratio(str(r["Ceiling_Type"]), P["quant_rules"]),
        axis=1
    )

    base_pts = qty.apply(lambda r: base_elec_points(str(r["Room_Type"]), P["quant_rules"]), axis=1)
    scale_pts = qty["Area_Sqft"].fillna(0) * float(P["quant_rules"]["electrical_area_scale"])
    qty["electrical_points"] = np.where(df["Has_Electrical"].fillna(False), (base_pts + scale_pts).round().astype(int), 0)

    if detections_jsonl and os.path.exists(detections_jsonl):
        det = []
        with open(detections_jsonl,"r") as f:
            for line in f:
                try: det.append(json.loads(line))
                except: pass
        if det:
            det_df = pd.DataFrame(det).drop_duplicates(subset=["Row_ID"], keep="last")
            qty = qty.merge(det_df, on="Row_ID", how="left", suffixes=("","_det"))
            for col in ["paintable_area_sqft","flooring_area_sqft","false_ceiling_area_sqft","electrical_points"]:
                over = col+"_det"
                if over in qty.columns:
                    qty[col] = qty[over].fillna(qty[col])
                    qty.drop(columns=[over], inplace=True, errors="ignore")

    qty.to_parquet(outp, index=False)
    if args.preview_csv:
        cols = ["Row_ID","Room_Type","Area_Sqft","paintable_area_sqft",
                "flooring_area_sqft","false_ceiling_area_sqft","electrical_points"]
        existing = [c for c in cols if c in qty.columns]
        qty[existing].head(args.preview_n).to_csv(args.preview_csv, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True)
    ap.add_argument("--detections", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.tab, args.detections, args.out)
    ap.add_argument("--preview-csv", default="")
    ap.add_argument("--preview-n", type=int, default=200)
