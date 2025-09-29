# src/quantity/estimate.py
import argparse
import json
import os
import pandas as pd
import numpy as np
import yaml

# ---------------- helpers ----------------

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def paint_multiplier(room, rules):
    m = rules["paint_wall_multiplier_by_room"]
    return m.get(room, m["default"])

def false_ceiling_ratio(ctype, rules):
    r = rules["false_ceiling_ratio_by_type"]
    return r.get(ctype, r["default"])

def base_elec_points(room, rules):
    return rules["electrical_points_base"].get(room, rules["electrical_points_base"]["default"])

# ---------------- core ----------------

def main(tab_pq, detections_jsonl, outp, preview_csv="", preview_n=200):
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    if preview_csv:
        os.makedirs(os.path.dirname(preview_csv), exist_ok=True)

    P = load_params()
    df = pd.read_parquet(tab_pq)

    # columns we use (robust to missing ones)
    cols = [
        "Row_ID","Room_Type","Area_Sqft","Ceiling_Type","Ceiling_Quality",
        "Paint_Quality","Floor_Type","Floor_Quality","Has_Electrical"
    ]
    keep = [c for c in cols if c in df.columns]
    qty = df[keep].copy()

    # Fill safe defaults for missing inputs
    qty["Room_Type"] = qty.get("Room_Type", pd.Series("", index=qty.index)).astype(str)
    qty["Area_Sqft"] = pd.to_numeric(qty.get("Area_Sqft", 0), errors="coerce").fillna(0)
    qty["Ceiling_Type"] = qty.get("Ceiling_Type", pd.Series("None", index=qty.index)).astype(str)

    has_elec = qty["Has_Electrical"] if "Has_Electrical" in qty.columns else pd.Series(False, index=qty.index)
    has_elec = has_elec.fillna(False).astype(bool)

    # Paintable wall area (heuristic)
    qty["paintable_area_sqft"] = [
        (a or 0) * paint_multiplier(str(rt), P["quant_rules"])
        for rt, a in zip(qty["Room_Type"], qty["Area_Sqft"])
    ]

    # Flooring area = plan area
    qty["flooring_area_sqft"] = qty["Area_Sqft"]

    # False ceiling area (ratio by type)
    qty["false_ceiling_area_sqft"] = [
        (a or 0) * false_ceiling_ratio(str(ct), P["quant_rules"])
        for ct, a in zip(qty["Ceiling_Type"], qty["Area_Sqft"])
    ]

    # Electrical points = base by room + area scaling (only if Has_Electrical)
    base_pts = np.array([base_elec_points(str(rt), P["quant_rules"]) for rt in qty["Room_Type"]], dtype=float)
    scale_pts = qty["Area_Sqft"].values.astype(float) * float(P["quant_rules"]["electrical_area_scale"])
    points = (base_pts + scale_pts).round().astype(int)
    qty["electrical_points"] = np.where(has_elec.values, points, 0)

    # Optional: apply overrides from detections.jsonl
    if detections_jsonl and os.path.exists(detections_jsonl):
        det_rows = []
        with open(detections_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    det_rows.append(json.loads(line))
                except Exception:
                    pass
        if det_rows:
            det_df = pd.DataFrame(det_rows).drop_duplicates(subset=["Row_ID"], keep="last")
            qty = qty.merge(det_df, on="Row_ID", how="left", suffixes=("", "_det"))
            for col in ["paintable_area_sqft", "flooring_area_sqft", "false_ceiling_area_sqft", "electrical_points"]:
                over = col + "_det"
                if over in qty.columns:
                    qty[col] = pd.to_numeric(qty[over], errors="coerce").fillna(qty[col])
                    qty.drop(columns=[over], inplace=True, errors="ignore")

    # Save full quantities as parquet
    qty.to_parquet(outp, index=False)

    # Optional small preview CSV to show module-wise result
    if preview_csv:
        preview_cols = [
            "Row_ID","Room_Type","Area_Sqft",
            "paintable_area_sqft","flooring_area_sqft","false_ceiling_area_sqft","electrical_points"
        ]
        exist = [c for c in preview_cols if c in qty.columns]
        qty[exist].head(int(preview_n)).to_csv(preview_csv, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", required=True, help="Path to M1 parquet (processed.parquet)")
    ap.add_argument("--detections", default="", help="Optional JSONL overrides for quantities")
    ap.add_argument("--out", required=True, help="Output parquet (quantities)")
    ap.add_argument("--preview-csv", default="", help="Optional CSV preview path")
    ap.add_argument("--preview-n", type=int, default=200, help="Rows for preview CSV")
    args = ap.parse_args()

    main(args.tab, args.detections, args.out, args.preview_csv, args.preview_n)
