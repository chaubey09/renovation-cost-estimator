import argparse, pandas as pd, numpy as np, os, json
BOOL_MAP = {"TRUE": True,"FALSE": False,"True": True,"False": False,True: True,False: False}

def main(inp, outp, preview_csv="", schema_json="", preview_n=200):
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    df = pd.read_csv(inp)

    if "As_Of_Date" in df.columns: df.drop(columns=["As_Of_Date"], inplace=True)

    for col in df.columns:
        if df[col].dtype == object and df[col].astype(str).str.upper().isin(["TRUE","FALSE"]).any():
            df[col] = df[col].map(BOOL_MAP).fillna(df[col])

    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols: df[c] = df[c].astype(str).str.strip()

    numeric_guess = [
        "Material_Price_Index","City_Multiplier","Labor_Day_Rate_Min","Labor_Day_Rate_Max","Area_Sqft",
        "Painting_Material_Cost","Painting_Labor_Cost","Flooring_Material_Cost","Flooring_Labor_Cost",
        "Ceiling_Material_Cost","Ceiling_Labor_Cost","Electrical_Material_Cost","Electrical_Labor_Cost",
        "Kitchen_Package_Cost","Bathroom_Package_Cost","Plumbing_Cost","Furniture_Cost",
        "Wastage_Sundries_Cost","Contractor_Overhead_Cost","GST_Amount","Grand_Total","Total_Cost_per_Sqft"
    ]
    for c in [c for c in numeric_guess if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "Material_Price_Index" not in df.columns: df["Material_Price_Index"] = 1.0
    df["Area_Sqft"] = pd.to_numeric(df.get("Area_Sqft", np.nan), errors="coerce")
    df.to_parquet(outp, index=False)

    if preview_csv:
        os.makedirs(os.path.dirname(preview_csv), exist_ok=True)
        df.head(int(preview_n)).to_csv(preview_csv, index=False)
    if schema_json:
        os.makedirs(os.path.dirname(schema_json), exist_ok=True)
        schema = {c: str(dt) for c, dt in df.dtypes.items()}
        with open(schema_json, "w", encoding="utf-8") as f:
            json.dump({"rows": int(len(df)), "schema": schema}, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview-csv", default="")
    ap.add_argument("--schema-json", default="")
    ap.add_argument("--preview-n", type=int, default=200)
    args = ap.parse_args()
    main(args.input, args.out, args.preview_csv, args.schema_json, args.preview_n)
