# src/service/features.py
from __future__ import annotations
from typing import Dict
import pandas as pd
from .ml_utils import to_number, dedupe_preserve_order

ROOM_TYPES = ["Bedroom", "Living_Room", "Kitchen", "Bathroom", "Other"]
DEFAULT_AREA = {"Bedroom":120.0,"Living_Room":180.0,"Kitchen":80.0,"Bathroom":45.0,"Other":100.0}

def spec_to_quality(spec: str) -> str:
    s = (spec or "").lower()
    if s in {"economy","basic"}: return "Economy"
    if s in {"premium","luxury"}: return "Premium"
    return "Standard"

def lookup_city_index(city: str, ci_df: pd.DataFrame, P: dict) -> dict:
    base_mpi = float(P["rates"].get("base_mpi", 1.0))
    out = {"Material_Price_Index": base_mpi, "City_Multiplier": 1.0, "City_Tier": ""}
    if ci_df is None or ci_df.empty or not city: return out
    hit = ci_df.loc[ci_df["City"].astype(str).str.lower() == str(city).lower()]
    if len(hit):
        for k in ("Material_Price_Index","City_Multiplier"):
            if k in hit.columns and pd.notna(hit[k].iloc[0]): out[k] = float(hit[k].iloc[0])
        if "City_Tier" in hit.columns and pd.notna(hit["City_Tier"].iloc[0]): out["City_Tier"] = str(hit["City_Tier"].iloc[0])
    return out

def compute_engineered_features(mini: dict, P: dict, ci_df: pd.DataFrame) -> pd.DataFrame:
    city  = str(mini.get("City",""))
    area  = to_number(mini.get("Area_Sqft",0.0), 0.0)
    level = str(mini.get("Renovation_Level","Economy"))
    room  = str(mini.get("Room_Type","Bedroom"))
    has_e = bool(mini.get("Has_Electrical", True))
    ceil_t= str(mini.get("Ceiling_Type","None"))

    qual = spec_to_quality(level)
    paint_q = str(mini.get("Paint_Quality", qual))
    floor_q = str(mini.get("Floor_Quality", qual))
    ceil_q  = str(mini.get("Ceiling_Quality", qual))
    floor_t = str(mini.get("Floor_Type","Ceramic_Tile"))
    k_pkg   = str(mini.get("Kitchen_Package","None"))
    b_pkg   = str(mini.get("Bathroom_Package","None"))

    city_idx  = lookup_city_index(city, ci_df, P)
    city_mult = float(city_idx["City_Multiplier"])
    mpi       = float(city_idx["Material_Price_Index"])
    city_tier = str(city_idx.get("City_Tier",""))

    paint_rules = dict(P["quant_rules"]["paint_wall_multiplier_by_room"])
    ceil_rules  = dict(P["quant_rules"]["false_ceiling_ratio_by_type"])
    elec_base   = dict(P["quant_rules"]["electrical_points_base"])
    elec_scale  = float(P["quant_rules"]["electrical_area_scale"])

    paint_tbl   = dict(P["rates"]["painting"]["material_per_sqft_by_quality"])
    floor_tbl   = dict(P["rates"]["flooring"]["material_per_sqft_by_type_quality"])
    ceil_tbl    = dict(P["rates"]["ceiling"]["material_per_sqft_by_type_quality"])
    paint_lab   = float(P["rates"]["painting"]["labor_per_sqft"])
    floor_lab   = float(P["rates"]["flooring"]["labor_per_sqft"])
    ceil_lab    = float(P["rates"]["ceiling"]["labor_per_sqft"])
    elec_mat_pp = float(P["rates"]["electrical"]["per_point_material"])
    elec_lab_pp = float(P["rates"]["electrical"]["per_point_labor"])
    pkg_k       = dict(P["rates"]["packages"]["kitchen"])
    pkg_b       = dict(P["rates"]["packages"]["bathroom"])

    wall_mult  = float(paint_rules.get(room, paint_rules.get("default", 3.2)))
    ceil_ratio = float(ceil_rules.get(ceil_t, ceil_rules.get("default", 0.0)))
    elec_base_pts = float(elec_base.get(room, elec_base.get("default", 2.0)))

    paintable = area * wall_mult
    flooring  = area
    ceil_area = area * ceil_ratio
    elec_pts  = int(round((elec_base_pts + area * elec_scale))) if has_e else 0

    paint_mat_rate = float(paint_tbl.get(paint_q, paint_tbl.get("Standard", 0.0)))
    base_floor = floor_tbl.get(floor_t, next(iter(floor_tbl.values())))
    floor_mat_rate = float(base_floor.get(floor_q, base_floor.get("Standard", 0.0)))
    if ceil_t == "None":
        ceil_mat_rate = 0.0
    else:
        base_ceil = ceil_tbl.get(ceil_t, next(iter(ceil_tbl.values())))
        ceil_mat_rate = float(base_ceil.get(ceil_q, base_ceil.get("Standard", 0.0)))

    PMat = paintable * paint_mat_rate * mpi * city_mult
    PLab = paintable * paint_lab       * city_mult
    FMat = flooring  * floor_mat_rate  * mpi * city_mult
    FLab = flooring  * floor_lab       * city_mult
    CMat = ceil_area * ceil_mat_rate   * mpi * city_mult
    CLab = ceil_area * ceil_lab        * city_mult
    EMat = elec_pts  * elec_mat_pp     * mpi * city_mult
    ELab = elec_pts  * elec_lab_pp     * city_mult
    KPkg = float(pkg_k.get(k_pkg, 0.0))
    BPkg = float(pkg_b.get(b_pkg, 0.0))

    Subtotal = PMat+PLab+FMat+FLab+CMat+CLab+EMat+ELab+KKpg if (KKpg:=KPkg) is not None else 0.0
    Subtotal += BPkg

    all_model_features = [
        'Material_Price_Index','City','City_Tier','City_Multiplier','Labor_Day_Rate_Min','Labor_Day_Rate_Max',
        'Room_Type','Area_Sqft','Renovation_Level','Paint_Quality','Floor_Type','Floor_Quality',
        'Ceiling_Type','Ceiling_Quality','Has_Electrical','Furniture_Level','Kitchen_Package','Bathroom_Package',
        'Painting_Material_Cost','Painting_Labor_Cost','Flooring_Material_Cost','Flooring_Labor_Cost',
        'Ceiling_Material_Cost','Ceiling_Labor_Cost','Electrical_Material_Cost','Electrical_Labor_Cost',
        'Kitchen_Package_Cost','Bathroom_Package_Cost','Plumbing_Cost','Furniture_Cost','Wastage_Sundries_Cost',
        'Contractor_Overhead_Cost','GST_Amount','Total_Cost_per_Sqft',
        'paintable_area_sqft','flooring_area_sqft','false_ceiling_area_sqft','electrical_points',
        'paint_mat_rate','paint_lab_rate','Painting_Material_Est','Painting_Labor_Est',
        'floor_mat_rate','floor_lab_rate','Flooring_Material_Est','Flooring_Labor_Est',
        'ceil_mat_rate','ceil_lab_rate','Ceiling_Material_Est','Ceiling_Labor_Est',
        'Electrical_Material_Est','Electrical_Labor_Est','Kitchen_Package_Cost_Est','Bathroom_Package_Cost_Est',
        'Wastage_Sundries_Est','Overhead_Est','PreGST_Total_Est','GST_Est'
    ]
    row = {
        'Material_Price_Index': mpi, 'City': city, 'City_Tier': city_tier, 'City_Multiplier': city_mult,
        'Labor_Day_Rate_Min': 0.0, 'Labor_Day_Rate_Max': 0.0,
        'Room_Type': room, 'Area_Sqft': area, 'Renovation_Level': level,
        'Paint_Quality': paint_q, 'Floor_Type': floor_t, 'Floor_Quality': floor_q,
        'Ceiling_Type': ceil_t, 'Ceiling_Quality': ceil_q, 'Has_Electrical': has_e,
        'Furniture_Level': 'Basic', 'Kitchen_Package': k_pkg, 'Bathroom_Package': b_pkg,
        'Painting_Material_Cost': 0, 'Painting_Labor_Cost': 0, 'Flooring_Material_Cost': 0, 'Flooring_Labor_Cost': 0,
        'Ceiling_Material_Cost': 0, 'Ceiling_Labor_Cost': 0, 'Electrical_Material_Cost': 0, 'Electrical_Labor_Cost': 0,
        'Kitchen_Package_Cost': 0, 'Bathroom_Package_Cost': 0, 'Plumbing_Cost': 0, 'Furniture_Cost': 0,
        'Wastage_Sundries_Cost': 0, 'Contractor_Overhead_Cost': 0, 'GST_Amount': 0, 'Total_Cost_per_Sqft': 0,
        'paintable_area_sqft': paintable, 'flooring_area_sqft': flooring, 'false_ceiling_area_sqft': ceil_area,
        'electrical_points': elec_pts, 'paint_mat_rate': paint_mat_rate, 'paint_lab_rate': paint_lab,
        'Painting_Material_Est': PMat, 'Painting_Labor_Est': PLab, 'floor_mat_rate': floor_mat_rate, 'floor_lab_rate': floor_lab,
        'Flooring_Material_Est': FMat, 'Flooring_Labor_Est': FLab, 'ceil_mat_rate': ceil_mat_rate, 'ceil_lab_rate': ceil_lab,
        'Ceiling_Material_Est': CMat, 'Ceiling_Labor_Est': CLab, 'Electrical_Material_Est': EMat, 'Electrical_Labor_Est': ELab,
        'Kitchen_Package_Cost_Est': KPkg, 'Bathroom_Package_Cost_Est': BPkg,
        'Wastage_Sundries_Est': 0, 'Overhead_Est': 0, 'PreGST_Total_Est': 0, 'GST_Est': 0,
    }
    for c in all_model_features:
        row.setdefault(c, 0)
    df = pd.DataFrame([row], columns=dedupe_preserve_order(all_model_features + ["Subtotal"]))
    df.loc[0, "Subtotal"] = Subtotal
    return df
