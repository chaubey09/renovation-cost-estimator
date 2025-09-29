import pandas as pd
import yaml
import numpy as np

def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def create_feature_dataframe(raw_input_dict: dict) -> pd.DataFrame:
    P = load_params()
    df = pd.DataFrame([raw_input_dict])

    # STAGE 2: QUANTITY ESTIMATION LOGIC
    paint_rules = P["quant_rules"]["paint_wall_multiplier_by_room"]
    ceiling_rules = P["quant_rules"]["false_ceiling_ratio_by_type"]
    elec_base_rules = P["quant_rules"]["electrical_points_base"]
    elec_scale = float(P["quant_rules"]["electrical_area_scale"])
    df['paintable_area_sqft'] = df['Area_Sqft'] * df['Room_Type'].map(paint_rules).fillna(paint_rules['default'])
    df['flooring_area_sqft'] = df['Area_Sqft']
    df['false_ceiling_area_sqft'] = df['Area_Sqft'] * df['Ceiling_Type'].map(ceiling_rules).fillna(ceiling_rules['default'])
    base_pts = df['Room_Type'].map(elec_base_rules).fillna(elec_base_rules['default'])
    scaled_pts = df['Area_Sqft'] * elec_scale
    df['electrical_points'] = np.where(df['Has_Electrical'], (base_pts + scaled_pts).round(), 0).astype(int)

    # STAGE 3: COST MAPPING LOGIC
    for col, default in [("Material_Price_Index", 1.0), ("City_Multiplier", 1.0)]:
        if col not in df.columns: df[col] = default
    paint_mat_tbl = P["rates"]["painting"]["material_per_sqft_by_quality"]
    floor_tbl = P["rates"]["flooring"]["material_per_sqft_by_type_quality"]
    ceil_tbl = P["rates"]["ceiling"]["material_per_sqft_by_type_quality"]
    df['paint_mat_rate'] = df['Paint_Quality'].map(paint_mat_tbl).fillna(paint_mat_tbl["Standard"])
    df['paint_lab_rate'] = P["rates"]["painting"]["labor_per_sqft"]
    df['Painting_Material_Est'] = df['paintable_area_sqft'] * df['paint_mat_rate'] * df['Material_Price_Index'] * df['City_Multiplier']
    df['Painting_Labor_Est'] = df['paintable_area_sqft'] * df['paint_lab_rate'] * df['City_Multiplier']
    def floor_rate(row):
        default_tile = next(iter(floor_tbl))
        base = floor_tbl.get(row['Floor_Type'], floor_tbl[default_tile])
        return base.get(row['Floor_Quality'], base["Standard"])
    df['floor_mat_rate'] = df.apply(floor_rate, axis=1)
    df['floor_lab_rate'] = P["rates"]["flooring"]["labor_per_sqft"]
    df['Flooring_Material_Est'] = df['flooring_area_sqft'] * df['floor_mat_rate'] * df['Material_Price_Index'] * df['City_Multiplier']
    df['Flooring_Labor_Est'] = df['flooring_area_sqft'] * df['floor_lab_rate'] * df['City_Multiplier']
    def ceil_rate(row):
        if row['Ceiling_Type'] == "None": return 0.0
        default_type = next(iter(ceil_tbl))
        base = ceil_tbl.get(row['Ceiling_Type'], ceil_tbl[default_type])
        return base.get(row['Ceiling_Quality'], base["Standard"])
    df['ceil_mat_rate'] = df.apply(ceil_rate, axis=1)
    df['ceil_lab_rate'] = P["rates"]["ceiling"]["labor_per_sqft"]
    df['Ceiling_Material_Est'] = df['false_ceiling_area_sqft'] * df['ceil_mat_rate'] * df['Material_Price_Index'] * df['City_Multiplier']
    df['Ceiling_Labor_Est'] = df['false_ceiling_area_sqft'] * df['ceil_lab_rate'] * df['City_Multiplier']
    df['Electrical_Material_Est'] = df['electrical_points'] * P["rates"]["electrical"]["per_point_material"] * df['Material_Price_Index'] * df['City_Multiplier']
    df['Electrical_Labor_Est'] = df['electrical_points'] * P["rates"]["electrical"]["per_point_labor"] * df['City_Multiplier']
    pkg_kitchen_defaults = P["rates"]["packages"]["kitchen"]
    pkg_bathroom_defaults = P["rates"]["packages"]["bathroom"]
    df['Kitchen_Package_Cost_Est'] = df['Kitchen_Package'].map(pkg_kitchen_defaults).fillna(0)
    df['Bathroom_Package_Cost_Est'] = df['Bathroom_Package'].map(pkg_bathroom_defaults).fillna(0)
    cost_cols = ["Painting_Material_Est", "Painting_Labor_Est", "Flooring_Material_Est", "Flooring_Labor_Est",
                 "Ceiling_Material_Est", "Ceiling_Labor_Est", "Electrical_Material_Est", "Electrical_Labor_Est",
                 "Kitchen_Package_Cost_Est", "Bathroom_Package_Cost_Est"]
    df['Subtotal'] = df[cost_cols].sum(axis=1)
    df['Wastage_Sundries_Est'] = df['Subtotal'] * float(P["cost"]["wastage_pct"])
    df['Overhead_Est'] = (df['Subtotal'] + df['Wastage_Sundries_Est']) * float(P["cost"]["contractor_overhead_pct"])
    df['PreGST_Total_Est'] = df['Subtotal'] + df['Wastage_Sundries_Est'] + df['Overhead_Est']
    df['GST_Est'] = df['PreGST_Total_Est'] * float(P["cost"]["gst_rate"])

    all_model_features = [
        'Material_Price_Index', 'City', 'City_Tier', 'City_Multiplier', 'Labor_Day_Rate_Min', 
        'Labor_Day_Rate_Max', 'Room_Type', 'Area_Sqft', 'Renovation_Level', 'Paint_Quality', 
        'Floor_Type', 'Floor_Quality', 'Ceiling_Type', 'Ceiling_Quality', 'Has_Electrical', 
        'Furniture_Level', 'Kitchen_Package', 'Bathroom_Package', 'Painting_Material_Cost', 
        'Painting_Labor_Cost', 'Flooring_Material_Cost', 'Flooring_Labor_Cost', 
        'Ceiling_Material_Cost', 'Ceiling_Labor_Cost', 'Electrical_Material_Cost', 
        'Electrical_Labor_Cost', 'Kitchen_Package_Cost', 'Bathroom_Package_Cost', 
        'Plumbing_Cost', 'Furniture_Cost', 'Wastage_Sundries_Cost', 'Contractor_Overhead_Cost', 
        'GST_Amount', 'Total_Cost_per_Sqft', 'paintable_area_sqft', 'flooring_area_sqft', 
        'false_ceiling_area_sqft', 'electrical_points', 'paint_mat_rate', 'paint_lab_rate', 
        'Painting_Material_Est', 'Painting_Labor_Est', 'floor_mat_rate', 'floor_lab_rate', 
        'Flooring_Material_Est', 'Flooring_Labor_Est', 'ceil_mat_rate', 'ceil_lab_rate', 
        'Ceiling_Material_Est', 'Ceiling_Labor_Est', 'Electrical_Material_Est', 
        'Electrical_Labor_Est', 'Kitchen_Package_Cost_Est', 'Bathroom_Package_Cost_Est', 
        'Subtotal', 'Wastage_Sundries_Est', 'Overhead_Est', 'PreGST_Total_Est', 'GST_Est'
    ]
    
    for col in all_model_features:
        if col not in df.columns:
            df[col] = 0
            
    return df[all_model_features]