# src/service/calc.py
# ===================== NEW FILE =====================
from __future__ import annotations
from typing import Dict

def calculate_realistic_quantity_cost(base_cost: float) -> Dict[str, float]:
    """
    Enhance the raw quantity estimate with realistic site overheads.

    Parameters
    ----------
    base_cost : float
        Sum of materials + labor (+ packages) BEFORE overheads/taxes.

    Returns
    -------
    dict with transparent components:
        {
            'base_cost': ...,
            'contractor_margin': ...,
            'wastage_sundries': ...,
            'contingency': ...,
            'pre_gst_total': ...,
            'gst': ...,
            'final_total_qty': ...
        }
    """
    base = float(base_cost or 0.0)

    # Policy (tune as needed or move into params.yaml later)
    contractor_margin_pct = 0.15   # 15%
    wastage_sundries_pct  = 0.05   # 5%
    contingency_pct       = 0.07   # 7%
    gst_pct               = 0.18   # 18%

    contractor_margin = base * contractor_margin_pct
    wastage_sundries = base * wastage_sundries_pct
    contingency      = base * contingency_pct

    pre_gst_total = base + contractor_margin + wastage_sundries + contingency
    gst           = pre_gst_total * gst_pct
    final_total   = pre_gst_total + gst

    return {
        "base_cost": base,
        "contractor_margin": contractor_margin,
        "wastage_sundries": wastage_sundries,
        "contingency": contingency,
        "pre_gst_total": pre_gst_total,
        "gst": gst,
        "final_total_qty": final_total,
    }

# src/service/calc.py (append below previous function)
# ===================== NEW FUNCTION =====================
from typing import Tuple

def _weights_for_level(level: str) -> Tuple[float, float]:
    """
    Returns (w_ml, w_qty) based on Renovation_Level.
    """
    level_norm = (level or "").strip().lower()
    if level_norm in {"economy", "basic"}:
        return (0.30, 0.70)  # favor quantity for low-budget scopes
    if level_norm in {"premium", "luxury"}:
        return (0.60, 0.40)  # favor ML for higher specs/variability
    # Default / Standard / not provided
    return (0.50, 0.50)

def generate_final_recommendation(
    ml_estimate: float,
    realistic_qty_estimate: float,
    user_inputs: dict
) -> Dict[str, float]:
    """
    Blend ML and realistic Qty using context from user_inputs (Renovation_Level).
    Returns:
        {
          'recommended_price': float,
          'price_range': (low, high),
          'weights': {'w_ml': ..., 'w_qty': ...},
          'rationale': '...'
        }
    """
    ml = float(ml_estimate or 0.0)
    qty = float(realistic_qty_estimate or 0.0)
    level = user_inputs.get("Renovation_Level", "") if isinstance(user_inputs, dict) else ""

    w_ml, w_qty = _weights_for_level(level)
    recommended = w_ml * ml + w_qty * qty

    # +/- 10% planning band
    band = 0.10
    low  = recommended * (1.0 - band)
    high = recommended * (1.0 + band)

    rationale = (
        f"Weighted average using Renovation_Level='{level or 'Standard'}' "
        f"(w_ml={w_ml:.2f}, w_qty={w_qty:.2f})."
    )

    return {
        "recommended_price": recommended,
        "price_range": (low, high),
        "weights": {"w_ml": w_ml, "w_qty": w_qty},
        "rationale": rationale,
    }