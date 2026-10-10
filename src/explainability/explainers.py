"""Interactive explanation API for Demand and Purchase-Price forecasts.

Generates local and global explanations with feature display name translation,
business family aggregation, positive/negative driver ranking in business units,
calibrated uncertainty interval integration, Telegram messaging, and dashboard JSON formatting.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from src.calibration.inference import (
    predict_demand_with_uncertainty,
    predict_price_with_uncertainty
)
from src.explainability.tree_shap import compute_tree_shap
from src.explainability.display_names import FeatureDisplayDictionary


# Global cached display dictionary
_DISPLAY_DICT: Optional[FeatureDisplayDictionary] = None


def get_display_dictionary(root: Path = PROJECT_ROOT) -> FeatureDisplayDictionary:
    """Retrieve or initialize singleton FeatureDisplayDictionary."""
    global _DISPLAY_DICT
    if _DISPLAY_DICT is None:
        _DISPLAY_DICT = FeatureDisplayDictionary.load(root=root)
    return _DISPLAY_DICT


def explain_demand_prediction(
    feature_frame: pd.DataFrame,
    horizon: Union[int, str],
    top_n: int = 5,
    root: Path = PROJECT_ROOT
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate comprehensive local explanation for demand predictions.
    
    Compatible with user-uploaded data frames. Does not require target columns.
    
    Args:
        feature_frame: DataFrame containing feature columns for one or more predictions
        horizon: Horizon integer (1, 3, 7, 14, 21, 30) or string ('1d', etc.)
        top_n: Number of top positive and negative drivers to extract
        root: Project root directory
        
    Returns:
        Explanation dictionary (if single row) or list of dictionaries (if multiple rows).
    """
    h = int(str(horizon).replace("d", ""))
    root = Path(root)
    display_dict = get_display_dictionary(root=root)

    # 1. Calibrated predictions and uncertainty
    pred_df = predict_demand_with_uncertainty(feature_frame=feature_frame, horizon=h, root=root)

    # 2. TreeSHAP contributions and base value
    shap_df, base_vals, feature_names = compute_tree_shap(
        task="demand",
        horizon=h,
        feature_frame=feature_frame,
        root=root
    )

    explanations = []
    N = len(feature_frame)

    for i in range(N):
        row_feat = feature_frame.iloc[i]
        row_pred = pred_df.iloc[i]
        phi_row = shap_df.iloc[i]
        base_val = float(base_vals[i])
        point_pred = float(row_pred["point_prediction"])
        p10 = float(row_pred["p10"])
        p50 = float(row_pred["p50"])
        p90 = float(row_pred["p90"])
        width = float(row_pred["interval_width"])

        # Extract entity metadata
        date_val = str(row_pred.get("date", "UNKNOWN"))
        store_id = str(row_pred.get("store_id", "UNKNOWN"))
        product_id = str(row_pred.get("product_id", "UNKNOWN"))

        # Feature contributions
        pos_drivers = []
        neg_drivers = []
        family_contributions: Dict[str, float] = {}

        for feat in feature_names:
            val = float(phi_row[feat])
            fam = display_dict.get_business_family(feat)
            family_contributions[fam] = family_contributions.get(fam, 0.0) + val

            d_name = display_dict.get_display_name(feat)
            raw_val = float(row_feat[feat]) if feat in row_feat else 0.0

            if val > 0:
                pos_drivers.append({
                    "feature": feat,
                    "display_name": d_name,
                    "business_family": fam,
                    "contribution": round(val, 3),
                    "impact": "INCREASING_DEMAND",
                    "unit": "units",
                    "feature_value": round(raw_val, 3)
                })
            elif val < 0:
                neg_drivers.append({
                    "feature": feat,
                    "display_name": d_name,
                    "business_family": fam,
                    "contribution": round(val, 3),
                    "impact": "DECREASING_DEMAND",
                    "unit": "units",
                    "feature_value": round(raw_val, 3)
                })

        # Sort drivers by absolute magnitude
        pos_drivers.sort(key=lambda x: x["contribution"], reverse=True)
        neg_drivers.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        top_pos = pos_drivers[:top_n]
        top_neg = neg_drivers[:top_n]

        # Sorted family contributions
        sorted_families = [
            {"business_family": k, "net_contribution": round(v, 3)}
            for k, v in sorted(family_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        ]

        # Narrative explanation
        direction_str = "higher than" if point_pred >= base_val else "lower than"
        diff_str = f"{abs(point_pred - base_val):.1f} units"
        
        top_pos_names = [f"{d['display_name']} (+{d['contribution']} units)" for d in top_pos[:2]]
        top_neg_names = [f"{d['display_name']} ({d['contribution']} units)" for d in top_neg[:2]]
        
        narrative_parts = [
            f"The forecasted demand for product {product_id} at store {store_id} ({h}-day horizon) is {point_pred:.1f} units, "
            f"which is {diff_str} {direction_str} the historical baseline ({base_val:.1f} units)."
        ]
        if top_pos_names:
            narrative_parts.append(f"Key upward drivers include: {', '.join(top_pos_names)}.")
        if top_neg_names:
            narrative_parts.append(f"Key downward dampers include: {', '.join(top_neg_names)}.")
        narrative_parts.append(
            f"With 80% confidence, actual demand is expected between {p10:.1f} and {p90:.1f} units (median: {p50:.1f})."
        )
        narrative_text = " ".join(narrative_parts)

        # Telegram markdown format
        telegram_lines = [
            f"*Demand Forecast Alert ({h}d Horizon)*",
            f"Store: `{store_id}` | Product: `{product_id}` | Date: `{date_val}`",
            f"*Point Forecast:* `{point_pred:.1f}` units",
            f"*80% Prediction Interval:* `{p10:.1f}` to `{p90:.1f}` units (P50: `{p50:.1f}`)",
            f"*Baseline Reference:* `{base_val:.1f}` units",
            "",
            "*Top Upward Drivers:*"
        ]
        for d in top_pos[:3]:
            telegram_lines.append(f"  + {d['display_name']}: `+{d['contribution']} units`")
        if not top_pos:
            telegram_lines.append("  (None)")

        telegram_lines.append("")
        telegram_lines.append("*Top Downward Dampers:*")
        for d in top_neg[:3]:
            telegram_lines.append(f"  - {d['display_name']}: `{d['contribution']} units`")
        if not top_neg:
            telegram_lines.append("  (None)")

        telegram_msg = "\n".join(telegram_lines)

        # Dashboard payload
        dashboard_payload = {
            "entity": {
                "task": "demand",
                "horizon": f"{h}d",
                "date": date_val,
                "store_id": store_id,
                "product_id": product_id
            },
            "forecast": {
                "point_forecast": round(point_pred, 2),
                "base_value": round(base_val, 2),
                "p10": round(p10, 2),
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "interval_width": round(width, 2),
                "unit": "units"
            },
            "drivers": {
                "top_positive": top_pos,
                "top_negative": top_neg,
                "family_contributions": sorted_families
            },
            "narrative": narrative_text
        }

        explanation = {
            "task": "demand",
            "horizon": h,
            "date": date_val,
            "store_id": store_id,
            "product_id": product_id,
            "point_prediction": round(point_pred, 3),
            "base_value": round(base_val, 3),
            "p10": round(p10, 3),
            "p50": round(p50, 3),
            "p90": round(p90, 3),
            "interval_width": round(width, 3),
            "unit": "units",
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "family_contributions": sorted_families,
            "narrative_explanation": narrative_text,
            "telegram_message": telegram_msg,
            "dashboard_payload": dashboard_payload
        }
        explanations.append(explanation)

    return explanations[0] if N == 1 else explanations


def explain_price_prediction(
    feature_frame: pd.DataFrame,
    horizon: Union[int, str],
    top_n: int = 5,
    root: Path = PROJECT_ROOT
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Generate comprehensive local explanation for supplier purchase-price predictions.
    
    Compatible with user-uploaded data frames. Does not require target columns.
    Converts model percentage change contributions into explicit MYR (RM) business contributions.
    
    Args:
        feature_frame: DataFrame containing feature columns for one or more predictions
        horizon: Horizon integer (1, 3, 7, 14, 21, 30) or string ('1d', etc.)
        top_n: Number of top positive and negative drivers to extract
        root: Project root directory
        
    Returns:
        Explanation dictionary (if single row) or list of dictionaries (if multiple rows).
    """
    h = int(str(horizon).replace("d", ""))
    root = Path(root)
    display_dict = get_display_dictionary(root=root)

    # 1. Calibrated predictions and uncertainty
    pred_df = predict_price_with_uncertainty(feature_frame=feature_frame, horizon=h, root=root)

    # 2. TreeSHAP percentage change contributions and base percentage value
    shap_df, base_vals, feature_names = compute_tree_shap(
        task="price",
        horizon=h,
        feature_frame=feature_frame,
        root=root
    )

    curr_col = "purchase_price" if "purchase_price" in feature_frame.columns else "current_purchase_price"

    explanations = []
    N = len(feature_frame)

    for i in range(N):
        row_feat = feature_frame.iloc[i]
        row_pred = pred_df.iloc[i]
        phi_row = shap_df.iloc[i]
        base_pct = float(base_vals[i])
        
        # Reference current purchase price
        curr_price = float(row_feat[curr_col]) if curr_col in row_feat else 10.0
        
        point_price = float(row_pred["point_prediction"])
        pred_pct = float(row_pred.get("predicted_percentage_change", (point_price / curr_price) - 1.0))
        p10 = float(row_pred["p10"])
        p50 = float(row_pred["p50"])
        p90 = float(row_pred["p90"])
        width = float(row_pred.get("interval_width_myr", p90 - p10))

        # Base expected price in MYR
        base_price_myr = curr_price * (1.0 + base_pct)

        # Extract entity metadata
        date_val = str(row_pred.get("date", "UNKNOWN"))
        store_id = str(row_pred.get("store_id", "UNKNOWN"))
        product_id = str(row_pred.get("product_id", "UNKNOWN"))
        supplier_id = str(row_pred.get("supplier_id", "UNKNOWN"))

        # Feature contributions in % and in MYR
        pos_drivers = []
        neg_drivers = []
        family_contributions_pct: Dict[str, float] = {}
        family_contributions_myr: Dict[str, float] = {}

        for feat in feature_names:
            val_pct = float(phi_row[feat])
            val_myr = curr_price * val_pct
            fam = display_dict.get_business_family(feat)

            family_contributions_pct[fam] = family_contributions_pct.get(fam, 0.0) + val_pct
            family_contributions_myr[fam] = family_contributions_myr.get(fam, 0.0) + val_myr

            d_name = display_dict.get_display_name(feat)
            raw_val = float(row_feat[feat]) if feat in row_feat else 0.0

            if val_pct > 0:
                pos_drivers.append({
                    "feature": feat,
                    "display_name": d_name,
                    "business_family": fam,
                    "contribution_myr": round(val_myr, 4),
                    "contribution_pct": round(val_pct * 100, 2),
                    "impact": "INCREASING_PRICE",
                    "unit": "MYR",
                    "feature_value": round(raw_val, 3)
                })
            elif val_pct < 0:
                neg_drivers.append({
                    "feature": feat,
                    "display_name": d_name,
                    "business_family": fam,
                    "contribution_myr": round(val_myr, 4),
                    "contribution_pct": round(val_pct * 100, 2),
                    "impact": "DECREASING_PRICE",
                    "unit": "MYR",
                    "feature_value": round(raw_val, 3)
                })

        # Sort drivers by absolute magnitude
        pos_drivers.sort(key=lambda x: x["contribution_myr"], reverse=True)
        neg_drivers.sort(key=lambda x: abs(x["contribution_myr"]), reverse=True)

        top_pos = pos_drivers[:top_n]
        top_neg = neg_drivers[:top_n]

        # Sorted family contributions
        sorted_families = [
            {
                "business_family": k,
                "net_contribution_myr": round(family_contributions_myr[k], 4),
                "net_contribution_pct": round(family_contributions_pct[k] * 100, 2)
            }
            for k, _ in sorted(family_contributions_myr.items(), key=lambda x: abs(x[1]), reverse=True)
        ]

        # Narrative explanation
        direction_str = "increase by" if point_price >= curr_price else "decrease by"
        diff_myr = abs(point_price - curr_price)
        diff_pct = abs(pred_pct) * 100

        top_pos_names = [f"{d['display_name']} (+RM {d['contribution_myr']:.2f} / +{d['contribution_pct']}%)" for d in top_pos[:2]]
        top_neg_names = [f"{d['display_name']} (-RM {abs(d['contribution_myr']):.2f} / {d['contribution_pct']}%)" for d in top_neg[:2]]

        narrative_parts = [
            f"The forecasted supplier purchase price for {product_id} from {supplier_id} ({h}-day horizon) is "
            f"RM {point_price:.2f}, expected to {direction_str} RM {diff_myr:.2f} ({diff_pct:.1f}%) relative to the current price of RM {curr_price:.2f}."
        ]
        if top_pos_names:
            narrative_parts.append(f"Key cost drivers increasing the price include: {', '.join(top_pos_names)}.")
        if top_neg_names:
            narrative_parts.append(f"Key downward price pressures include: {', '.join(top_neg_names)}.")
        narrative_parts.append(
            f"With 80% confidence, the future purchase price will range between RM {p10:.2f} and RM {p90:.2f} (median: RM {p50:.2f})."
        )
        narrative_text = " ".join(narrative_parts)

        # Telegram message
        telegram_lines = [
            f"*Purchase Price Forecast Alert ({h}d Horizon)*",
            f"Supplier: `{supplier_id}` | Product: `{product_id}` | Store: `{store_id}`",
            f"*Current Price:* `RM {curr_price:.2f}`",
            f"*Point Forecast:* `RM {point_price:.2f}` (`{pred_pct * 100:+.1f}%`)",
            f"*80% Prediction Interval:* `RM {p10:.2f}` to `RM {p90:.2f}` (P50: `RM {p50:.2f}`)",
            "",
            "*Top Cost Inflation Drivers:*"
        ]
        for d in top_pos[:3]:
            telegram_lines.append(f"  + {d['display_name']}: `+RM {d['contribution_myr']:.2f}` (`+{d['contribution_pct']}%`)")
        if not top_pos:
            telegram_lines.append("  (None)")

        telegram_lines.append("")
        telegram_lines.append("*Top Downward Deflation Drivers:*")
        for d in top_neg[:3]:
            telegram_lines.append(f"  - {d['display_name']}: `RM {d['contribution_myr']:.2f}` (`{d['contribution_pct']}%`)")
        if not top_neg:
            telegram_lines.append("  (None)")

        telegram_msg = "\n".join(telegram_lines)

        # Dashboard payload
        dashboard_payload = {
            "entity": {
                "task": "purchase_price",
                "horizon": f"{h}d",
                "date": date_val,
                "store_id": store_id,
                "product_id": product_id,
                "supplier_id": supplier_id
            },
            "forecast": {
                "current_price": round(curr_price, 2),
                "point_forecast": round(point_price, 2),
                "predicted_pct_change": round(pred_pct * 100, 2),
                "base_price_myr": round(base_price_myr, 2),
                "p10": round(p10, 2),
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "interval_width": round(width, 2),
                "unit": "MYR"
            },
            "drivers": {
                "top_positive": top_pos,
                "top_negative": top_neg,
                "family_contributions": sorted_families
            },
            "narrative": narrative_text
        }

        explanation = {
            "task": "purchase_price",
            "horizon": h,
            "date": date_val,
            "store_id": store_id,
            "product_id": product_id,
            "supplier_id": supplier_id,
            "current_price": round(curr_price, 3),
            "point_prediction": round(point_price, 3),
            "predicted_percentage_change": round(pred_pct, 4),
            "base_value_myr": round(base_price_myr, 3),
            "p10": round(p10, 3),
            "p50": round(p50, 3),
            "p90": round(p90, 3),
            "interval_width": round(width, 3),
            "unit": "MYR",
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "family_contributions": sorted_families,
            "narrative_explanation": narrative_text,
            "telegram_message": telegram_msg,
            "dashboard_payload": dashboard_payload
        }
        explanations.append(explanation)

    return explanations[0] if N == 1 else explanations
