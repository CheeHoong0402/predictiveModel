"""Action Priority Queue and Executive Priority Scoring."""
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from .taxonomy import UrgencyLevel, RobustnessStatus


URGENCY_RANKS = {
    UrgencyLevel.CRITICAL.value: 5,
    UrgencyLevel.HIGH.value: 4,
    UrgencyLevel.MEDIUM.value: 3,
    UrgencyLevel.LOW.value: 2,
    UrgencyLevel.NONE.value: 1,
}

ROBUSTNESS_WEIGHTS = {
    RobustnessStatus.ROBUST.value: 100.0,
    RobustnessStatus.MODERATELY_ROBUST.value: 80.0,
    RobustnessStatus.UNCERTAINTY_SENSITIVE.value: 50.0,
    RobustnessStatus.HIGHLY_UNCERTAIN.value: 30.0,
}


def calculate_executive_priority_score(
    urgency: str,
    stockout_prob_before: float,
    lost_units: float,
    price_change_pct: float,
    robustness: str,
    net_price_timing_benefit: float = 0.0,
    overstock_prob: float = 0.0,
    demand_trend: str = "STABLE",
    price_model_validated: bool = False
) -> float:
    """Compute transparent 0-100 Executive Priority Score.
    
    Formula components:
    - Urgency contribution (35%): urgency_rank (1-5) * 20
    - Stockout risk contribution (30%): stockout_prob_before * 100
    - Financial volume impact (15%): min(100, lost_units / 50 * 100)
    - Price timing opportunity (10%): awarded ONLY when price model is validated, net benefit > 0,
      demand supports procurement, and overstock risk is acceptable (< 0.35).
      When price model is not validated, price opportunity is strictly 0.0.
    - Recommendation robustness (10%): robustness weight (30-100)
    """
    u_score = URGENCY_RANKS.get(str(urgency).upper(), 1) * 20.0
    s_score = min(100.0, max(0.0, float(stockout_prob_before) * 100.0))
    vol_score = min(100.0, max(0.0, (float(lost_units) / 50.0) * 100.0))
    
    # Section 14: Price opportunity requires validated price model, positive financial benefit, demand support, and low overstock risk
    if price_model_validated and net_price_timing_benefit > 0.0 and overstock_prob < 0.35 and str(demand_trend).upper() not in ["FALLING", "LOW"]:
        p_score = min(100.0, max(0.0, float(price_change_pct) * 10.0))
    else:
        p_score = 0.0

    r_score = ROBUSTNESS_WEIGHTS.get(str(robustness).upper(), 50.0)

    score = (
        0.35 * u_score +
        0.30 * s_score +
        0.15 * vol_score +
        0.10 * p_score +
        0.10 * r_score
    )
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def build_action_priority_queue(recommendations_df: pd.DataFrame, root: Any = None) -> pd.DataFrame:
    """Sort and rank recommendations into an actionable priority queue."""
    from src.models.governance import price_model_usable_for_decision
    price_validated = price_model_usable_for_decision(root=root)
    df = recommendations_df.copy()
    
    if "urgency_rank" not in df.columns:
        df["urgency_rank"] = df["urgency"].map(lambda u: URGENCY_RANKS.get(str(u).upper(), 1))
        
    if "priority_score" not in df.columns:
        df["priority_score"] = [
            calculate_executive_priority_score(
                urgency=row.get("urgency", "NONE"),
                stockout_prob_before=row.get("stockout_probability_before", 0.0),
                lost_units=row.get("expected_lost_units", 0.0) if "expected_lost_units" in row else 0.0,
                price_change_pct=row.get("predicted_price_change_pct", 0.0),
                robustness=row.get("robustness_status", "ROBUST"),
                net_price_timing_benefit=row.get("net_price_timing_benefit", 0.0),
                overstock_prob=row.get("overstock_probability", 0.0),
                demand_trend=row.get("demand_trend", "STABLE"),
                price_model_validated=price_validated
            )
            for _, row in df.iterrows()
        ]
        
    # Sort strictly by: 1. Priority Score, 2. Urgency Rank, 3. Stockout Risk Before
    # Price change is only included in sort if price model is validated
    sort_cols = ["priority_score", "urgency_rank", "stockout_probability_before"]
    ascending_flags = [False, False, False]
    if price_validated and "predicted_price_change_pct" in df.columns:
        sort_cols.append("predicted_price_change_pct")
        ascending_flags.append(False)

    df = df.sort_values(
        by=sort_cols,
        ascending=ascending_flags
    ).reset_index(drop=True)
    
    df["queue_rank"] = range(1, len(df) + 1)
    return df

