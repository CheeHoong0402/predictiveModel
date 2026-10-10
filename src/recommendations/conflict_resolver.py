"""Conflict Resolution, Urgency Assignment, and Robustness Evaluation Engine."""
from typing import Any, Dict, List, Optional, Tuple
import math

from .taxonomy import ActionType, UrgencyLevel, RobustnessStatus, ReasonCode



def classify_price_signal(
    price_change_pct: float,
    prob_rise: float,
    prob_fall: float,
    price_p10: Optional[float] = None,
    price_p50: Optional[float] = None,
    price_p90: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool]:
    """Classify purchase-price signal based on both price movement magnitude and probability.
    
    Returns:
        (signal_class, is_price_uncertain)
    """
    if config is None:
        from .config import load_price_decision_config
        config = load_price_decision_config()
        
    thresholds = config.get("thresholds", {})
    min_pct = float(thresholds.get("minimum_price_change_pct", 2.0))
    strong_pct = float(thresholds.get("strong_price_change_pct", 5.0))
    min_prob = float(thresholds.get("minimum_direction_probability", 0.60))
    strong_prob = float(thresholds.get("strong_direction_probability", 0.75))
    penalty_thresh = float(thresholds.get("price_uncertainty_penalty_threshold", 0.35))
    
    # Uncertainty evaluation: relative interval spread (P90 - P10) / P50
    is_uncertain = False
    if price_p10 is not None and price_p50 is not None and price_p90 is not None:
        p50_denom = max(0.01, float(price_p50))
        relative_spread = (float(price_p90) - float(price_p10)) / p50_denom
        if relative_spread > penalty_thresh:
            is_uncertain = True
    
    # Direction probability check
    dominant_prob = prob_rise if price_change_pct >= 0 else prob_fall
    if dominant_prob < min_prob:
        is_uncertain = True

    # Signal determination based on magnitude and probability (Section 6)
    if price_change_pct >= strong_pct and prob_rise >= strong_prob:
        raw_signal = "STRONG_BUY_EARLY_SIGNAL"
    elif price_change_pct >= min_pct and prob_rise >= min_prob:
        raw_signal = "MODERATE_BUY_EARLY_SIGNAL"
    elif price_change_pct <= -strong_pct and prob_fall >= strong_prob:
        raw_signal = "STRONG_WAIT_SIGNAL"
    elif price_change_pct <= -min_pct and prob_fall >= min_prob:
        raw_signal = "MODERATE_WAIT_SIGNAL"
    else:
        raw_signal = "NEUTRAL_PRICE_SIGNAL"

    # High uncertainty weakens price signal (Section 5 Case G & Section 6)
    if is_uncertain:
        if raw_signal == "STRONG_BUY_EARLY_SIGNAL":
            final_signal = "MODERATE_BUY_EARLY_SIGNAL"
        elif raw_signal == "MODERATE_BUY_EARLY_SIGNAL":
            final_signal = "NEUTRAL_PRICE_SIGNAL"
        elif raw_signal == "STRONG_WAIT_SIGNAL":
            final_signal = "MODERATE_WAIT_SIGNAL"
        elif raw_signal == "MODERATE_WAIT_SIGNAL":
            final_signal = "NEUTRAL_PRICE_SIGNAL"
        else:
            final_signal = "NEUTRAL_PRICE_SIGNAL"
    else:
        final_signal = raw_signal

    return final_signal, is_uncertain


def resolve_action_and_urgency(
    current_inventory: float,
    current_coverage_days: float,
    lead_time_days: int,
    lead_time_demand: float,
    phase18_action: str,
    phase18_quantity: float,
    phase19_action: str,
    phase19_quantity: float,
    stockout_prob_before: float,
    stockout_prob_after: float,
    overstock_prob: float,
    wastage_prob: float,
    shelf_life_days: Optional[int],
    price_change_pct: float,
    price_direction: str,
    late_delivery_prob: float = 0.0,
    cost_diff_pct: float = 5.0,
    cvar_cost_ratio: float = 1.4,
    price_signal_strength: str = "NEUTRAL_PRICE_SIGNAL",
    net_price_timing_benefit: float = 0.0,
    demand_trend: str = "STABLE",
    is_price_uncertain: bool = False,
    budget_limit: Optional[float] = None,
    current_purchase_price: float = 0.0,
    price_available: bool = True,
    capacity_limit: Optional[float] = None,
    price_model_validated: Optional[bool] = None
) -> Tuple[str, str, float, List[str], List[str], str, bool]:
    """Resolve conflicting optimization and risk signals using strict priority hierarchy.
    
    Implements formal Demand x Price matrix (Cases A through G) and strict override rules.
    When price_model_validated is explicitly True, price signals participate in decision matrix.
    When price_model_validated is None, queries central price_model_usable_for_decision().
    When price_model_validated is False, price predictions have ZERO influence on decisions.
    
    Returns:
        (final_action, urgency, final_quantity, reason_codes, top_explanations, robustness_status, is_low_margin)
    """
    if price_model_validated is None:
        from src.models.governance import price_model_usable_for_decision
        price_model_validated = price_model_usable_for_decision()
    reason_codes: List[str] = []
    explanations: List[str] = []
    
    # Baseline quantity from Phase 19 risk-adjuster
    final_quantity = max(0.0, float(phase19_quantity if phase19_quantity is not None else phase18_quantity))
    
    # Compute decision margin flag
    is_low_margin = abs(cost_diff_pct) <= 1.0
    
    # Robustness classification
    if cvar_cost_ratio <= 1.5 and cost_diff_pct >= 5.0:
        robustness = RobustnessStatus.ROBUST.value
    elif cvar_cost_ratio <= 2.5 and cost_diff_pct >= 2.0:
        robustness = RobustnessStatus.MODERATELY_ROBUST.value
    elif cvar_cost_ratio <= 4.0 or cost_diff_pct >= 1.0:
        robustness = RobustnessStatus.UNCERTAINTY_SENSITIVE.value
    else:
        robustness = RobustnessStatus.HIGHLY_UNCERTAIN.value

    # Evaluate risk and operational signals
    is_critical_stockout = (
        current_inventory <= 0.0 or
        current_coverage_days < float(lead_time_days) or
        stockout_prob_before >= 0.30
    )
    is_high_stockout = stockout_prob_before >= 0.15
    is_overstocked = overstock_prob >= 0.35 or current_coverage_days >= 25.0
    is_perishable_constrained = (
        (shelf_life_days is not None and shelf_life_days <= 7) or
        (wastage_prob >= 0.10)
    )

    # When price model is NOT validated, all price signals are strictly deactivated
    if price_model_validated and price_available:
        is_price_rising = price_change_pct >= 2.0 or price_direction == "UP"
        is_price_falling = price_change_pct <= -2.0 or price_direction == "DOWN"
        is_wait_signal = (
            price_signal_strength in ["STRONG_WAIT_SIGNAL", "MODERATE_WAIT_SIGNAL", "STRONG_WAIT_PRICE_SIGNAL", "MODERATE_WAIT_PRICE_SIGNAL"] or
            is_price_falling
        )
        is_buy_early_signal = (
            price_signal_strength in ["STRONG_BUY_EARLY_SIGNAL", "MODERATE_BUY_EARLY_SIGNAL", "STRONG_BUY_EARLY_PRICE_SIGNAL", "MODERATE_BUY_EARLY_PRICE_SIGNAL"] or
            (is_price_rising and net_price_timing_benefit >= 5.0)
        )
    else:
        is_price_rising = False
        is_price_falling = False
        is_wait_signal = False
        is_buy_early_signal = False

    is_demand_high_or_rising = str(demand_trend).upper() in ["RISING", "HIGH", "SPIKE", "DEMAND_SPIKE"]
    is_demand_low_or_falling = str(demand_trend).upper() in ["FALLING", "LOW"]
    is_supplier_delayed = late_delivery_prob >= 0.10

    # -------------------------------------------------------------
    # REASON CODE COLLECTION
    # -------------------------------------------------------------
    if not price_model_validated:
        reason_codes.append(ReasonCode.PRICE_MODEL_NOT_VALIDATED.value)
    if not price_available:
        reason_codes.append(ReasonCode.PRICE_FORECAST_UNAVAILABLE.value)
    
    if current_inventory < lead_time_demand:
        reason_codes.append(ReasonCode.LEAD_TIME_SHORTAGE.value)
    if is_critical_stockout or is_high_stockout:
        reason_codes.append(ReasonCode.HIGH_STOCKOUT_RISK.value)
    if is_overstocked:
        reason_codes.append(ReasonCode.HIGH_OVERSTOCK_RISK.value)
    if wastage_prob >= 0.10:
        reason_codes.append(ReasonCode.HIGH_WASTAGE_RISK.value)
    if is_perishable_constrained:
        reason_codes.append(ReasonCode.SHELF_LIFE_CONSTRAINED.value)
    if is_supplier_delayed:
        reason_codes.append(ReasonCode.SUPPLIER_DELAY_RISK.value)

    # Price specific reason codes
    if price_model_validated and price_available:
        if is_price_uncertain:
            reason_codes.append(ReasonCode.PRICE_UNCERTAINTY_HIGH.value)

        if is_price_rising:
            reason_codes.append(ReasonCode.PRICE_EXPECTED_UP.value)
            if price_signal_strength in ["STRONG_BUY_EARLY_SIGNAL", "STRONG_BUY_EARLY_PRICE_SIGNAL"]:
                reason_codes.append(ReasonCode.STRONG_BUY_EARLY_PRICE_SIGNAL.value)
            elif price_signal_strength in ["MODERATE_BUY_EARLY_SIGNAL", "MODERATE_BUY_EARLY_PRICE_SIGNAL"]:
                reason_codes.append(ReasonCode.MODERATE_BUY_EARLY_PRICE_SIGNAL.value)
            
            if net_price_timing_benefit >= 5.0:
                reason_codes.append(ReasonCode.POSITIVE_NET_PRICE_TIMING_BENEFIT.value)
            else:
                reason_codes.append(ReasonCode.NEGATIVE_NET_PRICE_TIMING_BENEFIT.value)

        elif is_price_falling:
            reason_codes.append(ReasonCode.PRICE_EXPECTED_DOWN.value)
            if price_signal_strength in ["STRONG_WAIT_SIGNAL", "STRONG_WAIT_PRICE_SIGNAL"]:
                reason_codes.append(ReasonCode.STRONG_WAIT_PRICE_SIGNAL.value)
            elif price_signal_strength in ["MODERATE_WAIT_SIGNAL", "MODERATE_WAIT_PRICE_SIGNAL"]:
                reason_codes.append(ReasonCode.MODERATE_WAIT_PRICE_SIGNAL.value)
            reason_codes.append(ReasonCode.PRICE_DECLINE_WAIT_SIGNAL.value)
            if net_price_timing_benefit < 0.0:
                reason_codes.append(ReasonCode.NEGATIVE_NET_PRICE_TIMING_BENEFIT.value)
        else:
            if net_price_timing_benefit >= 5.0:
                reason_codes.append(ReasonCode.POSITIVE_NET_PRICE_TIMING_BENEFIT.value)
            elif net_price_timing_benefit < 0.0:
                reason_codes.append(ReasonCode.NEGATIVE_NET_PRICE_TIMING_BENEFIT.value)

    # -------------------------------------------------------------
    # DECISION HIERARCHY & DEMAND X PRICE MATRIX (Cases A through G)
    # -------------------------------------------------------------
    
    # Priority 1 (Case E): CRITICAL STOCKOUT / LEAD-TIME SHORTAGE
    # Overrides price drop wait! Even if supplier price drops, stockout penalty is far worse.
    if is_critical_stockout:
        final_action = ActionType.URGENT_REORDER.value
        urgency = UrgencyLevel.CRITICAL.value if current_inventory <= 0 else UrgencyLevel.HIGH.value
        
        # Ensure positive replenishment quantity
        if final_quantity <= 0:
            final_quantity = max(50.0, lead_time_demand * 1.5)
            
        explanations.append(
            f"Existing stock ({current_inventory:.0f} units, {current_coverage_days:.1f} days) is projected "
            f"to run out before supplier delivery (lead time: {lead_time_days} days)."
        )
        explanations.append(
            f"The recommended order of {final_quantity:.0f} units reduces simulated stockout probability "
            f"from {stockout_prob_before:.1%} to {stockout_prob_after:.1%}."
        )
        if is_wait_signal:
            reason_codes.append(ReasonCode.STOCKOUT_OVERRIDES_PRICE_DECLINE.value)
            reason_codes.append(ReasonCode.PRICE_SIGNAL_OVERRIDDEN_BY_STOCKOUT_RISK.value)
            explanations.append(
                f"Although supplier price is forecast to ease ({price_change_pct:+.1f}%), urgent stockout risk takes absolute priority."
            )
        elif is_price_rising:
            explanations.append(
                f"Supplier purchase price is forecasted to increase by {price_change_pct:+.1f}%, further justifying immediate reorder."
            )
        else:
            explanations.append("Order quantity satisfies operational MOQ and minimum case pack requirements.")

    # Priority 2 (Case F): SHELF-LIFE / OVERSTOCK PROTECTION
    # A BUY_EARLY price signal must not cause unsafe overbuying
    elif is_overstocked or (wastage_prob >= 0.20) or (shelf_life_days is not None and shelf_life_days <= 7 and current_coverage_days >= shelf_life_days):
        if overstock_prob >= 0.50:
            final_action = ActionType.OVERSTOCK_WARNING.value
            urgency = UrgencyLevel.HIGH.value
        else:
            final_action = ActionType.HOLD_CURRENT_STOCK.value
            urgency = UrgencyLevel.LOW.value
        final_quantity = 0.0  # Zero purchase under severe overstock or perishability cap
        
        explanations.append(
            f"Surplus inventory detected ({current_coverage_days:.1f} days coverage, simulated overstock probability {overstock_prob:.1%})."
        )
        if is_buy_early_signal or is_price_rising:
            if wastage_prob >= 0.10 or (shelf_life_days is not None and shelf_life_days <= 7):
                reason_codes.append(ReasonCode.PRICE_SIGNAL_OVERRIDDEN_BY_WASTAGE_RISK.value)
                reason_codes.append(ReasonCode.SHELF_LIFE_OVERRIDES_PRICE_PREBUY.value)
                explanations.append(
                    f"Perishable shelf life ({shelf_life_days}d) strictly caps coverage; price pre-buy was overridden to avoid spoilage write-offs."
                )
            else:
                reason_codes.append(ReasonCode.PRICE_SIGNAL_OVERRIDDEN_BY_OVERSTOCK_RISK.value)
                explanations.append(
                    "High overstock risk strictly limits inventory; early price pre-buy was overridden to preserve working capital."
                )
        else:
            explanations.append("Zero reorder recommended to prevent dead working capital and perishable spoilage write-offs.")
        explanations.append("Allow current inventory to draw down to target safety levels before issuing new purchase orders.")

    # Priority 3: LEAD-TIME SHORTAGE (Coverage tight but not yet at 0)
    elif current_coverage_days <= float(lead_time_days) + 2.0 or is_high_stockout:
        if is_wait_signal and current_coverage_days > float(lead_time_days):
            final_action = ActionType.BUY_SMALLER_QUANTITY.value
            urgency = UrgencyLevel.MEDIUM.value
            reason_codes.append(ReasonCode.PRICE_DECLINE_LEAN_BUY.value)
            if final_quantity <= 0:
                final_quantity = max(40.0, lead_time_demand)
            explanations.append(
                f"Stock coverage ({current_coverage_days:.1f}d) exceeds lead time ({lead_time_days}d); lean order of {final_quantity:.0f} units buffers demand while supplier price drops."
            )
        else:
            final_action = ActionType.BUY_NOW.value
            urgency = UrgencyLevel.HIGH.value
            if final_quantity <= 0:
                final_quantity = max(50.0, lead_time_demand)
                
            explanations.append(
                f"Stock coverage ({current_coverage_days:.1f} days) is approaching supplier lead time ({lead_time_days} days)."
            )
            explanations.append(
                f"Order of {final_quantity:.0f} units maintains safety coverage and limits stockout risk to {stockout_prob_after:.1%}."
            )
            if is_wait_signal:
                reason_codes.append(ReasonCode.STOCKOUT_OVERRIDES_PRICE_DECLINE.value)
                reason_codes.append(ReasonCode.PRICE_SIGNAL_OVERRIDDEN_BY_STOCKOUT_RISK.value)
                explanations.append(f"Imminent lead time constraint overrides forward price easing ({price_change_pct:+.1f}%).")
            elif is_price_rising:
                explanations.append(f"Upcoming price increase of {price_change_pct:+.1f}% favors prompt procurement.")
            else:
                explanations.append("Balanced reorder minimizes total holding and stockout penalties.")

    # Priority 4: NORMAL OPERATIONS DEMAND X PRICE MATRIX (Cases A, B, C, D)
    else:
        # Case D: Demand LOW/FALLING + Price FALLING -> WAIT / HOLD
        if is_demand_low_or_falling and is_wait_signal and current_coverage_days >= float(lead_time_days) + 3.0:
            final_action = ActionType.WAIT_FOR_LOWER_PRICE.value if current_coverage_days >= float(lead_time_days) + 5.0 else ActionType.HOLD_CURRENT_STOCK.value
            urgency = UrgencyLevel.LOW.value
            final_quantity = 0.0
            explanations.append(
                f"Current inventory safely covers {current_coverage_days:.1f} days of demand beyond supplier lead time."
            )
            explanations.append(
                f"Falling demand and forecasted price reduction of {abs(price_change_pct):.1f}% justify delaying procurement."
            )
            explanations.append("Monitor supplier quotations and re-evaluate upon next review cycle.")

        # Case B: Demand RISING/HIGH + Price FALLING -> BUY ENOUGH TO AVOID STOCKOUT (LEAN BUY)
        elif is_demand_high_or_rising and is_wait_signal:
            if current_coverage_days >= float(lead_time_days) + 5.0:
                final_action = ActionType.WAIT_FOR_LOWER_PRICE.value
                urgency = UrgencyLevel.LOW.value
                final_quantity = 0.0
                explanations.append("High demand is sufficiently buffered by current on-hand stock; holding purchase to capture lower future price.")
            else:
                final_action = ActionType.BUY_SMALLER_QUANTITY.value
                urgency = UrgencyLevel.MEDIUM.value
                reason_codes.append(ReasonCode.PRICE_DECLINE_LEAN_BUY.value)
                explanations.append("Lean replenishment ordered to maintain immediate lead time buffer while supplier prices soften.")
                explanations.append(f"Reduced order of {final_quantity:.0f} units controls working capital and avoids forward overpayment.")
                explanations.append(f"Simulated post-decision service level reaches {1.0 - stockout_prob_after:.1%}.")

        # Case C: Demand LOW/FALLING + Price RISING -> DO NOT AUTOMATICALLY BUY EXTRA
        elif is_demand_low_or_falling and is_price_rising:
            if net_price_timing_benefit >= 5.0 and not is_price_uncertain and overstock_prob < 0.20:
                final_action = ActionType.BUY_NOW.value
                urgency = UrgencyLevel.MEDIUM.value
                explanations.append("Expected wholesale price arbitrage exceeds holding cost despite sluggish demand.")
                explanations.append(f"Controlled order of {final_quantity:.0f} units placed to lock in pre-increase price.")
            else:
                # Holding costs exceed price benefit -> DO NOT buy extra!
                reason_codes.append(ReasonCode.PRICE_SIGNAL_DID_NOT_CHANGE_DECISION.value)
                if current_coverage_days >= float(lead_time_days) + 4.0:
                    final_action = ActionType.HOLD_CURRENT_STOCK.value
                    urgency = UrgencyLevel.LOW.value
                    final_quantity = 0.0
                    explanations.append("Although supplier price is forecast to rise, low demand makes pre-buying uneconomic.")
                    explanations.append("Holding current stock to avoid dead inventory and carrying cost escalation.")
                else:
                    final_action = ActionType.BUY_NOW.value
                    urgency = UrgencyLevel.LOW.value
                    final_quantity = max(0.0, float(phase18_quantity))
                    explanations.append("Routine replenishment without speculative price forward-buying.")
                    explanations.append("Carrying costs outweigh price timing arbitrage under low demand.")

        # Case A: Demand RISING/HIGH + Price RISING -> BUY_NOW (Potentially expanded coverage)
        elif is_demand_high_or_rising and is_price_rising:
            final_action = ActionType.BUY_NOW.value
            urgency = UrgencyLevel.HIGH.value if (net_price_timing_benefit >= 5.0 and not is_price_uncertain) else UrgencyLevel.MEDIUM.value
            explanations.append("High demand combined with confirmed supplier price increase justifies proactive procurement.")
            explanations.append(f"Order quantity of {final_quantity:.0f} units locks in wholesale price before anticipated increase.")
            explanations.append(f"Simulated post-decision service level reaches {1.0 - stockout_prob_after:.1%}.")

        # Standard Cost Optimisation / Fallback
        elif final_quantity > 0:
            if is_wait_signal and current_coverage_days >= float(lead_time_days) + 4.0:
                final_action = ActionType.WAIT_FOR_LOWER_PRICE.value
                urgency = UrgencyLevel.LOW.value
                final_quantity = 0.0
                explanations.append(f"Supplier price forecast to soften ({price_change_pct:+.1f}%); deferring procurement.")
            elif is_wait_signal and current_coverage_days >= float(lead_time_days) + 2.0:
                final_action = ActionType.BUY_SMALLER_QUANTITY.value
                urgency = UrgencyLevel.MEDIUM.value
                reason_codes.append(ReasonCode.PRICE_DECLINE_LEAN_BUY.value)
                explanations.append("Lean replenishment ordered to maintain immediate lead time buffer while supplier prices soften.")
            else:
                final_action = ActionType.BUY_NOW.value
                urgency = UrgencyLevel.HIGH.value if (is_price_rising and net_price_timing_benefit >= 5.0 and not is_price_uncertain) else UrgencyLevel.MEDIUM.value
                explanations.append("Routine replenishment recommended to maintain optimal inventory equilibrium.")
                explanations.append(f"Order quantity of {final_quantity:.0f} units satisfies forecasted demand while minimizing holding cost.")
                explanations.append(f"Simulated post-decision service level reaches {1.0 - stockout_prob_after:.1%}.")
                if (is_price_rising or is_price_falling) and net_price_timing_benefit < 5.0 and not is_critical_stockout:
                    reason_codes.append(ReasonCode.PRICE_SIGNAL_DID_NOT_CHANGE_DECISION.value)

        # Monitoring / Zero Reorder
        else:
            if is_low_margin:
                final_action = ActionType.MONITOR_CLOSELY.value
                urgency = UrgencyLevel.LOW.value
                explanations.append("Decision difference between purchasing and holding is marginal (within 1.0% cost band).")
                explanations.append("Close monitoring recommended; maintain current stock position.")
            else:
                final_action = ActionType.DO_NOT_REORDER.value
                urgency = UrgencyLevel.NONE.value
                explanations.append(f"Inventory coverage ({current_coverage_days:.1f} days) is in equilibrium with forward demand.")
                explanations.append("No replenishment order is required at this decision origin.")

    # -------------------------------------------------------------
    # BUDGET & CAPACITY CONSTRAINTS OVERRIDE
    # -------------------------------------------------------------
    if budget_limit is not None and current_purchase_price > 0.0 and final_quantity > 0.0:
        cost_of_order = final_quantity * current_purchase_price
        if cost_of_order > budget_limit:
            max_affordable_qty = max(0.0, float(math.floor(budget_limit / current_purchase_price)))
            if max_affordable_qty < final_quantity:
                final_quantity = max_affordable_qty
                reason_codes.append(ReasonCode.PRICE_SIGNAL_OVERRIDDEN_BY_BUDGET.value)
                explanations.append(
                    f"Procurement budget constraint (RM {budget_limit:.2f}) capped order quantity to {final_quantity:.0f} units."
                )

    if capacity_limit is not None and final_quantity > 0.0:
        if current_inventory + final_quantity > capacity_limit:
            final_quantity = max(0.0, capacity_limit - current_inventory)
            reason_codes.append(ReasonCode.CAPACITY_OVERRIDES_PRICE_PREBUY.value)
            explanations.append(
                f"Warehouse capacity limit ({capacity_limit:.0f} units) capped order quantity to {final_quantity:.0f} units."
            )

    # Final cleanup of reason codes
    if not reason_codes:
        reason_codes.append(ReasonCode.NO_ACTION_REQUIRED.value)

    # Deduplicate reason codes preserving order
    deduped_reasons: List[str] = []
    for r in reason_codes:
        if r not in deduped_reasons:
            deduped_reasons.append(r)

    return (
        final_action,
        urgency,
        final_quantity,
        deduped_reasons,
        explanations[:3],  # top 3 reasons
        robustness,
        is_low_margin
    )

