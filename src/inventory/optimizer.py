"""Core Inventory Optimisation and Purchase Decision Engine.

Evaluates candidate stock coverage days from 1 to 30, incorporates calibrated demand
and price uncertainty, enforces MOQ and case-size rounding, evaluates lead time,
shelf life, storage capacity, and budget constraints, and selects the minimum expected cost
procurement action.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from src.models.governance import (
    price_model_usable_for_decision,
    get_price_model_validation_status,
    load_model_governance_config
)
from .config import load_inventory_config, load_cost_assumptions
from .curve_fitting import interpolate_demand_curve, interpolate_price_curve
from .cost_engine import (
    calculate_candidate_cost,
    check_constraints,
    CostBreakdown,
    RiskEstimates,
    ConstraintCheckResult
)


@dataclass
class CandidateEvaluation:
    """Evaluation metrics for a single candidate coverage day."""
    coverage_days: int
    raw_order_quantity: float
    order_quantity: float
    expected_demand: float
    safety_stock: float
    predicted_future_price: float
    cost_breakdown: CostBreakdown
    risk_estimates: RiskEstimates
    is_feasible: bool
    rejection_reasons: List[str] = field(default_factory=list)
    net_timing_benefit: float = 0.0
    gross_timing_savings: float = 0.0


@dataclass
class InventoryDecision:
    """Final selected procurement decision for a product-store-supplier combination."""
    decision_date: str
    store_id: str
    product_id: str
    supplier_id: str
    recommended_action: str
    severity: str
    recommended_order_quantity: float
    recommended_stock_coverage_days: int
    current_inventory: float
    incoming_inventory: float
    current_purchase_price: float
    predicted_future_price: float
    expected_price_change_pct: float
    expected_demand_for_coverage: float
    demand_p10: float
    demand_p50: float
    demand_p90: float
    price_p10: float
    price_p50: float
    price_p90: float
    cost_breakdown: CostBreakdown
    risk_estimates: RiskEstimates
    expected_savings_vs_reference: float
    reference_strategy_name: str
    reference_order_quantity: float
    reference_total_cost: float
    robustness_status: str
    top_reasons: List[str]
    alternative_strategies: Dict[str, Dict[str, Any]]
    all_candidate_evaluations: List[CandidateEvaluation] = field(default_factory=list)
    baseline_demand_only_quantity: float = 0.0
    baseline_demand_only_coverage_days: int = 0
    price_adjusted_quantity_delta: float = 0.0
    price_adjusted_coverage_delta: int = 0
    net_price_timing_benefit: Optional[float] = None
    price_signal_strength: str = "NEUTRAL_PRICE_SIGNAL"
    buy_now_vs_wait_recommendation: str = "BUY_NOW"
    buy_now_cost: float = 0.0
    wait_cost: float = 0.0
    price_model_status: str = "NOT_VALIDATED"
    price_decision_enabled: bool = False
    price_influenced_order_quantity: bool = False
    price_influenced_coverage: bool = False
    price_order_adjustment_units: float = 0.0
    price_coverage_adjustment_days: int = 0
    decision_mode: str = "DEMAND_INVENTORY_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to structured dictionary."""
        return {
            "decision_date": self.decision_date,
            "store_id": self.store_id,
            "product_id": self.product_id,
            "supplier_id": self.supplier_id,
            "recommended_action": self.recommended_action,
            "severity": self.severity,
            "recommended_order_quantity": self.recommended_order_quantity,
            "recommended_stock_coverage_days": self.recommended_stock_coverage_days,
            "current_inventory": self.current_inventory,
            "incoming_inventory": self.incoming_inventory,
            "current_purchase_price": round(self.current_purchase_price, 2),
            "predicted_future_price": round(self.predicted_future_price, 2),
            "expected_price_change_pct": round(self.expected_price_change_pct, 2),
            "expected_demand_for_coverage": round(self.expected_demand_for_coverage, 1),
            "demand_p10": round(self.demand_p10, 1),
            "demand_p50": round(self.demand_p50, 1),
            "demand_p90": round(self.demand_p90, 1),
            "expected_total_cost": round(self.cost_breakdown.total_expected_cost, 2),
            "expected_purchase_cost": round(self.cost_breakdown.purchase_cost, 2),
            "expected_holding_cost": round(self.cost_breakdown.holding_cost, 2),
            "expected_stockout_cost": round(self.cost_breakdown.stockout_cost, 2),
            "expected_wastage_cost": round(self.cost_breakdown.wastage_cost, 2),
            "future_price_savings": round(self.cost_breakdown.future_price_savings, 2),
            "net_timing_benefit": round(self.cost_breakdown.net_timing_benefit, 2) if self.cost_breakdown.net_timing_benefit is not None else None,
            "buy_now_vs_wait_advantage": round(self.cost_breakdown.buy_now_vs_wait_advantage, 2) if self.cost_breakdown.buy_now_vs_wait_advantage is not None else None,
            "baseline_demand_only_quantity": round(self.baseline_demand_only_quantity, 1),
            "baseline_demand_only_coverage_days": self.baseline_demand_only_coverage_days,
            "price_adjusted_quantity_delta": round(self.price_adjusted_quantity_delta, 1),
            "price_adjusted_coverage_delta": self.price_adjusted_coverage_delta,
            "net_price_timing_benefit": round(self.net_price_timing_benefit, 2) if self.net_price_timing_benefit is not None else None,
            "price_signal_strength": self.price_signal_strength,
            "buy_now_vs_wait_recommendation": self.buy_now_vs_wait_recommendation,
            "buy_now_cost": round(self.buy_now_cost, 2),
            "wait_cost": round(self.wait_cost, 2),
            "price_model_status": self.price_model_status,
            "price_decision_enabled": self.price_decision_enabled,
            "price_influenced_order_quantity": self.price_influenced_order_quantity,
            "price_influenced_coverage": self.price_influenced_coverage,
            "price_order_adjustment_units": round(self.price_order_adjustment_units, 1),
            "price_coverage_adjustment_days": self.price_coverage_adjustment_days,
            "decision_mode": self.decision_mode,
            "preliminary_stockout_risk": self.risk_estimates.preliminary_stockout_risk,
            "preliminary_overstock_risk": self.risk_estimates.preliminary_overstock_risk,
            "expected_savings_vs_reference": round(self.expected_savings_vs_reference, 2),
            "robustness_status": self.robustness_status,
            "top_reasons": self.top_reasons,
            "alternative_strategies": self.alternative_strategies
        }


@dataclass
class DecisionBatchResult:
    """Aggregated outputs from batch inventory optimisation."""
    recommendations_df: pd.DataFrame
    candidate_evaluations_df: pd.DataFrame
    strategy_comparison_df: pd.DataFrame
    robustness_df: pd.DataFrame


def _round_to_constraints(raw_qty: float, moq: float, case_size: int) -> float:
    """Round order quantity up to supplier MOQ and pack/case-size multiples."""
    if raw_qty <= 0:
        return 0.0

    # Minimum order quantity constraint
    qty = max(raw_qty, float(moq))

    # Case size multiple rounding
    if case_size > 1:
        qty = math.ceil(qty / float(case_size)) * float(case_size)
    else:
        qty = math.ceil(qty)

    return float(qty)


def optimise_inventory_decision(
    demand_forecasts: Dict[int, float],
    price_forecasts: Dict[int, float],
    demand_p10: Optional[Dict[int, float]] = None,
    demand_p50: Optional[Dict[int, float]] = None,
    demand_p90: Optional[Dict[int, float]] = None,
    price_p10: Optional[Dict[int, float]] = None,
    price_p50: Optional[Dict[int, float]] = None,
    price_p90: Optional[Dict[int, float]] = None,
    current_inventory: float = 0.0,
    incoming_inventory: float = 0.0,
    current_purchase_price: float = 10.0,
    normal_selling_price: Optional[float] = None,
    lead_time_days: float = 2.0,
    supplier_minimum_order_quantity: float = 10.0,
    product_case_size: int = 1,
    shelf_life_days: Optional[float] = None,
    store_storage_capacity: Optional[float] = None,
    budget_limit: Optional[float] = None,
    cost_parameters: Optional[Dict[str, float]] = None,
    service_level: str = "balanced",
    store_id: str = "STORE_UNKNOWN",
    product_id: str = "PRODUCT_UNKNOWN",
    supplier_id: str = "SUPPLIER_UNKNOWN",
    decision_date: str = "2024-09-15",
    config: Optional[Dict[str, Any]] = None,
    price_enabled_override: Optional[bool] = None
) -> InventoryDecision:
    """Evaluate candidate stock coverage from 1 to 30 days and select optimal decision."""
    cfg = config or load_inventory_config()
    cost_params = cost_parameters or {
        "inventory_holding_cost_per_unit_per_day": 0.25,
        "storage_cost_per_unit_per_day": 0.15,
        "stockout_cost_per_unit": 3.50,
        "wastage_cost_per_unit": 7.20,
        "ordering_cost_per_po": 50.00
    }

    # Central Price Governance Gate
    if price_enabled_override is not None:
        price_decision_enabled = bool(price_enabled_override)
    else:
        try:
            price_decision_enabled = price_model_usable_for_decision(root=PROJECT_ROOT)
        except Exception:
            price_decision_enabled = False

    gov_status = get_price_model_validation_status(root=PROJECT_ROOT)
    price_status_label = "VALIDATED" if price_decision_enabled else "NOT_VALIDATED"
    decision_mode = "PRICE_AWARE_DEMAND_INVENTORY" if price_decision_enabled else "DEMAND_INVENTORY_ONLY"

    # 1. Interpolate cumulative demand and price curves across 1..30 days
    demand_curves = interpolate_demand_curve(
        horizon_forecasts=demand_forecasts,
        horizon_p10=demand_p10,
        horizon_p50=demand_p50,
        horizon_p90=demand_p90
    )

    price_curves = interpolate_price_curve(
        current_price=current_purchase_price,
        horizon_prices=price_forecasts,
        horizon_p10=price_p10,
        horizon_p50=price_p50,
        horizon_p90=price_p90
    )

    # Service-level safety stock multiplier
    sl_cfg = cfg.get("service_levels", {}).get(service_level, {})
    q_mult = float(sl_cfg.get("quantile_multiplier", 0.75))

    # Lead time demand (interpolated at lead_time_days)
    lt_idx = min(29, max(0, int(round(lead_time_days)) - 1))
    lead_time_demand = float(demand_curves["point_demand"][lt_idx])

    # 2. Evaluate all candidate coverage days d = 1..30
    evaluations: List[CandidateEvaluation] = []

    for d in range(1, 31):
        idx = d - 1
        d_pt = float(demand_curves["point_demand"][idx])
        d_10 = float(demand_curves["p10_demand"][idx])
        d_50 = float(demand_curves["p50_demand"][idx])
        d_90 = float(demand_curves["p90_demand"][idx])

        p_future = float(price_curves["point_price"][idx])

        # Safety stock based on predictive uncertainty
        ss = max(0.0, q_mult * (d_90 - d_50))

        # Required stock to cover d days plus buffer
        req_stock = d_pt + ss

        # Raw replenishment needed
        raw_qty = max(0.0, req_stock - current_inventory - incoming_inventory)

        # Operational rounding
        order_qty = _round_to_constraints(
            raw_qty=raw_qty,
            moq=supplier_minimum_order_quantity,
            case_size=product_case_size
        )

        # Check operational constraints
        c_res = check_constraints(
            candidate_coverage_days=d,
            order_quantity=order_qty,
            current_inventory=current_inventory,
            shelf_life_days=shelf_life_days,
            store_capacity=store_storage_capacity,
            current_purchase_price=current_purchase_price,
            budget_limit=budget_limit
        )

        # Cost and risk calculation
        breakdown, risks = calculate_candidate_cost(
            candidate_coverage_days=d,
            order_quantity=order_qty,
            current_inventory=current_inventory,
            incoming_inventory=incoming_inventory,
            expected_demand=d_pt,
            demand_p10=d_10,
            demand_p50=d_50,
            demand_p90=d_90,
            current_purchase_price=current_purchase_price,
            predicted_future_price=p_future,
            lead_time_days=lead_time_days,
            lead_time_demand=lead_time_demand,
            shelf_life_days=shelf_life_days,
            cost_parameters=cost_params,
            price_decision_enabled=price_decision_enabled
        )

        evaluations.append(CandidateEvaluation(
            coverage_days=d,
            raw_order_quantity=round(raw_qty, 2),
            order_quantity=order_qty,
            expected_demand=round(d_pt, 2),
            safety_stock=round(ss, 2),
            predicted_future_price=round(p_future, 2),
            cost_breakdown=breakdown,
            risk_estimates=risks,
            is_feasible=c_res.is_feasible,
            rejection_reasons=c_res.rejection_reasons,
            net_timing_benefit=breakdown.net_timing_benefit,
            gross_timing_savings=breakdown.future_price_savings
        ))

    # 3. Select optimal candidate with Demand x Price criteria
    feasible_cands = [c for c in evaluations if c.is_feasible]
    if not feasible_cands:
        # If all infeasible, pick least violating or day 1 with zero order
        best_cand = evaluations[0]
        baseline_cand = evaluations[0]
    else:
        # Baseline demand-only: candidate that minimizes cost when price timing arbitrage is excluded
        baseline_cand = min(
            feasible_cands,
            key=lambda c: (c.cost_breakdown.total_expected_cost + c.cost_breakdown.future_price_savings)
        )

        if not price_decision_enabled:
            # PRICE DISABLED: Best candidate is strictly baseline demand candidate!
            # Zero coverage shifts, zero order quantity adjustments from price forecasts.
            best_cand = baseline_cand
        else:
            # Price-aware candidate minimizes total cost including timing savings
            candidate_price_min = min(feasible_cands, key=lambda c: c.cost_breakdown.total_expected_cost)

            # Enforce price decision rules:
            # Maximum extra coverage days due to price rise is 7 days
            max_extra_days = 7
            min_net_benefit = 5.0

            if candidate_price_min.coverage_days > baseline_cand.coverage_days:
                extra_days = candidate_price_min.coverage_days - baseline_cand.coverage_days
                if (
                    extra_days <= max_extra_days
                    and candidate_price_min.cost_breakdown.net_timing_benefit is not None
                    and candidate_price_min.cost_breakdown.net_timing_benefit >= min_net_benefit
                ):
                    best_cand = candidate_price_min
                else:
                    best_cand = baseline_cand
            else:
                best_cand = candidate_price_min

    # 4. Reference Strategy (Fixed 14-day coverage)
    ref_days = int(cfg.get("evaluation", {}).get("reference_strategy_days", 14))
    ref_idx = min(29, max(0, ref_days - 1))
    ref_cand = evaluations[ref_idx]
    savings_vs_ref = ref_cand.cost_breakdown.total_expected_cost - best_cand.cost_breakdown.total_expected_cost

    # 5. Alternative Strategies
    low_days = int(cfg.get("strategies", {}).get("low_stock", {}).get("coverage_days", 7))
    high_days = int(cfg.get("strategies", {}).get("high_safety", {}).get("coverage_days", 21))

    # Bound alternatives within shelf-life if perishable
    if shelf_life_days is not None and shelf_life_days > 0:
        low_days = min(low_days, int(shelf_life_days))
        high_days = min(high_days, int(shelf_life_days))

    low_cand = evaluations[min(29, max(0, low_days - 1))]
    high_cand = evaluations[min(29, max(0, high_days - 1))]

    alt_strategies = {
        "LOW_STOCK_STRATEGY": {
            "coverage_days": low_cand.coverage_days,
            "order_quantity": low_cand.order_quantity,
            "expected_total_cost": round(low_cand.cost_breakdown.total_expected_cost, 2),
            "stockout_risk": low_cand.risk_estimates.preliminary_stockout_risk,
            "overstock_risk": low_cand.risk_estimates.preliminary_overstock_risk,
            "purchase_cost": round(low_cand.cost_breakdown.purchase_cost, 2)
        },
        "BALANCED_STRATEGY": {
            "coverage_days": best_cand.coverage_days,
            "order_quantity": best_cand.order_quantity,
            "expected_total_cost": round(best_cand.cost_breakdown.total_expected_cost, 2),
            "stockout_risk": best_cand.risk_estimates.preliminary_stockout_risk,
            "overstock_risk": best_cand.risk_estimates.preliminary_overstock_risk,
            "purchase_cost": round(best_cand.cost_breakdown.purchase_cost, 2)
        },
        "HIGH_SAFETY_STRATEGY": {
            "coverage_days": high_cand.coverage_days,
            "order_quantity": high_cand.order_quantity,
            "expected_total_cost": round(high_cand.cost_breakdown.total_expected_cost, 2),
            "stockout_risk": high_cand.risk_estimates.preliminary_stockout_risk,
            "overstock_risk": high_cand.risk_estimates.preliminary_overstock_risk,
            "purchase_cost": round(high_cand.cost_breakdown.purchase_cost, 2)
        }
    }

    # 6. Decision Robustness Analysis (P10 vs P50 vs P90 demand)
    # Check what order would be recommended under P10 demand vs P90 demand
    d_pt_sel = best_cand.expected_demand
    ss_p10 = max(0.0, q_mult * (demand_curves["p50_demand"][best_cand.coverage_days - 1] - demand_curves["p10_demand"][best_cand.coverage_days - 1]))
    raw_p10 = max(0.0, demand_curves["p10_demand"][best_cand.coverage_days - 1] + ss_p10 - current_inventory - incoming_inventory)
    order_p10 = _round_to_constraints(raw_p10, supplier_minimum_order_quantity, product_case_size)

    raw_p90 = max(0.0, demand_curves["p90_demand"][best_cand.coverage_days - 1] + best_cand.safety_stock - current_inventory - incoming_inventory)
    order_p90 = _round_to_constraints(raw_p90, supplier_minimum_order_quantity, product_case_size)

    tol = float(cfg.get("decision_thresholds", {}).get("robustness_tolerance_pct", 0.25))
    if best_cand.order_quantity > 0:
        spread = abs(order_p90 - order_p10) / best_cand.order_quantity
        robustness_status = "ROBUST" if spread <= (tol * 2) else "UNCERTAINTY_SENSITIVE"
    else:
        robustness_status = "ROBUST" if (order_p10 == 0 and order_p90 <= supplier_minimum_order_quantity) else "UNCERTAINTY_SENSITIVE"

    # 7. Action and Severity Classification according to Demand x Price Decision Matrix
    pct_price_change = ((best_cand.predicted_future_price / current_purchase_price) - 1.0) * 100.0

    # Stock exhaustion timeline (days until stockout)
    daily_rate = max(1.0, demand_curves["point_demand"][0])
    days_of_stock_left = current_inventory / daily_rate

    # Signal classification
    if not price_decision_enabled:
        price_signal_strength = "PRICE_PREDICTION_UNAVAILABLE"
    elif pct_price_change >= 5.0:
        price_signal_strength = "STRONG_BUY_EARLY_SIGNAL"
    elif pct_price_change >= 2.0:
        price_signal_strength = "MODERATE_BUY_EARLY_SIGNAL"
    elif pct_price_change <= -5.0:
        price_signal_strength = "STRONG_WAIT_SIGNAL"
    elif pct_price_change <= -2.0:
        price_signal_strength = "MODERATE_WAIT_SIGNAL"
    else:
        price_signal_strength = "NEUTRAL_PRICE_SIGNAL"

    # Buy Now vs Wait comparison
    buy_now_cost = best_cand.cost_breakdown.purchase_cost + best_cand.cost_breakdown.holding_cost + best_cand.cost_breakdown.stockout_cost + best_cand.cost_breakdown.wastage_cost + best_cand.cost_breakdown.ordering_cost
    wait_cost = buy_now_cost + (best_cand.cost_breakdown.buy_now_vs_wait_advantage or 0.0)

    if not price_decision_enabled:
        # PURE DEMAND-INVENTORY LOGIC: Action cannot be influenced by price rise/drop forecasts
        if current_inventory <= 0 or days_of_stock_left < lead_time_days:
            action = "URGENT_REORDER"
            severity = "CRITICAL"
            buy_now_vs_wait_rec = "BUY_NOW"
        elif best_cand.order_quantity == 0:
            if days_of_stock_left > 45 or (shelf_life_days and current_inventory > shelf_life_days * daily_rate):
                action = "OVERSTOCK_WARNING"
                severity = "HIGH" if (shelf_life_days and shelf_life_days <= 7) else "MEDIUM"
            else:
                action = "DO_NOT_REORDER"
                severity = "NONE"
            buy_now_vs_wait_rec = "HOLD_CURRENT_STOCK"
        else:
            action = "BUY_NOW"
            severity = "HIGH" if days_of_stock_left <= 7 else "MEDIUM"
            buy_now_vs_wait_rec = "BUY_NOW"
    else:
        # PRICE-AWARE LOGIC
        if current_inventory <= 0 or days_of_stock_left < lead_time_days:
            buy_now_vs_wait_rec = "BUY_NOW"
        elif pct_price_change <= -2.0 and days_of_stock_left >= (lead_time_days + 5.0):
            buy_now_vs_wait_rec = "WAIT"
        elif pct_price_change <= -2.0:
            buy_now_vs_wait_rec = "LEAN_BUY"
        elif pct_price_change >= 2.0 and (best_cand.cost_breakdown.net_timing_benefit or 0.0) >= 5.0:
            buy_now_vs_wait_rec = "BUY_NOW"
        else:
            buy_now_vs_wait_rec = "BUY_NOW" if best_cand.order_quantity > 0 else "WAIT"

        # Case E: Critical stockout overrides price drop wait!
        if current_inventory <= 0 or days_of_stock_left < lead_time_days:
            action = "URGENT_REORDER"
            severity = "CRITICAL"
        elif best_cand.order_quantity == 0:
            if days_of_stock_left > 45 or (shelf_life_days and current_inventory > shelf_life_days * daily_rate):
                action = "OVERSTOCK_WARNING"
                severity = "HIGH" if (shelf_life_days and shelf_life_days <= 7) else "MEDIUM"
            elif pct_price_change < -1.5 and days_of_stock_left >= (lead_time_days + 5):
                action = "WAIT_FOR_LOWER_PRICE"
                severity = "LOW"
            else:
                action = "DO_NOT_REORDER"
                severity = "NONE"
        else:
            # Order quantity > 0
            if pct_price_change < -1.5 and days_of_stock_left >= (lead_time_days + 3):
                action = "BUY_SMALLER_QUANTITY"
                severity = "MEDIUM"
            else:
                action = "BUY_NOW"
                severity = "HIGH" if (pct_price_change > 2.0 and days_of_stock_left <= 7) else "MEDIUM"

    # 8. Business Explanations / Reasons (Grounding the 4 Decision Criteria)
    reasons = []
    # Criterion 1: Demand Outlook
    reasons.append(
        f"Demand Outlook: Expected {best_cand.coverage_days}-day cumulative demand is {best_cand.expected_demand:.1f} units "
        f"(safety buffer: {best_cand.safety_stock:.1f} units)."
    )
    # Criterion 2: Inventory & Supply Risk
    reasons.append(
        f"Inventory Risk: Current inventory ({current_inventory:.0f} units) covers approximately {days_of_stock_left:.1f} days; "
        f"supplier lead time is {lead_time_days:.1f} days."
    )
    # Criterion 3: Purchase-Price Outlook & Signal
    if not price_decision_enabled:
        reasons.append(
            f"Price Outlook: Price model validation gate is active [PRICE_MODEL_STATUS: {price_status_label}]. "
            f"Speculative future price forecasts are quarantined; current price RM {current_purchase_price:.2f} is used purely for accounting."
        )
    elif pct_price_change >= 2.0:
        reasons.append(
            f"Price Outlook: Supplier price projected to rise by {pct_price_change:+.1f}% (RM {best_cand.predicted_future_price:.2f}) "
            f"[{price_signal_strength}], capturing RM {best_cand.cost_breakdown.future_price_savings:.2f} forward avoidance."
        )
    elif pct_price_change <= -2.0:
        reasons.append(
            f"Price Outlook: Supplier price projected to soften by {pct_price_change:+.1f}% (RM {best_cand.predicted_future_price:.2f}) "
            f"[{price_signal_strength}], favoring lean/controlled replenishment."
        )
    else:
        reasons.append(
            f"Price Outlook: Supplier price expected to remain stable at RM {best_cand.predicted_future_price:.2f} ({pct_price_change:+.1f}%)."
        )
    # Criterion 4: Economic & Overstock Trade-off
    if not price_decision_enabled:
        reasons.append(
            f"Economic Trade-Off: Inventory optimization operating in '{decision_mode}'. "
            f"Expected portfolio cost: RM {best_cand.cost_breakdown.total_expected_cost:.2f} (Zero price timing adjustments applied)."
        )
    else:
        timing_val = best_cand.cost_breakdown.net_timing_benefit if best_cand.cost_breakdown.net_timing_benefit is not None else 0.0
        adv_val = best_cand.cost_breakdown.buy_now_vs_wait_advantage if best_cand.cost_breakdown.buy_now_vs_wait_advantage is not None else 0.0
        reasons.append(
            f"Economic Trade-Off: Net price timing benefit is RM {timing_val:.2f} "
            f"(Buy Now advantage vs Wait: RM {adv_val:.2f})."
        )

    # Attach calibrated horizon details
    sel_idx = best_cand.coverage_days - 1
    d_10_sel = float(demand_curves["p10_demand"][sel_idx])
    d_50_sel = float(demand_curves["p50_demand"][sel_idx])
    d_90_sel = float(demand_curves["p90_demand"][sel_idx])

    p_10_sel = float(price_curves["p10_price"][sel_idx])
    p_50_sel = float(price_curves["p50_price"][sel_idx])
    p_90_sel = float(price_curves["p90_price"][sel_idx])

    # Calculate deltas relative to demand-only baseline
    baseline_qty = baseline_cand.order_quantity
    baseline_cov = baseline_cand.coverage_days
    qty_delta = best_cand.order_quantity - baseline_qty if price_decision_enabled else 0.0
    cov_delta = best_cand.coverage_days - baseline_cov if price_decision_enabled else 0
    net_timing_benefit = best_cand.cost_breakdown.net_timing_benefit if price_decision_enabled else None

    price_influenced_qty = bool(price_decision_enabled and abs(qty_delta) > 1e-6)
    price_influenced_cov = bool(price_decision_enabled and cov_delta != 0)

    return InventoryDecision(
        decision_date=decision_date,
        store_id=store_id,
        product_id=product_id,
        supplier_id=supplier_id,
        recommended_action=action,
        severity=severity,
        recommended_order_quantity=best_cand.order_quantity,
        recommended_stock_coverage_days=best_cand.coverage_days,
        current_inventory=current_inventory,
        incoming_inventory=incoming_inventory,
        current_purchase_price=current_purchase_price,
        predicted_future_price=best_cand.predicted_future_price,
        expected_price_change_pct=pct_price_change,
        expected_demand_for_coverage=best_cand.expected_demand,
        demand_p10=d_10_sel,
        demand_p50=d_50_sel,
        demand_p90=d_90_sel,
        price_p10=p_10_sel,
        price_p50=p_50_sel,
        price_p90=p_90_sel,
        cost_breakdown=best_cand.cost_breakdown,
        risk_estimates=best_cand.risk_estimates,
        expected_savings_vs_reference=savings_vs_ref,
        reference_strategy_name=f"FIXED_{ref_days}_DAY_STOCK",
        reference_order_quantity=ref_cand.order_quantity,
        reference_total_cost=ref_cand.cost_breakdown.total_expected_cost,
        robustness_status=robustness_status,
        top_reasons=reasons,
        alternative_strategies=alt_strategies,
        all_candidate_evaluations=evaluations,
        baseline_demand_only_quantity=baseline_qty,
        baseline_demand_only_coverage_days=baseline_cov,
        price_adjusted_quantity_delta=qty_delta,
        price_adjusted_coverage_delta=cov_delta,
        net_price_timing_benefit=net_timing_benefit,
        price_signal_strength=price_signal_strength,
        buy_now_vs_wait_recommendation=buy_now_vs_wait_rec,
        buy_now_cost=buy_now_cost,
        wait_cost=wait_cost,
        price_model_status=price_status_label,
        price_decision_enabled=price_decision_enabled,
        price_influenced_order_quantity=price_influenced_qty,
        price_influenced_coverage=price_influenced_cov,
        price_order_adjustment_units=qty_delta,
        price_coverage_adjustment_days=cov_delta,
        decision_mode=decision_mode
    )


def optimise_inventory_batch(
    prediction_frame: pd.DataFrame,
    inventory_frame: pd.DataFrame,
    product_metadata: pd.DataFrame,
    supplier_metadata: pd.DataFrame,
    store_metadata: Optional[pd.DataFrame] = None,
    decision_date: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> DecisionBatchResult:
    """Run batch inventory optimisation over all series in the provided prediction frame."""
    cfg = config or load_inventory_config()

    # Pre-index metadata for fast vector lookups
    prod_dict = product_metadata.set_index("product_id").to_dict(orient="index") if "product_id" in product_metadata.columns else {}
    supp_dict = supplier_metadata.set_index("supplier_id").to_dict(orient="index") if "supplier_id" in supplier_metadata.columns else {}
    store_dict = store_metadata.set_index("store_id").to_dict(orient="index") if (store_metadata is not None and "store_id" in store_metadata.columns) else {}

    # Extract unique series
    series_keys = prediction_frame[["store_id", "product_id"]].drop_duplicates().values.tolist()

    rec_records = []
    cand_records = []
    comp_records = []
    rob_records = []

    for store_id, product_id in series_keys:
        p_sub = prediction_frame[
            (prediction_frame["store_id"] == store_id) & (prediction_frame["product_id"] == product_id)
        ]
        if p_sub.empty:
            continue

        # Extract horizon forecasts
        d_fc = {}
        d_p10, d_p50, d_p90 = {}, {}, {}
        p_fc = {}
        p_p10, p_p50, p_p90 = {}, {}, {}

        for _, r in p_sub.iterrows():
            h = int(r["horizon"])
            d_fc[h] = float(r.get("point_prediction", r.get("demand_forecast", 0.0)))
            d_p10[h] = float(r.get("calibrated_p10", r.get("demand_p10", d_fc[h] * 0.8)))
            d_p50[h] = float(r.get("calibrated_p50", r.get("demand_p50", d_fc[h])))
            d_p90[h] = float(r.get("calibrated_p90", r.get("demand_p90", d_fc[h] * 1.2)))

            p_fc[h] = float(r.get("predicted_future_price", r.get("price_forecast", 10.0)))
            p_p10[h] = float(r.get("price_p10", p_fc[h] * 0.95))
            p_p50[h] = float(r.get("price_p50", p_fc[h]))
            p_p90[h] = float(r.get("price_p90", p_fc[h] * 1.05))

        # Look up current inventory
        inv_sub = inventory_frame[
            (inventory_frame["store_id"] == store_id) & (inventory_frame["product_id"] == product_id)
        ]
        if not inv_sub.empty:
            inv_row = inv_sub.iloc[-1]
            curr_inv = float(inv_row.get("current_inventory_quantity", inv_row.get("inventory_current", 0.0)))
            inc_inv = float(inv_row.get("incoming_inventory_quantity", 0.0))
            curr_price = float(inv_row.get("base_purchase_price", inv_row.get("purchase_price", 10.0)))
            d_date = str(inv_row.get("date", decision_date or "2024-09-15"))
        else:
            curr_inv = 50.0
            inc_inv = 0.0
            curr_price = 10.0
            d_date = decision_date or "2024-09-15"

        # Look up product metadata
        p_meta = prod_dict.get(product_id, {})
        supplier_id = str(p_meta.get("supplier_id", "SUP_01"))
        shelf_life = float(p_meta.get("product_shelf_life_days", 999.0))
        if shelf_life >= 100:
            shelf_life = None  # Non-perishable
        case_size = int(p_meta.get("product_case_size", 1))

        # Look up supplier metadata
        s_meta = supp_dict.get(supplier_id, {})
        lead_time = float(s_meta.get("supplier_lead_time_days", 2.0))
        moq = float(s_meta.get("supplier_minimum_order_quantity", 10.0))

        # Look up store metadata
        st_meta = store_dict.get(store_id, {})
        store_cap = float(st_meta.get("store_storage_capacity", 20000.0))

        # Run optimisation
        decision = optimise_inventory_decision(
            demand_forecasts=d_fc,
            price_forecasts=p_fc,
            demand_p10=d_p10,
            demand_p50=d_50,
            demand_p90=d_90,
            price_p10=p_p10,
            price_p50=p_p50,
            price_p90=p_p90,
            current_inventory=curr_inv,
            incoming_inventory=inc_inv,
            current_purchase_price=curr_price,
            lead_time_days=lead_time,
            supplier_minimum_order_quantity=moq,
            product_case_size=case_size,
            shelf_life_days=shelf_life,
            store_storage_capacity=store_cap,
            store_id=store_id,
            product_id=product_id,
            supplier_id=supplier_id,
            decision_date=d_date,
            config=cfg
        )

        # 1. Recommendation Record
        rec_records.append({
            "decision_date": decision.decision_date,
            "store_id": decision.store_id,
            "product_id": decision.product_id,
            "supplier_id": decision.supplier_id,
            "recommended_action": decision.recommended_action,
            "severity": decision.severity,
            "recommended_order_quantity": decision.recommended_order_quantity,
            "recommended_stock_coverage_days": decision.recommended_stock_coverage_days,
            "current_inventory": decision.current_inventory,
            "incoming_inventory": decision.incoming_inventory,
            "current_purchase_price": decision.current_purchase_price,
            "predicted_future_price": decision.predicted_future_price,
            "expected_price_change_pct": decision.expected_price_change_pct,
            "expected_demand_for_coverage": decision.expected_demand_for_coverage,
            "demand_p10": decision.demand_p10,
            "demand_p50": decision.demand_p50,
            "demand_p90": decision.demand_p90,
            "price_p10": decision.price_p10,
            "price_p50": decision.price_p50,
            "price_p90": decision.price_p90,
            "expected_total_cost": decision.cost_breakdown.total_expected_cost,
            "expected_purchase_cost": decision.cost_breakdown.purchase_cost,
            "expected_holding_cost": decision.cost_breakdown.holding_cost,
            "expected_stockout_cost": decision.cost_breakdown.stockout_cost,
            "expected_wastage_cost": decision.cost_breakdown.wastage_cost,
            "future_price_savings": decision.cost_breakdown.future_price_savings,
            "net_timing_benefit": decision.cost_breakdown.net_timing_benefit,
            "buy_now_vs_wait_advantage": decision.cost_breakdown.buy_now_vs_wait_advantage,
            "baseline_demand_only_quantity": decision.baseline_demand_only_quantity,
            "baseline_demand_only_coverage_days": decision.baseline_demand_only_coverage_days,
            "price_adjusted_quantity_delta": decision.price_adjusted_quantity_delta,
            "price_adjusted_coverage_delta": decision.price_adjusted_coverage_delta,
            "net_price_timing_benefit": decision.net_price_timing_benefit,
            "price_signal_strength": decision.price_signal_strength,
            "buy_now_vs_wait_recommendation": decision.buy_now_vs_wait_recommendation,
            "buy_now_cost": decision.buy_now_cost,
            "wait_cost": decision.wait_cost,
            "preliminary_stockout_risk": decision.risk_estimates.preliminary_stockout_risk,
            "preliminary_overstock_risk": decision.risk_estimates.preliminary_overstock_risk,
            "expected_savings_vs_reference": decision.expected_savings_vs_reference,
            "robustness_status": decision.robustness_status,
            "price_model_status": decision.price_model_status,
            "price_decision_enabled": decision.price_decision_enabled,
            "price_influenced_order_quantity": decision.price_influenced_order_quantity,
            "price_influenced_coverage": decision.price_influenced_coverage,
            "price_order_adjustment_units": decision.price_order_adjustment_units,
            "price_coverage_adjustment_days": decision.price_coverage_adjustment_days,
            "decision_mode": decision.decision_mode,
            "top_reason_1": decision.top_reasons[0] if len(decision.top_reasons) > 0 else "",
            "top_reason_2": decision.top_reasons[1] if len(decision.top_reasons) > 1 else "",
            "top_reason_3": decision.top_reasons[2] if len(decision.top_reasons) > 2 else ""
        })

        # 2. Candidate Evaluations (All 30 days)
        for c in decision.all_candidate_evaluations:
            cand_records.append({
                "decision_date": decision.decision_date,
                "store_id": decision.store_id,
                "product_id": decision.product_id,
                "supplier_id": decision.supplier_id,
                "candidate_coverage_days": c.coverage_days,
                "raw_order_quantity": c.raw_order_quantity,
                "order_quantity": c.order_quantity,
                "expected_demand": c.expected_demand,
                "safety_stock": c.safety_stock,
                "purchase_cost": c.cost_breakdown.purchase_cost,
                "holding_cost": c.cost_breakdown.holding_cost,
                "stockout_cost": c.cost_breakdown.stockout_cost,
                "wastage_cost": c.cost_breakdown.wastage_cost,
                "ordering_cost": c.cost_breakdown.ordering_cost,
                "future_price_savings": c.cost_breakdown.future_price_savings,
                "net_timing_benefit": c.cost_breakdown.net_timing_benefit,
                "buy_now_vs_wait_advantage": c.cost_breakdown.buy_now_vs_wait_advantage,
                "total_expected_cost": c.cost_breakdown.total_expected_cost,
                "preliminary_stockout_risk": c.risk_estimates.preliminary_stockout_risk,
                "preliminary_overstock_risk": c.risk_estimates.preliminary_overstock_risk,
                "is_feasible": c.is_feasible,
                "rejection_reasons": "; ".join(c.rejection_reasons)
            })

        # 3. Strategy Comparison Record
        for strat_name, s_data in decision.alternative_strategies.items():
            comp_records.append({
                "store_id": decision.store_id,
                "product_id": decision.product_id,
                "strategy": strat_name,
                "coverage_days": s_data["coverage_days"],
                "order_quantity": s_data["order_quantity"],
                "expected_total_cost": s_data["expected_total_cost"],
                "stockout_risk": s_data["stockout_risk"],
                "overstock_risk": s_data["overstock_risk"],
                "purchase_cost": s_data["purchase_cost"]
            })

        # Fixed 14-day reference row
        comp_records.append({
            "store_id": decision.store_id,
            "product_id": decision.product_id,
            "strategy": decision.reference_strategy_name,
            "coverage_days": 14,
            "order_quantity": decision.reference_order_quantity,
            "expected_total_cost": decision.reference_total_cost,
            "stockout_risk": 0.0,
            "overstock_risk": 0.0,
            "purchase_cost": decision.reference_order_quantity * decision.current_purchase_price
        })

        # 4. Robustness Record
        rob_records.append({
            "store_id": decision.store_id,
            "product_id": decision.product_id,
            "recommended_action": decision.recommended_action,
            "recommended_coverage_days": decision.recommended_stock_coverage_days,
            "recommended_order_quantity": decision.recommended_order_quantity,
            "robustness_status": decision.robustness_status
        })

    return DecisionBatchResult(
        recommendations_df=pd.DataFrame(rec_records),
        candidate_evaluations_df=pd.DataFrame(cand_records),
        strategy_comparison_df=pd.DataFrame(comp_records),
        robustness_df=pd.DataFrame(rob_records)
    )
