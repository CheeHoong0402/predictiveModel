"""Cost Calculation, Preliminary Risk Estimation, and Operational Constraints Engine."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CostBreakdown:
    """Detailed breakdown of expected costs and timing savings for a candidate decision."""
    purchase_cost: float
    holding_cost: float
    stockout_cost: float
    wastage_cost: float
    ordering_cost: float
    future_price_savings: float
    total_expected_cost: float
    net_timing_benefit: Optional[float] = None
    buy_now_vs_wait_advantage: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "purchase_cost": round(self.purchase_cost, 2),
            "holding_cost": round(self.holding_cost, 2),
            "stockout_cost": round(self.stockout_cost, 2),
            "wastage_cost": round(self.wastage_cost, 2),
            "ordering_cost": round(self.ordering_cost, 2),
            "future_price_savings": round(self.future_price_savings, 2),
            "total_expected_cost": round(self.total_expected_cost, 2),
            "net_timing_benefit": round(self.net_timing_benefit, 2) if self.net_timing_benefit is not None else None,
            "buy_now_vs_wait_advantage": round(self.buy_now_vs_wait_advantage, 2) if self.buy_now_vs_wait_advantage is not None else None
        }


@dataclass
class RiskEstimates:
    """Preliminary risk proxies based on calibrated prediction intervals."""
    preliminary_stockout_risk: float       # Scale 0.0 - 1.0
    preliminary_overstock_risk: float        # Scale 0.0 - 1.0
    preliminary_wastage_risk: float          # Scale 0.0 - 1.0
    lead_time_stockout_risk: float           # Stockout risk during supplier replenishment
    expected_lost_units: float
    expected_expired_units: float


@dataclass
class ConstraintCheckResult:
    """Outcome of constraint evaluations on a candidate order decision."""
    is_feasible: bool
    rejection_reasons: List[str] = field(default_factory=list)


def check_constraints(
    candidate_coverage_days: int,
    order_quantity: float,
    current_inventory: float,
    shelf_life_days: Optional[float],
    store_capacity: Optional[float],
    current_purchase_price: float,
    budget_limit: Optional[float] = None
) -> ConstraintCheckResult:
    """Evaluate operational constraints for a candidate decision."""
    reasons = []

    # 1. Non-negative order
    if order_quantity < 0:
        reasons.append(f"Negative order quantity ({order_quantity}) not permitted")

    # 2. Perishable shelf-life constraint
    if shelf_life_days is not None and shelf_life_days > 0:
        if candidate_coverage_days > shelf_life_days:
            reasons.append(
                f"Candidate coverage ({candidate_coverage_days}d) exceeds perishable shelf life ({shelf_life_days}d)"
            )

    # 3. Store storage capacity constraint
    if store_capacity is not None and store_capacity > 0:
        post_order_inventory = current_inventory + order_quantity
        if post_order_inventory > store_capacity:
            reasons.append(
                f"Projected inventory ({post_order_inventory:.0f}) exceeds store capacity ({store_capacity:.0f})"
            )

    # 4. Purchase budget constraint (optional)
    if budget_limit is not None and budget_limit > 0:
        order_cost = order_quantity * current_purchase_price
        if order_cost > budget_limit:
            reasons.append(
                f"Purchase cost (RM {order_cost:.2f}) exceeds available budget (RM {budget_limit:.2f})"
            )

    return ConstraintCheckResult(is_feasible=(len(reasons) == 0), rejection_reasons=reasons)


def calculate_candidate_cost(
    candidate_coverage_days: int,
    order_quantity: float,
    current_inventory: float,
    incoming_inventory: float,
    expected_demand: float,
    demand_p10: float,
    demand_p50: float,
    demand_p90: float,
    current_purchase_price: float,
    predicted_future_price: float,
    lead_time_days: float,
    lead_time_demand: float,
    shelf_life_days: Optional[float],
    cost_parameters: Dict[str, float],
    price_decision_enabled: bool = False
) -> Tuple[CostBreakdown, RiskEstimates]:
    """Calculate exact cost components and risk proxies for a single candidate decision.
    
    Expected Total Cost Formula:
        Total Cost = Purchase Cost
                   + Holding Cost
                   + Stockout Cost
                   + Wastage Cost
                   + Ordering Cost
                   - Future Price Savings (only if price_decision_enabled == True)
    """
    # Unpack cost rates with standard defaults
    c_hold_daily = float(cost_parameters.get("inventory_holding_cost_per_unit_per_day", 0.25))
    c_storage_daily = float(cost_parameters.get("storage_cost_per_unit_per_day", 0.15))
    c_hold_total_daily = c_hold_daily + c_storage_daily

    c_stockout = float(cost_parameters.get("stockout_cost_per_unit", 3.50))
    c_wastage = float(cost_parameters.get("wastage_cost_per_unit", 7.20))
    c_order_fixed = float(cost_parameters.get("ordering_cost_per_po", 50.00))

    # 1. Purchase Cost
    purchase_cost = order_quantity * current_purchase_price

    # 2. Ordering Cost (Fixed fee per purchase order placed)
    ordering_cost = c_order_fixed if order_quantity > 0 else 0.0

    # 3. Holding Cost
    # Start inventory after delivery arrives: current + incoming + order
    total_avail_stock = current_inventory + incoming_inventory + order_quantity
    # End inventory at end of coverage period
    end_stock = max(0.0, total_avail_stock - expected_demand)
    # Average inventory held over the period
    avg_inventory = (total_avail_stock + end_stock) / 2.0
    holding_cost = max(0.0, avg_inventory) * c_hold_total_daily * float(candidate_coverage_days)

    # 4. Stockout Risk and Cost
    # Stockout occurs if true demand reaches or exceeds P90 and exhausts available stock
    unfulfilled_demand = max(0.0, demand_p90 - total_avail_stock)
    stockout_cost = unfulfilled_demand * c_stockout

    # Risk proxy normalized against uncertainty width
    unc_width = max(1.0, demand_p90 - demand_p50)
    stockout_risk_proxy = float(np.clip(unfulfilled_demand / unc_width, 0.0, 1.0))

    # Lead-time specific stockout risk before order arrives
    lead_time_shortfall = max(0.0, lead_time_demand - current_inventory)
    lead_time_risk = float(np.clip(lead_time_shortfall / max(1.0, lead_time_demand), 0.0, 1.0))
    if current_inventory <= 0:
        lead_time_risk = 1.0

    # 5. Perishable Wastage Risk and Cost
    expired_units = 0.0
    wastage_cost = 0.0
    wastage_risk_proxy = 0.0

    if shelf_life_days is not None and shelf_life_days > 0:
        # If stock exceeds expected demand over shelf life period
        shelf_demand = expected_demand * (shelf_life_days / max(1.0, float(candidate_coverage_days)))
        if total_avail_stock > shelf_demand:
            expired_units = total_avail_stock - shelf_demand
            wastage_cost = expired_units * c_wastage
            wastage_risk_proxy = float(np.clip(expired_units / max(1.0, total_avail_stock), 0.0, 1.0))
        elif candidate_coverage_days > shelf_life_days:
            # Entire excess days spoiled
            excess_ratio = (candidate_coverage_days - shelf_life_days) / candidate_coverage_days
            expired_units = total_avail_stock * excess_ratio
            wastage_cost = expired_units * c_wastage
            wastage_risk_proxy = float(np.clip(excess_ratio, 0.0, 1.0))

    # 6. Overstock Risk Proxy
    # Stock exceeds P10 demand by significant buffer
    excess_over_p10 = max(0.0, total_avail_stock - (demand_p10 * 1.5))
    overstock_risk_proxy = float(np.clip(excess_over_p10 / max(1.0, total_avail_stock), 0.0, 1.0))

    # 7. Future Price Savings and Net Timing Benefit (Timing Arbitrage)
    if price_decision_enabled:
        price_delta = predicted_future_price - current_purchase_price
        gross_timing_saving = order_quantity * max(0.0, price_delta)
        delta_holding_cost = (order_quantity / 2.0) * c_hold_total_daily * float(candidate_coverage_days)
        net_timing_benefit: Optional[float] = gross_timing_saving - delta_holding_cost - wastage_cost
        future_price_savings = gross_timing_saving

        buy_now_cost = purchase_cost + holding_cost + stockout_cost + wastage_cost + ordering_cost
        if order_quantity > 0:
            future_purch_cost = order_quantity * predicted_future_price
            current_holding = (max(0.0, current_inventory) / 2.0) * c_hold_total_daily * float(candidate_coverage_days)
            extra_lead_time_shortfall = max(0.0, lead_time_demand - current_inventory)
            wait_stockout = stockout_cost + (extra_lead_time_shortfall * c_stockout)
            wait_cost = future_purch_cost + current_holding + wait_stockout + ordering_cost
            buy_now_vs_wait_advantage: Optional[float] = wait_cost - buy_now_cost
        else:
            buy_now_vs_wait_advantage = 0.0
    else:
        # Price decision DISABLED by model governance:
        # No speculative timing benefits or future price savings allowed.
        future_price_savings = 0.0
        net_timing_benefit = None
        buy_now_vs_wait_advantage = None

    # 8. Expected Total Cost
    total_expected_cost = (
        purchase_cost
        + holding_cost
        + stockout_cost
        + wastage_cost
        + ordering_cost
        - future_price_savings
    )

    breakdown = CostBreakdown(
        purchase_cost=purchase_cost,
        holding_cost=holding_cost,
        stockout_cost=stockout_cost,
        wastage_cost=wastage_cost,
        ordering_cost=ordering_cost,
        future_price_savings=future_price_savings,
        total_expected_cost=total_expected_cost,
        net_timing_benefit=net_timing_benefit,
        buy_now_vs_wait_advantage=buy_now_vs_wait_advantage
    )

    risks = RiskEstimates(
        preliminary_stockout_risk=round(stockout_risk_proxy, 3),
        preliminary_overstock_risk=round(overstock_risk_proxy, 3),
        preliminary_wastage_risk=round(wastage_risk_proxy, 3),
        lead_time_stockout_risk=round(lead_time_risk, 3),
        expected_lost_units=round(unfulfilled_demand, 2),
        expected_expired_units=round(expired_units, 2)
    )

    return breakdown, risks
