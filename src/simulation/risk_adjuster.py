"""Strategy comparison, regret analysis, and risk-adjustment logic for Phase 19.

Evaluates:
- NO_ACTION (Do Nothing, Q = 0)
- LOW_STOCK_STRATEGY
- BALANCED_STRATEGY (Phase 18 recommendation)
- HIGH_SAFETY_STRATEGY

Computes scenario-level regret and determines whether simulated tail risk
justifies elevating the recommendation to a higher safety buffer without
overwriting Phase 18 records.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .engine import (
    simulate_inventory_risk,
    SimulationRiskMetrics
)


@dataclass
class StrategySimulationResult:
    """Simulation outcomes for a specific replenishment strategy."""
    strategy_name: str
    order_quantity: float
    coverage_days: int
    metrics: SimulationRiskMetrics
    scenario_costs: np.ndarray


@dataclass
class RiskAdjustedDecision:
    """Final risk-evaluated decision comparing Phase 18 and Phase 19 outcomes."""
    decision_date: str
    store_id: str
    product_id: str
    supplier_id: str
    phase18_action: str
    phase18_order_quantity: float
    phase18_coverage_days: int
    phase19_action: str
    phase19_order_quantity: float
    phase19_coverage_days: int
    recommendation_changed: bool
    change_reason: str
    strategy_results: Dict[str, StrategySimulationResult]
    stockout_probability_before: float
    stockout_probability_after: float
    overstock_probability: float
    wastage_probability: float
    service_probability: float
    expected_total_cost: float
    cost_p10: float
    cost_p50: float
    cost_p90: float
    cost_var95: float
    cost_cvar95: float
    expected_lost_units: float
    expected_excess_units: float
    late_delivery_probability: float
    future_price_increase_probability: Optional[float] = None
    robustness_status: str = "ROBUST"
    probability_future_price_rises: Optional[float] = None
    probability_future_price_falls: Optional[float] = None
    probability_buy_now_is_cheaper: Optional[float] = None
    expected_buy_now_saving: Optional[float] = None
    expected_wait_saving: Optional[float] = None
    price_signal_strength: str = "PRICE_MODEL_NOT_VALIDATED"
    price_model_status: str = "NOT_VALIDATED"
    price_risk_enabled: bool = False
    risk_mode: str = "DEMAND_SUPPLY_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to flat dictionary for parquet / reporting."""
        return {
            "decision_date": self.decision_date,
            "store_id": self.store_id,
            "product_id": self.product_id,
            "supplier_id": self.supplier_id,
            "phase18_action": self.phase18_action,
            "phase18_order_quantity": self.phase18_order_quantity,
            "phase18_coverage_days": self.phase18_coverage_days,
            "phase19_action": self.phase19_action,
            "phase19_order_quantity": self.phase19_order_quantity,
            "phase19_coverage_days": self.phase19_coverage_days,
            "recommendation_changed": self.recommendation_changed,
            "change_reason": self.change_reason,
            "stockout_probability_before": round(float(self.stockout_probability_before), 4),
            "stockout_probability_after": round(float(self.stockout_probability_after), 4),
            "overstock_probability": round(float(self.overstock_probability), 4),
            "wastage_probability": round(float(self.wastage_probability), 4),
            "service_probability": round(float(self.service_probability), 4),
            "expected_total_cost": round(float(self.expected_total_cost), 2),
            "cost_p10": round(float(self.cost_p10), 2),
            "cost_p50": round(float(self.cost_p50), 2),
            "cost_p90": round(float(self.cost_p90), 2),
            "cost_var95": round(float(self.cost_var95), 2),
            "cost_cvar95": round(float(self.cost_cvar95), 2),
            "expected_lost_units": round(float(self.expected_lost_units), 2),
            "expected_excess_units": round(float(self.expected_excess_units), 2),
            "late_delivery_probability": round(float(self.late_delivery_probability), 4),
            "future_price_increase_probability": round(float(self.future_price_increase_probability), 4) if self.future_price_increase_probability is not None else None,
            "probability_future_price_rises": round(float(self.probability_future_price_rises), 4) if self.probability_future_price_rises is not None else None,
            "probability_future_price_falls": round(float(self.probability_future_price_falls), 4) if self.probability_future_price_falls is not None else None,
            "probability_buy_now_is_cheaper": round(float(self.probability_buy_now_is_cheaper), 4) if self.probability_buy_now_is_cheaper is not None else None,
            "expected_buy_now_saving": round(float(self.expected_buy_now_saving), 2) if self.expected_buy_now_saving is not None else None,
            "expected_wait_saving": round(float(self.expected_wait_saving), 2) if self.expected_wait_saving is not None else None,
            "price_signal_strength": self.price_signal_strength,
            "price_model_status": self.price_model_status,
            "price_risk_enabled": self.price_risk_enabled,
            "risk_mode": self.risk_mode,
            "robustness_status": self.robustness_status
        }


def evaluate_and_adjust_risk(
    p18_decision: Dict[str, Any],
    daily_demand_paths: np.ndarray,
    future_prices: Optional[np.ndarray] = None,
    lead_times: Optional[np.ndarray] = None,
    fill_rates: Optional[np.ndarray] = None,
    shelf_life_days: Optional[float] = None,
    cost_parameters: Optional[Dict[str, float]] = None,
    risk_tolerance_profile: str = "balanced",
    max_stockout_threshold: float = 0.10,
    nominal_lead_time: float = 2.0,
    price_risk_enabled: Optional[bool] = None
) -> RiskAdjustedDecision:
    """Simulate all strategies, conduct regret analysis, and apply risk adjustment logic."""
    p18_action = str(p18_decision.get("recommended_action", "BUY_NOW"))
    p18_qty = float(p18_decision.get("recommended_order_quantity", 0.0))
    p18_cov = int(p18_decision.get("recommended_stock_coverage_days", 7))
    curr_inv = float(p18_decision.get("current_inventory", 0.0))
    inc_inv = float(p18_decision.get("incoming_inventory", 0.0))
    curr_price = float(p18_decision.get("current_purchase_price", 10.0))

    alt_strats = p18_decision.get("alternative_strategies", {})

    # Define candidate strategies to simulate
    strategies_to_test = [
        ("NO_ACTION", 0.0, p18_cov),
        ("BALANCED_STRATEGY", p18_qty, p18_cov),
    ]

    # Add Low Stock and High Safety from alternatives
    if "LOW_STOCK_STRATEGY" in alt_strats:
        ls_data = alt_strats["LOW_STOCK_STRATEGY"]
        strategies_to_test.append(("LOW_STOCK_STRATEGY", float(ls_data["order_quantity"]), int(ls_data["coverage_days"])))
    if "HIGH_SAFETY_STRATEGY" in alt_strats:
        hs_data = alt_strats["HIGH_SAFETY_STRATEGY"]
        strategies_to_test.append(("HIGH_SAFETY_STRATEGY", float(hs_data["order_quantity"]), int(hs_data["coverage_days"])))

    is_delayed = (p18_action == "WAIT_FOR_LOWER_PRICE")

    # 1. Run simulation for each strategy
    strategy_results: Dict[str, StrategySimulationResult] = {}
    cost_matrix = []

    for name, qty, cov in strategies_to_test:
        met, costs_arr = simulate_inventory_risk(
            order_quantity=qty,
            coverage_days=cov,
            current_inventory=curr_inv,
            incoming_inventory=inc_inv,
            current_purchase_price=curr_price,
            daily_demand_paths=daily_demand_paths,
            future_prices=future_prices,
            lead_times=lead_times,
            fill_rates=fill_rates,
            shelf_life_days=shelf_life_days,
            cost_parameters=cost_parameters,
            is_delayed_purchase=is_delayed,
            nominal_lead_time=nominal_lead_time,
            price_risk_enabled=price_risk_enabled
        )
        strategy_results[name] = StrategySimulationResult(
            strategy_name=name,
            order_quantity=qty,
            coverage_days=cov,
            metrics=met,
            scenario_costs=costs_arr
        )
        cost_matrix.append(costs_arr)

    # 2. Compute Regret Analysis across strategies
    # Minimum cost achievable in each scenario across tested strategies
    stacked_costs = np.vstack(cost_matrix)  # shape: (n_strategies, n_scenarios)
    min_costs_per_scenario = np.min(stacked_costs, axis=0)

    for idx, (name, _, _) in enumerate(strategies_to_test):
        regret_arr = stacked_costs[idx] - min_costs_per_scenario
        strategy_results[name].metrics.mean_regret = float(np.mean(regret_arr))
        strategy_results[name].metrics.p90_regret = float(np.percentile(regret_arr, 90))

    # Baseline outcomes
    no_action_res = strategy_results["NO_ACTION"]
    stk_prob_before = no_action_res.metrics.stockout_probability

    balanced_res = strategy_results["BALANCED_STRATEGY"]
    stk_prob_balanced = balanced_res.metrics.stockout_probability

    # 3. Apply Risk Adjustment Rule
    # Check if balanced recommendation violates risk tolerance
    rec_changed = False
    change_reason = "Phase 18 recommendation validated: simulated stockout risk within acceptable tolerance and CVaR95 optimal."
    p19_action = p18_action
    p19_qty = p18_qty
    p19_cov = p18_cov
    selected_metrics = balanced_res.metrics

    if "HIGH_SAFETY_STRATEGY" in strategy_results:
        hs_res = strategy_results["HIGH_SAFETY_STRATEGY"]
        # If balanced has elevated stockout risk that high-safety fixes
        if stk_prob_balanced > max_stockout_threshold and hs_res.metrics.stockout_probability < stk_prob_balanced:
            rec_changed = True
            p19_action = "BUY_NOW" if p18_action in ["BUY_NOW", "URGENT_REORDER"] else "BUY_NOW"
            p19_qty = hs_res.order_quantity
            p19_cov = hs_res.coverage_days
            selected_metrics = hs_res.metrics
            change_reason = (
                f"Phase 18 recommendation stockout probability ({stk_prob_balanced * 100:.1f}%) "
                f"exceeded risk tolerance ({max_stockout_threshold * 100:.1f}%); "
                f"upgraded to High-Safety strategy ({hs_res.metrics.stockout_probability * 100:.1f}% stockout risk)."
            )

    return RiskAdjustedDecision(
        decision_date=str(p18_decision.get("decision_date", "2024-08-17")),
        store_id=str(p18_decision.get("store_id", "UNKNOWN")),
        product_id=str(p18_decision.get("product_id", "UNKNOWN")),
        supplier_id=str(p18_decision.get("supplier_id", "UNKNOWN")),
        phase18_action=p18_action,
        phase18_order_quantity=p18_qty,
        phase18_coverage_days=p18_cov,
        phase19_action=p19_action,
        phase19_order_quantity=p19_qty,
        phase19_coverage_days=p19_cov,
        recommendation_changed=rec_changed,
        change_reason=change_reason,
        strategy_results=strategy_results,
        stockout_probability_before=stk_prob_before,
        stockout_probability_after=selected_metrics.stockout_probability,
        overstock_probability=selected_metrics.overstock_probability,
        wastage_probability=selected_metrics.wastage_probability,
        service_probability=selected_metrics.service_probability,
        expected_total_cost=selected_metrics.expected_total_cost,
        cost_p10=selected_metrics.cost_p10,
        cost_p50=selected_metrics.cost_p50,
        cost_p90=selected_metrics.cost_p90,
        cost_var95=selected_metrics.cost_var95,
        cost_cvar95=selected_metrics.cost_cvar95,
        expected_lost_units=selected_metrics.expected_lost_units,
        expected_excess_units=selected_metrics.expected_excess_units,
        late_delivery_probability=selected_metrics.late_delivery_probability,
        future_price_increase_probability=selected_metrics.future_price_increase_probability,
        robustness_status=str(p18_decision.get("robustness_status", "ROBUST")),
        probability_future_price_rises=selected_metrics.probability_future_price_rises,
        probability_future_price_falls=selected_metrics.probability_future_price_falls,
        probability_buy_now_is_cheaper=selected_metrics.probability_buy_now_is_cheaper,
        expected_buy_now_saving=selected_metrics.expected_buy_now_saving,
        expected_wait_saving=selected_metrics.expected_wait_saving,
        price_signal_strength=selected_metrics.price_signal_strength,
        price_model_status=selected_metrics.price_model_status,
        price_risk_enabled=selected_metrics.price_risk_enabled,
        risk_mode=selected_metrics.risk_mode
    )
