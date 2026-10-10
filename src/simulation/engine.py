"""Vectorised Monte Carlo Inventory Simulation Engine.

Simulates thousands of plausible future realizations simultaneously across:
- Daily demand paths (30 days)
- Supplier delivery arrival days (lead times)
- Delivery fill rates (supplier reliability)
- Future supplier purchase prices

Tracks daily inventory depletion, stockouts, holding costs, perishable spoilage,
and computes expected costs, Value at Risk (VaR95), and Conditional Value at Risk (CVaR95).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.utils.paths import PROJECT_ROOT
from src.models.governance import price_model_usable_for_decision


@dataclass
class SimulationRiskMetrics:
    """Aggregated probabilistic risk metrics from Monte Carlo simulation."""
    stockout_probability: float
    overstock_probability: float
    wastage_probability: float
    service_probability: float
    expected_lost_units: float
    median_lost_units: float
    p90_lost_units: float
    expected_stockout_days: float
    prob_stockout_before_arrival: float
    expected_excess_units: float
    median_excess_units: float
    expected_wasted_units: float
    expected_wastage_cost: float
    expected_total_cost: float
    cost_p10: float
    cost_p50: float
    cost_p90: float
    cost_p95: float
    cost_var95: float
    cost_cvar95: float
    late_delivery_probability: float
    future_price_increase_probability: Optional[float] = None
    probability_future_price_rises: Optional[float] = None
    probability_future_price_falls: Optional[float] = None
    probability_buy_now_is_cheaper: Optional[float] = None
    expected_buy_now_saving: Optional[float] = None
    expected_wait_saving: Optional[float] = None
    price_signal_strength: str = "PRICE_MODEL_NOT_VALIDATED"
    price_model_status: str = "NOT_VALIDATED"
    price_risk_enabled: bool = False
    risk_mode: str = "DEMAND_SUPPLY_ONLY"
    n_scenarios: int = 1000
    mean_regret: float = 0.0
    p90_regret: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "stockout_probability": round(float(self.stockout_probability), 4),
            "overstock_probability": round(float(self.overstock_probability), 4),
            "wastage_probability": round(float(self.wastage_probability), 4),
            "service_probability": round(float(self.service_probability), 4),
            "expected_lost_units": round(float(self.expected_lost_units), 2),
            "median_lost_units": round(float(self.median_lost_units), 2),
            "p90_lost_units": round(float(self.p90_lost_units), 2),
            "expected_stockout_days": round(float(self.expected_stockout_days), 2),
            "prob_stockout_before_arrival": round(float(self.prob_stockout_before_arrival), 4),
            "expected_excess_units": round(float(self.expected_excess_units), 2),
            "median_excess_units": round(float(self.median_excess_units), 2),
            "expected_wasted_units": round(float(self.expected_wasted_units), 2),
            "expected_wastage_cost": round(float(self.expected_wastage_cost), 2),
            "expected_total_cost": round(float(self.expected_total_cost), 2),
            "cost_p10": round(float(self.cost_p10), 2),
            "cost_p50": round(float(self.cost_p50), 2),
            "cost_p90": round(float(self.cost_p90), 2),
            "cost_p95": round(float(self.cost_p95), 2),
            "cost_var95": round(float(self.cost_var95), 2),
            "cost_cvar95": round(float(self.cost_cvar95), 2),
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
            "n_scenarios": int(self.n_scenarios),
            "mean_regret": round(float(self.mean_regret), 2),
            "p90_regret": round(float(self.p90_regret), 2)
        }


def simulate_inventory_risk(
    order_quantity: float,
    coverage_days: int,
    current_inventory: float,
    incoming_inventory: float,
    current_purchase_price: float,
    daily_demand_paths: np.ndarray,
    future_prices: Optional[np.ndarray] = None,
    lead_times: Optional[np.ndarray] = None,
    fill_rates: Optional[np.ndarray] = None,
    shelf_life_days: Optional[float] = None,
    cost_parameters: Optional[Dict[str, float]] = None,
    is_delayed_purchase: bool = False,
    nominal_lead_time: float = 2.0,
    overstock_threshold_multiplier: float = 1.5,
    price_risk_enabled: Optional[bool] = None
) -> Tuple[SimulationRiskMetrics, np.ndarray]:
    """Execute vectorised Monte Carlo simulation for a specific order decision.

    Args:
        order_quantity: Units to order (>= 0).
        coverage_days: Intended planning horizon days (1..30).
        current_inventory: Initial on-hand inventory.
        incoming_inventory: Confirmed incoming pipeline inventory.
        current_purchase_price: Current supplier unit purchase price.
        daily_demand_paths: Array of shape (n_scenarios, 30).
        future_prices: Optional array of shape (n_scenarios,).
        lead_times: Array of shape (n_scenarios,) in integer days.
        fill_rates: Array of shape (n_scenarios,) in [0.0, 1.0].
        shelf_life_days: Maximum product shelf-life days (or None).
        cost_parameters: Unit cost parameter registry.
        is_delayed_purchase: True if buying in future at simulated future price (if price risk enabled).
        nominal_lead_time: Expected contract lead time for delay evaluation.
        overstock_threshold_multiplier: Excess inventory threshold relative to demand.
        price_risk_enabled: Whether stochastic price modeling is authorized by governance.

    Returns:
        Tuple of (SimulationRiskMetrics, scenario_total_costs array).
    """
    n_scenarios, days = daily_demand_paths.shape

    # Determine price governance status
    if price_risk_enabled is not None:
        price_enabled = bool(price_risk_enabled)
    else:
        try:
            price_enabled = price_model_usable_for_decision(root=PROJECT_ROOT)
        except Exception:
            price_enabled = False

    price_status_label = "VALIDATED" if price_enabled else "NOT_VALIDATED"
    risk_mode = "PRICE_DEMAND_SUPPLY" if price_enabled else "DEMAND_SUPPLY_ONLY"

    if lead_times is None:
        lead_times = np.full(n_scenarios, int(round(nominal_lead_time)), dtype=np.int32)
    if fill_rates is None:
        fill_rates = np.ones(n_scenarios, dtype=np.float32)

    costs = cost_parameters or {
        "inventory_holding_cost_per_unit_per_day": 0.25,
        "storage_cost_per_unit_per_day": 0.15,
        "stockout_cost_per_unit": 3.50,
        "wastage_cost_per_unit": 7.20,
        "ordering_cost_per_po": 50.00
    }
    c_hold_total = float(costs.get("inventory_holding_cost_per_unit_per_day", 0.25)) + float(costs.get("storage_cost_per_unit_per_day", 0.15))
    c_stockout = float(costs.get("stockout_cost_per_unit", 3.50))
    c_wastage = float(costs.get("wastage_cost_per_unit", 7.20))
    c_order = float(costs.get("ordering_cost_per_po", 50.00))

    # Actual received quantity per scenario
    received_quantity = np.round(order_quantity * fill_rates).astype(np.float32)

    # Tracking arrays across all scenarios
    current_stock = np.full(n_scenarios, current_inventory, dtype=np.float32)
    daily_holding_stock = np.zeros(n_scenarios, dtype=np.float32)
    lost_units = np.zeros(n_scenarios, dtype=np.float32)
    stockout_days = np.zeros(n_scenarios, dtype=np.float32)
    stockout_before_arrival = np.zeros(n_scenarios, dtype=bool)

    # Determine planning horizon for simulation evaluation
    if coverage_days is not None and coverage_days > 0:
        eval_horizon = min(days, int(coverage_days) + max(0, int(round(nominal_lead_time))))
    else:
        eval_horizon = days

    # Daily simulation step (vectorized over n_scenarios across planning horizon)
    for t in range(eval_horizon):
        day_num = t + 1
        d_t = daily_demand_paths[:, t]

        # 1. Incoming pipeline arrivals (assumed to arrive on Day 1)
        if day_num == 1 and incoming_inventory > 0:
            current_stock += incoming_inventory

        # 2. Reorder arrivals on day matching lead time
        arrived_mask = (lead_times == day_num)
        if np.any(arrived_mask):
            current_stock[arrived_mask] += received_quantity[arrived_mask]

        # If lead time is 0 (same day delivery), arrived on day 1
        if day_num == 1:
            same_day_mask = (lead_times <= 0)
            if np.any(same_day_mask):
                current_stock[same_day_mask] += received_quantity[same_day_mask]

        # 3. Serve demand
        unfulfilled = np.maximum(0.0, d_t - current_stock)
        has_stockout_today = unfulfilled > 0

        # Check if stockout occurred before reorder arrived
        not_yet_arrived = (lead_times > day_num)
        stockout_before_arrival |= (has_stockout_today & not_yet_arrived)

        lost_units += unfulfilled
        stockout_days += has_stockout_today.astype(np.float32)

        # Update end-of-day stock
        current_stock = np.maximum(0.0, current_stock - d_t)
        daily_holding_stock += current_stock

    # Perishable Wastage Evaluation
    wasted_units = np.zeros(n_scenarios, dtype=np.float32)
    if shelf_life_days is not None and shelf_life_days > 0 and shelf_life_days < eval_horizon:
        # Expected demand over shelf life period
        shelf_idx = int(min(eval_horizon - 1, max(0, round(shelf_life_days) - 1)))
        shelf_demand = np.sum(daily_demand_paths[:, :shelf_idx + 1], axis=1)
        total_inflow = current_inventory + incoming_inventory + received_quantity
        # Units surviving past shelf life that were not consumed
        potential_excess = total_inflow - shelf_demand
        wasted_units = np.maximum(0.0, potential_excess).astype(np.float32)

    # Overstock Evaluation
    # Excess inventory relative to expected remaining consumption
    expected_period_demand = np.sum(daily_demand_paths[:, :eval_horizon], axis=1)
    excess_units = np.maximum(0.0, current_stock - (expected_period_demand * (overstock_threshold_multiplier - 1.0)))
    has_overstock = current_stock > (expected_period_demand * 0.5)

    # Financial Cost Breakdown per scenario
    if order_quantity > 0:
        if is_delayed_purchase and price_enabled and future_prices is not None:
            purch_costs = received_quantity * future_prices
        else:
            # Current known price for immediate purchase or reference price when unvalidated
            purch_costs = np.full(n_scenarios, order_quantity * current_purchase_price, dtype=np.float32)
        ordering_costs = np.full(n_scenarios, c_order, dtype=np.float32)
    else:
        purch_costs = np.zeros(n_scenarios, dtype=np.float32)
        ordering_costs = np.zeros(n_scenarios, dtype=np.float32)

    holding_costs = daily_holding_stock * c_hold_total
    stockout_costs = lost_units * c_stockout
    wastage_costs = wasted_units * c_wastage

    total_scenario_costs = (
        purch_costs + ordering_costs + holding_costs + stockout_costs + wastage_costs
    ).astype(np.float32)

    # Statistical Aggregation
    has_stockout = (lost_units > 0)
    stockout_prob = float(np.mean(has_stockout))
    overstock_prob = float(np.mean(has_overstock))
    wastage_prob = float(np.mean(wasted_units > 0))
    service_prob = float(1.0 - stockout_prob)

    exp_lost = float(np.mean(lost_units))
    med_lost = float(np.median(lost_units[has_stockout])) if np.any(has_stockout) else 0.0
    p90_lost = float(np.percentile(lost_units, 90))

    exp_stockout_days = float(np.mean(stockout_days))
    prob_stk_before_arr = float(np.mean(stockout_before_arrival))

    exp_excess = float(np.mean(excess_units))
    med_excess = float(np.median(excess_units))
    exp_wasted = float(np.mean(wasted_units))
    exp_wastage_cost = float(np.mean(wastage_costs))

    exp_total_cost = float(np.mean(total_scenario_costs))
    cost_p10 = float(np.percentile(total_scenario_costs, 10))
    cost_p50 = float(np.percentile(total_scenario_costs, 50))
    cost_p90 = float(np.percentile(total_scenario_costs, 90))
    cost_p95 = float(np.percentile(total_scenario_costs, 95))

    cost_var95 = cost_p95
    worst_5pct_mask = total_scenario_costs >= cost_var95
    cost_cvar95 = float(np.mean(total_scenario_costs[worst_5pct_mask])) if np.any(worst_5pct_mask) else cost_var95

    late_deliv_prob = float(np.mean(lead_times > nominal_lead_time))

    # Price signal metrics: only compute if price_enabled is True and future_prices provided
    if price_enabled and future_prices is not None:
        price_inc_prob: Optional[float] = float(np.mean(future_prices > current_purchase_price))
        price_dec_prob: Optional[float] = float(np.mean(future_prices < current_purchase_price))
        buy_now_cheaper_prob: Optional[float] = float(np.mean(current_purchase_price < future_prices))
        exp_buy_now_saving: Optional[float] = float(np.mean(np.maximum(0.0, (future_prices - current_purchase_price) * order_quantity)))
        exp_wait_saving: Optional[float] = float(np.mean(np.maximum(0.0, (current_purchase_price - future_prices) * order_quantity)))

        mean_future_p = float(np.mean(future_prices))
        pct_p_change = ((mean_future_p / current_purchase_price) - 1.0) * 100.0

        if pct_p_change >= 5.0 and price_inc_prob >= 0.70:
            price_sig_str = "STRONG_BUY_EARLY_SIGNAL"
        elif pct_p_change >= 2.0 and price_inc_prob >= 0.60:
            price_sig_str = "MODERATE_BUY_EARLY_SIGNAL"
        elif pct_p_change <= -5.0 and price_dec_prob >= 0.70:
            price_sig_str = "STRONG_WAIT_SIGNAL"
        elif pct_p_change <= -2.0 and price_dec_prob >= 0.60:
            price_sig_str = "MODERATE_WAIT_SIGNAL"
        else:
            price_sig_str = "NEUTRAL_PRICE_SIGNAL"
    else:
        # Price model is quarantined / not validated:
        # Do NOT fabricate metrics. Explicitly set to None / "PRICE_MODEL_NOT_VALIDATED"
        price_inc_prob = None
        price_dec_prob = None
        buy_now_cheaper_prob = None
        exp_buy_now_saving = None
        exp_wait_saving = None
        price_sig_str = "PRICE_MODEL_NOT_VALIDATED"

    metrics = SimulationRiskMetrics(
        stockout_probability=stockout_prob,
        overstock_probability=overstock_prob,
        wastage_probability=wastage_prob,
        service_probability=service_prob,
        expected_lost_units=exp_lost,
        median_lost_units=med_lost,
        p90_lost_units=p90_lost,
        expected_stockout_days=exp_stockout_days,
        prob_stockout_before_arrival=prob_stk_before_arr,
        expected_excess_units=exp_excess,
        median_excess_units=med_excess,
        expected_wasted_units=exp_wasted,
        expected_wastage_cost=exp_wastage_cost,
        expected_total_cost=exp_total_cost,
        cost_p10=cost_p10,
        cost_p50=cost_p50,
        cost_p90=cost_p90,
        cost_p95=cost_p95,
        cost_var95=cost_var95,
        cost_cvar95=cost_cvar95,
        late_delivery_probability=late_deliv_prob,
        future_price_increase_probability=price_inc_prob,
        probability_future_price_rises=price_inc_prob,
        probability_future_price_falls=price_dec_prob,
        probability_buy_now_is_cheaper=buy_now_cheaper_prob,
        expected_buy_now_saving=exp_buy_now_saving,
        expected_wait_saving=exp_wait_saving,
        price_signal_strength=price_sig_str,
        price_model_status=price_status_label,
        price_risk_enabled=price_enabled,
        risk_mode=risk_mode,
        n_scenarios=n_scenarios
    )

    return metrics, total_scenario_costs
