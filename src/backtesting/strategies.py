"""Inventory strategy definitions for Phase 20 Historical Business Backtesting."""
from typing import Dict, Any, Optional
import math
import numpy as np

from src.inventory.optimizer import optimise_inventory_decision
from src.simulation.distributions import (
    sample_daily_demand_paths,
    sample_future_purchase_prices,
    sample_supplier_lead_times,
    sample_supplier_fill_rates
)
from src.simulation.risk_adjuster import evaluate_and_adjust_risk


def round_up_to_constraints(raw_qty: float, moq: float, pack_size: int) -> float:
    """Round order quantity up to supplier MOQ and pack size multiples."""
    if raw_qty <= 0:
        return 0.0
    qty = max(float(raw_qty), float(moq))
    if pack_size > 1:
        qty = math.ceil(qty / float(pack_size)) * float(pack_size)
    else:
        qty = math.ceil(qty)
    return float(qty)


def decide_no_action() -> Dict[str, Any]:
    """Strategy A: Do nothing, place 0 order."""
    return {
        "strategy": "NO_ACTION",
        "order_quantity": 0.0,
        "coverage_days": 0,
        "action_label": "DO_NOT_REORDER"
    }


def decide_fixed_coverage(
    target_days: int,
    current_inventory: float,
    demand_forecasts: Dict[int, float],
    moq: float,
    pack_size: int,
    safety_margin: float = 0.15
) -> Dict[str, Any]:
    """Strategies B, C, D: Fixed target days demand coverage."""
    # Lookup demand forecast for target_days, or interpolate
    d_target = float(demand_forecasts.get(target_days, demand_forecasts.get(14, 100.0)))
    target_inventory = d_target * (1.0 + safety_margin)
    raw_qty = max(0.0, target_inventory - current_inventory)
    order_qty = round_up_to_constraints(raw_qty, moq, pack_size)

    return {
        "strategy": f"FIXED_{target_days}_DAY",
        "order_quantity": order_qty,
        "coverage_days": target_days,
        "action_label": "BUY_NOW" if order_qty > 0 else "DO_NOT_REORDER"
    }


def decide_reorder_point(
    current_inventory: float,
    mean_daily_demand: float,
    lead_time_days: float,
    moq: float,
    pack_size: int,
    cycle_days: int = 14,
    z_service: float = 1.645
) -> Dict[str, Any]:
    """Strategy E: Conventional (s, S) Reorder Point policy."""
    lead_time_demand = mean_daily_demand * lead_time_days
    safety_stock = z_service * math.sqrt(max(1.0, lead_time_days)) * max(1.0, mean_daily_demand * 0.25)
    reorder_point = lead_time_demand + safety_stock
    order_up_to = reorder_point + (mean_daily_demand * cycle_days)

    if current_inventory <= reorder_point:
        raw_qty = max(moq, order_up_to - current_inventory)
        order_qty = round_up_to_constraints(raw_qty, moq, pack_size)
        action = "BUY_NOW"
    else:
        order_qty = 0.0
        action = "DO_NOT_REORDER"

    return {
        "strategy": "REORDER_POINT",
        "order_quantity": order_qty,
        "coverage_days": cycle_days if order_qty > 0 else 0,
        "action_label": action
    }


def decide_phase18_optimiser(
    demand_forecasts: Dict[int, float],
    price_forecasts: Dict[int, float],
    demand_p10: Dict[int, float],
    demand_p50: Dict[int, float],
    demand_p90: Dict[int, float],
    price_p10: Dict[int, float],
    price_p50: Dict[int, float],
    price_p90: Dict[int, float],
    current_inventory: float,
    current_purchase_price: float,
    moq: float,
    pack_size: int,
    shelf_life_days: Optional[float],
    lead_time_days: float,
    store_id: str,
    product_id: str,
    decision_date: str
) -> Dict[str, Any]:
    """Strategy F: Phase 18 Expected-Cost Optimiser alone."""
    p18_dec = optimise_inventory_decision(
        demand_forecasts=demand_forecasts,
        price_forecasts=price_forecasts,
        demand_p10=demand_p10,
        demand_p50=demand_p50,
        demand_p90=demand_p90,
        price_p10=price_p10,
        price_p50=price_p50,
        price_p90=price_p90,
        current_inventory=current_inventory,
        current_purchase_price=current_purchase_price,
        supplier_minimum_order_quantity=moq,
        product_case_size=pack_size,
        shelf_life_days=shelf_life_days,
        lead_time_days=lead_time_days,
        store_id=store_id,
        product_id=product_id,
        decision_date=decision_date
    )
    return {
        "strategy": "PHASE18_OPTIMISER",
        "order_quantity": float(p18_dec.recommended_order_quantity),
        "coverage_days": int(p18_dec.recommended_stock_coverage_days),
        "action_label": p18_dec.recommended_action,
        "raw_p18_decision": p18_dec.to_dict()
    }


def decide_phase20_ai_strategy(
    demand_forecasts: Dict[int, float],
    price_forecasts: Dict[int, float],
    demand_p10: Dict[int, float],
    demand_p50: Dict[int, float],
    demand_p90: Dict[int, float],
    price_p10: Dict[int, float],
    price_p50: Dict[int, float],
    price_p90: Dict[int, float],
    current_inventory: float,
    current_purchase_price: float,
    moq: float,
    pack_size: int,
    shelf_life_days: Optional[float],
    lead_time_days: float,
    store_id: str,
    product_id: str,
    decision_date: str,
    n_scenarios: int = 1000
) -> Dict[str, Any]:
    """Strategy G: Full Phase 19 Risk-Adjusted AI Strategy."""
    # 1. Base Phase 18 decision
    p18_dec = optimise_inventory_decision(
        demand_forecasts=demand_forecasts,
        price_forecasts=price_forecasts,
        demand_p10=demand_p10,
        demand_p50=demand_p50,
        demand_p90=demand_p90,
        price_p10=price_p10,
        price_p50=price_p50,
        price_p90=price_p90,
        current_inventory=current_inventory,
        current_purchase_price=current_purchase_price,
        supplier_minimum_order_quantity=moq,
        product_case_size=pack_size,
        shelf_life_days=shelf_life_days,
        lead_time_days=lead_time_days,
        store_id=store_id,
        product_id=product_id,
        decision_date=decision_date
    )
    p18_dict = p18_dec.to_dict()

    # 2. Monte Carlo Risk Assessment
    rng = np.random.default_rng(42)
    paths = sample_daily_demand_paths(
        demand_horizons=demand_forecasts,
        demand_p10=demand_p10,
        demand_p50=demand_p50,
        demand_p90=demand_p90,
        n_scenarios=n_scenarios,
        rng=rng
    )
    prices = sample_future_purchase_prices(
        current_price=current_purchase_price,
        price_horizons=price_forecasts,
        price_p10=price_p10,
        price_p50=price_p50,
        price_p90=price_p90,
        n_scenarios=n_scenarios,
        rng=rng
    )
    lts = sample_supplier_lead_times(
        nominal_lead_time=lead_time_days,
        n_scenarios=n_scenarios,
        rng=rng
    )
    fills = sample_supplier_fill_rates(
        n_scenarios=n_scenarios,
        rng=rng
    )

    adj_dec = evaluate_and_adjust_risk(
        p18_decision=p18_dict,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        shelf_life_days=shelf_life_days,
        nominal_lead_time=lead_time_days
    )

    return {
        "strategy": "PHASE20_AI_STRATEGY",
        "order_quantity": float(adj_dec.phase19_order_quantity),
        "coverage_days": int(adj_dec.phase19_coverage_days),
        "action_label": adj_dec.phase19_action,
        "recommendation_changed": adj_dec.recommendation_changed,
        "change_reason": adj_dec.change_reason,
        "stockout_prob_simulated": float(adj_dec.stockout_probability_after),
        "cost_cvar95_simulated": float(adj_dec.cost_cvar95)
    }
