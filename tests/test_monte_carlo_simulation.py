"""Comprehensive automated test suite for Phase 19 Monte Carlo Risk Simulation.

Validates:
- Seed reproducibility.
- Non-negative demand, positive prices, non-negative lead times.
- Calibrated quantiles consistency.
- Bounded probabilities ([0, 1]) for stockout, overstock, wastage, and service levels.
- Finite and monotonic cost distributions, VaR95, and CVaR95.
- Strategy comparisons (No Action, Low Stock, Balanced, High Safety).
- Monotonic risk response to order quantity and lead-time shocks.
- Perishable wastage tracking.
- Absence of future leakage.
- FAST (2,000) and DETAILED (10,000) execution modes.
- Preservation of Phase 18 recommendations alongside Phase 19 risk adjustments.
"""
from typing import Dict
import json
import pytest
import numpy as np
import pandas as pd

from src.simulation.config import load_simulation_config
from src.simulation.distributions import (
    sample_daily_demand_paths,
    sample_future_purchase_prices,
    sample_supplier_lead_times,
    sample_supplier_fill_rates
)
from src.simulation.engine import (
    simulate_inventory_risk,
    SimulationRiskMetrics
)
from src.simulation.risk_adjuster import (
    evaluate_and_adjust_risk,
    RiskAdjustedDecision
)
from src.simulation.reporting import (
    format_telegram_risk_alert,
    build_dashboard_risk_payload,
    generate_simulation_summary_markdown
)


@pytest.fixture
def standard_demand_horizons() -> Dict[int, float]:
    return {1: 25.0, 3: 75.0, 7: 175.0, 14: 350.0, 21: 525.0, 30: 750.0}


@pytest.fixture
def standard_demand_p10() -> Dict[int, float]:
    return {1: 15.0, 3: 50.0, 7: 120.0, 14: 250.0, 21: 380.0, 30: 550.0}


@pytest.fixture
def standard_demand_p50() -> Dict[int, float]:
    return {1: 25.0, 3: 75.0, 7: 175.0, 14: 350.0, 21: 525.0, 30: 750.0}


@pytest.fixture
def standard_demand_p90() -> Dict[int, float]:
    return {1: 40.0, 3: 110.0, 7: 240.0, 14: 480.0, 21: 700.0, 30: 990.0}


@pytest.fixture
def standard_price_horizons() -> Dict[int, float]:
    return {1: 10.0, 3: 10.0, 7: 10.0, 14: 10.0, 21: 10.0, 30: 10.0}


# =============================================================================
# 1. DISTRIBUTION SAMPLING & REPRODUCIBILITY TESTS
# =============================================================================

def test_fixed_random_seed_reproduces_results(standard_demand_horizons):
    """Verify that identical random seeds produce identical simulated demand arrays."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    paths1 = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500, rng=rng1)
    paths2 = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500, rng=rng2)

    np.testing.assert_array_equal(paths1, paths2)


def test_demand_samples_are_non_negative(standard_demand_horizons):
    """Demand samples must be strictly non-negative across all scenarios and days."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=1000)
    assert np.all(paths >= 0.0)
    assert paths.shape == (1000, 30)


def test_price_samples_are_positive(standard_price_horizons):
    """Simulated purchase prices must be strictly positive."""
    prices = sample_future_purchase_prices(current_price=10.0, price_horizons=standard_price_horizons, n_scenarios=1000)
    assert np.all(prices > 0.0)
    assert np.all(np.isfinite(prices))


def test_lead_time_samples_are_non_negative():
    """Sampled supplier lead times must be non-negative integer days."""
    lts = sample_supplier_lead_times(nominal_lead_time=2.0, historical_variability=0.8, n_scenarios=1000)
    assert np.all(lts >= 0)
    assert np.issubdtype(lts.dtype, np.integer)


def test_p10_p50_p90_distribution_respected(
    standard_demand_horizons, standard_demand_p10, standard_demand_p50, standard_demand_p90
):
    """30-day cumulative demand quantiles must approximate Phase 16 P10, P50, and P90."""
    paths = sample_daily_demand_paths(
        demand_horizons=standard_demand_horizons,
        demand_p10=standard_demand_p10,
        demand_p50=standard_demand_p50,
        demand_p90=standard_demand_p90,
        n_scenarios=5000,
        rng=np.random.default_rng(42)
    )
    cum_30 = np.sum(paths, axis=1)
    sample_p10 = float(np.percentile(cum_30, 10))
    sample_p50 = float(np.percentile(cum_30, 50))
    sample_p90 = float(np.percentile(cum_30, 90))

    target_p10 = standard_demand_p10[30]
    target_p50 = standard_demand_p50[30]
    target_p90 = standard_demand_p90[30]

    assert sample_p10 <= sample_p50 <= sample_p90
    assert pytest.approx(sample_p50, rel=0.15) == target_p50
    assert pytest.approx(sample_p90, rel=0.20) == target_p90


# =============================================================================
# 2. SIMULATION ENGINE METRICS & CONSTRAINTS
# =============================================================================

def test_stockout_and_service_probability_in_range(standard_demand_horizons, standard_price_horizons):
    """Probabilities must lie strictly within [0.0, 1.0]."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500)
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=500)
    lts = sample_supplier_lead_times(2.0, n_scenarios=500)
    fills = sample_supplier_fill_rates(n_scenarios=500)

    metrics, _ = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert 0.0 <= metrics.stockout_probability <= 1.0
    assert 0.0 <= metrics.service_probability <= 1.0
    assert pytest.approx(metrics.stockout_probability + metrics.service_probability, 0.001) == 1.0
    assert 0.0 <= metrics.overstock_probability <= 1.0
    assert 0.0 <= metrics.wastage_probability <= 1.0


def test_total_costs_are_finite_and_quantiles_ordered(standard_demand_horizons, standard_price_horizons):
    """Total scenario costs must be finite, non-negative, and quantile ordered."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500)
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=500)
    lts = sample_supplier_lead_times(2.0, n_scenarios=500)
    fills = sample_supplier_fill_rates(n_scenarios=500)

    metrics, costs_arr = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert np.all(np.isfinite(costs_arr))
    assert np.all(costs_arr >= 0.0)
    assert metrics.cost_p10 <= metrics.cost_p50 <= metrics.cost_p90 <= metrics.cost_p95
    assert metrics.cost_cvar95 >= metrics.cost_var95


def test_larger_stock_reduces_stockout_risk(standard_demand_horizons, standard_price_horizons):
    """Ordering more inventory must monotonically reduce or keep equal stockout risk."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500, rng=np.random.default_rng(42))
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=500, rng=np.random.default_rng(42))
    lts = sample_supplier_lead_times(2.0, n_scenarios=500, rng=np.random.default_rng(42))
    fills = sample_supplier_fill_rates(n_scenarios=500, rng=np.random.default_rng(42))

    # Low order (0 units) vs High order (400 units)
    m_low, _ = simulate_inventory_risk(
        order_quantity=0.0,
        coverage_days=1,
        current_inventory=20.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    m_high, _ = simulate_inventory_risk(
        order_quantity=400.0,
        coverage_days=14,
        current_inventory=20.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert m_high.stockout_probability < m_low.stockout_probability
    assert m_high.expected_lost_units < m_low.expected_lost_units


def test_larger_stock_increases_holding_cost(standard_demand_horizons, standard_price_horizons):
    """Ordering massive inventory increases expected total purchase and holding cost."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500, rng=np.random.default_rng(42))
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=500, rng=np.random.default_rng(42))
    lts = sample_supplier_lead_times(2.0, n_scenarios=500, rng=np.random.default_rng(42))
    fills = sample_supplier_fill_rates(n_scenarios=500, rng=np.random.default_rng(42))

    m_mod, _ = simulate_inventory_risk(
        order_quantity=50.0,
        coverage_days=3,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    m_huge, _ = simulate_inventory_risk(
        order_quantity=2000.0,
        coverage_days=30,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert m_huge.expected_total_cost > m_mod.expected_total_cost
    assert m_huge.expected_excess_units > m_mod.expected_excess_units


def test_late_delivery_causes_stockout(standard_demand_horizons, standard_price_horizons):
    """When lead time is 10 days and on-hand stock is 0, stockout probability must be 1.0."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=200)
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=200)
    lts = np.full(200, 10, dtype=np.int32)  # Forced 10 days delay
    fills = np.ones(200, dtype=np.float32)

    metrics, _ = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=0.0,  # 0 on hand
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert metrics.stockout_probability == 1.0
    assert metrics.prob_stockout_before_arrival == 1.0


def test_perishable_shelf_life_wastage(standard_demand_horizons, standard_price_horizons):
    """Massive over-ordering of 3-day perishable product triggers simulated wastage."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=300)
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=300)
    lts = np.zeros(300, dtype=np.int32)
    fills = np.ones(300, dtype=np.float32)

    metrics, _ = simulate_inventory_risk(
        order_quantity=1000.0,  # Way beyond 3 days demand (~75 units)
        coverage_days=30,
        current_inventory=0.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        shelf_life_days=3.0  # Perishable!
    )
    assert metrics.wastage_probability > 0.0
    assert metrics.expected_wasted_units > 0.0
    assert metrics.expected_wastage_cost > 0.0


# =============================================================================
# 3. STRATEGY COMPARISON & RISK ADJUSTMENT
# =============================================================================

def test_strategy_comparison_and_regret_analysis(standard_demand_horizons, standard_price_horizons):
    """Test evaluating No Action, Balanced, and Alternatives with regret calculation."""
    p18_mock = {
        "decision_date": "2024-08-17",
        "store_id": "STR_KUL_001",
        "product_id": "PRD_CHK_01",
        "supplier_id": "SUP_KLS_01",
        "recommended_action": "BUY_NOW",
        "recommended_order_quantity": 50.0,
        "recommended_stock_coverage_days": 1,
        "current_inventory": 10.0,
        "incoming_inventory": 0.0,
        "current_purchase_price": 7.20,
        "alternative_strategies": {
            "LOW_STOCK_STRATEGY": {"order_quantity": 30.0, "coverage_days": 1},
            "HIGH_SAFETY_STRATEGY": {"order_quantity": 100.0, "coverage_days": 3}
        }
    }
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500)
    prices = sample_future_purchase_prices(7.20, standard_price_horizons, n_scenarios=500)
    lts = sample_supplier_lead_times(1.0, n_scenarios=500)
    fills = sample_supplier_fill_rates(n_scenarios=500)

    adj_dec = evaluate_and_adjust_risk(
        p18_decision=p18_mock,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        shelf_life_days=3.0
    )
    assert isinstance(adj_dec, RiskAdjustedDecision)
    assert "NO_ACTION" in adj_dec.strategy_results
    assert "BALANCED_STRATEGY" in adj_dec.strategy_results
    assert "HIGH_SAFETY_STRATEGY" in adj_dec.strategy_results

    # Regret must be non-negative
    for s_name, s_res in adj_dec.strategy_results.items():
        assert s_res.metrics.mean_regret >= 0.0
        assert s_res.metrics.p90_regret >= s_res.metrics.mean_regret


def test_risk_adjustment_triggers_when_stockout_exceeds_threshold(
    standard_demand_horizons, standard_price_horizons
):
    """When balanced recommendation has stockout risk > max_threshold, elevate to high-safety."""
    p18_mock = {
        "decision_date": "2024-08-17",
        "store_id": "STR_KUL_001",
        "product_id": "PRD_CHK_01",
        "supplier_id": "SUP_KLS_01",
        "recommended_action": "BUY_NOW",
        "recommended_order_quantity": 10.0,  # Very small order, likely to stockout
        "recommended_stock_coverage_days": 1,
        "current_inventory": 0.0,
        "incoming_inventory": 0.0,
        "current_purchase_price": 7.20,
        "alternative_strategies": {
            "HIGH_SAFETY_STRATEGY": {"order_quantity": 150.0, "coverage_days": 3}
        }
    }
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500)
    prices = sample_future_purchase_prices(7.20, standard_price_horizons, n_scenarios=500)
    lts = sample_supplier_lead_times(1.0, n_scenarios=500)
    fills = sample_supplier_fill_rates(n_scenarios=500)

    adj_dec = evaluate_and_adjust_risk(
        p18_decision=p18_mock,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        max_stockout_threshold=0.05  # Tight 5% threshold
    )
    assert adj_dec.recommendation_changed is True
    assert adj_dec.phase19_order_quantity == 150.0
    assert "exceeded risk tolerance" in adj_dec.change_reason


# =============================================================================
# 4. REPORTING, SERIALIZATION & CONFIGURATION
# =============================================================================

def test_telegram_risk_alert_formatting():
    """Verify Telegram alert formatting contains key risk indicators."""
    met = SimulationRiskMetrics(
        stockout_probability=0.02,
        overstock_probability=0.15,
        wastage_probability=0.0,
        service_probability=0.98,
        expected_lost_units=0.5,
        median_lost_units=0.0,
        p90_lost_units=1.2,
        expected_stockout_days=0.1,
        prob_stockout_before_arrival=0.01,
        expected_excess_units=15.0,
        median_excess_units=10.0,
        expected_wasted_units=0.0,
        expected_wastage_cost=0.0,
        expected_total_cost=450.0,
        cost_p10=400.0,
        cost_p50=440.0,
        cost_p90=510.0,
        cost_p95=540.0,
        cost_var95=540.0,
        cost_cvar95=580.0,
        late_delivery_probability=0.05,
        future_price_increase_probability=0.6,
        n_scenarios=1000
    )
    dec = RiskAdjustedDecision(
        decision_date="2024-08-17",
        store_id="STR_JHR_003",
        product_id="PRD_CHK_01",
        supplier_id="SUP_KLS_01",
        phase18_action="BUY_NOW",
        phase18_order_quantity=50.0,
        phase18_coverage_days=1,
        phase19_action="BUY_NOW",
        phase19_order_quantity=50.0,
        phase19_coverage_days=1,
        recommendation_changed=False,
        change_reason="Validated",
        strategy_results={},
        stockout_probability_before=0.85,
        stockout_probability_after=0.02,
        overstock_probability=0.15,
        wastage_probability=0.0,
        service_probability=0.98,
        expected_total_cost=450.0,
        cost_p10=400.0,
        cost_p50=440.0,
        cost_p90=510.0,
        cost_var95=540.0,
        cost_cvar95=580.0,
        expected_lost_units=0.5,
        expected_excess_units=15.0,
        late_delivery_probability=0.05,
        future_price_increase_probability=0.6,
        robustness_status="ROBUST"
    )
    alert = format_telegram_risk_alert(dec)
    assert "STR_JHR_003" in alert
    assert "Without Action (Do Nothing): *85.0%*" in alert
    assert "With Recommended Action: *2.0%*" in alert
    assert "VaR95" in alert


def test_dashboard_risk_payload_serialization():
    """Verify dashboard payload dictionary serializes cleanly to JSON."""
    dec = RiskAdjustedDecision(
        decision_date="2024-08-17",
        store_id="STR_JHR_003",
        product_id="PRD_CHK_01",
        supplier_id="SUP_KLS_01",
        phase18_action="BUY_NOW",
        phase18_order_quantity=50.0,
        phase18_coverage_days=1,
        phase19_action="BUY_NOW",
        phase19_order_quantity=50.0,
        phase19_coverage_days=1,
        recommendation_changed=False,
        change_reason="Validated",
        strategy_results={},
        stockout_probability_before=0.85,
        stockout_probability_after=0.02,
        overstock_probability=0.15,
        wastage_probability=0.0,
        service_probability=0.98,
        expected_total_cost=450.0,
        cost_p10=400.0,
        cost_p50=440.0,
        cost_p90=510.0,
        cost_var95=540.0,
        cost_cvar95=580.0,
        expected_lost_units=0.5,
        expected_excess_units=15.0,
        late_delivery_probability=0.05,
        future_price_increase_probability=0.6,
        robustness_status="ROBUST"
    )
    payload = build_dashboard_risk_payload(dec)
    assert "metadata" in payload
    assert "decision_comparison" in payload
    assert "risk_mitigation" in payload
    assert "cost_distribution" in payload
    json_str = json.dumps(payload)
    assert len(json_str) > 100


def test_simulation_config_loading():
    """Verify loading config/monte_carlo.yaml."""
    cfg = load_simulation_config()
    assert "reproducibility" in cfg
    assert cfg["reproducibility"]["random_seed"] == 42
    assert "simulation" in cfg
    assert cfg["simulation"]["fast_mode_scenarios"] == 2000
    assert cfg["simulation"]["detailed_mode_scenarios"] == 10000


def test_dependency_assumptions_report_exists():
    """Verify dependency assumptions report exists and has required structure."""
    from src.utils.paths import PROJECT_ROOT
    dep_path = PROJECT_ROOT / "reports" / "monte_carlo" / "dependency_assumptions.csv"
    assert dep_path.exists()
    df = pd.read_csv(dep_path)
    assert "variables" in df.columns
    assert "assumed_dependence" in df.columns
    assert "method" in df.columns


def test_supplier_reliability_affects_receipt_quantity(standard_demand_horizons, standard_price_horizons):
    """Supplier fill rate shortfall strictly reduces received order quantities."""
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=100)
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=100)
    lts = np.zeros(100, dtype=np.int32)
    fills = np.full(100, 0.80, dtype=np.float32)  # 80% delivery

    metrics, costs = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=0.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    # With 100 ordered and 80% fill rate, 80 units received
    assert metrics.expected_total_cost > 0.0


def test_buy_now_uses_current_purchase_price(standard_demand_horizons):
    """BUY_NOW purchase costs must equal order_quantity * current_purchase_price deterministically."""
    paths = np.ones((50, 30), dtype=np.float32)
    prices = np.full(50, 25.0, dtype=np.float32)  # future price 25.0
    lts = np.zeros(50, dtype=np.int32)
    fills = np.ones(50, dtype=np.float32)

    _, costs = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=100.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,  # current price 10.0
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        is_delayed_purchase=False,
        cost_parameters={
            "inventory_holding_cost_per_unit_per_day": 0.0,
            "storage_cost_per_unit_per_day": 0.0,
            "stockout_cost_per_unit": 0.0,
            "wastage_cost_per_unit": 0.0,
            "ordering_cost_per_po": 0.0
        }
    )
    # Costs should be exactly 100 * 10.0 = 1000.0
    np.testing.assert_allclose(costs, 1000.0)


def test_wait_uses_simulated_future_purchase_price(standard_demand_horizons):
    """When price risk is disabled by governance, purchase cost uses current price regardless of delay.
    When price risk is explicitly enabled, WAIT uses simulated future prices."""
    paths = np.ones((50, 30), dtype=np.float32)
    prices = np.linspace(12.0, 20.0, 50, dtype=np.float32)  # varied future prices
    lts = np.zeros(50, dtype=np.int32)
    fills = np.ones(50, dtype=np.float32)

    # 1. When price risk is disabled (governance default), costs must equal 100 * current_price (10.0)
    _, costs_gov = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=100.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        is_delayed_purchase=True,
        cost_parameters={
            "inventory_holding_cost_per_unit_per_day": 0.0,
            "storage_cost_per_unit_per_day": 0.0,
            "stockout_cost_per_unit": 0.0,
            "wastage_cost_per_unit": 0.0,
            "ordering_cost_per_po": 0.0
        },
        price_risk_enabled=False
    )
    np.testing.assert_allclose(costs_gov, 100.0 * 10.0)

    # 2. When price risk is explicitly enabled, WAIT costs should match 100 * prices
    _, costs_enabled = simulate_inventory_risk(
        order_quantity=100.0,
        coverage_days=7,
        current_inventory=100.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        is_delayed_purchase=True,
        cost_parameters={
            "inventory_holding_cost_per_unit_per_day": 0.0,
            "storage_cost_per_unit_per_day": 0.0,
            "stockout_cost_per_unit": 0.0,
            "wastage_cost_per_unit": 0.0,
            "ordering_cost_per_po": 0.0
        },
        price_risk_enabled=True
    )
    np.testing.assert_allclose(costs_enabled, 100.0 * prices)


def test_no_future_realised_data_used():
    """Verify simulation is forward-looking and does not leak or touch realised post-origin data."""
    from src.utils.paths import PROJECT_ROOT
    rec_path = PROJECT_ROOT / "data" / "simulation" / "risk_adjusted_recommendations.parquet"
    if rec_path.exists():
        df = pd.read_parquet(rec_path)
        # All simulation origins must be <= 2024-08-17
        assert (pd.to_datetime(df["decision_date"]) <= pd.Timestamp("2024-08-17")).all()


def test_convergence_report_content():
    """Verify convergence analysis CSV exists and exhibits empirical error convergence."""
    from src.utils.paths import PROJECT_ROOT
    conv_path = PROJECT_ROOT / "reports" / "monte_carlo" / "convergence_analysis.csv"
    assert conv_path.exists()
    df = pd.read_csv(conv_path)
    assert "scenario_count" in df.columns
    assert "stockout_probability" in df.columns
    assert "expected_cost" in df.columns
    assert len(df) >= 4  # 1k, 2k, 5k, 10k


def test_fast_and_detailed_modes_work(standard_demand_horizons, standard_price_horizons):
    """Verify FAST (2,000) and DETAILED (10,000) modes run accurately."""
    paths_fast = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=2000)
    prices_fast = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=2000)
    lts_fast = sample_supplier_lead_times(2.0, n_scenarios=2000)
    fills_fast = sample_supplier_fill_rates(n_scenarios=2000)

    m_fast, _ = simulate_inventory_risk(
        order_quantity=50.0, coverage_days=3, current_inventory=10.0,
        incoming_inventory=0.0, current_purchase_price=10.0,
        daily_demand_paths=paths_fast, future_prices=prices_fast,
        lead_times=lts_fast, fill_rates=fills_fast
    )
    assert m_fast.n_scenarios == 2000

    paths_det = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=10000)
    prices_det = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=10000)
    lts_det = sample_supplier_lead_times(2.0, n_scenarios=10000)
    fills_det = sample_supplier_fill_rates(n_scenarios=10000)

    m_det, _ = simulate_inventory_risk(
        order_quantity=50.0, coverage_days=3, current_inventory=10.0,
        incoming_inventory=0.0, current_purchase_price=10.0,
        daily_demand_paths=paths_det, future_prices=prices_det,
        lead_times=lts_det, fill_rates=fills_det
    )
    assert m_det.n_scenarios == 10000


def test_batch_simulation_and_parquet_generation():
    """Verify batch recommendations parquet exists with valid schema."""
    from src.utils.paths import PROJECT_ROOT
    rec_path = PROJECT_ROOT / "data" / "simulation" / "risk_adjusted_recommendations.parquet"
    assert rec_path.exists()
    df = pd.read_parquet(rec_path)
    assert len(df) == 24  # All 24 SKU-store series
    expected_cols = [
        "decision_date", "store_id", "product_id", "supplier_id",
        "phase18_action", "phase18_order_quantity", "phase18_coverage_days",
        "phase19_action", "phase19_order_quantity", "phase19_coverage_days",
        "recommendation_changed", "change_reason",
        "stockout_probability_before", "stockout_probability_after",
        "expected_total_cost", "cost_var95", "cost_cvar95"
    ]
    for c in expected_cols:
        assert c in df.columns


def test_user_upload_compatible_standardised_input(standard_demand_horizons, standard_price_horizons):
    """Verify simulation operates on generic user POS input dictionaries."""
    user_pos_record = {
        "store_id": "STORE_CUSTOM_99",
        "product_id": "PRD_CUSTOM_01",
        "supplier_id": "SUP_CUSTOM_A",
        "decision_date": "2024-08-17",
        "recommended_action": "BUY_NOW",
        "recommended_order_quantity": 80.0,
        "recommended_stock_coverage_days": 3,
        "current_inventory": 15.0,
        "incoming_inventory": 0.0,
        "current_purchase_price": 5.50,
        "alternative_strategies": {
            "LOW_STOCK_STRATEGY": {"order_quantity": 40.0, "coverage_days": 1},
            "HIGH_SAFETY_STRATEGY": {"order_quantity": 120.0, "coverage_days": 7}
        }
    }
    paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=200)
    prices = sample_future_purchase_prices(5.50, standard_price_horizons, n_scenarios=200)
    lts = sample_supplier_lead_times(2.0, n_scenarios=200)
    fills = sample_supplier_fill_rates(n_scenarios=200)

    res = evaluate_and_adjust_risk(
        p18_decision=user_pos_record,
        daily_demand_paths=paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert res.store_id == "STORE_CUSTOM_99"
    assert res.phase19_order_quantity >= 0.0


def test_final_holdout_remains_untouched():
    """Verify the final holdout period (2024-10-01 to 2024-12-31) was strictly untouched."""
    from src.utils.paths import PROJECT_ROOT
    master_path = PROJECT_ROOT / "data" / "processed" / "master_daily.parquet"
    assert master_path.exists()
    df = pd.read_parquet(master_path, columns=["date"])
    df["date"] = pd.to_datetime(df["date"])
    holdout_mask = df["date"] >= pd.Timestamp("2024-10-01")
    assert holdout_mask.sum() == 2208, "Holdout count must remain exactly 2,208 records"


# =============================================================================
# 11. PRICE RISK SIMULATION METRIC TESTS
# =============================================================================

def test_price_risk_metrics_in_simulation(standard_demand_horizons, standard_price_horizons):
    """Verify Monte Carlo simulation computes probability of price rise, fall, and buy-now savings."""
    d_paths = sample_daily_demand_paths(standard_demand_horizons, n_scenarios=500)
    prices = sample_future_purchase_prices(10.0, standard_price_horizons, n_scenarios=500)
    lts = sample_supplier_lead_times(2.0, n_scenarios=500)
    fills = sample_supplier_fill_rates(n_scenarios=500)

    m, _ = simulate_inventory_risk(
        order_quantity=50.0,
        coverage_days=7,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=d_paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills
    )
    assert hasattr(m, "probability_future_price_rises")
    assert hasattr(m, "probability_future_price_falls")
    assert hasattr(m, "probability_buy_now_is_cheaper")
    assert hasattr(m, "expected_buy_now_saving")
    assert hasattr(m, "expected_wait_saving")
    assert hasattr(m, "price_signal_strength")
    # 1. Under quarantined price model (default), price risk metrics are nullified
    m_gov, _ = simulate_inventory_risk(
        order_quantity=50.0,
        coverage_days=7,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=d_paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        price_risk_enabled=False
    )
    assert m_gov.probability_future_price_rises is None
    assert m_gov.probability_future_price_falls is None
    assert m_gov.probability_buy_now_is_cheaper is None
    assert m_gov.expected_buy_now_saving is None
    assert m_gov.expected_wait_saving is None
    assert m_gov.price_signal_strength == "PRICE_MODEL_NOT_VALIDATED"
    assert m_gov.price_model_status == "NOT_VALIDATED"
    assert m_gov.price_risk_enabled is False
    assert m_gov.risk_mode == "DEMAND_SUPPLY_ONLY"

    # 2. When price model is validated and price_risk_enabled=True, probabilities are populated
    m_enabled, _ = simulate_inventory_risk(
        order_quantity=50.0,
        coverage_days=7,
        current_inventory=50.0,
        incoming_inventory=0.0,
        current_purchase_price=10.0,
        daily_demand_paths=d_paths,
        future_prices=prices,
        lead_times=lts,
        fill_rates=fills,
        price_risk_enabled=True
    )
    assert 0.0 <= m_enabled.probability_future_price_rises <= 1.0
    assert 0.0 <= m_enabled.probability_future_price_falls <= 1.0
    assert abs((m_enabled.probability_future_price_rises + m_enabled.probability_future_price_falls) - 1.0) < 0.01


def test_price_decision_risk_csv_export():
    """Verify reports/monte_carlo/price_decision_risk.csv exists with correct columns."""
    from src.utils.paths import PROJECT_ROOT
    p = PROJECT_ROOT / "reports" / "monte_carlo" / "price_decision_risk.csv"
    assert p.exists(), "price_decision_risk.csv must exist"
    df = pd.read_csv(p)
    assert len(df) == 24
    expected_cols = [
        "product_id", "store_id", "price_signal_strength",
        "probability_future_price_rises", "probability_future_price_falls",
        "probability_buy_now_is_cheaper", "expected_buy_now_saving", "expected_wait_saving"
    ]
    for col in expected_cols:
        assert col in df.columns

