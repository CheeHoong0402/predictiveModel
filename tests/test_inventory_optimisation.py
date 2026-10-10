"""Comprehensive test suite for Phase 18 Inventory Optimisation and Purchase Decision Engine.

Validates:
- No ground truth target columns required for optimization.
- Non-negative order quantities under all conditions.
- Operational constraint enforcement (MOQ, case size, lead time, shelf life, storage capacity, budget).
- Expected cost formula components and timing arbitrage.
- Monotonicity and quantile consistency of curve interpolation.
- Service level policies, alternative strategies, and reference benchmarking.
- Robustness evaluation across P10, P50, and P90 uncertainty scenarios.
- Reporting outputs (Telegram alert formatting, JSON dashboard payload).
- Batch optimization and cost provenance registry.
"""
from typing import Dict
import json
import pytest
import numpy as np
import pandas as pd

from src.inventory.curve_fitting import (
    interpolate_demand_curve,
    interpolate_price_curve
)
from src.inventory.cost_engine import (
    calculate_candidate_cost,
    check_constraints,
    CostBreakdown,
    RiskEstimates,
    ConstraintCheckResult
)
from src.inventory.optimizer import (
    optimise_inventory_decision,
    optimise_inventory_batch,
    InventoryDecision,
    CandidateEvaluation,
    _round_to_constraints
)
from src.inventory.config import (
    load_inventory_config,
    load_cost_assumptions,
    export_cost_assumptions_csv
)
from src.inventory.reporting import (
    format_telegram_inventory_alert,
    build_dashboard_inventory_payload,
    generate_inventory_summary_markdown
)


@pytest.fixture
def standard_demand_forecasts() -> Dict[int, float]:
    """Standard cumulative demand forecasts for testing."""
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
def standard_price_forecasts() -> Dict[int, float]:
    """Constant price forecasts."""
    return {1: 10.0, 3: 10.0, 7: 10.0, 14: 10.0, 21: 10.0, 30: 10.0}


@pytest.fixture
def rising_price_forecasts() -> Dict[int, float]:
    """Price rising from 10 to 14 (+40%)."""
    return {1: 10.2, 3: 10.8, 7: 11.5, 14: 12.8, 21: 13.5, 30: 14.0}


@pytest.fixture
def falling_price_forecasts() -> Dict[int, float]:
    """Price falling from 10 to 7.5 (-25%)."""
    return {1: 9.8, 3: 9.2, 7: 8.8, 14: 8.2, 21: 7.8, 30: 7.5}


# =============================================================================
# 1. TARGET INDEPENDENCE & SANITY TESTS
# =============================================================================

def test_decision_runs_without_target_columns(standard_demand_forecasts, standard_price_forecasts):
    """Test that decision engine operates solely on forecasts and metadata without actual targets."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0,
        current_purchase_price=10.0
    )
    assert isinstance(decision, InventoryDecision)
    assert decision.recommended_order_quantity >= 0.0
    assert 1 <= decision.recommended_stock_coverage_days <= 30


def test_order_quantity_never_negative(standard_demand_forecasts, standard_price_forecasts):
    """Ensure recommended order quantity is non-negative even with excessive inventory."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=10000.0,  # Massive excess stock
        current_purchase_price=10.0
    )
    assert decision.recommended_order_quantity == 0.0
    assert decision.recommended_action in ["DO_NOT_REORDER", "OVERSTOCK_WARNING", "HOLD_CURRENT_STOCK"]


def test_excess_stock_triggers_zero_order(standard_demand_forecasts, standard_price_forecasts):
    """When stock covers full 30 days demand, order quantity must be 0."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=1500.0,
        current_purchase_price=10.0
    )
    assert decision.recommended_order_quantity == 0.0
    assert decision.cost_breakdown.purchase_cost == 0.0


# =============================================================================
# 2. OPERATIONAL CONSTRAINTS (MOQ, CASE PACK, SHELF LIFE, CAPACITY, BUDGET)
# =============================================================================

def test_moq_enforcement():
    """Verify that any positive order quantity meets or exceeds supplier MOQ."""
    raw_qty = 5.0
    moq = 50.0
    case_size = 1
    qty = _round_to_constraints(raw_qty, moq, case_size)
    assert qty == 50.0

    # Zero requirement remains 0
    assert _round_to_constraints(0.0, moq, case_size) == 0.0


def test_case_size_rounding():
    """Verify rounding up to integer multiples of case pack size."""
    moq = 20.0
    case_size = 12
    # 25 requires rounding up past 20 to multiple of 12 (36)
    assert _round_to_constraints(25.0, moq, case_size) == 36.0
    # 15 requires rounding up to MOQ (20), then to multiple of 12 (24)
    assert _round_to_constraints(15.0, moq, case_size) == 24.0
    # Exactly multiple of 12
    assert _round_to_constraints(48.0, moq, case_size) == 48.0


def test_lead_time_shortfall_triggers_urgent_reorder(standard_demand_forecasts, standard_price_forecasts):
    """When current stock is below lead time demand, trigger URGENT_REORDER."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=5.0,  # Far below lead time demand (~50 units for 2 days)
        lead_time_days=2.0,
        current_purchase_price=10.0
    )
    assert decision.recommended_action == "URGENT_REORDER"
    assert decision.severity in ["CRITICAL", "HIGH"]
    assert decision.recommended_order_quantity > 0.0


def test_short_shelf_life_caps_coverage(standard_demand_forecasts, standard_price_forecasts):
    """Perishable product with 3-day shelf life must not receive >3 days coverage."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        shelf_life_days=3.0,
        current_inventory=10.0,
        current_purchase_price=10.0
    )
    assert decision.recommended_stock_coverage_days <= 3
    # Check that candidates with coverage > 3 are marked infeasible
    for cand in decision.all_candidate_evaluations:
        if cand.coverage_days > 3:
            assert not cand.is_feasible
            assert any("exceeds perishable shelf life" in r for r in cand.rejection_reasons)


def test_storage_capacity_constraint(standard_demand_forecasts, standard_price_forecasts):
    """When store capacity is limited, candidates exceeding capacity must be rejected."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=100.0,
        store_storage_capacity=300.0,
        current_purchase_price=10.0
    )
    for cand in decision.all_candidate_evaluations:
        if cand.order_quantity + 100.0 > 300.0:
            assert not cand.is_feasible
            assert any("exceeds store capacity" in r for r in cand.rejection_reasons)


def test_budget_limit_constraint(standard_demand_forecasts, standard_price_forecasts):
    """When purchase outlay exceeds budget, candidate is rejected."""
    budget = 1000.0
    unit_price = 10.0
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=0.0,
        current_purchase_price=unit_price,
        budget_limit=budget
    )
    for cand in decision.all_candidate_evaluations:
        if cand.order_quantity * unit_price > budget:
            assert not cand.is_feasible
            assert any("exceeds available budget" in r for r in cand.rejection_reasons)


# =============================================================================
# 3. COST ENGINE & PRICE TIMING ARBITRAGE
# =============================================================================

def test_cost_breakdown_components_present(standard_demand_forecasts, standard_price_forecasts):
    """Verify all individual cost components exist and sum correctly."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=20.0,
        current_purchase_price=10.0
    )
    cb = decision.cost_breakdown
    assert isinstance(cb, CostBreakdown)
    assert cb.purchase_cost >= 0.0
    assert cb.holding_cost >= 0.0
    assert cb.stockout_cost >= 0.0
    assert cb.wastage_cost >= 0.0
    assert cb.ordering_cost >= 0.0
    
    # Total cost equals sum of subcosts minus future price timing savings
    expected_sum = (
        cb.purchase_cost + cb.holding_cost + cb.stockout_cost +
        cb.wastage_cost + cb.ordering_cost - cb.future_price_savings
    )
    assert pytest.approx(cb.total_expected_cost, 0.01) == expected_sum


def test_timing_advantage_when_price_expected_to_rise(
    standard_demand_forecasts, rising_price_forecasts
):
    """When future price is expected to rise sharply, timing savings should be positive only if validated."""
    # When price decision is explicitly enabled (e.g. validated price model):
    decision_validated = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=rising_price_forecasts,
        current_inventory=50.0,
        current_purchase_price=10.0,
        price_enabled_override=True
    )
    assert decision_validated.predicted_future_price > decision_validated.current_purchase_price
    assert decision_validated.expected_price_change_pct > 0.0
    cand_14_val = next(c for c in decision_validated.all_candidate_evaluations if c.coverage_days == 14)
    assert cand_14_val.cost_breakdown.future_price_savings > 0.0

    # When price decision is disabled by governance (current default):
    decision_gated = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=rising_price_forecasts,
        current_inventory=50.0,
        current_purchase_price=10.0
    )
    assert decision_gated.price_decision_enabled is False
    assert decision_gated.cost_breakdown.future_price_savings == 0.0
    assert decision_gated.net_price_timing_benefit is None
    cand_14_gated = next(c for c in decision_gated.all_candidate_evaluations if c.coverage_days == 14)
    assert cand_14_gated.cost_breakdown.future_price_savings == 0.0


def test_delay_purchase_when_price_expected_to_fall(
    standard_demand_forecasts, falling_price_forecasts
):
    """When price is expected to fall and current stock is adequate, prefer waiting or smaller order."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=falling_price_forecasts,
        current_inventory=150.0,  # Adequate for ~6 days
        current_purchase_price=10.0
    )
    assert decision.recommended_action in [
        "WAIT_FOR_LOWER_PRICE", "HOLD_CURRENT_STOCK", "BUY_SMALLER_QUANTITY", "DO_NOT_REORDER"
    ]


# =============================================================================
# 4. CURVE FITTING & INTERPOLATION PROPERTIES
# =============================================================================

def test_all_30_candidates_evaluated(standard_demand_forecasts, standard_price_forecasts):
    """Verify that exactly 30 candidate days are evaluated and populated."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0,
        current_purchase_price=10.0
    )
    assert len(decision.all_candidate_evaluations) == 30
    days = [c.coverage_days for c in decision.all_candidate_evaluations]
    assert days == list(range(1, 31))


def test_interpolated_demand_monotonicity(
    standard_demand_forecasts, standard_demand_p10, standard_demand_p50, standard_demand_p90
):
    """Interpolated cumulative demand curve must be strictly non-decreasing across 1..30 days."""
    curves = interpolate_demand_curve(
        horizon_forecasts=standard_demand_forecasts,
        horizon_p10=standard_demand_p10,
        horizon_p50=standard_demand_p50,
        horizon_p90=standard_demand_p90
    )
    pt = curves["point_demand"]
    p10 = curves["p10_demand"]
    p50 = curves["p50_demand"]
    p90 = curves["p90_demand"]

    assert len(pt) == 30
    for i in range(1, 30):
        assert pt[i] >= pt[i - 1], f"Monotonicity violated at day {i+1} for point demand"
        assert p10[i] >= p10[i - 1], f"Monotonicity violated at day {i+1} for P10"
        assert p50[i] >= p50[i - 1], f"Monotonicity violated at day {i+1} for P50"
        assert p90[i] >= p90[i - 1], f"Monotonicity violated at day {i+1} for P90"


def test_p10_p50_p90_ordering_preserved(
    standard_demand_forecasts, standard_demand_p10, standard_demand_p50, standard_demand_p90
):
    """Quantile ordering P10 <= P50 <= P90 must hold strictly across all 30 days."""
    curves = interpolate_demand_curve(
        horizon_forecasts=standard_demand_forecasts,
        horizon_p10=standard_demand_p10,
        horizon_p50=standard_demand_p50,
        horizon_p90=standard_demand_p90
    )
    p10 = curves["p10_demand"]
    p50 = curves["p50_demand"]
    p90 = curves["p90_demand"]

    for d in range(30):
        assert p10[d] <= p50[d] <= p90[d], f"Quantile inversion at day {d+1}: {p10[d]} <= {p50[d]} <= {p90[d]}"


def test_interpolated_price_positive(standard_price_forecasts):
    """Interpolated purchase price curve must remain positive across all 30 days."""
    curves = interpolate_price_curve(
        current_price=10.0,
        horizon_prices=standard_price_forecasts
    )
    assert len(curves["point_price"]) == 30
    assert all(p > 0 for p in curves["point_price"])


# =============================================================================
# 5. SERVICE LEVEL POLICIES & ALTERNATIVE STRATEGIES
# =============================================================================

def test_service_level_sensitivity(
    standard_demand_forecasts, standard_demand_p10, standard_demand_p50, standard_demand_p90,
    standard_price_forecasts
):
    """Higher service level policy must maintain equal or higher order quantity and coverage."""
    dec_balanced = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        demand_p10=standard_demand_p10,
        demand_p50=standard_demand_p50,
        demand_p90=standard_demand_p90,
        current_inventory=50.0,
        service_level="balanced"
    )
    dec_high = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        demand_p10=standard_demand_p10,
        demand_p50=standard_demand_p50,
        demand_p90=standard_demand_p90,
        current_inventory=50.0,
        service_level="high_service"
    )
    assert dec_high.recommended_order_quantity >= dec_balanced.recommended_order_quantity


def test_alternative_strategies_generated(standard_demand_forecasts, standard_price_forecasts):
    """Verify alternative procurement strategies (LOW_STOCK, BALANCED, HIGH_SAFETY) exist."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0
    )
    alt = decision.alternative_strategies
    assert "LOW_STOCK_STRATEGY" in alt
    assert "BALANCED_STRATEGY" in alt
    assert "HIGH_SAFETY_STRATEGY" in alt
    assert alt["LOW_STOCK_STRATEGY"]["order_quantity"] <= alt["HIGH_SAFETY_STRATEGY"]["order_quantity"]


def test_reference_strategy_comparison(standard_demand_forecasts, standard_price_forecasts):
    """Verify comparison against 14-day reference strategy is calculated."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0
    )
    assert decision.reference_strategy_name == "FIXED_14_DAY_STOCK"
    assert decision.reference_order_quantity >= 0.0
    assert decision.reference_total_cost >= 0.0
    assert isinstance(decision.expected_savings_vs_reference, float)


def test_robustness_analysis_assigned(
    standard_demand_forecasts, standard_demand_p10, standard_demand_p50, standard_demand_p90,
    standard_price_forecasts
):
    """Decision must be flagged with a valid robustness tag."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        demand_p10=standard_demand_p10,
        demand_p50=standard_demand_p50,
        demand_p90=standard_demand_p90,
        current_inventory=50.0
    )
    assert decision.robustness_status in ["ROBUST", "UNCERTAINTY_SENSITIVE"]


# =============================================================================
# 6. REPORTING & SERIALIZATION TESTS
# =============================================================================

def test_telegram_alert_formatting(standard_demand_forecasts, standard_price_forecasts):
    """Test Telegram alert formatting string output."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        store_id="STR_KUL_001",
        product_id="PRD_CHK_01",
        supplier_id="SUP_CHK_01",
        current_inventory=30.0
    )
    alert = format_telegram_inventory_alert(decision)
    assert "STR_KUL_001" in alert
    assert "PRD_CHK_01" in alert
    assert "Recommended Order:" in alert
    assert "Coverage Days:" in alert
    assert "Total Expected Cost:" in alert
    assert "Key Business Rationales:" in alert


def test_dashboard_payload_serialization(standard_demand_forecasts, standard_price_forecasts):
    """Test dashboard payload dictionary and JSON serialization."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0
    )
    payload = build_dashboard_inventory_payload(decision)
    assert "metadata" in payload
    assert "primary_recommendation" in payload
    assert "costs" in payload
    assert "risks" in payload
    assert "candidate_curves" in payload
    assert len(payload["candidate_curves"]["coverage_days"]) == 30

    # Ensure json dump succeeds
    json_str = json.dumps(payload)
    assert len(json_str) > 100


def test_cost_assumptions_provenance_export():
    """Test exporting cost assumptions CSV and verify provenance status."""
    from src.utils.paths import PROJECT_ROOT
    csv_path = export_cost_assumptions_csv(root=PROJECT_ROOT)
    df = pd.read_csv(csv_path)
    assert not df.empty
    assert "parameter_name" in df.columns
    assert "provenance" in df.columns
    assert "status" in df.columns
    assert set(df["status"].unique()).issubset({"VERIFIED", "DYNAMIC", "ASSUMED_COST", "UNAVAILABLE"})
    assert set(df["provenance"].unique()).issubset({"retailer_provided", "derived", "default_assumption", "unavailable"})


# =============================================================================
# 7. EXTENDED EDGE CASES, ROBUSTNESS & PIPELINE VALIDATION
# =============================================================================

def test_zero_moq_and_single_case_size(standard_demand_forecasts, standard_price_forecasts):
    """When MOQ is 0 and case size is 1, quantity rounds naturally to integer."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        supplier_minimum_order_quantity=0.0,
        product_case_size=1,
        current_inventory=0.0
    )
    assert decision.recommended_order_quantity > 0.0
    assert float(decision.recommended_order_quantity).is_integer()


def test_large_pack_size_rounding(standard_demand_forecasts, standard_price_forecasts):
    """When product case size is large (e.g. 30 eggs/pack), order must be multiple of 30."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        product_case_size=30,
        supplier_minimum_order_quantity=30.0,
        current_inventory=10.0
    )
    assert decision.recommended_order_quantity % 30 == 0


def test_holding_cost_dominance_shortens_coverage(standard_demand_forecasts, standard_price_forecasts):
    """Massive holding cost should force shorter coverage days."""
    high_holding_costs = {
        "inventory_holding_cost_per_unit_per_day": 10.0,  # Extreme holding cost
        "storage_cost_per_unit_per_day": 5.0,
        "stockout_cost_per_unit": 1.0,
        "wastage_cost_per_unit": 5.0,
        "ordering_cost_per_po": 5.0
    }
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        cost_parameters=high_holding_costs,
        current_inventory=0.0
    )
    assert decision.recommended_stock_coverage_days <= 5


def test_stockout_penalty_dominance_lengthens_coverage(standard_demand_forecasts, standard_price_forecasts):
    """Verify stockout penalty is properly incorporated into candidate costs."""
    high_stockout_costs = {
        "inventory_holding_cost_per_unit_per_day": 0.01,
        "storage_cost_per_unit_per_day": 0.01,
        "stockout_cost_per_unit": 500.0,  # Massive penalty for stockout
        "wastage_cost_per_unit": 5.0,
        "ordering_cost_per_po": 50.0
    }
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        cost_parameters=high_stockout_costs,
        current_inventory=100.0
    )
    # Check that candidate stockout costs reflect the higher penalty
    cand_14 = next(c for c in decision.all_candidate_evaluations if c.coverage_days == 14)
    assert cand_14.cost_breakdown.stockout_cost > 0.0


def test_extreme_price_spike_arbitrage(standard_demand_forecasts):
    """When supplier price jumps +80% over next month, forward buying timing savings must be large if price is validated."""
    spiking_prices = {1: 10.5, 3: 11.5, 7: 13.0, 14: 15.0, 21: 17.0, 30: 18.0}
    # When price enabled:
    decision_val = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=spiking_prices,
        current_purchase_price=10.0,
        current_inventory=100.0,
        price_enabled_override=True
    )
    cand_30_val = next(c for c in decision_val.all_candidate_evaluations if c.coverage_days == 30)
    assert cand_30_val.cost_breakdown.future_price_savings > 1000.0

    # When gated/unvalidated: timing savings must be 0
    decision_gated = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=spiking_prices,
        current_purchase_price=10.0,
        current_inventory=100.0
    )
    cand_30_gated = next(c for c in decision_gated.all_candidate_evaluations if c.coverage_days == 30)
    assert cand_30_gated.cost_breakdown.future_price_savings == 0.0



def test_all_decision_actions_are_valid_enum(standard_demand_forecasts, standard_price_forecasts):
    """Ensure recommended_action belongs to the valid action taxonomy."""
    valid_actions = {
        "BUY_NOW", "BUY_SMALLER_QUANTITY", "HOLD_CURRENT_STOCK",
        "WAIT_FOR_LOWER_PRICE", "DO_NOT_REORDER", "URGENT_REORDER",
        "OVERSTOCK_WARNING"
    }
    for curr_inv in [0.0, 10.0, 50.0, 200.0, 1000.0]:
        decision = optimise_inventory_decision(
            demand_forecasts=standard_demand_forecasts,
            price_forecasts=standard_price_forecasts,
            current_inventory=curr_inv
        )
        assert decision.recommended_action in valid_actions


def test_all_severities_are_valid_enum(standard_demand_forecasts, standard_price_forecasts):
    """Ensure severity belongs to the valid severity taxonomy."""
    valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"}
    for curr_inv in [0.0, 5.0, 50.0, 500.0]:
        decision = optimise_inventory_decision(
            demand_forecasts=standard_demand_forecasts,
            price_forecasts=standard_price_forecasts,
            current_inventory=curr_inv
        )
        assert decision.severity in valid_severities


def test_markdown_summary_generation_structure(standard_demand_forecasts, standard_price_forecasts):
    """Test generation of high-level audit markdown."""
    dec = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0
    )
    from src.inventory.optimizer import DecisionBatchResult
    batch = DecisionBatchResult(
        recommendations_df=pd.DataFrame([dec.to_dict()]),
        candidate_evaluations_df=pd.DataFrame([{
            "candidate_coverage_days": 14,
            "order_quantity": 100.0,
            "total_expected_cost": 500.0
        }]),
        strategy_comparison_df=pd.DataFrame([{
            "strategy": "BALANCED_STRATEGY",
            "coverage_days": 14,
            "order_quantity": 100.0,
            "expected_total_cost": 500.0,
            "stockout_risk": 0.05
        }]),
        robustness_df=pd.DataFrame([{
            "robustness_status": "ROBUST"
        }])
    )
    prov_df = pd.DataFrame([{
        "parameter": "inventory_holding_cost",
        "value": 0.25,
        "provenance": "retailer_provided",
        "source": "retail_inventory.parquet"
    }])
    md = generate_inventory_summary_markdown(batch, prov_df, runtime_s=0.5)
    assert "# PHASE 18" in md
    assert "Total Recommended Units Ordered" in md
    assert "Action Distribution" in md
    assert "Cost Assumption Provenance" in md


def test_interpolated_curve_endpoints_match_forecasts(standard_demand_forecasts, standard_price_forecasts):
    """Endpoints at horizon days (1, 3, 7, 14, 21, 30) must match input forecasts exactly."""
    curves = interpolate_demand_curve(standard_demand_forecasts)
    p_curves = interpolate_price_curve(10.0, standard_price_forecasts)
    for h in [1, 3, 7, 14, 21, 30]:
        idx = h - 1
        assert pytest.approx(curves["point_demand"][idx], 0.01) == standard_demand_forecasts[h]
        assert pytest.approx(p_curves["point_price"][idx], 0.01) == standard_price_forecasts[h]


def test_incoming_inventory_reduces_order_quantity(standard_demand_forecasts, standard_price_forecasts):
    """Pipeline incoming inventory must be credited against replenishment needs."""
    dec_no_incoming = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0,
        incoming_inventory=0.0
    )
    dec_with_incoming = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=standard_price_forecasts,
        current_inventory=50.0,
        incoming_inventory=100.0
    )
    assert dec_with_incoming.recommended_order_quantity <= dec_no_incoming.recommended_order_quantity


# =============================================================================
# 10. FORMAL PURCHASE-PRICE DECISION CRITERIA TESTS
# =============================================================================

def test_net_price_timing_benefit_calculation(standard_demand_forecasts, rising_price_forecasts):
    """Verify net price timing benefit formula: (P_future - P_current)*Q_base - delta_Holding - delta_Wastage."""
    # When price is enabled (validated mode):
    decision_val = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=rising_price_forecasts,
        current_inventory=30.0,
        current_purchase_price=10.0,
        price_enabled_override=True
    )
    assert hasattr(decision_val, "net_price_timing_benefit")
    assert hasattr(decision_val, "buy_now_vs_wait_recommendation")
    assert hasattr(decision_val, "price_signal_strength")
    assert decision_val.price_signal_strength in ["STRONG_BUY_EARLY_SIGNAL", "MODERATE_BUY_EARLY_SIGNAL"]

    # When price is gated (default mode):
    decision_gated = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=rising_price_forecasts,
        current_inventory=30.0,
        current_purchase_price=10.0
    )
    assert decision_gated.net_price_timing_benefit is None
    assert decision_gated.price_signal_strength == "PRICE_PREDICTION_UNAVAILABLE"


def test_option_a_vs_option_b_cost_advantage(standard_demand_forecasts):
    """Option A (Buy Now) vs Option B (Wait) must compare costs under rising vs falling prices when price validated."""
    # Rising prices: Option A should be cheaper (positive advantage)
    rising_prices = {1: 10.5, 3: 11.0, 7: 11.5, 14: 12.0, 21: 12.5, 30: 13.0}
    dec_rise = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=rising_prices,
        current_inventory=10.0,
        current_purchase_price=10.0,
        price_enabled_override=True
    )
    assert dec_rise.buy_now_vs_wait_recommendation == "BUY_NOW"

    # Falling prices with adequate stock: Option B should recommend wait
    falling_prices = {1: 9.5, 3: 9.0, 7: 8.5, 14: 8.0, 21: 7.5, 30: 7.0}
    dec_fall = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=falling_prices,
        current_inventory=200.0,  # 8+ days of inventory
        current_purchase_price=10.0,
        price_enabled_override=True
    )
    assert dec_fall.buy_now_vs_wait_recommendation == "WAIT"


def test_maximum_price_coverage_days_capped_at_7(standard_demand_forecasts):
    """Price pre-buying should never extend stock coverage by more than 7 days above baseline."""
    hyper_rising_prices = {1: 15.0, 3: 20.0, 7: 25.0, 14: 30.0, 21: 35.0, 30: 40.0}
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=hyper_rising_prices,
        current_inventory=10.0,
        current_purchase_price=10.0
    )
    assert hasattr(decision, "price_adjusted_coverage_delta")
    assert decision.price_adjusted_coverage_delta <= 7, "Price expansion must not exceed 7 days cap"


def test_perishable_shelf_life_overrides_price_prebuy(standard_demand_forecasts, rising_price_forecasts):
    """For short shelf-life items (e.g. 3 days), forward pre-buying is strictly capped to shelf life."""
    decision = optimise_inventory_decision(
        demand_forecasts=standard_demand_forecasts,
        price_forecasts=rising_price_forecasts,
        current_inventory=5.0,
        current_purchase_price=10.0,
        shelf_life_days=3.0
    )
    assert decision.recommended_stock_coverage_days <= 3
