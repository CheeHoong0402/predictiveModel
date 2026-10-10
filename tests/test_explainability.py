"""Unit and integration test suite for Phase 17 Explainability, Model Transparency and Training Coverage Audit.

Validates:
1. Feature Display Dictionary and YAML configuration
2. Exact TreeSHAP additivity on LightGBM, CatBoost, XGBoost, and weighted ensembles
3. Local Demand Explanations API (schema, target independence, monotonicity, units, formatting)
4. Local Price Explanations API (schema, target independence, MYR conversions, formatting)
5. Training Data Duration and Entity Coverage Audit
6. Walk-Forward Folds (33 expanding folds) and row exclusions
7. Final Holdout strict protection and reservation
8. Output artifacts existence and file integrity (CSVs, Markdown, Parquet)
9. Visualizations existence and size (9 publication-grade figures at 300 DPI)
10. Inference latency thresholds (< 200ms per prediction)
"""
from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT
from src.explainability.display_names import FeatureDisplayDictionary
from src.explainability.tree_shap import compute_tree_shap
from src.explainability.explainers import explain_demand_prediction, explain_price_prediction
from src.explainability.training_coverage import (
    audit_model_training_coverage,
    audit_walk_forward_folds,
    audit_training_exclusions
)
from src.ensemble.inference import predict_final_demand, predict_final_purchase_price


@pytest.fixture(scope="module")
def display_dict():
    return FeatureDisplayDictionary.load(root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def sample_feature_row():
    return pd.DataFrame([{
        "date": "2024-09-15",
        "store_id": "STR_KUL_001",
        "product_id": "PRD_CHK_01",
        "supplier_id": "SUP_01",
        "purchase_price": 14.50,
        "demand_lag_1": 45.0,
        "demand_lag_7": 42.0,
        "rolling_mean_demand_28": 40.5,
        "diesel_price_lag_1": 3.35,
        "usd_myr_lag_1": 4.65,
        "myr_usd_change_30d": 0.02,
        "cpo_price_lag_1": 3950.0
    }])


@pytest.fixture(scope="module")
def multi_feature_rows():
    return pd.DataFrame([
        {
            "date": "2024-09-15",
            "store_id": "STR_KUL_001",
            "product_id": "PRD_CHK_01",
            "supplier_id": "SUP_01",
            "purchase_price": 14.50,
            "demand_lag_1": 45.0,
            "demand_lag_7": 42.0
        },
        {
            "date": "2024-09-15",
            "store_id": "STR_PNG_004",
            "product_id": "PRD_EGG_02",
            "supplier_id": "SUP_02",
            "purchase_price": 8.20,
            "demand_lag_1": 110.0,
            "demand_lag_7": 105.0
        }
    ])


# =====================================================================
# 1. FEATURE DISPLAY DICTIONARY & CONFIGURATION TESTS (7 tests)
# =====================================================================

def test_01_display_dict_loads_successfully(display_dict):
    """Test FeatureDisplayDictionary instantiates and loads metadata."""
    assert display_dict is not None
    assert len(display_dict._name_to_display) > 0


def test_02_yaml_configuration_exists_and_valid():
    """Test config/feature_display_names.yaml exists and is valid YAML."""
    yaml_path = PROJECT_ROOT / "config/feature_display_names.yaml"
    assert yaml_path.exists()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_03_csv_dictionary_schema():
    """Test reports/explainability/feature_display_dictionary.csv has complete schema."""
    csv_path = PROJECT_ROOT / "reports/explainability/feature_display_dictionary.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    expected_cols = ["raw_feature_name", "display_name", "description", "feature_family", "unit", "source", "task_applicability"]
    for col in expected_cols:
        assert col in df.columns


def test_04_explicit_aliases_resolution(display_dict):
    """Test explicitly required aliases resolve correctly."""
    assert "Demand 7 days ago" in display_dict.get_display_name("demand_lag_7")
    assert "28 days" in display_dict.get_display_name("rolling_mean_demand_28")
    assert "ringgit" in display_dict.get_display_name("myr_usd_change_30d").lower()


def test_05_business_family_resolution(display_dict):
    """Test resolution of raw feature names to business families."""
    assert display_dict.get_business_family("demand_lag_1") == "Historical Demand"
    assert display_dict.get_business_family("cpo_price_lag_1") == "Commodity & Global Markets"
    assert display_dict.get_business_family("diesel_price_lag_1") == "Fuel & Energy"


def test_06_fallback_display_name(display_dict):
    """Test graceful fallback for unknown feature name."""
    res = display_dict.get_display_name("unknown_sensor_signal_99")
    assert res == "Unknown Sensor Signal 99"


def test_07_fallback_business_family(display_dict):
    """Test graceful fallback for unknown business family."""
    fam = display_dict.get_business_family("xyz_unseen_term")
    assert fam == "General Drivers"


# =====================================================================
# 2. TREESHAP COMPUTATION & ADDITIVITY TESTS (6 tests)
# =====================================================================

def test_08_treeshap_additivity_demand_1d_ensemble(sample_feature_row):
    """Test exact additivity for Demand 1d weighted ensemble."""
    shap_df, base_val, feats = compute_tree_shap("demand", 1, sample_feature_row)
    pred_pt = predict_final_demand(sample_feature_row, 1)["final_point_forecast"].iloc[0]
    recon = shap_df.iloc[0].sum() + base_val[0]
    np.testing.assert_allclose(recon, pred_pt, rtol=1e-4)


def test_09_treeshap_additivity_demand_3d_xgboost(sample_feature_row):
    """Test exact additivity for Demand 3d XGBoost single model."""
    shap_df, base_val, feats = compute_tree_shap("demand", 3, sample_feature_row)
    pred_pt = predict_final_demand(sample_feature_row, 3)["final_point_forecast"].iloc[0]
    recon = shap_df.iloc[0].sum() + base_val[0]
    np.testing.assert_allclose(recon, pred_pt, rtol=1e-4)


def test_10_treeshap_additivity_demand_7d_catboost(sample_feature_row):
    """Test exact additivity for Demand 7d CatBoost single model."""
    shap_df, base_val, feats = compute_tree_shap("demand", 7, sample_feature_row)
    pred_pt = predict_final_demand(sample_feature_row, 7)["final_point_forecast"].iloc[0]
    recon = shap_df.iloc[0].sum() + base_val[0]
    np.testing.assert_allclose(recon, pred_pt, rtol=1e-4)


def test_11_treeshap_additivity_price_1d_lightgbm(sample_feature_row):
    """Test exact additivity for Price 1d LightGBM percentage change model."""
    shap_df, base_val, feats = compute_tree_shap("price", 1, sample_feature_row)
    pred_pct = predict_final_purchase_price(sample_feature_row, 1)["predicted_percentage_change"].iloc[0]
    recon = shap_df.iloc[0].sum() + base_val[0]
    np.testing.assert_allclose(recon, pred_pct, rtol=1e-4)


def test_12_treeshap_additivity_price_14d_ensemble(sample_feature_row):
    """Test exact additivity for Price 14d ensemble model."""
    shap_df, base_val, feats = compute_tree_shap("price", 14, sample_feature_row)
    pred_pct = predict_final_purchase_price(sample_feature_row, 14)["predicted_percentage_change"].iloc[0]
    recon = shap_df.iloc[0].sum() + base_val[0]
    np.testing.assert_allclose(recon, pred_pct, rtol=1e-4)


def test_13_treeshap_matrix_shape(multi_feature_rows):
    """Test TreeSHAP returns matching shape (N, n_features) and base_val (N,)."""
    shap_df, base_val, feats = compute_tree_shap("demand", 1, multi_feature_rows)
    assert shap_df.shape[0] == len(multi_feature_rows)
    assert len(base_val) == len(multi_feature_rows)
    assert len(feats) == shap_df.shape[1]


# =====================================================================
# 3. DEMAND LOCAL EXPLANATIONS API TESTS (7 tests)
# =====================================================================

def test_14_explain_demand_single_row_schema(sample_feature_row):
    """Test explain_demand_prediction returns dictionary with required keys."""
    exp = explain_demand_prediction(sample_feature_row, 1)
    assert isinstance(exp, dict)
    required_keys = [
        "task", "horizon", "date", "store_id", "product_id", "point_prediction",
        "base_value", "p10", "p50", "p90", "interval_width", "unit",
        "top_positive_drivers", "top_negative_drivers", "family_contributions",
        "narrative_explanation", "telegram_message", "dashboard_payload"
    ]
    for k in required_keys:
        assert k in exp, f"Missing key: {k}"


def test_15_explain_demand_target_independence(sample_feature_row):
    """Test demand explainer succeeds with no target column present."""
    df_no_target = sample_feature_row.copy()
    assert not any("target" in c or "demand" == c for c in df_no_target.columns)
    exp = explain_demand_prediction(df_no_target, 7)
    assert exp["point_prediction"] is not None


def test_16_explain_demand_uncertainty_monotonicity(sample_feature_row):
    """Test demand uncertainty interval satisfies P10 <= P50 <= P90."""
    exp = explain_demand_prediction(sample_feature_row, 1)
    assert exp["p10"] <= exp["p50"] <= exp["p90"]
    assert exp["interval_width"] == pytest.approx(exp["p90"] - exp["p10"], rel=1e-3)


def test_17_explain_demand_top_drivers_signs(sample_feature_row):
    """Test positive drivers have positive contribution and negative drivers have negative contribution."""
    exp = explain_demand_prediction(sample_feature_row, 1, top_n=3)
    for d in exp["top_positive_drivers"]:
        assert d["contribution"] > 0
        assert d["impact"] == "INCREASING_DEMAND"
    for d in exp["top_negative_drivers"]:
        assert d["contribution"] < 0
        assert d["impact"] == "DECREASING_DEMAND"


def test_18_explain_demand_top_n_respects_limit(sample_feature_row):
    """Test top_n limit is respected."""
    exp = explain_demand_prediction(sample_feature_row, 1, top_n=2)
    assert len(exp["top_positive_drivers"]) <= 2
    assert len(exp["top_negative_drivers"]) <= 2


def test_19_explain_demand_multi_row_batch(multi_feature_rows):
    """Test multi-row input returns list of dictionaries of matching length."""
    exps = explain_demand_prediction(multi_feature_rows, 1)
    assert isinstance(exps, list)
    assert len(exps) == 2


def test_20_explain_demand_dashboard_payload_serializable(sample_feature_row):
    """Test dashboard payload is valid JSON serializable."""
    exp = explain_demand_prediction(sample_feature_row, 1)
    json_str = json.dumps(exp["dashboard_payload"])
    assert len(json_str) > 0


# =====================================================================
# 4. PRICE LOCAL EXPLANATIONS API TESTS (7 tests)
# =====================================================================

def test_21_explain_price_single_row_schema(sample_feature_row):
    """Test explain_price_prediction returns dictionary with required keys."""
    exp = explain_price_prediction(sample_feature_row, 1)
    assert isinstance(exp, dict)
    required_keys = [
        "task", "horizon", "date", "store_id", "product_id", "supplier_id",
        "current_price", "point_prediction", "predicted_percentage_change",
        "base_value_myr", "p10", "p50", "p90", "interval_width", "unit",
        "top_positive_drivers", "top_negative_drivers", "family_contributions",
        "narrative_explanation", "telegram_message", "dashboard_payload"
    ]
    for k in required_keys:
        assert k in exp, f"Missing key: {k}"


def test_22_explain_price_target_independence(sample_feature_row):
    """Test price explainer succeeds without target columns."""
    df_no_target = sample_feature_row.copy()
    assert not any("target" in c for c in df_no_target.columns)
    exp = explain_price_prediction(df_no_target, 1)
    assert exp["point_prediction"] is not None


def test_23_explain_price_myr_conversion(sample_feature_row):
    """Test conversion from percentage SHAP to MYR business contribution."""
    exp = explain_price_prediction(sample_feature_row, 1)
    curr = exp["current_price"]
    for d in exp["top_positive_drivers"]:
        expected_myr = curr * (d["contribution_pct"] / 100.0)
        assert d["contribution_myr"] == pytest.approx(expected_myr, abs=0.01)


def test_24_explain_price_uncertainty_monotonicity(sample_feature_row):
    """Test price uncertainty interval satisfies P10 <= P50 <= P90."""
    exp = explain_price_prediction(sample_feature_row, 1)
    assert exp["p10"] <= exp["p50"] <= exp["p90"]
    assert exp["unit"] == "MYR"


def test_25_explain_price_top_drivers_impact_labels(sample_feature_row):
    """Test driver impact labels for price."""
    exp = explain_price_prediction(sample_feature_row, 1)
    for d in exp["top_positive_drivers"]:
        assert d["impact"] == "INCREASING_PRICE"
        assert d["contribution_myr"] >= 0
    for d in exp["top_negative_drivers"]:
        assert d["impact"] == "DECREASING_PRICE"
        assert d["contribution_myr"] <= 0


def test_26_explain_price_multi_row_batch(multi_feature_rows):
    """Test multi-row price input returns list of matching length."""
    exps = explain_price_prediction(multi_feature_rows, 1)
    assert isinstance(exps, list)
    assert len(exps) == 2


def test_27_explain_price_dashboard_payload_serializable(sample_feature_row):
    """Test price dashboard payload is valid JSON serializable."""
    exp = explain_price_prediction(sample_feature_row, 1)
    json_str = json.dumps(exp["dashboard_payload"])
    assert len(json_str) > 0


# =====================================================================
# 5. TRAINING COVERAGE AND HISTORICAL DURATION AUDIT TESTS (7 tests)
# =====================================================================

def test_28_training_coverage_csv_schema_and_rows():
    """Test model_training_coverage.csv exists and has 12 entries."""
    csv_path = PROJECT_ROOT / "reports/explainability/model_training_coverage.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 12  # 6 demand + 6 price
    assert "train_calendar_days" in df.columns
    assert "train_rows" in df.columns
    assert "final_holdout_reserved" in df.columns


def test_29_training_temporal_span_bounds():
    """Test training start date is 2021-01-01 and duration is approx 3.75 years."""
    df = pd.read_csv(PROJECT_ROOT / "reports/explainability/model_training_coverage.csv")
    assert (df["train_start_date"] == "2021-01-01").all()
    # 1d horizon reaches 2024-09-30 (1369 days)
    assert df["train_calendar_days"].max() == 1369
    assert df["train_calendar_days"].min() >= 1340


def test_30_final_holdout_strictly_reserved():
    """Test all models certify final holdout reservation of 92 days / 2,208 rows."""
    df = pd.read_csv(PROJECT_ROOT / "reports/explainability/model_training_coverage.csv")
    assert df["final_holdout_reserved"].all()
    assert (df["final_holdout_start"] == "2024-10-01").all()
    assert (df["final_holdout_end"] == "2024-12-31").all()
    assert (df["final_holdout_days"] == 92).all()
    assert (df["final_holdout_rows"] == 2208).all()


def test_31_entity_coverage():
    """Test coverage across 4 stores, 6 products, 6 suppliers."""
    df = pd.read_csv(PROJECT_ROOT / "reports/explainability/model_training_coverage.csv")
    assert (df["unique_stores"] == 4).all()
    assert (df["unique_products"] == 6).all()
    assert (df["unique_suppliers"] == 6).all()


def test_32_longest_lookback_vs_training_span():
    """Test 365-day feature lookback ratio is under 30% of training duration."""
    df = pd.read_csv(PROJECT_ROOT / "reports/explainability/model_training_coverage.csv")
    assert (df["longest_feature_lookback_days"] == 365).all()
    assert (df["lookback_to_training_ratio_pct"] < 30.0).all()


def test_33_walk_forward_folds_coverage():
    """Test walk_forward_training_coverage.csv contains all 33 expanding folds."""
    csv_path = PROJECT_ROOT / "reports/explainability/walk_forward_training_coverage.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 33
    assert df.iloc[0]["validation_fold"] == 1
    assert df.iloc[-1]["validation_fold"] == 33
    assert df.iloc[0]["training_start_date"] == "2021-01-01"


def test_34_training_exclusions_summary():
    """Test training_exclusion_summary.csv exists and documents boundaries."""
    csv_path = PROJECT_ROOT / "reports/explainability/training_exclusion_summary.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) >= 3
    assert "leakage_prevention_rationale" in df.columns


# =====================================================================
# 6. GLOBAL IMPORTANCE, RUNTIME & ARTIFACTS INTEGRITY TESTS (7 tests)
# =====================================================================

def test_35_global_demand_importance_exists():
    """Test global_demand_importance.csv exists, is non-empty, and sorted."""
    csv_path = PROJECT_ROOT / "reports/explainability/global_demand_importance.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) > 10
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[0]["mean_abs_shap"] >= df.iloc[1]["mean_abs_shap"]


def test_36_global_price_importance_exists():
    """Test global_price_importance.csv exists, is non-empty, and sorted."""
    csv_path = PROJECT_ROOT / "reports/explainability/global_price_importance.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) > 10
    assert df.iloc[0]["rank"] == 1
    assert df.iloc[0]["mean_abs_shap"] >= df.iloc[1]["mean_abs_shap"]


def test_37_explanation_runtime_benchmarks():
    """Test explanation_runtime.csv exists and all horizons meet sub-200ms threshold."""
    csv_path = PROJECT_ROOT / "reports/explainability/explanation_runtime.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 12  # 6 demand + 6 price
    assert df["meets_threshold"].all()
    assert (df["avg_latency_ms"] < 200.0).all()


def test_38_local_prediction_explanations_parquet_exists():
    """Test data/explanations/local_prediction_explanations.parquet exists and has rows."""
    p_path = PROJECT_ROOT / "data/explanations/local_prediction_explanations.parquet"
    assert p_path.exists()
    df = pd.read_parquet(p_path)
    assert len(df) > 0
    assert "demand_forecast" in df.columns
    assert "price_forecast_myr" in df.columns


def test_39_markdown_reports_exist():
    """Test all markdown reports exist and have substantial content."""
    md_files = [
        PROJECT_ROOT / "reports/explainability/model_training_coverage_summary.md",
        PROJECT_ROOT / "reports/explainability/explanation_limitations.md",
        PROJECT_ROOT / "reports/explainability/explainability_summary.md"
    ]
    for md in md_files:
        assert md.exists()
        assert len(md.read_text(encoding="utf-8")) > 200


def test_40_visualisation_figures_all_nine_exist():
    """Test all 9 publication-grade PNG figures exist with valid file size (>10KB)."""
    fig_names = [
        "demand_top15_features.png",
        "demand_feature_families.png",
        "demand_family_contribution_by_horizon.png",
        "price_top15_features.png",
        "price_feature_families.png",
        "price_family_contribution_by_horizon.png",
        "training_timeline.png",
        "local_demand_waterfall_example.png",
        "local_price_waterfall_example.png"
    ]
    fig_dir = PROJECT_ROOT / "reports/figures/explainability"
    for name in fig_names:
        fig_path = fig_dir / name
        assert fig_path.exists(), f"Figure {name} does not exist"
        assert fig_path.stat().st_size > 10_000, f"Figure {name} is unusually small: {fig_path.stat().st_size} bytes"


def test_41_no_data_leakage_in_explanations(sample_feature_row):
    """Test explanations do not use or reveal final holdout actuals."""
    # Ensure sample date is inside development period
    assert pd.Timestamp(sample_feature_row["date"].iloc[0]) < pd.Timestamp("2024-10-01")
    exp_d = explain_demand_prediction(sample_feature_row, 1)
    exp_p = explain_price_prediction(sample_feature_row, 1)
    assert "actual" not in exp_d["dashboard_payload"]["forecast"]
    assert "actual" not in exp_p["dashboard_payload"]["forecast"]
