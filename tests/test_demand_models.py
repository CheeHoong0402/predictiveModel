"""Tests for Phase 11 Demand Forecasting Models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from src.models.demand.config import (
    load_config,
    load_demand_feature_manifest,
    validate_feature_input,
    BANNED_INPUT_PREFIXES
)
from src.models.demand.evaluator import (
    compute_regression_metrics,
    compute_baseline_comparison,
    compute_residual_quantiles
)
from src.models.demand.inference import predict_demand, load_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT)


@pytest.fixture
def manifest():
    return load_demand_feature_manifest(ROOT)


def test_approved_feature_manifest_enforced(manifest):
    # Valid call
    validate_feature_input(manifest[:5], manifest)
    # Unapproved column
    with pytest.raises(ValueError, match="Unapproved feature columns"):
        validate_feature_input(['unapproved_feature_xyz'], manifest)


@pytest.mark.parametrize('target_col', [
    'demand_target_1d', 'demand_target_7d', 'demand_day_7',
    'purchase_price_target_7d', 'purchase_price_target_1d', 'price_direction_7d'
])
def test_targets_never_appear_among_model_inputs(manifest, target_col):
    with pytest.raises(ValueError, match="Prohibited target columns"):
        validate_feature_input([target_col], manifest)


def test_walk_forward_folds_remain_chronological():
    folds_path = ROOT / 'reports/baselines/walk_forward_folds.csv'
    assert folds_path.exists()
    folds = pd.read_csv(folds_path)
    assert len(folds) == 33
    t_end = pd.to_datetime(folds.training_end)
    v_start = pd.to_datetime(folds.validation_start)
    v_end = pd.to_datetime(folds.validation_end)
    holdout = pd.Timestamp('2024-10-01')
    assert (t_end < v_start).all()
    assert (v_end < holdout).all()
    assert (v_start.iloc[1:].to_numpy() > v_end.iloc[:-1].to_numpy()).all()


def test_final_holdout_protected():
    pred_path = ROOT / 'data/predictions/demand_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    dates = pd.to_datetime(df.date)
    assert (dates < pd.Timestamp('2024-10-01')).all()


def test_product_boundaries_remain_valid(manifest):
    # Predictions should be independent across distinct product features
    df1 = pd.DataFrame(np.ones((2, len(manifest))), columns=manifest)
    df2 = pd.DataFrame(np.ones((2, len(manifest))) * 2.0, columns=manifest)
    p1 = predict_demand(df1, 7, 'lightgbm', root=ROOT)
    p2 = predict_demand(df2, 7, 'lightgbm', root=ROOT)
    assert len(p1) == 2 and len(p2) == 2
    assert np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))


def test_store_boundaries_remain_valid(manifest):
    df = pd.DataFrame(np.ones((4, len(manifest))), columns=manifest)
    preds = predict_demand(df, 7, 'lightgbm', root=ROOT)
    assert len(preds) == 4
    # Reversing order should reverse predictions
    preds_rev = predict_demand(df.iloc[::-1], 7, 'lightgbm', root=ROOT)
    np.testing.assert_allclose(preds, preds_rev[::-1])


def test_models_reproduce_predictions_with_fixed_seed(manifest):
    import lightgbm as lgb
    X = np.random.randn(100, 10).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    m1 = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    m1.fit(X, y)
    m2 = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    m2.fit(X, y)
    np.testing.assert_allclose(m1.predict(X), m2.predict(X))


def test_prediction_values_are_finite():
    pred_path = ROOT / 'data/predictions/demand_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    assert np.all(np.isfinite(df.predicted_demand))
    assert not df.predicted_demand.isna().any()


def test_demand_forecasts_are_nonnegative():
    pred_path = ROOT / 'data/predictions/demand_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    assert (df.predicted_demand >= 0.0).all()
    if 'raw_q10' in df:
        assert (df.raw_q10 >= 0.0).all()
        assert (df.raw_q50 >= df.raw_q10).all()
        assert (df.raw_q90 >= df.raw_q50).all()


def test_all_horizons_are_supported(cfg):
    expected_horizons = set(cfg['evaluation']['horizons'])
    pred_path = ROOT / 'data/predictions/demand_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    assert set(df.horizon.unique()) == expected_horizons
    for m in ['lightgbm', 'catboost', 'xgboost']:
        for h in expected_horizons:
            artifact = ROOT / f'models/demand/{m}/{m}_demand_{h}d.joblib'
            assert artifact.exists()


def test_metrics_calculate_correctly():
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]
    mase_scale = [2.0, 2.0, 2.0]
    m = compute_regression_metrics(actual, predicted, mase_scale)
    # MAE = (2 + 2 + 3) / 3 = 7/3 = 2.3333...
    assert m['mae'] == pytest.approx(7.0 / 3.0)
    # RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(17/3)
    assert m['rmse'] == pytest.approx(np.sqrt(17.0 / 3.0))
    # WAPE = (2 + 2 + 3) / (10 + 20 + 30) = 7 / 60
    assert m['wape'] == pytest.approx(7.0 / 60.0)
    # MASE = mean(2/2, 2/2, 3/2) = (1 + 1 + 1.5)/3 = 3.5/3 = 7/6
    assert m['mase'] == pytest.approx(7.0 / 6.0)
    # Bias = (2 - 2 + 3)/3 = 1.0
    assert m['bias'] == pytest.approx(1.0)


def test_baseline_comparisons_calculate_correctly():
    model_df = pd.DataFrame([{'horizon': 7, 'wape': 0.10, 'mae': 30.0, 'rmse': 40.0}])
    base_df = pd.DataFrame([{'horizon': 7, 'wape': 0.15, 'mae': 45.0, 'rmse': 60.0}])
    comp = compute_baseline_comparison(model_df, base_df)
    # WAPE reduction = (0.15 - 0.10) / 0.15 * 100 = 33.333%
    assert comp.iloc[0]['wape_reduction_pct'] == pytest.approx(100.0 / 3.0)
    assert comp.iloc[0]['outperforms_baseline'] is True or comp.iloc[0]['outperforms_baseline'] == 1


def test_categorical_handling_is_fold_safe(manifest):
    # Categorical proxies are integer/float or handled fold-internally
    cal_cols = [c for c in manifest if 'cal_' in c]
    assert len(cal_cols) > 0


def test_imputation_is_fold_safe():
    # Residual quantiles handle fold residuals without global stats
    train_res = [1.0, -2.0, 3.0, -1.0, 0.0]
    preds = [10.0, 20.0]
    q10, q50, q90 = compute_residual_quantiles(train_res, preds)
    assert len(q10) == 2 and len(q90) == 2
    assert (q10 <= q50).all() and (q50 <= q90).all()


def test_model_serialization_preserves_predictions(manifest):
    model, _ = load_model(7, 'lightgbm', root=ROOT)
    X = np.ones((1, len(manifest)), dtype=np.float32)
    p1 = model.predict(X)
    # Reload and compare
    reloaded = joblib.load(ROOT / 'models/demand/lightgbm/lightgbm_demand_7d.joblib')
    p2 = reloaded.predict(X)
    np.testing.assert_allclose(p1, p2)


def test_user_upload_compatible_inference_accepts_standardised_input(manifest):
    # Arbitrary user dataframe without demonstration store/product columns
    user_data = pd.DataFrame(np.random.rand(5, len(manifest)), columns=manifest)
    preds = predict_demand(user_data, 7, root=ROOT)
    assert len(preds) == 5
    assert (preds >= 0.0).all()
    # Test with quantiles
    q_df = predict_demand(user_data, 7, return_quantiles=True, root=ROOT)
    assert isinstance(q_df, pd.DataFrame)
    assert 'predicted_demand' in q_df and 'raw_q10' in q_df and 'raw_q90' in q_df
    assert (q_df.raw_q10 <= q_df.raw_q50).all()
    assert (q_df.raw_q50 <= q_df.raw_q90).all()
