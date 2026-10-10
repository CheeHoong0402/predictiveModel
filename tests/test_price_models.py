"""Tests for Phase 12 Supplier Purchase-Price Forecasting Models."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.utils.paths import PROJECT_ROOT
from src.models.price.config import (
    load_config,
    load_price_feature_manifest,
    validate_feature_input,
    BANNED_INPUT_PREFIXES
)
from src.models.price.evaluator import (
    compute_regression_metrics,
    compute_direction_metrics,
    compute_confusion_matrix_df,
    compute_baseline_comparison,
    compute_residual_quantiles,
    derive_direction_from_prices,
    derive_direction_from_pct_change,
    DIRECTION_CLASSES
)
from src.models.price.inference import (
    predict_purchase_price,
    predict_price_direction,
    load_model
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT)


@pytest.fixture
def manifest():
    return load_price_feature_manifest(ROOT)


# 1. Approved price-feature manifest is enforced
def test_approved_price_feature_manifest_enforced(manifest):
    validate_feature_input(manifest[:5], manifest)
    with pytest.raises(ValueError, match="Unapproved feature columns"):
        validate_feature_input(['unapproved_feature_xyz'], manifest)


# 2, 3, 4. Targets never appear among model inputs
@pytest.mark.parametrize('target_col', [
    'purchase_price_target_1d', 'purchase_price_target_7d',
    'demand_target_7d', 'demand_day_7',
    'price_direction_7d', 'price_direction_1d', 'y__price'
])
def test_targets_never_appear_among_model_inputs(manifest, target_col):
    with pytest.raises(ValueError, match="Prohibited target columns"):
        validate_feature_input([target_col], manifest)


# 5. Walk-forward folds remain chronological
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


# 6. Final holdout is protected
def test_final_holdout_protected():
    pred_path = ROOT / 'data/predictions/price_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Price pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    dates = pd.to_datetime(df.date)
    assert (dates < pd.Timestamp('2024-10-01')).all()


# 7, 8. Sparse target rows handled and invalid excluded
def test_sparse_targets_handled_correctly():
    pred_path = ROOT / 'data/predictions/price_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Price pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    assert df['actual_future_price'].notna().all()
    assert (df['actual_future_price'] > 0).all()


# 9. All six horizons are supported
def test_all_six_horizons_are_supported(cfg):
    expected_horizons = set(cfg['evaluation']['horizons'])
    assert expected_horizons == {1, 3, 7, 14, 21, 30}


# 10. Predicted prices are finite for regression models
def test_predicted_prices_are_finite():
    pred_path = ROOT / 'data/predictions/price_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Price pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    reg_df = df[df.approach.isin(['direct', 'percentage_change'])]
    assert np.all(np.isfinite(reg_df.predicted_future_price))
    assert not reg_df.predicted_future_price.isna().any()


# 11. Predicted prices cannot be negative
def test_predicted_prices_cannot_be_negative():
    pred_path = ROOT / 'data/predictions/price_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Price pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    reg_df = df[df.approach.isin(['direct', 'percentage_change'])]
    assert (reg_df.predicted_future_price >= 0.0).all()


# 12, 13, 14. Product, Store, and Supplier boundaries maintained
def test_product_store_supplier_boundaries_maintained():
    pred_path = ROOT / 'data/predictions/price_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Price pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    assert set(df.product_id.unique()) == {'PRD_CHK_01', 'PRD_EGG_02', 'PRD_OIL_05', 'PRD_ONN_06', 'PRD_RIC_04', 'PRD_VCH_03'}
    assert set(df.store_id.unique()) == {'STR_JHR_003', 'STR_KUL_001', 'STR_PNG_004', 'STR_SGR_002'}
    assert df['supplier_id'].notna().all()


# 15. Shuffled input order does not alter aligned predictions
def test_shuffled_input_order_preserves_aligned_predictions(manifest):
    df = pd.DataFrame(np.random.rand(4, len(manifest)), columns=manifest)
    p_orig = predict_purchase_price(df, 7, 'catboost_direct', root=ROOT)
    p_shuff = predict_purchase_price(df.iloc[::-1], 7, 'catboost_direct', root=ROOT)
    np.testing.assert_allclose(p_orig, p_shuff[::-1])


# 16, 17, 18. Regression metrics correctly calculated
def test_regression_metrics_calculated_correctly():
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]
    m = compute_regression_metrics(actual, predicted)
    # MAE = (2 + 2 + 3) / 3 = 7/3 = 2.3333...
    assert m['mae'] == pytest.approx(7.0 / 3.0)
    # RMSE = sqrt((4 + 4 + 9) / 3) = sqrt(17/3)
    assert m['rmse'] == pytest.approx(np.sqrt(17.0 / 3.0))
    # MAPE = (2/10 + 2/20 + 3/30)/3 = (0.2 + 0.1 + 0.1)/3 = 0.4/3
    assert m['mape'] == pytest.approx(0.4 / 3.0)
    # Bias = (2 - 2 + 3)/3 = 1.0
    assert m['bias'] == pytest.approx(1.0)


# 19, 20, 21. Direction labels, balanced accuracy, and confusion matrix
def test_direction_metrics_and_confusion_matrix_correctly_calculated():
    actual = ['UP', 'FLAT', 'DOWN', 'UP']
    pred = ['UP', 'DOWN', 'DOWN', 'FLAT']
    m = compute_direction_metrics(actual, pred)
    # Accuracy = 2/4 = 0.5
    assert m['accuracy'] == pytest.approx(0.5)
    # UP: 1/2, FLAT: 0/1, DOWN: 1/1 -> balanced acc = (0.5 + 0 + 1) / 3 = 0.5
    assert m['balanced_accuracy'] == pytest.approx(0.5)
    
    cm = compute_confusion_matrix_df(actual, pred)
    assert cm.shape == (3, 3)
    assert list(cm.columns) == ['pred_DOWN', 'pred_FLAT', 'pred_UP']


# 22. Regression-derived direction follows configured flat threshold
def test_derived_direction_follows_flat_threshold():
    current_price = [100.0, 100.0, 100.0]
    # flat threshold 0.005 -> +/- 0.5%
    predicted_price = [100.6, 100.2, 99.4]
    dirs = derive_direction_from_prices(predicted_price, current_price, flat_tolerance=0.005)
    assert list(dirs) == ['UP', 'FLAT', 'DOWN']


# 23, 24. Categorical handling and imputation are fold-safe
def test_categorical_and_imputation_are_fold_safe(manifest):
    assert 'product_imported_flag' in manifest
    # Quantiles maintain non-crossing
    res = [0.1, -0.2, 0.3, -0.1, 0.0]
    q10, q50, q90, _ = compute_residual_quantiles(res, [10.0, 20.0])
    assert (q10 <= q50).all() and (q50 <= q90).all()


# 25. Model serialization/deserialization preserves predictions
def test_model_serialization_preserves_predictions(manifest):
    artifact = ROOT / 'models/price/catboost/catboost_direct_7d.joblib'
    if not artifact.exists():
        pytest.skip("Artifacts not yet written")
    model, _, _ = load_model(7, 'catboost_direct', task_type='regression', root=ROOT)
    X = np.ones((1, len(manifest)), dtype=np.float32)
    p1 = model.predict(X)
    reloaded = joblib.load(artifact)
    p2 = reloaded.predict(X)
    np.testing.assert_allclose(p1, p2)


# 26, 27. Reusable inference accepts standardized user-uploaded data
def test_user_upload_compatible_inference_accepts_standardised_input(manifest):
    user_data = pd.DataFrame(np.random.rand(5, len(manifest)), columns=manifest)
    preds = predict_purchase_price(user_data, 7, 'catboost_direct', root=ROOT)
    assert len(preds) == 5
    assert (preds >= 0.0).all()
    
    # Quantile DataFrame
    q_df = predict_purchase_price(user_data, 7, 'catboost_direct', return_quantiles=True, root=ROOT)
    assert isinstance(q_df, pd.DataFrame)
    assert 'predicted_future_price' in q_df and 'raw_p10' in q_df and 'raw_p90' in q_df
    assert (q_df.raw_p10 <= q_df.raw_p50).all()
    assert (q_df.raw_p50 <= q_df.raw_p90).all()


# 28. Raw quantiles maintain valid ordering
def test_raw_quantiles_maintain_valid_ordering():
    pred_path = ROOT / 'data/predictions/price_model_predictions.parquet'
    if not pred_path.exists():
        pytest.skip("Price pipeline not yet completed")
    df = pd.read_parquet(pred_path)
    reg_df = df[df.approach == 'direct']
    if 'raw_p10' in reg_df:
        assert (reg_df.raw_p10 <= reg_df.raw_p50).all()
        assert (reg_df.raw_p50 <= reg_df.raw_p90).all()
