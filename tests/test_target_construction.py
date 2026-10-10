"""Phase 9 adversarial, calendar, supplier, immutability and pipeline tests."""

from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest

from src.targets.target_pipeline import load_config, construct_targets, run_target_pipeline
from src.targets.target_validation import validate_targets, validate_manifests, assemble_training_view, sha256, LEGACY_NAME_COLLISIONS
from src.targets.target_registry import create_registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT)


@pytest.fixture
def sequence():
    # T has demand 1000, so erroneously including the origin is very visible.
    return pd.DataFrame(dict(date=pd.date_range("2024-01-01", periods=41), store_id="S1", product_id="P1",
        supplier_id="SUP1", category="test", estimated_true_demand_training_safe=[1000.] + list(np.arange(1., 41.)),
        estimated_true_demand_retrospective=999999., observed_units_sold=8888., purchase_price=np.arange(100., 141.)))


@pytest.mark.parametrize("h,expected", [(1, 1), (3, 6), (7, 28), (14, 105), (21, 231), (30, 465)])
def test_exact_future_cumulative_and_direct(sequence, config, h, expected):
    _, targets = construct_targets(sequence, config)
    assert targets.loc[0, f"demand_target_{h}d"] == expected
    assert targets.loc[0, f"demand_day_{h}"] == h
    assert targets.loc[0, f"demand_target_{h}d_valid"]
    assert targets.loc[0, f"demand_target_{h}d_coverage"] == 1
    assert targets[f"demand_target_{h}d"].tail(h).isna().all()


def test_group_boundaries_and_shuffle(sequence, config):
    frames = []
    for store, product, scale in [("S1", "P1", 1), ("S2", "P1", 10), ("S1", "P2", 100)]:
        f = sequence.copy()
        f["store_id"], f["product_id"] = store, product
        f["estimated_true_demand_training_safe"] *= scale
        frames.append(f)
    data = pd.concat(frames, ignore_index=True)
    _, a = construct_targets(data, config)
    _, b = construct_targets(data.sample(frac=1, random_state=13), config)
    pd.testing.assert_frame_equal(a, b)
    assert sorted(a.loc[a.date == data.date.min(), "demand_target_3d"].tolist()) == [6, 60, 600]


def test_training_safe_not_retrospective_and_null_fallback(sequence, config):
    _, expected = construct_targets(sequence, config)
    sequence["estimated_true_demand_retrospective"] *= 12345
    _, actual = construct_targets(sequence, config)
    pd.testing.assert_frame_equal(expected, actual)
    sequence.loc[1, "estimated_true_demand_training_safe"] = np.nan
    sequence.loc[1, "observed_units_sold"] = 7
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "demand_target_1d"] == 7
    assert t.loc[0, "demand_target_1d_fallback_days"] == 1
    sequence.loc[1, "observed_units_sold"] = np.nan
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "demand_target_1d"])


def test_missing_source_column_uses_documented_fallback(sequence, config):
    _, t = construct_targets(sequence.drop(columns="estimated_true_demand_training_safe"), config)
    assert t.loc[0, "demand_target_3d"] == 3 * 8888


def test_missing_calendar_day_not_shifted_or_zeroed(sequence, config):
    _, t = construct_targets(sequence.drop(index=2), config)
    assert pd.isna(t.loc[0, "demand_target_3d"])
    assert t.loc[0, "demand_target_3d_days_available"] == 2
    assert t.loc[0, "demand_target_3d_coverage"] == pytest.approx(2 / 3)
    assert t.loc[0, "demand_day_3"] == 3
    assert t.loc[0, "demand_target_3d_invalid_reason"] == "MISSING_DATE"


@pytest.mark.parametrize("column,value,reason", [
    ("store_active", 0, "STORE_CLOSED"), ("store_closed", 1, "STORE_CLOSED"),
    ("store_open", 0, "STORE_CLOSED"), ("product_active", 0, "INACTIVE_PRODUCT"),
    ("product_active_in_store", 0, "INACTIVE_PRODUCT"),
    ("product_available", 0, "UNAVAILABLE_PRODUCT"),
    ("product_discontinued", 1, "DISCONTINUED_PRODUCT"),
    ("store_active", np.nan, "UNKNOWN_ACTIVITY")])
def test_activity_invalidates_horizon(sequence, config, column, value, reason):
    sequence[column] = 0.0 if column in ["store_closed", "product_discontinued"] else 1.0
    sequence.loc[2, column] = value
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "demand_target_1d_valid"]
    assert not t.loc[0, "demand_target_3d_valid"]
    assert reason in t.loc[0, "demand_target_3d_invalid_reason"]


@pytest.mark.parametrize("column", ["product_discontinued_date", "store_closure_date"])
def test_lifecycle_end_exclusive(sequence, config, column):
    sequence[column] = pd.Timestamp("2024-01-03")
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "demand_target_1d"] == 1
    assert pd.isna(t.loc[0, "demand_target_3d"])
    assert pd.isna(t.loc[1, "demand_target_1d"])


def test_genuine_zero_and_corrected_stockout_valid(sequence, config):
    sequence.loc[1, "estimated_true_demand_training_safe"] = 0
    sequence["stockout_flag"] = 1
    sequence["current_inventory"] = 0
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "demand_target_1d"] == 0
    assert t.loc[0, "demand_target_3d"] == 5


@pytest.mark.parametrize("value", [-1., np.inf, -np.inf])
def test_invalid_demand_is_not_replaced_by_fallback(sequence, config, value):
    sequence.loc[1, "estimated_true_demand_training_safe"] = value
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "demand_target_1d"])


@pytest.mark.parametrize("h", [1, 3, 7, 14, 21, 30])
def test_price_sequence_dates_changes_direction(sequence, config, h):
    _, t = construct_targets(sequence, config)
    assert t.loc[0, f"purchase_price_target_{h}d"] == 100 + h
    assert t.loc[0, f"purchase_price_target_{h}d_observation_date"] == sequence.loc[h, "date"]
    assert t.loc[0, f"purchase_price_pct_change_{h}d"] == pytest.approx(h / 100)
    assert t.loc[0, f"price_direction_{h}d"] == "UP"
    assert t[f"purchase_price_target_{h}d"].tail(h).isna().all()


def test_sparse_price_bounded_forward_tolerance(sequence, config):
    sequence.loc[1:4, "purchase_price"] = np.nan
    config["price_target"]["tolerance_days"] = 2
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "purchase_price_target_1d"])
    assert t.loc[0, "purchase_price_target_3d"] == 105
    assert t.loc[0, "purchase_price_target_3d_date_difference_days"] == 2
    assert t.loc[0, "purchase_price_target_3d_selection_method"] == "NEXT_WITHIN_TOLERANCE"
    # A closer observation before the requested target date is not forward-filled.
    config["price_target"]["tolerance_days"] = 0
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "purchase_price_target_3d"])


def test_supplier_boundary(sequence, config):
    sequence.loc[1:, "supplier_id"] = "SUP2"
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "purchase_price_target_1d"])
    assert t.loc[0, "demand_target_3d"] == 6


def test_missing_supplier_cannot_cross_match(sequence, config):
    sequence["supplier_id"] = None
    _, t = construct_targets(sequence, config)
    assert t.purchase_price_target_1d.isna().all()


def test_no_future_baseline_or_unbounded_carry(sequence, config):
    sequence.loc[:9, "purchase_price"] = np.nan
    _, t = construct_targets(sequence, config)
    assert t.loc[:9, "purchase_price_at_origin"].isna().all()
    assert pd.isna(t.loc[9, "purchase_price_pct_change_1d"])
    sequence.loc[0, "purchase_price"] = 100
    _, t = construct_targets(sequence, config)
    assert t.loc[7, "purchase_price_at_origin"] == 100
    assert pd.isna(t.loc[8, "purchase_price_at_origin"])


@pytest.mark.parametrize("value", [0., -1., np.inf, np.nan])
def test_invalid_origin_price_safe(sequence, config, value):
    sequence.loc[0, "purchase_price"] = value
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "purchase_price_target_1d"] == 101
    assert pd.isna(t.loc[0, "purchase_price_pct_change_1d"])
    assert pd.isna(t.loc[0, "price_direction_1d"])


@pytest.mark.parametrize("price,label", [(100.5, "FLAT"), (99.5, "FLAT"), (100.51, "UP"), (99.49, "DOWN"), (100., "FLAT")])
def test_flat_boundary_is_inclusive(sequence, config, price, label):
    sequence.loc[1, "purchase_price"] = price
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "price_direction_1d"] == label


def test_explicit_effective_interval_only_and_bounded(sequence, config):
    config["price_target"]["allow_effective_price_carry_forward"] = True
    with pytest.raises(ValueError, match="explicit start/end"):
        construct_targets(sequence, config)
    sequence["price_effective_from"] = sequence.date
    sequence["price_effective_until"] = sequence.date + pd.Timedelta(days=10)
    sequence.loc[3:, "purchase_price"] = np.nan
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "purchase_price_target_7d"] == 102
    assert t.loc[0, "purchase_price_target_7d_selection_method"] == "EFFECTIVE_INTERVAL"
    assert pd.isna(t.loc[0, "purchase_price_target_14d"])
    # A quote already observed on/before origin is not a future observation.
    assert pd.isna(t.loc[2, "purchase_price_target_3d"])
    sequence["price_effective_until"] = sequence.date
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "purchase_price_target_7d"])


def test_extreme_events_retained(sequence, config):
    sequence.loc[1, "purchase_price"] = 1000
    sequence.loc[1, "estimated_true_demand_training_safe"] = 1e6
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "purchase_price_target_1d"] == 1000
    assert t.loc[0, "price_change_1d_extreme"]
    assert t.loc[0, "demand_target_1d"] == 1e6


def test_duplicate_key_rejected_and_source_not_mutated(sequence, config):
    original = sequence.copy(deep=True)
    _, t = construct_targets(sequence, config)
    pd.testing.assert_frame_equal(sequence, original)
    assert not t.duplicated(["date", "store_id", "product_id"]).any()
    with pytest.raises(ValueError, match="duplicate"):
        construct_targets(pd.concat([sequence, sequence.iloc[[0]]]), config)


def test_temporal_assertions_detect_tampering(sequence, config):
    _, t = construct_targets(sequence, config)
    t.loc[0, "purchase_price_target_1d_observation_date"] = sequence.loc[0, "date"]
    with pytest.raises(ValueError, match="strictly after"):
        validate_targets(t, config)
    _, t = construct_targets(sequence, config)
    t.loc[0, "feature_information_upper_bound"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="after origin"):
        validate_targets(t, config)


def test_no_target_identities_in_approved_manifests_and_safe_join(config):
    features = pd.read_parquet(ROOT / "data/processed/model_features.parquet")
    registry = create_registry(config)
    collisions = validate_manifests(ROOT, features.columns, registry.target_name)
    assert set(collisions.name) == LEGACY_NAME_COLLISIONS
    manifest = pd.read_csv(ROOT / "reports/leakage/demand_features_temporally_safe.csv")
    qualified = {"model_features." + n for n in manifest.feature_name}
    assert not qualified.intersection(registry.qualified_identity)
    source = pd.read_parquet(ROOT / "data/processed/master_daily_demand_corrected.parquet")
    _, targets = construct_targets(source, config)
    joined = assemble_training_view(features, targets, manifest, "purchase_price_pct_change_1d", "price_change_1d_valid")
    changed = targets.copy()
    changed["purchase_price_pct_change_1d"] = changed.purchase_price_pct_change_1d * 1e9
    alternate = assemble_training_view(features, changed, manifest, "purchase_price_pct_change_1d", "price_change_1d_valid")
    xcols = [c for c in joined if c.startswith("x__")]
    pd.testing.assert_frame_equal(joined[xcols], alternate[xcols])
    assert not joined["y__purchase_price_pct_change_1d"].equals(alternate["y__purchase_price_pct_change_1d"])
    target_metadata = set(targets.columns) - set(registry.target_name) - {"date", "store_id", "product_id", "supplier_id"}
    assert not target_metadata.intersection(c.removeprefix("x__") for c in xcols)
    assert len(xcols) == len(manifest)


def test_pipeline_outputs_and_upstream_byte_immutability(tmp_path, config):
    # Full integration in an isolated copy; protect the canonical feature file,
    # manifests and previous outputs, not just schema/row counts.
    for directory in ["config", "reports/leakage", "reports/features", "data/processed"]:
        shutil.copytree(ROOT / directory, tmp_path / directory)
    protected = [p for p in tmp_path.rglob("*") if p.is_file() and "targets" not in p.parts and p.name != "forecast_targets.parquet"]
    before = {p: sha256(p) for p in protected}
    targets = run_target_pipeline(tmp_path)
    assert all(sha256(p) == value for p, value in before.items())
    required = ["target_construction_summary.md", "target_registry.csv", "target_lineage.csv", "target_coverage.csv",
                "target_missingness.csv", "demand_target_distribution.csv", "price_target_distribution.csv",
                "target_outliers.csv", "manual_target_checks.csv", "training_row_masks.parquet",
                "price_direction_distribution.csv", "intermittent_demand_analysis.csv"]
    assert all((tmp_path / "reports/targets" / name).exists() for name in required)
    assert len(targets) == 35064
    proof = pd.read_csv(tmp_path / "reports/targets/manual_target_checks.csv")
    assert proof.passed.all()
    assert set(proof.horizon) == set(config["horizons"])
    masks = pd.read_parquet(tmp_path / "reports/targets/training_row_masks.parquet")
    for h in config["horizons"]:
        assert masks[f"demand_{h}d_training_rows"].sum() == 35064 - 24 * h
        assert masks[f"price_{h}d_training_rows"].sum() == 35064 - 24 * h
    saved = pd.read_parquet(tmp_path / "data/processed/forecast_targets.parquet")
    pd.testing.assert_frame_equal(saved, targets)


def test_config_rejects_partial_canonical_targets(tmp_path):
    shutil.copytree(ROOT / "config", tmp_path / "config")
    path = tmp_path / "config/target_construction.yaml"
    path.write_text(path.read_text().replace("minimum_coverage: 1.0", "minimum_coverage: 0.5"))
    with pytest.raises(ValueError, match="partial sums"):
        load_config(tmp_path)


def test_nondefault_flat_tolerance_and_price_validity_flag(sequence, config):
    config["price_target"]["flat_tolerance"] = .02
    _, t = construct_targets(sequence, config)
    assert t.loc[0, "price_direction_1d"] == "FLAT"
    sequence["price_observation_valid"] = True
    sequence.loc[1, "price_observation_valid"] = False
    _, t = construct_targets(sequence, config)
    assert pd.isna(t.loc[0, "purchase_price_target_1d"])


def test_input_calendar_dates_and_null_keys_rejected(sequence, config):
    sequence.loc[0, "date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="calendar dates"):
        construct_targets(sequence, config)
    sequence.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match="primary key"):
        construct_targets(sequence, config)


def test_reproducible_pipeline_tables(tmp_path):
    for directory in ["config", "reports/leakage", "reports/features", "data/processed"]:
        shutil.copytree(ROOT / directory, tmp_path / directory)
    first = run_target_pipeline(tmp_path)
    reports = tmp_path / "reports/targets"
    before = {p.name: sha256(p) for p in reports.glob("*.csv")}
    second = run_target_pipeline(tmp_path)
    pd.testing.assert_frame_equal(first, second)
    assert all(sha256(reports / name) == digest for name, digest in before.items())
