"""Reproduce Phase 9 artifacts: python -m src.targets.target_pipeline.

No models, scaling, random splits, or writes to preceding phase artifacts.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT
from .demand_targets import KEYS, SERIES, prepare_source, build_demand_targets
from .price_targets import build_price_targets
from .target_registry import create_registry, create_lineage
from .target_validation import sha256, validate_manifests, validate_targets


def load_config(root=PROJECT_ROOT):
    """Single Phase 9 loader; reuse the authoritative Phase 7 timing/sources."""
    root = Path(root)
    with (root / "config/target_construction.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with (root / "config/feature_engineering.yaml").open(encoding="utf-8") as f:
        existing = yaml.safe_load(f)
    timing = existing["timing_convention"]
    if timing["forecast_generation_point"] != "end_of_day_T" or timing["time_zone"] != "Asia/Kuala_Lumpur":
        raise ValueError("Unsupported forecast origin; review target contract")
    cfg["timing_convention"] = timing
    cfg["demand_source"] = existing["demand_features"]["target_source_column"]
    cfg["demand_fallback"] = existing["demand_features"]["fallback_source_column"]
    if cfg["demand_source"] != "estimated_true_demand_training_safe" or cfg["demand_fallback"] != "observed_units_sold":
        raise ValueError("Review changed upstream demand source contract")
    if cfg["horizons"] != [1, 3, 7, 14, 21, 30]:
        raise ValueError("Canonical horizons must be [1, 3, 7, 14, 21, 30]")
    if cfg["demand_target"]["minimum_coverage"] != 1.0:
        raise ValueError("Canonical cumulative labels require minimum_coverage=1.0; partial sums are not complete labels")
    if cfg["demand_target"]["missing_activity_policy"] != "observed_records_assumed_active":
        raise ValueError("Unknown activity policy")
    for name in ["tolerance_days", "maximum_carry_forward_days", "current_price_maximum_age_days"]:
        value = cfg["price_target"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ["flat_tolerance", "extreme_change_threshold"]:
        if not np.isfinite(cfg["price_target"][name]) or cfg["price_target"][name] < 0:
            raise ValueError(f"Invalid {name}")
    if not np.isfinite(cfg["outliers"]["iqr_multiplier"]) or cfg["outliers"]["iqr_multiplier"] <= 0:
        raise ValueError("iqr_multiplier must be positive")
    return cfg


def construct_targets(frame, config):
    source = prepare_source(frame, config)
    if source.empty:
        raise ValueError("No source observations")
    demand = build_demand_targets(source, config)
    price = build_price_targets(source, config)
    targets = demand.merge(price, on=KEYS, validate="one_to_one").sort_values(SERIES + ["date"]).reset_index(drop=True).copy()
    # This is an audited upper bound, not a fabricated measured timestamp.
    targets["feature_information_upper_bound"] = targets.date
    validate_targets(targets, config)
    return source, targets


def distribution(series):
    x = series.dropna()
    result = dict(count=len(x), missing_count=int(series.isna().sum()), mean=x.mean(), median=x.median(),
                  standard_deviation=x.std(), minimum=x.min(), maximum=x.max())
    result.update({f"p{p}": x.quantile(p / 100) for p in [10, 25, 75, 90, 95, 99]})
    return result


def manual_checks(source, targets, config):
    eligible = targets[targets.demand_target_30d_valid]
    chosen = eligible.sample(n=min(len(eligible), config["manual_checks"]["sample_count"]), random_state=config["manual_checks"]["seed"])
    rows = []
    lookup = source.set_index(SERIES + ["date"])
    for _, origin in chosen.iterrows():
        for h in config["horizons"]:
            dates = pd.date_range(origin.date + pd.Timedelta(days=1), periods=h)
            index = pd.MultiIndex.from_arrays([[origin.store_id] * h, [origin.product_id] * h, dates], names=SERIES + ["date"])
            contributing = lookup.reindex(index)
            values = contributing._valid_demand.to_numpy(float)
            expected = values.sum()
            actual = origin[f"demand_target_{h}d"]
            if not np.isclose(expected, actual):
                raise ValueError("Independent manual demand reconstruction failed")
            rows.append(dict(kind="demand", **{k: origin[k] for k in KEYS}, horizon=h,
                future_dates=json.dumps(dates.strftime("%Y-%m-%d").tolist()), demand_values=json.dumps(values.tolist()),
                independently_reconstructed=expected, target=actual, passed=bool(np.isclose(expected, actual))))
            p = f"purchase_price_target_{h}d"
            obs_date = origin[p + "_observation_date"]
            obs = source[(source.store_id == origin.store_id) & (source.product_id == origin.product_id)
                         & (source.supplier_id == origin.supplier_id) & (source.date == obs_date)]
            reconstructed = obs.purchase_price.iloc[0] if len(obs) else np.nan
            passed = (pd.isna(reconstructed) and pd.isna(origin[p])) or np.isclose(reconstructed, origin[p])
            if not passed:
                raise ValueError("Independent manual price reconstruction failed")
            rows.append(dict(kind="price", **{k: origin[k] for k in KEYS}, supplier_id=origin.supplier_id, horizon=h,
                current_purchase_price=origin.purchase_price_at_origin,
                current_price_observation_date=origin.purchase_price_origin_observation_date,
                requested_date=origin[p + "_requested_date"], observation_date=obs_date,
                date_difference_days=origin[p + "_date_difference_days"], selection_method=origin[p + "_selection_method"],
                independently_reconstructed=reconstructed, target=origin[p],
                percentage_change=origin[f"purchase_price_pct_change_{h}d"], direction=origin[f"price_direction_{h}d"], passed=passed))
    return pd.DataFrame(rows)


def intermittent_analysis(source):
    rows = []
    for key, group in source.groupby(SERIES):
        values = group._valid_demand.dropna()
        positive = group.loc[group._valid_demand.gt(0)]
        intervals = positive.date.diff().dt.days.dropna()
        zero_pct = 100 * values.eq(0).mean()
        adi = len(values) / len(positive) if len(positive) else np.nan
        cv2 = (positive._valid_demand.std() / positive._valid_demand.mean()) ** 2 if len(positive) > 1 else np.nan
        rows.append(dict(zip(SERIES, key), observed_days=len(values), nonzero_days=len(positive),
            zero_demand_pct=zero_pct, average_demand_interval_observations=adi,
            average_calendar_interval_nonzero=intervals.mean(), demand_standard_deviation=values.std(),
            positive_demand_cv_squared=cv2, intermittent=bool(adi >= 1.32) if np.isfinite(adi) else True,
            sparse=bool(zero_pct >= 80 or len(positive) < 10)))
    return pd.DataFrame(rows)


def create_reports(source, targets, registry, config, report_dir):
    joined = targets.merge(source[KEYS + ["category"]] if "category" in source else source[KEYS].assign(category="UNKNOWN"), on=KEYS, validate="one_to_one")
    coverage, missing, demand_stats, price_stats, directions, outliers = [], [], [], [], [], []
    training = targets[KEYS].copy()
    for _, definition in registry.iterrows():
        name, h = definition.target_name, int(definition.horizon)
        missing.append(dict(target_name=name, horizon=h, total_rows=len(targets), valid_count=int(targets[name].notna().sum()),
                            missing_count=int(targets[name].isna().sum()), missing_pct=100 * targets[name].isna().mean()))
    for h in config["horizons"]:
        d, p = f"demand_target_{h}d", f"purchase_price_target_{h}d"
        training[f"demand_{h}d_training_rows"] = targets[d + "_valid"]
        training[f"price_{h}d_training_rows"] = targets[p + "_valid"]
        training[f"price_change_{h}d_training_rows"] = targets[f"price_change_{h}d_valid"]
        coverage.append(dict(horizon=h, total_rows=len(targets), demand_valid=int(targets[d + "_valid"].sum()),
            demand_missing=int(targets[d].isna().sum()), mean_demand_coverage=targets[d + "_coverage"].mean(),
            future_days_required=h, future_valid_days_total=int(targets[d + "_days_available"].sum()),
            price_valid=int(targets[p + "_valid"].sum()), price_missing=int(targets[p].isna().sum()),
            price_change_valid=int(targets[f"price_change_{h}d_valid"].sum())))
        demand_stats.append(dict(target_name=d, horizon=h, scope="overall", store_id="ALL", category="ALL", **distribution(targets[d])))
        for scope, group_cols in [("store", ["store_id"]), ("category", ["category"]), ("store_category", ["store_id", "category"])]:
            for key, group in joined.groupby(group_cols):
                key = key if isinstance(key, tuple) else (key,)
                labels = dict(store_id="ALL", category="ALL")
                labels.update(dict(zip(group_cols, key)))
                demand_stats.append(dict(target_name=d, horizon=h, scope=scope, **labels, **distribution(group[d])))
        for name in [p, f"purchase_price_pct_change_{h}d"]:
            price_stats.append(dict(target_name=name, horizon=h, **distribution(targets[name])))
        for label, count in targets[f"price_direction_{h}d"].fillna("MISSING").value_counts().items():
            directions.append(dict(horizon=h, direction=label, count=int(count), pct_all_rows=100 * count / len(targets)))
        # Descriptive audit thresholds only, not fitted model transformations.
        for _, group in joined.groupby(SERIES):
            q1, q3 = group[d].quantile([0.25, 0.75])
            threshold = q3 + config["outliers"]["iqr_multiplier"] * (q3 - q1)
            for _, row in group[group[d] > threshold].iterrows():
                outliers.append(dict(**{k: row[k] for k in KEYS}, target_name=d, value=row[d],
                                     threshold=threshold, reason="ABOVE_SERIES_Q3_PLUS_CONFIGURED_IQR", action="RETAIN_REVIEW"))
        for _, row in joined[joined[f"price_change_{h}d_extreme"]].iterrows():
            outliers.append(dict(**{k: row[k] for k in KEYS}, target_name=f"purchase_price_pct_change_{h}d",
                value=row[f"purchase_price_pct_change_{h}d"], threshold=config["price_target"]["extreme_change_threshold"],
                reason="EXTREME_ABSOLUTE_PRICE_CHANGE", action="RETAIN_REVIEW"))
    outputs = {
        "target_coverage.csv": pd.DataFrame(coverage), "target_missingness.csv": pd.DataFrame(missing),
        "demand_target_distribution.csv": pd.DataFrame(demand_stats), "price_target_distribution.csv": pd.DataFrame(price_stats),
        "price_direction_distribution.csv": pd.DataFrame(directions),
        "target_outliers.csv": pd.DataFrame(outliers, columns=KEYS + ["target_name", "value", "threshold", "reason", "action"]),
        "manual_target_checks.csv": manual_checks(source, targets, config),
        "intermittent_demand_analysis.csv": intermittent_analysis(source),
        "target_registry.csv": registry, "target_lineage.csv": create_lineage(registry, config),
    }
    for name, data in outputs.items():
        data.to_csv(report_dir / name, index=False)
    training.to_parquet(report_dir / "training_row_masks.parquet", index=False)
    # Individual key indexes keep validity horizon-specific and do not split time.
    indexes = report_dir / "training_indexes"
    indexes.mkdir(exist_ok=True)
    for name in training.columns.difference(KEYS):
        training.loc[training[name], KEYS].to_parquet(indexes / (name + ".parquet"), index=False)
    return outputs


def run_target_pipeline(root=PROJECT_ROOT):
    root = Path(root)
    config = load_config(root)
    features_path = root / "data/processed/model_features.parquet"
    source_path = root / "data/processed/master_daily_demand_corrected.parquet"
    protected = [source_path, root / "data/processed/master_daily.parquet", features_path]
    protected += sorted((root / "reports/leakage").glob("*")) + sorted((root / "reports/features").glob("*"))
    protected += sorted((root / "config").glob("*.yaml"))
    before = {str(p.relative_to(root)): sha256(p) for p in protected if p.is_file()}
    features = pd.read_parquet(features_path)
    registry = create_registry(config)
    collisions = validate_manifests(root, features.columns, registry.target_name)
    source, targets = construct_targets(pd.read_parquet(source_path), config)
    feature_keys = features[KEYS].copy()
    feature_keys["date"] = pd.to_datetime(feature_keys.date)
    if feature_keys.duplicated(KEYS).any() or not feature_keys.merge(targets[KEYS], on=KEYS, how="outer", indicator=True)._merge.eq("both").all():
        raise ValueError("Source and feature primary keys do not match exactly")
    report_dir = root / "reports/targets"
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs = create_reports(source, targets, registry, config, report_dir)
    collisions.to_csv(report_dir / "feature_target_name_collisions.csv", index=False)
    targets.to_parquet(root / "data/processed/forecast_targets.parquet", index=False)
    integrity = [dict(path=name, sha256_before=digest, sha256_after=sha256(root / name)) for name, digest in before.items()]
    if any(row["sha256_before"] != row["sha256_after"] for row in integrity):
        raise RuntimeError("Protected upstream artifact changed")
    pd.DataFrame(integrity).to_csv(report_dir / "upstream_integrity.csv", index=False)
    (report_dir / "temporal_assertions.json").write_text(json.dumps(validate_targets(targets, config), indent=2), encoding="utf-8")
    coverage = outputs["target_coverage.csv"]
    demand_overall = outputs["demand_target_distribution.csv"].query("scope == 'overall'")
    activity_columns = [c for c in ["store_active", "store_open", "store_closed", "product_active", "product_active_in_store", "product_available", "product_discontinued", "product_discontinued_date", "store_closure_date"] if c in source]
    gap = source.groupby(SERIES + ["supplier_id"]).date.diff().dt.days
    summary = f"""# Phase 9 — Forecast Target Construction

Generated {len(targets):,} unique date/store/product rows. No model training, random splits, optimisation or target scaling.

## Forecast and observation contract
Origin: end_of_day_T, 23:59:59 Asia/Kuala_Lumpur, inherited from config/feature_engineering.yaml.
Future demand includes calendar days T+1 through T+H and never T. Cumulative targets require 100% valid coverage.
Auxiliary demand_day_H is demand on exactly T+H, for later direct-vs-cumulative evaluation; inventory targets are cumulative.
Demand source: {config['demand_source']}; null-only fallback: {config['demand_fallback']} ({int(source._fallback.sum())} source rows).
If both are missing, the target is missing. Phase 7's final zero fallback is deliberately not used to fabricate target observations.
Retrospective demand is never read by target calculations. Demand retains units, prices MYR, changes decimal ratios.

## Price policy and data evidence
Input price observations: {int(source.purchase_price.notna().sum()):,}/{len(source):,}; observed row gap median/max: {gap.median()}/{gap.max()} calendar days.
Daily supplier price records include non-purchase days and are not filtered by purchase_quantity.
Default tolerance: {config['price_target']['tolerance_days']} days (exact-day labels supported by complete daily source).
Configurable fallback selects the first observed price in [T+H, T+H+tolerance], strictly after T and within the origin supplier's series.
Effective carry is {config['price_target']['allow_effective_price_carry_forward']}; if enabled, explicit price_effective_from/until metadata and an observation after T are required; maximum age {config['price_target']['maximum_carry_forward_days']} days.
There is no ordinary target forward-fill. Origin baseline is the last valid same-supplier observation at/before T, at most {config['price_target']['current_price_maximum_age_days']} days old.
FLAT includes changes within +/-{config['price_target']['flat_tolerance']:.2%}; UP/DOWN are outside that band (floating-point comparison epsilon 1e-12).

## Lifecycle and closure limits
Available explicit activity columns: {activity_columns or 'none'}.
Earlier exports omit the skeleton activity flags, and source store/product masters have no dated closure/discontinuation records.
For this dataset, observed operational rows are assumed active, as in Phase 5; this is an explicit assumption, not verified store-opening evidence.
Missing dates stay missing; no lifecycle is inferred from future disappearance. Supplied activity, availability, closure and discontinuation flags/dates are enforced.
Null supplied flags mean unknown activity. Genuine observed zeros remain valid; unavailable products, closure, discontinuation, and missing observations are invalidated distinctly.
Stockout-corrected demand remains eligible; zero inventory alone does not mean discontinued.

## Coverage by horizon
```
{coverage.to_string(index=False)}
```

## Demand distributions (overall)
```
{demand_overall.to_string(index=False)}
```
Full distribution files include store, category and store/category breakdowns; price levels and percentage changes have the same quantiles.

## Separation and temporal safety
Canonical feature table and Phase 8 manifests are byte-for-byte unchanged (upstream_integrity.csv).
All-row temporal assertions pass, with feature information bounded by T using approved Phase 8 metadata.
This bound inherits the Phase 8 audit; per-feature physical source timestamps are not available and are not invented.
All chosen price observation dates are > T; all valid demand windows are T+1..T+H.

Three requested target names already occur in approved historical features: purchase_price_pct_change_1d, _7d and _30d.
It is impossible to make the literal name sets disjoint while preserving both the requested target names and authoritative manifests.
The historical features measure P(T)/P(T-H)-1; targets measure P(T+H)/P(T)-1. They remain separate, qualified dataset identities.
feature_target_name_collisions.csv documents this compatibility exception. assemble_training_view selects X only from canonical model_features and uses x__/y__ aliases.
Never select features from a bare merged target/feature frame. Target coverage, validity, observation dates, outlier flags and masks are audit metadata, not predictors.

## Review and proof artifacts
Outlier flags: {len(outputs['target_outliers.csv']):,} target observations; retained without clipping/removal.
Demand flags exceed within-series Q3 + {config['outliers']['iqr_multiplier']} IQR; price flags exceed absolute {config['price_target']['extreme_change_threshold']:.0%} change.
These are full-history descriptive audit thresholds, never model inputs or fitted preprocessing.
Intermittent/sparse series: {int(outputs['intermittent_demand_analysis.csv'].intermittent.sum())}/{int(outputs['intermittent_demand_analysis.csv']['sparse'].sum())}.
manual_target_checks.csv records exact contributing dates/values and independently reconstructed totals, plus selected price dates, baseline, changes and labels.
Seed {config['manual_checks']['seed']} samples observations for audit only, not train/test partitions.
Horizon-specific key indexes and masks are under reports/targets; price-change eligibility is separate from price-level eligibility.

## Reproduction
From the project folder with its existing environment: `python -m src.targets.target_pipeline`.
Tests: `python -m pytest tests/test_target_construction.py tests/test_temporal_integrity.py tests/test_feature_engineering.py tests/test_demand_correction.py tests/test_master_daily_table.py`.
"""
    (report_dir / "target_construction_summary.md").write_text(summary, encoding="utf-8")
    print(coverage.to_string(index=False))
    print(f"Rows: {len(targets):,}; protected artifacts unchanged: {len(integrity)}; outlier flags: {len(outputs['target_outliers.csv'])}")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run_target_pipeline(args.root)


if __name__ == "__main__":
    main()
