"""Definitions and lineage for every model label, separate from audit columns."""

import pandas as pd


def create_registry(config):
    rows = []
    for h in config["horizons"]:
        specs = [
            (f"demand_target_{h}d", "Total corrected purchasing demand over next H calendar days", "units", "sum D(T+k), k=1..H", config["demand_source"], "regression", "All H future dates valid; no negative/missing demand; activity valid", "1.0"),
            (f"purchase_price_target_{h}d", "Supplier unit purchase price at procurement horizon", "MYR", "P(T+H), or configured bounded observation/effective interval", "purchase_price", "regression", "Positive finite supplier-matched future observation under configured policy", "one valid price"),
            (f"purchase_price_pct_change_{h}d", "Future supplier price change relative to origin", "decimal ratio", "(future price - origin price) / origin price", "purchase_price", "regression", "Valid future price and positive finite bounded-age origin price", "two valid prices"),
            (f"price_direction_{h}d", "Future supplier price movement", "UP / FLAT / DOWN", "UP if change > flat_tolerance; DOWN if < -flat_tolerance; otherwise FLAT (inclusive)", "purchase_price", "classification", "Valid percentage change", "two valid prices"),
        ]
        if config["demand_target"]["create_direct_targets"]:
            specs.append((f"demand_day_{h}", "Demand exactly H calendar days ahead (auxiliary)", "units", "D(T+H)", config["demand_source"], "regression", "Exact target date with valid demand and activity", "one valid day"))
        for name, meaning, unit, calculation, source, model, validity, coverage in specs:
            rows.append(dict(target_name=name, qualified_identity=f"forecast_targets.{name}", business_meaning=meaning,
                unit=unit, horizon=h, calculation=calculation, required_coverage=coverage,
                missingness_rule="Missing unless validity rule passes; never impute labels", target_source=source,
                model_type=model, target_validity_rule=validity, feature_eligible=False))
    return pd.DataFrame(rows)


def create_lineage(registry, config):
    out = registry[["target_name", "target_source", "horizon", "calculation"]].copy()
    out["source_artifact"] = "data/processed/master_daily_demand_corrected.parquet"
    out["output_artifact"] = "data/processed/forecast_targets.parquet"
    out["demand_fallback"] = config["demand_fallback"]
    out["retrospective_allowed"] = False
    out["forecast_origin"] = "end_of_day_T; 23:59:59 Asia/Kuala_Lumpur"
    return out
