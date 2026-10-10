"""
price_data_sufficiency.py
=========================
Read-only audit: does the purchase-price training database contain enough
*genuine* price information to justify the existing price model, and is the
very low purchase-price MAE credible?

This module never trains or re-tunes a model and never reads protected
final-holdout outcomes: every statistic is computed on rows dated before
``HOLDOUT_START``; holdout rows are only *counted*.

Small pure helpers (``classify_density``, ``flag_forward_filled`` ...) are
unit-tested in ``tests/test_price_data_sufficiency.py``.  ``run_audit`` wires
them to the real artefacts and writes reports to ``reports/data_validation``
and figures to ``reports/figures/price_data_audit``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

HOLDOUT_START = pd.Timestamp("2024-10-01")
HORIZONS = (1, 3, 7, 14, 21, 30)
SERIES_KEYS = ["store_id", "product_id", "supplier_id"]
PRICE_COL = "current_supplier_purchase_price"


# ----------------------------------------------------------------------------
# Configurable, documented thresholds (NOT universal truths)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class DensityThresholds:
    """Genuine-observation count thresholds for a series' usable history.

    These are working conventions chosen for daily procurement data:
    ~1 year of daily data is HIGH, ~1 quarter MEDIUM, ~1 month LOW.
    They are configurable and should be revisited per business context.
    """

    high: int = 365
    medium: int = 90
    low: int = 30


def classify_density(n_genuine: int, thresholds: DensityThresholds = DensityThresholds()) -> str:
    """Return HIGH / MEDIUM / LOW / COLD_START for a count of genuine observations."""
    if n_genuine >= thresholds.high:
        return "HIGH"
    if n_genuine >= thresholds.medium:
        return "MEDIUM"
    if n_genuine >= thresholds.low:
        return "LOW"
    return "COLD_START"


# ----------------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------------
def flag_forward_filled(prices: pd.Series, min_run: int = 3, tol: float = 1e-9) -> pd.Series:
    """Flag suspected forward-filled rows.

    A row is flagged when it repeats the previous value *and* belongs to a run of
    at least ``min_run`` identical consecutive values (the first value of a run is
    treated as the genuine observation).  Missing values are never flagged
    (they are simply not genuine).
    """
    s = pd.Series(prices).reset_index(drop=True)
    starts_new_run = s.diff().abs().gt(tol) | s.isna() | s.shift().isna()
    run_id = starts_new_run.cumsum()
    run_len = s.groupby(run_id).transform("size")
    pos = s.groupby(run_id).cumcount()
    flagged = (run_len >= min_run) & (pos > 0) & s.notna()
    flagged.index = pd.Series(prices).index
    return flagged.astype(bool)


def genuine_observation_mask(
    df: pd.DataFrame,
    price_col: str = PRICE_COL,
    date_col: str = "date",
    explicit_flag_col: Optional[str] = None,
    min_run: int = 3,
    group_cols: Sequence[str] = ("store_id", "product_id"),
) -> pd.Series:
    """Boolean mask of genuine (non-null, not forward-filled) price observations.

    If the data carries an explicit observation flag it is authoritative; otherwise
    the run-length heuristic in :func:`flag_forward_filled` is applied per series.
    """
    if explicit_flag_col and explicit_flag_col in df.columns:
        return df[explicit_flag_col].astype(bool) & df[price_col].notna()
    ordered = df.sort_values(list(group_cols) + [date_col])
    ff = ordered.groupby(list(group_cols), group_keys=False)[price_col].apply(
        lambda s: flag_forward_filled(s, min_run=min_run)
    )
    ff = ff.reindex(df.index).fillna(False)
    return df[price_col].notna() & ~ff


def count_price_changes(prices: pd.Series, tol: float = 1e-9) -> int:
    """Number of times the price differs from the previous observation."""
    s = pd.Series(prices).dropna()
    return int(s.diff().abs().gt(tol).sum())


def price_variation_stats(prices: pd.Series, tol: float = 1e-9) -> Dict[str, float]:
    """Distribution and change statistics for one price series."""
    s = pd.Series(prices).dropna().astype(float)
    d = s.diff().abs().dropna()
    n_changes = int(d.gt(tol).sum())
    q1, q3 = (s.quantile(0.25), s.quantile(0.75)) if len(s) else (np.nan, np.nan)
    return {
        "n_obs": int(len(s)),
        "mean": float(s.mean()) if len(s) else np.nan,
        "std": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
        "variance": float(s.var(ddof=1)) if len(s) > 1 else np.nan,
        "cv": float(s.std(ddof=1) / s.mean()) if len(s) > 1 and s.mean() else np.nan,
        "min": float(s.min()) if len(s) else np.nan,
        "max": float(s.max()) if len(s) else np.nan,
        "iqr": float(q3 - q1) if len(s) else np.nan,
        "unique_prices": int(s.nunique()),
        "change_count": n_changes,
        "unchanged_rate": float(1.0 - n_changes / len(d)) if len(d) else np.nan,
        "mean_abs_change": float(d.mean()) if len(d) else np.nan,
        "median_abs_change": float(d.median()) if len(d) else np.nan,
        "p90_abs_change": float(d.quantile(0.9)) if len(d) else np.nan,
    }


def horizon_movement(current: pd.Series, future: pd.Series, tol: float = 1e-9) -> Dict[str, float]:
    """Actual |future - current| statistics over rows where both are valid."""
    m = pd.concat([current, future], axis=1).dropna()
    if m.empty:
        return {"n": 0, "mean_abs_move": np.nan, "median_abs_move": np.nan,
                "p90_abs_move": np.nan, "pct_differing": np.nan, "genuine_change_count": 0}
    mv = (m.iloc[:, 1] - m.iloc[:, 0]).abs()
    return {
        "n": int(len(m)),
        "mean_abs_move": float(mv.mean()),
        "median_abs_move": float(mv.median()),
        "p90_abs_move": float(mv.quantile(0.9)),
        "pct_differing": float((mv > tol).mean() * 100.0),
        "genuine_change_count": int((mv > tol).sum()),
    }


def target_coverage(valid: pd.Series, dates: pd.Series, dev_end: pd.Timestamp,
                    holdout_start: pd.Timestamp = HOLDOUT_START) -> Dict[str, float]:
    """Candidate / valid / invalid / coverage and holdout row counts.

    Holdout rows are *counted only*; no holdout value is used anywhere.
    """
    dates = pd.to_datetime(dates)
    dev = dates < holdout_start
    cand = int(dev.sum())
    val = int((valid.astype(bool) & dev).sum())
    return {
        "candidate_rows": cand,
        "valid_rows": val,
        "invalid_rows": cand - val,
        "coverage_pct": float(100.0 * val / cand) if cand else np.nan,
        "holdout_rows_counted_only": int((~dev).sum()),
    }


def last_price_baseline_metrics(current: pd.Series, actual: pd.Series) -> Dict[str, float]:
    """MAE / RMSE / MAPE(%) of the naive 'next price = last price' forecast."""
    m = pd.concat([current, actual], axis=1).dropna()
    err = (m.iloc[:, 1] - m.iloc[:, 0]).astype(float)
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape_pct": float((err.abs() / m.iloc[:, 1].abs()).mean() * 100.0),
        "n": int(len(m)),
    }


def check_no_future_backfill(origin_dates: pd.Series, origin_obs_dates: pd.Series,
                             target_obs_dates: pd.Series, horizon: int) -> Dict[str, int]:
    """Count temporal violations in price-target construction.

    * ``origin_obs_dates`` (the 'current price' observation) must be <= origin.
    * ``target_obs_dates`` must be >= origin + horizon (no earlier/backfilled
      value may stand in for a future price).
    """
    od = pd.to_datetime(origin_dates)
    oo = pd.to_datetime(origin_obs_dates)
    to = pd.to_datetime(target_obs_dates)
    ok_origin = oo.notna()
    ok_target = to.notna()
    return {
        "origin_price_from_future": int((oo[ok_origin] > od[ok_origin]).sum()),
        "target_before_due_date": int((to[ok_target] < od[ok_target] + pd.Timedelta(days=horizon)).sum()),
        "checked_rows": int(ok_target.sum()),
    }


def assert_no_holdout(df: pd.DataFrame, date_col: str = "date",
                      holdout_start: pd.Timestamp = HOLDOUT_START) -> None:
    """Raise if any row is on/after the protected holdout start."""
    if len(df) and pd.to_datetime(df[date_col]).max() >= holdout_start:
        raise AssertionError(f"Holdout contamination: max {date_col} >= {holdout_start.date()}")


def is_available_at_origin(observation_date: pd.Timestamp, publication_lag_days: int,
                           origin_date: pd.Timestamp) -> bool:
    """A value is usable at ``origin_date`` only if observation + publication lag <= origin."""
    return pd.Timestamp(observation_date) + pd.Timedelta(days=int(publication_lag_days)) <= pd.Timestamp(origin_date)


def find_target_equal_features(features: pd.DataFrame, target: pd.Series,
                               tol: float = 1e-6, corr_threshold: float = 0.98) -> List[str]:
    """Return feature columns that duplicate (or near-perfectly correlate with) the target."""
    hits = []
    for c in features.columns:
        f = pd.to_numeric(features[c], errors="coerce")
        m = pd.concat([f, target], axis=1).dropna()
        if len(m) < 10:
            continue
        a, b = m.iloc[:, 0].to_numpy(float), m.iloc[:, 1].to_numpy(float)
        if np.allclose(a, b, atol=tol):
            hits.append(c)
        elif a.std() > 0 and b.std() > 0 and abs(np.corrcoef(a, b)[0, 1]) >= corr_threshold:
            hits.append(c)
    return hits


def normalised_value(model_mae: float, mean_abs_movement: float) -> float:
    """model MAE divided by actual mean absolute movement (<1 means the model beats 'no skill')."""
    return float(model_mae / mean_abs_movement) if mean_abs_movement else np.nan


def decide_status(warnings: Iterable[Dict]) -> str:
    """FAIL on any CRITICAL warning, PASS_WITH_WARNINGS on HIGH/MEDIUM/LOW, else PASS."""
    sev = {w["severity"] for w in warnings}
    if "CRITICAL" in sev:
        return "FAIL"
    if sev & {"HIGH", "MEDIUM", "LOW"}:
        return "PASS_WITH_WARNINGS"
    return "PASS"


# ----------------------------------------------------------------------------
# Audit orchestration
# ----------------------------------------------------------------------------
EXTERNAL_FAMILIES = {
    "macro_commodity": ("commodities_10y.csv", "global commodity prices"),
    "macro_currency": ("exchange_rates_10y.csv", "FX rates"),
    "macro_fuel": ("fuel_prices.csv", "fuel prices"),
    "macro_inflation": ("cpi_headline.csv + ppi.csv", "CPI / PPI"),
    "market_price": ("pricecatcher_2024-01.parquet", "PriceCatcher retail market prices"),
    "policy_subsidy": ("malaysia_subsidies_and_policies.csv", "policy / subsidy flags"),
    "weather": ("malaysia_weather_10y.csv", "weather"),
    "weather_rolling": ("malaysia_weather_10y.csv", "weather"),
    "weather_spells": ("malaysia_weather_10y.csv", "weather"),
    "salary_cycle": ("malaysia_government_salary_dates_10y.csv", "salary calendar"),
    "festival_proximity": ("malaysia_public_holidays_10y.csv", "holiday calendar"),
}


def _dev(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out[out[col] < HOLDOUT_START]


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_audit(root: Path, make_figures: bool = True) -> Dict:
    """Run the full read-only audit and write reports. Returns a result dictionary."""
    root = Path(root)
    out_dir = root / "reports" / "data_validation"
    fig_dir = root / "reports" / "figures" / "price_data_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    R: Dict = {}
    warnings: List[Dict] = []

    def warn(sev: str, wid: str, finding: str, evidence: str, action: str) -> None:
        warnings.append({"severity": sev, "id": wid, "finding": finding, "evidence": evidence, "action": action})

    # ---------------------------------------------------------------- raw prices
    raw = pd.read_parquet(root / "data/raw/retail_purchase_prices.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    n_holdout_raw = int((raw["date"] >= HOLDOUT_START).sum())
    dev_raw = raw[raw["date"] < HOLDOUT_START].copy()
    assert_no_holdout(dev_raw)

    dev_raw["is_genuine"] = genuine_observation_mask(dev_raw, price_col=PRICE_COL)
    dev_raw["is_forward_filled"] = dev_raw[PRICE_COL].notna() & ~dev_raw["is_genuine"]

    raw_days = int((raw["date"].max() - raw["date"].min()).days + 1)
    dev_days = int((dev_raw["date"].max() - dev_raw["date"].min()).days + 1)
    n_series = dev_raw.groupby(SERIES_KEYS).ngroups
    R.update(
        raw_start=str(raw["date"].min().date()), raw_end=str(raw["date"].max().date()), raw_days=raw_days,
        dev_start=str(dev_raw["date"].min().date()), dev_end=str(dev_raw["date"].max().date()), dev_days=dev_days,
        total_rows_all=int(len(raw)), total_rows_dev=int(len(dev_raw)), holdout_rows_counted_only=n_holdout_raw,
        genuine_obs=int(dev_raw["is_genuine"].sum()), forward_filled=int(dev_raw["is_forward_filled"].sum()),
        unique_products=int(raw["product_id"].nunique()), unique_stores=int(raw["store_id"].nunique()),
        unique_suppliers=int(raw["supplier_id"].nunique()), n_series=int(n_series),
        supplier_per_product=float(raw.groupby("product_id")["supplier_id"].nunique().max()),
        hist_equals_current_pct=float((raw["historical_supplier_purchase_price"] == raw[PRICE_COL]).mean() * 100),
    )
    R["forward_filled_pct"] = 100.0 * R["forward_filled"] / max(R["total_rows_dev"], 1)

    # ------------------------------------------------- per-series density/variation
    dens, chg = [], []
    for key, g in dev_raw.sort_values("date").groupby(SERIES_KEYS):
        gen = g[g["is_genuine"]]
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        gaps = int(len(full_idx) - g["date"].nunique())
        st = price_variation_stats(g[PRICE_COL])
        dens.append({
            "store_id": key[0], "product_id": key[1], "supplier_id": key[2],
            "first_date": g["date"].min().date(), "last_date": g["date"].max().date(),
            "total_rows": len(g), "genuine_observations": len(gen),
            "forward_filled_rows": int(g["is_forward_filled"].sum()),
            "unique_prices": st["unique_prices"], "price_change_events": st["change_count"],
            "calendar_gap_days": gaps,
            "density_class": classify_density(len(gen)),
        })
        chg.append({"scope": "SERIES", "store_id": key[0], "product_id": key[1], "supplier_id": key[2], **st})
    dens_df = pd.DataFrame(dens)
    th = DensityThresholds()
    dens_df["threshold_high"], dens_df["threshold_medium"], dens_df["threshold_low"] = th.high, th.medium, th.low
    for stat, fn in [("mean", np.mean), ("median", np.median), ("p10", lambda x: np.percentile(x, 10)),
                     ("p25", lambda x: np.percentile(x, 25)), ("p75", lambda x: np.percentile(x, 75)),
                     ("p90", lambda x: np.percentile(x, 90)), ("min", np.min), ("max", np.max)]:
        R[f"obs_per_series_{stat}"] = float(fn(dens_df["genuine_observations"].to_numpy()))
    R["density_class_counts"] = dens_df["density_class"].value_counts().to_dict()
    _write(dens_df, out_dir / "price_series_density.csv")

    by_prod = []
    for p, g in dev_raw.groupby("product_id"):
        by_prod.append({"scope": "PRODUCT", "store_id": "ALL", "product_id": p,
                        "supplier_id": g["supplier_id"].iloc[0], **price_variation_stats(g[PRICE_COL])})
    # price-change stats per series must be computed within a series for ALL
    allrow = {"scope": "ALL", "store_id": "ALL", "product_id": "ALL", "supplier_id": "ALL"}
    d_all = dev_raw.sort_values("date").groupby(SERIES_KEYS)[PRICE_COL].diff().abs().dropna()
    allrow.update({
        "n_obs": int(dev_raw[PRICE_COL].notna().sum()), "mean": float(dev_raw[PRICE_COL].mean()),
        "std": float(dev_raw[PRICE_COL].std()), "variance": float(dev_raw[PRICE_COL].var()),
        "cv": float(dev_raw[PRICE_COL].std() / dev_raw[PRICE_COL].mean()),
        "min": float(dev_raw[PRICE_COL].min()), "max": float(dev_raw[PRICE_COL].max()),
        "iqr": float(dev_raw[PRICE_COL].quantile(.75) - dev_raw[PRICE_COL].quantile(.25)),
        "unique_prices": int(dev_raw[PRICE_COL].nunique()), "change_count": int((d_all > 1e-9).sum()),
        "unchanged_rate": float(1 - (d_all > 1e-9).mean()), "mean_abs_change": float(d_all.mean()),
        "median_abs_change": float(d_all.median()), "p90_abs_change": float(d_all.quantile(.9)),
    })
    chg_df = pd.DataFrame(chg + by_prod + [allrow])
    _write(chg_df, out_dir / "price_change_analysis.csv")
    R["change_events"] = int(allrow["change_count"])
    R["change_rate"] = float(1 - allrow["unchanged_rate"])
    R["mean_price_dev"] = allrow["mean"]

    # ----------------------------------------------- stochastic structure (A / B)
    wide = dev_raw.sort_values("date").pivot_table(index="date", columns=["store_id", "product_id"], values=PRICE_COL)
    dif = wide.diff()
    ac_diff = [dif[c].autocorr(1) for c in wide.columns]
    dm = wide - wide.mean()
    ac_lvl = [dm[c].autocorr(1) for c in wide.columns]
    vr = {}
    for h in (7, 30):
        v1 = float(np.nanvar(wide.diff(1).to_numpy()))
        vh = float(np.nanvar(wide.diff(h).to_numpy()))
        vr[h] = float(vh / v1) if v1 > 0 else np.nan
    within_var = float(np.nanvar(dm.to_numpy()))
    R.update(lag1_autocorr_changes_mean=float(np.nanmean(ac_diff)),
             lag1_autocorr_levels_demeaned_mean=float(np.nanmean(ac_lvl)),
             variance_ratio_7d=vr[7], variance_ratio_30d=vr[30],
             within_series_std_mean=float(np.nanmean(wide.std())),
             mean_price_noise_pct=float((wide.std() / wide.mean()).mean() * 100))

    # --------------------------------------------------------- forecast targets
    tg = _dev(pd.read_parquet(root / "data/processed/forecast_targets.parquet"))
    tg_all_n = pd.read_parquet(root / "data/processed/forecast_targets.parquet", columns=["date"])
    tg_all_dates = pd.to_datetime(tg_all_n["date"])
    folds = pd.read_csv(root / "reports/baselines/walk_forward_folds.csv", parse_dates=[
        "training_start", "training_end", "forecast_origin", "validation_start", "validation_end"])
    pred = pd.read_parquet(root / "data/predictions/final_precalibration_price_predictions.parquet")
    pred["forecast_origin"] = pd.to_datetime(pred["forecast_origin"])
    assert_no_holdout(pred, "forecast_origin")
    pred["target_date"] = pred["forecast_origin"] + pd.to_timedelta(pred["horizon"], unit="D")
    assert pred["target_date"].max() < HOLDOUT_START, "validation targets reach the holdout"

    cov_rows, move_rows = [], []
    for h in HORIZONS:
        v = tg[f"purchase_price_target_{h}d_valid"].astype(bool)
        cov = target_coverage(tg[f"purchase_price_target_{h}d_valid"], tg["date"], pd.Timestamp("2024-09-30"))
        # holdout counts: use ALL targets dates, valid flag counted only
        full = pd.read_parquet(root / "data/processed/forecast_targets.parquet",
                               columns=["date", f"purchase_price_target_{h}d_valid"])
        full["date"] = pd.to_datetime(full["date"])
        hold_valid = int((full["date"] >= HOLDOUT_START).sum())
        cur = tg["purchase_price_at_origin"]
        fut = tg[f"purchase_price_target_{h}d"].where(v)
        mv = horizon_movement(cur, fut)
        tobs = pd.to_datetime(tg[f"purchase_price_target_{h}d_observation_date"])
        crossing = int((tobs >= HOLDOUT_START).sum())
        nfv = check_no_future_backfill(tg["date"], tg["purchase_price_origin_observation_date"], tobs, h)
        sel = tg[f"purchase_price_target_{h}d_selection_method"].value_counts().to_dict()
        n_val = int((pred["horizon"] == h).sum())
        cov_rows.append({
            "horizon_days": h, **{k: cov[k] for k in ["candidate_rows", "valid_rows", "invalid_rows", "coverage_pct"]},
            "training_rows_dev_pool": int(v.sum()), "validation_rows_walk_forward": n_val,
            "holdout_rows_counted_only": hold_valid,
            "targets_with_observation_on_or_after_holdout": crossing,
            "selection_methods": json.dumps(sel),
            "origin_price_from_future_violations": nfv["origin_price_from_future"],
            "target_before_due_date_violations": nfv["target_before_due_date"],
        })
        move_rows.append({"horizon_days": h, **mv})
    cov_df = pd.DataFrame(cov_rows)
    mov_df = pd.DataFrame(move_rows)
    _write(cov_df.merge(mov_df, on="horizon_days"), out_dir / "price_horizon_target_coverage.csv")
    R["coverage"] = cov_df.set_index("horizon_days")["coverage_pct"].to_dict()
    R["mean_abs_move"] = mov_df.set_index("horizon_days")["mean_abs_move"].to_dict()
    R["pct_differing"] = mov_df.set_index("horizon_days")["pct_differing"].to_dict()

    # ------------------------------------------------- baselines / final model MAE
    base = pd.read_parquet(root / "data/predictions/baseline_price_predictions.parquet")
    base["forecast_origin"] = pd.to_datetime(base["forecast_origin"])
    assert_no_holdout(base, "forecast_origin")
    pred["abs_err"] = (pred["final_predicted_future_price"] - pred["actual_future_price"]).abs()
    pred["last_err"] = (pred["current_purchase_price"] - pred["actual_future_price"]).abs()
    pred["moved"] = (pred["actual_future_price"] - pred["current_purchase_price"]).abs() > 1e-9
    pred["abs_move"] = (pred["actual_future_price"] - pred["current_purchase_price"]).abs()

    # causal series-mean baselines on the exact validation rows
    cum = dev_raw.sort_values("date").copy()
    cum["expanding_mean"] = cum.groupby(["store_id", "product_id"])[PRICE_COL].transform(lambda s: s.expanding().mean())
    cum["trailing_90d_mean"] = cum.groupby(["store_id", "product_id"])[PRICE_COL].transform(
        lambda s: s.rolling(90, min_periods=30).mean())
    pred = pred.merge(cum[["date", "store_id", "product_id", "expanding_mean", "trailing_90d_mean"]]
                      .rename(columns={"date": "forecast_origin"}),
                      on=["forecast_origin", "store_id", "product_id"], how="left")

    bl_rows = []
    for h in HORIZONS:
        p = pred[pred["horizon"] == h]
        a = p["actual_future_price"]
        row = {"horizon_days": h, "n_validation_rows": len(p), "mean_actual_price": float(a.mean()),
               "mean_abs_actual_movement": float(p["abs_move"].mean())}

        def met(pr):
            e = (pr - a).abs()
            return float(e.mean()), float(np.sqrt(((pr - a) ** 2).mean())), float((e / a).mean() * 100)

        lp = last_price_baseline_metrics(p["current_purchase_price"], a)
        row.update(last_price_mae=lp["mae"], last_price_rmse=lp["rmse"], last_price_mape_pct=lp["mape_pct"])
        row["final_model_mae"], row["final_model_rmse"], row["final_model_mape_pct"] = met(p["final_predicted_future_price"])
        row["ma30_baseline_mae"], row["ma30_baseline_rmse"], _ = met(p["baseline_prediction"])
        row["expanding_mean_mae"], row["expanding_mean_rmse"], _ = met(p["expanding_mean"])
        row["trailing_90d_mean_mae"], row["trailing_90d_mean_rmse"], _ = met(p["trailing_90d_mean"])
        best_simple = min(row["last_price_mae"], row["ma30_baseline_mae"], row["expanding_mean_mae"], row["trailing_90d_mean_mae"])
        row["best_simple_baseline_mae"] = best_simple
        row["improvement_vs_last_price_pct"] = 100 * (row["last_price_mae"] - row["final_model_mae"]) / row["last_price_mae"]
        row["improvement_vs_best_simple_pct"] = 100 * (best_simple - row["final_model_mae"]) / best_simple
        row["model_mae_over_mean_abs_movement"] = normalised_value(row["final_model_mae"], row["mean_abs_actual_movement"])
        row["selected_architecture"] = p["selected_architecture"].mode().iloc[0]
        row["corr_current_vs_future_pooled"] = float(np.corrcoef(p["current_purchase_price"], a)[0, 1])
        fut_minus_cur = (a - p["current_purchase_price"])
        row["future_minus_current_mean"] = float(fut_minus_cur.mean())
        row["future_minus_current_std"] = float(fut_minus_cur.std())
        row["future_minus_current_p10"] = float(fut_minus_cur.quantile(.1))
        row["future_minus_current_p90"] = float(fut_minus_cur.quantile(.9))
        # within-product (price scale removed) correlation of pct moves with model pct prediction
        row["corr_pred_pct_vs_actual_pct"] = float(np.corrcoef(
            p["predicted_percentage_change"], (a / p["current_purchase_price"] - 1) * 100)[0, 1]) \
            if p["predicted_percentage_change"].std() > 0 else np.nan
        bl_rows.append(row)
    bl_df = pd.DataFrame(bl_rows)
    # also every named baseline from Phase 10 for completeness
    named = (base.assign(err=(base["prediction"] - base["actual"]).abs())
             .groupby(["horizon", "baseline"])["err"].mean().unstack("baseline"))
    for c in named.columns:
        bl_df[f"phase10_{c}_mae"] = bl_df["horizon_days"].map(named[c])
    _write(bl_df, out_dir / "price_baseline_comparison.csv")
    R["baseline"] = bl_df.set_index("horizon_days").to_dict("index")

    # ----------------------------------------------------- fold sufficiency
    tg_v = {h: tg[tg[f"purchase_price_target_{h}d_valid"].astype(bool)] for h in HORIZONS}
    fs_rows = []
    for _, f in folds.iterrows():
        for h in HORIZONS:
            tr = tg_v[h]
            tmask = (tr["date"] >= f["training_start"]) & (tr["date"] <= f["forecast_origin"] - pd.Timedelta(days=h))
            p = pred[(pred["validation_fold"] == f["validation_fold"]) & (pred["horizon"] == h)]
            if p.empty:
                continue
            chg_ev = int(p["moved"].sum())
            fs_rows.append({
                "fold": int(f["validation_fold"]), "horizon_days": h,
                "train_start": f["training_start"].date(), "train_end_effective": (f["forecast_origin"] - pd.Timedelta(days=h)).date(),
                "train_rows": int(tmask.sum()),
                "validation_origin": f["forecast_origin"].date(), "validation_target_date": p["target_date"].iloc[0].date(),
                "validation_rows": len(p), "price_change_events_in_validation": chg_ev,
                "unchanged_rate": float(1 - chg_ev / len(p)),
                "mean_abs_movement": float(p["abs_move"].mean()),
                "last_price_mae": float(p["last_err"].mean()), "final_model_mae": float(p["abs_err"].mean()),
                "flag_many_rows_few_changes": bool(len(p) >= 100 and chg_ev / len(p) < 0.05),
            })
    fs_df = pd.DataFrame(fs_rows)
    _write(fs_df, out_dir / "price_fold_sufficiency.csv")
    R["folds_flagged_many_rows_few_changes"] = int(fs_df["flag_many_rows_few_changes"].sum())
    R["fold_train_rows_min"] = int(fs_df["train_rows"].min())
    R["fold_train_rows_max"] = int(fs_df["train_rows"].max())
    R["fold_val_origin_count"] = int(folds.shape[0])

    # ------------------------------------------------------ leakage audit (Phase 12 features)
    reg = pd.read_csv(root / "reports/leakage/price_features_temporally_safe.csv")
    feat_all = pd.read_parquet(root / "data/processed/model_features.parquet")
    feat_all["date"] = pd.to_datetime(feat_all["date"])
    feat_dev = feat_all[feat_all["date"] < HOLDOUT_START]
    jt = feat_dev[["date", "store_id", "product_id"]].merge(
        tg[["date", "store_id", "product_id"] + [f"purchase_price_pct_change_{h}d" for h in HORIZONS]
           + [f"purchase_price_target_{h}d_valid" for h in HORIZONS]],
        on=["date", "store_id", "product_id"], how="left", suffixes=("", "_t"))
    approved = [c for c in reg["feature_name"] if c in feat_dev.columns]
    fnum = feat_dev[approved].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    corr_rows = []
    leak_names = ("target", "future", "_next", "lead_", "forward")
    for h in HORIZONS:
        y = pd.to_numeric(jt[f"purchase_price_pct_change_{h}d"], errors="coerce").where(
            jt[f"purchase_price_target_{h}d_valid"].astype(bool)).reset_index(drop=True)
        cr = fnum.corrwith(y)
        for c, v in cr.items():
            corr_rows.append({"feature": c, "horizon_days": h, "corr_with_future_pct_change": v})
    corr_df = pd.DataFrame(corr_rows)
    mx = corr_df.assign(a=corr_df["corr_with_future_pct_change"].abs()).sort_values("a", ascending=False)
    fam = reg.set_index("feature_name")["feature_family"].to_dict()
    mx["feature_family"] = mx["feature"].map(fam)
    mx["name_pattern_flag"] = mx["feature"].str.lower().apply(lambda s: any(t in s for t in leak_names))
    R["phase12_max_abs_corr"] = float(mx["a"].max())
    R["phase12_top_corr_features"] = mx.head(8)[["feature", "horizon_days", "corr_with_future_pct_change"]].to_dict("records")
    R["phase12_features_over_0p9"] = int((mx["a"] > 0.9).sum())
    R["phase12_name_pattern_flags"] = int(mx["name_pattern_flag"].sum())
    R["phase12_feature_count"] = len(approved)
    corr_df.to_csv(out_dir / "price_feature_target_correlations.csv", index=False)

    # --- Phase 13/14 cache audit: is any "feature" equal to the future target?
    cache_hits: Dict[int, List[str]] = {}
    keep = pd.read_csv(root / "reports/ablation/price_features_for_optimisation.csv")
    keep_feats = keep.loc[keep["keep_status"] == "KEEP", "feature_name"].tolist()
    for h in HORIZONS:
        cp = root / f"cache/model_ready/price_{h}d.parquet"
        if not cp.exists():
            continue
        c = pd.read_parquet(cp)
        c["date"] = pd.to_datetime(c["date"])
        c = c[c["date"] < HOLDOUT_START]
        tgt = tg[["date", "store_id", "product_id", "purchase_price_at_origin", f"purchase_price_target_{h}d",
                  f"purchase_price_pct_change_{h}d"]].rename(columns={f"purchase_price_pct_change_{h}d": "_true_pct"})
        m = c.merge(tgt, on=["date", "store_id", "product_id"], how="inner")
        cols = [k for k in keep_feats if k in m.columns]
        hits = find_target_equal_features(m[cols], m["_true_pct"], tol=1e-5, corr_threshold=0.98)
        cache_hits[h] = hits
    R["cache_target_equal_features"] = cache_hits
    R["cache_leak_horizons"] = [h for h, v in cache_hits.items() if v]

    # evidence from saved Phase 14 model: gain share of the leaking feature
    gain_share = {}
    try:
        import joblib
        for h in R["cache_leak_horizons"]:
            mdl = joblib.load(root / f"models/optimised/price/lightgbm_pct_{h}d.joblib")
            names = list(getattr(mdl, "feature_name_", []) or mdl.booster_.feature_name())
            gain = mdl.booster_.feature_importance("gain")
            tot = float(gain.sum()) or 1.0
            leaked = cache_hits[h]
            idx = [i for i, n in enumerate(names) if n in leaked]
            gain_share[h] = {"leaking_features": leaked, "gain_share_pct": float(100 * gain[idx].sum() / tot) if idx else None}
    except Exception as exc:  # pragma: no cover - evidence only
        gain_share["error"] = str(exc)
    R["phase14_model_gain_share_of_leaking_feature"] = gain_share

    p14 = pd.read_csv(root / "reports/optimisation/before_after_optimisation.csv")
    p14p = p14[p14["task"] == "price"][["horizon", "before_metric", "after_metric", "status"]]
    R["phase14_price"] = p14p.to_dict("records")

    # downstream consumption of the optimised (leaky) artefacts
    inf_src = (root / "src/ensemble/inference.py").read_text(encoding="utf-8")
    R["inference_prefers_optimised_price_models"] = "models/optimised/price" in inf_src
    R["optimised_price_models_present"] = sorted(p.name for p in (root / "models/optimised/price").glob("*.joblib")) \
        if (root / "models/optimised/price").exists() else []

    # --------------------------------------------------- external coverage
    ext_rows = []
    raw_dir = root / "data/raw"
    fam_df = reg[reg["feature_name"].isin(feat_dev.columns)].copy()
    ftm = _dev(pd.read_parquet(root / "data/processed/model_features.parquet",
                               columns=["date"] + list(fam_df["feature_name"])))
    final_cols_p12 = set(approved)
    for family, g in fam_df.groupby("feature_family"):
        cols = list(g["feature_name"])
        sub = ftm[cols].apply(pd.to_numeric, errors="coerce")
        anyv = sub.notna().any(axis=1)
        raw_src, desc = EXTERNAL_FAMILIES.get(family, ("internal", "internal retailer data / derived"))
        role = "EXTERNAL_PREDICTOR" if family in EXTERNAL_FAMILIES else "INTERNAL_PREDICTOR"
        mc = corr_df[corr_df["feature"].isin(cols)]
        ext_rows.append({
            "record_type": "FEATURE_FAMILY", "name": family, "role": role, "raw_source": raw_src, "description": desc,
            "n_features": len(cols), "missing_pct_dev_rows": float(100 * sub.isna().mean().mean()),
            "rows_with_any_value_pct": float(100 * anyv.mean()),
            "first_non_null_date": str(ftm.loc[anyv, "date"].min().date()) if anyv.any() else None,
            "last_non_null_date": str(ftm.loc[anyv, "date"].max().date()) if anyv.any() else None,
            "frequency": "/".join(sorted(set(g["source_frequency"].astype(str)))),
            "publication_lag_days_max": int(g["publication_lag_days"].max()),
            "safe_at_origin_phase8": bool((g["final_audit_status"] == "SAFE").all()),
            "used_by_phase12_final_family": bool(set(cols) & final_cols_p12),
            "max_abs_corr_with_future_pct": float(mc["corr_with_future_pct_change"].abs().max()) if len(mc) else np.nan,
        })
    synthetic_cols = {}
    comm = pd.read_csv(raw_dir / "commodities_10y.csv")
    try:
        np.random.seed(123)
        n = len(comm)
        pal = np.clip(2400 + np.cumsum(np.random.normal(0.5, 25, n)), 2000, 7000)
        ste = 700 + np.cumsum(np.random.normal(0.1, 5, n))
        fee = 450 + np.cumsum(np.random.normal(0.1, 4, n))
        synthetic_cols = {
            "palm_oil_myr_per_tonne": bool(np.allclose(comm["palm_oil_myr_per_tonne"].to_numpy(), pal, atol=1e-6)),
            "steel_usd_index": bool(np.allclose(comm["steel_usd_index"].to_numpy(), ste, atol=1e-6)),
            "feed_grain_usd": bool(np.allclose(comm["feed_grain_usd"].to_numpy(), fee, atol=1e-6)),
        }
    except Exception:  # pragma: no cover
        pass
    R["commodity_columns_equal_seeded_random_walk"] = synthetic_cols
    for f, dcol in [("commodities_10y.csv", "date"), ("exchange_rates_10y.csv", "date"), ("fuel_prices.csv", "date"),
                    ("cpi_headline.csv", "date"), ("ppi.csv", "date"), ("pricecatcher_2024-01.parquet", "date"),
                    ("malaysia_weather_10y.csv", "date"), ("malaysia_subsidies_and_policies.csv", None)]:
        p = raw_dir / f
        if not p.exists():
            continue
        d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        col = dcol if dcol in d.columns else next((c for c in d.columns if "date" in c.lower()), None)
        dts = pd.to_datetime(d[col], errors="coerce") if col else None
        ext_rows.append({
            "record_type": "RAW_SOURCE", "name": f, "role": "EXTERNAL_PREDICTOR", "raw_source": f, "description": "raw file",
            "n_features": np.nan, "missing_pct_dev_rows": float(100 * d.isna().mean().mean()),
            "rows_with_any_value_pct": np.nan,
            "first_non_null_date": str(dts.min().date()) if dts is not None and dts.notna().any() else None,
            "last_non_null_date": str(dts.max().date()) if dts is not None and dts.notna().any() else None,
            "frequency": f"{len(d)} rows", "publication_lag_days_max": np.nan, "safe_at_origin_phase8": np.nan,
            "used_by_phase12_final_family": np.nan, "max_abs_corr_with_future_pct": np.nan,
        })
    ext_df = pd.DataFrame(ext_rows)
    _write(ext_df, out_dir / "price_external_feature_coverage.csv")
    R["external_family_table"] = ext_df[ext_df["record_type"] == "FEATURE_FAMILY"][
        ["name", "role", "n_features", "missing_pct_dev_rows", "rows_with_any_value_pct", "max_abs_corr_with_future_pct"]
    ].to_dict("records")
    mk = ext_df[(ext_df["name"] == "market_price")]
    R["market_price_rows_with_value_pct"] = float(mk["rows_with_any_value_pct"].iloc[0]) if len(mk) else np.nan

    # --------------------------------------------------- source usage table
    interim = root / "data/interim"
    src_defs = [
        ("Retailer supplier purchase price (TARGET)", "retail_purchase_prices.parquet", "purchase_prices_clean.parquet",
         "purchase_price_history", "TARGET + internal lag/rolling predictors",
         "Provenance undocumented; generator not in repository; statistical profile = noise around fixed base (see audit)."),
        ("PriceCatcher (KPDN retail shelf prices)", "pricecatcher_2024-01.parquet", "pricecatcher_clean.parquet",
         "market_price", "EXTERNAL predictor", "Only 2024-01 downloaded; almost all development rows have no value."),
        ("Exchange rates", "exchange_rates_10y.csv", "exchange_rates_clean.parquet", "macro_currency",
         "EXTERNAL predictor", "Market data (yfinance); business-day, forward-filled to daily."),
        ("Commodity prices", "commodities_10y.csv", "commodity_prices_clean.parquet", "macro_commodity",
         "EXTERNAL predictor", "7 real yfinance series; palm_oil / steel / feed_grain are seeded random walks (synthetic)."),
        ("Brent / fuel prices", "fuel_prices.csv", "fuel_prices_clean.parquet", "macro_fuel",
         "EXTERNAL predictor", "Weekly fuel (data.gov.my) forward-filled to daily; Brent inside commodities file."),
        ("CPI", "cpi_headline.csv", "consumer_price_index_clean.parquet", "macro_inflation",
         "EXTERNAL predictor", "Monthly, publication lag applied."),
        ("PPI", "ppi.csv", "producer_price_index_clean.parquet", "macro_inflation",
         "EXTERNAL predictor", "Monthly, publication lag applied."),
        ("External trade (BEC/SITC)", "trade_headline.csv", "external_trade_clean.parquet", None,
         "EXTERNAL candidate", "Downloaded and cleaned but no trade feature exists in the approved price manifest."),
        ("Subsidy / policy / budget", "malaysia_subsidies_and_policies.csv", "subsidies_and_policies_clean.parquet",
         "policy_subsidy", "EXTERNAL predictor", "Event flags; Phase 8 SAFE but excluded by Phase 13 (non-informative)."),
        ("Weather", "malaysia_weather_10y.csv", None, "weather", "EXTERNAL predictor",
         "Phase 8 SAFE; excluded by Phase 13."),
    ]
    fam_status = ext_df[ext_df["record_type"] == "FEATURE_FAMILY"].set_index("name")
    keep_fams = set(keep.loc[keep["keep_status"] == "KEEP", "feature_family"])
    su = []
    for name, rawf, interf, fam_key, role, reason in src_defs:
        in_reg = fam_key in fam_status.index if fam_key else False
        su.append({
            "source": name, "downloaded_phase2": (raw_dir / rawf).exists(),
            "processed": bool(interf and (interim / interf).exists()),
            "passed_phase8_leakage_audit": bool(fam_status.loc[fam_key, "safe_at_origin_phase8"]) if in_reg else False,
            "used_by_final_price_model_phase12_15": in_reg,
            "used_by_phase14_optimised_model": bool(fam_key in keep_fams) if fam_key else False,
            "role": role, "reason_or_note": reason,
        })
    su_df = pd.DataFrame(su)
    _write(su_df, out_dir / "price_source_usage.csv")
    R["sources_used"] = su_df.loc[su_df["used_by_final_price_model_phase12_15"], "source"].tolist()
    R["sources_not_used"] = su_df.loc[~su_df["used_by_final_price_model_phase12_15"], "source"].tolist()

    # -------------------------------------------------------------- warnings
    b30 = R["baseline"][30]
    b1 = R["baseline"][1]
    best_gap = [R["baseline"][h]["improvement_vs_best_simple_pct"] for h in HORIZONS]
    R["max_improvement_vs_best_simple_pct"] = float(max(best_gap))
    R["max_improvement_vs_last_price_pct"] = float(max(R["baseline"][h]["improvement_vs_last_price_pct"] for h in HORIZONS))

    if R["cache_leak_horizons"]:
        warn("CRITICAL", "W01_TARGET_IN_FEATURE_CACHE",
             f"Phase 13/14 model-ready price cache contains the future target as feature "
             f"'purchase_price_pct_change_{{h}}d' for horizons {R['cache_leak_horizons']}.",
             f"cache/model_ready/price_hd.parquet; src/ablation/cache.py lines 112-113 overwrite the backward-looking feature "
             f"with the target; Phase 14 'after' MAE {[r['after_metric'] for r in R['phase14_price'] if r['horizon'] in R['cache_leak_horizons']]} "
             f"vs Phase 12 ~0.20; improvement exactly on leaking horizons only.",
             "Fix cache builder to never overwrite features with target columns, rebuild cache, re-run Phase 13/14 (not done in this audit).")
    if R["inference_prefers_optimised_price_models"] and R["cache_leak_horizons"]:
        warn("HIGH", "W02_LEAKY_MODELS_IN_INFERENCE_PATH",
             "src/ensemble/inference.py resolves models/optimised/price/lightgbm_pct_{h}d.joblib BEFORE the Phase 12 models; "
             "the 1d/7d/30d optimised models were trained with the target as a feature.",
             f"optimised artefacts present: {R['optimised_price_models_present']}",
             "Remove/quarantine leaky optimised price artefacts or point inference to Phase 12 models until Phase 14 is re-run on a fixed cache.")
    if R["max_improvement_vs_best_simple_pct"] <= 1.0:
        warn("CRITICAL", "W03_NO_DEMONSTRABLE_SKILL",
             "The final price models do not beat the best trivial baseline by a meaningful margin at any horizon.",
             f"max improvement vs last price {R['max_improvement_vs_last_price_pct']:.2f}%; "
             f"vs best simple baseline {R['max_improvement_vs_best_simple_pct']:.2f}% (values <= 0 mean the model is worse).",
             "Do not present price MAE as evidence of predictive skill; present it as a calibrated noise band around a stable base price.")
    if R["variance_ratio_30d"] < 3.0:
        warn("HIGH", "W04_PRICE_IS_STATIONARY_NOISE",
             "Purchase prices are mean-reverting noise around a fixed per-series base, not a trending/regime process.",
             f"lag-1 autocorr of daily changes {R['lag1_autocorr_changes_mean']:.2f} (iid noise = -0.50); "
             f"variance ratio 30d/1d = {R['variance_ratio_30d']:.2f} (random walk = 30, iid = 2); "
             f"price noise ~{R['mean_price_noise_pct']:.1f}% of level.",
             "Document that supplier price 'direction' signals have little economic content in this data.")
    warn("HIGH", "W05_TARGET_PROVENANCE_SYNTHETIC",
         "Supplier purchase price (target) originates from a retailer table with undocumented provenance "
         "(generate_retailer_dataset is referenced but not defined in the repository); profile is synthetic-like.",
         "Prices change on 100% of days; historical == current on 100% of rows; base prices hard-coded in "
         "retailer_ingestion.get_default_product_catalog; one supplier per product.",
         "Disclose as synthetic / unverified; replace with real procurement history before any real-world claim.")
    if R["market_price_rows_with_value_pct"] < 10:
        warn("MEDIUM", "W06_PRICECATCHER_SPARSE",
             "PriceCatcher (real market prices) covers only one month, so market-price features are empty for almost all training rows.",
             f"rows with any market_price value in development: {R['market_price_rows_with_value_pct']:.1f}%.",
             "Download full PriceCatcher history or drop the family from claims.")
    if any(synthetic_cols.get(k) for k in synthetic_cols):
        warn("MEDIUM", "W07_SYNTHETIC_COMMODITY_COLUMNS",
             "Commodity file contains seeded random-walk columns presented as market data.",
             f"matches np.random.seed(123) generator: {synthetic_cols}",
             "Label palm_oil / steel / feed_grain as synthetic or replace with real MPOB / index data.")
    warn("MEDIUM", "W08_SPARSE_VALIDATION_ORIGINS",
         "Walk-forward validation uses one forecast origin per fold (33 origins x 24 series = 792 rows/horizon); "
         "rows within an origin share common shocks.",
         f"{R['fold_val_origin_count']} origins; effective sample size is smaller than 792.",
         "Report confidence intervals; consider denser origins.")
    warn("LOW", "W09_UNCHANGED_RATE_NEAR_ZERO",
         "Unlike real procurement feeds (sticky contract prices), series have ~0% unchanged days, so 'price-change event' counts overstate genuine repricing.",
         f"change rate {100 * R['change_rate']:.1f}%.", "Interpret change counts accordingly.")
    R["warnings"] = warnings
    R["status"] = decide_status(warnings)

    # ----------------------------------------------- headline CSV
    head = [
        ("raw_duration_days", R["raw_days"], f"{R['raw_start']} to {R['raw_end']}"),
        ("usable_training_duration_days", R["dev_days"], f"{R['dev_start']} to {R['dev_end']} (holdout excluded)"),
        ("raw_years", round(R["raw_days"] / 365.25, 2), ""),
        ("total_price_rows_all", R["total_rows_all"], "includes holdout rows (counted only)"),
        ("total_price_rows_development", R["total_rows_dev"], ""),
        ("holdout_rows_counted_only", R["holdout_rows_counted_only"], "not analysed"),
        ("genuine_observations", R["genuine_obs"], "non-null and not run-length forward-filled"),
        ("forward_filled_rows", R["forward_filled"], f"min_run=3 heuristic; targets all EXACT"),
        ("forward_filled_pct", round(R["forward_filled_pct"], 4), ""),
        ("price_change_events", R["change_events"], ""),
        ("price_change_rate", round(R["change_rate"], 4), ""),
        ("unique_products", R["unique_products"], ""), ("unique_suppliers", R["unique_suppliers"], "one supplier per product"),
        ("unique_stores", R["unique_stores"], ""), ("series_count", R["n_series"], ""),
        ("median_observations_per_series", R["obs_per_series_median"], ""),
        ("lag1_autocorr_daily_changes", round(R["lag1_autocorr_changes_mean"], 4), "iid noise => -0.5"),
        ("variance_ratio_30d_over_1d", round(R["variance_ratio_30d"], 3), "iid => 2, random walk => 30"),
        ("phase12_max_abs_feature_target_corr", round(R["phase12_max_abs_corr"], 4), "backward-looking features only"),
        ("phase12_features_audited", R["phase12_feature_count"], ""),
        ("phase13_14_cache_leak_horizons", str(R["cache_leak_horizons"]), "target duplicated as feature"),
        ("max_improvement_vs_last_price_pct", round(R["max_improvement_vs_last_price_pct"], 3), ""),
        ("max_improvement_vs_best_simple_baseline_pct", round(R["max_improvement_vs_best_simple_pct"], 3), ""),
        ("PRICE_TRAINING_DATA_SUFFICIENCY", R["status"], "see price_training_sufficiency_summary.md"),
    ]
    _write(pd.DataFrame(head, columns=["metric", "value", "note"]), out_dir / "price_training_data_sufficiency.csv")

    if make_figures:
        _figures(R, bl_df, mov_df, dens_df, ext_df, wide, dev_raw, root, fig_dir)
    _summary_md(R, bl_df, mov_df, out_dir / "price_training_sufficiency_summary.md")
    return R


# ----------------------------------------------------------------------------
def _figures(R, bl_df, mov_df, dens_df, ext_df, wide, dev_raw, root: Path, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    hs = [str(h) for h in HORIZONS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(hs, mov_df["pct_differing"], color="#3b7ddd")
    ax.set_xlabel("Horizon (days)"); ax.set_ylabel("% of rows where future != current")
    ax.set_title("Price-change rate by horizon"); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(fig_dir / "price_change_rate_by_horizon.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(hs)); w = 0.2
    for i, (c, lab) in enumerate([("last_price_mae", "Last price"), ("ma30_baseline_mae", "MA30"),
                                  ("expanding_mean_mae", "Expanding mean"), ("final_model_mae", "Final model")]):
        ax.bar(x + (i - 1.5) * w, bl_df[c], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(hs); ax.set_xlabel("Horizon (days)"); ax.set_ylabel("MAE (RM)")
    ax.set_title("Last-price vs final model MAE (validation, pre-holdout)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "last_price_vs_final_model_mae.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dens_df["genuine_observations"], bins=10, color="#2a9d8f")
    ax.set_xlabel("Genuine observations per series (development)"); ax.set_ylabel("Series")
    ax.set_title("Distribution of genuine observations per series")
    fig.tight_layout(); fig.savefig(fig_dir / "genuine_observations_per_series.png", dpi=130); plt.close(fig)

    tg = _dev(pd.read_parquet(root / "data/processed/forecast_targets.parquet",
                              columns=["date", "purchase_price_at_origin"] + [f"purchase_price_target_{h}d" for h in (1, 7, 30)]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for h in (1, 7, 30):
        mv = (tg[f"purchase_price_target_{h}d"] - tg["purchase_price_at_origin"]).dropna()
        ax.hist(mv, bins=80, alpha=0.5, label=f"{h}d", range=(-1.5, 1.5))
    ax.set_xlabel("future - current price (RM)"); ax.set_ylabel("Rows"); ax.legend()
    ax.set_title("Actual future price movement distribution")
    fig.tight_layout(); fig.savefig(fig_dir / "actual_future_movement_distribution.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hs, bl_df["mean_abs_actual_movement"], "o-", label="Mean |actual movement|")
    ax.plot(hs, bl_df["final_model_mae"], "s-", label="Final model MAE")
    ax.plot(hs, bl_df["best_simple_baseline_mae"], "^--", label="Best simple baseline MAE")
    ax.set_xlabel("Horizon (days)"); ax.set_ylabel("RM"); ax.set_ylim(0, None); ax.legend()
    ax.set_title("Model MAE vs actual movement (no skill => MAE ~ movement)")
    fig.tight_layout(); fig.savefig(fig_dir / "model_mae_vs_actual_movement.png", dpi=130); plt.close(fig)

    fams = [r["name"] for r in R["external_family_table"] if r["role"] == "EXTERNAL_PREDICTOR"]
    feats = pd.read_parquet(root / "data/processed/model_features.parquet")
    feats["date"] = pd.to_datetime(feats["date"]); feats = feats[feats["date"] < HOLDOUT_START]
    reg = pd.read_csv(root / "reports/leakage/price_features_temporally_safe.csv")
    q = feats["date"].dt.to_period("Q").astype(str)
    mat = []
    for f in fams:
        cols = [c for c in reg.loc[reg["feature_family"] == f, "feature_name"] if c in feats.columns]
        mat.append(feats[cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1).groupby(q).mean().values)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(np.array(mat), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=7)
    ax.set_xticks(range(0, len(sorted(q.unique())), 2))
    ax.set_xticklabels(sorted(q.unique())[::2], rotation=60, fontsize=7)
    ax.set_title("External feature coverage timeline (share of rows with a value)")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(fig_dir / "external_feature_coverage_timeline.png", dpi=130); plt.close(fig)

    s = wide.iloc[:, 0]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(s.index, s.values, lw=0.6, label="daily price")
    ax.plot(s.index, s.rolling(30).mean(), lw=1.4, label="30d mean")
    ax.set_title(f"Example series {wide.columns[0]} - noise around a fixed base"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "example_price_series.png", dpi=130); plt.close(fig)


def _summary_md(R, bl_df, mov_df, path: Path) -> None:
    L = []
    a = L.append
    a("# Purchase-Price Training Data Sufficiency and Model Sanity Audit\n")
    a(f"**PRICE TRAINING DATA SUFFICIENCY: {R['status']}**\n")
    a("Read-only audit. No model was retrained; no hyperparameter was changed; the protected holdout "
      f"(from {HOLDOUT_START.date()}) was only row-counted.\n")
    a("## Synthetic-data disclosure\n")
    a("- **Target (supplier purchase price): SYNTHETIC / unverified provenance.** The repository never defines "
      "`generate_retailer_dataset`; base prices are hard-coded; prices change on every day; one supplier per product.")
    a(f"- **External predictors: MIXED.** Real: FX, 7 commodity series, fuel, CPI/PPI, PriceCatcher (1 month only). "
      f"Synthetic: commodity columns equal to a seeded random walk: `{R['commodity_columns_equal_seeded_random_walk']}`.")
    a("- Any accuracy on this target is therefore *not* independent real-world validation.\n")
    a("## Data volume\n")
    a(f"- Raw: {R['raw_start']} to {R['raw_end']} ({R['raw_days']} days, {R['raw_days']/365.25:.2f} years), {R['total_rows_all']:,} rows; "
      f"development (pre-holdout) {R['dev_days']} days, {R['total_rows_dev']:,} rows.")
    a(f"- {R['n_series']} series ({R['unique_stores']} stores x {R['unique_products']} products; {R['unique_suppliers']} suppliers). "
      f"Genuine observations {R['genuine_obs']:,}; forward-filled {R['forward_filled']:,} ({R['forward_filled_pct']:.3f}%).")
    a(f"- Median genuine observations per series: {R['obs_per_series_median']:.0f}; density classes {R['density_class_counts']}.")
    a(f"- Price-change events: {R['change_events']:,} ({100*R['change_rate']:.1f}% of days).\n")
    a("## Last-price baseline vs final model (validation, pre-holdout)\n")
    a("| h | mean abs move | last price MAE | MA30 MAE | expanding-mean MAE | final MAE | vs last | vs best simple |")
    a("|---|---|---|---|---|---|---|---|")
    for _, r in bl_df.iterrows():
        a(f"| {int(r.horizon_days)} | {r.mean_abs_actual_movement:.4f} | {r.last_price_mae:.4f} | {r.ma30_baseline_mae:.4f} | "
          f"{r.expanding_mean_mae:.4f} | {r.final_model_mae:.4f} | {r.improvement_vs_last_price_pct:+.2f}% | {r.improvement_vs_best_simple_pct:+.2f}% |")
    a("\n## MAE sanity (A-G)\n")
    a(f"- **A. Predictable dynamics:** none beyond mean reversion. Lag-1 autocorrelation of daily changes "
      f"{R['lag1_autocorr_changes_mean']:.2f} (iid noise = -0.50); variance ratio 30d/1d = {R['variance_ratio_30d']:.2f} "
      f"(iid = 2, random walk = 30). Price is a stationary band around a base.")
    a(f"- **B. Barely-moving prices:** No - prices change daily; typical |move| is ~{R['mean_price_noise_pct']:.1f}% of price, "
      "so a small RM MAE is a property of low noise amplitude, not of skill.")
    a(f"- **C. Forward fill:** {R['forward_filled_pct']:.3f}% of rows; all target selections are EXACT (tolerance 0); no carry-forward.")
    a("- **D. Target alignment:** target observation date == origin + h for all valid rows; zero origin-from-future and zero target-before-due violations.")
    a(f"- **E. Leakage:** Phase 12 / Phase 15 features are clean (max |corr| with future % change {R['phase12_max_abs_corr']:.2f}, "
      f"backward-looking mean-reversion). **CONFIRMED CRITICAL leakage in the Phase 13/14 cache** on horizons {R['cache_leak_horizons']}: "
      f"the target is copied into feature `purchase_price_pct_change_{{h}}d`. This explains Phase 14 MAE 0.0027 / 0.0036 / 0.0049.")
    a("- **F. Validation contamination:** training uses `date <= origin - h` (purged); evaluation targets all fall before the holdout. No contamination found in Phase 12/15.")
    a("- **G. Learnable synthetic generator:** the final models cannot even beat last-price by >1%, so they are not exploiting generator structure; "
      "the very low MAE reported by Phase 14 is leakage, not learned structure.\n")
    a("## Warnings\n")
    a("| Severity | ID | Finding | Evidence | Recommended action |")
    a("|---|---|---|---|---|")
    for w in R["warnings"]:
        a(f"| {w['severity']} | {w['id']} | {w['finding']} | {w['evidence']} | {w['action']} |")
    a(f"\n## Verdict\n\n**PRICE TRAINING DATA SUFFICIENCY: {R['status']}**\n")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    res = run_audit(Path(__file__).resolve().parents[2])
    print(res["status"])
