"""Calendar-aligned future demand labels with explicit observation validity."""

import numpy as np
import pandas as pd

KEYS = ["date", "store_id", "product_id"]
SERIES = ["store_id", "product_id"]


def prepare_source(frame, config):
    """Copy and validate input. Missing demand is not an implicit zero."""
    df = frame.copy(deep=True)
    if not set(KEYS).issubset(df):
        raise ValueError("Missing primary-key columns")
    df["date"] = pd.to_datetime(df.date, errors="raise")
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError("Null or duplicate primary key")
    if df.date.dt.tz is not None or not df.date.eq(df.date.dt.normalize()).all():
        raise ValueError("Use timezone-naive calendar dates at midnight")
    df = df.sort_values(SERIES + ["date"]).reset_index(drop=True)
    src = config["demand_source"]
    fallback = config["demand_fallback"]
    if src == "estimated_true_demand_retrospective" or fallback == "estimated_true_demand_retrospective":
        raise ValueError("Retrospective demand is prohibited")
    base = pd.to_numeric(df.get(src, pd.Series(np.nan, index=df.index)), errors="raise")
    observed = pd.to_numeric(df.get(fallback, pd.Series(np.nan, index=df.index)), errors="raise")
    df["_demand"] = base.fillna(observed)
    df["_fallback"] = base.isna() & observed.notna()
    reason = pd.Series("VALID", index=df.index)
    reason.loc[~np.isfinite(df._demand) | df._demand.lt(0)] = "INVALID_OR_MISSING_DEMAND"
    # An explicit null status is unknown, never interpreted as open/active.
    for col, label in [("product_available", "UNAVAILABLE_PRODUCT"),
                       ("product_active_in_store", "INACTIVE_PRODUCT"),
                       ("product_active", "INACTIVE_PRODUCT"),
                       ("store_active", "STORE_CLOSED"), ("store_open", "STORE_CLOSED")]:
        if col in df:
            reason.loc[df[col].eq(0)] = label
            reason.loc[~df[col].isin([0, 1])] = "UNKNOWN_ACTIVITY"
    for col, label in [("product_discontinued", "DISCONTINUED_PRODUCT"),
                       ("store_closed", "STORE_CLOSED")]:
        if col in df:
            reason.loc[df[col].eq(1)] = label
            reason.loc[~df[col].isin([0, 1])] = "UNKNOWN_ACTIVITY"
    # End dates are exclusive: no demand is valid on the discontinuation day.
    for col in ["product_discontinued_date", "store_closure_date"]:
        if col in df:
            end = pd.to_datetime(df[col], errors="raise")
            reason.loc[df.date.ge(end)] = "DISCONTINUED_PRODUCT" if col.startswith("product") else "STORE_CLOSED"
    for col in ["product_start_date", "store_opening_date"]:
        if col in df:
            reason.loc[df.date.lt(pd.to_datetime(df[col], errors="raise"))] = "NOT_YET_ACTIVE"
    df["_demand_reason"] = reason
    df["_valid_demand"] = df._demand.where(reason.eq("VALID"))
    return df


def build_demand_targets(source, config):
    """Match T+k by date, never by row offset; return labels only."""
    blocks = []
    max_h = max(config["horizons"])
    for _, group in source.groupby(SERIES, sort=True):
        g = group.set_index("date", drop=False)
        dates = pd.DatetimeIndex(g.date)
        future_dates = [dates + pd.Timedelta(days=k) for k in range(1, max_h + 1)]
        values = np.column_stack([g._valid_demand.reindex(d).to_numpy(float) for d in future_dates])
        reasons = np.column_stack([g._demand_reason.reindex(d).fillna("MISSING_DATE").to_numpy() for d in future_dates])
        fallback = np.column_stack([g._fallback.reindex(d).fillna(False).to_numpy(bool) for d in future_dates])
        result = g[KEYS].reset_index(drop=True)
        result["demand_origin_status"] = g._demand_reason.to_numpy()
        for h in config["horizons"]:
            name = f"demand_target_{h}d"
            v = values[:, :h]
            available = np.isfinite(v).sum(axis=1)
            valid = available == h
            result[name] = np.where(valid, np.nansum(v, axis=1), np.nan)
            result[name + "_valid"] = valid
            result[name + "_days_required"] = h
            result[name + "_days_available"] = available
            result[name + "_coverage"] = available / h
            result[name + "_fallback_days"] = (fallback[:, :h] & np.isfinite(v)).sum(axis=1)
            result[name + "_start_date"] = pd.Series(dates + pd.Timedelta(days=1)).where(valid)
            result[name + "_end_date"] = pd.Series(dates + pd.Timedelta(days=h)).where(valid)
            result[name + "_invalid_reason"] = [
                "VALID" if ok else "|".join(sorted(set(r) - {"VALID"}))
                for ok, r in zip(valid, reasons[:, :h])
            ]
            if config["demand_target"]["create_direct_targets"]:
                # Direct demand helps later compare daily vs cumulative forecasts.
                result[f"demand_day_{h}"] = values[:, h - 1]
                result[f"demand_day_{h}_valid"] = np.isfinite(values[:, h - 1])
        blocks.append(result)
    return pd.concat(blocks, ignore_index=True)
