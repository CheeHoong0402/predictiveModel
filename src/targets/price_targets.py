"""Supplier-specific observed price targets, with bounded optional fallback."""

import numpy as np
import pandas as pd
from .demand_targets import KEYS, SERIES


def build_price_targets(source, config):
    """Exact date -> first observation on/after due date within tolerance ->
    explicit effective interval. Ordinary observations are never forward-filled
    into targets. Supplier identity is the supplier at the forecast origin.
    """
    policy = config["price_target"]
    df = source.copy()
    if "supplier_id" not in df:
        raise ValueError("Supplier identity is required for price targets")
    blocks = []
    for _, g in df.groupby(SERIES + ["supplier_id"], dropna=False, sort=True):
        g = g.sort_values("date")
        result = g[KEYS + ["supplier_id"]].reset_index(drop=True)
        dates = pd.DatetimeIndex(g.date)
        values = pd.to_numeric(g.purchase_price, errors="raise").to_numpy(float)
        valid = np.isfinite(values) & (values > 0) & g.supplier_id.notna().to_numpy()
        if "price_observation_valid" in g:
            valid &= g.price_observation_valid.eq(1).to_numpy()
        observations = dates[valid]
        prices = values[valid]
        n = len(g)
        # Last genuinely observed price known by T, with a bounded age.
        idx = observations.searchsorted(dates, side="right") - 1
        current = np.full(n, np.nan)
        current_dates = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        good = idx >= 0
        if len(observations):
            chosen = observations.to_numpy()[idx.clip(0)]
            good &= ((dates.to_numpy() - chosen) / np.timedelta64(1, "D")) <= policy["current_price_maximum_age_days"]
            current[good] = prices[idx[good]]
            current_dates[good] = chosen[good]
        result["purchase_price_at_origin"] = current
        result["purchase_price_origin_observation_date"] = current_dates
        for h in config["horizons"]:
            due = dates + pd.Timedelta(days=h)
            pos = observations.searchsorted(due, side="left")
            future = np.full(n, np.nan)
            selected = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
            method = np.full(n, "MISSING", dtype=object)
            effective_start = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
            effective_end = effective_start.copy()
            if len(observations):
                candidates = observations.to_numpy()[pos.clip(0, len(observations) - 1)]
                delta = (candidates - due.to_numpy()) / np.timedelta64(1, "D")
                good = (pos < len(observations)) & (delta >= 0) & (delta <= policy["tolerance_days"])
                future[good] = prices[pos[good]]
                selected[good] = candidates[good]
                method[good] = np.where(delta[good] == 0, "EXACT", "NEXT_WITHIN_TOLERANCE")
            if policy["allow_effective_price_carry_forward"]:
                required = {"price_effective_from", "price_effective_until"}
                if not required.issubset(g):
                    raise ValueError("Effective-price carry requires explicit start/end metadata")
                starts = pd.to_datetime(g.price_effective_from, errors="raise").to_numpy()
                ends = pd.to_datetime(g.price_effective_until, errors="raise").to_numpy()
                if ((ends < starts) & valid).any():
                    raise ValueError("Invalid effective-price interval")
                for i in np.flatnonzero(np.isnan(future)):
                    age = (due[i] - dates).days
                    # Strict future observation constraint prevents an old quote
                    # from masquerading as a newly observed future price.
                    eligible = valid & (dates > dates[i]) & (dates <= due[i])
                    eligible &= (age <= policy["maximum_carry_forward_days"])
                    eligible &= (starts <= due[i].to_datetime64()) & (ends >= due[i].to_datetime64())
                    choices = np.flatnonzero(eligible)
                    if len(choices):
                        j = choices[-1]
                        future[i], selected[i], method[i] = values[j], dates[j].to_datetime64(), "EFFECTIVE_INTERVAL"
                        effective_start[i], effective_end[i] = starts[j], ends[j]
            name = f"purchase_price_target_{h}d"
            pct = np.divide(future - current, current, out=np.full(n, np.nan), where=np.isfinite(current) & (current > 0))
            direction = np.full(n, None, dtype=object)
            known = np.isfinite(pct)
            flat = policy["flat_tolerance"]
            direction[known] = np.where(pct[known] > flat + 1e-12, "UP", np.where(pct[known] < -flat - 1e-12, "DOWN", "FLAT"))
            result[name] = future
            result[name + "_valid"] = np.isfinite(future)
            result[name + "_requested_date"] = due
            result[name + "_observation_date"] = selected
            result[name + "_date_difference_days"] = (selected - due.to_numpy()) / np.timedelta64(1, "D")
            result[name + "_selection_method"] = method
            result[name + "_effective_from"] = effective_start
            result[name + "_effective_until"] = effective_end
            result[f"purchase_price_pct_change_{h}d"] = pct
            result[f"price_direction_{h}d"] = direction
            result[f"price_change_{h}d_valid"] = known
            result[f"price_change_{h}d_extreme"] = np.abs(pct) > policy["extreme_change_threshold"]
        blocks.append(result)
    return pd.concat(blocks, ignore_index=True).sort_values(SERIES + ["date"]).reset_index(drop=True)
