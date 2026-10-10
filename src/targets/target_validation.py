"""Fail-closed temporal checks and safe, namespace-aware training assembly."""

import hashlib
import numpy as np
import pandas as pd
from .demand_targets import KEYS

# These existing Phase 7 columns measure P(T)/P(T-H)-1, whereas identically
# named Phase 9 labels measure P(T+H)/P(T)-1. Qualified identity is mandatory.
LEGACY_NAME_COLLISIONS = {f"purchase_price_pct_change_{h}d" for h in [1, 7, 30]}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifests(root, feature_columns, target_names):
    collisions = []
    registry = pd.read_csv(root / "reports/features/feature_registry.csv").set_index("name")
    for model in ["demand", "price"]:
        manifest = pd.read_csv(root / "reports/leakage" / f"{model}_features_temporally_safe.csv")
        if manifest.empty or manifest.feature_name.duplicated().any():
            raise ValueError("Empty or duplicate approved manifest")
        if not manifest.final_audit_status.isin(["SAFE", "SAFE_WITH_ASSUMPTION"]).all():
            raise ValueError("Unapproved feature in manifest")
        if not set(manifest.feature_name).issubset(feature_columns):
            raise ValueError("Approved feature absent from canonical feature table")
        overlap = set(manifest.feature_name) & set(target_names)
        if overlap - LEGACY_NAME_COLLISIONS:
            raise ValueError("Target column contaminates approved manifest")
        for name in sorted(overlap):
            definition = registry.loc[name]
            if definition.feature_family != "price_momentum" or "lag_" not in definition.calculation or definition.source_columns != "purchase_price":
                raise ValueError("Legacy collision no longer describes an approved historical feature")
            collisions.append(dict(model=model, name=name,
                feature_identity=f"model_features.{name}", target_identity=f"forecast_targets.{name}",
                resolution="Keep separate tables; use x__ / y__ aliases in safe assembly"))
        if any(n.startswith(("demand_target_", "demand_day_", "price_direction_", "purchase_price_target_", "y__"))
               or "retrospective" in n for n in manifest.feature_name):
            raise ValueError("Future or retrospective value in feature manifest")
    return pd.DataFrame(collisions)


def assemble_training_view(features, targets, manifest, target_name, valid_column):
    """Optional join contract for later phases, without training or splitting.

    X is ALWAYS selected from the canonical feature frame, never the target
    frame. Qualified x__/y__ names eliminate existing historical-name collisions.
    """
    if not manifest.final_audit_status.isin(["SAFE", "SAFE_WITH_ASSUMPTION"]).all():
        raise ValueError("Manifest is not approved")
    names = manifest.feature_name.tolist()
    if len(names) != len(set(names)) or any(n.startswith(("demand_target_", "demand_day_", "purchase_price_target_", "price_direction_", "y__")) or "retrospective" in n for n in names):
        raise ValueError("Invalid feature manifest")
    if target_name in names and target_name not in LEGACY_NAME_COLLISIONS:
        raise ValueError("Target is listed as feature")
    x = features[KEYS + names].copy().rename(columns={n: "x__" + n for n in names})
    y = targets.loc[targets[valid_column] & targets[target_name].notna(), KEYS + [target_name]].copy()
    y = y.rename(columns={target_name: "y__" + target_name})
    x["date"], y["date"] = pd.to_datetime(x.date), pd.to_datetime(y.date)
    return x.merge(y, on=KEYS, validate="one_to_one")


def validate_targets(targets, config):
    """All-row assertions: strict future dates, complete coverage, positive
    prices, bounded selection offsets, and consistent change/direction labels.
    Feature availability is inherited from the approved Phase 8 registry;
    no invented per-feature observation timestamps are claimed here.
    """
    t = targets
    if t[KEYS].isna().any().any() or t.duplicated(KEYS).any():
        raise ValueError("Target primary key is not unique/non-null")
    if not t.feature_information_upper_bound.le(t.date).all():
        raise ValueError("Feature information after origin")
    prior = None
    for h in config["horizons"]:
        d = f"demand_target_{h}d"
        valid = t[d + "_valid"]
        if not valid.equals(t[d].notna()):
            raise ValueError("Demand validity and missingness disagree")
        if not (t.loc[valid, d] >= 0).all() or not np.isfinite(t.loc[valid, d]).all():
            raise ValueError("Invalid demand target")
        if not t[d + "_coverage"].eq(t[d + "_days_available"] / h).all():
            raise ValueError("Incorrect coverage metadata")
        if not t[d + "_days_required"].eq(h).all() or not t[d + "_days_available"].between(0, h).all():
            raise ValueError("Invalid demand day counts")
        if not valid.eq(t[d + "_days_available"].eq(h)).all():
            raise ValueError("Incomplete cumulative target")
        for suffix, offset in [("_start_date", 1), ("_end_date", h)]:
            if not t.loc[valid, d + suffix].eq(t.loc[valid, "date"] + pd.Timedelta(days=offset)).all():
                raise ValueError("Demand temporal boundary violation")
        if prior:
            common = valid & t[prior + "_valid"]
            if not (t.loc[common, d] + 1e-8 >= t.loc[common, prior]).all():
                raise ValueError("Nonmonotonic cumulative demand")
        prior = d
        p = f"purchase_price_target_{h}d"
        pv = t[p + "_valid"]
        if not pv.equals(t[p].notna()) or not (t.loc[pv, p] > 0).all() or not np.isfinite(t.loc[pv, p]).all():
            raise ValueError("Invalid future price")
        if not t.loc[pv, p + "_observation_date"].gt(t.loc[pv, "date"]).all():
            raise ValueError("Price observation must be strictly after origin")
        if not t[p + "_requested_date"].eq(t.date + pd.Timedelta(days=h)).all():
            raise ValueError("Incorrect price requested date")
        ordinary = pv & t[p + "_selection_method"].isin(["EXACT", "NEXT_WITHIN_TOLERANCE"])
        delta = (t[p + "_observation_date"] - t[p + "_requested_date"]).dt.days
        if not np.allclose(delta, t[p + "_date_difference_days"], equal_nan=True):
            raise ValueError("Incorrect price date difference metadata")
        if not t.loc[pv, p + "_selection_method"].isin(["EXACT", "NEXT_WITHIN_TOLERANCE", "EFFECTIVE_INTERVAL"]).all():
            raise ValueError("Unknown price selection method")
        if not delta[t[p + "_selection_method"].eq("EXACT")].eq(0).all():
            raise ValueError("Exact price is not on requested date")
        if not delta[ordinary].between(0, config["price_target"]["tolerance_days"]).all():
            raise ValueError("Price tolerance exceeded")
        effective = pv & t[p + "_selection_method"].eq("EFFECTIVE_INTERVAL")
        if effective.any():
            if not config["price_target"]["allow_effective_price_carry_forward"]:
                raise ValueError("Unapproved carry")
            if not (-delta[effective]).between(0, config["price_target"]["maximum_carry_forward_days"]).all():
                raise ValueError("Effective price too old")
            if not (t.loc[effective, p + "_effective_from"] <= t.loc[effective, p + "_requested_date"]).all() or not (t.loc[effective, p + "_effective_until"] >= t.loc[effective, p + "_requested_date"]).all():
                raise ValueError("Effective interval does not cover target date")
        current_ok = np.isfinite(t.purchase_price_at_origin) & t.purchase_price_at_origin.gt(0)
        if not t.loc[current_ok, "purchase_price_origin_observation_date"].le(t.loc[current_ok, "date"]).all():
            raise ValueError("Future baseline price")
        age = (t.date - t.purchase_price_origin_observation_date).dt.days
        if not age[current_ok].between(0, config["price_target"]["current_price_maximum_age_days"]).all():
            raise ValueError("Stale baseline price")
        pct = f"purchase_price_pct_change_{h}d"
        expected = ((t[p] - t.purchase_price_at_origin) / t.purchase_price_at_origin).where(current_ok)
        if not np.allclose(expected, t[pct], equal_nan=True):
            raise ValueError("Incorrect percentage change")
        if not np.isfinite(t.loc[t[pct].notna(), pct]).all():
            raise ValueError("Nonfinite price change")
        if not t[f"price_change_{h}d_valid"].eq(t[pct].notna()).all():
            raise ValueError("Incorrect price-change validity")
        flat = config["price_target"]["flat_tolerance"]
        labels = pd.Series(np.where(t[pct] > flat + 1e-12, "UP", np.where(t[pct] < -flat - 1e-12, "DOWN", "FLAT")), index=t.index).where(t[pct].notna())
        if not labels.fillna("MISSING").eq(t[f"price_direction_{h}d"].fillna("MISSING")).all():
            raise ValueError("Direction does not agree with price change")
    return {"rows_checked": len(t), "status": "PASS", "feature_bound_basis": "Approved Phase 8 metadata; unchanged canonical feature artifact"}
