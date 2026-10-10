"""Configuration loader and feature manifest manager for Phase 13 Feature Ablation."""
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT


def load_config(root: Path = PROJECT_ROOT) -> dict:
    """Load config/feature_ablation.yaml."""
    cfg_path = Path(root) / "config/feature_ablation.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Feature ablation config not found at: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_demand_feature_manifest(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load safe demand feature manifest from Phase 8."""
    path = Path(root) / "reports/leakage/demand_features_temporally_safe.csv"
    if not path.exists():
        raise FileNotFoundError(f"Safe demand feature manifest not found: {path}")
    df = pd.read_csv(path)
    # Ensure audit status is safe (SAFE or SAFE_WITH_ASSUMPTION)
    safe_df = df[df["final_audit_status"].isin(["SAFE", "SAFE_WITH_ASSUMPTION"])]
    return safe_df


def load_price_feature_manifest(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load safe price feature manifest from Phase 8."""
    path = Path(root) / "reports/leakage/price_features_temporally_safe.csv"
    if not path.exists():
        raise FileNotFoundError(f"Safe price feature manifest not found: {path}")
    df = pd.read_csv(path)
    safe_df = df[df["final_audit_status"].isin(["SAFE", "SAFE_WITH_ASSUMPTION"])]
    return safe_df


def get_demand_family_to_features(root: Path = PROJECT_ROOT) -> Dict[str, List[str]]:
    """Group approved safe demand features by higher-level family group.
    
    Returns a dict mapping group_name -> list of feature_names.
    """
    manifest_df = load_demand_feature_manifest(root)
    cfg = load_config(root)
    groups_cfg = cfg["demand_family_groups"]

    tag_to_features = manifest_df.groupby("feature_family")["feature_name"].apply(list).to_dict()

    group_to_features: Dict[str, List[str]] = {}
    assigned_features = set()

    for group_name, tags in groups_cfg.items():
        feats = []
        for tag in tags:
            if tag in tag_to_features:
                feats.extend(tag_to_features[tag])
        feats = sorted(list(set(feats)))
        group_to_features[group_name] = feats
        assigned_features.update(feats)

    # Check for any unassigned safe features
    all_safe_features = set(manifest_df["feature_name"])
    unassigned = all_safe_features - assigned_features
    if unassigned:
        # Fallback group if any
        group_to_features["other_features"] = sorted(list(unassigned))

    return group_to_features


def get_price_family_to_features(root: Path = PROJECT_ROOT) -> Dict[str, List[str]]:
    """Group approved safe price features by higher-level family group.
    
    Returns a dict mapping group_name -> list of feature_names.
    """
    manifest_df = load_price_feature_manifest(root)
    cfg = load_config(root)
    groups_cfg = cfg["price_family_groups"]

    tag_to_features = manifest_df.groupby("feature_family")["feature_name"].apply(list).to_dict()

    group_to_features: Dict[str, List[str]] = {}
    assigned_features = set()

    for group_name, tags in groups_cfg.items():
        feats = []
        for tag in tags:
            if tag in tag_to_features:
                feats.extend(tag_to_features[tag])
        feats = sorted(list(set(feats)))
        group_to_features[group_name] = feats
        assigned_features.update(feats)

    all_safe_features = set(manifest_df["feature_name"])
    unassigned = all_safe_features - assigned_features
    if unassigned:
        group_to_features["other_features"] = sorted(list(unassigned))

    return group_to_features


def get_buildup_feature_sets(task: str, root: Path = PROJECT_ROOT) -> List[Tuple[str, List[str]]]:
    """Retrieve ordered build-up stages and cumulative feature lists for a task."""
    cfg = load_config(root)
    if task == "demand":
        family_map = get_demand_family_to_features(root)
        manifest_df = load_demand_feature_manifest(root)
        all_features = sorted(manifest_df["feature_name"].tolist())
        stages = cfg["buildup_stages"]["demand"]
    elif task == "price":
        family_map = get_price_family_to_features(root)
        manifest_df = load_price_feature_manifest(root)
        all_features = sorted(manifest_df["feature_name"].tolist())
        stages = cfg["buildup_stages"]["price"]
    else:
        raise ValueError(f"Unknown task: {task}")

    result = []
    for stage in stages:
        name = stage["name"]
        fams = stage["families"]
        if fams == "ALL":
            stage_feats = all_features
        else:
            feats = set()
            for f in fams:
                if f in family_map:
                    feats.update(family_map[f])
            stage_feats = sorted(list(feats))
        result.append((name, stage_feats))
    return result


def get_internal_vs_external_feature_sets(task: str, root: Path = PROJECT_ROOT) -> Tuple[List[str], List[str]]:
    """Retrieve (internal_features, full_features) for internal vs external data testing."""
    cfg = load_config(root)
    if task == "demand":
        family_map = get_demand_family_to_features(root)
        manifest_df = load_demand_feature_manifest(root)
        all_features = sorted(manifest_df["feature_name"].tolist())
        internal_fams = cfg["internal_vs_external"]["demand"]["internal_families"]
    elif task == "price":
        family_map = get_price_family_to_features(root)
        manifest_df = load_price_feature_manifest(root)
        all_features = sorted(manifest_df["feature_name"].tolist())
        internal_fams = cfg["internal_vs_external"]["price"]["internal_families"]
    else:
        raise ValueError(f"Unknown task: {task}")

    internal_feats = set()
    for fam in internal_fams:
        if fam in family_map:
            internal_feats.update(family_map[fam])

    return sorted(list(internal_feats)), all_features


def validate_ablation_features(
    features_to_use: List[str],
    full_features: List[str],
    excluded_features: List[str],
    manifest_features: List[str]
) -> bool:
    """Validate strict integrity of feature ablation sets.
    
    1. Features to use must be a subset of full features.
    2. None of the excluded features may be present.
    3. full_features minus excluded_features must match features_to_use.
    4. No target columns or unapproved features are present.
    """
    set_use = set(features_to_use)
    set_full = set(full_features)
    set_excl = set(excluded_features)
    set_manifest = set(manifest_features)

    # Must be approved features
    if not set_use.issubset(set_manifest):
        raise ValueError("features_to_use contains features not present in safe manifest!")

    # Must not contain excluded features
    overlap = set_use.intersection(set_excl)
    if overlap:
        raise ValueError(f"Excluded features leaked into features_to_use: {overlap}")

    # Must equal set_full - set_excl
    expected = set_full - set_excl
    if set_use != expected:
        raise ValueError("features_to_use does not exactly match (full_features - excluded_features)!")

    # Guard against target leakage
    for f in features_to_use:
        if "target" in f.lower():
            raise ValueError(f"Target column detected in feature set: {f}")

    return True
