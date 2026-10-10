"""Model-ready matrix cache builder and loader for Phase 13 Feature Ablation."""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.utils.paths import PROJECT_ROOT
from .config import load_config, load_demand_feature_manifest, load_price_feature_manifest


def read_before(path: Path, boundary: pd.Timestamp, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Filter rows before boundary, handling string vs timestamp date schema types."""
    data = ds.dataset(str(path), format='parquet')
    date_type = data.schema.field('date').type
    if pa.types.is_string(date_type) or pa.types.is_large_string(date_type):
        limit = str(pd.Timestamp(boundary).date())
    else:
        limit = pd.Timestamp(boundary).to_pydatetime()
    table = data.to_table(columns=columns, filter=ds.field('date') < limit)
    return table.to_pandas()


def build_model_ready_caches(root: Path = PROJECT_ROOT, verbose: bool = True) -> Dict[str, Path]:
    """Create reusable model-ready parquet caches in cache/model_ready/."""
    root = Path(root)
    cfg = load_config(root)
    horizons = cfg['evaluation']['horizons']
    holdout_start = pd.Timestamp(cfg['evaluation']['holdout_start'])

    cache_dir = root / "cache/model_ready"
    cache_dir.mkdir(parents=True, exist_ok=True)

    demand_manifest = load_demand_feature_manifest(root)["feature_name"].tolist()
    price_manifest = load_price_feature_manifest(root)["feature_name"].tolist()

    if verbose:
        print("--- Building Reusable Model-Ready Caches ---", flush=True)
        print(f"Holdout start: {holdout_start.date()}", flush=True)
        print(f"Horizons: {horizons}", flush=True)

    # 1. Load pre-holdout features
    feat_df = read_before(root / "data/processed/model_features.parquet", holdout_start)
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    # 2. Load pre-holdout targets
    targ_df = read_before(root / "data/processed/forecast_targets.parquet", holdout_start)
    targ_df["date"] = pd.to_datetime(targ_df["date"])

    # 3. Load baseline metadata for segment flags
    base_preds = pd.read_parquet(root / "data/predictions/baseline_price_predictions.parquet")
    base_preds["date"] = pd.to_datetime(base_preds["date"])
    
    # Metadata columns for segmentations
    meta_cols = [
        'date', 'store_id', 'product_id', 'horizon', 'validation_fold',
        'current_price', 'actual', 'promotion_period', 'festival_period',
        'high_volatility', 'policy_change', 'commodity_shock', 'price_shock',
        'category', 'supplier_id'
    ]
    meta_cols = [c for c in meta_cols if c in base_preds.columns]
    unique_meta = base_preds[meta_cols].drop_duplicates(subset=['date', 'store_id', 'product_id', 'horizon'])

    # Merge features with targets on primary key
    key_cols = ['date', 'store_id', 'product_id']
    merged_df = pd.merge(feat_df, targ_df, on=key_cols, how='inner', suffixes=('', '__targ'))

    created_caches = {}

    # Build Demand caches
    for h in horizons:
        cache_path = cache_dir / f"demand_{h}d.parquet"
        targ_col = f"demand_target_{h}d"
        valid_col = f"demand_target_{h}d_valid"

        demand_cols = key_cols + demand_manifest + [targ_col, valid_col]
        # Include product/store static info and segment flags if in merged_df
        aux_cols = [c for c in ['product_shelf_life_days', 'product_imported_flag', 'promo_active_flag', 'major_festival_window_active'] if c in merged_df.columns and c not in demand_cols]
        d_df = merged_df[demand_cols + aux_cols].copy()

        # Join fold and segment metadata for horizon h
        h_meta = unique_meta[unique_meta['horizon'] == h]
        d_df = pd.merge(d_df, h_meta[['date', 'store_id', 'product_id', 'validation_fold', 'promotion_period', 'festival_period', 'high_volatility', 'policy_change', 'commodity_shock', 'price_shock', 'category']], on=key_cols, how='left')

        # Optimize numeric dtypes to float32
        for col in demand_manifest:
            d_df[col] = d_df[col].astype(np.float32)
        d_df[targ_col] = d_df[targ_col].astype(np.float32)

        d_df.to_parquet(cache_path, index=False)
        created_caches[f"demand_{h}d"] = cache_path
        if verbose:
            print(f"Created {cache_path.name} ({len(d_df)} rows, {len(d_df.columns)} cols)", flush=True)

    # Build Price caches
    from src.models.governance import assert_no_target_leakage

    # Validate that price_manifest itself has zero target leakage
    assert_no_target_leakage(price_manifest)

    for h in horizons:
        cache_path = cache_dir / f"price_{h}d.parquet"
        targ_col = f"purchase_price_target_{h}d"
        valid_col = f"purchase_price_target_{h}d_valid"
        dir_col = f"price_direction_{h}d"

        # Explicitly isolate features from targets.
        # NEVER overwrite or replace backward feature with forward target!
        # The backward-looking feature 'purchase_price_pct_change_1d' / '30d' in price_manifest
        # must remain purely backward-looking.
        price_cols = key_cols + price_manifest + [targ_col, valid_col]
        if dir_col in merged_df.columns:
            price_cols.append(dir_col)

        p_df = merged_df[price_cols].copy()

        # Join fold and segment metadata for horizon h (includes current_price and actual)
        h_meta = unique_meta[unique_meta['horizon'] == h]
        p_df = pd.merge(
            p_df,
            h_meta[['date', 'store_id', 'product_id', 'validation_fold', 'current_price', 'actual', 'promotion_period', 'festival_period', 'high_volatility', 'policy_change', 'commodity_shock', 'price_shock', 'category', 'supplier_id']],
            on=key_cols,
            how='left'
        )

        for col in price_manifest:
            p_df[col] = p_df[col].astype(np.float32)
        p_df[targ_col] = p_df[targ_col].astype(np.float32)

        # Enforce that no target columns exist within the feature manifest
        assert_no_target_leakage(price_manifest, target_columns=[targ_col, valid_col])

        p_df.to_parquet(cache_path, index=False)
        created_caches[f"price_{h}d"] = cache_path
        if verbose:
            print(f"Created {cache_path.name} ({len(p_df)} rows, {len(p_df.columns)} cols)", flush=True)

    return created_caches


def load_model_ready_cache(task: str, horizon: int, root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load model-ready cached parquet dataframe for task and horizon."""
    cache_path = Path(root) / f"cache/model_ready/{task}_{horizon}d.parquet"
    if not cache_path.exists():
        build_model_ready_caches(root=root, verbose=False)
    return pd.read_parquet(cache_path)
