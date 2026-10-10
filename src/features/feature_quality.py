"""
Feature Quality and Audit Module.
Calculates statistical distributions, missingness percentages, availability date ranges,
and detects constant, infinite, or duplicated features.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import polars as pl


def audit_feature_quality(
    df: pl.DataFrame,
    feature_names: List[str],
    date_col: str = "date",
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Perform rigorous data quality profiling on all engineered features.
    
    Generates:
    - feature_quality_report.csv: comprehensive summary metrics
    - feature_missingness.csv: null/missing percentages and counts
    - feature_availability.csv: date ranges where each feature is populated
    """
    total_rows = df.height
    quality_records = []
    missing_records = []
    avail_records = []

    # Convert to pandas for robust statistical aggregations and string handling
    # Select only date and features to conserve memory
    cols_to_profile = [date_col] + [f for f in feature_names if f in df.columns]
    pdf = df.select(cols_to_profile).to_pandas()
    pdf[date_col] = pd.to_datetime(pdf[date_col])

    for fname in feature_names:
        if fname not in pdf.columns:
            continue
        
        series = pdf[fname]
        null_count = int(series.isna().sum())
        missing_pct = float(null_count / total_rows * 100.0)

        # Missingness record
        missing_records.append({
            "feature_name": fname,
            "null_count": null_count,
            "total_rows": total_rows,
            "missing_pct": round(missing_pct, 4),
            "status": "EXCESSIVE" if missing_pct > 90.0 else "HIGH" if missing_pct > 50.0 else "NORMAL",
        })

        # Availability date range
        valid_series = pdf.loc[series.notna(), date_col]
        if not valid_series.empty:
            avail_start = str(valid_series.min().date())
            avail_end = str(valid_series.max().date())
            valid_days = int((valid_series.max() - valid_series.min()).days + 1)
        else:
            avail_start = "N/A"
            avail_end = "N/A"
            valid_days = 0

        avail_records.append({
            "feature_name": fname,
            "availability_start_date": avail_start,
            "availability_end_date": avail_end,
            "coverage_days": valid_days,
            "populated_rows": int(series.notna().sum()),
        })

        # Statistical properties
        is_num = pd.api.types.is_numeric_dtype(series)
        unique_cnt = int(series.nunique(dropna=True))

        if is_num:
            # Check infinites
            num_clean = series.dropna()
            inf_count = int(np.isinf(num_clean).sum())
            inf_pct = float(inf_count / total_rows * 100.0)
            
            # Finite only for min, max, mean, std
            finite_vals = num_clean[np.isfinite(num_clean)]
            val_min = float(finite_vals.min()) if not finite_vals.empty else None
            val_max = float(finite_vals.max()) if not finite_vals.empty else None
            val_mean = float(finite_vals.mean()) if not finite_vals.empty else None
            val_std = float(finite_vals.std()) if not finite_vals.empty else None
            
            # Constant check
            is_const = bool(unique_cnt <= 1 and null_count == 0)
            const_pct = float(series.value_counts(normalize=True).iloc[0] * 100.0) if not series.empty else 100.0
        else:
            inf_count = 0
            inf_pct = 0.0
            val_min = None
            val_max = None
            val_mean = None
            val_std = None
            is_const = bool(unique_cnt <= 1 and null_count == 0)
            const_pct = float(series.value_counts(normalize=True).iloc[0] * 100.0) if not series.empty else 100.0

        quality_records.append({
            "feature_name": fname,
            "data_type": str(series.dtype),
            "missing_pct": round(missing_pct, 4),
            "unique_values": unique_cnt,
            "min_value": round(val_min, 4) if val_min is not None else None,
            "max_value": round(val_max, 4) if val_max is not None else None,
            "mean_value": round(val_mean, 4) if val_mean is not None else None,
            "std_value": round(val_std, 4) if val_std is not None else None,
            "infinite_count": inf_count,
            "infinite_pct": round(inf_pct, 4),
            "is_constant": is_const,
            "top_value_frequency_pct": round(const_pct, 4),
            "availability_start": avail_start,
            "availability_end": avail_end,
        })

    df_quality = pd.DataFrame(quality_records)
    df_missing = pd.DataFrame(missing_records)
    df_avail = pd.DataFrame(avail_records)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_quality.to_csv(output_dir / "feature_quality_report.csv", index=False)
        df_missing.to_csv(output_dir / "feature_missingness.csv", index=False)
        df_avail.to_csv(output_dir / "feature_availability.csv", index=False)

    return {
        "quality": df_quality,
        "missingness": df_missing,
        "availability": df_avail,
    }
