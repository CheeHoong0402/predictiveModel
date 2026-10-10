import pandas as pd
from typing import List, Dict, Any

def check_required_columns(df: pd.DataFrame, required_cols: List[str], dataset_name: str = "Dataset") -> Dict[str, Any]:
    """Validates that a dataframe contains the expected columns."""
    missing = [col for col in required_cols if col not in df.columns]
    status = len(missing) == 0
    return {
        "dataset": dataset_name,
        "valid": status,
        "missing_columns": missing,
        "total_columns": len(df.columns),
        "total_rows": len(df)
    }

def summarize_dataframe_health(df: pd.DataFrame, date_col: str = None) -> Dict[str, Any]:
    """Computes basic health metrics including nulls, duplicates, and date span."""
    health = {
        "rows": len(df),
        "columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }
    if date_col and date_col in df.columns:
        date_series = pd.to_datetime(df[date_col], errors="coerce")
        health["date_min"] = str(date_series.min())
        health["date_max"] = str(date_series.max())
    return health
