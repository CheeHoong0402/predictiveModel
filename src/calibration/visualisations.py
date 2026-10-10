"""Visualization suite for demand and purchase-price uncertainty calibration."""
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT


def plot_demand_forecast_with_intervals(
    df_cal_demand: pd.DataFrame,
    out_dir: Path,
    store_id: str = "STR_JHR_003",
    product_id: str = "PRD_DRK_001",
    horizon: int = 7
) -> Path:
    """Plot time series of actual demand vs P50 forecast with calibrated P10-P90 band."""
    sub = df_cal_demand[
        (df_cal_demand["store_id"] == store_id) &
        (df_cal_demand["product_id"] == product_id) &
        (df_cal_demand["horizon"] == horizon)
    ].sort_values("forecast_origin")

    fig, ax = plt.subplots(figsize=(12, 6))
    dates = pd.to_datetime(sub["forecast_origin"])

    ax.plot(dates, sub["actual_demand"], label="Actual Demand", color="#1f77b4", marker="o", linewidth=2)
    ax.plot(dates, sub["calibrated_p50"], label="Calibrated P50 (Median)", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.fill_between(
        dates,
        sub["calibrated_p10"],
        sub["calibrated_p90"],
        color="#ff7f0e",
        alpha=0.25,
        label="80% Calibrated Interval (P10 - P90)"
    )

    ax.set_title(f"Demand Forecast with Calibrated 80% Uncertainty Band ({horizon}d Ahead)\n{store_id} - {product_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Origin Date", fontsize=11)
    ax.set_ylabel("Demand (Units)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()

    out_path = out_dir / "demand_forecast_with_intervals.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_demand_coverage_by_horizon(
    df_summary: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot nominal vs raw vs calibrated demand interval coverage across all horizons."""
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = df_summary["horizon"].astype(str) + "d"
    x = np.arange(len(horizons))
    width = 0.35

    ax.bar(x - width/2, df_summary["raw_coverage"] * 100, width, label="Raw Uncalibrated Coverage", color="#aec7e8")
    ax.bar(x + width/2, df_summary["calibrated_coverage"] * 100, width, label="Calibrated Coverage", color="#1f77b4")
    ax.axhline(80, color="red", linestyle="--", linewidth=1.5, label="Nominal Target (80%)")
    ax.axhspan(77, 83, color="green", alpha=0.15, label="Target Tolerance Zone (77% - 83%)")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylim(50, 100)
    ax.set_title("Demand 80% Prediction Interval Coverage by Horizon", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = out_dir / "demand_coverage_by_horizon.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_demand_interval_width_by_horizon(
    df_summary: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot mean demand interval width across horizons before and after calibration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = df_summary["horizon"].astype(str) + "d"
    x = np.arange(len(horizons))
    width = 0.35

    ax.bar(x - width/2, df_summary["raw_mean_width"], width, label="Raw Mean Width", color="#ffbb78")
    ax.bar(x + width/2, df_summary["calibrated_mean_width"], width, label="Calibrated Mean Width", color="#d62728")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title("Demand Interval Width (Units) Across Forecast Horizons", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Mean Interval Width (P90 - P10 Units)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()

    out_path = out_dir / "demand_interval_width_by_horizon.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_demand_calibration_before_after(
    df_summary: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Comparison plot of demand coverage error and Winkler score before vs after calibration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    horizons = df_summary["horizon"].astype(str) + "d"
    x = np.arange(len(horizons))
    width = 0.35

    # Absolute Coverage Error
    raw_err = np.abs(df_summary["raw_coverage"] - 0.80) * 100
    cal_err = np.abs(df_summary["calibrated_coverage"] - 0.80) * 100

    ax1.bar(x - width/2, raw_err, width, label="Raw Error (|Cov - 80%|)", color="#98df8a")
    ax1.bar(x + width/2, cal_err, width, label="Calibrated Error", color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_title("Absolute Coverage Error from Nominal 80%", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Percentage Points Error", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Winkler Score
    ax2.bar(x - width/2, df_summary["raw_winkler_score"], width, label="Raw Winkler Score", color="#c5b0d5")
    ax2.bar(x + width/2, df_summary["calibrated_winkler_score"], width, label="Calibrated Winkler Score", color="#9467bd")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_title("Winkler Interval Score (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Winkler Score", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.suptitle("Demand Uncertainty Calibration: Before vs After", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / "demand_calibration_before_after.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_price_forecast_with_intervals(
    df_cal_price: pd.DataFrame,
    out_dir: Path,
    store_id: str = "STR_JHR_003",
    product_id: str = "PRD_DRK_001",
    horizon: int = 14
) -> Path:
    """Plot time series of actual future purchase price vs forecast with calibrated P10-P90 band."""
    sub = df_cal_price[
        (df_cal_price["store_id"] == store_id) &
        (df_cal_price["product_id"] == product_id) &
        (df_cal_price["horizon"] == horizon)
    ].sort_values("forecast_origin")

    fig, ax = plt.subplots(figsize=(12, 6))
    dates = pd.to_datetime(sub["forecast_origin"])

    ax.plot(dates, sub["actual_future_price"], label="Actual Future Price", color="#1f77b4", marker="s", linewidth=2)
    ax.plot(dates, sub["calibrated_p50"], label="Calibrated P50 (Forecast Price)", color="#2ca02c", linestyle="--", linewidth=2)
    ax.fill_between(
        dates,
        sub["calibrated_p10"],
        sub["calibrated_p90"],
        color="#2ca02c",
        alpha=0.25,
        label="80% Calibrated Interval (P10 - P90)"
    )

    ax.set_title(f"Purchase Price Forecast with Calibrated 80% Uncertainty Band ({horizon}d Ahead)\n{store_id} - {product_id}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Origin Date", fontsize=11)
    ax.set_ylabel("Price (MYR)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()

    out_path = out_dir / "price_forecast_with_intervals.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_price_coverage_by_horizon(
    df_summary: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot nominal vs raw vs calibrated price interval coverage across horizons."""
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = df_summary["horizon"].astype(str) + "d"
    x = np.arange(len(horizons))
    width = 0.35

    ax.bar(x - width/2, df_summary["raw_coverage"] * 100, width, label="Raw Uncalibrated Coverage (Over-covered)", color="#c7c7c7")
    ax.bar(x + width/2, df_summary["calibrated_coverage"] * 100, width, label="Calibrated Coverage", color="#2ca02c")
    ax.axhline(80, color="red", linestyle="--", linewidth=1.5, label="Nominal Target (80%)")
    ax.axhspan(77, 83, color="green", alpha=0.15, label="Target Tolerance Zone (77% - 83%)")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_ylim(60, 105)
    ax.set_title("Purchase-Price 80% Prediction Interval Coverage by Horizon", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Empirical Coverage (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()

    out_path = out_dir / "price_coverage_by_horizon.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_price_interval_width_by_horizon(
    df_summary: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Plot mean price interval width in MYR across horizons before and after calibration."""
    fig, ax = plt.subplots(figsize=(10, 5))
    horizons = df_summary["horizon"].astype(str) + "d"
    x = np.arange(len(horizons))
    width = 0.35

    ax.bar(x - width/2, df_summary["raw_mean_width"], width, label="Raw Mean Width (Excessive)", color="#e377c2")
    ax.bar(x + width/2, df_summary["calibrated_mean_width"], width, label="Calibrated Mean Width (Sharpened)", color="#17becf")

    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.set_title("Purchase-Price Interval Width (MYR) Across Forecast Horizons", fontsize=14, fontweight="bold")
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Mean Interval Width (MYR)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()

    out_path = out_dir / "price_interval_width_by_horizon.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_price_calibration_before_after(
    df_summary: pd.DataFrame,
    out_dir: Path
) -> Path:
    """Comparison plot of price coverage error and Winkler score before vs after calibration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    horizons = df_summary["horizon"].astype(str) + "d"
    x = np.arange(len(horizons))
    width = 0.35

    # Coverage Error
    raw_err = np.abs(df_summary["raw_coverage"] - 0.80) * 100
    cal_err = np.abs(df_summary["calibrated_coverage"] - 0.80) * 100

    ax1.bar(x - width/2, raw_err, width, label="Raw Overcoverage Error", color="#bcbd22")
    ax1.bar(x + width/2, cal_err, width, label="Calibrated Error", color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_title("Absolute Coverage Error from Nominal 80%", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Percentage Points Error", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Winkler Score
    ax2.bar(x - width/2, df_summary["raw_winkler_score"], width, label="Raw Winkler Score", color="#ff9896")
    ax2.bar(x + width/2, df_summary["calibrated_winkler_score"], width, label="Calibrated Winkler Score", color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_title("Winkler Interval Score (Lower is Better)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Winkler Score (MYR)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.suptitle("Purchase-Price Uncertainty Calibration: Before vs After", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = out_dir / "price_calibration_before_after.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
