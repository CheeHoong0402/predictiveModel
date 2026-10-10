"""Visualisation generator for Phase 15 Ensemble Modelling."""
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.paths import PROJECT_ROOT


def generate_ensemble_plots(
    comparison_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: Path = PROJECT_ROOT / "reports/figures/ensemble"
) -> None:
    """Generate all required visualization figures for Phase 15."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Baseline vs Single vs Ensemble
    c_df = comparison_df.rename(columns={c: c.lower().replace(" ", "_") for c in comparison_df.columns})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    d_comp = c_df[c_df["task"] == "demand"]
    p_comp = c_df[c_df["task"] == "price"]

    # Demand subplot
    if not d_comp.empty:
        x = np.arange(len(d_comp))
        w = 0.25
        axes[0].bar(x - w, d_comp["champion_baseline_metric"] * 100, width=w, label="Champion Baseline", color="#9E9E9E")
        axes[0].bar(x, d_comp["best_single_model_metric"] * 100, width=w, label="Best Single Model", color="#2196F3")
        axes[0].bar(x + w, d_comp["weighted_ensemble_metric"] * 100, width=w, label="Weighted Ensemble", color="#4CAF50")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f"{h}d" for h in d_comp["horizon"]])
        axes[0].set_ylabel("WAPE (%)")
        axes[0].set_title("Demand Forecasting: Baseline vs Single vs Ensemble")
        axes[0].legend()

    # Price subplot
    if not p_comp.empty:
        x = np.arange(len(p_comp))
        w = 0.25
        axes[1].bar(x - w, p_comp["champion_baseline_metric"], width=w, label="Champion Baseline", color="#9E9E9E")
        axes[1].bar(x, p_comp["best_single_model_metric"], width=w, label="Best Single Model", color="#2196F3")
        axes[1].bar(x + w, p_comp["weighted_ensemble_metric"], width=w, label="Weighted Ensemble", color="#FF9800")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"{h}d" for h in p_comp["horizon"]])
        axes[1].set_ylabel("MAE (MYR)")
        axes[1].set_title("Purchase-Price Forecasting: Baseline vs Single vs Ensemble")
        axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_dir / "baseline_vs_single_vs_ensemble.png", dpi=200)
    plt.close(fig)

    # 2. Error by Horizon
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not d_comp.empty:
        ax.plot(d_comp["horizon"], d_comp["best_single_model_metric"] * 100, marker="o", label="Best Single (Demand WAPE %)", color="#2196F3")
        ax.plot(d_comp["horizon"], d_comp["weighted_ensemble_metric"] * 100, marker="s", linestyle="--", label="Ensemble (Demand WAPE %)", color="#4CAF50")
    ax.set_xlabel("Forecast Horizon (Days)")
    ax.set_ylabel("Error Metric (%)")
    ax.set_title("Demand Forecast Error Trajectory Across Horizons")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "error_by_horizon.png", dpi=200)
    plt.close(fig)

    # 3. Ensemble Weights
    if not weights_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        d_w = weights_df[weights_df["task"] == "demand"]
        if not d_w.empty:
            pvt_w = d_w.pivot_table(index="horizon", columns="model", values="weight", fill_value=0.0)
            pvt_w.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
            ax.set_ylabel("Model Weight")
            ax.set_title("Demand Ensemble Component Weights by Horizon")
            ax.set_xticklabels([f"{h}d" for h in pvt_w.index], rotation=0)
            ax.legend(title="Member Model")
        plt.tight_layout()
        fig.savefig(output_dir / "ensemble_weights.png", dpi=200)
        plt.close(fig)

    # 4. Fold Stability Comparison
    if not stability_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        d_stab = stability_df[stability_df["task"] == "demand"]
        if not d_stab.empty:
            for m_name, grp in d_stab.groupby("model_or_ensemble"):
                ax.errorbar(grp["horizon"], grp["mean_error"] * 100, yerr=grp["std_error"] * 100, marker="o", label=m_name, capsize=4)
            ax.set_xlabel("Horizon (Days)")
            ax.set_ylabel("Mean Fold Error ± 1 Std (%)")
            ax.set_title("Walk-Forward Fold Error Stability")
            ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / "fold_stability.png", dpi=200)
        plt.close(fig)

    # 5. Model Residual Correlation Heatmap
    if not diversity_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sub_div = diversity_df[diversity_df["task"] == "demand"]
        if not sub_div.empty:
            pairs = sub_div["model_A"] + " vs " + sub_div["model_B"]
            y_pos = np.arange(len(pairs))
            ax.barh(y_pos, sub_div["residual_correlation"], color="#3F51B5")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pairs)
            ax.set_xlabel("Residual Correlation")
            ax.set_title("Demand Model Residual Diversity Across Pairs")
            ax.set_xlim(0.8, 1.0)
        plt.tight_layout()
        fig.savefig(output_dir / "model_residual_correlation.png", dpi=200)
        plt.close(fig)

    # 6. Worst-Case Error Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not d_comp.empty:
        ax.plot(d_comp["horizon"], d_comp["best_single_model_metric"] * 100, marker="o", label="Best Single (Mean)", color="#2196F3")
        ax.plot(d_comp["horizon"], d_comp["weighted_ensemble_metric"] * 100, marker="s", label="Ensemble (Mean)", color="#4CAF50")
    ax.set_xlabel("Horizon (Days)")
    ax.set_ylabel("Error Metric (%)")
    ax.set_title("Worst-Case and Tail Error Trajectory")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "worst_case_error_comparison.png", dpi=200)
    plt.close(fig)
