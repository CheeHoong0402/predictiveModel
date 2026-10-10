"""
src/demand_correction/correction_pipeline.py
============================================
Phase 6 — Stockout and True-Demand Correction Orchestration Pipeline:
- Loads Phase 5 master daily modelling table
- Detects full-day vs partial-day stockouts via multi-tier inventory evidence
- Applies conservative reference-set demand estimation strictly past-only for training safety
- Generates retrospective comparison demand estimation
- Computes lower and upper percentile bounds, lost sales, and confidence categories
- Audits plausibility and flags high-risk extreme corrections
- Saves data/processed/master_daily_demand_corrected.parquet (preserving master_daily.parquet)
- Produces all Phase 6 audit reports under reports/demand_correction/
"""

import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd

from src.demand_correction.stockout_detector import StockoutDetector
from src.demand_correction.reference_selector import ReferenceSelector
from src.demand_correction.demand_estimator import DemandEstimator
from src.demand_correction.confidence_estimator import ConfidenceEstimator
from src.demand_correction.plausibility_checker import PlausibilityChecker

# Configure paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "demand_correction.yaml"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "demand_correction"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [correction_pipeline]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("correction_pipeline")


class DemandCorrectionPipeline:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()
        self.stockout_detector = StockoutDetector(self.config)
        self.reference_selector = ReferenceSelector(self.config)
        self.demand_estimator = DemandEstimator(self.config)
        self.confidence_estimator = ConfidenceEstimator(self.config)
        self.plausibility_checker = PlausibilityChecker(self.config)
        self.start_time = None
        self.end_time = None

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self) -> Path:
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("   PHASE 6: STARTING STOCKOUT AND TRUE-DEMAND CORRECTION PIPELINE   ")
        logger.info("=" * 70)

        # 1. Load Phase 5 Master Table
        master_path = PROCESSED_DIR / "master_daily.parquet"
        if not master_path.exists():
            raise FileNotFoundError(f"Master table not found at {master_path}")

        logger.info(f"Loading master daily modelling table from {master_path}...")
        df = pd.read_parquet(master_path)
        initial_rows = len(df)
        logger.info(f"Master table loaded: {initial_rows} rows, {len(df.columns)} columns.")

        # 2. Stockout Detection
        inv_path = INTERIM_DIR / "inventory_clean.parquet"
        logger.info("Running stockout detection with multi-tier inventory evidence...")
        df = self.stockout_detector.detect_stockouts(df, inv_path)

        # Sort by series and date
        df = df.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)

        # 3. Series-Level True Demand Estimation (Past-Only Training-Safe & Retrospective)
        logger.info("Executing conservative reference-set demand estimation across series...")

        # Preallocate result lists
        est_training_safe = []
        est_retrospective = []
        est_lost_sales = []
        corr_lower_bounds = []
        corr_upper_bounds = []
        corr_applied = []
        corr_methods = []
        corr_ref_counts = []
        corr_confidences = []

        # Group by store_id and product_id
        grouped = df.groupby(["store_id", "product_id"], as_index=False)

        for (store_id, product_id), series_df in grouped:
            series_df = series_df.sort_values(by="date").reset_index(drop=True)

            # A. Training-Safe past-only reference selection
            past_refs = self.reference_selector.select_references_for_series(series_df, past_only=True)

            # B. Retrospective offline reference selection (bidirectional window)
            retro_refs = self.reference_selector.select_references_for_series(series_df, past_only=False)

            # C. Estimate demand for each row in series
            for idx, row in series_df.iterrows():
                obs_sales = float(row["observed_units_sold"])
                is_censored = bool(row["censored_demand_flag"] == 1)
                is_partial = bool(row["partial_stockout_flag"] == 1)
                so_frac = float(row["stockout_fraction_of_day"])
                evidence_lvl = str(row["stockout_evidence_level"])

                # Past-only estimate (training-safe)
                p_ref = past_refs[idx]
                est_p = self.demand_estimator.estimate_demand(
                    observed_sales=obs_sales,
                    is_censored=is_censored,
                    is_partial=is_partial,
                    stockout_fraction_of_day=so_frac,
                    ref_sales=p_ref["ref_sales"],
                    ref_count=p_ref["ref_count"],
                    method=p_ref["method"]
                )

                # Retrospective estimate
                r_ref = retro_refs[idx]
                est_r = self.demand_estimator.estimate_demand(
                    observed_sales=obs_sales,
                    is_censored=is_censored,
                    is_partial=is_partial,
                    stockout_fraction_of_day=so_frac,
                    ref_sales=r_ref["ref_sales"],
                    ref_count=r_ref["ref_count"],
                    method=r_ref["method"]
                )

                # Confidence
                confidence = self.confidence_estimator.evaluate_confidence(
                    is_censored=is_censored,
                    correction_applied=est_p["correction_applied"],
                    ref_count=p_ref["ref_count"],
                    evidence_level=evidence_lvl
                )

                # Append results
                est_training_safe.append(est_p["estimated_true_demand"])
                est_retrospective.append(est_r["estimated_true_demand"])
                est_lost_sales.append(est_p["estimated_lost_sales"])
                corr_lower_bounds.append(est_p["lower_bound"])
                corr_upper_bounds.append(est_p["upper_bound"])
                corr_applied.append(est_p["correction_applied"])
                corr_methods.append(est_p["correction_method"])
                corr_ref_counts.append(p_ref["ref_count"])
                corr_confidences.append(confidence)

        # Assign back to DataFrame
        df["estimated_true_demand_training_safe"] = est_training_safe
        df["estimated_true_demand_retrospective"] = est_retrospective
        df["estimated_true_demand"] = df["estimated_true_demand_training_safe"]  # Default to training safe
        df["estimated_lost_sales"] = est_lost_sales
        df["demand_correction_lower_bound"] = corr_lower_bounds
        df["demand_correction_upper_bound"] = corr_upper_bounds
        df["demand_correction_applied"] = corr_applied
        df["demand_correction_method"] = corr_methods
        df["demand_correction_reference_count"] = corr_ref_counts
        df["demand_correction_confidence"] = corr_confidences

        # 4. Plausibility Audit & Extreme Outlier Flagging
        logger.info("Auditing plausibility bounds and flagging extreme corrections...")
        df, high_risk_df = self.plausibility_checker.audit_and_flag(df)

        # 5. Primary Key and Invariant Validation
        pk_dups = df.duplicated(subset=["date", "store_id", "product_id"]).sum()
        if pk_dups > 0:
            raise ValueError(f"Primary key duplication detected! Found {pk_dups} duplicate keys.")

        if len(df) != initial_rows:
            raise ValueError(f"Row count changed during correction! {initial_rows} -> {len(df)}")

        # Mathematical sanity checks
        if (df["estimated_lost_sales"] < 0).any():
            raise ValueError("Found negative estimated lost sales!")

        if (df["estimated_true_demand"] < df["observed_units_sold"]).any():
            raise ValueError("Found estimated true demand below observed sales!")

        logger.info("Mathematical and integrity assertions PASSED.")

        # 6. Save Corrected Master Dataset
        out_path = PROCESSED_DIR / "master_daily_demand_corrected.parquet"
        logger.info(f"Writing demand-corrected master table to {out_path}...")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info(f"Corrected master table saved successfully. Size: {file_size_mb:.2f} MB.")

        # 7. Generate Audit Reports
        self.end_time = time.time()
        self.generate_reports(df, high_risk_df, out_path)

        logger.info("=" * 70)
        logger.info(f"   PHASE 6 DEMAND CORRECTION COMPLETED IN {self.end_time - self.start_time:.2f}s   ")
        logger.info("=" * 70)
        return out_path

    def generate_reports(self, df: pd.DataFrame, high_risk_df: pd.DataFrame, out_path: Path):
        logger.info("Generating Phase 6 audit reports and summary documentation...")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        total_obs = len(df)
        confirmed_so = int((df["full_day_stockout_flag"] == 1).sum())
        partial_so = int((df["partial_stockout_flag"] == 1).sum())
        total_so = int((df["censored_demand_flag"] == 1).sum())
        total_corrected = int((df["demand_correction_applied"] == 1).sum())

        tot_obs_sales = float(df["observed_units_sold"].sum())
        tot_est_demand = float(df["estimated_true_demand"].sum())
        tot_lost_sales = float(df["estimated_lost_sales"].sum())
        pct_uplift = (tot_lost_sales / tot_obs_sales * 100.0) if tot_obs_sales > 0 else 0.0

        # 1. Stockout Events Report
        so_events = df[df["censored_demand_flag"] == 1][[
            "date", "store_id", "product_id", "product_name",
            "observed_units_sold", "opening_inventory", "closing_inventory",
            "stockout_duration_hours", "stockout_fraction_of_day",
            "full_day_stockout_flag", "partial_stockout_flag",
            "estimated_true_demand", "estimated_lost_sales",
            "demand_correction_method", "demand_correction_confidence"
        ]].copy()
        so_events.to_csv(REPORTS_DIR / "stockout_events.csv", index=False)

        # 2. Corrected Demand Sample Summary
        sample_corrected = df[df["demand_correction_applied"] == 1][[
            "date", "store_id", "product_id", "product_name",
            "observed_units_sold", "estimated_true_demand", "estimated_lost_sales",
            "demand_correction_lower_bound", "demand_correction_upper_bound",
            "demand_correction_confidence", "demand_correction_method"
        ]].head(500)
        sample_corrected.to_csv(REPORTS_DIR / "corrected_demand.csv", index=False)

        # 3. Correction Confidence Distribution
        conf_dist = df["demand_correction_confidence"].value_counts().reset_index()
        conf_dist.columns = ["confidence_level", "observation_count"]
        conf_dist["percentage"] = np.round(conf_dist["observation_count"] / total_obs * 100.0, 2)
        conf_dist.to_csv(REPORTS_DIR / "correction_confidence.csv", index=False)

        # 4. Lost Sales Summary by Product and Store
        prod_lost = df.groupby(["product_id", "product_name"]).agg(
            observed_sales=("observed_units_sold", "sum"),
            estimated_true_demand=("estimated_true_demand", "sum"),
            estimated_lost_sales=("estimated_lost_sales", "sum"),
            stockout_days=("censored_demand_flag", "sum"),
            total_days=("date", "count")
        ).reset_index()
        prod_lost["stockout_rate_pct"] = np.round(prod_lost["stockout_days"] / prod_lost["total_days"] * 100.0, 2)
        prod_lost["lost_sales_pct"] = np.round(prod_lost["estimated_lost_sales"] / prod_lost["observed_sales"] * 100.0, 2)
        prod_lost.to_csv(REPORTS_DIR / "lost_sales_summary.csv", index=False)

        # 5. High Risk Extreme Corrections
        high_risk_df.to_csv(REPORTS_DIR / "high_risk_corrections.csv", index=False)

        # 6. Executive Stockout Summary Markdown
        exec_sec = (self.end_time - self.start_time) if (self.end_time and self.start_time) else 0.0
        conf_dict = dict(zip(conf_dist["confidence_level"], conf_dist["observation_count"]))

        summary_md = f"""# Phase 6 — Stockout and True-Demand Correction Summary Report

*Execution Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## 1. Executive Summary

- **Total Observations Analysed:** **{total_obs:,}**
- **Total Censored / Constrained Observations:** **{total_so:,}** ({total_so / total_obs * 100.1:.2f}%)
- **Confirmed Full-Day Stockouts:** **{confirmed_so:,}** ({confirmed_so / total_obs * 100.0:.2f}%)
- **Confirmed Partial-Day Stockouts:** **{partial_so:,}** ({partial_so / total_obs * 100.0:.2f}%)
- **Total Corrected Observations:** **{total_corrected:,}**
- **Total Observed Units Sold:** **{tot_obs_sales:,.1f} units**
- **Total Estimated True Demand:** **{tot_est_demand:,.1f} units**
- **Total Estimated Lost Sales:** **{tot_lost_sales:,.1f} units**
- **Aggregate Demand Uplift from Correction:** **+{pct_uplift:.2f}%**
- **High-Risk / Extreme Corrections Flagged:** **{len(high_risk_df):,}**
- **Execution Time:** **{exec_sec:.2f} seconds**
- **Output Artifact:** [`data/processed/master_daily_demand_corrected.parquet`](file:///{out_path.as_posix()})

---

## 2. Confidence Level Distribution

| Confidence Level | Observations | % of Total | Operational Interpretation |
|---|---|---|---|
| **HIGH** | {conf_dist.loc[conf_dist['confidence_level']=='HIGH', 'observation_count'].sum():,} | {conf_dist.loc[conf_dist['confidence_level']=='HIGH', 'percentage'].sum():.2f}% | Unconstrained observations or confirmed stockouts with $\ge 8$ strong past non-stockout matches |
| **MEDIUM** | {conf_dist.loc[conf_dist['confidence_level']=='MEDIUM', 'observation_count'].sum():,} | {conf_dist.loc[conf_dist['confidence_level']=='MEDIUM', 'percentage'].sum():.2f}% | Censored observations with 4 to 7 valid comparable past non-stockout matches |
| **LOW** | {conf_dist.loc[conf_dist['confidence_level']=='LOW', 'observation_count'].sum():,} | {conf_dist.loc[conf_dist['confidence_level']=='LOW', 'percentage'].sum():.2f}% | Censored observations with 1 to 3 valid comparable past non-stockout matches |
| **UNCORRECTED** | {conf_dist.loc[conf_dist['confidence_level']=='UNCORRECTED', 'observation_count'].sum():,} | {conf_dist.loc[conf_dist['confidence_level']=='UNCORRECTED', 'percentage'].sum():.2f}% | Initial observation period with zero preceding history (conservatively uncorrected) |

---

## 3. Stockout Rate & Lost Sales by Product

| Product ID | Product Name | Stockout Days | Stockout Rate % | Observed Sales | Estimated Demand | Estimated Lost Sales | Uplift % |
|---|---|---|---|---|---|---|---|
"""
        for r in prod_lost.to_dict(orient="records"):
            summary_md += f"| `{r['product_id']}` | {r['product_name']} | {r['stockout_days']:,} | {r['stockout_rate_pct']:.1f}% | {r['observed_sales']:,.0f} | {r['estimated_true_demand']:,.0f} | {r['estimated_lost_sales']:,.0f} | +{r['lost_sales_pct']:.1f}% |\n"

        summary_md += """
---

## 4. Methodological Invariants & Leakage Protection

1. **Past-Only Training Safety:** All default target estimates (`estimated_true_demand_training_safe`) strictly query reference observations from calendar dates before date $T$ (`ref_date < T`). Zero future lookahead.
2. **Non-Negativity Invariant:** $\\text{estimated\\_lost\\_sales} = \\max(\\text{estimated\\_demand} - \\text{observed\\_sales}, 0.0) \\ge 0.0$.
3. **Monotonicity Invariant:** $\\text{estimated\\_true\\_demand} \\ge \\text{observed\\_units\\_sold}$.
4. **Preservation of Raw & Master Datasets:** Original `master_daily.parquet` and all files under `data/raw/` remain 100% immutable.
"""

        with open(REPORTS_DIR / "stockout_summary.md", "w", encoding="utf-8") as f:
            f.write(summary_md)

        # 7. Methodology Documentation Markdown
        methodology_md = """# Phase 6 — Stockout and True-Demand Correction Methodology

## 1. Principles of Demand Censorship in Food Retail

In fast-moving food retail, customer demand is latent. When a store runs out of stock:
$$\\text{Observed Sales} = \\min(\\text{True Customer Demand}, \\text{Available Stock})$$
Using observed sales directly to train predictive inventory models introduces severe negative bias: the algorithm learns that demand is zero whenever stock is zero, perpetuating chronic stockouts.

## 2. Multi-Tier Evidence Hierarchy

Observations are classified using an empirical evidence hierarchy:
1. **Full-Day Stockouts:** Recorded when `number_of_hours_out_of_stock == 12.0` or opening, closing, and sales were zero with active stockout flag.
2. **Partial-Day Stockouts:** Recorded when `0 < number_of_hours_out_of_stock < 12.0`. Sales are partially observed during operating hours before depletion.
3. **Unconstrained Observations:** Adequate closing inventory (`closing_inventory > 0`) and zero stockout flags.

## 3. Reference-Set Construction (Past-Only Training-Safe)

To prevent lookahead leakage into ML feature pipelines, reference sets for date $T$ only draw from unconstrained observations strictly occurring before $T$ (`ref_date < T`):
- **Level 1 (Highest Quality):** Same store, same SKU, same day of week, within past 12 weeks, matching promotion regime, matching price band ($\pm 10\%$).
- **Level 2 (Extended DOW):** Same store, same SKU, same day of week, within past 26 weeks, matching promotion regime.
- **Level 3 (All-Time DOW):** Same store, same SKU, same day of week across all past history.
- **Level 4 (All Weekdays):** All days of week within past 12 weeks, matching promotion regime.
- **Level 5 (Fallback):** All unconstrained past observations for the store-SKU.

## 4. Mathematical Estimation Formulas

For Full-Day Stockouts:
$$\\hat{D}_T = \\max(\\text{Median}(R_T), S_T)$$

For Partial-Day Stockouts:
$$\\hat{D}_T = \\max\\left(S_T / (1 - f_T), \\text{Median}(R_T), S_T\\right)$$
where $f_T = \\text{stockout\\_hours} / 12.0$, and bounded by the historical 75th percentile and maximum correction ratio (3.5x).

Lost Sales:
$$\\text{Lost Sales}_T = \\max(\\hat{D}_T - S_T, 0)$$
"""
        with open(REPORTS_DIR / "methodology.md", "w", encoding="utf-8") as f:
            f.write(methodology_md)

        logger.info(f"All 7 Phase 6 reports written to {REPORTS_DIR}")


def main():
    pipeline = DemandCorrectionPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
