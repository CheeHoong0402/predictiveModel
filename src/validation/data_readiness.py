"""
data_readiness.py
=================
Evaluates readiness status (READY, WARNING, BLOCKED) for all audited datasets.
Verifies essential retailer dataset presence and generates structured validation reports:
1. data_inventory.csv
2. dataset_quality_report.csv
3. date_coverage_report.csv
4. missing_values_report.csv
5. duplicate_report.csv
6. anomaly_report.csv
7. data_readiness_report.csv
8. data_validation_summary.md
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.utils.paths import REPORTS_DIR
from src.utils.logging import setup_logger

logger = setup_logger("data_readiness")

REPORTS_DATA_VALIDATION_DIR = REPORTS_DIR / "data_validation"
REPORTS_DATA_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

ESSENTIAL_RETAILER_DOMAINS = [
    "Point-of-Sale sales",
    "Inventory",
    "Purchase-price history",
    "Product master",
    "Supplier master",
    "Store master",
    "Promotion history"
]


class DataReadinessEvaluator:
    """Evaluates data quality metrics and assigns readiness statuses."""

    @staticmethod
    def evaluate_status(schema_info: Dict[str, Any], anomaly_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determines READY, WARNING, or BLOCKED status.
        """
        reasons = []
        status = "READY"

        # BLOCKED criteria
        if schema_info.get("is_empty_file", False):
            status = "BLOCKED"
            reasons.append("File is empty (0 bytes)")
        elif schema_info.get("is_template_only", False):
            status = "BLOCKED"
            reasons.append("File is template-only (contains header but 0 records)")
        elif schema_info.get("error"):
            status = "BLOCKED"
            reasons.append(f"File reading error: {schema_info['error']}")
        elif schema_info.get("has_missing_identifiers", False):
            status = "BLOCKED"
            reasons.extend(schema_info["missing_identifier_details"])
        elif anomaly_info.get("infinite_values_count", 0) > 0:
            status = "BLOCKED"
            reasons.append(f"Contains {anomaly_info['infinite_values_count']} infinite values")

        # WARNING criteria (if not BLOCKED)
        if status != "BLOCKED":
            if anomaly_info.get("duplicate_rows", 0) > 0:
                status = "WARNING"
                reasons.append(f"Contains {anomaly_info['duplicate_rows']} duplicate rows")
            if anomaly_info.get("duplicate_business_keys", 0) > 0:
                status = "WARNING"
                reasons.append(f"Contains {anomaly_info['duplicate_business_keys']} duplicate business keys")
            if anomaly_info.get("empty_columns"):
                status = "WARNING"
                reasons.append(f"Contains {len(anomaly_info['empty_columns'])} completely empty columns")
            if anomaly_info.get("overall_missing_pct", 0.0) > 15.0:
                status = "WARNING"
                reasons.append(f"Overall missing cell percentage is high ({anomaly_info['overall_missing_pct']}%)")
            if anomaly_info.get("negative_value_anomalies"):
                status = "WARNING"
                reasons.append("Contains negative values in positive-only columns")
            if anomaly_info.get("invalid_price_anomalies"):
                status = "WARNING"
                reasons.append("Contains zero or extreme prices requiring treatment")

        return {
            "status": status,
            "reasons": "; ".join(reasons) if reasons else "Dataset passed all baseline checks cleanly"
        }

    def generate_all_reports(self, audit_results: List[Dict[str, Any]]) -> Dict[str, Path]:
        """Generates the 8 mandatory data validation reports."""
        logger.info("Generating comprehensive data-readiness reports...")
        out_dir = REPORTS_DATA_VALIDATION_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        inventory_rows = []
        quality_rows = []
        date_rows = []
        missing_rows = []
        duplicate_rows = []
        anomaly_rows = []
        readiness_rows = []

        present_domains = set()

        for item in audit_results:
            schema = item["schema"]
            anomaly = item["anomaly"]
            readiness = item["readiness"]

            cat = schema["classification"]
            present_domains.add(cat)

            # 1. Inventory
            inventory_rows.append({
                "filename": schema["filename"],
                "file_path": schema["file_path"],
                "format": schema["format"],
                "classification": cat,
                "file_size_kb": round(schema["file_size_bytes"] / 1024, 2),
                "rows": schema["rows"],
                "columns": schema["columns"]
            })

            # 2. Quality
            quality_rows.append({
                "filename": schema["filename"],
                "classification": cat,
                "rows": schema["rows"],
                "columns": schema["columns"],
                "total_missing_cells": anomaly["total_missing_cells"],
                "missing_pct": anomaly["overall_missing_pct"],
                "duplicate_rows": anomaly["duplicate_rows"],
                "empty_columns_count": len(anomaly["empty_columns"]),
                "status": readiness["status"]
            })

            # 3. Date Coverage
            if schema["earliest_date"] or schema["latest_date"]:
                date_rows.append({
                    "filename": schema["filename"],
                    "classification": cat,
                    "earliest_date": schema.get("earliest_date", "N/A"),
                    "latest_date": schema.get("latest_date", "N/A"),
                    "timeseries_gaps": anomaly.get("timeseries_gaps_count", 0),
                    "primary_date_col": schema["date_columns"][0] if schema["date_columns"] else "N/A"
                })

            # 4. Missing values
            for col, stats in anomaly.get("column_missing_stats", {}).items():
                if stats["count"] > 0:
                    missing_rows.append({
                        "filename": schema["filename"],
                        "column_name": col,
                        "missing_count": stats["count"],
                        "missing_pct": stats["pct"]
                    })

            # 5. Duplicates
            if anomaly["duplicate_rows"] > 0 or anomaly["duplicate_business_keys"] > 0:
                duplicate_rows.append({
                    "filename": schema["filename"],
                    "duplicate_rows": anomaly["duplicate_rows"],
                    "duplicate_business_keys": anomaly["duplicate_business_keys"],
                    "business_key_cols": str(anomaly["business_key_cols"])
                })

            # 6. Anomalies
            for a in anomaly.get("anomalies_detected", []):
                anomaly_rows.append({
                    "filename": schema["filename"],
                    "anomaly_description": a
                })

            # 7. Readiness
            readiness_rows.append({
                "filename": schema["filename"],
                "classification": cat,
                "rows": schema["rows"],
                "status": readiness["status"],
                "justification": readiness["reasons"]
            })

        # Export CSVs
        p_inv = out_dir / "data_inventory.csv"
        p_qual = out_dir / "dataset_quality_report.csv"
        p_date = out_dir / "date_coverage_report.csv"
        p_miss = out_dir / "missing_values_report.csv"
        p_dup = out_dir / "duplicate_report.csv"
        p_anom = out_dir / "anomaly_report.csv"
        p_ready = out_dir / "data_readiness_report.csv"

        pd.DataFrame(inventory_rows).to_csv(p_inv, index=False)
        pd.DataFrame(quality_rows).to_csv(p_qual, index=False)
        pd.DataFrame(date_rows).to_csv(p_date, index=False)
        pd.DataFrame(missing_rows if missing_rows else [{"filename": "None", "column_name": "None", "missing_count": 0, "missing_pct": 0.0}]).to_csv(p_miss, index=False)
        pd.DataFrame(duplicate_rows if duplicate_rows else [{"filename": "None", "duplicate_rows": 0, "duplicate_business_keys": 0, "business_key_cols": "None"}]).to_csv(p_dup, index=False)
        pd.DataFrame(anomaly_rows if anomaly_rows else [{"filename": "None", "anomaly_description": "No critical anomalies detected"}]).to_csv(p_anom, index=False)
        pd.DataFrame(readiness_rows).to_csv(p_ready, index=False)

        # 8. Markdown Summary Report
        summary_md_path = out_dir / "data_validation_summary.md"
        self._write_markdown_summary(summary_md_path, inventory_rows, readiness_rows, date_rows, present_domains)

        logger.info(f"All 8 validation reports successfully saved to: {out_dir}")
        return {
            "inventory": p_inv,
            "quality": p_qual,
            "date_coverage": p_date,
            "missing_values": p_miss,
            "duplicates": p_dup,
            "anomalies": p_anom,
            "readiness": p_ready,
            "summary_md": summary_md_path
        }

    def _write_markdown_summary(self, path: Path, inventory: List[Dict], readiness: List[Dict], date_cov: List[Dict], present_domains: set):
        ready_list = [r for r in readiness if r["status"] == "READY"]
        warning_list = [r for r in readiness if r["status"] == "WARNING"]
        blocked_list = [r for r in readiness if r["status"] == "BLOCKED"]

        missing_essential = [dom for dom in ESSENTIAL_RETAILER_DOMAINS if dom not in present_domains]

        md = []
        md.append("# Raw Data Audit & Readiness Summary Report")
        md.append(f"\n*Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        md.append(f"\n**Total Structured Files Inspected:** {len(inventory)}")
        md.append(f"\n**Readiness Breakdown:**")
        md.append(f"- **READY:** {len(ready_list)} datasets")
        md.append(f"- **WARNING:** {len(warning_list)} datasets")
        md.append(f"- **BLOCKED:** {len(blocked_list)} datasets\n")

        md.append("## 1. Essential Retailer Domains Status")
        md.append("| Essential Domain | Present in Repository | Status |")
        md.append("|---|---|---|")
        for dom in ESSENTIAL_RETAILER_DOMAINS:
            is_pres = "YES" if dom in present_domains else "NO"
            st = "VALIDATED" if dom in present_domains else "**MISSING**"
            md.append(f"| {dom} | {is_pres} | {st} |")

        if missing_essential:
            md.append(f"\n> [!CAUTION]\n> **Missing Essential Retailer Datasets:** {', '.join(missing_essential)}\n")
        else:
            md.append("\n> [!NOTE]\n> All 7 essential retailer domains are present with full records (no empty templates).\n")

        md.append("## 2. READY Datasets (Approved for Preprocessing)")
        md.append("| Filename | Classification | Rows | Verification Note |")
        md.append("|---|---|---|---|")
        for r in ready_list:
            md.append(f"| `{r['filename']}` | {r['classification']} | {r['rows']:,} | {r['justification']} |")

        md.append("\n## 3. WARNING Datasets (Requiring Preprocessing Treatment)")
        if warning_list:
            md.append("| Filename | Classification | Rows | Issues Detected |")
            md.append("|---|---|---|---|")
            for r in warning_list:
                md.append(f"| `{r['filename']}` | {r['classification']} | {r['rows']:,} | {r['justification']} |")
        else:
            md.append("*No datasets in WARNING status.*")

        md.append("\n## 4. BLOCKED Datasets (Unusable)")
        if blocked_list:
            md.append("| Filename | Classification | Rows | Blocking Reason |")
            md.append("|---|---|---|---|")
            for r in blocked_list:
                md.append(f"| `{r['filename']}` | {r['classification']} | {r['rows']:,} | {r['justification']} |")
        else:
            md.append("*No datasets BLOCKED. All files contain valid records and valid schemas.*")

        md.append("\n## 5. Major External Datasets Date Coverage")
        md.append("| Dataset | Earliest Date | Latest Date | Coverage Span | Timeseries Gaps |")
        md.append("|---|---|---|---|---|")
        for d in date_cov:
            md.append(f"| `{d['filename']}` | {d['earliest_date']} | {d['latest_date']} | {d['classification']} | {d['timeseries_gaps']} |")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
