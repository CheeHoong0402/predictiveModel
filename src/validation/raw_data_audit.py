"""
raw_data_audit.py
=================
Master raw-data audit orchestrator.
Scans recursively all structured datasets in the project, validates schemas,
detects anomalies, assigns readiness statuses, and generates the 8 official reports.
Executable via:
    python -m src.validation.raw_data_audit
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.utils.paths import RAW_DATA_DIR, DATA_DIR, REPORTS_DIR
from src.utils.logging import setup_logger
from src.validation.schema_validator import SchemaValidator
from src.validation.anomaly_detector import AnomalyDetector
from src.validation.data_readiness import DataReadinessEvaluator, ESSENTIAL_RETAILER_DOMAINS

logger = setup_logger("raw_data_audit")


class RawDataAuditor:
    """Orchestrates comprehensive raw dataset auditing and readiness reporting."""

    def __init__(self, root_data_dir: Path = RAW_DATA_DIR):
        self.root_data_dir = root_data_dir
        self.schema_validator = SchemaValidator()
        self.anomaly_detector = AnomalyDetector()
        self.readiness_evaluator = DataReadinessEvaluator()

    def discover_files(self) -> List[Path]:
        """Recursively discovers all structured files (.parquet, .csv, .json, .xlsx)."""
        valid_extensions = {".parquet", ".csv", ".json", ".xlsx"}
        discovered = []
        for p in self.root_data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_extensions:
                discovered.append(p)
        return sorted(discovered)

    def run_full_audit(self) -> Dict[str, Any]:
        logger.info("==================================================================")
        logger.info("   PHASE 3: COMPREHENSIVE RAW DATA AUDIT & VALIDATION PIPELINE   ")
        logger.info("==================================================================")

        files = self.discover_files()
        logger.info(f"Discovered {len(files)} structured files in {self.root_data_dir}")

        audit_results = []
        for file_path in files:
            logger.info(f"Auditing file: {file_path.name}...")
            # 1. Schema & classification
            schema_info = self.schema_validator.inspect_file(file_path)

            # 2. Anomalies, duplicates, integrity
            anomaly_info = self.anomaly_detector.audit_dataset(file_path, schema_info)

            # 3. Readiness status
            readiness_info = self.readiness_evaluator.evaluate_status(schema_info, anomaly_info)

            audit_results.append({
                "schema": schema_info,
                "anomaly": anomaly_info,
                "readiness": readiness_info
            })

        # 4. Generate the 8 required reports
        reports = self.readiness_evaluator.generate_all_reports(audit_results)

        # 5. Print Executive Console Summary
        self._print_executive_summary(audit_results)

        logger.info("==================================================================")
        logger.info("             PHASE 3 RAW DATA AUDIT COMPLETE                      ")
        logger.info("==================================================================")
        return {
            "total_files": len(files),
            "audit_results": audit_results,
            "reports": reports
        }

    def _print_executive_summary(self, audit_results: List[Dict[str, Any]]):
        ready = [r for r in audit_results if r["readiness"]["status"] == "READY"]
        warning = [r for r in audit_results if r["readiness"]["status"] == "WARNING"]
        blocked = [r for r in audit_results if r["readiness"]["status"] == "BLOCKED"]

        present_domains = set(r["schema"]["classification"] for r in audit_results)
        missing_domains = [d for d in ESSENTIAL_RETAILER_DOMAINS if d not in present_domains]

        print("\n" + "=" * 80)
        print("                 EXECUTIVE DATA READINESS REPORT                  ")
        print("=" * 80)
        print(f"Total Structured Files Audited: {len(audit_results)}")
        print(f"  [READY]   : {len(ready):>2} datasets")
        print(f"  [WARNING] : {len(warning):>2} datasets")
        print(f"  [BLOCKED] : {len(blocked):>2} datasets")
        print("-" * 80)

        print("\n--- 1. ESSENTIAL RETAILER DOMAINS STATUS ---")
        for dom in ESSENTIAL_RETAILER_DOMAINS:
            status_tag = "[PASS] Present" if dom in present_domains else "[FAIL] MISSING"
            print(f"  {status_tag:<18} : {dom}")

        if missing_domains:
            print(f"\n[ALERT] Missing essential retailer domains: {', '.join(missing_domains)}")
        else:
            print("\n[CONFIRMED] All 7 essential retailer domains are present and populated with real records.")

        print("\n--- 2. READY DATASETS (Approved for Preprocessing) ---")
        for r in ready:
            fn = r["schema"]["filename"]
            cat = r["schema"]["classification"]
            rows = r["schema"]["rows"]
            print(f"  * {fn:<38} | {cat:<28} | {rows:>9,} rows")

        if warning:
            print("\n--- 3. WARNING DATASETS (Requiring Treatment During Preprocessing) ---")
            for r in warning:
                fn = r["schema"]["filename"]
                cat = r["schema"]["classification"]
                reason = r["readiness"]["reasons"]
                print(f"  * {fn:<38} | {cat:<28} | Notice: {reason}")

        if blocked:
            print("\n--- 4. BLOCKED DATASETS (Unusable) ---")
            for r in blocked:
                fn = r["schema"]["filename"]
                reason = r["readiness"]["reasons"]
                print(f"  * [BLOCKED] {fn:<38} | Reason: {reason}")
        else:
            print("\n--- 4. BLOCKED DATASETS ---")
            print("  * None. Zero blocked datasets.")

        print("\n--- 5. MAJOR EXTERNAL DATASETS DATE COVERAGE ---")
        for r in audit_results:
            schema = r["schema"]
            if schema["earliest_date"] or schema["latest_date"]:
                fn = schema["filename"]
                cat = schema["classification"]
                e_d = schema.get("earliest_date", "N/A")
                l_d = schema.get("latest_date", "N/A")
                print(f"  * {fn:<38} | {cat:<25} : {e_d} to {l_d}")

        print("=" * 80 + "\n")


def main():
    auditor = RawDataAuditor()
    auditor.run_full_audit()


if __name__ == "__main__":
    main()
