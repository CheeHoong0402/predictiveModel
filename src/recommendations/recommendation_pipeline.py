"""Executable Pipeline for Final Recommendation Engine and Decision Orchestration (Phase 21)."""
import time
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.utils.paths import PROJECT_ROOT
from src.utils.logging import setup_logger
from .engine import generate_final_recommendation, generate_recommendations_batch
from .reporting import generate_recommendation_reports

logger = setup_logger("recommendation_pipeline")


def run_recommendation_pipeline(root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Execute end-to-end recommendation orchestration pipeline."""
    start_total = time.perf_counter()
    logger.info("Initializing Phase 21 Final Recommendation Engine...")

    data_dir = root / "data/recommendations"
    reports_dir = root / "reports/recommendations"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    runtime_records: List[Dict[str, Any]] = []

    # Benchmark 1: Single product recommendation
    t0 = time.perf_counter()
    single_rec = generate_final_recommendation(
        store_id="STR_JHR_003",
        product_id="PRD_CHK_01",
        root=root
    )
    single_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"Single recommendation runtime: {single_ms:.2f} ms")
    runtime_records.append({
        "task": "single_product_recommendation",
        "count": 1,
        "runtime_seconds": round(single_ms / 1000.0, 4),
        "runtime_ms": round(single_ms, 2)
    })

    # Benchmark 2: 100 recommendation batch (simulated loop or repeated calls)
    t1 = time.perf_counter()
    for _ in range(4):  # 4 * 24 = 96 recommendations ~ 100 recommendations
        for p in ["PRD_CHK_01", "PRD_EGG_02", "PRD_OIL_05", "PRD_ONN_06", "PRD_RIC_04", "PRD_VCH_03"]:
            for s in ["STR_KUL_001", "STR_SGR_002", "STR_JHR_003", "STR_PNG_004"]:
                _ = generate_final_recommendation(store_id=s, product_id=p, root=root)
    hundred_ms = (time.perf_counter() - t1) * 1000.0
    logger.info(f"~100 recommendations runtime: {hundred_ms:.2f} ms")
    runtime_records.append({
        "task": "100_recommendations_batch",
        "count": 96,
        "runtime_seconds": round(hundred_ms / 1000.0, 4),
        "runtime_ms": round(hundred_ms, 2)
    })

    # Benchmark 3: Production batch across all 24 SKU-Stores
    t2 = time.perf_counter()
    final_df, pq_df, audit_df = generate_recommendations_batch(root=root)
    batch_ms = (time.perf_counter() - t2) * 1000.0
    logger.info(f"Production catalog batch (24 items): {batch_ms:.2f} ms")
    runtime_records.append({
        "task": "production_catalog_batch",
        "count": len(final_df),
        "runtime_seconds": round(batch_ms / 1000.0, 4),
        "runtime_ms": round(batch_ms, 2)
    })

    # Save Parquet outputs
    final_path = data_dir / "final_recommendations.parquet"
    pq_path = data_dir / "action_priority_queue.parquet"
    audit_path = data_dir / "recommendation_audit_log.parquet"

    final_df.to_parquet(final_path, index=False)
    pq_df.to_parquet(pq_path, index=False)
    audit_df.to_parquet(audit_path, index=False)

    logger.info(f"Saved final recommendations: {final_path} ({len(final_df)} rows)")
    logger.info(f"Saved action priority queue: {pq_path} ({len(pq_df)} rows)")
    logger.info(f"Saved recommendation audit log: {audit_path} ({len(audit_df)} rows)")

    # Generate Reports
    total_sec = time.perf_counter() - start_total
    runtime_records.append({
        "task": "end_to_end_pipeline",
        "count": len(final_df),
        "runtime_seconds": round(total_sec, 4),
        "runtime_ms": round(total_sec * 1000.0, 2)
    })

    generate_recommendation_reports(
        final_df=final_df,
        priority_queue_df=pq_df,
        audit_df=audit_df,
        runtime_records=runtime_records,
        output_dir=reports_dir
    )

    logger.info(f"Phase 21 Recommendation Pipeline completed in {total_sec:.2f}s.")
    return {
        "status": "COMPLETED",
        "total_recommendations": len(final_df),
        "total_runtime_seconds": round(total_sec, 2),
        "single_rec_ms": round(single_ms, 2),
        "batch_rec_ms": round(batch_ms, 2)
    }


if __name__ == "__main__":
    res = run_recommendation_pipeline()
    print("Recommendation Pipeline Execution Result:", res)
