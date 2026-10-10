"""
inventory_reconciliation.py
===========================
Validates and reconciles physical inventory movements:
`expected closing inventory = opening inventory + received quantity + transfers in - units sold - transfers out - damaged quantity - expired quantity`
Calculates `inventory_balance_difference` and classifies reconciliation by severity:
- Exact Match (diff == 0)
- Minor Mismatch (abs(diff) <= 2 units)
- Major Mismatch (abs(diff) > 2 units)
- Insufficient Data (missing movement components)
Does NOT automatically overwrite reported inventory without audit justification.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


class InventoryReconciler:
    """Executes conservation-of-inventory equation audits across stores and SKUs."""

    def reconcile_inventory(
        self,
        df: pd.DataFrame,
        opening_col: str = "opening_inventory_quantity",
        closing_col: str = "closing_inventory_quantity",
        received_col: str = "quantity_received",
        transfers_in_col: str = "quantity_transferred_into_store",
        sold_col: str = "historical_quantity_sold",
        transfers_out_col: str = "quantity_transferred_out_of_store",
        damaged_col: str = "damaged_quantity",
        expired_col: str = "expired_quantity"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates expected closing inventory and flags balance differences.
        Returns audited DataFrame and summary statistics table.
        """
        df = df.copy()

        # Check required columns
        req_cols = [opening_col, closing_col, received_col, transfers_in_col, sold_col, transfers_out_col, damaged_col, expired_col]
        missing_cols = [c for c in req_cols if c not in df.columns]

        if missing_cols:
            df["reconciliation_status"] = "Insufficient Data"
            df["inventory_balance_difference"] = np.nan
            summary = pd.DataFrame([{
                "store_id": "All",
                "product_id": "All",
                "total_records": len(df),
                "exact_matches": 0,
                "minor_mismatches": 0,
                "major_mismatches": 0,
                "status": f"Missing columns: {missing_cols}"
            }])
            return df, summary

        # Fill null movements with 0 for formula computation
        open_inv = df[opening_col].fillna(0)
        close_inv = df[closing_col].fillna(0)
        rec = df[received_col].fillna(0)
        t_in = df[transfers_in_col].fillna(0)
        sold = df[sold_col].fillna(0)
        t_out = df[transfers_out_col].fillna(0)
        dam = df[damaged_col].fillna(0)
        exp = df[expired_col].fillna(0)

        # Expected Closing Inventory Equation
        expected_close = open_inv + rec + t_in - sold - t_out - dam - exp
        df["expected_closing_inventory"] = expected_close
        diff = close_inv - expected_close
        df["inventory_balance_difference"] = diff

        # Severity Classification
        conditions = [
            diff == 0,
            diff.abs() <= 2,
            diff.abs() > 2
        ]
        choices = [
            "Exact Match",
            "Minor Mismatch",
            "Major Mismatch"
        ]
        df["reconciliation_status"] = np.select(conditions, choices, default="Insufficient Data")

        # Generate summary aggregation by store and product
        group_cols = [c for c in ["store_id", "product_id"] if c in df.columns]
        if group_cols:
            summary = df.groupby(group_cols).agg(
                total_records=(closing_col, "count"),
                exact_matches=("reconciliation_status", lambda s: (s == "Exact Match").sum()),
                minor_mismatches=("reconciliation_status", lambda s: (s == "Minor Mismatch").sum()),
                major_mismatches=("reconciliation_status", lambda s: (s == "Major Mismatch").sum()),
                max_discrepancy=("inventory_balance_difference", lambda d: d.abs().max())
            ).reset_index()
        else:
            summary = pd.DataFrame([{
                "total_records": len(df),
                "exact_matches": (df["reconciliation_status"] == "Exact Match").sum(),
                "minor_mismatches": (df["reconciliation_status"] == "Minor Mismatch").sum(),
                "major_mismatches": (df["reconciliation_status"] == "Major Mismatch").sum(),
                "max_discrepancy": df["inventory_balance_difference"].abs().max()
            }])

        return df, summary
