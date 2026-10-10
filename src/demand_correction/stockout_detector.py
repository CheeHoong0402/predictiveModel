"""
src/demand_correction/stockout_detector.py
==========================================
Classifies daily product-store observations into Unconstrained vs. Censored by inventory:
- Distinguishes Full-Day Stockouts vs. Partial-Day Stockouts vs. Unconstrained observations
- Uses multi-tier evidence hierarchy (hours out of stock, closing inventory, stockout status)
- Computes duration, fraction of day unavailable, and inventory boundaries
- Strictly prevents treating genuine zero demand in well-stocked conditions as stockouts
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, Any


class StockoutDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trading_hours = float(config.get("stockout_definition", {}).get("trading_hours_per_day", 12.0))
        self.full_so_threshold = float(config.get("stockout_definition", {}).get("full_stockout_hours_threshold", 12.0))

    def detect_stockouts(self, df: pd.DataFrame, inventory_clean_path: Path) -> pd.DataFrame:
        """
        Enriches master dataframe with comprehensive stockout and censorship indicators.
        """
        df = df.copy()

        # Load intra-day hours out of stock from interim inventory file
        inv_hours = pd.read_parquet(
            inventory_clean_path,
            columns=["date", "store_id", "product_id", "number_of_hours_out_of_stock"]
        )

        df = pd.merge(df, inv_hours, on=["date", "store_id", "product_id"], how="left")
        df["number_of_hours_out_of_stock"] = df["number_of_hours_out_of_stock"].fillna(0.0).astype(float)

        # Baseline indicators
        hours = df["number_of_hours_out_of_stock"].values
        close_inv = df["closing_inventory"].values
        open_inv = df["opening_inventory"].values
        so_flag = df["stockout_flag"].values
        sales = df["observed_units_sold"].values

        # 1. Full-day stockout
        # Condition: hours out of stock >= 12 OR (open == 0 and close == 0 and sales == 0)
        full_so = (hours >= self.full_so_threshold) | ((open_inv == 0) & (close_inv == 0) & (sales == 0) & (so_flag == 1))

        # 2. Partial-day stockout
        # Condition: 0 < hours < 12 OR (close == 0 and open > 0 and hours == 0 and so_flag == 1)
        partial_so = ((hours > 0) & (hours < self.full_so_threshold)) | ((close_inv == 0) & (open_inv > 0) & (hours == 0) & (so_flag == 1))

        # 3. Censored demand flag
        # An observation is censored if sales were constrained by inventory
        censored = full_so | partial_so | (so_flag == 1) | (close_inv == 0)

        # 4. Inventory constrained flag
        # Broader flag including cases where stock became critically low (< 3 units and close <= 1)
        constrained = censored | ((close_inv <= 1) & (open_inv > 0) & (sales > 0))

        # Assign back
        df["full_day_stockout_flag"] = full_so.astype(int)
        df["partial_stockout_flag"] = partial_so.astype(int)
        df["censored_demand_flag"] = censored.astype(int)
        df["inventory_constrained_flag"] = constrained.astype(int)

        # Effective stockout duration hours
        duration_hours = np.zeros(len(df), dtype=float)
        duration_hours[full_so] = self.trading_hours
        duration_hours[partial_so] = np.maximum(hours[partial_so], 4.0)  # Default minimum 4 hours if partial confirmed
        # Bound duration by trading hours
        duration_hours = np.clip(duration_hours, 0.0, self.trading_hours)
        df["stockout_duration_hours"] = duration_hours

        # Stockout fraction of day
        df["stockout_fraction_of_day"] = np.round(df["stockout_duration_hours"] / self.trading_hours, 4)

        # Inventory boundaries
        df["available_inventory_start"] = df["opening_inventory"]
        df["available_inventory_end"] = df["closing_inventory"]

        # Evidence certainty level
        evidence = np.full(len(df), "NONE", dtype=object)
        evidence[constrained] = "MODERATE"
        evidence[(so_flag == 1) & (hours > 0)] = "STRONG"
        evidence[full_so] = "STRONG"
        df["stockout_evidence_level"] = evidence

        # Clean interim column
        df = df.drop(columns=["number_of_hours_out_of_stock"])
        return df
