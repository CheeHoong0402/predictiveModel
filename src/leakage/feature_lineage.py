"""
Feature Lineage Mapping Module.
Traces comprehensive 4-stage data pedigree for every engineered feature:
Raw Source -> Cleaned Interim -> Master Table Column -> Engineered Feature.
"""

from pathlib import Path
from typing import Dict, List
import pandas as pd


class FeatureLineageTracker:
    """Builds and exports complete data pedigree traces for model governance."""

    def __init__(self):
        self.family_lineage_rules: Dict[str, Dict[str, str]] = {
            "demand_history": {
                "raw_source": "retailer_operational_data.parquet / POS Transactions",
                "cleaned_interim": "pos_sales_clean.parquet & demand_correction",
                "master_column": "estimated_true_demand_training_safe",
                "transformation_type": "Temporal lag shift (shift(k) from T-1)",
            },
            "rolling_demand": {
                "raw_source": "retailer_operational_data.parquet / POS Transactions",
                "cleaned_interim": "pos_sales_clean.parquet & demand_correction",
                "master_column": "estimated_true_demand_training_safe",
                "transformation_type": "Rolling window statistical aggregation on shift(1)",
            },
            "demand_momentum": {
                "raw_source": "retailer_operational_data.parquet / POS Transactions",
                "cleaned_interim": "pos_sales_clean.parquet & demand_correction",
                "master_column": "estimated_true_demand_training_safe",
                "transformation_type": "Moving window ratios and percentage growths",
            },
            "weekly_seasonality": {
                "raw_source": "retailer_operational_data.parquet / POS Transactions",
                "cleaned_interim": "pos_sales_clean.parquet & demand_correction",
                "master_column": "estimated_true_demand_training_safe",
                "transformation_type": "Same-weekday rolling median over prior 4/8 weeks",
            },
            "annual_seasonality": {
                "raw_source": "retailer_operational_data.parquet / POS Transactions",
                "cleaned_interim": "pos_sales_clean.parquet & demand_correction",
                "master_column": "estimated_true_demand_training_safe",
                "transformation_type": "Annual lag 365d and 52-week rolling window",
            },
            "observed_sales": {
                "raw_source": "retail_pos_sales.parquet / POS Cash Registers",
                "cleaned_interim": "pos_sales_clean.parquet",
                "master_column": "observed_units_sold",
                "transformation_type": "Historical sales lags and rolling window statistics",
            },
            "inventory_state": {
                "raw_source": "retail_inventory.parquet / WMS Daily Balance",
                "cleaned_interim": "inventory_clean.parquet",
                "master_column": "current_inventory, opening_inventory, closing_inventory",
                "transformation_type": "Stock balance snapshot and prior-day closing lag",
            },
            "stockout_history": {
                "raw_source": "retail_inventory.parquet / POS Sales Reconciliation",
                "cleaned_interim": "inventory_clean.parquet & demand_correction",
                "master_column": "stockout_flag, full_day_stockout_flag",
                "transformation_type": "Rolling stockout count, rate, and days since event",
            },
            "replenishment_history": {
                "raw_source": "retail_inventory.parquet / GRN Receiving Slips",
                "cleaned_interim": "inventory_clean.parquet",
                "master_column": "quantity_received",
                "transformation_type": "Rolling receipts sum and delivery receipt counts",
            },
            "inventory_coverage": {
                "raw_source": "retail_inventory.parquet & pos_sales.parquet",
                "cleaned_interim": "inventory_clean.parquet & master_table",
                "master_column": "current_inventory & demand_mean_7d",
                "transformation_type": "Days of supply coverage ratio",
            },
            "purchase_price_history": {
                "raw_source": "retail_purchase_prices.parquet / Supplier Invoices",
                "cleaned_interim": "purchase_prices_clean.parquet",
                "master_column": "purchase_price",
                "transformation_type": "Historical purchase price lags (shift(k) from T-1)",
            },
            "purchase_price_rolling": {
                "raw_source": "retail_purchase_prices.parquet / Supplier Invoices",
                "cleaned_interim": "purchase_prices_clean.parquet",
                "master_column": "purchase_price",
                "transformation_type": "Rolling mean, standard deviation, and volatility on shift(1)",
            },
            "price_momentum": {
                "raw_source": "retail_purchase_prices.parquet / Supplier Invoices",
                "cleaned_interim": "purchase_prices_clean.parquet",
                "master_column": "purchase_price",
                "transformation_type": "1d/7d/30d percentage changes and price acceleration",
            },
            "selling_price": {
                "raw_source": "retail_pos_sales.parquet / Shelf Price Records",
                "cleaned_interim": "pos_sales_clean.parquet",
                "master_column": "selling_price, normal_selling_price",
                "transformation_type": "Discounts, price ratios, and rolling selling price means",
            },
            "price_elasticity_proxy": {
                "raw_source": "retail_pos_sales.parquet & demand_correction",
                "cleaned_interim": "pos_sales_clean.parquet & master_table",
                "master_column": "selling_price & demand_growth_7d",
                "transformation_type": "Arc elasticity proxy clipped to safe domain bounds",
            },
            "procurement_supplier": {
                "raw_source": "retail_suppliers.csv & purchase_prices.parquet",
                "cleaned_interim": "supplier_master_clean.parquet",
                "master_column": "expected_lead_time_days, actual_lead_time_days",
                "transformation_type": "Lead time gaps, rolling mean lead times, on-time rate",
            },
            "promotion": {
                "raw_source": "retail_promotions.csv / Marketing Calendar",
                "cleaned_interim": "promotions_clean.parquet",
                "master_column": "promotion_active, discount_percentage",
                "transformation_type": "Campaign active indicators, durations, scheduled upcoming promos",
            },
            "calendar": {
                "raw_source": "System Calendar / Gregorian Date Module",
                "cleaned_interim": "Master Calendar Skeleton",
                "master_column": "date",
                "transformation_type": "DOW, DOM, month, quarter, and cyclical sine/cosine transforms",
            },
            "holiday_proximity": {
                "raw_source": "malaysia_public_holidays_10y.csv / Cabinet Gazette",
                "cleaned_interim": "public_holidays_clean.parquet",
                "master_column": "public_holiday_flag",
                "transformation_type": "Holiday lead (-1d to -7d) and lag (+1d to +7d) indicators",
            },
            "festival_proximity": {
                "raw_source": "Solar-Lunar Astronomical Almanac / National Holidays",
                "cleaned_interim": "public_holidays_clean.parquet",
                "master_column": "public_holiday_name",
                "transformation_type": "Calendar distance to/from CNY, Raya, Deepavali, Christmas",
            },
            "school_holiday": {
                "raw_source": "malaysia_school_holidays_10y.csv / MOE Gazette",
                "cleaned_interim": "school_holidays_clean.parquet",
                "master_column": "school_holiday_flag",
                "transformation_type": "Contiguous break durations, days elapsed, days remaining",
            },
            "salary_cycle": {
                "raw_source": "malaysia_government_salary_dates_10y.csv / JANM",
                "cleaned_interim": "salary_dates_clean.parquet",
                "master_column": "cal_day_of_month",
                "transformation_type": "Government pay day distance (23rd) and post-pay windows",
            },
            "government_assistance": {
                "raw_source": "malaysia_government_assistance_10y.csv / MOF",
                "cleaned_interim": "government_assistance_clean.parquet",
                "master_column": "government_assistance_event, amount",
                "transformation_type": "Days since BPR/BKM/STR crediting and post-aid flags",
            },
            "weather": {
                "raw_source": "malaysia_weather_10y.csv / MET Malaysia",
                "cleaned_interim": "weather_clean.parquet",
                "master_column": "temperature, rainfall, humidity",
                "transformation_type": "Store meteorological station daily readings",
            },
            "weather_rolling": {
                "raw_source": "malaysia_weather_10y.csv / MET Malaysia",
                "cleaned_interim": "weather_clean.parquet",
                "master_column": "rainfall, average_temperature",
                "transformation_type": "Historical rolling rainfall sums and mean temps on shift(1)",
            },
            "weather_spells": {
                "raw_source": "malaysia_weather_10y.csv / MET Malaysia",
                "cleaned_interim": "weather_clean.parquet",
                "master_column": "rainfall, maximum_temperature",
                "transformation_type": "Consecutive rainy days (>1mm) and heatwave spells on shift(1)",
            },
            "market_price": {
                "raw_source": "pricecatcher_2024-01.parquet / KPDN",
                "cleaned_interim": "pricecatcher_clean.parquet",
                "master_column": "market_median_price, market_mean_price",
                "transformation_type": "Premise competitor median/mean and retailer price gaps",
            },
            "macro_fuel": {
                "raw_source": "fuel_prices.csv / MOF APM Weekly Gazette",
                "cleaned_interim": "fuel_prices_clean.parquet",
                "master_column": "subsidised_petrol_price, unsubsidised_diesel_price",
                "transformation_type": "Official retail pump prices and 7d/28d percentage shocks",
            },
            "macro_currency": {
                "raw_source": "exchange_rates_10y.csv / Bank Negara Malaysia",
                "cleaned_interim": "exchange_rates_clean.parquet",
                "master_column": "relevant_exchange_rate",
                "transformation_type": "Daily FX rate, 1d/7d/30d changes, and 30d rolling volatility",
            },
            "macro_commodity": {
                "raw_source": "commodities_10y.csv / World Bank & Global Exchanges",
                "cleaned_interim": "commodity_prices_clean.parquet",
                "master_column": "relevant_commodity_price",
                "transformation_type": "Mapped benchmark spot prices and 7d/30d percentage changes",
            },
            "macro_inflation": {
                "raw_source": "cpi_headline.csv & ppi.csv / DOSM",
                "cleaned_interim": "cpi_clean.parquet & ppi_clean.parquet",
                "master_column": "Consumer_Price_Index, Producer_Price_Index",
                "transformation_type": "Headline and category indices with publication release lags",
            },
            "policy_subsidy": {
                "raw_source": "malaysia_subsidies_and_policies.csv / Federal Gazette",
                "cleaned_interim": "subsidies_and_policies_clean.parquet",
                "master_column": "subsidy_active_flag, policy_active_flag",
                "transformation_type": "Targeted diesel float, chicken ceilings, minimum wage RM1500",
            },
            "static_product": {
                "raw_source": "retail_products.csv / ERP Item Master",
                "cleaned_interim": "product_master_clean.parquet",
                "master_column": "shelf_life_days, imported_flag",
                "transformation_type": "Perishability tiers, imported flag, controlled goods status",
            },
            "static_store": {
                "raw_source": "retail_stores.csv / Premise Registry",
                "cleaned_interim": "store_master_clean.parquet",
                "master_column": "storage_capacity, latitude, longitude",
                "transformation_type": "Capacity units, relative scale ratio, geographical coordinates",
            },
            "interactions": {
                "raw_source": "Cross-domain Master Tables",
                "cleaned_interim": "Engineered features combination",
                "master_column": "Multiple domain columns",
                "transformation_type": "Multiplicative interaction between campaign and macro drivers",
            },
            "missingness_indicator": {
                "raw_source": "External dataset presence flags",
                "cleaned_interim": "master_daily.parquet",
                "master_column": "market_observation_count, weather_available",
                "transformation_type": "Binary indicators for unobserved external market data",
            },
        }

    def trace_feature_lineage(self, feature_registry_df: pd.DataFrame, output_dir: Path) -> Path:
        """
        Generate feature lineage CSV mapping every engineered feature through the 4 stages.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "feature_lineage.csv"

        records = []
        for _, row in feature_registry_df.iterrows():
            fname = row["name"]
            family = row["feature_family"]
            rule = self.family_lineage_rules.get(family, {
                "raw_source": "Retail operational data / External APIs",
                "cleaned_interim": "Cleaned interim Parquet",
                "master_column": row.get("source_columns", "Multiple"),
                "transformation_type": row.get("calculation", "Deterministic transform"),
            })

            records.append({
                "feature_name": fname,
                "feature_family": family,
                "raw_source": rule["raw_source"],
                "cleaned_interim_layer": rule["cleaned_interim"],
                "master_table_column": rule["master_column"],
                "engineered_feature": fname,
                "transformation_type": rule["transformation_type"],
                "data_type": row.get("data_type", "float64"),
                "target_model_applicability": row.get("target_model_applicability", "both"),
            })

        df_lineage = pd.DataFrame(records)
        df_lineage.to_csv(csv_path, index=False)
        return csv_path
