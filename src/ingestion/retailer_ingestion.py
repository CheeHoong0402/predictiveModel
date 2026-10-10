"""
retailer_ingestion.py
=====================
Modular ingestion layer for internal Retailer data:
1. Retail Point-of-Sale sales (POS transactions)
2. Inventory (Daily positions, movements, shelf-life spoilage, stockouts)
3. Purchase prices (Supplier purchase price history, POs, transport costs)
4. Supplier data (Supplier master, lead times, MOQs, reliability metrics)
5. Promotion history (Active promotions, promo types, discount percentages)
6. Product information (Product master, shelf life, expiry periods, origin, dimensions)

Supports loading external raw CSV/Parquet files and generating comprehensive
high-fidelity 10-year Malaysian retail operational datasets conforming to all criteria.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.utils.paths import RAW_DATA_DIR
from src.utils.logging import setup_logger

logger = setup_logger("retailer_ingestion")


class ProductInformationLoader:
    """Manages product catalog and master attributes."""

    @staticmethod
    def get_default_product_catalog() -> pd.DataFrame:
        products = [
            {
                "product_id": "PRD_CHK_01",
                "product_name": "Ayam Bulat Segar (Standard Whole Chicken)",
                "product_category": "Poultry & Meat",
                "product_subcategory": "Fresh Chicken",
                "product_brand": "Peladang Jaya",
                "product_size": "1.8kg - 2.0kg",
                "product_unit_of_measurement": "kg",
                "product_pack_size": 1,
                "product_case_size": 10,
                "product_shelf_life_days": 3,
                "product_expiry_period_days": 3,
                "product_country_of_origin": "Malaysia",
                "imported_or_locally_produced": "Local",
                "normal_selling_price": 9.40,
                "base_purchase_price": 7.20,
                "supplier_id": "SUP_KLS_01",
                "controlled_goods_status": 1,
                "eligibility_under_assistance": 1,
            },
            {
                "product_id": "PRD_EGG_02",
                "product_name": "Telur Ayam Gred A (30 biji)",
                "product_category": "Dairy & Eggs",
                "product_subcategory": "Eggs",
                "product_brand": "NutriPlus",
                "product_size": "Tray of 30",
                "product_unit_of_measurement": "tray",
                "product_pack_size": 30,
                "product_case_size": 10,
                "product_shelf_life_days": 21,
                "product_expiry_period_days": 21,
                "product_country_of_origin": "Malaysia",
                "imported_or_locally_produced": "Local",
                "normal_selling_price": 13.50,
                "base_purchase_price": 11.20,
                "supplier_id": "SUP_LST_02",
                "controlled_goods_status": 1,
                "eligibility_under_assistance": 1,
            },
            {
                "product_id": "PRD_VCH_03",
                "product_name": "Kobis Bulat Cameron (Cameron Cabbage)",
                "product_category": "Fresh Vegetables",
                "product_subcategory": "Leafy Vegetables",
                "product_brand": "Cameron Agro",
                "product_size": "1.0kg",
                "product_unit_of_measurement": "kg",
                "product_pack_size": 1,
                "product_case_size": 20,
                "product_shelf_life_days": 5,
                "product_expiry_period_days": 5,
                "product_country_of_origin": "Malaysia",
                "imported_or_locally_produced": "Local",
                "normal_selling_price": 4.80,
                "base_purchase_price": 3.20,
                "supplier_id": "SUP_CAM_03",
                "controlled_goods_status": 0,
                "eligibility_under_assistance": 0,
            },
            {
                "product_id": "PRD_RIC_04",
                "product_name": "Beras Putih Super Tempatan (SST 5%) 10kg",
                "product_category": "Dry Staples",
                "product_subcategory": "Rice",
                "product_brand": "Jasmine",
                "product_size": "10kg",
                "product_unit_of_measurement": "bag",
                "product_pack_size": 1,
                "product_case_size": 5,
                "product_shelf_life_days": 365,
                "product_expiry_period_days": 365,
                "product_country_of_origin": "Malaysia",
                "imported_or_locally_produced": "Local",
                "normal_selling_price": 26.00,
                "base_purchase_price": 22.50,
                "supplier_id": "SUP_BER_04",
                "controlled_goods_status": 1,
                "eligibility_under_assistance": 1,
            },
            {
                "product_id": "PRD_OIL_05",
                "product_name": "Minyak Masak Sawit Tulen 5kg",
                "product_category": "Cooking Oils & Fats",
                "product_subcategory": "Palm Cooking Oil",
                "product_brand": "Seri Murni",
                "product_size": "5kg",
                "product_unit_of_measurement": "bottle",
                "product_pack_size": 1,
                "product_case_size": 4,
                "product_shelf_life_days": 540,
                "product_expiry_period_days": 540,
                "product_country_of_origin": "Malaysia",
                "imported_or_locally_produced": "Local",
                "normal_selling_price": 31.50,
                "base_purchase_price": 27.20,
                "supplier_id": "SUP_SOP_05",
                "controlled_goods_status": 1,
                "eligibility_under_assistance": 1,
            },
            {
                "product_id": "PRD_ONN_06",
                "product_name": "Bawang Merah Kecil Import (Rose Shallots)",
                "product_category": "Fresh Vegetables",
                "product_subcategory": "Onions & Shallots",
                "product_brand": "Rose Brand",
                "product_size": "1.0kg",
                "product_unit_of_measurement": "kg",
                "product_pack_size": 1,
                "product_case_size": 15,
                "product_shelf_life_days": 14,
                "product_expiry_period_days": 14,
                "product_country_of_origin": "India",
                "imported_or_locally_produced": "Imported",
                "normal_selling_price": 8.50,
                "base_purchase_price": 5.80,
                "supplier_id": "SUP_IMP_06",
                "controlled_goods_status": 0,
                "eligibility_under_assistance": 0,
            }
        ]
        return pd.DataFrame(products)

    def load(self, source_path: Optional[Path] = None, output_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        if source_path and source_path.exists():
            df = pd.read_csv(source_path) if source_path.suffix == ".csv" else pd.read_parquet(source_path)
            logger.info(f"Loaded external product catalog: {len(df)} items")
        else:
            df = self.get_default_product_catalog()
            df.to_csv(output_dir / "retail_products.csv", index=False)
            logger.info(f"Saved default product catalog: {len(df)} items")
        return df


class SupplierDataLoader:
    """Manages supplier profiles, locations, lead times, and reliability metrics."""

    @staticmethod
    def get_default_suppliers() -> pd.DataFrame:
        suppliers = [
            {
                "supplier_id": "SUP_KLS_01",
                "supplier_name": "Klang Valley Poultry Sdn Bhd",
                "supplier_location": "Banting",
                "supplier_state": "Selangor",
                "supplier_country": "Malaysia",
                "supplier_lead_time_days": 1,
                "historical_supplier_lead_time": 1.2,
                "supplier_minimum_order_quantity": 50,
                "supplier_pack_size": 10,
                "supplier_delivery_cost": 120.0,
                "supplier_freight_cost": 25.0,
                "supplier_reliability_rate": 0.96,
                "supplier_on_time_delivery_rate": 0.94,
                "supplier_complete_delivery_rate": 0.98,
                "lead_time_variability_days": 0.3
            },
            {
                "supplier_id": "SUP_LST_02",
                "supplier_name": "Lay Hong Farming Sdn Bhd",
                "supplier_location": "Ijok",
                "supplier_state": "Selangor",
                "supplier_country": "Malaysia",
                "supplier_lead_time_days": 2,
                "historical_supplier_lead_time": 2.1,
                "supplier_minimum_order_quantity": 30,
                "supplier_pack_size": 10,
                "supplier_delivery_cost": 100.0,
                "supplier_freight_cost": 20.0,
                "supplier_reliability_rate": 0.98,
                "supplier_on_time_delivery_rate": 0.96,
                "supplier_complete_delivery_rate": 0.99,
                "lead_time_variability_days": 0.4
            },
            {
                "supplier_id": "SUP_CAM_03",
                "supplier_name": "Highland Agro Supply Sdn Bhd",
                "supplier_location": "Ringlet",
                "supplier_state": "Pahang",
                "supplier_country": "Malaysia",
                "supplier_lead_time_days": 2,
                "historical_supplier_lead_time": 2.4,
                "supplier_minimum_order_quantity": 40,
                "supplier_pack_size": 20,
                "supplier_delivery_cost": 150.0,
                "supplier_freight_cost": 35.0,
                "supplier_reliability_rate": 0.93,
                "supplier_on_time_delivery_rate": 0.90,
                "supplier_complete_delivery_rate": 0.95,
                "lead_time_variability_days": 0.7
            },
            {
                "supplier_id": "SUP_BER_04",
                "supplier_name": "Bernas Distribution Sdn Bhd",
                "supplier_location": "Shah Alam",
                "supplier_state": "Selangor",
                "supplier_country": "Malaysia",
                "supplier_lead_time_days": 3,
                "historical_supplier_lead_time": 3.0,
                "supplier_minimum_order_quantity": 20,
                "supplier_pack_size": 5,
                "supplier_delivery_cost": 80.0,
                "supplier_freight_cost": 15.0,
                "supplier_reliability_rate": 0.99,
                "supplier_on_time_delivery_rate": 0.98,
                "supplier_complete_delivery_rate": 0.99,
                "lead_time_variability_days": 0.2
            },
            {
                "supplier_id": "SUP_SOP_05",
                "supplier_name": "Sime Darby Oils Marketing",
                "supplier_location": "Telok Panglima Garang",
                "supplier_state": "Selangor",
                "supplier_country": "Malaysia",
                "supplier_lead_time_days": 3,
                "historical_supplier_lead_time": 3.2,
                "supplier_minimum_order_quantity": 25,
                "supplier_pack_size": 4,
                "supplier_delivery_cost": 90.0,
                "supplier_freight_cost": 20.0,
                "supplier_reliability_rate": 0.97,
                "supplier_on_time_delivery_rate": 0.95,
                "supplier_complete_delivery_rate": 0.98,
                "lead_time_variability_days": 0.3
            },
            {
                "supplier_id": "SUP_IMP_06",
                "supplier_name": "Global Spice & Produce Importers",
                "supplier_location": "Port Klang",
                "supplier_state": "Selangor",
                "supplier_country": "Malaysia",
                "supplier_lead_time_days": 4,
                "historical_supplier_lead_time": 4.6,
                "supplier_minimum_order_quantity": 30,
                "supplier_pack_size": 15,
                "supplier_delivery_cost": 140.0,
                "supplier_freight_cost": 45.0,
                "supplier_reliability_rate": 0.91,
                "supplier_on_time_delivery_rate": 0.88,
                "supplier_complete_delivery_rate": 0.94,
                "lead_time_variability_days": 1.1
            }
        ]
        return pd.DataFrame(suppliers)

    def load(self, source_path: Optional[Path] = None, output_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        if source_path and source_path.exists():
            df = pd.read_csv(source_path) if source_path.suffix == ".csv" else pd.read_parquet(source_path)
            logger.info(f"Loaded external supplier master: {len(df)} suppliers")
        else:
            df = self.get_default_suppliers()
            df.to_csv(output_dir / "retail_suppliers.csv", index=False)
            logger.info(f"Saved default supplier master: {len(df)} suppliers")
        return df


class PromotionHistoryLoader:
    """Manages promotional campaigns, discount percentages, and campaign dates."""

    @staticmethod
    def generate_promotions(start_year: int = 2021, end_year: int = 2024) -> pd.DataFrame:
        records = []
        promo_id = 1
        for yr in range(start_year, end_year + 1):
            # Festive promotions
            events = [
                ("CNY Mega Sale", datetime(yr, 1, 20), datetime(yr, 2, 5), 0.15, "Festive Promotion"),
                ("Raya Aidilfitri Jualan Rahmah", datetime(yr, 4, 5), datetime(yr, 4, 25), 0.12, "Festive Promotion"),
                ("Deepavali Light Specials", datetime(yr, 10, 20), datetime(yr, 11, 2), 0.10, "Festive Promotion"),
                ("National Day Merdeka Promo", datetime(yr, 8, 25), datetime(yr, 9, 2), 0.08, "National Day"),
                ("Year-End Festive Clearance", datetime(yr, 12, 18), datetime(yr, 12, 31), 0.15, "Year End"),
            ]
            for name, start_dt, end_dt, disc, p_type in events:
                records.append({
                    "promotion_id": f"PRM_{yr}_{promo_id:04d}",
                    "promotion_name": name,
                    "promotion_type": p_type,
                    "promotion_start_date": start_dt.strftime("%Y-%m-%d"),
                    "promotion_end_date": end_dt.strftime("%Y-%m-%d"),
                    "discount_percentage": disc * 100,
                    "target_categories": "Poultry & Meat, Fresh Vegetables, Cooking Oils & Fats, Dry Staples"
                })
                promo_id += 1

        return pd.DataFrame(records)

    def load(self, source_path: Optional[Path] = None, output_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        if source_path and source_path.exists():
            df = pd.read_csv(source_path)
        else:
            df = self.generate_promotions()
            df.to_csv(output_dir / "retail_promotions.csv", index=False)
            logger.info(f"Saved promotion history: {len(df)} campaigns")
        return df


class POSSalesLoader:
    """Manages POS sales transactions (units sold, transactions, prices, returns)."""

    def extract_from_operational(self, operational_df: pd.DataFrame, output_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        sales_cols = [
            "date", "store_id", "product_id", "historical_quantity_sold",
            "historical_number_of_transactions", "average_quantity_purchased_per_transaction",
            "historical_selling_price", "normal_selling_price", "discounted_selling_price",
            "promotion_status", "promotion_type", "discount_percentage",
            "product_return_quantity", "cancelled_sales_quantity"
        ]
        available_cols = [c for c in sales_cols if c in operational_df.columns]
        df = operational_df[available_cols].copy()
        df.to_parquet(output_dir / "retail_pos_sales.parquet", index=False)
        logger.info(f"Saved POS sales dataset: {output_dir / 'retail_pos_sales.parquet'} ({len(df):,} rows)")
        return df


class InventoryLoader:
    """Manages daily inventory movements, spoilage, damages, and stockout tracking."""

    def extract_from_operational(self, operational_df: pd.DataFrame, output_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        inv_cols = [
            "date", "store_id", "product_id", "opening_inventory_quantity",
            "closing_inventory_quantity", "current_inventory_quantity", "quantity_received",
            "quantity_transferred_into_store", "quantity_transferred_out_of_store",
            "damaged_quantity", "expired_quantity", "returned_inventory_quantity",
            "out_of_stock_status", "number_of_hours_out_of_stock", "number_of_days_out_of_stock",
            "backorder_quantity", "incoming_inventory_quantity", "expected_incoming_inventory_date",
            "inventory_holding_cost_per_unit_per_day", "storage_cost", "stockout_cost_per_unit", "wastage_cost_per_unit"
        ]
        available_cols = [c for c in inv_cols if c in operational_df.columns]
        df = operational_df[available_cols].copy()
        df.to_parquet(output_dir / "retail_inventory.parquet", index=False)
        logger.info(f"Saved Inventory dataset: {output_dir / 'retail_inventory.parquet'} ({len(df):,} rows)")
        return df


class PurchasePriceLoader:
    """Manages historical purchase orders, supplier purchase prices, and delivery costs."""

    def extract_from_operational(self, operational_df: pd.DataFrame, output_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
        po_cols = [
            "date", "store_id", "product_id", "supplier_id",
            "historical_supplier_purchase_price", "current_supplier_purchase_price",
            "historical_purchase_quantity", "purchase_order_date",
            "expected_supplier_delivery_date", "actual_supplier_delivery_date",
            "supplier_lead_time", "supplier_delivery_cost", "supplier_freight_cost",
            "gross_profit_margin_per_unit"
        ]
        available_cols = [c for c in po_cols if c in operational_df.columns]
        df = operational_df[available_cols].copy()
        df.to_parquet(output_dir / "retail_purchase_prices.parquet", index=False)
        logger.info(f"Saved Purchase Price dataset: {output_dir / 'retail_purchase_prices.parquet'} ({len(df):,} rows)")
        return df


class RetailerIngestionManager:
    """Master orchestrator for ingesting all 6 retailer domain tables."""

    def __init__(self, raw_dir: Path = RAW_DATA_DIR):
        self.raw_dir = raw_dir
        self.product_loader = ProductInformationLoader()
        self.supplier_loader = SupplierDataLoader()
        self.promo_loader = PromotionHistoryLoader()
        self.pos_loader = POSSalesLoader()
        self.inv_loader = InventoryLoader()
        self.po_loader = PurchasePriceLoader()

    def run_full_ingestion(self) -> Dict[str, pd.DataFrame]:
        logger.info("=== STARTING RETAILER INGESTION SUITE (6 DOMAIN TABLES) ===")
        
        # 1. Product master
        prod_df = self.product_loader.load(output_dir=self.raw_dir)

        # 2. Supplier master
        supp_df = self.supplier_loader.load(output_dir=self.raw_dir)

        # 3. Promotions schedule
        promo_df = self.promo_loader.load(output_dir=self.raw_dir)

        # 4. Check or generate full operational table
        op_path = self.raw_dir / "retailer_operational_data.parquet"
        if not op_path.exists():
            from src.ingestion.load_retailer_data import generate_retailer_dataset
            op_df = generate_retailer_dataset(output_dir=self.raw_dir)
        else:
            op_df = pd.read_parquet(op_path)

        # 5. POS Sales
        sales_df = self.pos_loader.extract_from_operational(op_df, output_dir=self.raw_dir)

        # 6. Inventory positions & spoilage
        inv_df = self.inv_loader.extract_from_operational(op_df, output_dir=self.raw_dir)

        # 7. Purchase prices & PO history
        po_df = self.po_loader.extract_from_operational(op_df, output_dir=self.raw_dir)

        logger.info("=== RETAILER INGESTION SUITE COMPLETED SUCCESSFULLY ===")
        return {
            "products": prod_df,
            "suppliers": supp_df,
            "promotions": promo_df,
            "sales": sales_df,
            "inventory": inv_df,
            "purchase_prices": po_df
        }


if __name__ == "__main__":
    mgr = RetailerIngestionManager()
    mgr.run_full_ingestion()
