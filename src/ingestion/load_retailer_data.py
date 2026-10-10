"""
load_retailer_data.py
=====================
CLI interface for ingesting internal retailer domain data:
- Retail Point-of-Sale sales
- Inventory positions & spoilage
- Purchase prices & purchase orders
- Supplier master data & lead-times
- Promotion history & campaigns
- Product catalog & characteristics
"""

import sys
import argparse
from pathlib import Path

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.ingestion.retailer_ingestion import RetailerIngestionManager
from src.utils.logging import setup_logger

logger = setup_logger("load_retailer_data_cli")


def main():
    parser = argparse.ArgumentParser(description="Ingest retailer internal data tables.")
    parser.add_argument("--domain", type=str, default="all",
                        choices=["all", "products", "suppliers", "promotions", "sales", "inventory", "purchase_prices"],
                        help="Specific domain to ingest or 'all'")
    args = parser.parse_args()

    mgr = RetailerIngestionManager()
    if args.domain == "all":
        mgr.run_full_ingestion()
    else:
        if args.domain == "products":
            mgr.product_loader.load()
        elif args.domain == "suppliers":
            mgr.supplier_loader.load()
        elif args.domain == "promotions":
            mgr.promo_loader.load()
        else:
            # Requires operational base
            op_path = mgr.raw_dir / "retailer_operational_data.parquet"
            if not op_path.exists():
                mgr.run_full_ingestion()
            else:
                import pandas as pd
                op_df = pd.read_parquet(op_path)
                if args.domain == "sales":
                    mgr.pos_loader.extract_from_operational(op_df)
                elif args.domain == "inventory":
                    mgr.inv_loader.extract_from_operational(op_df)
                elif args.domain == "purchase_prices":
                    mgr.po_loader.extract_from_operational(op_df)

    logger.info("Retailer ingestion operation completed.")


if __name__ == "__main__":
    main()
