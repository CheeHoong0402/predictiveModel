"""
download_public_data.py
=======================
CLI interface for downloading public Malaysian retail datasets:
PriceCatcher, CPI, PPI, Fuel Prices, Exchange Rates, Commodities,
Weather, Holidays, School Holidays, Trade Activity, Imports/Exports,
Government Assistance, Subsidy Events, and Budget Policy events.
"""

import sys
import argparse
from pathlib import Path

# Path setup
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.ingestion.public_data_downloaders import PublicDataMasterManager
from src.utils.logging import setup_logger

logger = setup_logger("download_public_data_cli")


def main():
    parser = argparse.ArgumentParser(description="Download public Malaysian datasets for retail predictive modeling.")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "pricecatcher", "cpi", "ppi", "fuel", "exchange",
                                 "commodities", "weather", "holidays", "trade", "external_trade",
                                 "assistance", "subsidies", "budget"],
                        help="Select dataset to download or 'all'")
    parser.add_argument("--start-year", type=int, default=2015, help="Start year for historical datasets")
    parser.add_argument("--end-year", type=int, default=2025, help="End year for historical datasets")

    args = parser.parse_args()
    mgr = PublicDataMasterManager()

    if args.dataset == "all":
        mgr.download_all()
    elif args.dataset == "pricecatcher":
        mgr.pricecatcher.download_lookups()
        mgr.pricecatcher.download_monthly_sample()
    elif args.dataset == "cpi":
        mgr.cpi.download()
    elif args.dataset == "ppi":
        mgr.ppi.download()
    elif args.dataset == "fuel":
        mgr.fuel.download()
    elif args.dataset == "exchange":
        mgr.exchange.download(args.start_year, args.end_year)
    elif args.dataset == "commodities":
        mgr.commodities.download(args.start_year, args.end_year)
    elif args.dataset == "weather":
        mgr.weather.download(args.start_year, min(args.end_year, 2024))
    elif args.dataset == "holidays":
        mgr.holidays.download(args.start_year, args.end_year)
        mgr.school_holidays.download(args.start_year, args.end_year)
    elif args.dataset == "trade":
        mgr.trade_activity.download()
    elif args.dataset == "external_trade":
        mgr.external_trade.download()
    elif args.dataset == "assistance":
        mgr.assistance.download(args.start_year, args.end_year)
    elif args.dataset == "subsidies":
        mgr.subsidies.download()
    elif args.dataset == "budget":
        mgr.budget.download()

    logger.info("Download operation completed.")


if __name__ == "__main__":
    main()
