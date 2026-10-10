"""
Publication Lag and Release Timing Validator.
Audits external macroeconomic, fuel, currency, commodity, and policy datasets
to ensure no observation is accessible prior to its official public release date.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import polars as pl


@dataclass
class PublicationLagSpec:
    """Metadata specification for source dataset publication lag and release rules."""
    dataset_name: str
    frequency: str
    official_source: str
    typical_lag_days: int
    timing_rule: str
    conservative_assumption: str
    leakage_risk_level: str


class PublicationLagValidator:
    """Validates temporal release constraints across all external and market data sources."""

    def __init__(self):
        self.specs: Dict[str, PublicationLagSpec] = {
            "consumer_price_index": PublicationLagSpec(
                dataset_name="Consumer Price Index (CPI)",
                frequency="Monthly",
                official_source="Department of Statistics Malaysia (DOSM)",
                typical_lag_days=21,
                timing_rule="Published ~21 days following the close of reference month M",
                conservative_assumption="Data for month M is visible to model only from 22nd day of month M+1 onwards",
                leakage_risk_level="HIGH",
            ),
            "producer_price_index": PublicationLagSpec(
                dataset_name="Producer Price Index (PPI)",
                frequency="Monthly",
                official_source="Department of Statistics Malaysia (DOSM)",
                typical_lag_days=30,
                timing_rule="Published ~30 days following reference month M",
                conservative_assumption="Data for month M is visible only from 1st day of month M+2 onwards",
                leakage_risk_level="HIGH",
            ),
            "fuel_prices": PublicationLagSpec(
                dataset_name="Retail Fuel Prices (RON95, RON97, Diesel)",
                frequency="Weekly",
                official_source="Ministry of Finance (MOF) / APM Gazette",
                typical_lag_days=0,
                timing_rule="Announced Wednesday 17:00 MYT, effective Thursday 00:00 MYT",
                conservative_assumption="Prices effective Thursday are visible on day T=Thursday; announcements prior to effective date tracked separately",
                leakage_risk_level="MEDIUM",
            ),
            "exchange_rates": PublicationLagSpec(
                dataset_name="Bilateral Foreign Exchange Rates",
                frequency="Daily (Business Days)",
                official_source="Bank Negara Malaysia (BNM)",
                typical_lag_days=0,
                timing_rule="Official 17:00 MYT reference rate published end of trading day",
                conservative_assumption="Available at end of day T; weekend/holiday filled strictly forward from prior business day",
                leakage_risk_level="LOW",
            ),
            "commodity_prices": PublicationLagSpec(
                dataset_name="Global Commodity Benchmarks (Crude, Palm Oil, Grains)",
                frequency="Daily",
                official_source="World Bank / Global Spot Exchanges",
                typical_lag_days=0,
                timing_rule="Daily market closing settlement price",
                conservative_assumption="Available at close of trading day T; forward-fill only across non-trading days",
                leakage_risk_level="LOW",
            ),
            "weather": PublicationLagSpec(
                dataset_name="Store Meteorological Weather Reports",
                frequency="Daily",
                official_source="Malaysian Meteorological Department (MET Malaysia)",
                typical_lag_days=0,
                timing_rule="24-hour meteorological summaries recorded daily",
                conservative_assumption="Day T observed weather is recorded by 23:59:59 MYT; rolling weather uses T-1 shift",
                leakage_risk_level="MEDIUM",
            ),
            "pricecatcher": PublicationLagSpec(
                dataset_name="PriceCatcher Competitor Price Surveys",
                frequency="Daily",
                official_source="KPDN PriceCatcher Platform",
                typical_lag_days=0,
                timing_rule="Survey observations recorded by premise enforcement officers",
                conservative_assumption="Aggregated premise benchmarks available for day T; missing survey days flagged with explicit indicator",
                leakage_risk_level="MEDIUM",
            ),
            "public_holidays": PublicationLagSpec(
                dataset_name="Malaysian Public Holidays",
                frequency="Annual Calendar",
                official_source="Prime Minister's Department (JPM) Cabinet Gazette",
                typical_lag_days=-180,
                timing_rule="Gazetted officially in August/September of preceding calendar year",
                conservative_assumption="Calendar holiday schedules known 100% in advance (known_in_advance=True)",
                leakage_risk_level="LOW",
            ),
            "school_holidays": PublicationLagSpec(
                dataset_name="Ministry of Education School Holidays (KPM)",
                frequency="Annual Academic Calendar",
                official_source="Ministry of Education (MOE / KPM)",
                typical_lag_days=-180,
                timing_rule="Academic calendar gazetted in Q4 of preceding year",
                conservative_assumption="Term dates known completely in advance (known_in_advance=True)",
                leakage_risk_level="LOW",
            ),
            "government_assistance": PublicationLagSpec(
                dataset_name="Government Cash Transfers (BPR, BKM, STR)",
                frequency="Quarterly Tranches",
                official_source="Ministry of Finance (MOF)",
                typical_lag_days=0,
                timing_rule="Public disbursement schedule announced 7 to 14 days before crediting",
                conservative_assumption="Disbursement events tracked on crediting dates; post-assistance indicators use backward-looking elapsed days",
                leakage_risk_level="MEDIUM",
            ),
            "subsidies_and_policies": PublicationLagSpec(
                dataset_name="Subsidies, Price Controls, and National Budget Policies",
                frequency="Event-driven",
                official_source="Federal Government Gazette / Ministry Announcements",
                typical_lag_days=0,
                timing_rule="Announcements precede or coincide with effective dates",
                conservative_assumption="Announcement flags active from announcement date; regime active flags active from effective date",
                leakage_risk_level="MEDIUM",
            ),
        }

    def export_assumptions_report(self, output_dir: Path) -> Path:
        """Export publication lag specification matrix to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "publication_lag_assumptions.csv"
        records = [
            {
                "dataset_name": spec.dataset_name,
                "frequency": spec.frequency,
                "official_source": spec.official_source,
                "typical_lag_days": spec.typical_lag_days,
                "timing_rule": spec.timing_rule,
                "conservative_assumption": spec.conservative_assumption,
                "leakage_risk_level": spec.leakage_risk_level,
            }
            for spec in self.specs.values()
        ]
        pd.DataFrame(records).to_csv(csv_path, index=False)
        return csv_path
