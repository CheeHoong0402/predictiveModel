"""Feature display name translation and business family aggregation module."""
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import pandas as pd

from src.utils.paths import PROJECT_ROOT

# Mapping from granular technical families to high-level business families
BUSINESS_FAMILY_MAPPING = {
    # Demand families
    "demand_history": "Historical Demand",
    "rolling_demand": "Historical Demand",
    "demand_momentum": "Historical Demand",
    "weekly_seasonality": "Historical Demand",
    "annual_seasonality": "Historical Demand",
    "observed_sales": "Observed Sales",
    "inventory_state": "Inventory & Stockouts",
    "stockout_history": "Inventory & Stockouts",
    "replenishment_history": "Inventory & Stockouts",
    "inventory_coverage": "Inventory & Stockouts",
    "selling_price": "Selling Price & Elasticity",
    "price_elasticity_proxy": "Selling Price & Elasticity",
    "promotion": "Promotions",
    "calendar": "Calendar & Seasonality",
    "holiday_proximity": "Calendar & Seasonality",
    "festival_proximity": "Calendar & Seasonality",
    "school_holiday": "Calendar & Seasonality",
    "salary_cycle": "Pay & Salary Cycles",
    "government_assistance": "Government Assistance",
    "weather": "Weather & Climate",
    "weather_rolling": "Weather & Climate",
    "weather_spells": "Weather & Climate",
    "market_price": "Market & Competitor Prices",
    "macro_fuel": "Fuel & Energy",
    "macro_currency": "Currency Exchange Rates",
    "macro_commodity": "Commodity & Global Markets",
    "macro_inflation": "Macroeconomic & Inflation",
    "policy_subsidy": "Government Policy & Subsidies",
    "static_product": "Product Attributes",
    "static_store": "Store Geography",
    "procurement_supplier": "Procurement & Supplier",
    "interactions": "Cross-Feature Interactions",
    "missingness_indicator": "Data Quality Indicators",
    "purchase_price_history": "Historical Purchase Price",
    "purchase_price_rolling": "Historical Purchase Price",
    "price_momentum": "Historical Purchase Price"
}

def generate_display_metadata(raw_name: str, family: str) -> Tuple[str, str, str]:
    """Generate display name, unit, and source for a given feature name."""
    name = raw_name.lower()

    # 1. Demand & Sales Lags
    if name.startswith("demand_lag_"):
        lag = name.replace("demand_lag_", "")
        return f"Demand {lag} days ago", "units", "Point-of-Sale History"
    if name.startswith("sales_lag_"):
        lag = name.replace("sales_lag_", "")
        return f"Sales {lag} days ago", "units", "Point-of-Sale History"
    if name.startswith("purchase_price_lag_"):
        lag = name.replace("purchase_price_lag_", "")
        return f"Supplier purchase price {lag} days ago", "MYR", "Procurement Records"

    # 2. Rolling Aggregations
    if name.startswith("rolling_mean_demand_"):
        w = name.replace("rolling_mean_demand_", "")
        return f"Average demand over previous {w} days", "units", "Demand History"
    if name.startswith("rolling_std_demand_"):
        w = name.replace("rolling_std_demand_", "")
        return f"Demand volatility (std) over {w} days", "units", "Demand History"
    if name.startswith("rolling_mean_sales_"):
        w = name.replace("rolling_mean_sales_", "")
        return f"Average sales over previous {w} days", "units", "Sales History"
    if name.startswith("rolling_mean_price_"):
        w = name.replace("rolling_mean_price_", "")
        return f"Average purchase price over previous {w} days", "MYR", "Procurement Records"
    if name.startswith("rolling_mean_purchase_price_"):
        w = name.replace("rolling_mean_purchase_price_", "")
        return f"Average purchase price over previous {w} days", "MYR", "Procurement Records"

    # 3. Macro & FX
    if name in ["fx_rate", "myr_usd_rate"]:
        return "Malaysian ringgit / US dollar exchange rate", "MYR/USD", "Central Bank (BNM)"
    if name in ["fx_pct_change_30d", "myr_usd_change_30d"]:
        return "30-day Malaysian ringgit / United States dollar movement", "%", "Central Bank (BNM)"
    if name.startswith("fx_pct_change_"):
        d = name.replace("fx_pct_change_", "")
        return f"{d} MYR/USD exchange rate movement", "%", "Central Bank (BNM)"
    if name == "fx_change_1d":
        return "1-day MYR/USD exchange rate change", "MYR/USD", "Central Bank (BNM)"
    if name.startswith("commodity_"):
        clean_comm = name.replace("commodity_", "").replace("_", " ").title()
        return f"Commodity index: {clean_comm}", "index/MYR", "Commodity Exchange"
    if name.startswith("fuel_"):
        clean_fuel = name.replace("fuel_", "").replace("_", " ").title()
        return f"Fuel price index: {clean_fuel}", "MYR/litre", "Ministry of Finance"
    if name.startswith("cpi_"):
        clean_cpi = name.replace("cpi_", "").replace("_", " ").title()
        return f"Inflation index: {clean_cpi}", "index/%", "DOSM"

    # 4. Specific Business Features
    special_cases = {
        "promo_active_flag": ("Active promotion scheduled", "binary", "Promotion Schedule"),
        "is_promo": ("Active promotion scheduled", "binary", "Promotion Schedule"),
        "promo_discount_percentage": ("Promotion discount percentage", "%", "Promotion Schedule"),
        "promo_discount_pct": ("Promotion discount percentage", "%", "Promotion Schedule"),
        "promo_days_active_so_far": ("Days promotion active so far", "days", "Promotion Schedule"),
        "planned_promo_within_7d": ("Promotion scheduled within next 7 days", "binary", "Promotion Schedule"),
        "planned_promo_within_14d": ("Promotion scheduled within next 14 days", "binary", "Promotion Schedule"),
        "planned_promo_within_30d": ("Promotion scheduled within next 30 days", "binary", "Promotion Schedule"),
        "selling_price": ("Current retail shelf price", "MYR", "Store Price Management"),
        "selling_price_diff_mean": ("Deviation from product mean selling price", "MYR", "Store Price Management"),
        "stockout_flag": ("Active stockout indicator", "binary", "Inventory Management"),
        "stockout_days_7d": ("Stockout days in previous 7 days", "days", "Inventory Management"),
        "inventory_current": ("Current inventory level on hand", "units", "Inventory Management"),
        "inventory_opening": ("Opening inventory level", "units", "Inventory Management"),
        "inventory_coverage_days": ("Current days of inventory coverage", "days", "Inventory Management"),
        "current_purchase_price": ("Current supplier purchase price", "MYR", "Procurement Records"),
        "purchase_price": ("Current supplier purchase price", "MYR", "Procurement Records"),
        "temp_mean": ("Mean daily temperature", "Celsius", "Malaysian Meteorological Dept"),
        "temp_max": ("Maximum daily temperature", "Celsius", "Malaysian Meteorological Dept"),
        "rainfall_mm": ("Daily rainfall precipitation", "mm", "Malaysian Meteorological Dept"),
        "heavy_rain_flag": ("Heavy rainfall spell indicator", "binary", "Malaysian Meteorological Dept"),
        "is_weekend": ("Weekend indicator (Saturday/Sunday)", "binary", "Calendar"),
        "day_of_week": ("Day of week (Monday to Sunday)", "day", "Calendar"),
        "day_of_month": ("Day of month (1 to 31)", "day", "Calendar"),
        "month": ("Month of year (1 to 12)", "month", "Calendar"),
        "is_public_holiday": ("National public holiday indicator", "binary", "National Gazette"),
        "days_to_major_festival": ("Days until next major festive holiday", "days", "National Gazette"),
        "is_festival_season": ("Active festive holiday season (Raya/CNY/Deepavali)", "binary", "National Gazette"),
        "is_salary_payday": ("Monthly civil service/private payday window", "binary", "Economic Calendar"),
        "is_school_holiday": ("School holiday period indicator", "binary", "Ministry of Education"),
        "is_diesel_rationalisation_active": ("Diesel subsidy rationalisation policy active", "binary", "Ministry of Finance"),
        "is_chicken_subsidy_active": ("Chicken price ceiling/subsidy policy active", "binary", "Ministry of Agriculture"),
        "is_minimum_wage_rm1500_active": ("Minimum wage RM1500 policy active", "binary", "Ministry of Human Resources")
    }

    if name in special_cases:
        return special_cases[name]

    # Clean generic fallback
    clean_display = name.replace("_", " ").title()
    return clean_display, "scalar", "Feature Engineering"


class FeatureDisplayDictionary:
    """Provides fast translation between raw engineering feature names and business display representations."""

    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = Path(root)
        self.config_yaml_path = self.root / "config/feature_display_names.yaml"
        self.csv_dict_path = self.root / "reports/explainability/feature_display_dictionary.csv"
        self._name_to_display: Dict[str, str] = {}
        self._name_to_family: Dict[str, str] = {}
        self._metadata_df: Optional[pd.DataFrame] = None
        self._load_or_build()

    @classmethod
    def load(cls, root: Path = PROJECT_ROOT) -> "FeatureDisplayDictionary":
        """Factory method to load or instantiate dictionary."""
        return cls(root=root)

    def _load_or_build(self) -> None:
        if self.config_yaml_path.exists() and self.csv_dict_path.exists():
            with open(self.config_yaml_path, "r", encoding="utf-8") as f:
                self._name_to_display = yaml.safe_load(f) or {}
            self._metadata_df = pd.read_csv(self.csv_dict_path)
            self._name_to_family = dict(zip(self._metadata_df["raw_feature_name"], self._metadata_df["feature_family"]))
        else:
            self.build_and_save()

    def build_and_save(self) -> None:
        """Construct dictionary from feature registry and manifests, then write YAML and CSV artifacts."""
        reg_path = self.root / "reports/features/feature_registry.csv"
        d_man_path = self.root / "reports/features/demand_feature_manifest.csv"
        p_man_path = self.root / "reports/features/price_feature_manifest.csv"

        df_reg = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame(columns=["name", "description", "feature_family"])
        d_feats = set(pd.read_csv(d_man_path)["name"]) if d_man_path.exists() else set()
        p_feats = set(pd.read_csv(p_man_path)["name"]) if p_man_path.exists() else set()

        records = []
        name_map = {}

        # Add explicit special aliases required by prompt
        explicit_aliases = [
            ("demand_lag_7", "demand_history", "Demand 7 days ago", "units", "Point-of-Sale History"),
            ("rolling_mean_demand_28", "rolling_demand", "Average demand over the previous 28 days", "units", "Demand History"),
            ("myr_usd_change_30d", "macro_currency", "30-day Malaysian ringgit / United States dollar movement", "%", "Central Bank (BNM)")
        ]

        seen_names = set()

        for raw_name, raw_fam, disp_name, unit, src in explicit_aliases:
            biz_fam = BUSINESS_FAMILY_MAPPING.get(raw_fam, raw_fam.replace("_", " ").title())
            name_map[raw_name] = disp_name
            seen_names.add(raw_name)
            records.append({
                "raw_feature_name": raw_name,
                "display_name": disp_name,
                "description": disp_name,
                "feature_family": biz_fam,
                "unit": unit,
                "source": src,
                "task_applicability": "DEMAND_AND_PRICE"
            })

        for _, row in df_reg.iterrows():
            raw_name = str(row["name"])
            if raw_name in seen_names:
                continue
            seen_names.add(raw_name)

            raw_fam = str(row.get("feature_family", "general"))
            desc = str(row.get("description", raw_name))

            biz_fam = BUSINESS_FAMILY_MAPPING.get(raw_fam, raw_fam.replace("_", " ").title())
            display_name, unit, source = generate_display_metadata(raw_name, raw_fam)

            in_d = raw_name in d_feats
            in_p = raw_name in p_feats
            if in_d and in_p:
                applicability = "DEMAND_AND_PRICE"
            elif in_d:
                applicability = "DEMAND"
            elif in_p:
                applicability = "PRICE"
            else:
                applicability = "UNIVERSAL"

            name_map[raw_name] = display_name
            records.append({
                "raw_feature_name": raw_name,
                "display_name": display_name,
                "description": desc,
                "feature_family": biz_fam,
                "unit": unit,
                "source": source,
                "task_applicability": applicability
            })

        df_out = pd.DataFrame(records)

        # Write YAML
        self.config_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(name_map, f, sort_keys=False, allow_unicode=True)

        # Write CSV
        self.csv_dict_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(self.csv_dict_path, index=False, encoding="utf-8")

        self._name_to_display = name_map
        self._metadata_df = df_out
        self._name_to_family = dict(zip(df_out["raw_feature_name"], df_out["feature_family"]))

    def get_display_name(self, raw_feature: str) -> str:
        """Translate raw feature name into user-friendly display name."""
        if raw_feature in self._name_to_display:
            return self._name_to_display[raw_feature]
        disp, _, _ = generate_display_metadata(raw_feature, "")
        if disp != raw_feature.replace("_", " ").title():
            return disp
        return raw_feature.replace("_", " ").title()

    def get_feature_family(self, raw_feature: str) -> str:
        """Resolve feature to its high-level business family."""
        if raw_feature in self._name_to_family:
            return self._name_to_family[raw_feature]
        name = raw_feature.lower()
        if "commodity" in name or "cpo" in name or "brent" in name or "palm_oil" in name:
            return "Commodity & Global Markets"
        if "fuel" in name or "petrol" in name or "diesel" in name or "ron95" in name:
            return "Fuel & Energy"
        if "fx_" in name or "currency" in name or "usd" in name or "myr" in name:
            return "Currency Exchange Rates"
        if "cpi" in name or "inflation" in name:
            return "Macroeconomic & Inflation"
        if "subsidy" in name or "policy" in name or "wage" in name:
            return "Government Policy & Subsidies"
        if "weather" in name or "temp" in name or "rain" in name:
            return "Weather & Climate"
        if "holiday" in name or "festival" in name or "date" in name or "day" in name or "month" in name or "weekend" in name:
            return "Calendar & Seasonality"
        if "promo" in name:
            return "Promotions"
        if "demand" in name or "sales" in name:
            return "Historical Demand"
        if "price" in name or "cost" in name:
            return "Historical Purchase Price"
        return "General Drivers"

    def get_business_family(self, raw_feature: str) -> str:
        """Alias for get_feature_family."""
        return self.get_feature_family(raw_feature)

    def get_metadata(self, raw_feature: str) -> Dict[str, str]:
        """Retrieve complete dictionary record for a feature."""
        if self._metadata_df is not None and not self._metadata_df.empty:
            match = self._metadata_df[self._metadata_df["raw_feature_name"] == raw_feature]
            if not match.empty:
                return match.iloc[0].to_dict()
        return {
            "raw_feature_name": raw_feature,
            "display_name": self.get_display_name(raw_feature),
            "description": raw_feature,
            "feature_family": self.get_feature_family(raw_feature),
            "unit": "scalar",
            "source": "Feature Engineering",
            "task_applicability": "UNIVERSAL"
        }
