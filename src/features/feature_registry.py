"""
Feature Registry Module.
Defines metadata schema and registry container for all engineered features.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
import pandas as pd


@dataclass
class FeatureDefinition:
    """Metadata specification for an engineered feature."""
    name: str
    description: str
    source_columns: str
    calculation: str
    data_type: str
    feature_family: str
    known_at_prediction_time: bool
    expected_missingness: str
    target_model_applicability: Literal["demand", "price", "both"]
    leakage_risk: str = "none"


@dataclass
class ExcludedFeatureDefinition:
    """Metadata specification for an explicitly excluded or quarantined feature."""
    feature_name: str
    feature_family: str
    reason_for_exclusion: str
    risk_type: str


class FeatureRegistry:
    """Central registry container for engineered features and manifests."""

    def __init__(self):
        self._features: Dict[str, FeatureDefinition] = {}
        self._excluded: List[ExcludedFeatureDefinition] = []

    def register(self, feature: FeatureDefinition) -> None:
        """Register a feature definition."""
        if feature.name in self._features:
            raise ValueError(f"Feature '{feature.name}' is already registered in FeatureRegistry.")
        self._features[feature.name] = feature

    def register_many(self, features: List[FeatureDefinition]) -> None:
        """Register a collection of feature definitions."""
        for feat in features:
            self.register(feat)

    def register_excluded(self, excluded: ExcludedFeatureDefinition) -> None:
        """Register an excluded feature with explicit justification."""
        self._excluded.append(excluded)

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Retrieve definition for a single feature."""
        return self._features.get(name)

    @property
    def all_features(self) -> List[FeatureDefinition]:
        """Return all registered feature definitions."""
        return list(self._features.values())

    @property
    def feature_names(self) -> List[str]:
        """Return names of all registered features."""
        return list(self._features.keys())

    @property
    def demand_features(self) -> List[FeatureDefinition]:
        """Return features applicable to demand forecasting model."""
        return [f for f in self._features.values() if f.target_model_applicability in ("demand", "both")]

    @property
    def price_features(self) -> List[FeatureDefinition]:
        """Return features applicable to purchase price forecasting model."""
        return [f for f in self._features.values() if f.target_model_applicability in ("price", "both")]

    def to_dataframe(self) -> pd.DataFrame:
        """Export registry to a pandas DataFrame."""
        records = [asdict(f) for f in self._features.values()]
        return pd.DataFrame(records)

    def save_reports(self, output_dir: Path) -> Dict[str, Path]:
        """
        Generate and save feature registry, demand manifest, price manifest,
        and excluded features CSV reports.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Full feature registry
        df_registry = self.to_dataframe()
        registry_path = output_dir / "feature_registry.csv"
        df_registry.to_csv(registry_path, index=False)

        # 2. Demand manifest
        df_demand = pd.DataFrame([asdict(f) for f in self.demand_features])
        demand_manifest_path = output_dir / "demand_feature_manifest.csv"
        df_demand.to_csv(demand_manifest_path, index=False)

        # 3. Price manifest
        df_price = pd.DataFrame([asdict(f) for f in self.price_features])
        price_manifest_path = output_dir / "price_feature_manifest.csv"
        df_price.to_csv(price_manifest_path, index=False)

        # 4. Excluded features
        df_excluded = pd.DataFrame([asdict(e) for e in self._excluded])
        excluded_path = output_dir / "excluded_features.csv"
        df_excluded.to_csv(excluded_path, index=False)

        return {
            "registry": registry_path,
            "demand_manifest": demand_manifest_path,
            "price_manifest": price_manifest_path,
            "excluded_features": excluded_path,
        }
