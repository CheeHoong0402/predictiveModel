"""
src/demand_correction/confidence_estimator.py
=============================================
Assigns categorical confidence levels (HIGH, MEDIUM, LOW, UNCORRECTED)
based on reference sample size, evidence certainty, and demand stability.
"""

from typing import Dict, Any


class ConfidenceEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        conf_cfg = config.get("confidence_thresholds", {})
        self.high_min_obs = int(conf_cfg.get("high", {}).get("min_observations", 8))
        self.med_min_obs = int(conf_cfg.get("medium", {}).get("min_observations", 4))
        self.low_min_obs = int(conf_cfg.get("low", {}).get("min_observations", 1))

    def evaluate_confidence(
        self,
        is_censored: bool,
        correction_applied: int,
        ref_count: int,
        evidence_level: str
    ) -> str:
        """
        Determines confidence category for an observation.
        """
        if not is_censored:
            return "HIGH"  # Unconstrained observation is highly confident as observed demand

        # Censored cases
        if ref_count == 0:
            return "UNCORRECTED"
        elif ref_count >= self.high_min_obs and evidence_level == "STRONG":
            return "HIGH"
        elif ref_count >= self.med_min_obs:
            return "MEDIUM"
        elif ref_count >= self.low_min_obs:
            return "LOW"
        else:
            return "UNCORRECTED"
