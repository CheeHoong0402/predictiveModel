"""Calibrator model implementations for Demand and Purchase-Price uncertainty quantification."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.calibration.metrics import enforce_monotonic_quantiles


class BaseCalibrator(ABC):
    """Abstract base class for uncertainty calibrators."""

    def __init__(
        self,
        task: str,
        horizon: int,
        alpha: float = 0.20,
        min_samples: int = 24
    ):
        self.task = str(task).lower()
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self.is_fitted = False
        self.sample_count = 0
        self.method_name = "base"
        self.empirical_residuals_: np.ndarray = np.array([])

    @abstractmethod
    def fit(
        self,
        y_true: Union[np.ndarray, pd.Series],
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "BaseCalibrator":
        """Fit calibration parameters on historical out-of-fold observations."""
        pass

    @abstractmethod
    def predict_uncertainty(
        self,
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Produce calibrated P10, P50, P90 quantiles for new predictions."""
        pass

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize calibrator artifact using joblib."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseCalibrator":
        """Load calibrator artifact from disk."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Calibrator artifact not found at {p}")
        return joblib.load(p)


class ConformalQuantileCalibrator(BaseCalibrator):
    """Preferred Method A: Conformalized Quantile Regression (CQR) adjustment.
    
    Adjusts raw model-native quantiles using conformal nonconformity scores:
    s_i = max(q10_i - y_i, y_i - q90_i)
    Correction quantile: Q_{1 - alpha}(s)
    Calibrated P10 = raw_q10 - Q
    Calibrated P90 = raw_q90 + Q
    """

    def __init__(
        self,
        task: str,
        horizon: int,
        alpha: float = 0.20,
        min_samples: int = 24
    ):
        super().__init__(task=task, horizon=horizon, alpha=alpha, min_samples=min_samples)
        self.method_name = "conformal_quantile_regression"
        self.q_correction_: float = 0.0
        self.fallback_q_low_: float = 0.0
        self.fallback_q_high_: float = 0.0
        self.fallback_q_med_: float = 0.0

    def fit(
        self,
        y_true: Union[np.ndarray, pd.Series],
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "ConformalQuantileCalibrator":
        y = np.asarray(y_true, dtype=float)
        y_hat = np.asarray(point_pred, dtype=float)
        n = len(y)
        self.sample_count = n

        # Fallback residual distribution
        residuals = y - y_hat
        self.empirical_residuals_ = residuals.copy()
        
        if n < self.min_samples:
            # Fallback for insufficient sample size
            self.fallback_q_low_ = float(np.percentile(residuals, self.alpha / 2 * 100)) if n > 0 else 0.0
            self.fallback_q_high_ = float(np.percentile(residuals, (1 - self.alpha / 2) * 100)) if n > 0 else 0.0
            self.fallback_q_med_ = float(np.median(residuals)) if n > 0 else 0.0
            self.q_correction_ = 0.0
            self.is_fitted = True
            return self

        self.fallback_q_low_ = float(np.percentile(residuals, (self.alpha / 2) * 100))
        self.fallback_q_high_ = float(np.percentile(residuals, (1.0 - self.alpha / 2) * 100))
        self.fallback_q_med_ = float(np.median(residuals))

        if raw_q10 is not None and raw_q90 is not None:
            q10 = np.asarray(raw_q10, dtype=float)
            q90 = np.asarray(raw_q90, dtype=float)
            s = np.maximum(q10 - y, y - q90)
            # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
            target_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
            self.q_correction_ = float(np.quantile(s, target_level))
        else:
            self.q_correction_ = 0.0

        self.is_fitted = True
        return self

    def predict_uncertainty(
        self,
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predict_uncertainty is called.")

        y_hat = np.asarray(point_pred, dtype=float)

        if raw_q10 is not None and raw_q90 is not None:
            q10 = np.asarray(raw_q10, dtype=float)
            q90 = np.asarray(raw_q90, dtype=float)
            q50 = np.asarray(raw_q50, dtype=float) if raw_q50 is not None else y_hat
            p10 = q10 - self.q_correction_
            p90 = q90 + self.q_correction_
            p50 = q50
        else:
            p10 = y_hat + self.fallback_q_low_
            p50 = y_hat
            p90 = y_hat + self.fallback_q_high_

        return enforce_monotonic_quantiles(p10, p50, p90, task=self.task)


class AsymmetricResidualCalibrator(BaseCalibrator):
    """Method B: Asymmetric Out-of-Fold Residual Conformal Interval.
    
    Constructs uncertainty directly from empirical residual percentiles:
    e_i = y_i - y_hat_i
    P10 = y_hat + Q_{alpha/2}(e)
    P50 = y_hat
    P90 = y_hat + Q_{1 - alpha/2}(e)
    """

    def __init__(
        self,
        task: str,
        horizon: int,
        alpha: float = 0.20,
        min_samples: int = 24
    ):
        super().__init__(task=task, horizon=horizon, alpha=alpha, min_samples=min_samples)
        self.method_name = "asymmetric_residual_conformal"
        self.q_low_: float = 0.0
        self.q_high_: float = 0.0
        self.q_med_: float = 0.0

    def fit(
        self,
        y_true: Union[np.ndarray, pd.Series],
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "AsymmetricResidualCalibrator":
        y = np.asarray(y_true, dtype=float)
        y_hat = np.asarray(point_pred, dtype=float)
        n = len(y)
        self.sample_count = n

        residuals = y - y_hat
        self.empirical_residuals_ = residuals.copy()

        if n < self.min_samples:
            self.q_low_ = float(np.percentile(residuals, (self.alpha / 2) * 100)) if n > 0 else 0.0
            self.q_high_ = float(np.percentile(residuals, (1.0 - self.alpha / 2) * 100)) if n > 0 else 0.0
            self.q_med_ = float(np.median(residuals)) if n > 0 else 0.0
            self.is_fitted = True
            return self

        self.q_low_ = float(np.percentile(residuals, (self.alpha / 2) * 100))
        self.q_high_ = float(np.percentile(residuals, (1.0 - self.alpha / 2) * 100))
        self.q_med_ = float(np.median(residuals))
        self.is_fitted = True
        return self

    def predict_uncertainty(
        self,
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predict_uncertainty is called.")

        y_hat = np.asarray(point_pred, dtype=float)
        p10 = y_hat + self.q_low_
        p50 = y_hat
        p90 = y_hat + self.q_high_

        return enforce_monotonic_quantiles(p10, p50, p90, task=self.task)


class SymmetricResidualCalibrator(BaseCalibrator):
    """Method C: Symmetric Absolute Residual Conformal Interval.
    
    r_i = |y_i - y_hat_i|
    P10 = y_hat - Q_{1 - alpha}(r)
    P50 = y_hat
    P90 = y_hat + Q_{1 - alpha}(r)
    """

    def __init__(
        self,
        task: str,
        horizon: int,
        alpha: float = 0.20,
        min_samples: int = 24
    ):
        super().__init__(task=task, horizon=horizon, alpha=alpha, min_samples=min_samples)
        self.method_name = "symmetric_residual_conformal"
        self.q_abs_: float = 0.0

    def fit(
        self,
        y_true: Union[np.ndarray, pd.Series],
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> "SymmetricResidualCalibrator":
        y = np.asarray(y_true, dtype=float)
        y_hat = np.asarray(point_pred, dtype=float)
        n = len(y)
        self.sample_count = n

        residuals = y - y_hat
        self.empirical_residuals_ = residuals.copy()

        abs_residuals = np.abs(residuals)
        target_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n)) if n >= self.min_samples else (1.0 - self.alpha)
        self.q_abs_ = float(np.quantile(abs_residuals, target_level)) if n > 0 else 0.0
        self.is_fitted = True
        return self

    def predict_uncertainty(
        self,
        point_pred: Union[np.ndarray, pd.Series],
        raw_q10: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q50: Optional[Union[np.ndarray, pd.Series]] = None,
        raw_q90: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predict_uncertainty is called.")

        y_hat = np.asarray(point_pred, dtype=float)
        p10 = y_hat - self.q_abs_
        p50 = y_hat
        p90 = y_hat + self.q_abs_

        return enforce_monotonic_quantiles(p10, p50, p90, task=self.task)
