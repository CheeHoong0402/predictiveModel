"""Optuna study manager with persistent SQLite storage and pruning."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import optuna

from src.utils.paths import PROJECT_ROOT


class PlateauStopper:
    """Early stopping callback for Optuna studies when improvement plateaus."""
    def __init__(self, patience: int = 3, min_improvement_pct: float = 0.1):
        self.patience = patience
        self.min_improvement_pct = min_improvement_pct
        self.best_score = float("inf")
        self.no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        current_score = trial.value
        if current_score is None:
            return

        if self.best_score == float("inf"):
            self.best_score = current_score
            self.no_improvement_count = 0
            return

        improvement_pct = ((self.best_score - current_score) / (abs(self.best_score) + 1e-9)) * 100.0
        if improvement_pct >= self.min_improvement_pct:
            self.best_score = current_score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                study.stop()


def get_or_create_study(
    study_name: str,
    db_path: Path,
    seed: int = 42,
    direction: str = "minimize"
) -> optuna.Study:
    """Load or create persistent Optuna study using local SQLite database."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{db_path.as_posix()}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=2,
        interval_steps=1
    )
    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=direction,
        pruner=pruner,
        sampler=sampler
    )
    return study


def compute_param_importances(study: optuna.Study) -> Dict[str, float]:
    """Extract hyperparameter importance if study has enough completed trials."""
    try:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) >= 3:
            importances = optuna.importance.get_param_importances(study)
            return {k: round(float(v), 4) for k, v in importances.items()}
    except Exception:
        pass
    return {}
