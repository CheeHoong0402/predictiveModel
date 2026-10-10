import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Core Directories
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DIR = PROCESSED_DATA_DIR
PREDICTIONS_DATA_DIR = DATA_DIR / "predictions"

MODELS_DIR = PROJECT_ROOT / "models"
DEMAND_MODELS_DIR = MODELS_DIR / "demand"
PRICE_MODELS_DIR = MODELS_DIR / "price"
METADATA_MODELS_DIR = MODELS_DIR / "metadata"

REPORTS_DIR = PROJECT_ROOT / "reports"
VALIDATION_REPORTS_DIR = REPORTS_DIR / "validation"
FIGURES_REPORTS_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_REPORTS_DIR = REPORTS_DIR / "experiments"

# Ensure crucial data subdirectories exist
for d in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DATA_DIR,
          DEMAND_MODELS_DIR, PRICE_MODELS_DIR, METADATA_MODELS_DIR,
          VALIDATION_REPORTS_DIR, FIGURES_REPORTS_DIR, EXPERIMENTS_REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
