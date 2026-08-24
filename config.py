from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
SCORES_DIR = RESULTS_DIR / "scores"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Reproducibility
RANDOM_SEED = 42

# LFW dataset
MIN_FACES_PER_PERSON = 10
LFW_RESIZE = 1.0
LFW_COLOR = True
LFW_FUNNELED = True