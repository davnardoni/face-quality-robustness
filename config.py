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

# Experimental protocol
SAMPLES_PER_SUBJECT = 10

TRAIN_SAMPLES_PER_SUBJECT = 5
GALLERY_SAMPLES_PER_SUBJECT = 1
PROBE_SAMPLES_PER_SUBJECT = 4

# Classical recognition preprocessing
CLASSICAL_IMAGE_SIZE = (94, 125)  # OpenCV uses (width, height)