from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BATCH_DIR = PROCESSED_DATA_DIR / "batches"

# Data split ratios
REFERENCE_RATIO = 0.45      # 45% for training
VALIDATION_RATIO = 0.10     # 10% for validation
PRODUCTION_RATIO = 0.45     # 45% for production simulation

# Number of production batches to create
NUM_BATCHES = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# Column names
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
LABEL_COLUMN = "Class"
