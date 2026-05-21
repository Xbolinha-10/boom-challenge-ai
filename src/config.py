from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"

FORWARD_DIR = DATA_RAW / "forward_prediction"

TRAIN_PATH = FORWARD_DIR / "train.csv"
TRAIN_LABELS_PATH = FORWARD_DIR / "train_labels.csv"
TEST_PATH = FORWARD_DIR / "test.csv"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

TARGET_COLUMNS = [
    "P80",
    "fines_frac",
    "oversize_frac",
    "R95",
    "R50_fines",
    "R50_oversize",
]