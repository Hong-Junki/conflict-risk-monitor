from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed" / "dataset"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SUBMISSION_DIR = ARTIFACT_DIR / "submissions"

TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "val.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
FULL_PATH = DATA_DIR / "full.parquet"
FULL_SE_PATH = DATA_DIR / "full_se.parquet"

TARGET_COL = "y_escalation"
ID_COLS = ["date", "country"]
LABEL_META_COLS = [
    "y",
    "y_onset",
    "y_escalation",
    "fatalities_next3d",
    "event_count_next3d",
    "past14d_event_count",
    "past14d_fatalities_mean",
]

SEED = 42
TRAIN_END = "2023-12-31"
VAL_END = "2024-06-30"
