import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("ADENOSINE_DATA_DIR", str(ROOT_DIR / "data")))
RAW_DATA_DIR = Path(os.getenv("ADENOSINE_RAW_DATA_DIR", str(DATA_DIR / "raw")))
PROCESSED_DATA_DIR = Path(os.getenv("ADENOSINE_PROCESSED_DATA_DIR", str(DATA_DIR / "processed")))
MODELS_DIR = Path(os.getenv("ADENOSINE_MODELS_DIR", str(ROOT_DIR / "models")))
OUTPUTS_DIR = Path(os.getenv("ADENOSINE_OUTPUTS_DIR", str(ROOT_DIR / "outputs")))

SUBTYPES = ["A1", "A2A", "A2B", "A3"]

VALID_STANDARD_TYPES = {"KI", "KD", "IC50", "EC50", "AC50"}
REQUIRED_CONFIDENCE = 6
DECOY_PCHEMBL = 4.0

SCAFFOLD_SPLIT_SEED = 42
SCAFFOLD_TEST_SIZE = 0.2

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

MAPIE_CV_FOLDS = 5
MAPIE_CONFIDENCE = 0.90
CONFORMAL_ALPHA = 0.10

Y_RAND_ITERATIONS = 20
GNN_EPOCHS = 100
GNN_HIDDEN_DIM = 256
GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.2
GNN_BATCH_SIZE = 64
GNN_LR = 1e-3
GNN_PATIENCE = 15

NESTED_CV_OUTER_FOLDS = 5
NESTED_CV_INNER_FOLDS = 3
NESTED_CV_TRIALS = 20

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 15
RF_MAX_FEATURES = "sqrt"

FEATURE_NAN_THRESHOLD = 0.05
FEATURE_VAR_THRESHOLD = 0.01
FEATURE_CORR_THRESHOLD = 0.90

SELECTIVITY_MIN_PAIRED = 50
SELECTIVITY_N_ESTIMATORS = 300
SELECTIVITY_LR = 0.05
SELECTIVITY_MAX_DEPTH = 5

ACTIVE_THRESHOLD = 6.0

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "adenosine-selectivity")

LOG_LEVEL = os.getenv("ADENOSINE_LOG_LEVEL", "INFO").upper()

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

# ── Run ID (generated once per process, lazy on first access) ──
import time as _time

_RUN_ID: str | None = None
_RUN_TIMESTAMP: str | None = None


def _get_run_id() -> str:
    global _RUN_ID
    if _RUN_ID is None:
        import uuid
        _RUN_ID = f"ADENO_{_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    return _RUN_ID


def _get_run_timestamp() -> str:
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        _RUN_TIMESTAMP = _time.strftime("%Y-%m-%d %H:%M:%S")
    return _RUN_TIMESTAMP


def __getattr__(name):
    if name == "RUN_ID":
        return _get_run_id()
    if name == "RUN_TIMESTAMP":
        return _get_run_timestamp()
    raise AttributeError(f"module 'src.config' has no attribute {name!r}")

