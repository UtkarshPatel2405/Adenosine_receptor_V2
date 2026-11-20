import os
import time as _time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("ADENOSINE_DATA_DIR", str(ROOT_DIR / "data")))
RAW_DATA_DIR = Path(os.getenv("ADENOSINE_RAW_DATA_DIR", str(DATA_DIR / "raw")))
PROCESSED_DATA_DIR = Path(os.getenv("ADENOSINE_PROCESSED_DATA_DIR", str(DATA_DIR / "processed")))
MODELS_DIR = Path(os.getenv("ADENOSINE_MODELS_DIR", str(ROOT_DIR / "models")))
OUTPUTS_DIR = Path(os.getenv("ADENOSINE_OUTPUTS_DIR", str(ROOT_DIR / "outputs")))

APP_VERSION = "2.0.0"

SUBTYPES = ["A1", "A2A", "A2B", "A3"]
CHEMBL_TARGETS = {
    "A1": "CHEMBL226",
    "A2A": "CHEMBL251",
    "A2B": "CHEMBL255",
    "A3": "CHEMBL256",
}

VALID_STANDARD_TYPES = {"KI", "KD"}
VALID_FUNCTIONAL_TYPES = {"IC50", "EC50"}
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

FEATURE_NAN_THRESHOLD = 0.05
FEATURE_VAR_THRESHOLD = 0.01
FEATURE_CORR_THRESHOLD = 0.90

ACTIVE_THRESHOLD = 6.0

LOG_LEVEL = os.getenv("ADENOSINE_LOG_LEVEL", "INFO").upper()

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

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


def __getattr__(name: str):
    if name == "RUN_ID":
        return _get_run_id()
    if name == "RUN_TIMESTAMP":
        return _get_run_timestamp()
    raise AttributeError(f"module 'src.config' has no attribute {name!r}")
