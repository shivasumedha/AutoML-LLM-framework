from pathlib import Path
import os

# backend/ folder
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("AUTOML_DATA_DIR", BASE_DIR)).resolve()

UPLOAD_DIR = Path(os.getenv("AUTOML_UPLOAD_DIR", DATA_DIR / "uploads")).resolve()
MODEL_DIR = Path(os.getenv("AUTOML_MODEL_DIR", DATA_DIR / "models")).resolve()
MODEL_FILE = MODEL_DIR / "trained_model.pkl"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

API_VERSION = "1.1.0"

# Any localhost port is allowed, so 5173 / 5174 / 3000 all work.
CORS_ORIGIN_REGEX = r"http://(localhost|127\.0\.0\.1):\d+"
