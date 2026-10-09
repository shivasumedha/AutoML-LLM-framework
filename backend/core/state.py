"""In-memory app state (current dataset + trained model), shared by all routers."""

import pickle
from pathlib import Path
from typing import Any, Optional

from backend.core.config import MODEL_FILE, UPLOAD_DIR


class AppState:
    def __init__(self) -> None:
        self.dataset_path: Optional[Path] = None
        self.dataset_name: Optional[str] = None
        self.model: Any = None
        self.model_name: Optional[str] = None
        self.target_column: Optional[str] = None

    # ---- dataset -------------------------------------------------------

    def restore_last_dataset(self) -> None:
        """After a restart, pick the most recently uploaded CSV."""
        files = sorted(
            UPLOAD_DIR.glob("*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            self.dataset_path = files[0]
            self.dataset_name = files[0].name

    def set_dataset(self, path: Path) -> None:
        self.dataset_path = path
        self.dataset_name = path.name
        self.clear_model()  # a new dataset makes the old model invalid

    # ---- model ---------------------------------------------------------

    def clear_model(self) -> None:
        self.model = None
        self.model_name = None
        self.target_column = None
        MODEL_FILE.unlink(missing_ok=True)

    def save_model(self, model: Any, name: str, target: str) -> None:
        self.model = model
        self.model_name = name
        self.target_column = target
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(
                {"model": model, "model_name": name, "target_column": target}, f
            )

    def load_saved_model(self) -> bool:
        if not MODEL_FILE.exists():
            return False
        try:
            with open(MODEL_FILE, "rb") as f:
                saved = pickle.load(f)
            self.model = saved.get("model")
            self.model_name = saved.get("model_name")
            self.target_column = saved.get("target_column")
            return True
        except Exception:
            return False

    def ensure_model(self) -> bool:
        """True if a model is available (loads it from disk if needed)."""
        if self.model is None:
            self.load_saved_model()
        return self.model is not None


state = AppState()
