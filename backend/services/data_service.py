"""CSV loading helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException

from backend.core.state import state


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read a CSV, falling back to latin-1 for Excel-exported files."""
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def load_current_df() -> pd.DataFrame:
    if state.dataset_path is None:
        raise HTTPException(status_code=404, detail="Please upload a dataset first.")
    df = read_csv_safe(state.dataset_path)
    return df.replace([np.inf, -np.inf], np.nan)


def records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")
