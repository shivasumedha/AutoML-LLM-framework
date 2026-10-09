from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import pickle
import re
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from backend.services.assistant_service import ask_groq
from backend.services.data_service import read_csv_safe
from backend.services import persistence


# =========================================================
# DIRECTORIES
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AUTOML_DATA_DIR", BASE_DIR)).resolve()
UPLOAD_DIR = Path(os.getenv("AUTOML_UPLOAD_DIR", DATA_DIR / "uploads")).resolve()
MODEL_DIR = Path(os.getenv("AUTOML_MODEL_DIR", DATA_DIR / "models")).resolve()
MODEL_FILE = MODEL_DIR / "trained_model.pkl"
VERSION_DIR = Path(os.getenv("AUTOML_VERSION_DIR", DATA_DIR / "dataset_versions")).resolve()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VERSION_DIR.mkdir(parents=True, exist_ok=True)

# Read an optional backend/.env without ever sending its contents to the browser.
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    for env_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        env_line = env_line.strip()
        if env_line and not env_line.startswith("#") and "=" in env_line:
            env_key, env_value = env_line.split("=", 1)
            os.environ.setdefault(env_key.strip(), env_value.strip().strip("\"'"))

# Migrate the old configured model in memory after loading .env. Keep the user's
# key and other settings untouched while ensuring this app uses the requested GPT model.
if os.getenv("GROQ_MODEL") == "llama-3.3-70b-versatile":
    os.environ["GROQ_MODEL"] = "openai/gpt-oss-120b"


# =========================================================
# GLOBAL STATE
# =========================================================

current_dataset_path = None
current_dataset_name = None
trained_model = None
trained_model_name = None
target_column = None
training_summary = None
dataset_version_history = []
pending_chat_actions = {}
_version_history_by_session = {}
_pending_actions_by_session = {}


# =========================================================
# STARTUP: restore the last dataset and model after a restart
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_dataset_path, current_dataset_name, trained_model, trained_model_name, target_column, training_summary

    persistence.initialize()
    saved_dataset = persistence.latest_dataset()
    if saved_dataset:
        saved_name, saved_content = saved_dataset
        restored_path = UPLOAD_DIR / Path(saved_name).name
        restored_path.write_bytes(saved_content)
        current_dataset_path = restored_path
        current_dataset_name = restored_path.name

    files = sorted(
        [*UPLOAD_DIR.glob("*.csv"), *VERSION_DIR.glob("*.csv")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if current_dataset_path is None and files:
        current_dataset_path = files[0]
        current_dataset_name = files[0].name
        persistence.save_dataset(current_dataset_name, current_dataset_path.read_bytes())

    load_saved_model()

    print("=" * 60)
    print("AutoML FastAPI Backend Started")
    print(f"Dataset : {current_dataset_name}")
    print(f"Model   : {trained_model_name}")
    print("Docs    : http://127.0.0.1:8000/docs")
    print("=" * 60)
    yield


# =========================================================
# APP + CORS
# =========================================================

app = FastAPI(
    title="AutoML LLM Framework API",
    description="Backend API for AutoML dataset processing and chatbot",
    version="1.1.0",
    lifespan=lifespan,
)

# Local Vite ports work by default. Production origins are configured on Render
# as a comma-separated CORS_ORIGINS value.
cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def restore_session_workspace(session_id):
    """Load the selected chat's independent dataset and model into the legacy services."""
    global current_dataset_path, current_dataset_name, trained_model, trained_model_name
    global target_column, training_summary, dataset_version_history, pending_chat_actions
    dataset_version_history = _version_history_by_session.setdefault(session_id, [])
    pending_chat_actions = _pending_actions_by_session.setdefault(session_id, {})
    current_dataset_path = current_dataset_name = None
    trained_model = trained_model_name = target_column = training_summary = None

    saved_dataset = persistence.latest_dataset()
    if saved_dataset:
        name, content = saved_dataset
        session_dir = UPLOAD_DIR / "workspaces" / Path(session_id).name
        session_dir.mkdir(parents=True, exist_ok=True)
        current_dataset_path = session_dir / Path(name).name
        current_dataset_path.write_bytes(content)
        current_dataset_name = current_dataset_path.name

    MODEL_FILE.unlink(missing_ok=True)
    saved_model = persistence.get_artifact("trained_model")
    if saved_model:
        MODEL_FILE.write_bytes(saved_model[0])
        load_saved_model()


@app.middleware("http")
async def select_chat_workspace(request, call_next):
    session_id = request.headers.get("x-chat-session")
    if not session_id:
        return await call_next(request)
    persistence.create_session(session_id)
    token = persistence.bind_session(session_id)
    try:
        restore_session_workspace(session_id)
        return await call_next(request)
    finally:
        persistence.reset_session(token)


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)
    confirm_action: str | None = None
    session_id: str | None = None


class RenameSessionRequest(BaseModel):
    title: str


class TrainRequest(BaseModel):
    target_column: str
    model_name: str | None = None


class PredictionRequest(BaseModel):
    data: dict


class CleanDatasetRequest(BaseModel):
    remove_duplicates: bool = True
    fill_missing: bool = True
    remove_outliers: bool = True
    outlier_columns: list[str] | None = None


# =========================================================
# HELPERS
# =========================================================

def load_current_df():
    if current_dataset_path is None:
        raise HTTPException(
            status_code=404,
            detail="Please upload a dataset first.",
        )
    df = read_csv_safe(current_dataset_path)
    return df.replace([np.inf, -np.inf], np.nan)


def records(df):
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def load_saved_model():
    global trained_model, trained_model_name, target_column, training_summary

    try:
        stored = persistence.get_artifact("trained_model")
        if stored:
            content, _metadata = stored
            MODEL_FILE.write_bytes(content)
        if not MODEL_FILE.exists():
            return False
        saved = pickle.loads(MODEL_FILE.read_bytes())
        if not stored:
            persistence.save_artifact("trained_model", MODEL_FILE.read_bytes(), {})
        trained_model = saved.get("model")
        trained_model_name = saved.get("model_name")
        target_column = saved.get("target_column")
        training_summary = saved.get("training_summary") or {
            "selected_model": trained_model_name,
            "target_column": target_column,
            "results": [],
            "best_model": trained_model_name,
        }
        return True
    except Exception:
        return False


# =========================================================
# ROOT + HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "AutoML FastAPI backend is running",
        "version": "1.1.0",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "backend": "FastAPI"}


# =========================================================
# UPLOAD DATASET
# =========================================================

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    global current_dataset_path, current_dataset_name
    global trained_model, trained_model_name, target_column
    global training_summary
    global dataset_version_history, pending_chat_actions

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename received.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        safe_name = Path(file.filename).name
        session_id = persistence.current_session()
        upload_root = UPLOAD_DIR / "workspaces" / Path(session_id).name if session_id else UPLOAD_DIR
        upload_root.mkdir(parents=True, exist_ok=True)
        file_path = upload_root / safe_name
        if file_path.exists():
            file_path = upload_root / f"{Path(safe_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.csv"
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        try:
            df = read_csv_safe(file_path)
        except Exception as csv_error:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid CSV file: {csv_error}",
            )

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV contains no rows.")

        current_dataset_path = file_path
        current_dataset_name = file_path.name
        persistence.save_dataset(current_dataset_name, content)

        # A new dataset makes the old model invalid
        trained_model = None
        trained_model_name = None
        target_column = None
        training_summary = None
        dataset_version_history.clear()
        pending_chat_actions.clear()
        MODEL_FILE.unlink(missing_ok=True)
        persistence.delete_artifact("trained_model")

        return {
            "status": "success",
            "message": "Dataset uploaded successfully.",
            "filename": file_path.name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset upload failed: {e}")


# =========================================================
# DATASET INFO
# =========================================================

@app.get("/dataset")
def get_dataset():
    try:
        df = load_current_df()
        return {
            "status": "success",
            "filename": current_dataset_name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "preview": records(df.head(10)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/preview")
def preview_dataset(rows: int = 10):
    try:
        df = load_current_df()
        rows = max(1, min(rows, 100))
        return {"status": "success", "data": records(df.head(rows))}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def dataset_quality(df):
    numeric = df.select_dtypes(include=np.number)
    outliers = {}
    union_mask = pd.Series(False, index=df.index)
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            count = 0
        else:
            q1, q3 = values.quantile(.25), values.quantile(.75)
            spread = q3 - q1
            mask = ((df[column] < q1 - 1.5 * spread) | (df[column] > q3 + 1.5 * spread)).fillna(False)
            count = int(mask.sum())
            union_mask |= mask
        outliers[str(column)] = count
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "numeric_columns": [str(c) for c in numeric.columns],
        "outliers_by_column": outliers,
        "outlier_rows": int(union_mask.sum()),
    }


def clean_dataset_frame(df, options):
    original_rows = len(df)
    duplicate_count = int(df.duplicated().sum()) if options.remove_duplicates else 0
    cleaned = df.drop_duplicates().copy() if options.remove_duplicates else df.copy()
    filled_by_column = {}

    if options.fill_missing:
        for column in cleaned.columns:
            missing = int(cleaned[column].isna().sum())
            if not missing:
                continue
            if pd.api.types.is_numeric_dtype(cleaned[column]):
                value = cleaned[column].median()
            else:
                modes = cleaned[column].mode(dropna=True)
                value = modes.iloc[0] if not modes.empty else None
            if value is not None and not pd.isna(value):
                cleaned[column] = cleaned[column].fillna(value)
                filled_by_column[str(column)] = missing

    outlier_removed = 0
    selected = options.outlier_columns if options.outlier_columns is not None else [str(c) for c in cleaned.select_dtypes(include=np.number).columns]
    if options.remove_outliers:
        invalid = [column for column in selected if column not in cleaned.columns or not pd.api.types.is_numeric_dtype(cleaned[column])]
        if invalid:
            raise ValueError(f"Outlier columns must be numeric columns in this dataset: {invalid}")
        keep = pd.Series(True, index=cleaned.index)
        for column in selected:
            values = cleaned[column].dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile(.25), values.quantile(.75)
            spread = q3 - q1
            keep &= ((cleaned[column] >= q1 - 1.5 * spread) & (cleaned[column] <= q3 + 1.5 * spread)).fillna(True)
        cleaned = cleaned.loc[keep].copy()
        outlier_removed = int(len(keep) - keep.sum())

    if cleaned.empty:
        raise ValueError("These cleaning choices would remove every row. Select fewer outlier columns or keep outliers.")
    return cleaned, {
        "rows_before": int(original_rows),
        "rows_after": int(len(cleaned)),
        "duplicates_removed": duplicate_count,
        "outlier_rows_removed": outlier_removed,
        "missing_values_filled": int(sum(filled_by_column.values())),
        "filled_by_column": filled_by_column,
        "missing_values_remaining": int(cleaned.isna().sum().sum()),
    }


@app.get("/dataset/quality")
def get_dataset_quality():
    try:
        return dataset_quality(load_current_df())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset quality analysis failed: {e}")


@app.post("/dataset/clean-preview")
def preview_dataset_cleaning(request: CleanDatasetRequest):
    try:
        cleaned, summary = clean_dataset_frame(load_current_df(), request)
        return {"summary": summary, "column_names": [str(c) for c in cleaned.columns], "preview": records(cleaned.head(10))}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not preview cleaning: {e}")


@app.post("/dataset/clean")
def apply_dataset_cleaning(request: CleanDatasetRequest):
    try:
        cleaned, summary = clean_dataset_frame(load_current_df(), request)
        save_dataset_version(cleaned)
        return {"status": "success", "filename": current_dataset_name, "column_names": [str(c) for c in cleaned.columns], "summary": summary}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clean dataset: {e}")


@app.get("/dataset/download")
def download_current_dataset():
    try:
        df = load_current_df()
        content = df.to_csv(index=False).encode("utf-8-sig")
        filename = f"cleaned_{Path(current_dataset_name or 'dataset.csv').name}"
        return Response(content=content, media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not download dataset: {e}")


@app.get("/dataset-history")
def get_dataset_history():
    version_root = VERSION_DIR / Path(persistence.current_session()).name if persistence.current_session() else VERSION_DIR
    if current_dataset_name:
        origin = re.sub(r"_v\d{8}_\d{6}_\d{6}$", "", Path(current_dataset_name).stem)
        version_files = [p for p in version_root.glob("*.csv") if p.stem.startswith(f"{origin}_v")]
    else:
        version_files = []
    versions = sorted(version_files, key=lambda p: p.stat().st_mtime, reverse=True)
    entries = [
        {"filename": p.name, "created_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(), "current": p == current_dataset_path}
        for p in versions
    ]
    if current_dataset_path and not any(p == current_dataset_path for p in versions):
        entries.insert(0, {"filename": current_dataset_name, "created_at": datetime.fromtimestamp(current_dataset_path.stat().st_mtime, timezone.utc).isoformat(), "current": True})
    return {"current": current_dataset_name, "versions": entries}


def apply_dataset_action(df, args):
    operation = args.get("operation")
    column = args.get("column")
    if operation == "remove_duplicates":
        updated = df.drop_duplicates().copy()
        message = f"Removed {len(df) - len(updated)} duplicate rows."
    elif operation == "drop_column":
        if column not in df.columns:
            raise ValueError(f"Column not found. Available columns: {list(df.columns)}")
        if len(df.columns) < 2:
            raise ValueError("The dataset must keep at least one column.")
        updated = df.drop(columns=[column]).copy()
        message = f"Dropped column '{column}'."
    elif operation == "fill_missing":
        if column not in df.columns:
            raise ValueError(f"Column not found. Available columns: {list(df.columns)}")
        strategy = args.get("strategy") or ("median" if pd.api.types.is_numeric_dtype(df[column]) else "mode")
        if strategy in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"{strategy} can only fill a numeric column.")
            value = df[column].mean() if strategy == "mean" else df[column].median()
        elif strategy == "mode":
            modes = df[column].mode(dropna=True)
            if modes.empty:
                raise ValueError("This column has no non-missing value to use as its mode.")
            value = modes.iloc[0]
        elif strategy == "value":
            value = args.get("value")
            if value is None:
                raise ValueError("Provide a value to fill missing cells.")
        else:
            raise ValueError("Choose mean, median, mode, or value as the fill strategy.")
        missing = int(df[column].isna().sum())
        updated = df.copy()
        updated[column] = updated[column].fillna(value)
        message = f"Filled {missing} missing values in '{column}' using {strategy}."
    elif operation == "filter_rows":
        if column not in df.columns:
            raise ValueError(f"Column not found. Available columns: {list(df.columns)}")
        operator, value = args.get("operator", "eq"), args.get("value")
        series = df[column]
        if operator == "contains": mask = series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
        else:
            if pd.api.types.is_numeric_dtype(series):
                value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(value): raise ValueError("Filter value must be numeric for this column.")
            comparisons = {"eq": lambda: series == value, "ne": lambda: series != value,
                           "gt": lambda: series > value, "gte": lambda: series >= value,
                           "lt": lambda: series < value, "lte": lambda: series <= value}
            if operator not in comparisons: raise ValueError("Unsupported filter operator.")
            try: mask = comparisons[operator]()
            except TypeError: raise ValueError("That comparison is not valid for this column.")
        updated = df.loc[mask].copy()
        if updated.empty: raise ValueError("That filter would remove every row; no change was made.")
        message = f"Kept {len(updated)} of {len(df)} rows where {column} {operator} {value}."
    else:
        raise ValueError("Unsupported dataset edit.")
    return updated, message


def save_dataset_version(updated):
    global current_dataset_path, current_dataset_name, trained_model, trained_model_name, target_column, training_summary
    if current_dataset_path is None:
        raise ValueError("Upload a dataset before editing it.")
    previous_path = current_dataset_path
    previous_name = current_dataset_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = re.sub(r"_v\d{8}_\d{6}_\d{6}$", "", Path(previous_name or "dataset.csv").stem)
    version_root = VERSION_DIR / Path(persistence.current_session()).name if persistence.current_session() else VERSION_DIR
    version_root.mkdir(parents=True, exist_ok=True)
    if previous_path.parent != version_root and previous_path.exists():
        original_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (version_root / f"{base}_v{original_stamp}.csv").write_bytes(previous_path.read_bytes())
    version_path = version_root / f"{base}_v{stamp}.csv"
    updated.to_csv(version_path, index=False)
    dataset_version_history.append((previous_path, previous_name))
    current_dataset_path = version_path
    current_dataset_name = version_path.name
    persistence.save_dataset(current_dataset_name, version_path.read_bytes())
    trained_model = trained_model_name = target_column = None
    training_summary = None
    MODEL_FILE.unlink(missing_ok=True)
    persistence.delete_artifact("trained_model")
    return version_path


def undo_dataset_version():
    global current_dataset_path, current_dataset_name, trained_model, trained_model_name, target_column, training_summary
    if dataset_version_history:
        current_dataset_path, current_dataset_name = dataset_version_history.pop()
    else:
        version_root = VERSION_DIR / Path(persistence.current_session()).name if persistence.current_session() else VERSION_DIR
        if not current_dataset_name or not current_dataset_path or current_dataset_path.parent != version_root:
            return False
        origin = re.sub(r"_v\d{8}_\d{6}_\d{6}$", "", Path(current_dataset_name).stem)
        versions = sorted((p for p in version_root.glob("*.csv") if p != current_dataset_path and p.stem.startswith(f"{origin}_v")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not versions:
            return False
        current_dataset_path = versions[0]
        current_dataset_name = versions[0].name
    trained_model = trained_model_name = target_column = None
    training_summary = None
    MODEL_FILE.unlink(missing_ok=True)
    persistence.delete_artifact("trained_model")
    if current_dataset_path and current_dataset_path.exists():
        persistence.save_dataset(current_dataset_name, current_dataset_path.read_bytes())
    return True


# =========================================================
# ANALYSIS
# =========================================================

@app.get("/analyze")
def analyze_dataset():
    try:
        df = load_current_df()

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

        missing_values = {
            str(k): int(v)
            for k, v in df.isnull().sum().sort_values(ascending=False).items()
        }

        numeric_summary = {}
        if numeric_columns:
            desc = df[numeric_columns].describe()
            numeric_summary = {
                str(col): {
                    str(stat): (None if pd.isna(val) else float(val))
                    for stat, val in desc[col].items()
                }
                for col in desc.columns
            }

        return {
            "status": "success",
            "filename": current_dataset_name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "missing_values": missing_values,
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_summary": numeric_summary,
            "preview": records(df.head(10)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.get("/charts")
def chart_data(bins: int = 10, max_categories: int = 10):
    try:
        df = load_current_df()
        bins = max(3, min(bins, 30))
        numeric_df = df.select_dtypes(include=np.number)

        histograms = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if s.empty:
                continue
            counts, edges = np.histogram(s, bins=bins)
            histograms[str(col)] = [
                {
                    "range": f"{edges[i]:.2f} - {edges[i + 1]:.2f}",
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ]

        categories = {}
        for col in df.select_dtypes(exclude=np.number).columns:
            vc = df[col].value_counts().head(max_categories)
            categories[str(col)] = [
                {"name": str(k), "count": int(v)} for k, v in vc.items()
            ]

        missing = [
            {"column": str(c), "missing": int(v)}
            for c, v in df.isnull().sum().items()
            if v > 0
        ]

        correlation = []
        corr_cols = numeric_df.columns[:12]
        if len(corr_cols) >= 2:
            corr = numeric_df[corr_cols].corr().round(3)
            for x in corr.columns:
                for y in corr.columns:
                    value = corr.loc[x, y]
                    correlation.append(
                        {
                            "x": str(x),
                            "y": str(y),
                            "value": None if pd.isna(value) else float(value),
                        }
                    )

        return {
            "status": "success",
            "histograms": histograms,
            "categories": categories,
            "missing": missing,
            "correlation": correlation,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart data failed: {e}")


# =========================================================
# TARGET COLUMN OPTIONS
# =========================================================

@app.get("/target-columns")
def get_target_columns():
    try:
        df = load_current_df()
        return {
            "status": "success",
            "columns": [
                {
                    "name": str(col),
                    "unique_values": int(df[col].nunique()),
                    "data_type": str(df[col].dtype),
                }
                for col in df.columns
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# TRAINING
# =========================================================

def create_preprocessor(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


@app.post("/train")
def train_models(request: TrainRequest):
    global trained_model, trained_model_name, target_column, training_summary

    try:
        df = load_current_df()
        target = request.target_column.strip()

        if target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target}' not found. Available: {df.columns.tolist()}",
            )

        df = df.dropna(subset=[target])
        y = df[target]
        X = df.drop(columns=[target])

        if y.nunique() < 2:
            raise HTTPException(
                status_code=400,
                detail="Target column must contain at least 2 classes.",
            )

        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"'{target}' looks continuous ({y.nunique()} unique values). "
                    "This trainer handles classification. Pick a column with a few classes."
                ),
            )

        if X.shape[1] == 0:
            raise HTTPException(
                status_code=400,
                detail="Dataset needs at least one feature column besides the target.",
            )

        # Stratify only when every class has at least 2 rows
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        preprocessor = create_preprocessor(X)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight="balanced"
            ),
            "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        }
        if request.model_name and request.model_name not in models:
            raise HTTPException(status_code=400, detail=f"Unknown model '{request.model_name}'. Choose one of: {list(models)}")

        results = []
        fitted_models = {}
        best_f1 = -1.0
        best_pipeline = None
        best_model_name = None

        for name, model in models.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", model),
                ]
            )
            pipeline.fit(X_train, y_train)
            fitted_models[name] = pipeline
            preds = pipeline.predict(X_test)

            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
            results.append(
                {
                    "model": name,
                    "accuracy": round(float(accuracy_score(y_test, preds)), 4),
                    "precision": round(
                        float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4
                    ),
                    "recall": round(
                        float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4
                    ),
                    "f1_score": round(float(f1), 4),
                }
            )

            if f1 > best_f1:
                best_f1 = f1
                best_pipeline = pipeline
                best_model_name = name

        selected_model_name = request.model_name or best_model_name
        trained_model = fitted_models[selected_model_name]
        trained_model_name = selected_model_name
        target_column = target
        selected_metrics = next(r for r in results if r["model"] == selected_model_name)
        training_summary = {
            "selected_model": selected_model_name,
            "target_column": target,
            "best_model": best_model_name,
            "best_f1_score": round(float(best_f1), 4),
            "selected_metrics": selected_metrics,
            "results": results,
            "trained_dataset": current_dataset_name,
        }

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(
                {
                    "model": trained_model,
                    "model_name": trained_model_name,
                    "target_column": target_column,
                    "training_summary": training_summary,
                },
                f,
            )
        persistence.save_artifact("trained_model", MODEL_FILE.read_bytes(), {
            "model_name": trained_model_name,
            "target_column": target_column,
            "training_summary": training_summary,
        })

        return {
            "status": "success",
            "message": "Models trained successfully.",
            "target_column": target,
            "best_model": best_model_name,
            "selected_model": selected_model_name,
            "best_f1_score": round(float(best_f1), 4),
            "selected_metrics": selected_metrics,
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")


# =========================================================
# FEATURES (builds the Prediction form)
# =========================================================

@app.get("/model-status")
def get_model_status():
    if trained_model is None:
        load_saved_model()
    if trained_model is None:
        return {"status": "success", "available": False, "model": None, "target_column": None, "training": None}
    return {
        "status": "success",
        "available": True,
        "model": trained_model_name,
        "target_column": target_column,
        "training": training_summary or {"selected_model": trained_model_name, "target_column": target_column, "results": []},
    }

@app.get("/features")
def get_features():
    if trained_model is None:
        load_saved_model()

    if trained_model is None:
        raise HTTPException(
            status_code=404,
            detail="No trained model available. Train a model first.",
        )

    try:
        df = load_current_df()
        features = []

        for col in df.columns:
            if col == target_column:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].dropna()
                features.append(
                    {
                        "name": str(col),
                        "type": "number",
                        "min": float(s.min()) if not s.empty else None,
                        "max": float(s.max()) if not s.empty else None,
                        "median": float(s.median()) if not s.empty else None,
                    }
                )
            else:
                features.append(
                    {
                        "name": str(col),
                        "type": "category",
                        "options": [str(v) for v in df[col].dropna().unique()[:50]],
                    }
                )

        return {
            "status": "success",
            "target_column": target_column,
            "model": trained_model_name,
            "features": features,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(request: PredictionRequest):
    global trained_model

    if trained_model is None:
        load_saved_model()

    if trained_model is None:
        raise HTTPException(
            status_code=404,
            detail="No trained model available. Train a model first.",
        )

    try:
        preprocessor = trained_model.named_steps["preprocessor"]
        expected = list(preprocessor.feature_names_in_)

        numeric_cols = []
        for name, _, cols in preprocessor.transformers_:
            if name == "numeric":
                numeric_cols = list(cols)

        row = {}
        for col in expected:
            value = request.data.get(col)
            row[col] = np.nan if value is None or value == "" else value

        input_df = pd.DataFrame([row])
        for col in numeric_cols:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

        prediction = trained_model.predict(input_df)[0]
        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        response = {
            "status": "success",
            "prediction": prediction,
            "model": trained_model_name,
            "target_column": target_column,
        }

        if hasattr(trained_model, "predict_proba"):
            try:
                probs = trained_model.predict_proba(input_df)[0]
                response["probabilities"] = [
                    {"class": str(c), "probability": round(float(p), 4)}
                    for c, p in zip(trained_model.classes_, probs)
                ]
            except Exception:
                pass

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# =========================================================
# GROQ DATASET ASSISTANT
# =========================================================

def reply(text, **extra):
    return {"status": "success", "reply": text, **extra}


@app.get("/chat/sessions")
def get_chat_sessions():
    return {"sessions": persistence.list_sessions()}


@app.post("/chat/sessions")
def create_chat_session():
    session = persistence.create_session(uuid4().hex)
    return {"session": session}


@app.patch("/chat/sessions/{session_id}")
def rename_chat_session(session_id: str, request: RenameSessionRequest):
    if not persistence.rename_session(session_id, request.title):
        raise HTTPException(status_code=404, detail="Chat session not found or title is empty.")
    return {"status": "success", "title": " ".join(request.title.split())[:64]}


@app.get("/chat/sessions/{session_id}")
def get_chat_session(session_id: str):
    messages = persistence.session_messages(session_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"messages": messages}


@app.delete("/chat/sessions/{session_id}")
def remove_chat_session(session_id: str):
    if not persistence.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"status": "success"}


@app.post("/chat")
async def chat(request: ChatRequest):
    global pending_chat_actions
    session_id = request.session_id or uuid4().hex
    persistence.create_session(session_id)
    user_message = request.message.strip()
    if user_message:
        persistence.append_message(session_id, "user", user_message)

    def finish(response):
        metadata = {key: response[key] for key in ("chart", "pending_action", "confirmation_id", "training_result") if key in response}
        persistence.append_message(session_id, "assistant", response.get("reply", ""), metadata)
        response["session_id"] = session_id
        return response

    if current_dataset_path is None:
        return finish(reply("Please upload a CSV dataset first. Then I can analyze and chart it."))
    if not user_message and not request.confirm_action:
        return finish(reply("Please enter a question."))
    try:
        if request.confirm_action:
            pending = pending_chat_actions.pop(request.confirm_action, None)
            if not pending:
                raise HTTPException(status_code=404, detail="That confirmation expired or was already used. Ask the chatbot again.")
            if pending.get("operation") == "train_models":
                result = train_models(TrainRequest(target_column=pending["arguments"].get("target_column", ""), model_name=pending["arguments"].get("model_name")))
                return finish(reply(f"Training finished. Active model: {result['selected_model']}; best weighted-F1 model: {result['best_model']} ({result['best_f1_score']}). The Models and Prediction pages now use this saved model.", training_result=result, model_changed=True))
            df = load_current_df()
            updated, message = apply_dataset_action(df, pending["arguments"])
            save_dataset_version(updated)
            return finish(reply(f"{message} The updated dataset is now loaded. Analysis has been refreshed, and you can say ‘undo that’ to restore the previous version.", dataset_changed=True, filename=current_dataset_name))

        df = load_current_df()
        result = await ask_groq(user_message, request.history, df, {"trained": trained_model is not None, "name": trained_model_name, "target": target_column, "training": training_summary, "_model": trained_model})
        if result.get("error"):
            return finish(reply(result["error"], provider="groq"))
        if result.get("pending_action"):
            token = uuid4().hex
            pending_chat_actions[token] = result["pending_action"]
            result["confirmation_id"] = token
        if result.get("undo"):
            if not undo_dataset_version():
                return finish(reply("There is no earlier dataset version to restore."))
            result["reply"] = "Restored the previous dataset version. Analysis has been refreshed."
            result["dataset_changed"] = True
            result["filename"] = current_dataset_name
        result["status"] = "success"
        return finish(result)
    except HTTPException:
        raise
    except httpx.TimeoutException:
        return finish(reply("Groq took too long to respond. Please try again.", provider="groq"))
    except httpx.HTTPError as e:
        return finish(reply(f"Could not reach Groq: {e}", provider="groq"))
    except ValueError as e:
        return finish(reply(str(e), provider="local"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")


@app.get("/chat-test")
def chat_test():
    if current_dataset_path is None:
        return reply("Backend is working. Upload a dataset first.")
    return reply(f"Chatbot backend is working and dataset '{current_dataset_name}' is loaded.")
