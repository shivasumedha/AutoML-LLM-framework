from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.core.config import UPLOAD_DIR
from backend.core.state import state
from backend.services.data_service import load_current_df, read_csv_safe, records

router = APIRouter(tags=["Dataset"])


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename received.")

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_path = UPLOAD_DIR / Path(file.filename).name
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        try:
            df = read_csv_safe(file_path)
        except Exception as csv_error:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Invalid CSV file: {csv_error}")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV contains no rows.")

        state.set_dataset(file_path)

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


@router.get("/dataset")
def get_dataset():
    try:
        df = load_current_df()
        return {
            "status": "success",
            "filename": state.dataset_name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "preview": records(df.head(10)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/preview")
def preview_dataset(rows: int = 10):
    try:
        df = load_current_df()
        rows = max(1, min(rows, 100))
        return {"status": "success", "data": records(df.head(rows))}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/target-columns")
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
