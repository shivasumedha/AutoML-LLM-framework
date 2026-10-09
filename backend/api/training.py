from fastapi import APIRouter, HTTPException

from backend.schemas import TrainRequest
from backend.services.data_service import load_current_df
from backend.services.ml_service import TrainingError, train_models

router = APIRouter(tags=["Training"])


@router.post("/train")
def train(request: TrainRequest):
    try:
        df = load_current_df()
        return train_models(df, request.target_column)
    except HTTPException:
        raise
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")
