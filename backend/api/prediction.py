from fastapi import APIRouter, HTTPException

from backend.core.state import state
from backend.schemas import PredictionRequest
from backend.services.data_service import load_current_df
from backend.services.ml_service import build_feature_schema, predict_one

router = APIRouter(tags=["Prediction"])

NO_MODEL = "No trained model available. Train a model first."


@router.get("/features")
def get_features():
    if not state.ensure_model():
        raise HTTPException(status_code=404, detail=NO_MODEL)

    try:
        df = load_current_df()
        return {
            "status": "success",
            "target_column": state.target_column,
            "model": state.model_name,
            "features": build_feature_schema(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def predict(request: PredictionRequest):
    if not state.ensure_model():
        raise HTTPException(status_code=404, detail=NO_MODEL)

    try:
        return predict_one(request.data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
