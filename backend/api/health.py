from fastapi import APIRouter

from backend.core.config import API_VERSION

router = APIRouter(tags=["Health"])


@router.get("/")
def root():
    return {
        "status": "success",
        "message": "AutoML FastAPI backend is running",
        "version": API_VERSION,
    }


@router.get("/health")
def health():
    return {"status": "healthy", "backend": "FastAPI"}
