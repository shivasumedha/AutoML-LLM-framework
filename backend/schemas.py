from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    confirm_action: str | None = None


class TrainRequest(BaseModel):
    target_column: str


class PredictionRequest(BaseModel):
    data: dict
