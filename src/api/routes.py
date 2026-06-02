from fastapi import APIRouter
from pydantic import BaseModel

from src.services.finbert_service import (
    predict_sentiment,
    get_current_model_info,
)

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


@router.get("/health")
def health():
    return {
        "status": "ok",
        "model": get_current_model_info(),
    }


@router.get("/model-info")
def model_info():
    return get_current_model_info()


@router.post("/predict")
def predict(request: SentimentRequest):
    return predict_sentiment(request.text)