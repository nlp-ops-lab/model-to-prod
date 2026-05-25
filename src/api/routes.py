from fastapi import APIRouter
from pydantic import BaseModel

from src.pipelines.sentiment_flow import sentiment_pipeline


router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict")
def predict(request: SentimentRequest):

    result = sentiment_pipeline(request.text)

    return result