from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from src.services.finbert_service import (
    get_current_model_info,
    predict_batch,
    predict_quantized_sentiment,
    predict_sentiment,
)

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


def _raise_prediction_error(exc: Exception) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    try:
        return predict_sentiment(request.text)
    except Exception as exc:
        _raise_prediction_error(exc)


@router.post("/predict-quantized")
def predict_quantized(request: SentimentRequest):
    try:
        return predict_quantized_sentiment(request.text)
    except Exception as exc:
        _raise_prediction_error(exc)


@router.post("/predict-batch")
def predict_batch_route(
    sentences: list[str] = Body(..., description="A JSON list of sentences."),
    use_quantized: bool = Query(False, description="Use the quantized model for inference."),
):
    try:
        return predict_batch(sentences, use_quantized=use_quantized)
    except Exception as exc:
        _raise_prediction_error(exc)
