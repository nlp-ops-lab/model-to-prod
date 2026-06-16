from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from src.services.feedback_service import record_feedback
from src.services.finbert_service import (
    are_models_ready,
    get_current_model_info,
    predict_batch,
    predict_quantized_sentiment,
    predict_sentiment,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class SentimentRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    true_label: str


def _raise_prediction_error(exc: Exception) -> None:
    logger.exception("Inference request failed: %s", exc)
    status_code = 503 if isinstance(exc, (FileNotFoundError, RuntimeError)) else 500
    raise HTTPException(
        status_code=status_code,
        detail={
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        },
    ) from exc


def _raise_feedback_error(exc: Exception) -> None:
    logger.exception("Feedback request failed: %s", exc)
    raise HTTPException(
        status_code=500,
        detail={
            "status": "error",
            "error": type(exc).__name__,
            "message": str(exc),
        },
    ) from exc


@router.get("/health")
def health():
    try:
        return {
            "status": "ok",
            "model": get_current_model_info(),
        }
    except Exception as exc:
        _raise_prediction_error(exc)


@router.get("/ready")
def ready():
    return {"status": "ready" if are_models_ready() else "not_ready"}


@router.get("/model-info")
def model_info():
    try:
        return get_current_model_info()
    except Exception as exc:
        _raise_prediction_error(exc)


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


@router.post("/feedback")
def feedback(request: FeedbackRequest):
    try:
        record_feedback(
            text=request.text,
            predicted_label=request.predicted_label,
            true_label=request.true_label,
        )
        return {
            "status": "success",
            "message": "feedback recorded",
        }
    except Exception as exc:
        _raise_feedback_error(exc)
