from __future__ import annotations

import time

from fastapi import APIRouter, Body, HTTPException, Query
from pydantic import BaseModel

from src.monitoring.monitor import (
    get_metrics_summary,
    log_prediction,
    record_request,
    save_feedback,
)
from src.services.finbert_service import (
    get_current_model_info,
    get_readiness_status,
    predict_batch,
    predict_quantized_sentiment,
    predict_sentiment,
)

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    text: str
    predicted_label: str
    true_label: str


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


@router.get("/ready")
def ready():
    return get_readiness_status()


@router.get("/model-info")
def model_info():
    return get_current_model_info()


@router.post("/predict")
def predict(request: SentimentRequest):
    start = time.time()
    try:
        result = predict_sentiment(request.text)
    except Exception as exc:
        _raise_prediction_error(exc)
        return  # unreachable, keeps type-checkers happy
    end = time.time()

    latency = end - start
    record_request(latency, score=result.get("score"))
    log_prediction(request.text, result.get("label", ""), result.get("score", 0.0), latency)

    return result


@router.post("/predict-quantized")
def predict_quantized(request: SentimentRequest):
    start = time.time()
    try:
        result = predict_quantized_sentiment(request.text)
    except Exception as exc:
        _raise_prediction_error(exc)
        return
    end = time.time()

    latency = end - start
    record_request(latency, score=result.get("score"))
    log_prediction(request.text, result.get("label", ""), result.get("score", 0.0), latency)

    return result


@router.post("/predict-batch")
def predict_batch_route(
    sentences: list[str] = Body(..., description="A JSON list of sentences."),
    use_quantized: bool = Query(False, description="Use the quantized model for inference."),
):
    start = time.time()
    try:
        results = predict_batch(sentences, use_quantized=use_quantized)
    except Exception as exc:
        _raise_prediction_error(exc)
        return
    end = time.time()

    # split elapsed time evenly across the batch so request_count/latency
    # stats stay meaningful when batch endpoints are mixed with single ones
    per_item_latency = (end - start) / len(results) if results else (end - start)
    for item in results:
        record_request(per_item_latency, score=item.get("score"))
        log_prediction(item.get("text", ""), item.get("label", ""), item.get("score", 0.0), per_item_latency)

    return results


@router.post("/feedback")
def feedback(request: FeedbackRequest):
    """Collect ground-truth feedback for a prediction. Used for accuracy
    tracking, drift detection, and future retraining."""
    entry = save_feedback(
        text=request.text,
        predicted_label=request.predicted_label,
        true_label=request.true_label,
    )
    return {"status": "saved", "is_correct": entry["is_correct"]}


@router.get("/metrics")
def metrics():
    """Simple monitoring endpoint: avg/max latency, request count, accuracy
    from feedback, and drift score (PSI)."""
    return get_metrics_summary()
