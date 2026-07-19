from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.pipelines.drift_detector import (
    compute_drift,
    record_feedback_event,
    record_prediction_event,
    reset_drift_state,
)
from src.services.feedback_service import record_feedback as persist_feedback

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MONITORING_DIR = ARTIFACTS_DIR / "monitoring"
FEEDBACK_FILE = MONITORING_DIR / "feedback.jsonl"
PREDICTIONS_LOG_FILE = MONITORING_DIR / "predictions_log.jsonl"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

# baseline ("expected") distribution of model confidence scores, fitted
# once from the original test set the first time the module is imported.
DEFAULT_TEST_FILE = BASE_DIR / "data" / "test" / "test.csv"

PSI_STABLE_THRESHOLD = 0.1
PSI_DRIFT_THRESHOLD = 0.2

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 1. Latency / request monitoring
# ---------------------------------------------------------------------------
@dataclass
class _MonitorState:
    request_times: list[float] = field(default_factory=list)
    request_count: int = 0
    correct_count: int = 0
    labeled_count: int = 0


_state = _MonitorState()


def record_request(latency_seconds: float, score: float | None = None) -> None:
    """Call this on every prediction request to log latency + output distribution."""
    with _lock:
        _state.request_times.append(latency_seconds)
        _state.request_count += 1


def record_feedback_accuracy(is_correct: bool) -> None:
    """Call this whenever feedback with a true_label arrives, to track running accuracy."""
    with _lock:
        _state.labeled_count += 1
        if is_correct:
            _state.correct_count += 1


# ---------------------------------------------------------------------------
# 2. Feedback collection (for drift / future retraining)
# ---------------------------------------------------------------------------
def save_feedback(text: str, predicted_label: str, true_label: str) -> dict[str, Any]:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    is_correct = predicted_label.strip().lower() == true_label.strip().lower()
    record_feedback_accuracy(is_correct)

    feedback_entry = persist_feedback(
        text=text,
        predicted_label=predicted_label,
        true_label=true_label,
    )

    entry = {
        **feedback_entry,
        "timestamp": time.time(),
        "is_correct": is_correct,
    }

    with _lock:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    record_feedback_event(
        text=text,
        predicted_label=predicted_label,
        true_label=true_label,
    )

    return entry


def log_prediction(text: str, label: str, score: float, latency: float) -> None:
    """Optional persistent log of every prediction (input + output), useful
    for offline drift analysis beyond the in-memory PSI tracker."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "text": text,
        "label": label,
        "score": score,
        "latency": latency,
    }
    with _lock:
        with open(PREDICTIONS_LOG_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    record_prediction_event(text=text, label=label, score=score)


# ---------------------------------------------------------------------------
# 3. Metrics summary (for GET /metrics)
# ---------------------------------------------------------------------------
def get_metrics_summary() -> dict[str, Any]:
    with _lock:
        request_times = list(_state.request_times)
        request_count = _state.request_count
        labeled_count = _state.labeled_count
        correct_count = _state.correct_count

    avg_latency = float(np.mean(request_times)) if request_times else 0.0
    max_latency = float(np.max(request_times)) if request_times else 0.0
    min_latency = float(np.min(request_times)) if request_times else 0.0

    drift_info = compute_drift()

    accuracy = (correct_count / labeled_count) if labeled_count else None

    return {
        "avg_latency": round(avg_latency, 4),
        "max_latency": round(max_latency, 4),
        "min_latency": round(min_latency, 4),
        "request_count": request_count,
        "labeled_feedback_count": labeled_count,
        "accuracy_from_feedback": round(accuracy, 4) if accuracy is not None else None,
        "drift_score": round(drift_info["psi"], 4),
        "drift_status": drift_info["status"],
        "retrain_needed": drift_info["retrain_needed"],
    }


def reset_state() -> None:
    """Used by tests to reset in-memory monitoring state between runs."""
    with _lock:
        _state.request_times.clear()
        _state.request_count = 0
        _state.correct_count = 0
        _state.labeled_count = 0
    reset_drift_state()
