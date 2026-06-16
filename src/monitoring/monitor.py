from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
    score_distribution: list[float] = field(default_factory=list)
    expected_scores: list[float] = field(default_factory=list)
    drift_score: float = 0.0
    retrain_needed: bool = False


_state = _MonitorState()


def record_request(latency_seconds: float, score: float | None = None) -> None:
    """Call this on every prediction request to log latency + output distribution."""
    with _lock:
        _state.request_times.append(latency_seconds)
        _state.request_count += 1
        if score is not None:
            _state.score_distribution.append(float(score))


def record_feedback_accuracy(is_correct: bool) -> None:
    """Call this whenever feedback with a true_label arrives, to track running accuracy."""
    with _lock:
        _state.labeled_count += 1
        if is_correct:
            _state.correct_count += 1


def _seed_expected_distribution() -> None:
    """Build the baseline ('expected') score distribution from the test set,
    used as the reference distribution for PSI drift detection."""
    if _state.expected_scores:
        return
    if not DEFAULT_TEST_FILE.exists():
        return
    try:
        import pandas as pd

        df = pd.read_csv(DEFAULT_TEST_FILE)
        # use text length as a cheap, dependency-free proxy signal for the
        # baseline input distribution (works even before any predictions exist)
        lengths = df["text"].astype(str).str.len().tolist()
        with _lock:
            _state.expected_scores = [float(x) for x in lengths]
    except Exception:
        pass


_seed_expected_distribution()


# ---------------------------------------------------------------------------
# 2. PSI (Population Stability Index) drift detection
# ---------------------------------------------------------------------------
def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between an expected (baseline) and an
    actual (current) numeric distribution. <0.1 stable, 0.1-0.2 moderate,
    >0.2 significant drift."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    breakpoints = np.histogram_bin_edges(expected, bins=bins)

    expected_hist, _ = np.histogram(expected, bins=breakpoints)
    actual_hist, _ = np.histogram(actual, bins=breakpoints)

    expected_dist = expected_hist / len(expected)
    actual_dist = actual_hist / len(actual)

    psi_value = np.sum(
        (expected_dist - actual_dist)
        * np.log((expected_dist + 1e-6) / (actual_dist + 1e-6))
    )
    return float(psi_value)


def compute_drift(bins: int = 10) -> dict[str, Any]:
    """Compute PSI between the expected baseline distribution and the
    distribution of inputs/scores collected so far, and set retrain flag."""
    with _lock:
        expected = list(_state.expected_scores)
        actual = list(_state.score_distribution)

    if not expected or not actual:
        return {
            "psi": 0.0,
            "status": "insufficient_data",
            "retrain_needed": False,
            "expected_samples": len(expected),
            "actual_samples": len(actual),
        }

    psi_value = psi(expected, actual, bins=bins)

    if psi_value < PSI_STABLE_THRESHOLD:
        status = "stable"
    elif psi_value < PSI_DRIFT_THRESHOLD:
        status = "moderate_drift"
    else:
        status = "drift_detected"

    retrain_needed = psi_value > PSI_DRIFT_THRESHOLD

    with _lock:
        _state.drift_score = psi_value
        _state.retrain_needed = _state.retrain_needed or retrain_needed

    return {
        "psi": psi_value,
        "status": status,
        "retrain_needed": retrain_needed,
        "expected_samples": len(expected),
        "actual_samples": len(actual),
    }


def feed_actual_distribution(values: list[float]) -> None:
    """Inject a batch of new numeric values (e.g. text lengths from a new
    dataset) into the 'actual' distribution, used to test drift detection
    with synthetic/foreign data (see test_drift.py)."""
    with _lock:
        _state.score_distribution.extend(float(v) for v in values)


# ---------------------------------------------------------------------------
# 3. Feedback collection (for drift / future retraining)
# ---------------------------------------------------------------------------
def save_feedback(text: str, predicted_label: str, true_label: str) -> dict[str, Any]:
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    is_correct = predicted_label.strip().lower() == true_label.strip().lower()
    record_feedback_accuracy(is_correct)

    entry = {
        "timestamp": time.time(),
        "text": text,
        "predicted_label": predicted_label,
        "true_label": true_label,
        "is_correct": is_correct,
    }

    with _lock:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # feed text length into the actual-distribution used for drift detection
    feed_actual_distribution([len(text)])

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


# ---------------------------------------------------------------------------
# 4. Metrics summary (for GET /metrics)
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
        _state.score_distribution.clear()
        _state.drift_score = 0.0
        _state.retrain_needed = False
    _seed_expected_distribution()
