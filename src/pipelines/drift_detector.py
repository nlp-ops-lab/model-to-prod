from __future__ import annotations

import csv
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pipelines.incremental_update_pipeline import incremental_update_pipeline
from src.services.feedback_service import load_feedback_data


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_TEST_FILE = BASE_DIR / "data" / "test" / "test.csv"
PRODUCTION_MODEL_FILE = MODELS_DIR / "production_model.txt"
QUANTIZED_ONNX_FILE = "model_quantized.onnx"
RETRAINING_DIR = ARTIFACTS_DIR / "retraining"
RETRAINING_DATA_FILE = RETRAINING_DIR / "feedback_incremental.csv"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

PSI_STABLE_THRESHOLD = 0.1
PSI_DRIFT_THRESHOLD = 0.2
ACCURACY_THRESHOLD = 0.90
DOMAIN_SHIFT_THRESHOLD = 0.35
MIN_PREDICTION_SAMPLES = 25
MIN_FEEDBACK_SAMPLES = 20
RECENT_FEEDBACK_WINDOW = 50
CONFIDENCE_BASELINE_BOOTSTRAP = 50
MAX_BUFFER_SIZE = 2000
RETRAINING_COOLDOWN_SECONDS = 900
AUTO_RETRAIN_ENABLED_ENV = "FINBERT_AUTO_RETRAIN"

_TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")
_state_lock = threading.RLock()


@dataclass
class DriftState:
    baseline_text_lengths: list[float] = field(default_factory=list)
    baseline_label_ids: list[int] = field(default_factory=list)
    baseline_vocabulary: set[str] = field(default_factory=set)
    baseline_confidences: list[float] = field(default_factory=list)
    incoming_text_lengths: deque[float] = field(default_factory=deque)
    incoming_confidences: deque[float] = field(default_factory=deque)
    incoming_predicted_label_ids: deque[int] = field(default_factory=deque)
    incoming_true_label_ids: deque[int] = field(default_factory=deque)
    incoming_correctness: deque[bool] = field(default_factory=deque)
    incoming_domain_novelty: deque[float] = field(default_factory=deque)
    retrain_needed: bool = False
    retraining_in_progress: bool = False
    last_retrain_trigger_ts: float = 0.0
    last_retrain_feedback_count: int = 0
    last_retrain_version: str | None = None
    last_retrain_error: str | None = None
    last_report: dict[str, Any] = field(default_factory=dict)


_state = DriftState()


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _auto_retrain_enabled() -> bool:
    env_value = os.getenv(AUTO_RETRAIN_ENABLED_ENV)
    if env_value is None:
        return True
    return _is_truthy(env_value)


def _append_limited(buffer: deque[Any], value: Any) -> None:
    buffer.append(value)
    if len(buffer) > MAX_BUFFER_SIZE:
        buffer.popleft()


def _normalize_label(label: str) -> str | None:
    normalized = str(label).strip().lower()
    if normalized in LABEL2ID:
        return normalized
    return None


def _label_to_id(label: str) -> int | None:
    normalized = _normalize_label(label)
    if normalized is None:
        return None
    return LABEL2ID[normalized]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(str(text).lower())


def _to_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _resolve_model_path(saved_path: str) -> Path:
    path = Path(saved_path)
    return path if path.is_absolute() else BASE_DIR / path


def _discover_standard_local_model_path() -> Path:
    candidates = [
        path
        for path in MODELS_DIR.iterdir()
        if path.is_dir()
        and not (path / QUANTIZED_ONNX_FILE).exists()
        and (path / "config.json").exists()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    if not candidates:
        raise FileNotFoundError("No local standard model directory is available.")
    return candidates[0]


def _get_current_base_model_path() -> Path:
    if PRODUCTION_MODEL_FILE.exists():
        configured_path = PRODUCTION_MODEL_FILE.read_text(encoding="utf-8").strip()
        if configured_path:
            model_path = _resolve_model_path(configured_path)
            if model_path.exists() and model_path.is_dir():
                return model_path
    return _discover_standard_local_model_path()


def _seed_baseline() -> None:
    with _state_lock:
        if _state.baseline_text_lengths and _state.baseline_label_ids and _state.baseline_vocabulary:
            return

    if not DEFAULT_TEST_FILE.exists():
        return

    try:
        df = pd.read_csv(DEFAULT_TEST_FILE)
    except Exception:
        logger.exception("Failed to load baseline test file for drift detection.")
        return

    if "text" not in df.columns or "label" not in df.columns:
        logger.warning("Baseline test file is missing required columns: %s", DEFAULT_TEST_FILE)
        return

    texts = df["text"].astype(str).tolist()
    labels = [
        label_id
        for label_id in (
            _label_to_id(label)
            for label in df["label"].astype(str).tolist()
        )
        if label_id is not None
    ]
    vocabulary = {
        token
        for text in texts
        for token in _tokenize(text)
    }
    lengths = [float(len(text)) for text in texts]

    with _state_lock:
        _state.baseline_text_lengths = lengths
        _state.baseline_label_ids = labels
        _state.baseline_vocabulary = vocabulary


_seed_baseline()


def psi(expected: list[float] | np.ndarray, actual: list[float] | np.ndarray, bins: int = 10) -> float:
    expected_arr = np.asarray(expected, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)

    if len(expected_arr) == 0 or len(actual_arr) == 0:
        return 0.0

    breakpoints = np.histogram_bin_edges(expected_arr, bins=bins)
    expected_hist, _ = np.histogram(expected_arr, bins=breakpoints)
    actual_hist, _ = np.histogram(actual_arr, bins=breakpoints)

    expected_dist = expected_hist / len(expected_arr)
    actual_dist = actual_hist / len(actual_arr)

    psi_value = np.sum(
        (expected_dist - actual_dist)
        * np.log((expected_dist + 1e-6) / (actual_dist + 1e-6))
    )
    return float(psi_value)


def _psi_status(psi_value: float, expected_samples: int, actual_samples: int) -> tuple[str, bool]:
    enough_data = expected_samples > 0 and actual_samples > 0
    if not enough_data:
        return "insufficient_data", False
    if psi_value < PSI_STABLE_THRESHOLD:
        return "stable", False
    if psi_value < PSI_DRIFT_THRESHOLD:
        return "moderate_drift", False
    return "drift_detected", True


def _domain_status(avg_novelty_ratio: float, sample_count: int) -> tuple[str, bool]:
    if sample_count < MIN_PREDICTION_SAMPLES:
        return "insufficient_data", False
    if avg_novelty_ratio < DOMAIN_SHIFT_THRESHOLD * 0.5:
        return "stable", False
    if avg_novelty_ratio < DOMAIN_SHIFT_THRESHOLD:
        return "moderate_drift", False
    return "drift_detected", True


def _copy_state() -> dict[str, Any]:
    with _state_lock:
        return {
            "baseline_text_lengths": list(_state.baseline_text_lengths),
            "baseline_label_ids": list(_state.baseline_label_ids),
            "baseline_confidences": list(_state.baseline_confidences),
            "baseline_vocabulary_size": len(_state.baseline_vocabulary),
            "incoming_text_lengths": list(_state.incoming_text_lengths),
            "incoming_confidences": list(_state.incoming_confidences),
            "incoming_true_label_ids": list(_state.incoming_true_label_ids),
            "incoming_correctness": list(_state.incoming_correctness),
            "incoming_domain_novelty": list(_state.incoming_domain_novelty),
            "retraining_in_progress": _state.retraining_in_progress,
            "last_retrain_trigger_ts": _state.last_retrain_trigger_ts,
            "last_retrain_feedback_count": _state.last_retrain_feedback_count,
            "last_retrain_version": _state.last_retrain_version,
            "last_retrain_error": _state.last_retrain_error,
            "retrain_needed": _state.retrain_needed,
        }


def _build_report(snapshot: dict[str, Any], bins: int = 10) -> dict[str, Any]:
    covariate_psi = psi(snapshot["baseline_text_lengths"], snapshot["incoming_text_lengths"], bins=bins)
    covariate_status, covariate_retrain = _psi_status(
        covariate_psi,
        len(snapshot["baseline_text_lengths"]),
        len(snapshot["incoming_text_lengths"]),
    )
    covariate_shift = {
        "psi": covariate_psi,
        "status": covariate_status,
        "expected_samples": len(snapshot["baseline_text_lengths"]),
        "actual_samples": len(snapshot["incoming_text_lengths"]),
        "retrain_needed": covariate_retrain,
    }

    confidence_psi = psi(snapshot["baseline_confidences"], snapshot["incoming_confidences"], bins=bins)
    confidence_status, confidence_retrain = _psi_status(
        confidence_psi,
        len(snapshot["baseline_confidences"]),
        len(snapshot["incoming_confidences"]),
    )
    confidence_shift = {
        "psi": confidence_psi,
        "status": confidence_status,
        "baseline_samples": len(snapshot["baseline_confidences"]),
        "actual_samples": len(snapshot["incoming_confidences"]),
        "retrain_needed": confidence_retrain,
    }

    label_psi = psi(snapshot["baseline_label_ids"], snapshot["incoming_true_label_ids"], bins=len(LABEL2ID))
    label_status, label_retrain = _psi_status(
        label_psi,
        len(snapshot["baseline_label_ids"]),
        len(snapshot["incoming_true_label_ids"]),
    )
    label_shift = {
        "psi": label_psi,
        "status": label_status,
        "expected_samples": len(snapshot["baseline_label_ids"]),
        "actual_samples": len(snapshot["incoming_true_label_ids"]),
        "retrain_needed": label_retrain,
    }

    avg_novelty_ratio = float(np.mean(snapshot["incoming_domain_novelty"])) if snapshot["incoming_domain_novelty"] else 0.0
    domain_status, domain_retrain = _domain_status(
        avg_novelty_ratio,
        len(snapshot["incoming_domain_novelty"]),
    )
    domain_shift = {
        "avg_novelty_ratio": avg_novelty_ratio,
        "status": domain_status,
        "samples": len(snapshot["incoming_domain_novelty"]),
        "baseline_vocabulary_size": snapshot["baseline_vocabulary_size"],
        "retrain_needed": domain_retrain,
    }

    feedback_samples = len(snapshot["incoming_correctness"])
    accuracy = (
        float(np.mean(snapshot["incoming_correctness"]))
        if snapshot["incoming_correctness"]
        else None
    )
    recent_correctness = snapshot["incoming_correctness"][-RECENT_FEEDBACK_WINDOW:]
    recent_accuracy = (
        float(np.mean(recent_correctness))
        if recent_correctness
        else None
    )
    if feedback_samples < MIN_FEEDBACK_SAMPLES:
        concept_status = "insufficient_data"
        concept_retrain = False
    else:
        effective_accuracy = recent_accuracy if recent_accuracy is not None else accuracy
        if effective_accuracy is not None and effective_accuracy < ACCURACY_THRESHOLD:
            concept_status = "accuracy_degraded"
            concept_retrain = True
        elif effective_accuracy is not None and effective_accuracy < ACCURACY_THRESHOLD + 0.03:
            concept_status = "moderate_drift"
            concept_retrain = False
        else:
            concept_status = "stable"
            concept_retrain = False
    concept_drift = {
        "accuracy": accuracy,
        "recent_accuracy": recent_accuracy,
        "feedback_samples": feedback_samples,
        "status": concept_status,
        "retrain_needed": concept_retrain,
    }

    overall_psi = max(covariate_psi, confidence_psi, label_psi)
    statuses = [
        covariate_shift["status"],
        confidence_shift["status"],
        label_shift["status"],
        concept_drift["status"],
        domain_shift["status"],
    ]
    if any(status in {"drift_detected", "accuracy_degraded"} for status in statuses):
        overall_status = "drift_detected"
    elif any(status == "moderate_drift" for status in statuses):
        overall_status = "moderate_drift"
    elif all(status == "insufficient_data" for status in statuses):
        overall_status = "insufficient_data"
    else:
        overall_status = "stable"

    retrain_needed = any(
        signal["retrain_needed"]
        for signal in (
            covariate_shift,
            confidence_shift,
            label_shift,
            concept_drift,
            domain_shift,
        )
    )

    return {
        "psi": overall_psi,
        "status": overall_status,
        "retrain_needed": retrain_needed,
        "covariate_shift": covariate_shift,
        "confidence_shift": confidence_shift,
        "label_shift": label_shift,
        "concept_drift": concept_drift,
        "domain_shift": domain_shift,
        "retraining": {
            "in_progress": snapshot["retraining_in_progress"],
            "last_trigger_ts": snapshot["last_retrain_trigger_ts"],
            "last_feedback_count": snapshot["last_retrain_feedback_count"],
            "last_version": snapshot["last_retrain_version"],
            "last_error": snapshot["last_retrain_error"],
        },
    }


def _write_feedback_training_csv(feedback_rows: list[dict[str, Any]]) -> Path:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in feedback_rows:
        text = str(row.get("text", "")).strip()
        true_label = _normalize_label(str(row.get("true_label", "")))
        if not text or true_label is None:
            continue
        key = (text, true_label)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "label": true_label})

    if len(rows) < MIN_FEEDBACK_SAMPLES:
        raise RuntimeError(
            f"Not enough labeled feedback rows for retraining. Found {len(rows)}, expected at least {MIN_FEEDBACK_SAMPLES}."
        )

    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    with RETRAINING_DATA_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return RETRAINING_DATA_FILE


def _run_retraining_job() -> None:
    try:
        feedback_rows = load_feedback_data()
        feedback_count = len(feedback_rows)
        training_data_path = _write_feedback_training_csv(feedback_rows)
        base_model_path = _get_current_base_model_path()
        version_name = f"{base_model_path.name}_retrain_{int(time.time())}"
        output_model_path = MODELS_DIR / version_name

        logger.warning(
            "Automatic retraining triggered. Base model: %s, feedback samples: %s, output model: %s",
            base_model_path,
            feedback_count,
            output_model_path,
        )

        incremental_update_pipeline(
            base_model=_to_relative_path(base_model_path),
            new_data=_to_relative_path(training_data_path),
            output_model=_to_relative_path(output_model_path),
            version_name=version_name,
        )

        with _state_lock:
            _state.retraining_in_progress = False
            _state.last_retrain_feedback_count = feedback_count
            _state.last_retrain_version = version_name
            _state.last_retrain_error = None
            _state.retrain_needed = False
    except Exception as exc:
        logger.exception("Automatic retraining failed.")
        with _state_lock:
            _state.retraining_in_progress = False
            _state.last_retrain_error = str(exc)


def _maybe_trigger_retraining(report: dict[str, Any]) -> None:
    if not _auto_retrain_enabled():
        return
    if not report["retrain_needed"]:
        return

    feedback_samples = report["concept_drift"]["feedback_samples"]
    if feedback_samples < MIN_FEEDBACK_SAMPLES:
        return

    with _state_lock:
        if _state.retraining_in_progress:
            return
        if feedback_samples <= _state.last_retrain_feedback_count:
            return
        now = time.time()
        if now - _state.last_retrain_trigger_ts < RETRAINING_COOLDOWN_SECONDS:
            _state.retrain_needed = True
            return
        _state.retraining_in_progress = True
        _state.last_retrain_trigger_ts = now
        _state.retrain_needed = True
        _state.last_retrain_error = None

    threading.Thread(
        target=_run_retraining_job,
        name="finbert-auto-retrain",
        daemon=True,
    ).start()


def evaluate_retraining_need(psi_value: float, accuracy: float) -> dict[str, float | bool]:
    retrain_needed = psi_value > PSI_DRIFT_THRESHOLD or accuracy < ACCURACY_THRESHOLD
    return {
        "psi_value": psi_value,
        "accuracy": accuracy,
        "retrain_needed": retrain_needed,
    }


def record_prediction_event(text: str, label: str, score: float) -> dict[str, Any]:
    _seed_baseline()

    tokens = _tokenize(text)
    token_count = len(tokens)
    with _state_lock:
        baseline_vocabulary = set(_state.baseline_vocabulary)

    unseen_tokens = sum(1 for token in tokens if token not in baseline_vocabulary)
    novelty_ratio = (unseen_tokens / token_count) if token_count else 0.0
    predicted_label_id = _label_to_id(label)

    with _state_lock:
        _append_limited(_state.incoming_text_lengths, float(len(text)))
        _append_limited(_state.incoming_domain_novelty, float(novelty_ratio))
        if predicted_label_id is not None:
            _append_limited(_state.incoming_predicted_label_ids, predicted_label_id)

        confidence = float(score)
        if len(_state.baseline_confidences) < CONFIDENCE_BASELINE_BOOTSTRAP:
            _state.baseline_confidences.append(confidence)
        else:
            _append_limited(_state.incoming_confidences, confidence)

    report = compute_drift()
    _maybe_trigger_retraining(report)
    return report


def record_feedback_event(text: str, predicted_label: str, true_label: str) -> dict[str, Any]:
    _seed_baseline()

    true_label_id = _label_to_id(true_label)
    normalized_predicted = _normalize_label(predicted_label)
    normalized_true = _normalize_label(true_label)

    if true_label_id is not None and normalized_predicted is not None and normalized_true is not None:
        with _state_lock:
            _append_limited(_state.incoming_true_label_ids, true_label_id)
            _append_limited(_state.incoming_correctness, normalized_predicted == normalized_true)

    report = compute_drift()
    _maybe_trigger_retraining(report)
    return report


def compute_drift(bins: int = 10) -> dict[str, Any]:
    _seed_baseline()
    snapshot = _copy_state()
    report = _build_report(snapshot, bins=bins)
    with _state_lock:
        _state.last_report = report
        _state.retrain_needed = report["retrain_needed"] or _state.retraining_in_progress
    return report


def get_last_drift_report() -> dict[str, Any]:
    with _state_lock:
        if _state.last_report:
            return dict(_state.last_report)
    return compute_drift()


def reset_drift_state() -> None:
    with _state_lock:
        baseline_text_lengths = list(_state.baseline_text_lengths)
        baseline_label_ids = list(_state.baseline_label_ids)
        baseline_vocabulary = set(_state.baseline_vocabulary)
        _state.baseline_text_lengths = baseline_text_lengths
        _state.baseline_label_ids = baseline_label_ids
        _state.baseline_vocabulary = baseline_vocabulary
        _state.baseline_confidences = []
        _state.incoming_text_lengths.clear()
        _state.incoming_confidences.clear()
        _state.incoming_predicted_label_ids.clear()
        _state.incoming_true_label_ids.clear()
        _state.incoming_correctness.clear()
        _state.incoming_domain_novelty.clear()
        _state.retrain_needed = False
        _state.retraining_in_progress = False
        _state.last_retrain_trigger_ts = 0.0
        _state.last_retrain_feedback_count = 0
        _state.last_retrain_version = None
        _state.last_retrain_error = None
        _state.last_report = {}
