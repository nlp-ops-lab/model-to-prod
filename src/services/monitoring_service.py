from __future__ import annotations

import threading


_metrics_lock = threading.Lock()
_request_count = 0
_avg_latency = 0.0
_drift_score = 0.0


def record_request_latency(latency_seconds: float) -> None:
    global _request_count
    global _avg_latency

    with _metrics_lock:
        _request_count += 1
        _avg_latency += (latency_seconds - _avg_latency) / _request_count


def set_drift_score(drift_score: float) -> None:
    global _drift_score

    with _metrics_lock:
        _drift_score = drift_score


def get_metrics() -> dict[str, float | int]:
    with _metrics_lock:
        return {
            "avg_latency": float(_avg_latency),
            "request_count": int(_request_count),
            "drift_score": float(_drift_score),
        }
