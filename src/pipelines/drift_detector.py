from __future__ import annotations


def evaluate_retraining_need(psi_value: float, accuracy: float) -> dict[str, float | bool]:
    retrain_needed = psi_value > 0.2 or accuracy < 0.90
    return {
        "psi_value": psi_value,
        "accuracy": accuracy,
        "retrain_needed": retrain_needed,
    }
