from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = BASE_DIR / "artifacts" / "feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"
_feedback_write_lock = threading.Lock()


def _build_feedback_record(
    text: str,
    predicted_label: str,
    true_label: str,
) -> dict[str, str]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "predicted_label": predicted_label,
        "true_label": true_label,
    }


def record_feedback(
    text: str,
    predicted_label: str,
    true_label: str,
) -> dict[str, str]:
    record = _build_feedback_record(
        text=text,
        predicted_label=predicted_label,
        true_label=true_label,
    )

    with _feedback_write_lock:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def load_feedback_data() -> list[dict[str, Any]]:
    with _feedback_write_lock:
        if not FEEDBACK_FILE.exists():
            return []

        with FEEDBACK_FILE.open("r", encoding="utf-8") as file:
            return [
                json.loads(line)
                for line in file
                if line.strip()
            ]
