from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
PRODUCTION_MODEL_FILE = MODELS_DIR / "production_model.txt"
QUANTIZED_MODEL_FILE = MODELS_DIR / "quantized_model.txt"
DEFAULT_STANDARD_MODEL_PATH = MODELS_DIR / "FinbertConfiguration"
QUANTIZED_ONNX_FILE = "model_quantized.onnx"


@dataclass
class PipelineCache:
    classifier: Any
    model_path: Path
    pointer_mtime_ns: int | None


_standard_pipeline_cache: PipelineCache | None = None
_quantized_pipeline_cache: PipelineCache | None = None


def _resolve_model_path(saved_path: str) -> Path:
    path = Path(saved_path)
    return path if path.is_absolute() else BASE_DIR / path


def _read_pointer_file(pointer_file: Path, fallback_path: Path | None = None) -> Path:
    if pointer_file.exists():
        configured_path = pointer_file.read_text(encoding="utf-8").strip()
        if not configured_path:
            raise RuntimeError(f"Model pointer file is empty: {pointer_file}")
        return _resolve_model_path(configured_path)

    if fallback_path is not None:
        return fallback_path

    raise FileNotFoundError(
        f"Model pointer file not found: {pointer_file}. "
        "Run the quantization script first to create the quantized model."
    )


def _get_pointer_mtime_ns(pointer_file: Path) -> int | None:
    if not pointer_file.exists():
        return None
    return pointer_file.stat().st_mtime_ns


def get_production_model_path() -> Path:
    return _read_pointer_file(PRODUCTION_MODEL_FILE, fallback_path=DEFAULT_STANDARD_MODEL_PATH)


def get_quantized_model_path() -> Path:
    return _read_pointer_file(QUANTIZED_MODEL_FILE)


def _ensure_model_directory(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Configured model path does not exist: {model_path}")
    return model_path


def _load_standard_classifier(model_path: Path):
    return pipeline(
        "sentiment-analysis",
        model=str(model_path),
        tokenizer=str(model_path),
    )


def _load_quantized_classifier(model_path: Path):
    quantized_onnx_path = model_path / QUANTIZED_ONNX_FILE
    if not quantized_onnx_path.exists():
        raise FileNotFoundError(
            f"Quantized ONNX file not found: {quantized_onnx_path}. "
            "Run src/pipelines/quantize.py to generate it."
        )

    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path),
        file_name=QUANTIZED_ONNX_FILE,
        provider="CPUExecutionProvider",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def _get_cached_classifier(use_quantized: bool):
    global _standard_pipeline_cache
    global _quantized_pipeline_cache

    pointer_file = QUANTIZED_MODEL_FILE if use_quantized else PRODUCTION_MODEL_FILE
    model_path = get_quantized_model_path() if use_quantized else get_production_model_path()
    model_path = _ensure_model_directory(model_path)
    pointer_mtime_ns = _get_pointer_mtime_ns(pointer_file)

    cache = _quantized_pipeline_cache if use_quantized else _standard_pipeline_cache
    if (
        cache is None
        or cache.model_path != model_path
        or cache.pointer_mtime_ns != pointer_mtime_ns
    ):
        classifier = (
            _load_quantized_classifier(model_path)
            if use_quantized
            else _load_standard_classifier(model_path)
        )
        cache = PipelineCache(
            classifier=classifier,
            model_path=model_path,
            pointer_mtime_ns=pointer_mtime_ns,
        )
        if use_quantized:
            _quantized_pipeline_cache = cache
        else:
            _standard_pipeline_cache = cache

    return cache.classifier


def _format_prediction(text: str, prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": text,
        "label": prediction["label"],
        "score": float(prediction["score"]),
    }


def predict_sentiment(text: str) -> dict[str, Any]:
    classifier = _get_cached_classifier(use_quantized=False)
    prediction = classifier(text)[0]
    return _format_prediction(text, prediction)


def predict_quantized_sentiment(text: str) -> dict[str, Any]:
    classifier = _get_cached_classifier(use_quantized=True)
    prediction = classifier(text)[0]
    return _format_prediction(text, prediction)


def predict_batch(sentences: list[str], use_quantized: bool) -> list[dict[str, Any]]:
    if not sentences:
        return []

    classifier = _get_cached_classifier(use_quantized=use_quantized)
    predictions = classifier(sentences)
    return [
        _format_prediction(text, prediction)
        for text, prediction in zip(sentences, predictions, strict=False)
    ]


def _model_status(pointer_file: Path, fallback_path: Path | None = None) -> dict[str, Any]:
    try:
        model_path = _read_pointer_file(pointer_file, fallback_path=fallback_path)
        quantized_onnx_path = model_path / QUANTIZED_ONNX_FILE
        is_quantized = pointer_file == QUANTIZED_MODEL_FILE
        return {
            "pointer_file": str(pointer_file),
            "configured_path": str(model_path),
            "exists": model_path.exists(),
            "ready": quantized_onnx_path.exists() if is_quantized else model_path.exists(),
            "quantized": is_quantized,
        }
    except Exception as exc:
        return {
            "pointer_file": str(pointer_file),
            "configured_path": None,
            "exists": False,
            "ready": False,
            "quantized": pointer_file == QUANTIZED_MODEL_FILE,
            "error": str(exc),
        }


def get_current_model_info() -> dict[str, Any]:
    return {
        "standard_model": _model_status(
            PRODUCTION_MODEL_FILE,
            fallback_path=DEFAULT_STANDARD_MODEL_PATH,
        ),
        "quantized_model": _model_status(QUANTIZED_MODEL_FILE),
    }
