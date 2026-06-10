from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

from src.pipelines.mlflow_tracking import (
    LatestMLflowModel,
    download_model_artifacts_from_mlflow,
    get_latest_model_run,
)


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PRODUCTION_MODEL_FILE = MODELS_DIR / "production_model.txt"
QUANTIZED_MODEL_FILE = MODELS_DIR / "quantized_model.txt"
DEFAULT_STANDARD_MODEL_PATH = MODELS_DIR / "FinbertConfiguration"
QUANTIZED_ONNX_FILE = "model_quantized.onnx"
USE_MLFLOW_LATEST_ENV = "FINBERT_USE_MLFLOW_LATEST"


@dataclass(frozen=True)
class ModelReference:
    source: str
    cache_key: str
    model_path: Path | None = None
    version_name: str | None = None
    pointer_mtime_ns: int | None = None
    mlflow_model: LatestMLflowModel | None = None


@dataclass
class PipelineCache:
    classifier: Any
    model_path: Path
    cache_key: str


_standard_pipeline_cache: PipelineCache | None = None
_quantized_pipeline_cache: PipelineCache | None = None


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
        "Run the quantized model pipeline first to create the quantized model."
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
            "Run src/pipelines/quantized_model_pipeline.py to generate it."
        )

    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path),
        file_name=QUANTIZED_ONNX_FILE,
        provider="CPUExecutionProvider",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


def _should_use_mlflow(prefer_mlflow: bool | None) -> bool:
    if prefer_mlflow is not None:
        return prefer_mlflow
    return _is_truthy(os.getenv(USE_MLFLOW_LATEST_ENV))


def _get_local_model_reference(use_quantized: bool) -> ModelReference:
    pointer_file = QUANTIZED_MODEL_FILE if use_quantized else PRODUCTION_MODEL_FILE
    model_path = get_quantized_model_path() if use_quantized else get_production_model_path()
    return ModelReference(
        source="pointer_file",
        cache_key=f"pointer:{model_path}:{_get_pointer_mtime_ns(pointer_file)}",
        model_path=_ensure_model_directory(model_path),
        pointer_mtime_ns=_get_pointer_mtime_ns(pointer_file),
    )


def _get_mlflow_model_reference(use_quantized: bool) -> ModelReference:
    latest_model = get_latest_model_run(use_quantized=use_quantized)
    return ModelReference(
        source="mlflow",
        cache_key=f"mlflow:{latest_model.run_id}",
        version_name=latest_model.version_name,
        mlflow_model=latest_model,
    )


def _resolve_model_reference(
    use_quantized: bool,
    prefer_mlflow: bool | None = None,
) -> ModelReference:
    if _should_use_mlflow(prefer_mlflow):
        try:
            return _get_mlflow_model_reference(use_quantized=use_quantized)
        except FileNotFoundError:
            if prefer_mlflow is True:
                raise
            return _get_local_model_reference(use_quantized=use_quantized)
    return _get_local_model_reference(use_quantized=use_quantized)


def _materialize_model_path(reference: ModelReference) -> Path:
    if reference.source == "mlflow":
        if reference.mlflow_model is None:
            raise RuntimeError("MLflow model metadata is missing from the model reference.")
        downloaded_path = download_model_artifacts_from_mlflow(
            latest_model=reference.mlflow_model,
            destination_root=ARTIFACTS_DIR / "mlflow_cache",
        )
        return _ensure_model_directory(downloaded_path)

    if reference.model_path is None:
        raise RuntimeError("Local model path is missing from the model reference.")
    return _ensure_model_directory(reference.model_path)


def get_classifier(use_quantized: bool = False, prefer_mlflow: bool | None = None):
    global _standard_pipeline_cache
    global _quantized_pipeline_cache

    reference = _resolve_model_reference(
        use_quantized=use_quantized,
        prefer_mlflow=prefer_mlflow,
    )
    cache = _quantized_pipeline_cache if use_quantized else _standard_pipeline_cache

    if cache is None or cache.cache_key != reference.cache_key:
        model_path = _materialize_model_path(reference)
        classifier = (
            _load_quantized_classifier(model_path)
            if use_quantized
            else _load_standard_classifier(model_path)
        )
        cache = PipelineCache(
            classifier=classifier,
            model_path=model_path,
            cache_key=reference.cache_key,
        )
        if use_quantized:
            _quantized_pipeline_cache = cache
        else:
            _standard_pipeline_cache = cache

    return cache.classifier


def load_latest_model_from_mlflow(use_quantized: bool = False) -> Path:
    reference = _get_mlflow_model_reference(use_quantized=use_quantized)
    return _materialize_model_path(reference)


def _format_prediction(text: str, prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": text,
        "label": prediction["label"],
        "score": float(prediction["score"]),
    }


def predict_sentiment(text: str) -> dict[str, Any]:
    classifier = get_classifier(use_quantized=False)
    prediction = classifier(text)[0]
    return _format_prediction(text, prediction)


def predict_quantized_sentiment(text: str) -> dict[str, Any]:
    classifier = get_classifier(use_quantized=True)
    prediction = classifier(text)[0]
    return _format_prediction(text, prediction)


def predict_batch(sentences: list[str], use_quantized: bool) -> list[dict[str, Any]]:
    if not sentences:
        return []

    classifier = get_classifier(use_quantized=use_quantized)
    predictions = classifier(sentences)
    return [
        _format_prediction(text, prediction)
        for text, prediction in zip(sentences, predictions, strict=False)
    ]


def _local_model_status(pointer_file: Path, fallback_path: Path | None = None) -> dict[str, Any]:
    try:
        model_path = _read_pointer_file(pointer_file, fallback_path=fallback_path)
        quantized_onnx_path = model_path / QUANTIZED_ONNX_FILE
        is_quantized = pointer_file == QUANTIZED_MODEL_FILE
        return {
            "source": "pointer_file",
            "pointer_file": str(pointer_file),
            "configured_path": str(model_path),
            "exists": model_path.exists(),
            "ready": quantized_onnx_path.exists() if is_quantized else model_path.exists(),
            "quantized": is_quantized,
        }
    except Exception as exc:
        return {
            "source": "pointer_file",
            "pointer_file": str(pointer_file),
            "configured_path": None,
            "exists": False,
            "ready": False,
            "quantized": pointer_file == QUANTIZED_MODEL_FILE,
            "error": str(exc),
        }


def _mlflow_model_status(use_quantized: bool) -> dict[str, Any]:
    try:
        latest_model = get_latest_model_run(use_quantized=use_quantized)
        cached_path = (
            ARTIFACTS_DIR
            / "mlflow_cache"
            / latest_model.model_type
            / latest_model.run_id
            / "model"
        )
        return {
            "source": "mlflow",
            "quantized": use_quantized,
            "experiment_name": latest_model.experiment_name,
            "version_name": latest_model.version_name,
            "run_id": latest_model.run_id,
            "cached_path": str(cached_path),
            "cached": cached_path.exists(),
            "ready": True,
        }
    except Exception as exc:
        return {
            "source": "mlflow",
            "quantized": use_quantized,
            "ready": False,
            "error": str(exc),
        }


def get_current_model_info() -> dict[str, Any]:
    return {
        "active_source": "mlflow" if _should_use_mlflow(None) else "pointer_file",
        "mlflow_flag_env": USE_MLFLOW_LATEST_ENV,
        "standard_model": _local_model_status(
            PRODUCTION_MODEL_FILE,
            fallback_path=DEFAULT_STANDARD_MODEL_PATH,
        ),
        "quantized_model": _local_model_status(QUANTIZED_MODEL_FILE),
        "mlflow_standard_model": _mlflow_model_status(use_quantized=False),
        "mlflow_quantized_model": _mlflow_model_status(use_quantized=True),
    }
