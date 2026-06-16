from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

from src.pipelines.mlflow_tracking import (
    LatestMLflowModel,
    download_model_artifacts_from_mlflow,
    get_latest_model_run,
)


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PRODUCTION_MODEL_FILE = MODELS_DIR / "production_model.txt"
QUANTIZED_MODEL_FILE = MODELS_DIR / "quantized_model.txt"
DEFAULT_STANDARD_MODEL_PATH = MODELS_DIR / "FinbertConfiguration"
QUANTIZED_ONNX_FILE = "model_quantized.onnx"
USE_MLFLOW_LATEST_ENV = "FINBERT_USE_MLFLOW_LATEST"
WARMUP_TEXT = "Warmup text for FinBERT inference."
FALLBACK_MODEL_PATH = DEFAULT_STANDARD_MODEL_PATH


@dataclass(frozen=True)
class ModelReference:
    source: str
    cache_key: str
    model_path: Path | None = None
    version_name: str | None = None
    pointer_file: Path | None = None
    pointer_mtime_ns: int | None = None
    mlflow_model: LatestMLflowModel | None = None


@dataclass
class PipelineCache:
    classifier: Any
    model_path: Path
    cache_key: str
    source: str
    version_name: str | None
    use_quantized: bool


class DummySentimentClassifier:
    def __call__(self, payload: str | list[str]) -> list[dict[str, float | str]]:
        if isinstance(payload, str):
            return [self._predict_one(payload)]
        return [self._predict_one(text) for text in payload]

    @staticmethod
    def _predict_one(_: str) -> dict[str, float | str]:
        return {
            "label": "neutral",
            "score": 1.0,
        }


_standard_pipeline_cache: PipelineCache | None = None
_quantized_pipeline_cache: PipelineCache | None = None
_standard_load_lock = threading.Lock()
_quantized_load_lock = threading.Lock()


def _model_label(use_quantized: bool) -> str:
    return "quantized" if use_quantized else "standard"


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_path(saved_path: str) -> Path:
    path = Path(saved_path)
    return path if path.is_absolute() else BASE_DIR / path


def _read_pointer_file(pointer_file: Path) -> Path:
    if not pointer_file.exists():
        raise FileNotFoundError(f"Model pointer file not found: {pointer_file}")

    configured_path = pointer_file.read_text(encoding="utf-8").strip()
    if not configured_path:
        raise RuntimeError(f"Model pointer file is empty: {pointer_file}")

    return _resolve_model_path(configured_path)


def _get_pointer_mtime_ns(pointer_file: Path) -> int | None:
    if not pointer_file.exists():
        return None
    return pointer_file.stat().st_mtime_ns


def _ensure_model_directory(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Configured model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise NotADirectoryError(f"Configured model path is not a directory: {model_path}")
    return model_path


def _list_model_directories() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return [path for path in MODELS_DIR.iterdir() if path.is_dir()]


def _discover_standard_local_model_path() -> Path:
    candidates = [
        path
        for path in _list_model_directories()
        if path.name != DEFAULT_STANDARD_MODEL_PATH.name
        and not (path / QUANTIZED_ONNX_FILE).exists()
        and (path / "config.json").exists()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)

    if candidates:
        return candidates[0]

    if DEFAULT_STANDARD_MODEL_PATH.exists():
        return DEFAULT_STANDARD_MODEL_PATH

    raise FileNotFoundError(
        "No local standard model directory is available. "
        "Expected a fine-tuned model under models/ or the default FinbertConfiguration."
    )


def _discover_quantized_local_model_path() -> Path:
    candidates = [
        path
        for path in _list_model_directories()
        if (path / QUANTIZED_ONNX_FILE).exists()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No local quantized model directory is available. "
        "Run the quantized model pipeline first."
    )


def get_production_model_path() -> Path:
    try:
        return _ensure_model_directory(_read_pointer_file(PRODUCTION_MODEL_FILE))
    except Exception:
        return _ensure_model_directory(_discover_standard_local_model_path())


def get_quantized_model_path() -> Path:
    try:
        return _ensure_model_directory(_read_pointer_file(QUANTIZED_MODEL_FILE))
    except Exception:
        return _ensure_model_directory(_discover_quantized_local_model_path())


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
    env_value = os.getenv(USE_MLFLOW_LATEST_ENV)
    if env_value is None:
        return True
    return _is_truthy(env_value)


def _is_ci_environment() -> bool:
    return _is_truthy(os.getenv("CI"))


def _get_pointer_model_reference(use_quantized: bool) -> ModelReference:
    pointer_file = QUANTIZED_MODEL_FILE if use_quantized else PRODUCTION_MODEL_FILE
    model_path = _ensure_model_directory(_read_pointer_file(pointer_file))
    return ModelReference(
        source="pointer_file",
        cache_key=f"pointer:{model_path}:{_get_pointer_mtime_ns(pointer_file)}",
        model_path=model_path,
        pointer_file=pointer_file,
        pointer_mtime_ns=_get_pointer_mtime_ns(pointer_file),
        version_name=model_path.name,
    )


def _get_local_model_reference(use_quantized: bool) -> ModelReference:
    model_path = (
        _discover_quantized_local_model_path()
        if use_quantized
        else _discover_standard_local_model_path()
    )
    return ModelReference(
        source="local_model",
        cache_key=f"local:{model_path}",
        model_path=_ensure_model_directory(model_path),
        version_name=model_path.name,
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
    resolvers: list[tuple[str, Callable[[], ModelReference]]] = []
    if _should_use_mlflow(prefer_mlflow):
        resolvers.append(("mlflow", lambda: _get_mlflow_model_reference(use_quantized)))
    resolvers.extend(
        [
            ("pointer_file", lambda: _get_pointer_model_reference(use_quantized)),
            ("local_model", lambda: _get_local_model_reference(use_quantized)),
        ]
    )

    resolution_errors: list[str] = []
    for source_name, resolver in resolvers:
        try:
            reference = resolver()
            logger.info(
                "Resolved %s model from %s",
                _model_label(use_quantized),
                source_name,
            )
            return reference
        except Exception as exc:
            resolution_errors.append(f"{source_name}: {exc}")
            logger.warning(
                "Failed to resolve %s model from %s: %s",
                _model_label(use_quantized),
                source_name,
                exc,
            )

    raise RuntimeError(
        f"Unable to resolve a {_model_label(use_quantized)} model. "
        "Tried MLflow latest -> pointer file -> local model. "
        f"Details: {' | '.join(resolution_errors)}"
    )


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


def _get_pipeline_cache(use_quantized: bool) -> PipelineCache | None:
    return _quantized_pipeline_cache if use_quantized else _standard_pipeline_cache


def _set_pipeline_cache(use_quantized: bool, cache: PipelineCache) -> None:
    global _standard_pipeline_cache
    global _quantized_pipeline_cache

    if use_quantized:
        _quantized_pipeline_cache = cache
    else:
        _standard_pipeline_cache = cache


def _get_load_lock(use_quantized: bool) -> threading.Lock:
    return _quantized_load_lock if use_quantized else _standard_load_lock


def _warmup_classifier(classifier: Any, use_quantized: bool) -> None:
    warmup_start = perf_counter()
    classifier(WARMUP_TEXT)
    logger.info(
        "Warmup inference completed for %s model in %.3fs",
        _model_label(use_quantized),
        perf_counter() - warmup_start,
    )


def _build_dummy_pipeline_cache(use_quantized: bool) -> PipelineCache:
    logger.warning(
        "No production model found. Falling back to dummy neutral classifier for %s model.",
        _model_label(use_quantized),
    )
    classifier = DummySentimentClassifier()
    _warmup_classifier(classifier, use_quantized=use_quantized)
    return PipelineCache(
        classifier=classifier,
        model_path=FALLBACK_MODEL_PATH,
        cache_key=f"dummy_fallback:{_model_label(use_quantized)}",
        source="dummy_fallback",
        version_name="dummy_neutral",
        use_quantized=use_quantized,
    )


def _build_ci_safe_fallback_cache(
    use_quantized: bool,
    resolution_error: Exception,
) -> PipelineCache:
    logger.warning(
        "No production model found for %s model. Falling back to base FinBERT model for CI safety. Reason: %s",
        _model_label(use_quantized),
        resolution_error,
    )

    if FALLBACK_MODEL_PATH.exists():
        try:
            load_start = perf_counter()
            classifier = _load_standard_classifier(FALLBACK_MODEL_PATH)
            logger.info(
                "Loaded %s fallback model from %s in %.3fs",
                _model_label(use_quantized),
                FALLBACK_MODEL_PATH,
                perf_counter() - load_start,
            )
            _warmup_classifier(classifier, use_quantized=use_quantized)
            return PipelineCache(
                classifier=classifier,
                model_path=FALLBACK_MODEL_PATH,
                cache_key=f"base_fallback:{_model_label(use_quantized)}",
                source="base_model_fallback",
                version_name=FALLBACK_MODEL_PATH.name,
                use_quantized=use_quantized,
            )
        except Exception as fallback_exc:
            logger.warning(
                "Failed to load base FinBERT fallback model for %s model: %s",
                _model_label(use_quantized),
                fallback_exc,
            )

    return _build_dummy_pipeline_cache(use_quantized)


def _build_pipeline_cache(
    use_quantized: bool,
    prefer_mlflow: bool | None = None,
) -> PipelineCache:
    try:
        reference = _resolve_model_reference(
            use_quantized=use_quantized,
            prefer_mlflow=prefer_mlflow,
        )
    except RuntimeError as exc:
        if not _is_ci_environment():
            raise
        return _build_ci_safe_fallback_cache(
            use_quantized=use_quantized,
            resolution_error=exc,
        )
    model_path = _materialize_model_path(reference)

    load_start = perf_counter()
    classifier = (
        _load_quantized_classifier(model_path)
        if use_quantized
        else _load_standard_classifier(model_path)
    )
    logger.info(
        "Loaded %s model from %s (%s) in %.3fs",
        _model_label(use_quantized),
        reference.source,
        model_path,
        perf_counter() - load_start,
    )

    _warmup_classifier(classifier, use_quantized=use_quantized)

    return PipelineCache(
        classifier=classifier,
        model_path=model_path,
        cache_key=reference.cache_key,
        source=reference.source,
        version_name=reference.version_name or model_path.name,
        use_quantized=use_quantized,
    )


def get_classifier(use_quantized: bool = False, prefer_mlflow: bool | None = None):
    cache = _get_pipeline_cache(use_quantized)
    if cache is not None:
        return cache.classifier

    load_lock = _get_load_lock(use_quantized)
    with load_lock:
        cache = _get_pipeline_cache(use_quantized)
        if cache is None:
            logger.info("Initializing %s model", _model_label(use_quantized))
            try:
                cache = _build_pipeline_cache(
                    use_quantized=use_quantized,
                    prefer_mlflow=prefer_mlflow,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize %s model",
                    _model_label(use_quantized),
                )
                raise
            _set_pipeline_cache(use_quantized, cache)

    return cache.classifier


def preload_models() -> dict[str, Any]:
    get_classifier(use_quantized=False)
    get_classifier(use_quantized=True)
    return get_current_model_info()


def is_model_ready(use_quantized: bool = False) -> bool:
    return _get_pipeline_cache(use_quantized) is not None


def are_models_ready() -> bool:
    return is_model_ready(use_quantized=False) and is_model_ready(use_quantized=True)


def load_latest_model_from_mlflow(use_quantized: bool = False) -> Path:
    reference = _get_mlflow_model_reference(use_quantized=use_quantized)
    return _materialize_model_path(reference)


def _format_prediction(text: str, prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": text,
        "label": prediction["label"],
        "score": float(prediction["score"]),
    }


def _predict_with_logging(classifier: Any, payload: str | list[str], use_quantized: bool):
    start = perf_counter()
    predictions = classifier(payload)
    logger.info(
        "Inference completed with %s model in %.3fs",
        _model_label(use_quantized),
        perf_counter() - start,
    )
    return predictions


def predict_sentiment(text: str) -> dict[str, Any]:
    classifier = get_classifier(use_quantized=False)
    prediction = _predict_with_logging(classifier, text, use_quantized=False)[0]
    return _format_prediction(text, prediction)


def predict_quantized_sentiment(text: str) -> dict[str, Any]:
    classifier = get_classifier(use_quantized=True)
    prediction = _predict_with_logging(classifier, text, use_quantized=True)[0]
    return _format_prediction(text, prediction)


def predict_batch(sentences: list[str], use_quantized: bool) -> list[dict[str, Any]]:
    if not sentences:
        return []

    classifier = get_classifier(use_quantized=use_quantized)
    predictions = _predict_with_logging(classifier, sentences, use_quantized=use_quantized)
    return [
        _format_prediction(text, prediction)
        for text, prediction in zip(sentences, predictions, strict=False)
    ]


def _loaded_model_status(use_quantized: bool) -> dict[str, Any]:
    cache = _get_pipeline_cache(use_quantized)
    if cache is None:
        return {
            "loaded": False,
            "ready": False,
            "quantized": use_quantized,
        }

    return {
        "loaded": True,
        "ready": True,
        "quantized": use_quantized,
        "source": cache.source,
        "model_path": str(cache.model_path),
        "version_name": cache.version_name,
    }


def _pointer_model_status(pointer_file: Path) -> dict[str, Any]:
    try:
        model_path = _ensure_model_directory(_read_pointer_file(pointer_file))
        is_quantized = pointer_file == QUANTIZED_MODEL_FILE
        quantized_ready = not is_quantized or (model_path / QUANTIZED_ONNX_FILE).exists()
        return {
            "source": "pointer_file",
            "pointer_file": str(pointer_file),
            "configured_path": str(model_path),
            "exists": model_path.exists(),
            "ready": quantized_ready,
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


def _local_fallback_status(use_quantized: bool) -> dict[str, Any]:
    try:
        model_path = (
            _discover_quantized_local_model_path()
            if use_quantized
            else _discover_standard_local_model_path()
        )
        return {
            "source": "local_model",
            "quantized": use_quantized,
            "configured_path": str(model_path),
            "exists": model_path.exists(),
            "ready": True,
        }
    except Exception as exc:
        return {
            "source": "local_model",
            "quantized": use_quantized,
            "ready": False,
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
        "status": "ready" if are_models_ready() else "not_ready",
        "fallback_order": ["mlflow", "pointer_file", "local_model"],
        "mlflow_flag_env": USE_MLFLOW_LATEST_ENV,
        "runtime_standard_model": _loaded_model_status(use_quantized=False),
        "runtime_quantized_model": _loaded_model_status(use_quantized=True),
        "standard_model": _pointer_model_status(PRODUCTION_MODEL_FILE),
        "quantized_model": _pointer_model_status(QUANTIZED_MODEL_FILE),
        "local_standard_model": _local_fallback_status(use_quantized=False),
        "local_quantized_model": _local_fallback_status(use_quantized=True),
        "mlflow_standard_model": _mlflow_model_status(use_quantized=False),
        "mlflow_quantized_model": _mlflow_model_status(use_quantized=True),
    }
