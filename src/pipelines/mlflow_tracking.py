import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_ARTIFACT_PATH = "model"
EVALUATION_ARTIFACT_PATH = "evaluation"
STANDARD_MODEL_TYPE = "standard"
QUANTIZED_MODEL_TYPE = "quantized"
LEGACY_STANDARD_EXPERIMENT_NAME = "finbert-week2-stateful-finetuning"
STANDARD_EXPERIMENT_NAME = "finbert-standard-stateful-finetuning"
QUANTIZED_EXPERIMENT_NAME = "finbert-quantized-stateful-finetuning"
DEFAULT_STANDARD_HYPERPARAMETERS = {
    "base_model_type": "FinBERT",
    "training_strategy": "stateful_fine_tuning",
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 2e-5,
}


@dataclass(frozen=True)
class LatestMLflowModel:
    model_type: str
    experiment_id: str
    experiment_name: str
    run_id: str
    version_name: str


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _normalize_model_type(model_type: str) -> str:
    normalized = model_type.strip().lower()
    if normalized not in {STANDARD_MODEL_TYPE, QUANTIZED_MODEL_TYPE}:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Expected one of: {STANDARD_MODEL_TYPE}, {QUANTIZED_MODEL_TYPE}."
        )
    return normalized


def get_experiment_name(model_type: str) -> str:
    normalized = _normalize_model_type(model_type)
    if normalized == QUANTIZED_MODEL_TYPE:
        return QUANTIZED_EXPERIMENT_NAME
    return STANDARD_EXPERIMENT_NAME


def _get_experiment_aliases(model_type: str) -> tuple[str, ...]:
    normalized = _normalize_model_type(model_type)
    if normalized == QUANTIZED_MODEL_TYPE:
        return (QUANTIZED_EXPERIMENT_NAME,)
    return (STANDARD_EXPERIMENT_NAME, LEGACY_STANDARD_EXPERIMENT_NAME)


def _load_evaluation_metrics(evaluation_file: str | Path) -> tuple[dict[str, Any], Path]:
    evaluation_path = _resolve_path(evaluation_file)
    with open(evaluation_path, "r", encoding="utf-8") as file:
        evaluation = json.load(file)
    metrics = evaluation["metrics"]
    return metrics, evaluation_path


def _log_params(params: dict[str, Any]) -> None:
    for key, value in params.items():
        mlflow.log_param(key, value)


def _log_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_model_to_mlflow(
    version_name: str,
    model_path: str | Path,
    evaluation_file: str | Path,
    *,
    model_type: str = STANDARD_MODEL_TYPE,
    hyperparameters: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    normalized_model_type = _normalize_model_type(model_type)
    model_dir = _resolve_path(model_path)
    metrics, evaluation_path = _load_evaluation_metrics(evaluation_file)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    run_params: dict[str, Any] = {
        "model_version": version_name,
        "model_type": normalized_model_type,
    }

    if normalized_model_type == STANDARD_MODEL_TYPE:
        run_params.update(DEFAULT_STANDARD_HYPERPARAMETERS)

    if hyperparameters:
        run_params.update(hyperparameters)

    if extra_params:
        run_params.update(extra_params)

    experiment_name = get_experiment_name(normalized_model_type)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=version_name) as run:
        mlflow.set_tags(
            {
                "model_version": version_name,
                "model_type": normalized_model_type,
                "experiment_name": experiment_name,
            }
        )
        _log_params(run_params)
        _log_metrics(metrics)
        mlflow.log_artifact(str(evaluation_path), artifact_path=EVALUATION_ARTIFACT_PATH)
        mlflow.log_artifacts(str(model_dir), artifact_path=MODEL_ARTIFACT_PATH)

    print(
        f"Logged {version_name} to MLflow successfully. "
        f"Run ID: {run.info.run_id}. Experiment: {experiment_name}"
    )
    return run.info.run_id


def log_quantized_model_to_mlflow(
    version_name: str,
    model_path: str | Path,
    evaluation_file: str | Path,
    *,
    source_model_path: str | Path,
    is_static: bool = False,
    per_channel: bool = False,
    provider: str = "CPUExecutionProvider",
    quantization_backend: str = "onnxruntime",
    extra_params: dict[str, Any] | None = None,
) -> str:
    source_path = _resolve_path(source_model_path)
    hyperparameters = {
        "base_model_type": "FinBERT",
        "quantization_backend": quantization_backend,
        "quantization_provider": provider,
        "quantization_is_static": is_static,
        "quantization_per_channel": per_channel,
        "source_model_path": str(source_path),
        "source_model_version": source_path.name,
    }

    return log_model_to_mlflow(
        version_name=version_name,
        model_path=model_path,
        evaluation_file=evaluation_file,
        model_type=QUANTIZED_MODEL_TYPE,
        hyperparameters=hyperparameters,
        extra_params=extra_params,
    )


def _get_client() -> MlflowClient:
    return MlflowClient()


def _get_experiment_ids(model_type: str) -> list[str]:
    aliases = set(_get_experiment_aliases(model_type))
    experiments = _get_client().search_experiments(max_results=500)
    return [
        experiment.experiment_id
        for experiment in experiments
        if experiment.name in aliases
    ]


def get_latest_model_run(use_quantized: bool) -> LatestMLflowModel:
    model_type = QUANTIZED_MODEL_TYPE if use_quantized else STANDARD_MODEL_TYPE
    experiment_ids = _get_experiment_ids(model_type)

    if not experiment_ids:
        expected = ", ".join(_get_experiment_aliases(model_type))
        raise FileNotFoundError(
            f"No MLflow experiment found for {model_type} models. "
            f"Expected one of: {expected}"
        )

    runs = _get_client().search_runs(
        experiment_ids=experiment_ids,
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )

    if not runs:
        raise FileNotFoundError(f"No finished MLflow runs found for {model_type} models.")

    latest_run = runs[0]
    experiment_name_lookup = {
        experiment.experiment_id: experiment.name
        for experiment in _get_client().search_experiments(max_results=500)
    }
    version_name = (
        latest_run.data.params.get("model_version")
        or latest_run.info.run_name
        or latest_run.info.run_id
    )

    return LatestMLflowModel(
        model_type=model_type,
        experiment_id=latest_run.info.experiment_id,
        experiment_name=experiment_name_lookup.get(
            latest_run.info.experiment_id,
            get_experiment_name(model_type),
        ),
        run_id=latest_run.info.run_id,
        version_name=version_name,
    )


def download_model_artifacts_from_mlflow(
    latest_model: LatestMLflowModel,
    destination_root: str | Path | None = None,
) -> Path:
    root = _resolve_path(destination_root or BASE_DIR / "artifacts" / "mlflow_cache")
    run_cache_dir = root / latest_model.model_type / latest_model.run_id
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_model_dir = run_cache_dir / MODEL_ARTIFACT_PATH

    if cached_model_dir.exists() and any(cached_model_dir.iterdir()):
        return cached_model_dir

    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=latest_model.run_id,
        artifact_path=MODEL_ARTIFACT_PATH,
        dst_path=str(run_cache_dir),
    )
    return Path(downloaded_path)


def fetch_latest_model_from_mlflow(
    use_quantized: bool,
    destination_root: str | Path | None = None,
) -> Path:
    latest_model = get_latest_model_run(use_quantized=use_quantized)
    return download_model_artifacts_from_mlflow(
        latest_model=latest_model,
        destination_root=destination_root,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--evaluation_file", required=True)
    parser.add_argument(
        "--model_type",
        choices=[STANDARD_MODEL_TYPE, QUANTIZED_MODEL_TYPE],
        default=STANDARD_MODEL_TYPE,
    )
    parser.add_argument("--source_model_path")
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--per_channel", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_type == QUANTIZED_MODEL_TYPE:
        if not args.source_model_path:
            raise ValueError("--source_model_path is required for quantized model logging.")
        log_quantized_model_to_mlflow(
            version_name=args.version_name,
            model_path=args.model_path,
            evaluation_file=args.evaluation_file,
            source_model_path=args.source_model_path,
            is_static=args.is_static,
            per_channel=args.per_channel,
        )
    else:
        log_model_to_mlflow(
            version_name=args.version_name,
            model_path=args.model_path,
            evaluation_file=args.evaluation_file,
            model_type=args.model_type,
        )
