import argparse
import json
from pathlib import Path

import mlflow


BASE_DIR = Path(__file__).resolve().parents[2]


def log_model_to_mlflow(version_name: str, model_path: str, evaluation_file: str):
    model_dir = BASE_DIR / model_path
    eval_path = BASE_DIR / evaluation_file

    with open(eval_path, "r", encoding="utf-8") as file:
        evaluation = json.load(file)

    metrics = evaluation["metrics"]

    mlflow.set_experiment("finbert-week2-stateful-finetuning")

    with mlflow.start_run(run_name=version_name):
        mlflow.log_param("model_version", version_name)
        mlflow.log_param("base_model_type", "FinBERT")
        mlflow.log_param("training_strategy", "stateful_fine_tuning")
        mlflow.log_param("epochs", 1)
        mlflow.log_param("batch_size", 8)
        mlflow.log_param("learning_rate", 2e-5)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        mlflow.log_artifact(str(eval_path), artifact_path="evaluation")
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        print(f"Logged {version_name} to MLflow successfully.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--evaluation_file", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log_model_to_mlflow(
        version_name=args.version_name,
        model_path=args.model_path,
        evaluation_file=args.evaluation_file,
    )