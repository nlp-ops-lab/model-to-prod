from pathlib import Path

from prefect import flow, task
from src.pipelines.model_registry import promote_model_to_production
from src.pipelines.train_finetune import fine_tune_model
from src.pipelines.evaluate import evaluate_model
from src.pipelines.mlflow_tracking import log_model_to_mlflow


BASE_DIR = Path(__file__).resolve().parents[2]


@task
def train_task(base_model, train_file, output_model, version_name):
    fine_tune_model(
        base_model_path=BASE_DIR / base_model,
        train_file=BASE_DIR / train_file,
        output_model_path=BASE_DIR / output_model,
        version_name=version_name,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
    )


@task
def evaluate_task(model_path, test_file, version_name):
    evaluate_model(
        model_path=BASE_DIR / model_path,
        test_file=BASE_DIR / test_file,
        version_name=version_name,
    )


@task
def mlflow_task(version_name, model_path, evaluation_file):
    log_model_to_mlflow(
        version_name=version_name,
        model_path=model_path,
        evaluation_file=evaluation_file,
    )

@task
def promote_task(model_path):
    promote_model_to_production(model_path)

@flow(name="week2-stateful-finetuning-pipeline")
def week2_training_pipeline():
    versions = [
        {
            "base_model": "models/FinbertConfiguration",
            "train_file": "data/batches/batch_0.csv",
            "output_model": "models/finbert_v1",
            "version_name": "finbert_v1",
        },
        {
            "base_model": "models/finbert_v1",
            "train_file": "data/batches/batch_1.csv",
            "output_model": "models/finbert_v2",
            "version_name": "finbert_v2",
        },
        {
            "base_model": "models/finbert_v2",
            "train_file": "data/batches/batch_2.csv",
            "output_model": "models/finbert_v3",
            "version_name": "finbert_v3",
        },
        {
            "base_model": "models/finbert_v3",
            "train_file": "data/batches/batch_3.csv",
            "output_model": "models/finbert_v4",
            "version_name": "finbert_v4",
        },
    ]

    test_file = "data/test/test.csv"

    for item in versions:
        train_task(
            item["base_model"],
            item["train_file"],
            item["output_model"],
            item["version_name"],
        )

        evaluate_task(
            item["output_model"],
            test_file,
            item["version_name"],
        )

        mlflow_task(
            item["version_name"],
            item["output_model"],
            f"artifacts/evaluation/{item['version_name']}_evaluation.json",
        )
        promote_task(item["output_model"])


if __name__ == "__main__":
    week2_training_pipeline()