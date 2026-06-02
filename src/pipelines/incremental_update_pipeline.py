import argparse
from pathlib import Path

from prefect import flow, task

from src.pipelines.train_finetune import fine_tune_model
from src.pipelines.evaluate import evaluate_model
from src.pipelines.mlflow_tracking import log_model_to_mlflow
from src.pipelines.model_registry import promote_model_to_production

BASE_DIR = Path(__file__).resolve().parents[2]
TEST_FILE = "data/test/test.csv"


@task
def train_incremental_task(base_model, new_data, output_model, version_name):
    fine_tune_model(
        base_model_path=BASE_DIR / base_model,
        train_file=BASE_DIR / new_data,
        output_model_path=BASE_DIR / output_model,
        version_name=version_name,
        epochs=1,
        batch_size=8,
        learning_rate=2e-5,
    )


@task
def evaluate_incremental_task(output_model, version_name):
    evaluate_model(
        model_path=BASE_DIR / output_model,
        test_file=BASE_DIR / TEST_FILE,
        version_name=version_name,
    )


@task
def mlflow_incremental_task(version_name, output_model):
    evaluation_file = f"artifacts/evaluation/{version_name}_evaluation.json"

    log_model_to_mlflow(
        version_name=version_name,
        model_path=output_model,
        evaluation_file=evaluation_file,
    )

@task
def promote_incremental_task(output_model):
    promote_model_to_production(output_model)

@flow(name="incremental-stateful-update-pipeline")
def incremental_update_pipeline(
    base_model: str,
    new_data: str,
    output_model: str,
    version_name: str,
):
    train_incremental_task(
        base_model,
        new_data,
        output_model,
        version_name,
    )

    evaluate_incremental_task(
        output_model,
        version_name,
    )

    mlflow_incremental_task(
        version_name,
        output_model,
    )
    
    promote_incremental_task(output_model)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--new_data", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--version_name", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    incremental_update_pipeline(
        base_model=args.base_model,
        new_data=args.new_data,
        output_model=args.output_model,
        version_name=args.version_name,
    )