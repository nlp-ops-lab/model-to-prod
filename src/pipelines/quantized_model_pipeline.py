import argparse
from pathlib import Path

from prefect import flow, task

from src.pipelines.evaluate import evaluate_model
from src.pipelines.mlflow_tracking import log_quantized_model_to_mlflow
from src.pipelines.quantize_model import quantize_model


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEST_FILE = "data/test/test.csv"


@task
def quantize_task(
    source_model: str,
    output_model: str,
    is_static: bool,
    per_channel: bool,
):
    quantized_model_path = quantize_model(
        model_path=BASE_DIR / source_model,
        output_path=BASE_DIR / output_model,
        is_static=is_static,
        per_channel=per_channel,
    )
    return str(quantized_model_path)


@task
def evaluate_quantized_task(output_model: str, test_file: str, version_name: str):
    evaluation_path = evaluate_model(
        model_path=BASE_DIR / output_model,
        test_file=BASE_DIR / test_file,
        version_name=version_name,
        use_quantized=True,
    )
    return str(evaluation_path)


@task
def mlflow_quantized_task(
    version_name: str,
    source_model: str,
    output_model: str,
    evaluation_file: str,
    is_static: bool,
    per_channel: bool,
):
    return log_quantized_model_to_mlflow(
        version_name=version_name,
        model_path=BASE_DIR / output_model,
        evaluation_file=evaluation_file,
        source_model_path=BASE_DIR / source_model,
        is_static=is_static,
        per_channel=per_channel,
    )


@flow(name="quantized-model-pipeline")
def quantized_model_pipeline(
    source_model: str,
    output_model: str,
    version_name: str,
    test_file: str = DEFAULT_TEST_FILE,
    is_static: bool = False,
    per_channel: bool = False,
):
    quantized_model_path = quantize_task(
        source_model=source_model,
        output_model=output_model,
        is_static=is_static,
        per_channel=per_channel,
    )

    evaluation_file = evaluate_quantized_task(
        output_model=output_model,
        test_file=test_file,
        version_name=version_name,
    )

    mlflow_run_id = mlflow_quantized_task(
        version_name=version_name,
        source_model=source_model,
        output_model=output_model,
        evaluation_file=evaluation_file,
        is_static=is_static,
        per_channel=per_channel,
    )

    return {
        "quantized_model_path": quantized_model_path,
        "evaluation_file": evaluation_file,
        "mlflow_run_id": mlflow_run_id,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_model", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--version_name", required=True)
    parser.add_argument("--test_file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--per_channel", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    quantized_model_pipeline(
        source_model=args.source_model,
        output_model=args.output_model,
        version_name=args.version_name,
        test_file=args.test_file,
        is_static=args.is_static,
        per_channel=args.per_channel,
    )
