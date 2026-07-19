from __future__ import annotations

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from src.pipelines.mlflow_tracking import log_quantized_model_to_mlflow


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
QUANTIZED_MODEL_FILE = MODELS_DIR / "quantized_model.txt"
QUANTIZED_ONNX_FILE = "model_quantized.onnx"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _to_stored_path(path_value: Path) -> str:
    try:
        return str(path_value.relative_to(BASE_DIR))
    except ValueError:
        return str(path_value)


def _write_quantized_pointer(output_path: Path) -> None:
    QUANTIZED_MODEL_FILE.write_text(_to_stored_path(output_path), encoding="utf-8")


def quantize_model(
    model_path: str | Path,
    output_path: str | Path,
    *,
    is_static: bool = False,
    per_channel: bool = False,
    provider: str = "CPUExecutionProvider",
) -> Path:
    source_model_path = _resolve_path(model_path)
    quantized_output_path = _resolve_path(output_path)

    if not source_model_path.exists():
        raise FileNotFoundError(f"Standard model path does not exist: {source_model_path}")

    quantized_output_path.mkdir(parents=True, exist_ok=True)

    quantized_onnx_path = quantized_output_path / QUANTIZED_ONNX_FILE
    if quantized_onnx_path.exists():
        print(f"Quantized model already exists: {quantized_onnx_path}")
        print("Skipping quantization to keep the pipeline idempotent.")
        _write_quantized_pointer(quantized_output_path)
        return quantized_output_path

    model = ORTModelForSequenceClassification.from_pretrained(
        str(source_model_path),
        export=True,
        provider=provider,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_model_path))

    model.save_pretrained(str(quantized_output_path))
    tokenizer.save_pretrained(str(quantized_output_path))

    quantizer = ORTQuantizer.from_pretrained(str(quantized_output_path))
    quantization_config = AutoQuantizationConfig.avx2(
        is_static=is_static,
        per_channel=per_channel,
    )
    quantizer.quantize(
        save_dir=str(quantized_output_path),
        quantization_config=quantization_config,
    )

    _write_quantized_pointer(quantized_output_path)
    return quantized_output_path


def quantize_and_log_model(
    model_path: str | Path,
    output_path: str | Path,
    *,
    version_name: str,
    evaluation_file: str | Path,
    is_static: bool = False,
    per_channel: bool = False,
    provider: str = "CPUExecutionProvider",
) -> tuple[Path, str]:
    quantized_output_path = quantize_model(
        model_path=model_path,
        output_path=output_path,
        is_static=is_static,
        per_channel=per_channel,
        provider=provider,
    )
    run_id = log_quantized_model_to_mlflow(
        version_name=version_name,
        model_path=quantized_output_path,
        evaluation_file=evaluation_file,
        source_model_path=model_path,
        is_static=is_static,
        per_channel=per_channel,
        provider=provider,
    )
    return quantized_output_path, run_id


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a standard FinBERT model to ONNX, quantize it, and optionally log it to MLflow.",
    )
    parser.add_argument(
        "model_path",
        help="Path to the standard model directory. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "output_path",
        help="Directory where the quantized model will be saved. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--version_name",
        help="Logical version name for MLflow logging. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--evaluation_file",
        help="Evaluation JSON file to log alongside the quantized model in MLflow.",
    )
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--per_channel", action="store_true")
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--skip_mlflow", action="store_true")
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    version_name = args.version_name or Path(args.output_path).name

    quantized_output_path = quantize_model(
        args.model_path,
        args.output_path,
        is_static=args.is_static,
        per_channel=args.per_channel,
        provider=args.provider,
    )

    if not args.skip_mlflow:
        if not args.evaluation_file:
            raise ValueError(
                "--evaluation_file is required unless --skip_mlflow is provided."
            )
        run_id = log_quantized_model_to_mlflow(
            version_name=version_name,
            model_path=quantized_output_path,
            evaluation_file=args.evaluation_file,
            source_model_path=args.model_path,
            is_static=args.is_static,
            per_channel=args.per_channel,
            provider=args.provider,
        )
        print(f"MLflow run created: {run_id}")

    print(f"Quantized model saved to: {quantized_output_path}")
    print(f"Latest quantized model pointer updated: {QUANTIZED_MODEL_FILE}")


if __name__ == "__main__":
    main()
