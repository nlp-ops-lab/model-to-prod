from __future__ import annotations

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
QUANTIZED_MODEL_FILE = MODELS_DIR / "quantized_model.txt"


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_DIR / path


def _write_quantized_pointer(output_path: Path) -> None:
    try:
        stored_path = output_path.relative_to(BASE_DIR)
    except ValueError:
        stored_path = output_path

    QUANTIZED_MODEL_FILE.write_text(str(stored_path), encoding="utf-8")


def quantize_model(model_path: str | Path, output_path: str | Path) -> Path:
    source_model_path = _resolve_path(model_path)
    quantized_output_path = _resolve_path(output_path)

    if not source_model_path.exists():
        raise FileNotFoundError(f"Standard model path does not exist: {source_model_path}")

    quantized_output_path.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(
        str(source_model_path),
        export=True,
        provider="CPUExecutionProvider",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(source_model_path))

    model.save_pretrained(str(quantized_output_path))
    tokenizer.save_pretrained(str(quantized_output_path))

    quantizer = ORTQuantizer.from_pretrained(str(quantized_output_path))
    quantization_config = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=False,
    )
    quantizer.quantize(
        save_dir=str(quantized_output_path),
        quantization_config=quantization_config,
    )

    _write_quantized_pointer(quantized_output_path)
    return quantized_output_path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a standard FinBERT model to ONNX and quantize it for ONNX Runtime.",
    )
    parser.add_argument(
        "model_path",
        help="Path to the standard model directory. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "output_path",
        help="Directory where the quantized model will be saved. Relative paths are resolved from the project root.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    quantized_output_path = quantize_model(args.model_path, args.output_path)
    print(f"Quantized model saved to: {quantized_output_path}")
    print(f"Latest quantized model pointer updated: {QUANTIZED_MODEL_FILE}")


if __name__ == "__main__":
    main()
