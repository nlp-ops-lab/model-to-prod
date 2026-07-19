from src.pipelines.quantize_model import (
    QUANTIZED_MODEL_FILE,
    QUANTIZED_ONNX_FILE,
    main,
    quantize_and_log_model,
    quantize_model,
)


__all__ = [
    "QUANTIZED_MODEL_FILE",
    "QUANTIZED_ONNX_FILE",
    "quantize_model",
    "quantize_and_log_model",
]


if __name__ == "__main__":
    main()
