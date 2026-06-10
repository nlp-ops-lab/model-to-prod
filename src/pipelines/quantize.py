from pathlib import Path
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
from transformers import AutoTokenizer
import os

BASE_DIR = Path(__file__).resolve().parents[2]

def quantize_model(model_path: str, output_path: str):
    model_path = BASE_DIR / model_path
    output_path = BASE_DIR / output_path

    model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    quantizer = ORTQuantizer.from_pretrained(output_path)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=output_path, quantization_config=qconfig)

    onnx_original = output_path / "model.onnx"
    if onnx_original.exists():
        os.remove(onnx_original)

    print(f"Model quantized successfully: {output_path}")

if __name__ == "__main__":
    quantize_model("models/finbert_v4", "models/finbert_v4_quantized")
