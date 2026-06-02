import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from transformers import pipeline


BASE_DIR = Path(__file__).resolve().parents[2]

PRODUCTION_MODEL_FILE = (
    BASE_DIR / "models" / "production_model.txt"
)

_classifier = None
_loaded_model_path = None


def get_production_model_path() -> Path:
    if PRODUCTION_MODEL_FILE.exists():
        relative_path = (
            PRODUCTION_MODEL_FILE
            .read_text(encoding="utf-8")
            .strip()
        )

        return BASE_DIR / relative_path

    return BASE_DIR / "models" / "FinbertConfiguration"


def get_classifier():
    global _classifier
    global _loaded_model_path

    model_path = get_production_model_path()

    if (
        _classifier is None
        or _loaded_model_path != model_path
    ):
        print(f"Loading model: {model_path}")

        _classifier = pipeline(
            "sentiment-analysis",
            model=str(model_path),
            tokenizer=str(model_path),
        )

        _loaded_model_path = model_path

    return _classifier


def predict_sentiment(text: str):
    classifier = get_classifier()

    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": float(result["score"]),
    }


def get_current_model_info():
    model_path = get_production_model_path()

    return {
        "model_path": str(model_path),
        "exists": model_path.exists(),
    }