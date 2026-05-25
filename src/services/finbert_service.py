import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "FinbertConfiguration")

classifier = pipeline(
    "sentiment-analysis",
    model=MODEL_PATH,
    tokenizer=MODEL_PATH
)


def predict_sentiment(text: str):
    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": float(result["score"])
    }