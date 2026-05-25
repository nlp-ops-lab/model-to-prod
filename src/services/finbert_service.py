from transformers import pipeline
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "finbert")

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