import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification


os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


BASE_DIR = Path(__file__).resolve().parents[2]

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_test_data(test_file: Path):
    df = pd.read_csv(test_file)
    df = df.dropna()
    df = df[df["label"].isin(LABEL2ID.keys())]

    texts = df["text"].astype(str).tolist()
    labels = df["label"].map(LABEL2ID).tolist()

    return texts, labels


def predict(model, tokenizer, texts, batch_size=16):
    model.eval()

    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()

        predictions.extend(batch_preds)

    return predictions


def evaluate_model(model_path: Path, test_file: Path, version_name: str):
    texts, true_labels = load_test_data(test_file)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    predicted_labels = predict(model, tokenizer, texts)

    accuracy = accuracy_score(true_labels, predicted_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(true_labels, predicted_labels).tolist()

    report = {
        "version_name": version_name,
        "model_path": str(model_path),
        "test_file": str(test_file),
        "test_samples": len(true_labels),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        "confusion_matrix": cm,
        "labels": ID2LABEL,
    }

    output_dir = BASE_DIR / "artifacts" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{version_name}_evaluation.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)

    print("Evaluation completed.")
    print(json.dumps(report, ensure_ascii=False, indent=4))
    print(f"Evaluation report saved to: {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--version_name", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_model(
        model_path=BASE_DIR / args.model_path,
        test_file=BASE_DIR / args.test_file,
        version_name=args.version_name,
    )