import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


BASE_DIR = Path(__file__).resolve().parents[2]
QUANTIZED_ONNX_FILE = "model_quantized.onnx"

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


def _prediction_label_to_id(label: str) -> int:
    normalized = label.strip().lower()
    if normalized in LABEL2ID:
        return LABEL2ID[normalized]
    if normalized.startswith("label_"):
        return int(normalized.split("_", maxsplit=1)[1])
    if normalized.isdigit():
        return int(normalized)
    raise ValueError(f"Unsupported prediction label: {label}")


def predict_quantized(model_path: Path, texts: list[str], batch_size=16):
    quantized_onnx_path = model_path / QUANTIZED_ONNX_FILE
    if not quantized_onnx_path.exists():
        raise FileNotFoundError(f"Quantized ONNX file not found: {quantized_onnx_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path),
        file_name=QUANTIZED_ONNX_FILE,
        provider="CPUExecutionProvider",
    )
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_predictions = classifier(batch_texts, batch_size=batch_size)
        predictions.extend(
            _prediction_label_to_id(prediction["label"])
            for prediction in batch_predictions
        )

    return predictions


def evaluate_model(
    model_path: Path,
    test_file: Path,
    version_name: str,
    use_quantized: bool = False,
    batch_size: int = 16,
) -> Path:
    texts, true_labels = load_test_data(test_file)

    if use_quantized:
        predicted_labels = predict_quantized(model_path, texts, batch_size=batch_size)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        predicted_labels = predict(model, tokenizer, texts, batch_size=batch_size)

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
        "model_type": "quantized" if use_quantized else "standard",
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
    return output_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--version_name", required=True)
    parser.add_argument("--use_quantized", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_model(
        model_path=BASE_DIR / args.model_path,
        test_file=BASE_DIR / args.test_file,
        version_name=args.version_name,
        use_quantized=args.use_quantized,
        batch_size=args.batch_size,
    )
