import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


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


class FinancialDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df[df["label"].isin(LABEL2ID.keys())]

    texts = df["text"].astype(str).tolist()
    labels = df["label"].map(LABEL2ID).tolist()

    return texts, labels


def fine_tune_model(
    base_model_path: Path,
    train_file: Path,
    output_model_path: Path,
    version_name: str,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    if output_model_path.exists():
        print(f"Model already exists: {output_model_path}")
        print("Skipping training to keep pipeline idempotent.")
        return

    texts, labels = load_dataset(train_file)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=3,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    train_dataset = FinancialDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=str(BASE_DIR / "artifacts" / "training" / version_name),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()

    output_model_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    print(f"Fine-tuned model saved to: {output_model_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--version_name", required=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    fine_tune_model(
        base_model_path=BASE_DIR / args.base_model,
        train_file=BASE_DIR / args.train_file,
        output_model_path=BASE_DIR / args.output_model,
        version_name=args.version_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )