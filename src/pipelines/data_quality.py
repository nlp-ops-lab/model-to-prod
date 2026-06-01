from pathlib import Path
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_FILE = BASE_DIR / "data" / "processed" / "financial_phrasebank_75.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORT_FILE = ARTIFACTS_DIR / "data_report.json"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE)


def generate_data_quality_report(df: pd.DataFrame) -> dict:
    label_counts = df["label"].value_counts().to_dict()

    duplicate_count = int(df.duplicated(subset=["text"]).sum())

    text_lengths = df["text"].astype(str).str.len()

    report = {
        "dataset_name": "Financial PhraseBank - Sentences_75Agree",
        "total_samples": int(len(df)),
        "label_distribution": {
            "positive": int(label_counts.get("positive", 0)),
            "negative": int(label_counts.get("negative", 0)),
            "neutral": int(label_counts.get("neutral", 0)),
        },
        "duplicate_count": duplicate_count,
        "average_text_length": float(text_lengths.mean()),
        "max_text_length": int(text_lengths.max()),
        "min_text_length": int(text_lengths.min()),
        "missing_values": {
            "text": int(df["text"].isna().sum()),
            "label": int(df["label"].isna().sum()),
        },
        "invalid_labels": int(
            (~df["label"].isin(["positive", "negative", "neutral"])).sum()
        ),
    }

    return report


def save_report(report: dict):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=4)


def run_data_quality_check():
    df = load_data()
    report = generate_data_quality_report(df)
    save_report(report)

    print("Data quality report generated successfully.")
    print(f"Report path: {REPORT_FILE}")
    print(json.dumps(report, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    run_data_quality_check()