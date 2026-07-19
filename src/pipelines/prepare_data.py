from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = BASE_DIR / "data" / "raw" / "Sentences_75Agree.txt"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
BATCHES_DIR = BASE_DIR / "data" / "batches"
TEST_DIR = BASE_DIR / "data" / "test"

PROCESSED_FILE = PROCESSED_DIR / "financial_phrasebank_75.csv"


def read_financial_phrasebank(file_path: Path) -> pd.DataFrame:
    rows = []

    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if "@" not in line:
                continue

            sentence, label = line.rsplit("@", 1)

            rows.append({
                "text": sentence.strip(),
                "label": label.strip()
            })

    return pd.DataFrame(rows)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = df.drop_duplicates(subset=["text"])
    df = df[df["label"].isin(["positive", "negative", "neutral"])]
    df = df[df["text"].str.len() > 5]

    return df.reset_index(drop=True)


def save_main_csv(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8")


def split_into_batches(df: pd.DataFrame):
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        stratify=df["label"]
    )

    batch_0, remaining = train_test_split(
        train_df,
        test_size=0.3333,
        random_state=42,
        stratify=train_df["label"]
    )

    batch_1, remaining = train_test_split(
        remaining,
        test_size=0.6666,
        random_state=42,
        stratify=remaining["label"]
    )

    batch_2, batch_3 = train_test_split(
        remaining,
        test_size=0.50,
        random_state=42,
        stratify=remaining["label"]
    )

    batch_0.to_csv(BATCHES_DIR / "batch_0.csv", index=False, encoding="utf-8")
    batch_1.to_csv(BATCHES_DIR / "batch_1.csv", index=False, encoding="utf-8")
    batch_2.to_csv(BATCHES_DIR / "batch_2.csv", index=False, encoding="utf-8")
    batch_3.to_csv(BATCHES_DIR / "batch_3.csv", index=False, encoding="utf-8")
    test_df.to_csv(TEST_DIR / "test.csv", index=False, encoding="utf-8")

    print("Dataset split completed:")
    print(f"batch_0: {len(batch_0)}")
    print(f"batch_1: {len(batch_1)}")
    print(f"batch_2: {len(batch_2)}")
    print(f"batch_3: {len(batch_3)}")
    print(f"test: {len(test_df)}")


def prepare_data():
    df = read_financial_phrasebank(RAW_FILE)
    df = clean_dataset(df)

    save_main_csv(df)
    split_into_batches(df)

    print(f"Processed CSV saved to: {PROCESSED_FILE}")


if __name__ == "__main__":
    prepare_data()