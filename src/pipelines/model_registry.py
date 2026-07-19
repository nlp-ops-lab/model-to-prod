from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
PRODUCTION_MODEL_FILE = BASE_DIR / "models" / "production_model.txt"


def promote_model_to_production(model_path: str):
    full_model_path = BASE_DIR / model_path

    if not full_model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {full_model_path}")

    PRODUCTION_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(PRODUCTION_MODEL_FILE, "w", encoding="utf-8") as file:
        file.write(model_path)

    print(f"Production model updated to: {model_path}")