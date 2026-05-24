# NLP MLOps Pipeline

An end-to-end, reproducible MLOps pipeline that takes raw NLP data all the way to a production-ready model, using industry-standard tools.

---

## 🔍 Project Overview

This project implements the full ML lifecycle for an NLP task in a team of 4:

- Data ingestion from multiple sources (CSV/JSON/API/DB)
- Data anonymization / pseudonymization for PII
- Reproducible preprocessing and ETL pipelines
- Model training / fine-tuning and evaluation
- Experiment tracking and model versioning
- Preparation for deployment via an API service

The focus is not only on building a model, but on making the **entire process reproducible, traceable, and maintainable**.

---

## 🧰 Tech Stack

- **Language & ML:** Python, NLP (e.g. pretrained models + fine-tuning)
- **Version Control:** Git (code), DVC (data & model artifacts)
- **Reproducibility:** Docker
- **Orchestration:** Airflow or Prefect
- **Experiment Tracking & Registry:** MLflow
- **Serving (later):** FastAPI (for model inference API)

---

## 📁 Project Structure

```text
.
├── src/                # Source code (ingestion, preprocessing, training, inference)
├── data/               # Data managed by DVC (raw & processed)
│   ├── raw/
│   └── processed/
├── models/             # Trained model artifacts (also via DVC)
├── pipelines/          # Workflow definitions (Airflow DAGs / Prefect flows)
├── notebooks/          # Optional: EDA and experimentation
├── mlruns/             # Local MLflow runs (ignored in Git)
├── Dockerfile          # Reproducible environment definition
├── dvc.yaml            # Data & model pipeline definition for DVC
├── requirements.txt    # Python dependencies
└── README.md
```

> Note: `data/` and `models/` are **versioned with DVC**, not Git, to handle large files and ensure reproducibility.

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <REPO_URL>
   cd <REPO_NAME>
   ```

2. **Build Docker image**
   ```bash
   docker build -t nlp-mlops-pipeline .
   ```

3. **Restore data and models (if DVC remote is configured)**
   ```bash
   dvc pull
   ```

4. **Run the basic pipeline (example)**
   - Via Python/CLI:
     ```bash
     python -m src.data.fetch_data
     python -m src.data.preprocess
     python -m src.models.train
     ```
   - Or via orchestration tool (Airflow/Prefect), as configured in `pipelines/`.

---

## 🧪 Experiments & Model Tracking

We use **MLflow** to log:

- Parameters (hyperparameters, data versions)
- Metrics (accuracy/F1/etc.)
- Artifacts (models, plots)
- Model versions (via MLflow Model Registry)

This enables comparing different runs and promoting models from *candidate* → *staging* → *production*.

---

## 👥 Team

Developed as a collaborative MLOps exercise by a 4-person team, focusing on real-world practices in reproducible ML, data-centric AI, and production-ready NLP.

```
