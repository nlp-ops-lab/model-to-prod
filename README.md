# 🚀 Financial Sentiment Analysis MLOps Pipeline

An end-to-end, reproducible MLOps pipeline for financial sentiment analysis using **FinBERT**, built with industry-standard tools for experiment tracking, model versioning, workflow orchestration, and continuous model updates.

---

## 🔍 Project Overview

This project demonstrates the complete machine learning lifecycle for a Financial NLP application, from raw data preparation to automated model updates in a production-oriented environment.

The project focuses on building a **reproducible, traceable, maintainable, and continuously updatable ML system** rather than only training a model.

### Key Features

✅ Financial Sentiment Analysis using FinBERT

✅ Reproducible Data Processing Pipeline

✅ Data-Centric AI Workflow

✅ Stateful Fine-Tuning Strategy

✅ Automated Training Pipelines with Prefect

✅ Experiment Tracking with MLflow

✅ Model Versioning using DVC

✅ Incremental Model Updates

✅ Idempotent Pipeline Execution

---

## 🧰 Tech Stack

### 🤖 Machine Learning & NLP

* Python
* PyTorch
* Hugging Face Transformers
* FinBERT

### 🔄 MLOps & Reproducibility

* Git (Code Versioning)
* DVC (Model Versioning)
* Docker
* Prefect (Workflow Orchestration)
* MLflow (Experiment Tracking)

### 🌐 Serving Layer

* FastAPI (Inference API)

---

## 🏗️ System Architecture

```text
Raw Dataset
     │
     ▼
Data Preparation
     │
     ▼
Data Quality Validation
     │
     ▼
Batch Generation
     │
     ▼
Stateful Fine-Tuning
     │
     ▼
Model Evaluation
     │
     ▼
MLflow Tracking
     │
     ▼
DVC Versioning
     │
     ▼
Production Model
```

---

## 📁 Project Structure

```text
.
├── src/
│   ├── api/
│   │
│   ├── services/
│   │   └── finbert_service.py
│   │
│   └── pipelines/
│       ├── sentiment_flow.py
│       ├── prepare_data.py
│       ├── data_quality.py
│       ├── train_finetune.py
│       ├── evaluate.py
│       ├── mlflow_tracking.py
│       ├── week2_training_pipeline.py
│       └── incremental_update_pipeline.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── batches/
│   └── test/
│
├── models/
│   ├── FinbertConfiguration/
│   ├── finbert_v1/
│   ├── finbert_v2/
│   ├── finbert_v3/
│   └── finbert_v4/
│
├── artifacts/
│   ├── evaluation/
│   └── training/
│
├── models.dvc
├── dvc.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

The project uses the **Financial PhraseBank** dataset.

Selected version:

```text
Sentences_75Agree.txt
```

The dataset contains financial news statements manually labeled by financial experts as:

```text
positive
neutral
negative
```

To simulate real-world production environments, the dataset is divided into multiple batches representing incoming data over time.

---

## 🧹 Data-Centric AI

Before training, the dataset undergoes quality validation and preprocessing.

Implemented checks include:

* Duplicate Detection
* Label Distribution Analysis
* Missing Value Detection
* Invalid Label Detection
* Text Length Statistics

Generated artifact:

```text
artifacts/data_report.json
```

This follows the Data-Centric AI principle of improving data quality before increasing model complexity.

---

## 🧠 Stateful Fine-Tuning Strategy

The project adopts a **Stateful Training** approach.

Instead of retraining from scratch, every new model version starts from the previous model version.

```text
FinBERT Base
     │
     ▼
batch_0
     ▼
finbert_v1
     │
     ▼
batch_1
     ▼
finbert_v2
     │
     ▼
batch_2
     ▼
finbert_v3
     │
     ▼
batch_3
     ▼
finbert_v4
```

This closely resembles real-world production model updates.

---

## ⚙️ Automated Training Pipeline

A Prefect-based pipeline orchestrates the entire lifecycle.

Pipeline:

```text
week2_training_pipeline.py
```

Workflow:

```text
Train
   ▼
Evaluate
   ▼
Log to MLflow
   ▼
Version Model
```

Execution:

```bash
python -m src.pipelines.week2_training_pipeline
```

Features:

* Fully Automated
* Reproducible
* Stateful Updates
* MLflow Integration
* Idempotent Execution

---

## 🔄 Incremental Model Updates

For production scenarioswhere new data arrives periodically, a dedicated incremental pipeline is provided.

Pipeline:

```text
incremental_update_pipeline.py
```

Workflow:

```text
Latest Model
      +
 New Dataset
      ▼
 Fine-Tuning
      ▼
 Evaluation
      ▼
 MLflow Tracking
```

Example:

```bash
python -m src.pipelines.incremental_update_pipeline \
  --base_model models/finbert_v4 \
  --new_data data/incoming/new_batch.csv \
  --output_model models/finbert_v5 \
  --version_name finbert_v5
```

This enables continuous learning without retraining from scratch.

---

## 📈 Experiment Tracking with MLflow

All training runs are tracked using MLflow.

Tracked Information:

### Parameters

* Learning Rate
* Batch Size
* Epochs
* Model Version

### Metrics

* Accuracy
* Precision
* Recall
* F1 Score

### Artifacts

* Trained Models
* Evaluation Reports
* Confusion Matrices

Start MLflow UI:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

---

## 📦 Model Versioning with DVC

Model artifacts are managed through DVC.

Track model versions:

```bash
dvc add models
```

Commit metadata:

```bash
git add models.dvc
git commit -m "Track model versions with DVC"
```

Benefits:

* Model Reproducibility
* Version History
* Lightweight Git Repository
* Artifact Management

---

## 🔐 Idempotent Design

The training pipeline is designed to be idempotent.

Example:

```text
If model version already exists:
    Skip training
```

This prevents:

* Duplicate Model Training
* Accidental Overwrites
* Unnecessary Compute Consumption

---

## 🚀 Quick Start

### Clone Repository

```bash
git clone <REPOSITORY_URL>
cd model-to-prod
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Data Preparation

```bash
python -m src.pipelines.prepare_data
```

### Run Data Quality Checks

```bash
python -m src.pipelines.data_quality
```

### Run Full Training Pipeline

```bash
python -m src.pipelines.week2_training_pipeline
```

### Start MLflow UI

```bash
mlflow ui
```

---

## 📌 Current Achievements

* Financial PhraseBank Integration
* FinBERT Fine-Tuning
* Data-Centric AI Workflow
* Automated Prefect Pipelines
* Stateful Model Updates
* MLflow Experiment Tracking
* DVC Model Versioning
* Incremental Retraining Pipeline
* Production-Oriented Architecture

---

## 👥 Team

Developed as a collaborative MLOps project focusing on:

* Reproducible Machine Learning
* Financial NLP
* Data-Centric AI
* Stateful Training
* Experiment Tracking
* Production-Ready MLOps Practices

🚀 From Data to Production — End-to-End MLOps for Financial NLP.
