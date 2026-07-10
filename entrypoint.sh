#!/bin/bash
set -e

echo "=== Step 1: Running training pipeline ==="
python -m src.pipelines.week2_training_pipeline

echo "=== Step 2: Quantizing finbert_v4 ==="
python -m src.pipelines.quantized_model_pipeline \
    --source_model models/finbert_v4 \
    --output_model models/finbert_v4_quantized \
    --version_name finbert_v4_quantized \
    --test_file data/test/test.csv

echo "=== Step 3: Starting API ==="
exec uvicorn main:app --host 0.0.0.0 --port 8000
