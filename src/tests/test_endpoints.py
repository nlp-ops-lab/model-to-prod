# test_endpoints_debug.py
import pandas as pd
import requests
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

API_URL = "http://127.0.0.1:8000"

# ---------- بارگذاری داده‌های آزمون ----------
test_df = pd.read_csv("data/test/test.csv")  # مسیر test.csv
print("Columns in test CSV:", test_df.columns.tolist())

# اطمینان از نام ستون‌ها
text_col = 'text' if 'text' in test_df.columns else test_df.columns[0]
label_col = 'label' if 'label' in test_df.columns else test_df.columns[1]

texts = test_df[text_col].tolist()
labels = test_df[label_col].tolist()
print(f"Loaded {len(texts)} samples for testing.\n")

# ---------- تعریف تابع کمکی برای تست یک endpoint ----------
def test_endpoint(endpoint, batch_input=False, use_quantized=False):
    preds = []
    start_time = time.time()
    if batch_input:
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            try:
                resp = requests.post(f"{API_URL}/{endpoint}?use_quantized={str(use_quantized).lower()}", json=batch)
                resp.raise_for_status()
                preds.extend([r['label'] for r in resp.json()])
                print(f"Processed batch {i//batch_size + 1}")
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
    else:
        for i, text in enumerate(texts):
            try:
                resp = requests.post(f"{API_URL}/{endpoint}", json={"text": text})
                resp.raise_for_status()
                preds.append(resp.json()['label'])
                if (i+1) % 50 == 0 or i == len(texts)-1:
                    print(f"Processed {i+1}/{len(texts)} samples")
            except Exception as e:
                print(f"Error in sample {i+1}: {e}")
    elapsed = time.time() - start_time
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average='macro'),
        "precision": precision_score(labels, preds, average='macro'),
        "recall": recall_score(labels, preds, average='macro'),
        "time": elapsed,
        "preds": preds
    }
    return metrics

# ---------- اجرای تست هر سه endpoint ----------
endpoints = [
    ("predict", False, False),
    ("predict-quantized", False, True),
    ("predict-batch", True, True)
]

results = {}
for ep, batch_input, use_quantized in endpoints:
    print(f"\n=== Testing endpoint: {ep} ===")
    results[ep] = test_endpoint(ep, batch_input=batch_input, use_quantized=use_quantized)
    print(f"Time taken: {results[ep]['time']:.2f} seconds")
    print(f"Accuracy: {results[ep]['accuracy']:.4f}, F1: {results[ep]['f1']:.4f}\n")

# ---------- خلاصه نتایج ----------
print("\n=== Summary of all endpoints ===")
for key, val in results.items():
    print(f"{key}: Accuracy={val['accuracy']:.4f}, F1={val['f1']:.4f}, Time={val['time']:.2f}s")