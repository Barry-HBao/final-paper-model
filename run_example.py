"""Quick demo script to show preprocessing and local inference usage.

This script is safe to run: it will not train a model automatically. It prepares processed data (small sample) and demonstrates VADER and supervised inference if a model exists.
"""
from pathlib import Path
from src.data.preprocess import prepare_agnews
from src.unsupervised.vader_analyzer import VaderSentimentWrapper
from src.models.inference import SentimentModel
from src.config import DATA_DIR, DEFAULT_OUTPUT_DIR

processed_path = DATA_DIR / "processed_agnews_demo.csv"

if not processed_path.exists():
    print("Preparing a small demo processed file (2000 samples)...")
    prepare_agnews(processed_path, max_samples=2000)
else:
    print(f"Processed demo file already exists at {processed_path}")

# VADER demo
v = VaderSentimentWrapper()
texts = ["Stocks soared after the announcement.", "The company reported terrible losses.", "It was an ordinary day."]
print("VADER results:")
for r in v.analyze(texts):
    print(r)

# Supervised demo (only if a trained model is present)
if Path(DEFAULT_OUTPUT_DIR).exists():
    print("Found trained model. Running supervised inference...")
    m = SentimentModel()
    res = m.predict(texts)
    for r in res:
        print(r)
else:
    print("No trained model found. To train a model run:")
    print("python -m src.models.train --data processed/processed_agnews_demo.csv --output models/distilbert_sentiment --epochs 1 --sample 2000")

print("If you want to run the REST API: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
