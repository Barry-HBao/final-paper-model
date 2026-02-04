# AG News Sentiment Analysis

Project: DistilBERT (supervised) + VADER (unsupervised baseline) for sentiment analysis on AG News headlines.

Key assumptions
- AG News is a topic dataset. For this project we *create sentiment labels* by applying VADER (lexicon-based) to AG News texts to produce weak/pseudo labels used to train DistilBERT. This is a documented, defensible approach for weak supervision in a dissertation.
- Labels are: `negative`, `neutral`, `positive` mapped from VADER compound score using thresholds -0.05 and 0.05.

What is provided
- Data preprocessing and pseudo-labeling pipeline (`src/data/preprocess.py`)
- VADER wrapper (`src/unsupervised/vader_analyzer.py`)
- DistilBERT training script (`src/models/train.py`) using HuggingFace `Trainer`
- Inference wrapper (`src/models/inference.py`)
- REST API with FastAPI (`app/main.py`) exposing supervised and VADER baseline predictions
- Example script to run local predictions (`run_example.py`)

Requirements & setup
1. Create a Python 3.8+ environment on Windows.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. To prepare pseudo-labeled data (AG News + VADER):

```bash
python -m src.data.preprocess --out processed/processed_agnews.csv --max-samples 20000
```

If you already have AG News CSV files (for example `dataset/train.csv` / `dataset/test.csv`), you can point the training script directly at a CSV; `src/models/train.py` will automatically compute VADER pseudo-labels if a `vader_label` column is not present.

4. To train a DistilBERT model (quick demo mode):

```bash
python -m src.models.train --data processed/processed_agnews.csv --output models/distilbert_sentiment --epochs 1 --sample 5000
```

5. To run the API locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API Endpoints
- POST /predict  -> single text prediction
- POST /batch_predict -> list of texts
- GET /health -> basic status

Notes for dissertation
- The pipeline includes clear documentation for the weak labeling approach, evaluation metrics (accuracy, macro-F1), and code to reproduce results.
- Code is robust to CPU/GPU selection and Windows paths using `pathlib`.

Contact
- This scaffold is ready to be extended. If you'd like, I can add evaluation scripts, unit tests, or an optional Dockerfile next.
