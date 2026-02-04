import argparse
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List
import re
import logging

from src.config import DATA_DIR, VADER_POS_THRESHOLD, VADER_NEG_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analyzer = SentimentIntensityAnalyzer()


def clean_text(t: str) -> str:
    if t is None:
        return ""
    t = str(t)
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t


def vader_label_from_compound(c: float) -> int:
    """Map compound score to label: 0=negative, 1=neutral, 2=positive"""
    if c <= VADER_NEG_THRESHOLD:
        return 0
    if c >= VADER_POS_THRESHOLD:
        return 2
    return 1


def compute_vader(texts: List[str]):
    compounds = []
    labels = []
    for t in texts:
        sc = analyzer.polarity_scores(t)
        c = sc.get("compound", 0.0)
        compounds.append(c)
        labels.append(vader_label_from_compound(c))
    return compounds, labels


def prepare_agnews(output_path: Path, max_samples: int = None):
    """Load AG News, clean text, compute VADER pseudo-labels, and save CSV."""
    logger.info("Loading AG News dataset from HuggingFace datasets...")
    try:
        ds = load_dataset("ag_news")
    except Exception as e:
        logger.error("Failed to load AG News dataset: %s", e)
        raise

    # AG News has 'train' and 'test' splits with 'text' and 'label' (topic label). We'll use 'text'.
    combined = []
    for split in ["train", "test"]:
        for ex in ds[split]:
            combined.append(ex["text"])
    if max_samples is not None and max_samples > 0:
        combined = combined[:max_samples]

    logger.info("Cleaning texts... (first pass)")
    cleaned = [clean_text(t) for t in tqdm(combined)]

    logger.info("Computing VADER scores and pseudo-labels...")
    compounds, labels = compute_vader(cleaned)

    df = pd.DataFrame({"text": cleaned, "vader_compound": compounds, "vader_label": labels})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed dataset to %s (rows=%d)", output_path, len(df))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(DATA_DIR / "processed_agnews.csv"), help="Output CSV path")
    parser.add_argument("--max-samples", type=int, default=20000, help="Limit number of samples for quick runs (0=all)")
    args = parser.parse_args()

    max_samples = args.max_samples if args.max_samples > 0 else None
    path = prepare_agnews(args.out, max_samples=max_samples)
    print(f"Processed data saved to {path}")
