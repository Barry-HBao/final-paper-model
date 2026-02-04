from pathlib import Path
import os
import torch

# Base project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "processed"
MODELS_DIR = ROOT / "models"

# Default model/tokenizer to use
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
DEFAULT_OUTPUT_DIR = MODELS_DIR / "distilbert_sentiment"

# Tokenizer / model max length for news headlines
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", 128))

# Device detection (handles forced CPU for Windows if needed)
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"

def get_device():
    if FORCE_CPU:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = get_device()

# VADER thresholds (configurable via env)
VADER_POS_THRESHOLD = float(os.environ.get("VADER_POS_THRESHOLD", 0.05))
VADER_NEG_THRESHOLD = float(os.environ.get("VADER_NEG_THRESHOLD", -0.05))

# Utility
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"ROOT: {ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"Device: {DEVICE}")
