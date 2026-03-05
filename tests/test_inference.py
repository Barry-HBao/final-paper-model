"""Basic tests for the inference wrapper.

These tests are lightweight: they create a temporary minimal model (from the pre-trained base) and ensure `SentimentModel` can load it and predict.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

# torch is an optional heavy dependency; skip tests if not available
try:
    import torch
except ImportError:
    import pytest

    pytest.skip("torch not installed, skipping inference tests", allow_module_level=True)

from src.models.inference import SentimentModel


def test_sentiment_model_loads_and_predicts(monkeypatch):
    """Offline test: stub the transformers constructors to avoid network calls to HuggingFace hub."""
    tmpdir = Path(tempfile.mkdtemp())
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Stub tokenizer that returns simple torch tensors
    class StubTokenizer:
        def __call__(self, texts, truncation=True, padding=True, max_length=128, return_tensors="pt"):
            batch = len(texts)
            input_ids = torch.ones((batch, 8), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    # Stub model that returns deterministic logits
    class StubModel:
        def to(self, device):
            return self

        def eval(self):
            return None

        def __call__(self, **enc):
            batch = enc["input_ids"].shape[0]
            logits = torch.tensor([[0.1, 0.8, 0.1]] * batch)
            return SimpleNamespace(logits=logits)

    # Monkeypatch the symbols imported into the inference module to avoid touching HuggingFace internals
    import src.models.inference as inf_mod
    monkeypatch.setattr(inf_mod, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: StubTokenizer()))
    monkeypatch.setattr(inf_mod, "AutoModelForSequenceClassification", SimpleNamespace(from_pretrained=lambda *a, **k: StubModel()))

    sm = SentimentModel(model_dir=str(tmpdir))
    preds = sm.predict(["This is great!", "This is bad."])

    assert isinstance(preds, list)
    assert len(preds) == 2
    for p in preds:
        assert "label" in p and "confidence" in p and "probabilities" in p
        assert 0.0 <= p["confidence"] <= 1.0
        assert set(p["probabilities"].keys()) == {"negative", "neutral", "positive"}
