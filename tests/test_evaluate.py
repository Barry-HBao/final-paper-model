import pandas as pd
import pytest

from src.models import evaluate


def test_inspect_labels(tmp_path, capsys):
    df = pd.DataFrame({"text": ["hello"], "label": [3]})
    path = tmp_path / "temp.csv"
    df.to_csv(path, index=False)

    # should print dtype information and the unique value 3
    evaluate.inspect_labels(str(path))
    captured = capsys.readouterr()
    assert "dtypes" in captured.out
    assert "unique labels" in captured.out
    assert "3" in captured.out


def test_normalization_clips_and_maps():
    # out-of-range integer should be clipped to 2
    raw = [0, 1, 3, -5]
    normalized = evaluate._normalize_truth_labels(raw)
    # mapping: 0->negative,1->neutral,2->positive
    assert normalized == ["negative", "neutral", "positive", "negative"]

    # string that are valid should pass through
    raw = ["negative", "positive"]
    assert evaluate._normalize_truth_labels(raw) == raw

    # bad strings return None
    assert evaluate._normalize_truth_labels(["foo"]) is None
    # mixed types return None
    assert evaluate._normalize_truth_labels([0, "neutral"]) is None


def test_evaluate_with_out_of_range_labels(tmp_path, monkeypatch, capsys):
    df = pd.DataFrame({"text": ["a", "b"], "label": [0, 3]})
    test_file = tmp_path / "test.csv"
    df.to_csv(test_file, index=False)

    # patch the models to avoid heavy computation
    class DummySup:
        def __init__(self, model_dir):
            pass

        def predict(self, texts, batch_size=32):
            return [{"label": "negative"} for _ in texts]

    class DummyVader:
        def analyze(self, texts):
            return [{"label": "negative"} for _ in texts]

    monkeypatch.setattr(evaluate, "SentimentModel", DummySup)
    monkeypatch.setattr(evaluate, "VaderSentimentWrapper", DummyVader)

    evaluate.evaluate("fake_dir", str(test_file))
    out = capsys.readouterr().out
    assert "clipping to 0..2" in out
    # metrics should appear even though one label was out of range
    assert "Accuracy:" in out
    assert "Macro F1:" in out
