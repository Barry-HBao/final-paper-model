from fastapi.testclient import TestClient
import app.main as am


class StubModel:
    def __init__(self, model_dir=None):
        if model_dir == "bad":
            raise FileNotFoundError("bad model")
        self.model_dir = model_dir

    def predict(self, texts, batch_size=16):
        return [{"text": t, "label": "neutral", "confidence": 0.5, "probabilities": {}} for t in texts]


def test_health_and_load(monkeypatch):
    # monkeypatch the SentimentModel used by the app
    monkeypatch.setattr(am, "SentimentModel", StubModel)

    client = TestClient(am.app)
    # startup_event should run on first request
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    # health endpoint attempts to auto-load; stub always succeeds
    assert data["model_loaded"] is True

    # load a model successfully (should simply overwrite)
    r = client.post("/load_model", json={"path": "somepath"})
    assert r.status_code == 200
    assert r.json()["status"] == "loaded"

    # prediction should now include supervised
    r = client.post("/predict", json={"text": "hello"})
    assert r.status_code == 200
    out = r.json()
    assert out["supervised"]["label"] == "neutral"
    assert out["vader"]["label"] in ["negative", "neutral", "positive"]

    # unloading model
    r = client.post("/unload_model")
    assert r.status_code == 200
    assert r.json()["status"] == "unloaded"

    # health will attempt to reload again (lazy autoload) when model is None
    r = client.get("/health")
    assert r.json()["model_loaded"] is True


def test_load_failure(monkeypatch):
    monkeypatch.setattr(am, "SentimentModel", StubModel)
    client = TestClient(am.app)
    r = client.post("/load_model", json={"path": "bad"})
    assert r.status_code == 400
    assert "bad model" in r.json()["detail"]


def test_health_autoload(monkeypatch):
    # model should be loaded lazily by health endpoint when not present initially
    monkeypatch.setattr(am, "SentimentModel", StubModel)
    client = TestClient(am.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True
    # subsequent predict should return supervised results
    r = client.post("/predict", json={"text": "hi"})
    assert r.json()["supervised"]["label"] == "neutral"
