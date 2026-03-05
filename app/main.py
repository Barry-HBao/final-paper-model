from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from src.unsupervised.vader_analyzer import VaderSentimentWrapper
from src.models.inference import SentimentModel
from src.config import DEVICE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AG News Sentiment API", version="0.1")

vader = VaderSentimentWrapper()
model = None
model_load_error: Optional[str] = None

class PredictRequest(BaseModel):
    text: str

class BatchPredictRequest(BaseModel):
    texts: List[str]

class SupervisedResult(BaseModel):
    label: str
    confidence: float
    probabilities: dict

class VaderResult(BaseModel):
    label: str
    compound: float
    scores: dict

class PredictResponse(BaseModel):
    supervised: Optional[SupervisedResult]
    vader: VaderResult


@app.on_event("startup")
async def startup_event():
    global model, model_load_error
    # attempt to load from config default, falling back to any folder inside models/
    from src.config import DEFAULT_OUTPUT_DIR, MODELS_DIR
    candidates = [Path(DEFAULT_OUTPUT_DIR)]
    # add any subdirectory in MODELS_DIR that contains config.json
    for sub in Path(MODELS_DIR).iterdir():
        if sub.is_dir() and (sub / "config.json").exists():
            candidates.append(sub)
    loaded = False
    for cand in candidates:
        try:
            model = SentimentModel(model_dir=str(cand))
            model_load_error = None
            logger.info(f"Supervised model loaded from {cand}")
            loaded = True
            break
        except Exception as e:
            logger.debug(f"Failed to load model from {cand}: {e}")
    if not loaded:
        model = None
        model_load_error = "no valid model directory found"
        logger.warning("Supervised model not available: %s", model_load_error)


@app.get("/health")
def health():
    """Return service status and try to load a supervised model if absent."""
    global model, model_load_error
    if model is None:
        try:
            model = SentimentModel()
            model_load_error = None
            logger.info("Model automatically loaded in health check")
        except Exception as e:
            model_load_error = str(e)
    return {"status": "ok", "device": str(DEVICE), "model_loaded": model is not None, "model_error": model_load_error}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text
    vader_res = vader.analyze([text])[0]
    vad_res = VaderResult(label=vader_res["label"], compound=vader_res["compound"], scores=vader_res["scores"])

    sup_res = None
    if model is not None:
        preds = model.predict([text])
        p = preds[0]
        sup_res = SupervisedResult(label=p["label"], confidence=p["confidence"], probabilities=p["probabilities"])

    return PredictResponse(supervised=sup_res, vader=vad_res)




class LoadModelRequest(BaseModel):
    path: str


@app.post("/load_model")
def load_model(req: LoadModelRequest):
    """Load a supervised model from given directory during runtime."""
    global model, model_load_error
    try:
        model = SentimentModel(model_dir=req.path)
        model_load_error = None
        return {"status": "loaded", "model_dir": req.path}
    except Exception as e:
        model = None
        model_load_error = str(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/unload_model")
def unload_model():
    """Unload any currently loaded supervised model."""
    global model
    model = None
    return {"status": "unloaded"}


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    texts = req.texts
    vader_batch = vader.analyze(texts)
    vader_out = [{"text": v["text"], "label": v["label"], "compound": v["compound"], "scores": v["scores"]} for v in vader_batch]

    supervised_out = None
    if model is not None:
        preds = model.predict(texts)
        supervised_out = preds

    return {"vader": vader_out, "supervised": supervised_out}
