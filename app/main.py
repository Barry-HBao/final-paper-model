from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
    try:
        model = SentimentModel()
        logger.info("Supervised model loaded successfully")
    except Exception as e:
        model = None
        model_load_error = str(e)
        logger.warning("Supervised model not available: %s", e)


@app.get("/health")
def health():
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
