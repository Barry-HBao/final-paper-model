from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict
from src.config import VADER_POS_THRESHOLD, VADER_NEG_THRESHOLD

class VaderSentimentWrapper:
    def __init__(self, pos_threshold: float = VADER_POS_THRESHOLD, neg_threshold: float = VADER_NEG_THRESHOLD):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

    def score(self, text: str) -> Dict[str, float]:
        sc = self.analyzer.polarity_scores(text)
        return sc

    def label_from_compound(self, compound: float) -> str:
        if compound <= self.neg_threshold:
            return "negative"
        if compound >= self.pos_threshold:
            return "positive"
        return "neutral"

    def analyze(self, texts: List[str]):
        results = []
        for t in texts:
            sc = self.score(t)
            lab = self.label_from_compound(sc.get("compound", 0.0))
            results.append({"text": t, "compound": sc.get("compound", 0.0), "label": lab, "scores": sc})
        return results


if __name__ == "__main__":
    w = VaderSentimentWrapper()
    print(w.analyze(["I love this!", "This is terrible.", "It's okay."]))
