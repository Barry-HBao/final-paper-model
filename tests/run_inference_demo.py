from src.models.inference import SentimentModel

if __name__ == '__main__':
    m = SentimentModel(model_dir='models/distilbert_sentiment_demo')
    texts = [
        "Stocks soared after the earnings report.",
        "The product failed and customers are angry.",
    ]
    res = m.predict(texts)
    for r in res:
        print(r)
