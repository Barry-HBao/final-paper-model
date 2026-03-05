import pandas as pd
from src.unsupervised.vader_analyzer import VaderSentimentWrapper
from src.models.evaluate import evaluate

# prepare small dataset with pseudo labels

df = pd.read_csv('dataset/train.csv').head(200)
vader = VaderSentimentWrapper()
v = vader.analyze(df['text'].astype(str).tolist())
labels = [0 if x['label']=='negative' else 2 if x['label']=='positive' else 1 for x in v]
df['label'] = labels

temp = 'dataset/temp_eval.csv'
df.to_csv(temp, index=False)
print('Saved evaluation file with', len(df), 'rows')

# perform evaluation
evaluate('models/distilbert_sentiment_demo', temp)
