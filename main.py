# main.py
#import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import random
import joblib

app = FastAPI(
    title="Sentiment Analysis API",
    description="A FastAPI backend for sentiment prediction using a trained model.",
    version="1.0.0"
)

try:
    with open("model.joblib", "rb") as f:
        model = joblib.load(f)
except FileNotFoundError:
    raise RuntimeError("not found.")

try:
    df = pd.read_csv("IMDB Dataset.csv")
except FileNotFoundError:
    raise RuntimeError("IMDB dataset not found.")

class TextRequest(BaseModel):
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    try:
        prediction = model.predict([request.text])[0]
        return {"sentiment": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_proba")
def predict_sentiment_with_proba(request: TextRequest):
    try:
        sentiment = model.predict([request.text])[0]
        probas = model.predict_proba([request.text])[0]
        # Get index of predicted class
        class_index = list(model.classes_).index(sentiment)
        confidence = float(probas[class_index])
        return {"sentiment": sentiment, "probability": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/example")
def get_random_example():
    try:
        random_row = df.sample(n=1).iloc[0]
        review_text = random_row.get("review", "")
        return {"review": review_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
