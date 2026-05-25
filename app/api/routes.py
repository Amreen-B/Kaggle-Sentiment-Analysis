from fastapi import FastAPI
from pydantic import BaseModel
from app.inference.predict import predict_sentiment

app = FastAPI(
    title = "Sentiment Analysis API",
    description = "Production ready NLP sentiment analysis system",
    version = "1.0.0"
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment Intelligence API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    result = predict_sentiment(request.text)
    return result
