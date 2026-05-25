from transformers import pipeline

# Load pretrained sentiment analysis pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def predict_sentiment(text: str):
    """
    Predict sentiment for input text.
    """

    result = classifier(text)[0]

    return {
        "text": text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4)
    }