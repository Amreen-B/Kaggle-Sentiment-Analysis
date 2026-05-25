from app.inference.predict import predict_sentiment

sample_text = "This AI engineering project is amazing"

result = predict_sentiment(sample_text)

print(result)