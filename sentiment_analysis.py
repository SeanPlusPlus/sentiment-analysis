from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
result = sentiment("I love programming!")
print(result)
