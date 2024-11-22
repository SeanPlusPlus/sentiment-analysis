from transformers import pipeline

# Load the fine-tuned model
sentiment_pipeline = pipeline("sentiment-analysis", model="./fine_tuned_sentiment_model")

# Test on a few examples
test_reviews = [
    "This movie was amazing! The performances were incredible.",
    "It was boring and way too long. I almost fell asleep.",
    "The plot was interesting, but the execution was poor."
]

for review in test_reviews:
    print(f"Review: {review}")
    print(f"Prediction: {sentiment_pipeline(review)}\n")
