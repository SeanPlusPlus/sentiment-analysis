from transformers import pipeline

# Load the pre-trained sentiment analysis pipeline
sentiment = pipeline("sentiment-analysis")

# Define a list of custom reviews to test
reviews = [
    "The movie was absolutely fantastic! I loved every moment of it.",
    "I couldn't stand the acting. It was so bad that I left halfway through.",
    "It was an average movie. Some parts were good, but others were forgettable.",
    "The plot was intriguing, and the performances were top-notch!",
    "I wouldn't recommend this movie to anyone. It was a waste of time.",
]

# Analyze each review
for i, review in enumerate(reviews):
    result = sentiment(review)
    print(f"Review {i + 1}: {review}")
    print(f"Prediction: {result}\n")
