from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_sentiment_model")

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load IMDb dataset
dataset = load_dataset("imdb")

# Get a few examples from the test set
test_samples = dataset["test"].select(range(5))  # Adjust range as needed

# Run predictions on test samples
for i, sample in enumerate(test_samples):
    text = sample["text"]
    prediction = sentiment_analyzer(text)
    print(f"Review {i + 1}: {text}")
    print(f"Prediction: {prediction}")
    print("-" * 50)
