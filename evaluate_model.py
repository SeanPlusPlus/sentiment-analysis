from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import load_dataset

# Load the fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_sentiment_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_sentiment_model")

# Create a sentiment analysis pipeline
sentiment_analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Load and shuffle the dataset
dataset = load_dataset("imdb")

# Use a seed for reproducibility
shuffled_dataset = dataset["test"].shuffle(seed=42, load_from_cache_file=False)


# Get a few examples from the test set
test_samples = shuffled_dataset.select(range(12))  # Adjust range as needed

for i, sample in enumerate(test_samples):
    print(f"Sample {i + 1}: Label={sample['label']} Text={sample['text'][:100]}...")

# Run predictions on test samples
for i, sample in enumerate(test_samples):
    text = sample["text"]
    true_label = "POSITIVE" if sample["label"] == 1 else "NEGATIVE"
    prediction = sentiment_analyzer(text)
    predicted_label = prediction[0]["label"]
    confidence = prediction[0]["score"]
    print(f"Review {i + 1}: {text}")
    print(f"True Label: {true_label}")
    print(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    print("-" * 50)
