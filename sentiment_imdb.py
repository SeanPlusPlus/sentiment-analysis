from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import pipeline

# Load a pre-trained model fine-tuned on sentiment classification tasks
sentiment = pipeline("sentiment-analysis")

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Access the train and test splits
train_data = dataset['train']
test_data = dataset['test']

# Load the tokenizer for distilbert
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize a batch of data
def preprocess(data):
    return tokenizer(data['text'], padding="max_length", truncation=True)

# Apply preprocessing
train_data = train_data.map(preprocess, batched=True)
test_data = test_data.map(preprocess, batched=True)

# Evaluate on a few examples
for i in range(5):
    print(f"Review: {test_data[i]['text']}")
    print(f"Label: {test_data[i]['label']}")
    print(f"Prediction: {sentiment(test_data[i]['text'])}\n")
