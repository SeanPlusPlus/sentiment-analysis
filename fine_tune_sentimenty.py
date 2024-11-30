from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from collections import Counter

# Step 1: Load IMDb dataset
print("Loading dataset...")
dataset = load_dataset("imdb")

# Step 2: Balance the training subset
print("Balancing the training data...")
shuffled_train = dataset["train"].shuffle(seed=42)  # Shuffle to mix labels

# Select 250 positive and 250 negative samples
positive_samples = [example for example in shuffled_train if example["label"] == 1][:250]
negative_samples = [example for example in shuffled_train if example["label"] == 0][:250]

balanced_train = positive_samples + negative_samples

# Print the label distribution for verification
label_counts = Counter([example["label"] for example in balanced_train])
print(f"Balanced training label distribution: {label_counts}")

# Step 3: Tokenize the data
print("Tokenizing the dataset...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)

# Tokenize the balanced training data
tokenized_train = {
    "text": [example["text"] for example in balanced_train],
    "label": [example["label"] for example in balanced_train],
}
tokenized_train = dataset["train"].from_dict(tokenized_train).map(preprocess, batched=True)

# Tokenize the test dataset
tokenized_test = dataset["test"].map(preprocess, batched=True)

# Step 4: Load the pre-trained model
print("Loading pre-trained model...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 5: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,  # Increase the number of epochs for better learning
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    gradient_accumulation_steps=2,
)

# Step 6: Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Step 7: Fine-tune the model
print("Starting training...")
trainer.train()

# Step 8: Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./fine_tuned_sentiment_model")
tokenizer.save_pretrained("./fine_tuned_sentiment_model")

print("Fine-tuning complete!")
