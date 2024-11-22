from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load IMDb dataset
dataset = load_dataset("imdb")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def preprocess(data):
    return tokenizer(data["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(preprocess, batched=True)

# Load pre-trained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Reduce from 16 to 8
    num_train_epochs=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    gradient_accumulation_steps=2,  # Accumulates gradients for 2 steps (simulates batch size of 16)
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_sentiment_model")
