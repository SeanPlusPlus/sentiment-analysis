# Sentiment Analysis with IMDb Dataset

This project is a hands-on implementation of a sentiment analysis model using the IMDb dataset and Hugging Face Transformers. The goal is to learn how to fine-tune a pre-trained model for text classification tasks.

I started with this prompt:

<img width="827" alt="image" src="https://github.com/user-attachments/assets/36b6b209-fc83-40ef-8d96-e01f4ad609fb">

... And iteratively built out these scripts as well as this corresponding document ...

It's been a pretty cool first for me - using ChatGPT to autogenerate an iterative tutorial with functioning code and documentation!

Oh! And see the `notes` dir for more general stuff that I worked on with the 🤖 while building out this tutorial. Basically, it's a collection of docs where we distill some of our chat Q&A into a little explainer.

- https://github.com/SeanPlusPlus/sentiment-analysis/blob/main/notes

---

## Project Setup

### Prerequisites

- Python 3.11.6 (managed via pyenv or installed directly).
- virtualenv for creating an isolated Python environment.
- git for version control.

---

### Setting Up Python 3.11.6

Install pyenv
- On macOS, use Homebrew to install pyenv:

```
brew install pyenv
```

Install Python 3.11.6
- Install the desired Python version using pyenv:

```
pyenv install 3.11.6
```

Clone the repo to your local machine

```
git clone git@github.com:SeanPlusPlus/sentiment-analysis.git
```

Set Python 3.11.6 as the Local Version
 - Navigate to the project directory and set the local version:

```
cd sentiment-analysis
pyenv local 3.11.6
```

Verify the Python Version
- Check the Python version in the terminal:

```
python --version
```

You should see:

```
Python 3.11.6
```

### Creating a Virtual Environment

Create the Virtual Environment
- Use virtualenv to create an isolated environment with Python 3.11.6:

```
virtualenv venv --python=$(pyenv which python)
```

Activate the Virtual Environment

```
source venv/bin/activate
```

### Installing Required Libraries

Install the Python libraries:

```
pip install -r requirements.txt
```

### Running the Project

```
python sentiment_analysis.py
```

You should see output like

```
[{'label': 'POSITIVE', 'score': 0.999...}]
```

Alright, time to add some more interesting scripts to our tutorial!

---

## Sentiment IMDB

`sentiment_imdb.py`

This script demonstrates how to perform sentiment analysis using a pre-trained model (`distilbert-base-uncased-finetuned-sst-2-english`) on the IMDb dataset. It covers the following steps:

### **What the Script Does**

1. **Loads a Pre-trained Model**:
   - Uses the Hugging Face Transformers library to load a pre-trained sentiment analysis pipeline (`distilbert-base-uncased-finetuned-sst-2-english`).

2. **Loads the IMDb Dataset**:
   - Downloads the IMDb movie review dataset, which contains thousands of labeled reviews (positive/negative sentiment).

3. **Preprocesses the Data**:
   - Tokenizes the text data using the `distilbert-base-uncased` tokenizer.
   - Prepares the data for input to the model by padding and truncating text to a fixed length.

4. **Evaluates the Model on Sample Reviews**:
   - Iterates over a few test examples and prints:
     - The review text.
     - The actual sentiment label (`POSITIVE` or `NEGATIVE`).
     - The model’s prediction and confidence score.

---

## Experiment Sentiment

`experiment_sentiment.py`

This script is designed to explore how the pre-trained sentiment analysis pipeline handles custom input text. It allows for experimentation with different types of sentences, including positive, negative, and neutral examples, to understand the model's behavior and predictions.

### **What the Script Does**

1. **Loads a Pre-trained Model**:
   - Uses the Hugging Face Transformers library to load the `distilbert-base-uncased-finetuned-sst-2-english` sentiment analysis pipeline.

2. **Processes Custom Reviews**:
   - Analyzes a predefined list of reviews, each with different tones and sentiments:
     - Positive reviews (e.g., "The movie was absolutely fantastic!").
     - Negative reviews (e.g., "I couldn't stand the acting.").
     - Neutral or mixed reviews (e.g., "It was an average movie.").

3. **Generates Predictions**:
   - Outputs the sentiment prediction (`POSITIVE` or `NEGATIVE`) along with the model's confidence score for each review.

4. **Prints Results to the Console**:
   - Each review and its corresponding prediction are printed in a readable format.

Run the script:

```
python experiment_sentiment.py
```

Example Output:

```
Review 1: The movie was absolutely fantastic! I loved every moment of it.
Prediction: [{'label': 'POSITIVE', 'score': 0.9998730421066284}]

Review 2: I couldn't stand the acting. It was so bad that I left halfway through.
Prediction: [{'label': 'NEGATIVE', 'score': 0.998234987234}]

Review 3: It was an average movie. Some parts were good, but others were forgettable.
Prediction: [{'label': 'NEGATIVE', 'score': 0.567823410987234}]
```

---

## Fine Tune Sentiment

(This will take some time to run)

`python fine_tune_sentimenty.py`

### Understanding the Fine-Tuning Process

**What is Fine-Tuning?**

Fine-tuning is the process of taking a pre-trained model (like `distilbert-base-uncased`) and adapting it to a specific task (in this case, sentiment analysis on movie reviews from the IMDb dataset).

This approach leverages the concept of transfer learning:

- The model has already been trained on a large, generic dataset to understand language structure and meaning.
- By fine-tuning, you adapt this general-purpose knowledge for your specific task using a smaller, task-specific dataset.

**What's Happening in `fine_tune_sentiment.py`?**

1. Pre-Trained Model:

- The script starts with a pre-trained distilbert-base-uncased model from Hugging Face's Transformers library.
- This model already "knows" a lot about language, thanks to prior training on massive datasets.

2. IMDb Dataset:

- The IMDb dataset contains movie reviews labeled as POSITIVE or NEGATIVE.
- The dataset is tokenized and prepared for the model.

3. Fine-Tuning:

- The pre-trained model's weights are updated based on the IMDb data.
- During training, the model learns to associate patterns in the text (e.g., "fantastic" or "terrible") with the correct sentiment labels.

4. Result:

- The fine-tuned model becomes specialized for IMDb sentiment analysis.
- It is saved locally in the `./fine_tuned_sentiment_model` directory for later use.

**Is It "Your Own Model"?**

Yes and no:

- Yes: The fine-tuned model is now uniquely yours, tailored to your task and dataset. It is distinct from the original pre-trained model because its weights have been updated.
- No: The model architecture and base knowledge come from the pre-trained `distilbert-base-uncased`, not something built from scratch.

**Why Fine-Tune Instead of Training From Scratch?**

Fine-tuning is:

1. Efficient: You only need to train for a few epochs on a small dataset because the model already understands general language.
2. Effective: Transfer learning ensures better performance with less data and compute resources.

Training from scratch would require:

- A random model with no pre-learned knowledge.
- Large datasets and extensive computational resources (e.g., GPUs/TPUs running for weeks).

**Key Takeaways**

- Fine-tuning adapts a pre-trained model to your specific task.
- The fine-tuned model is unique to you and your dataset, even though it’s based on a pre-trained architecture.
- This approach combines the power of pre-trained language models with your specific task’s requirements.
