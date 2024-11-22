# Sentiment Analysis with IMDb Dataset

This project is a hands-on implementation of a sentiment analysis model using the IMDb dataset and Hugging Face Transformers. The goal is to learn how to fine-tune a pre-trained model for text classification tasks.

I started with this prompt:

<img width="827" alt="image" src="https://github.com/user-attachments/assets/36b6b209-fc83-40ef-8d96-e01f4ad609fb">

... And iteratively built out these scripts as well as this corresponding document. 

It's been a pretty cool first for me - using ChatGPT to autogenerate an iterative tutorial with functioning code and documentation!

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

---

## `sentiment_imdb.py`

```
python sentiment_imdb.py
```

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

## `experiment_sentiment.py`

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
