# Sentiment Analysis with IMDb Dataset

This project is a hands-on implementation of a sentiment analysis model using the IMDb dataset and Hugging Face Transformers. The goal is to learn how to fine-tune a pre-trained model for text classification tasks.

## Project Setup

### Prerequisites

- Python 3.11.6 (managed via pyenv or installed directly).
- virtualenv for creating an isolated Python environment.
- git for version control.

---

### Setting Up Python 3.11.6

1. Install pyenv
- On macOS, use Homebrew to install pyenv:

```
brew install pyenv
```

2. Install Python 3.11.6
- Install the desired Python version using pyenv:

```
pyenv install 3.11.6
```

3. Set Python 3.11.6 as the Local Version
 - Navigate to the project directory and set the local version:

```
cd sentiment-analysis
pyenv local 3.11.6
```

4. Verify the Python Version
- Check the Python version in the terminal:

```
python --version
```

You should see:

```
Python 3.11.6
```

### Creating a Virtual Environment

1. Create the Virtual Environment
- Use virtualenv to create an isolated environment with Python 3.11.6:

```
virtualenv venv --python=$(pyenv which python)
```

2. Activate the Virtual Environment

```
source venv/bin/activate
```

### Installing Required Libraries

Install the necessary Python libraries:

```
pip install transformers datasets torch
```

Save the dependencies to requirements.txt:

```
pip freeze > requirements.txt
```

### Running the Project

```
python sentiment_analysis.py
```

And should see output like

```
[{'label': 'POSITIVE', 'score': 0.999...}]
```