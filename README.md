# Sentiment Analysis with IMDb Dataset

This project is a hands-on implementation of a sentiment analysis model using the IMDb dataset and Hugging Face Transformers. The goal is to learn how to fine-tune a pre-trained model for text classification tasks.

## Project Setup

### Prerequisites
- Python 3.11.6 (managed via `pyenv` or installed directly).
- `virtualenv` for creating an isolated Python environment.
- `git` for version control.

### Steps Completed So Far

1. **Project Initialization**
   - Created a project directory: `sentiment-analysis`.
   - Initialized a Git repository for version control.

2. **Environment Setup**
   - Created and activated a virtual environment:
     ```bash
     virtualenv venv
     source venv/bin/activate
     ```
   - Installed the required libraries:
     ```bash
     pip install transformers datasets torch
     ```

3. **Initial Script**
   - Created a Python script (`sentiment_analysis.py`) to test the Hugging Face Transformers library:
     ```python
     from transformers import pipeline

     sentiment = pipeline("sentiment-analysis")
     result = sentiment("I love programming!")
     print(result)
     ```
   - Verified the environment is working as expected.

4. **Python Version Management**
   - Set Python 3.11.6 as the local version using `pyenv`:
     ```bash
     pyenv local 3.11.6
    
