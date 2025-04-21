# Fake News Detection System

A Streamlit application that uses machine learning to detect potential fake news.

## Features

- Search for similar headlines from reputable news sources
- Analyze headlines using a pre-trained CNN model
- Display credibility assessment with confidence scores

## Installation

1. Clone this repository or download the files
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download NLTK stopwords:
   ```python
   python -c "import nltk; nltk.download('stopwords')"
   ```

## Required Model Files

Before running the application, ensure these files are in the app directory:
- `new_cnn_news_model.h5`: The trained CNN model
- `new_tokenizer.pickle`: The tokenizer for text processing


## Running the Application

Run the Streamlit app:
```
streamlit run app.py
```


## How It Works

1. User enters a news headline
2. App searches for similar headlines from reputable news sources
3. Found headlines are processed and analyzed by a CNN model
4. Results show the credibility assessment with confidence scores
