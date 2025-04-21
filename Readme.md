# Fake News Detection System

A Streamlit application that uses machine learning to detect potential fake news by comparing user-provided headlines with trusted news sources.

## Features

- Search for similar headlines from reputable news sources
- Analyze headlines using a pre-trained CNN model
- Display credibility assessment with confidence scores
- User-friendly interface with detailed results

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

These files should be moved from your original location:
```
D:/Python/news_detection/new_tokenizer.pickle
D:/Python/news_detection/new_cnn_news_model.h5
```

## Running the Application

Run the Streamlit app:
```
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Create a GitHub repository and push all files
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the application by selecting app.py as the main file

## How It Works

1. User enters a news headline
2. App searches for similar headlines from reputable news sources
3. Found headlines are processed and analyzed by a CNN model
4. Results show the credibility assessment with confidence scores

## Customization

- You can modify the list of trusted news sources in the `TopNewsWebsitesCrawler` class
- Adjust the similarity threshold for headline matching
- Change the UI layout and styling as needed