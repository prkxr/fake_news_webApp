import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import difflib
import numpy as np
import pickle as pck
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This application uses NLP and machine learning to detect potential fake news. "
    "Enter a news headline, and the app will search for similar headlines "
    "from reputable sources and analyze them to determine credibility."
)

st.sidebar.title("How it works")
st.sidebar.markdown(
    """
    1. Enter a news headline
    2. App searches for similar headlines on reputable news sites
    3. Headlines are processed and analyzed by NLP and ML model
    4. Results show credibility assessment with confidence scores
    """
)

st.title("Fake News Detection System")
st.markdown("Enter a news headline to check its credibility")

class TopNewsWebsitesCrawler:
    def __init__(self, user_agent=None):
        self.headers = {
            'User-Agent': user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Top 10 news websites including Indian and world news sources
        self.top_news_sites = [
            # Indian news sites
            "timesofindia.indiatimes.com",
            "ndtv.com",
            "hindustantimes.com",
            "indianexpress.com",
            "news18.com",
            # World news sites
            "bbc.com",
            "cnn.com",
            "reuters.com",
            "theguardian.com",
            "aljazeera.com"
        ]
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def search_articles(self, query, max_results=10):
        """Search for news headlines related to the query on top news websites."""
        search_url = "https://html.duckduckgo.com/html/"
        
        
        site_filter = " OR ".join([f"site:{site}" for site in self.top_news_sites])
        search_query = f"{query} ({site_filter})"
        
        data = {"q": search_query}
        
        try:
            response = requests.post(search_url, headers=self.headers, data=data)
            headlines = []
            domains = []

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for result in soup.find_all('div', class_='result__body'):
                    
                    headline_element = result.find('a', class_='result__a')
                    if not headline_element:
                        continue
                        
                    headline = headline_element.get_text(strip=True)
                    
                    
                    url_element = result.find('a', class_='result__url')
                    domain = "unknown"
                    
                    if url_element:
                        url_text = url_element.get_text(strip=True)
                        
                        domain_match = re.search(r'(?:https?:\/\/)?(?:www\.)?([^\/]+)', url_text)
                        if domain_match:
                            domain = domain_match.group(1).lower()
                    
                    # Check if domain is in our target sites
                    if any(site in domain for site in self.top_news_sites):
                        
                        if headline not in headlines:
                            headlines.append(headline)
                            domains.append(domain)
                            
                    if len(headlines) >= max_results:
                        break
                        
            
            if not headlines:
                soup = BeautifulSoup(response.text, 'html.parser')
                for result in soup.find_all('a', class_='result__a'):
                    headline = result.get_text(strip=True)
                    
                    
                    if headline not in headlines:
                        headlines.append(headline)
                        domains.append("news site")  
                        
                    if len(headlines) >= max_results:
                        break
            
            return list(zip(headlines, domains))
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            return []
    
    def get_similar_headlines(self, query, max_results=10):
        """
        Get headlines from top news sites that are most similar to the query
        using sequence matcher similarity.
        """
        
        candidates = self.search_articles(query, max_results=max_results*2)
    
        if not candidates:
            return []
        
        
        headlines = [headline for headline, _ in candidates]
        domains = [domain for _, domain in candidates]
    
        # Calculate similarity scores using only SequenceMatcher
        sequence_similarities = [difflib.SequenceMatcher(None, query.lower(), headline.lower()).ratio() 
                            for headline in headlines]
    
        # Sort by sequence similarity score
        sorted_results = sorted(zip(sequence_similarities, headlines, domains), 
                            key=lambda x: x[0], reverse=True)
    
        results = []
        
        for score, headline, domain in sorted_results[:max_results]:
            if score >= 0.5:  
                results.append({
                    "headline": headline,
                    "domain": domain,
                    "similarity_score": score
                })

        return results

# Text processing functions
port_stem = PorterStemmer()
def stem_summaries(summaries):
    stemmed_summaries = []
    for content in summaries:
        content = str(content)
        content = re.sub('[^a-zA-Z]', ' ', content)
        content = content.lower().split()
        content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
        stemmed_summary = ' '.join(content)
        stemmed_summaries.append(stemmed_summary)
    return stemmed_summaries

@st.cache_resource
def load_model_and_tokenizer():
    """Load and cache the model and tokenizer"""
    
    model_path = 'D:/Python/news_detection/new_cnn_news_model.h5'
    tokenizer_path = 'D:/Python/news_detection/new_tokenizer.pickle'
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        st.error("Model or tokenizer file not found. Make sure you have the required files in the app directory.")
        return None, None
    
    tokenizer=Tokenizer()
    
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pck.load(handle)
    
    return model, tokenizer

def predict_news(news_list, model, tokenizer):
    """Process news headlines and make predictions"""
    if not model or not tokenizer:
        return []
    
    prediction_values = []
    
    
    for news in news_list:
        # Tokenize and pad the headline
        sequence = tokenizer.texts_to_sequences([news])
        padded_sequence = pad_sequences(sequence, maxlen=250)
        
        #prediction
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        prediction_values.append(prediction)
    
    return prediction_values

def run_ml_prediction(stemmed_summaries, model, tokenizer):
    """Run the ML model on processed headlines"""
    predictions = predict_news(stemmed_summaries, model, tokenizer)
    
    if not predictions:
        return []
    
    avg_prediction = sum(predictions) / len(predictions)
    
    results = []
    for summary, pred in zip(stemmed_summaries, predictions):
        prediction_label = "Most Likely True" if pred > 0.5 else "Most Likely False"
        confidence = pred if pred > 0.5 else 1 - pred
        results.append({
            'headline': summary,
            'prediction_value': pred,
            'prediction': prediction_label,
            'confidence': confidence * 100  # Convert to percentage
        })
    
    return results, avg_prediction


def main():
    model, tokenizer = load_model_and_tokenizer()
    
    if not model or not tokenizer:
        st.warning("Please ensure model and tokenizer files are present in the app directory")
        st.stop()
    
    # Input
    query = st.text_input("Enter News Headline to Verify:", "")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        search_button = st.button("Verify News")
        max_results = st.slider("Maximum results to analyze", 3, 15, 10)
    
    with col2:
        st.info("For best results, enter the complete headline. The system will search for similar headlines on reputable news sites.")
    
    
    if search_button and query:
        with st.spinner("Searching for similar headlines..."):
            crawler = TopNewsWebsitesCrawler()
            processed_headlines = crawler.get_similar_headlines(query, max_results=max_results)
        
        if not processed_headlines:
            st.error("No similar headlines found. This news is likely false.")
        elif len(processed_headlines) <= 2:
            st.warning(f"Only {len(processed_headlines)} similar headlines found. This news might be fake or uncommon.")
            
            st.subheader("Found Headlines:")
            for i, headline_data in enumerate(processed_headlines, 1):
                st.write(f"{i}. {headline_data['headline']}")
                st.write(f"   Source: {headline_data['domain']}")
                st.write(f"   Similarity: {headline_data['similarity_score']:.2f}")
                st.write("---")
        else:
            
            headline_texts = [item['headline'] for item in processed_headlines]
            stemmed_summaries = stem_summaries(headline_texts)
            
            # Combine stemmed text with domain information
            combined_texts = []
            for i, item in enumerate(processed_headlines):
                combined_texts.append(f"{stemmed_summaries[i]} {item['domain']}")
            
            
            with st.spinner("Analyzing headlines..."):
                results, avg_prediction = run_ml_prediction(combined_texts, model, tokenizer)
            
            # Display overall prediction
            overall_prediction = "Most Likely True" if avg_prediction > 0.5 else "Most Likely False"
            confidence = avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
            
            col1, col2 = st.columns(2)
            with col1:
                if overall_prediction == "Most Likely True":
                    st.success(f"Overall Assessment: {overall_prediction}")
                else:
                    st.error(f"Overall Assessment: {overall_prediction}")
            
            with col2:
                st.metric("Confidence Score", f"{confidence*100:.2f}%")
            
            # Display the results
            st.subheader("Found Similar Headlines:")
            with st.expander("View Details", expanded=True):
                for i, (headline_data, prediction) in enumerate(zip(processed_headlines, results), 1):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{headline_data['headline']}**")
                        st.write(f"Source: {headline_data['domain']}")
                    with col2:
                        if prediction['prediction'] == "Most Likely True":
                            st.success(prediction['prediction'])
                        else:
                            st.error(prediction['prediction'])
                    with col3:
                        st.write(f"Confidence: {prediction['confidence']:.2f}%")
                        st.write(f"Similarity: {headline_data['similarity_score']:.2f}")
                    st.write("---")

if __name__ == "__main__":
    main()