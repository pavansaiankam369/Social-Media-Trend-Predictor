from transformers import pipeline
import numpy as np
import streamlit as st

@st.cache_resource
def load_sentiment_classifier():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.error(f"Error loading sentiment classifier: {str(e)}")
        return None

def analyze_sentiment(hashtag, classifier):
    if classifier is None:
        return 0.0  # Default value if classifier fails
    mock_posts = [f"I love #{hashtag}", f"#{hashtag} is terrible", f"Amazing #{hashtag}"]
    sentiments = classifier(mock_posts)
    avg_sentiment = np.mean([s["score"] if s["label"] == "POSITIVE" else -s["score"] for s in sentiments])
    return avg_sentiment