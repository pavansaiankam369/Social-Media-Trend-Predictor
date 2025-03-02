import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from utils.data_fetcher import fetch_x_data
from utils.preprocessing import preprocess_data
from utils.model_builder import build_and_train_lstm_model
from utils.predictor import predict_trends
from utils.sentiment_analysis import load_sentiment_classifier, analyze_sentiment
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta



# Set page config as the FIRST Streamlit command
st.set_page_config(page_title="Trend Predictor", layout="wide")

def main():
    st.title("Social Media Trend Predictor")
    st.write("Predict trending hashtags based on historical X data and sentiment analysis.")

    # Load sentiment classifier
    classifier = load_sentiment_classifier()
    if classifier is None:
        st.warning("Sentiment analysis unavailable due to loading error. Proceeding without it.")
    
    # User inputs
    hashtag = st.text_input("Enter a hashtag to analyze (e.g., AI):", "AI")
    days_history = st.slider("Days of historical data:", 3, 30, 7)
    days_predict = st.slider("Days to predict:", 1, 7, 3)

    if st.button("Predict Trends"):
        with st.spinner("Fetching data and making predictions..."):
            # Fetch and preprocess data
            data = fetch_x_data(hashtag, days=days_history)
            X, y, scaler = preprocess_data(data)

            # Build or load model
            model = build_and_train_lstm_model((X.shape[1], 1), X, y)

            # Predict trends
            last_data = scaler.transform(data["frequency"].values.reshape(-1, 1))
            predictions = predict_trends(model, last_data, scaler, X.shape[1], days_predict)

            # Sentiment analysis
            sentiment_score = analyze_sentiment(hashtag, classifier)

            # Visualization
            future_dates = [data["date"].iloc[-1] + timedelta(days=i + 1) for i in range(days_predict)]
            fig, ax = plt.subplots(figsize=(12, 6))  # Larger figure size
            ax.plot(data["date"], data["frequency"], label="Historical Data")
            ax.plot(future_dates, predictions, label="Predicted Trends", linestyle="--")
            ax.set_xlabel("Date")
            ax.set_ylabel("Hashtag Frequency")
            ax.legend()

            # Fix clashing dates
            plt.xticks(rotation=45, ha="right")  # Rotate labels
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))  # Limit ticks
            ax.tick_params(axis="x", labelsize=8)  # Smaller font
            plt.tight_layout()

            st.pyplot(fig)

            st.write(f"Sentiment Score for #{hashtag}: {sentiment_score:.2f} (-1 to 1)")
            st.write("Positive score indicates positive sentiment.")

if __name__ == "__main__":
    main()