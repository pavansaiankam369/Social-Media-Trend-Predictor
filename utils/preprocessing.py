from sklearn.preprocessing import MinMaxScaler
import numpy as np
import streamlit as st

@st.cache_data
def preprocess_data(df, look_back=3):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df["frequency"].values.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i + look_back])
        y.append(scaled_data[i + look_back])
    return np.array(X), np.array(y), scaler  