from fastapi import FastAPI
from utils.data_fetcher import fetch_x_data
from utils.preprocessing import preprocess_data
from utils.model_builder import build_and_train_lstm_model
from utils.predictor import predict_trends
import numpy as np

app = FastAPI()

@app.get("/predict/{hashtag}")
async def predict_api(hashtag: str, days_history: int = 7, days_predict: int = 3):
    data = fetch_x_data(hashtag, days=days_history)
    X, y, scaler = preprocess_data(data)
    model = build_and_train_lstm_model((X.shape[1], 1), X, y)
    last_data = scaler.transform(data["frequency"].values.reshape(-1, 1))
    predictions = predict_trends(model, last_data, scaler, X.shape[1], days_predict)
    return {"hashtag": hashtag, "predictions": predictions.tolist()}