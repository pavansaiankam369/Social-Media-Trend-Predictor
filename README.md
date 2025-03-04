﻿

# Social-Media-Trend-Predictor


## Overview
The **Social Media Trend Predictor** is a machine learning-based application that forecasts trending hashtags on social media platforms using historical data and sentiment analysis. It utilizes **LSTM models** for time-series forecasting and integrates **FastAPI** and **Streamlit** for backend services and frontend visualization.

## Features
- Fetches historical hashtag data.
- Uses an **LSTM model** to predict future trends.
- Performs **sentiment analysis** on hashtag-related posts.
- Provides an interactive **Streamlit UI** for user-friendly analysis.
- Offers a **FastAPI** endpoint for programmatic access.

## Technologies Used
- **Python**
- **Streamlit** (Frontend visualization)
- **FastAPI** (Backend API)
- **TensorFlow/Keras** (LSTM model for trend prediction)
- **Scikit-Learn** (Data preprocessing)
- **Transformers** (Sentiment analysis)
- **Matplotlib** (Data visualization)

## Project Structure
```
trend_predictor/
│
├── app.py                     # Main Streamlit app
├── api.py                     # FastAPI backend
├── utils/                      # Utility functions
│   ├── data_fetcher.py         # Fetches hashtag data
│   ├── preprocessing.py        # Data preprocessing logic
│   ├── model_builder.py        # LSTM model training
│   ├── predictor.py            # Prediction logic
│   └── sentiment_analysis.py   # Sentiment analysis
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/pavansaiankam369/Social-Media-Trend-Predictor.git
cd Social-Media-Trend-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

### 4. Run the FastAPI backend
```bash
uvicorn api:app --reload
```

## Usage
### Using the Streamlit Interface
- Enter a hashtag in the **text input field**.
- Select the **historical data range** and **prediction days**.
- Click **"Predict Trends"** to generate forecasts and sentiment analysis.

### Using FastAPI Endpoints
#### Predict Trends API
```bash
GET /predict/{hashtag}
```
Example:
```bash
curl -X GET "http://127.0.0.1:8000/predict/AI"
```
Response:
```json
{
  "hashtag": "AI",
  "predictions": [120, 135, 150]
}
```

## Future Enhancements
- Implement real-time social media API integration.
- Improve model accuracy with additional feature engineering.
- Add user authentication and hashtag tracking.

## License
This project is open-source and available under the [MIT License](LICENSE).

---
Contributions and feedback are welcome!

