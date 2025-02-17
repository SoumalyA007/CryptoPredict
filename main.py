import os
import uvicorn
import numpy as np
import tensorflow as tf
import pandas as pd
from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pycoingecko import CoinGeckoAPI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize API, sentiment analyzer, and FastAPI app
cg = CoinGeckoAPI()
sanalyzer = SentimentIntensityAnalyzer()
app = FastAPI()

# Load trained model
model_path = "model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Ensure it is uploaded.")

model = load_model(model_path)

# Function to fetch historical price data
def get_historical_data(coin_id, vs_currency, days=365):
    try:
        if days > 365:
            raise HTTPException(status_code=400, detail="Free API only allows 365 days of historical data.")
        
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_30'] = df['price'].rolling(window=30).mean()
        df.dropna(inplace=True)
        return df[['price', 'ma_7', 'ma_30']]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

# Function to fetch news sentiment score
def get_news_sentiment():
    try:
        url = "https://newsapi.org/v2/everything?q=cryptocurrency&apiKey=YOUR_NEWSAPI_KEY"
        response = requests.get(url).json()
        scores = [sanalyzer.polarity_scores(article['title'])['compound'] for article in response['articles']]
        return np.mean(scores) if scores else 0
    except Exception as e:
        return 0  # Return neutral sentiment if error occurs

# Function to prepare data for prediction
def prepare_data(df, seq_length=30):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X = [df_scaled[-seq_length:]]  # Use last 30 days for prediction
    return np.array(X), scaler

# Prediction endpoint
@app.get("/predict/")
def predict_price(coin: str = "bitcoin", days: int = 30):
    """
    Predict future cryptocurrency price based on LSTM model.
    Allowed days: 30, 90, 180, 330.
    Example URL: /predict/?coin=bitcoin&days=30
    """
    try:
        if days not in [30, 90, 180, 330]:
            raise HTTPException(status_code=400, detail="Allowed days: 30, 90, 180, 330.")

        df = get_historical_data(coin, "usd", 365)
        df['sentiment'] = get_news_sentiment()

        X, scaler = prepare_data(df)
        future_prices = {}

        for _ in range(days):
            future_price_scaled = model.predict(X)[0][0]
            X = np.roll(X, -1, axis=1)
            X[0, -1, 0] = future_price_scaled  # Shift new prediction into input

        future_price = scaler.inverse_transform([[future_price_scaled, 0, 0, 0]])[0][0]
        future_prices[f"Predicted price in {days} days"] = future_price

        return {"coin": coin, "predicted_price": future_prices}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Root endpoint
@app.get("/")
def home():
    return {"message": "Crypto Price Prediction API is running!"}

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
