from fastapi import FastAPI, HTTPException
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from pycoingecko import CoinGeckoAPI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from model import load_model_and_scaler, predict_future_prices

app = FastAPI()

# Initialize CoinGecko API and sentiment analyzer
cg = CoinGeckoAPI()
sanalyzer = SentimentIntensityAnalyzer()

# Load LSTM Model and Scaler
model, scaler = load_model_and_scaler()

# Fetch historical price data
def get_historical_data(coin_id, vs_currency, days=365):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency=vs_currency, days=days)
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        df['ma_7'] = df['price'].rolling(window=7).mean()
        df['ma_30'] = df['price'].rolling(window=30).mean()
        df.dropna(inplace=True)
        return df[['price', 'ma_7', 'ma_30']]
    except:
        raise HTTPException(status_code=404, detail="Error fetching historical data")

# Fetch current price
def get_current_price(coin_id, vs_currency):
    try:
        data = cg.get_price(ids=coin_id, vs_currencies=vs_currency)
        return data.get(coin_id, {}).get(vs_currency, None)
    except:
        return None

# Fetch news sentiment
def get_news_sentiment():
    url = "https://newsapi.org/v2/everything?q=cryptocurrency&apiKey=YOUR_NEWSAPI_KEY"
    response = requests.get(url).json()
    scores = [sanalyzer.polarity_scores(article['title'])['compound'] for article in response.get('articles', [])]
    return np.mean(scores) if scores else 0

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crypto Prediction API"}

@app.get("/predict/{coin_id}")
def predict(coin_id: str):
    try:
        currency = "usd"
        current_price = get_current_price(coin_id, currency)
        if current_price is None:
            raise HTTPException(status_code=404, detail="Invalid cryptocurrency name")

        sentiment_score = get_news_sentiment()
        df = get_historical_data(coin_id, currency)
        predictions = predict_future_prices(model, scaler, df)

        return {
            "coin": coin_id,
            "current_price": current_price,
            "sentiment_score": sentiment_score,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
