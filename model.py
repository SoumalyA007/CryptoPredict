import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load Model & Scaler
def load_model_and_scaler():
    model = load_model("model.h5")
    scaler = MinMaxScaler()
    return model, scaler

# Predict future prices
def predict_future_prices(model, scaler, df, seq_length=30):
    last_data = df[-seq_length:].values
    last_data_scaled = scaler.fit_transform(last_data)
    last_data_scaled = np.expand_dims(last_data_scaled, axis=0)

    future_dates = [30, 90, 180, 330]  # Updated prediction periods
    predictions = {}

    for days in future_dates:
        for _ in range(days):
            future_price_scaled = model.predict(last_data_scaled)[0][0]
            last_data_scaled = np.roll(last_data_scaled, -1, axis=1)
            last_data_scaled[0, -1, 0] = future_price_scaled

        future_price = scaler.inverse_transform([[future_price_scaled, 0, 0, 0]])[0][0]
        predictions[f'Price after {days} days'] = future_price

    return predictions
