

import json
from pathlib import Path


import joblib
import tensorflow as tf

from tensorflow.keras.models import load_model
from app.config import LSTM_META, MODEL_LSTM, SCALER_PATH


loaded_lstm_model = load_model(
    MODEL_LSTM,
    compile=False
)

loaded_lstm_scaler = joblib.load(
    SCALER_PATH
)


with open(LSTM_META, "r") as file:
    loaded_lstm_meta = json.load(file)



print("\n" + "=" * 55)
print("🚀 Weather Forecasting Models Initialized")
print("=" * 55)

print(f"✅ LSTM Model Loaded")
print(f"📦 Forecast Horizon : {loaded_lstm_meta['forecast_n']} hours")
print(f"📊 Feature Count    : {loaded_lstm_meta['n_features']}")
print(f"📉 Validation MAE   : {loaded_lstm_meta['metrics']['mae_celsius']} °C")

print("=" * 55 + "\n")