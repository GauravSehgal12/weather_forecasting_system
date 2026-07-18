
import os
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Suppress TensorFlow logs before importing
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.preprocessing import MinMaxScaler


sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.config import (
    MODEL_XGBOOST_DIR, XGBOOST_MODEL_FILE, XGBOOST_META_FILE,
    MODEL_LSTM_DIR,    LSTM_MODEL_FILE,    LSTM_META_FILE,
    SCALER_DIR,        SCALER_FILE,
    FORECAST_N,        LOOKBACK,
)



_xgb_model  : Optional[XGBClassifier] = None
_xgb_meta   : Optional[dict[str, Any]] = None
_lstm_model : Optional[Any] = None
_lstm_meta   : Optional[dict[str, Any]] = None
_scaler      : Optional[MinMaxScaler] = None



XGB_FEATURES = [
    "temperature",
    "surface_pressure",
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
    "precipitation",
    "humidity",
    "wind_speed",
    "temp_rolling_6",
    "temp_lag_1",
    "temp_lag_24",
    "month",
]


LSTM_FEATURES = [
    "temperature",
    "temp_lag_1",
    "temp_lag_24",
    "temp_rolling_6",
    "humidity",
    "surface_pressure",
    "total_cloud_cover",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

LSTM_TARGET_IDX = 0  



def load_models() -> None:
    """
    Load XGBoost classifier, LSTM model, scaler, and both
    metadata files from disk into module-level variables.

    Raises:
        FileNotFoundError if any model file is missing.
        Call train_xgboost.py and train_lstm.py first.
    """
    global _xgb_model, _xgb_meta, _lstm_model, _lstm_meta, _scaler

    # ── XGBoost ───────────────────────────────────────────
    xgb_model_path = Path(MODEL_XGBOOST_DIR) / XGBOOST_MODEL_FILE
    xgb_meta_path  = Path(MODEL_XGBOOST_DIR) / XGBOOST_META_FILE

    if not xgb_model_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {xgb_model_path}.\n"
            f"Run: python pipeline/train_xgboost.py"
        )

    print(f"  Loading XGBoost → {xgb_model_path}")
    _xgb_model = XGBClassifier()
    _xgb_model.load_model(str(xgb_model_path))

    with open(xgb_meta_path, "r", encoding="utf-8") as f:
        _xgb_meta = json.load(f)
    assert _xgb_meta is not None

    print(f"  XGBoost loaded ✅  threshold={_xgb_meta.get('best_threshold', 0.5)}")

    # ── LSTM ──────────────────────────────────────────────
    lstm_model_path = Path(MODEL_LSTM_DIR) / LSTM_MODEL_FILE
    lstm_meta_path  = Path(MODEL_LSTM_DIR) / LSTM_META_FILE

    if not lstm_model_path.exists():
        raise FileNotFoundError(
            f"LSTM model not found at {lstm_model_path}.\n"
            f"Run: python pipeline/train_lstm.py"
        )

    print(f"  Loading LSTM    → {lstm_model_path}")
    _lstm_model = keras_load_model(str(lstm_model_path))

    with open(lstm_meta_path, "r", encoding="utf-8") as f:
        _lstm_meta = json.load(f)
    assert _lstm_meta is not None

    print(f"  LSTM loaded ✅   lookback={_lstm_meta.get('lookback')}h  forecast_n={_lstm_meta.get('forecast_n')}h")

  
    scaler_path = Path(SCALER_DIR) / SCALER_FILE

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}.\n"
            f"Run: python pipeline/train_lstm.py"
        )

    print(f"  Loading Scaler  → {scaler_path}")
    _scaler = joblib.load(str(scaler_path))
    print(f"  Scaler loaded ✅")


def _check_models_loaded() -> None:
    """Raise a clear error if load_models() was never called."""
    if (
        _xgb_model is None
        or _xgb_meta is None
        or _lstm_model is None
        or _lstm_meta is None
        or _scaler is None
    ):
        raise RuntimeError(
            "Models not loaded. "
            "load_models() must be called at app startup via lifespan."
        )


def predict_rain(features: dict) -> dict:
    """
    Predict whether it will rain tomorrow.

    Args:
        features : dict with keys matching XGB_FEATURES
                   (built by app/utils/fetch_weather.py → xgboost_features)

    Returns:
        {
            "rain_tomorrow"  : bool,
            "probability"    : float,   # 0.0 – 1.0
            "confidence"     : str,
            "threshold_used" : float,
        }
    """
    _check_models_loaded()
    assert _xgb_model is not None
    assert _xgb_meta is not None

    # Best threshold from training — more accurate than default 0.5
    threshold = float(_xgb_meta.get("best_threshold", 0.5))

    # Build single-row DataFrame in correct feature order
    xgb_features = _xgb_meta.get("features", XGB_FEATURES)
    row  = pd.DataFrame([{feat: features[feat] for feat in xgb_features}])
    prob = float(_xgb_model.predict_proba(row)[0][1])

    # Confidence label
    if prob > 0.75:
        confidence = "high (rain likely)"
    elif prob > 0.55:
        confidence = "medium"
    elif prob < 0.25:
        confidence = "high (no rain likely)"
    else:
        confidence = "low (uncertain)"

    return {
        "rain_tomorrow"  : bool(prob >= threshold),
        "probability"    : round(prob, 4),
        "confidence"     : confidence,
        "threshold_used" : threshold,
    }



def predict_temperature(hourly_df: pd.DataFrame, n_hours: int | None = None) -> dict:
    """
    Forecast temperature for the next N hours.

    Args:
        hourly_df : DataFrame with LSTM_FEATURES columns,
                    at least LOOKBACK (24) rows.
                    Built by app/utils/fetch_weather.py → hourly_df
        n_hours   : how many hours to forecast (default: from model meta)

    Returns:
        {
            "forecast": [
                {"hour": "+1h", "temperature_C": 28.4},
                {"hour": "+2h", "temperature_C": 29.1},
                ...
            ],
            "n_hours"     : int,
            "mae_celsius" : float,
        }
    """
    _check_models_loaded()

    # Help static type checkers: _check_models_loaded() ensures these are set
    assert _scaler is not None
    assert _lstm_model is not None
    assert _lstm_meta is not None

    lookback   = int(_lstm_meta.get("lookback",   LOOKBACK))
    forecast_n = int(_lstm_meta.get("forecast_n", FORECAST_N))
    n          = n_hours or forecast_n
    
    # Load features list and target index dynamically from metadata
    features   = _lstm_meta.get("features", LSTM_FEATURES)
    target_idx = int(_lstm_meta.get("target_idx", LSTM_TARGET_IDX))
    n_features = len(features)

    assert n <= forecast_n, (
        f"n_hours={n} exceeds trained forecast_n={forecast_n}. "
        f"Retrain with larger FORECAST_N."
    )
    assert len(hourly_df) >= lookback, (
        f"Need at least {lookback} rows, got {len(hourly_df)}."
    )

    # Build input window — last lookback rows, correct feature order
    window      = hourly_df[features].iloc[-lookback:].values.astype(np.float32)
    scaled      = _scaler.transform(window).reshape(1, lookback, n_features)

    # Predict
    pred_scaled = _lstm_model.predict(scaled, verbose=0)[0][:n]

    # Inverse transform temperature column only
    dummy = np.zeros((n, n_features), dtype=np.float32)
    dummy[:, target_idx] = pred_scaled
    temps = _scaler.inverse_transform(dummy)[:, target_idx]

    return {
        "forecast": [
            {"hour": f"+{i+1}h", "temperature_C": round(float(t), 2)}
            for i, t in enumerate(temps)
        ],
        "n_hours"     : n,
        "mae_celsius" : round(float(_lstm_meta.get("metrics", {}).get("mae_celsius", 0)), 4),
    }



def get_loaded_meta() -> dict:
    """
    Return already-loaded metadata dicts without re-reading files.
    Faster than reading JSON from disk on every /model-meta call.
    """
    _check_models_loaded()
    return {
        "lstm"    : _lstm_meta,
        "xgboost" : _xgb_meta,
    }