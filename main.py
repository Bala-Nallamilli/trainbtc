import os
import time
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# Constants
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "bitcoin_model.h5")
SCALER_PATH = os.path.join(CHECKPOINT_DIR, "scaler.pkl")

# Fetch historical Bitcoin price data
def fetch_historical_data(interval="1h", limit=1000):
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": limit,
    }
    response = requests.get(BINANCE_API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["close"] = df["close"].astype(float)
    return df[["timestamp", "close"]]

# Preprocess the data
def preprocess_data(df, scaler=None):
    # Scale the data
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df["scaled_close"] = scaler.fit_transform(df[["close"]])
    else:
        df["scaled_close"] = scaler.transform(df[["close"]])

    # Create sequences
    sequence_length = 60
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df["scaled_close"].iloc[i:i + sequence_length].values)
        y.append(df["scaled_close"].iloc[i + sequence_length])

    return np.array(X), np.array(y), scaler

# Build the LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Load or initialize the model
def load_or_initialize_model(input_shape):
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        model = build_model(input_shape)
        print("Initialized a new model.")
    return model

# Real-time graphs for loss and success rate
def setup_realtime_graphs():
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Batch")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("Success Rate")
    ax[1].set_xlabel("Batch")
    ax[1].set_ylabel("Success (%)")
    return fig, ax

# Train the model
def train_model_continuously():
    # Fetch initial data
    df = fetch_historical_data(interval="1h", limit=1000)
    X, y, scaler = preprocess_data(df)

    # Load or initialize the model
    model = load_or_initialize_model(input_shape=(X.shape[1], 1))

    # Real-time graph setup
    fig, ax = setup_realtime_graphs()
    loss_values = []
    success_rates = []

    # Continuous training loop
    print("Starting continuous training...")
    start_time = time.time()
    while True:
        # Train the model and collect batch loss
        history = model.fit(X, y, epochs=1, batch_size=32, verbose=1)
        loss = history.history["loss"][0]
        loss_values.append(loss)

        # Calculate success rate
        predictions = model.predict(X)
        actuals = y.reshape(-1, 1)
        success = np.mean(np.abs(predictions - actuals) / actuals <= 0.05) * 100  # 5% tolerance
        success_rates.append(success)

        # Update the graphs
        ax[0].plot(loss_values, label="Loss", color="blue")
        ax[1].plot(success_rates, label="Success Rate", color="green")
        plt.pause(0.1)

        # Save model and scaler every 6 hours
        if int(time.time() - start_time) % (6 * 60 * 60) == 0:
            if not os.path.exists(CHECKPOINT_DIR):
                os.makedirs(CHECKPOINT_DIR)
            model.save(MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

        # Fetch new data every hour and update dataset
        print("Fetching new data...")
        df_new = fetch_historical_data(interval="1h", limit=60)
        df = pd.concat([df, df_new]).drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
        X, y, scaler = preprocess_data(df, scaler)

        # Terminate after 7 days
        elapsed_time = time.time() - start_time
        if elapsed_time >= 7 * 24 * 60 * 60:  # 7 days
            print("Training completed after 7 days.")
            model.save(MODEL_PATH)
            print(f"Final model saved to {MODEL_PATH}")
            break

if __name__ == "__main__":
    train_model_continuously()
