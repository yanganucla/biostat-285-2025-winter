import streamlit as st
import yfinance as yf
import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from dotenv import load_dotenv

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to fetch stock or crypto data
def get_stock_data(ticker, start="2020-01-01"):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    return df

# Function to predict future stock prices using LSTM
def predict_stock_price(df, days=30):
    data = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(50, len(data_scaled) - days):
        X.append(data_scaled[i-50:i, 0])
        y.append(data_scaled[i + days, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=16, epochs=10, verbose=0)

    # Generate future predictions
    future_predictions = []
    last_50_days = data_scaled[-50:].reshape(1, 50, 1)
    for _ in range(days):
        pred = model.predict(last_50_days, verbose=0)[0][0]
        future_predictions.append(pred)
        last_50_days = np.roll(last_50_days, -1)
        last_50_days[0, -1, 0] = pred

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Function to get AI-generated investment insights
def get_ai_insight(ticker):
    prompt = f"""Analyze the stock {ticker} and provide investment insights.
    Consider fundamentals, technicals, and market trends."""

    # ✅ New OpenAI API Format (compatible with openai>=1.0.0)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a financial analyst."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message["content"]

# Streamlit Web Application
st.title("📈 Stock & Crypto Price Predictor")

# User input for stock/crypto symbol
ticker = st.text_input("Enter Stock or Crypto Symbol", "AAPL")
days = st.slider("Prediction Days", 7, 60, 30)

# Run prediction and display results
if st.button("Predict"):
    df = get_stock_data(ticker)
    future_prices = predict_stock_price(df, days)

    # Plot historical and predicted prices
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['Close'][-60:], name="Historical Prices"))
    fig.add_trace(go.Scatter(y=future_prices.flatten(), name="Predicted Prices"))
    st.plotly_chart(fig)

    # AI-generated investment insights
    insight = get_ai_insight(ticker)
    st.write("### 📊 AI Investment Insights:")
    st.write(insight)
