# Stock & Crypto Price Predictor

This is a **Streamlit-based web application** that predicts stock and cryptocurrency prices using a **Long Short-Term Memory (LSTM) model** and provides **AI-generated investment insights** using **OpenAI's GPT-4o**.

## 🚀 Features
- **📈 Stock & Crypto Price Prediction:** Uses historical price data to forecast future prices.
- **🧠 AI Investment Insights:** GPT-4o analyzes market trends and provides investment advice.
- **📊 Interactive Charts:** Visualize historical and predicted prices with Plotly.
- **🔍 Supports Major Stocks & Cryptos:** Fetches real-time data from Yahoo Finance.

---

## 📌 Installation & Setup

### **1️⃣ Install Required Packages**
Run the following command to install dependencies:
```bash
pip install streamlit yfinance openai numpy pandas tensorflow plotly scikit-learn python-dotenv
```

### **2️⃣ Set OpenAI API Key**
You need an OpenAI API Key to enable AI-generated investment insights.

#### **Option 1: Set as Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key"
```

#### **Option 2: Store in a `.env` File**
Create a `.env` file in the project directory and add:
```
OPENAI_API_KEY=your-api-key
```

---

## 🎯 Running the Application
Once dependencies are installed and the API key is set, run the following command:
```bash
streamlit run stock_crypto_predictor.py
```

After execution, open your browser and navigate to:
```
http://localhost:8501
```

---

## 📊 How to Use
1. **Enter a valid stock or crypto symbol** (e.g., `AAPL`, `BTC-USD`).
2. **Select the prediction period** (7-60 days).
3. **Click "Predict"** to generate price forecasts and AI investment insights.

---

## 🛠 Troubleshooting
- **Incorrect Stock Symbol Error:** Ensure you enter the correct stock ticker (e.g., `AAPL`, not "Apple").
- **OpenAI API Issues:** If you see API errors, check your OpenAI Key and balance.
- **Application Not Running?** Press `Ctrl + C` in the terminal and restart with `streamlit run stock_crypto_predictor.py`.

---

## 📌 Future Improvements
- ✅ Add sentiment analysis from financial news & social media.
- ✅ Improve AI insights with macroeconomic indicators.
- ✅ Deploy to cloud platforms (AWS/GCP/Streamlit Sharing).

📩 **Contributions & Feedback:** Feel free to suggest features or report issues!

---

### 💡 Built With
- **Python** 🐍
- **Streamlit** 🎨
- **TensorFlow (LSTM Model)** 🤖
- **OpenAI GPT-4o** 🧠
- **Yahoo Finance API** 💰
