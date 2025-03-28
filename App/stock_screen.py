import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def show():  # ✅ This function allows navigation
    # 📌 Page Configuration
    st.title("📊 Stock Screener")

    # 📌 Input for Stock Ticker
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")

    # Function to fetch stock data
    def get_stock_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            stock_data = stock.history(period="6mo")  # Get last 6 months data
            return stock_data
        except Exception as e:
            return None

    # 📌 Function to Calculate Technical Indicators
    def compute_technical_indicators(data):
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["SMA_200"] = data["Close"].rolling(window=200).mean()
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
        data["RSI"] = 100 - (100 / (1 + data["Close"].pct_change().rolling(14).mean() /
                                     data["Close"].pct_change().rolling(14).std()))
        
        # Bollinger Bands
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        data["BB_Upper"] = data["BB_Middle"] + (data["Close"].rolling(window=20).std() * 2)
        data["BB_Lower"] = data["BB_Middle"] - (data["Close"].rolling(window=20).std() * 2)
        
        return data

    # 📌 Function to Compute Sentiment
    def get_sentiment(data):
        if data["RSI"].iloc[-1] > 70:
            return "Bearish (Overbought)", "red"
        elif data["RSI"].iloc[-1] < 30:
            return "Bullish (Oversold)", "green"
        else:
            return "Neutral", "gray"

    # 📌 Load Stock Data
    if ticker:
        stock_data = get_stock_data(ticker)

        if stock_data is None or stock_data.empty:
            st.warning("⚠ Stock data not found. Check the ticker symbol.")
        else:
            st.success(f"✅ Data loaded for {ticker}")

            # 📌 Compute Technical Indicators
            stock_data = compute_technical_indicators(stock_data)

            # 📌 Display Stock Price Chart
            st.subheader(f"{ticker} Closing Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["Close"], name="Close Price"))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_50"], name="SMA 50", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["SMA_200"], name="SMA 200", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Upper"], name="BB Upper", line=dict(color="green", dash="dot")))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data["BB_Lower"], name="BB Lower", line=dict(color="red", dash="dot")))
            st.plotly_chart(fig, use_container_width=True)

            # 📌 Display Technical Indicators Table
            st.subheader("📊 Technical Indicators")
            st.dataframe(stock_data[["Close", "SMA_50", "SMA_200", "EMA_20", "RSI", "BB_Upper", "BB_Lower"]].tail(10))

            # 📌 Display Sentiment Meter
            sentiment, color = get_sentiment(stock_data)
            st.subheader("📈 Stock Sentiment Meter")
            st.markdown(f"""
            <div style="text-align: center;">
                <span style="color: {color}; font-size: 24px; font-weight: bold;">{sentiment}</span>
            </div>
            """, unsafe_allow_html=True)
