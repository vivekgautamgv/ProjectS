import streamlit as st
import yfinance as yf
import pandas as pd

st.title("📊 Stock Screener")

# Input field for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL")

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period="6mo")  # Get last 6 months data
        return stock_data
    except Exception as e:
        return None

if ticker:
    stock_data = get_stock_data(ticker)

    if stock_data is None or stock_data.empty:
        st.warning("⚠ Stock data not found. Check the ticker symbol.")
    else:
        st.success(f"✅ Data loaded for {ticker}")

        # Display stock price chart
        st.subheader(f"{ticker} Closing Price Chart")
        st.line_chart(stock_data["Close"])

        # Display raw data table
        st.subheader("Stock Data Table")
        st.dataframe(stock_data.tail(10))  # Show last 10 days

