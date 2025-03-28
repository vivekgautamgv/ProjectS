# data/stock_data.py
import yfinance as yf
import pandas as pd

def get_stock_data(ticker, period='1y'):
    """Fetch historical stock data"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist.drop(columns=['Dividends', 'Stock Splits'])