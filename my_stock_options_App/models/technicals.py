# models/technicals.py
import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """Calculate multiple technical indicators"""
    # SMA
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + 2*df['Close'].rolling(window=20).std()
    df['Lower_Band'] = df['Middle_Band'] - 2*df['Close'].rolling(window=20).std()
    
    """Add 2 more indicators (now total 6)"""
    # Existing code...
    
    # Stochastic Oscillator (14-day)
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['%D'] = df['%K'].rolling(3).mean()
    
    # Volume Weighted Average Price (VWAP)
    cum_vol = df['Volume'].cumsum()
    cum_vol_price = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
    df['VWAP'] = cum_vol_price / cum_vol
    
    return df

def generate_signals(df):
    """Generate trading signals based on indicators"""
    signals = pd.DataFrame(index=df.index)
    signals['price'] = df['Close']
    
    # SMA Crossover
    signals['SMA_Signal'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
    
    # RSI Signals
    signals['RSI_Signal'] = 0
    signals.loc[df['RSI'] < 30, 'RSI_Signal'] = 1
    signals.loc[df['RSI'] > 70, 'RSI_Signal'] = -1
    
    # MACD Crossover
    signals['MACD_Signal'] = np.where(df['MACD'] > df['Signal_Line'], 1, -1)
    
    # Combine signals
    signals['Final_Signal'] = signals[['SMA_Signal', 'RSI_Signal', 'MACD_Signal']].mean(axis=1)

    """Enhanced signal generation"""
    # Existing signals...
    
    # Stochastic Signals
    signals['Stoch_Signal'] = 0
    signals.loc[(df['%K'] < 20) & (df['%D'] < 20), 'Stoch_Signal'] = 1
    signals.loc[(df['%K'] > 80) & (df['%D'] > 80), 'Stoch_Signal'] = -1
    
    # VWAP Signal
    signals['VWAP_Signal'] = np.where(df['Close'] > df['VWAP'], 1, -1)
    
    # Combine all signals with weighting
    weights = {
        'SMA_Signal': 0.2,
        'RSI_Signal': 0.25,
        'MACD_Signal': 0.25,
        'Stoch_Signal': 0.15,
        'VWAP_Signal': 0.15
    }
    
    signals['Final_Signal'] = sum(signals[col] * weight for col, weight in weights.items())
    return signals