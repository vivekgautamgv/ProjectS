# main_app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from stock_data import get_stock_data # type: ignore
from models.technicals import calculate_technical_indicators, generate_signals
from models.pricing_models import black_scholes, monte_carlo_option_price
from models.pricing_models import black_scholes, monte_carlo_option_price
from scipy.optimize import newton
import requests
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Trading Analytics Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic sidebar configuration
st.sidebar.header("Global Parameters")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.5) / 100


def plot_stock_analysis(data, signals):
    """Create interactive visualization with Plotly"""
    fig = go.Figure()

    # Price and Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50', line=dict(color='red', dash='dot')))
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], 
                         line=dict(color='rgba(0,0,0,0.4)'), showlegend=False))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], 
                         fill='tonexty', fillcolor='rgba(173,216,230,0.2)',
                         line=dict(color='rgba(0,0,0,0.4)'), name='Bollinger Bands'))
    
    # Add markers for signals
    buy_signals = signals[signals['Final_Signal'] > 0]
    sell_signals = signals[signals['Final_Signal'] < 0]
    
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['price'],
                            mode='markers', name='Buy Signal',
                            marker=dict(color='green', size=10, symbol='triangle-up')))
    
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['price'],
                            mode='markers', name='Sell Signal',
                            marker=dict(color='red', size=10, symbol='triangle-down')))
    
    fig.update_layout(title='Price Analysis with Trading Signals',
                     xaxis_title='Date',
                     yaxis_title='Price',
                     template='plotly_white',
                     height=600)
    return fig

# Stock Screening Section
st.header("Stock Technical Analysis Screener")
col1, col2 = st.columns([1, 3])
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    period = st.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)

try:
    data = get_stock_data(ticker, period)
    if data.empty:
        st.warning("No data found for this ticker!")
    else:
        with st.spinner('Analyzing Stock...'):
            data = calculate_technical_indicators(data)
            signals = generate_signals(data)
            
            # Display Metrics
            latest_signal = "Buy" if signals['Final_Signal'].iloc[-1] > 0 else "Sell" if signals['Final_Signal'].iloc[-1] < 0 else "Neutral"
            st.metric("Current Recommendation", latest_signal, 
                     help="Based on combined technical indicators")
            
            # Plot main chart
            st.plotly_chart(plot_stock_analysis(data, signals), use_container_width=True)
            
            # Additional Indicators
            with st.expander("Detailed Technical Indicators"):
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
                fig2.add_hline(y=70, line_dash="dash", line_color="red")
                fig2.add_hline(y=30, line_dash="dash", line_color="green")
                fig2.update_layout(title='RSI Analysis', height=300)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'))
                fig3.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], name='Signal Line'))
                fig3.update_layout(title='MACD Analysis', height=300)
                
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig3, use_container_width=True)

except Exception as e:
    st.error(f"Error fetching data: {str(e)}")

# Update main_app.py (add after stock section)
st.header("Options Pricing Analysis")
option_tab1, option_tab2, option_tab3 = st.tabs(["Black-Scholes", "Monte Carlo", "Model Comparison"])

with option_tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Black-Scholes Parameters")
        S = st.number_input("Underlying Price", 10.0, 1000.0, 100.0, key='bs_price')
        K = st.number_input("Strike Price", 10.0, 1000.0, 105.0, key='bs_strike')
        T = st.slider("Time to Expiry (Years)", 0.1, 5.0, 1.0, key='bs_time')
        sigma = st.slider("Volatility (%)", 5.0, 150.0, 20.0, key='bs_vol') / 100
        option_type = st.radio("Option Type", ["Call", "Put"], key='bs_type')

    with col2:
        try:
            price = black_scholes(S, K, T, risk_free_rate, sigma, option_type.lower())
            st.metric("Theoretical Price", f"${price:.2f}")
            
            # Sensitivity Analysis
            st.subheader("Price Sensitivity")
            vol_range = np.linspace(0.1, 1.5, 50)
            prices = [black_scholes(S, K, T, risk_free_rate, vol, option_type.lower()) 
                     for vol in vol_range]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vol_range, y=prices, name='Option Price'))
            fig.update_layout(title='Volatility Impact',
                            xaxis_title='Volatility',
                            yaxis_title='Price',
                            height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")

with option_tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Monte Carlo Parameters")
        mc_S = st.number_input("Underlying Price", 10.0, 1000.0, 100.0, key='mc_price')
        mc_K = st.number_input("Strike Price", 10.0, 1000.0, 105.0, key='mc_strike')
        mc_T = st.slider("Time to Expiry (Years)", 0.1, 5.0, 1.0, key='mc_time')
        mc_sigma = st.slider("Volatility (%)", 5.0, 150.0, 20.0, key='mc_vol') / 100
        mc_sims = st.selectbox("Simulations", [1e4, 1e5, 1e6], index=1, format=lambda x: f"{int(x):,}")
        mc_option_type = st.radio("Option Type", ["Call", "Put"], key='mc_type')

    with col2:
        try:
            with st.spinner("Running Monte Carlo Simulation..."):
                mc_price = monte_carlo_option_price(mc_S, mc_K, mc_T, risk_free_rate,
                                                  mc_sigma, mc_option_type.lower(), int(mc_sims))
                
                st.metric("Estimated Price", f"${mc_price:.2f}")
                
                # Path Visualization
                st.subheader("Simulation Paths")
                fig2 = go.Figure()
                paths = generate_mc_paths(mc_S, mc_T, risk_free_rate, mc_sigma, 50, int(mc_T*252))
                for i in range(5):
                    fig2.add_trace(go.Scatter(x=np.linspace(0, mc_T, len(paths[i])),
                                  y=paths[i], 
                                  line=dict(width=1)))
                fig2.update_layout(title='Sample Monte Carlo Paths',
                                  xaxis_title='Time',
                                  yaxis_title='Price',
                                  showlegend=False,
                                  height=400)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Payoff Distribution
                st.subheader("Payoff Distribution")
                final_prices = paths[:, -1]
                payoffs = np.maximum(final_prices - mc_K, 0) if mc_option_type == 'Call' else np.maximum(mc_K - final_prices, 0)
                
                fig3 = go.Figure()
                fig3.add_trace(go.Histogram(x=payoffs, nbinsx=50, 
                                          marker_color='#636EFA',
                                          opacity=0.75))
                fig3.update_layout(title='Simulated Payoffs Distribution',
                                xaxis_title='Payoff',
                                yaxis_title='Frequency',
                                height=400)
                st.plotly_chart(fig3, use_container_width=True)
                
        except Exception as e:
            st.error(f"Simulation error: {str(e)}")

with option_tab3:
    st.subheader("Model Comparison")
    # Add comparison logic here

# Update main_app.py (add new section)
from utils.visualization import create_vol_surface_plot

st.header("Options Chain Analyzer")
chain_tab1, chain_tab2, chain_tab3 = st.tabs(["Live Chain", "Volatility Analysis", "Synthetic Chain"])

with chain_tab1:
    st.subheader("Live Options Chain Analysis")
    col1, col2 = st.columns([1, 4])
    with col1:
        chain_ticker = st.text_input("Underlying Ticker", "SPY")
        expiry_date = st.selectbox("Select Expiry", ["Loading..."])
        
        # Fetch expiries dynamically
        try:
            ticker = yf.Ticker(chain_ticker)
            expiries = ticker.options
            expiry_date = st.selectbox("Select Expiry", expiries)
        except Exception as e:
            st.error(f"Couldn't fetch options chain: {str(e)}")
    
    with col2:
        try:
            chain = ticker.option_chain(expiry_date)
            calls = chain.calls
            puts = chain.puts
            
            # Calculate model prices
            S = ticker.history(period='1d').iloc[-1]['Close']
            T = (pd.to_datetime(expiry_date) - pd.Timestamp.today()).days / 365
            r = risk_free_rate
            
            with st.spinner("Calculating Theoretical Prices..."):
                calls['BS Price'] = calls.apply(lambda row: black_scholes(
                    S, row['strike'], T, r, row['impliedVolatility'], 'call'), axis=1)
                puts['BS Price'] = puts.apply(lambda row: black_scholes(
                    S, row['strike'], T, r, row['impliedVolatility'], 'put'), axis=1)
                
                # Premium difference analysis
                calls['Premium Diff'] = calls['lastPrice'] - calls['BS Price']
                puts['Premium Diff'] = puts['lastPrice'] - puts['BS Price']
            
            # Display interactive chain
            st.subheader(f"Calls (Expiry: {expiry_date})")
            st.dataframe(calls[['strike', 'lastPrice', 'BS Price', 'Premium Diff', 
                              'impliedVolatility', 'volume', 'openInterest']]
                        .sort_values('strike')
                        .style.format("{:.2f}")
                        .background_gradient(subset=['Premium Diff'], cmap='RdYlGn'),
                        height=300)
            
            st.subheader(f"Puts (Expiry: {expiry_date})")
            st.dataframe(puts[['strike', 'lastPrice', 'BS Price', 'Premium Diff',
                             'impliedVolatility', 'volume', 'openInterest']]
                        .sort_values('strike', ascending=False)
                        .style.format("{:.2f}")
                        .background_gradient(subset=['Premium Diff'], cmap='RdYlGn'),
                        height=300)
            
        except Exception as e:
            st.error(f"Options chain error: {str(e)}")

with chain_tab2:
    st.subheader("Volatility Analysis")
    try:
        # Volatility Smile Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calls['strike'], y=calls['impliedVolatility'],
                               name='Calls IV', mode='markers'))
        fig.add_trace(go.Scatter(x=puts['strike'], y=puts['impliedVolatility'],
                               name='Puts IV', mode='markers'))
        fig.update_layout(title=f'Volatility Smile ({expiry_date})',
                         xaxis_title='Strike Price',
                         yaxis_title='Implied Volatility',
                         height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Vol Surface
        st.plotly_chart(create_vol_surface_plot(), use_container_width=True)
        
    except Exception as e:
        st.warning(f"Vol analysis unavailable: {str(e)}")

with chain_tab3:
    st.subheader("Synthetic Options Chain Generator")
    col1, col2 = st.columns(2)
    with col1:
        syn_S = st.number_input("Underlying Price", 50.0, 500.0, 100.0)
        syn_K_start = st.number_input("Min Strike", 50.0, 500.0, 80.0)
        syn_K_end = st.number_input("Max Strike", 50.0, 500.0, 120.0)
        syn_K_step = st.number_input("Strike Step", 1.0, 20.0, 5.0)
    
    with col2:
        syn_T = st.slider("Time to Expiry (Years)", 0.1, 2.0, 0.5)
        syn_sigma = st.slider("Volatility (%)", 10.0, 100.0, 30.0) / 100
        syn_r = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 2.5) / 100
    
    if st.button("Generate Synthetic Chain"):
        strikes = np.arange(syn_K_start, syn_K_end + syn_K_step, syn_K_step)
        synthetic = pd.DataFrame({
            'Strike': strikes,
            'Call Price': [black_scholes(syn_S, K, syn_T, syn_r, syn_sigma, 'call') for K in strikes],
            'Put Price': [black_scholes(syn_S, K, syn_T, syn_r, syn_sigma, 'put') for K in strikes],
            'Call Delta': [black_scholes_greeks(syn_S, K, syn_T, syn_r, syn_sigma, 'call')['delta'] for K in strikes],
            'Put Delta': [black_scholes_greeks(syn_S, K, syn_T, syn_r, syn_sigma, 'put')['delta'] for K in strikes]
        })
        
        st.subheader("Synthetic Options Pricing")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Call Options")
            st.dataframe(synthetic[['Strike', 'Call Price', 'Call Delta']]
                        .style.format("{:.2f}")
                        .background_gradient(subset=['Call Price'], cmap='Blues'),
                        height=400)
        with col2:
            st.write("Put Options")
            st.dataframe(synthetic[['Strike', 'Put Price', 'Put Delta']]
                        .style.format("{:.2f}")
                        .background_gradient(subset=['Put Price'], cmap='Reds'),
                        height=400)
            
# main_app.py (add these new sections after previous code)
from scipy.optimize import newton
import requests
from datetime import datetime, timedelta

# --------------------------
# Portfolio Risk Analyzer
# --------------------------
st.header("Portfolio Risk Analysis")
risk_tab1, risk_tab2 = st.tabs(["Value at Risk", "Stress Testing"])

with risk_tab1:
    st.subheader("Portfolio Value at Risk (VaR)")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        portfolio = st.text_area("Enter Portfolio (Ticker:Quantity)", 
                               "AAPL:100\nGOOG:50\nTSLA:200")
        confidence_level = st.slider("Confidence Level", 90, 99, 95)
        lookback_days = st.number_input("Lookback Period (Days)", 30, 1000, 252)
        
    with col2:
        if st.button("Calculate VaR"):
            try:
                # Parse portfolio
                positions = {line.split(':')[0]:int(line.split(':')[1]) 
                            for line in portfolio.split('\n') if line.strip()}
                
                # Get historical prices
                prices = yf.download(list(positions.keys()), 
                                    period=f"{lookback_days}d")['Adj Close']
                
                # Calculate daily returns
                returns = prices.pct_change().dropna()
                
                # Portfolio returns
                weights = np.array(list(positions.values()))
                port_returns = (returns * weights).sum(axis=1)
                
                # VaR calculation
                var = np.percentile(port_returns, 100 - confidence_level)
                current_value = (prices.iloc[-1] * weights).sum()
                
                st.metric(f"{confidence_level}% Daily VaR", 
                         f"${-var*current_value:.2f}",
                         help="Maximum expected loss under normal market conditions")
                
                # Plot distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=port_returns, name='Returns'))
                fig.add_vline(x=var, line_dash="dash", line_color="red",
                             annotation_text=f"VaR Threshold: {var:.2%}")
                fig.update_layout(title='Portfolio Returns Distribution',
                                 height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"VaR Calculation Error: {str(e)}")

# --------------------------
# Options Strategy Backtester
# --------------------------
st.header("Options Strategy Analyzer")
strategy_type = st.selectbox("Select Strategy", 
                            ["Straddle", "Strangle", "Iron Condor", "Custom"])

if strategy_type != "Custom":
    col1, col2 = st.columns(2)
    with col1:
        s_underlying = st.number_input("Underlying Price", 50.0, 500.0, 100.0)
        s_vol = st.number_input("Volatility (%)", 10.0, 150.0, 30.0) / 100
        s_days = st.number_input("Days to Expiry", 1, 365, 30)
        
    with col2:
        # Strategy-specific parameters
        if strategy_type == "Straddle":
            strike = st.number_input("Strike Price", 50.0, 500.0, 100.0)
            payoff = calculate_straddle_payoff(s_underlying, strike, s_vol, s_days/365)
        elif strategy_type == "Iron Condor":
            # Add IC parameters
            pass
            
    # Display payoff diagram
    fig = plot_strategy_payoff(payoff)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Implied Volatility Calculator
# --------------------------
with st.expander("Implied Volatility Calculator"):
    iv_col1, iv_col2 = st.columns(2)
    with iv_col1:
        iv_price = st.number_input("Option Market Price", 0.01, 1000.0, 10.0)
        iv_S = st.number_input("Underlying Price", 0.01, 1000.0, 100.0)
        iv_K = st.number_input("Strike Price", 0.01, 1000.0, 110.0)
        iv_T = st.number_input("Days to Expiry", 1, 3650, 30) / 365
        iv_r = risk_free_rate
        iv_type = st.radio("Option Type", ["Call", "Put"])
        
    with iv_col2:
        try:
            iv = calculate_implied_volatility(iv_price, iv_S, iv_K, iv_T, iv_r, iv_type.lower())
            st.metric("Implied Volatility", f"{iv:.2%}")
            
            # IV vs Historical comparison
            hist_vol = get_historical_volatility(iv_S, 30)  # 30-day historical vol
            st.write(f"30-Day Historical Volatility: {hist_vol:.2%}")
            
            # Plot IV surface for strike/expiry
            if st.button("Show IV Surface"):
                st.plotly_chart(plot_iv_surface(iv_S), use_container_width=True)
                
        except Exception as e:
            st.error(f"IV Calculation Error: {str(e)}")

# --------------------------
# Market News Integration
# --------------------------
st.header("Market News & Sentiment")
news_source = st.selectbox("News Source", ["General", "Stocks", "Crypto", "Options"])

try:
    news = get_market_news(news_source)
    for item in news[:5]:
        with st.expander(item['title']):
            st.write(f"**Source:** {item['source']}")
            st.write(f"**Published:** {item['published']}")
            st.write(item['summary'])
            st.markdown(f"[Read More]({item['url']})")
except Exception as e:
    st.warning(f"News feed unavailable: {str(e)}")

# --------------------------
# Portfolio Tracker
# --------------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

with st.sidebar.expander("My Portfolio"):
    pt_col1, pt_col2 = st.columns(2)
    with pt_col1:
        pt_ticker = st.text_input("Add Ticker")
    with pt_col2:
        pt_qty = st.number_input("Shares", 1, 10000, 100)
        
    if st.button("Add to Portfolio"):
        if pt_ticker:
            st.session_state.portfolio[pt_ticker] = pt_qty
            
    st.subheader("Current Holdings")
    for ticker, qty in st.session_state.portfolio.items():
        st.write(f"{ticker}: {qty} shares")




