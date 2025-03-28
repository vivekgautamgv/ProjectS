import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as si


# 🎯 **Main Show Function**
def show():
    st.title("📊 Options Pricing & Analysis")

    # 📌 **Model Selection**
    st.sidebar.header("Select a Page")
    model_choice = st.radio("Select Pricing Model:", 
                            ["Black-Scholes", "Monte Carlo Simulation", "Option Greeks", "Option Chain Analysis"])

    # 📌 **User Inputs**
    st.subheader("Enter Option Parameters")
    ticker = st.text_input("Stock Ticker", "AAPL")
    S = st.number_input("Stock Price ($)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.number_input("Time to Expiration (Years, T)", value=1.0)
    r = st.number_input("Risk-Free Rate (r)", value=0.05)
    sigma = st.number_input("Volatility (σ)", value=0.2)
    option_type = st.selectbox("Option Type", ["Call", "Put"])

    # 📌 **Model Execution**
    if model_choice == "Black-Scholes":
        if st.button("Calculate Black-Scholes Price"):
            price = black_scholes(S, K, T, r, sigma, option_type)
            st.success(f"📌 The {option_type} Option Price (Black-Scholes) is: **${price:.2f}**")

    elif model_choice == "Monte Carlo Simulation":
        if st.button("Run Monte Carlo Simulation"):
            mc_price = monte_carlo(S, K, T, r, sigma, option_type)
            st.success(f"📌 The {option_type} Option Price (Monte Carlo) is: **${mc_price:.2f}**")

    elif model_choice == "Option Greeks":
        if st.button("Calculate Option Greeks"):
            delta, gamma, vega, theta, rho = calculate_greeks(S, K, T, r, sigma, option_type)
            st.write(f"📌 **Delta:** {delta:.4f}")
            st.write(f"📌 **Gamma:** {gamma:.4f}")
            st.write(f"📌 **Vega:** {vega:.4f}")
            st.write(f"📌 **Theta:** {theta:.4f}")
            st.write(f"📌 **Rho:** {rho:.4f}")

    elif model_choice == "Option Chain Analysis":
        if st.button("Load Option Chain Data"):
            option_chain_data = get_option_chain(ticker)
            st.write("📊 **Options Chain Data**")
            st.dataframe(option_chain_data)
            visualize_bid_ask_spread(option_chain_data)
            visualize_volatility_surface(option_chain_data)


# 📌 **Black-Scholes Model**
def black_scholes(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S * si.norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * si.norm.cdf(d2, 0, 1)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0, 1) - S * si.norm.cdf(-d1, 0, 1)

    return price


# 📌 **Monte Carlo Simulation for Option Pricing**
def monte_carlo(S, K, T, r, sigma, option_type, num_simulations=10000):
    np.random.seed(42)
    payoff = []

    for _ in range(num_simulations):
        ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn())
        if option_type == "Call":
            payoff.append(max(ST - K, 0))
        else:
            payoff.append(max(K - ST, 0))

    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price


# 📌 **Option Greeks Calculation**
def calculate_greeks(S, K, T, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = si.norm.cdf(d1) if option_type == "Call" else -si.norm.cdf(-d1)
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    theta = - (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2) if option_type == "Call" else -K * T * np.exp(-r * T) * si.norm.cdf(-d2)

    return delta, gamma, vega, theta, rho


# 📌 **Retrieve Option Chain Data**
def get_option_chain(ticker):
    # Dummy data for now (Replace with actual API call)
    strikes = np.arange(100, 200, 10)
    bid = np.random.uniform(1, 5, len(strikes))
    ask = bid + np.random.uniform(0.1, 1, len(strikes))
    open_interest = np.random.randint(0, 100, len(strikes))
    implied_volatility = np.random.uniform(0.1, 0.5, len(strikes))

    data = pd.DataFrame({
        "Strike": strikes,
        "Bid": bid,
        "Ask": ask,
        "Open Interest": open_interest,
        "Implied Volatility": implied_volatility
    })
    return data


# 📌 **Visualization for Bid-Ask Spread**
def visualize_bid_ask_spread(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Strike"], y=data["Bid"], mode="lines+markers", name="Bid"))
    fig.add_trace(go.Scatter(x=data["Strike"], y=data["Ask"], mode="lines+markers", name="Ask"))
    fig.update_layout(title="Bid-Ask Spread", xaxis_title="Strike Price", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)


# 📌 **Volatility Surface Plot**
def visualize_volatility_surface(data):
    fig = go.Figure(data=[go.Surface(z=data["Implied Volatility"], x=data["Strike"], y=data["Open Interest"])])
    fig.update_layout(title="Implied Volatility Surface", xaxis_title="Strike Price", yaxis_title="Open Interest")
    st.plotly_chart(fig, use_container_width=True)


# 📌 **Run the App**
if __name__ == "__main__":
    show()
