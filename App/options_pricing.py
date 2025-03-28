import streamlit as st
import numpy as np
import scipy.stats as si
import math

def show():
    st.title("📈 Options Pricing & Analysis")

    # 📌 Select Model
    model_choice = st.radio("Select Pricing Model:", ["Black-Scholes", "Monte Carlo Simulation", "Option Greeks"], horizontal=True)

    # 📌 Input Fields
    st.subheader("Enter Option Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        S = st.number_input("Stock Price (S)", value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", value=100.0, step=1.0)
        T = st.number_input("Time to Expiration (Years, T)", value=1.0, step=0.1)
    
    with col2:
        r = st.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
        sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01)
        option_type = st.selectbox("Option Type", ["Call", "Put"])

    # 📌 Black-Scholes Formula
    def black_scholes(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "Call":
            price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

        return round(price, 2)

    # 📌 Monte Carlo Simulation
    def monte_carlo_simulation(S, K, T, r, sigma, option_type, simulations=10000):
        np.random.seed(42)
        dt = T / 252  # Assume 252 trading days in a year
        stock_prices = np.zeros(simulations)

        for i in range(simulations):
            Z = np.random.standard_normal()
            ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
            stock_prices[i] = max(ST - K, 0) if option_type == "Call" else max(K - ST, 0)

        option_price = np.exp(-r * T) * np.mean(stock_prices)
        return round(option_price, 2)

    # 📌 Greeks Calculation
    def option_greeks(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        delta = si.norm.cdf(d1) if option_type == "Call" else si.norm.cdf(d1) - 1
        gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        theta = (- (S * si.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))) - (r * K * np.exp(-r * T) * si.norm.cdf(d2))
        vega = S * si.norm.pdf(d1) * np.sqrt(T)
        rho = K * T * np.exp(-r * T) * si.norm.cdf(d2) if option_type == "Call" else -K * T * np.exp(-r * T) * si.norm.cdf(-d2)

        return round(delta, 3), round(gamma, 3), round(theta / 365, 3), round(vega / 100, 3), round(rho / 100, 3)

    # 📌 Calculate & Display Result
    st.subheader("📊 Pricing Result")

    if st.button("Calculate Price"):
        if model_choice == "Black-Scholes":
            price = black_scholes(S, K, T, r, sigma, option_type)
            st.success(f"Black-Scholes {option_type} Option Price: **${price}**")

        elif model_choice == "Monte Carlo Simulation":
            price = monte_carlo_simulation(S, K, T, r, sigma, option_type)
            st.success(f"Monte Carlo {option_type} Option Price: **${price}**")

        elif model_choice == "Option Greeks":
            delta, gamma, theta, vega, rho = option_greeks(S, K, T, r, sigma, option_type)
            st.success(f"**Greeks for {option_type} Option**")
            st.write(f"📍 **Delta:** {delta}")
            st.write(f"📍 **Gamma:** {gamma}")
            st.write(f"📍 **Theta (Daily Decay):** {theta}")
            st.write(f"📍 **Vega (Volatility Sensitivity):** {vega}")
            st.write(f"📍 **Rho (Interest Rate Sensitivity):** {rho}")

    # 📌 Explanation
    st.subheader("📚 Model Explanation")
    if model_choice == "Black-Scholes":
        st.write("The **Black-Scholes Model** is used for pricing European call and put options.")
    elif model_choice == "Monte Carlo Simulation":
        st.write("Monte Carlo Simulation estimates option prices by simulating thousands of possible stock price movements.")
    elif model_choice == "Option Greeks":
        st.write("Option Greeks help measure the sensitivity of an option's price to different factors like stock price changes, volatility, and time decay.")

