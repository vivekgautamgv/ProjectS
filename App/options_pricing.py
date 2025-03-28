import streamlit as st
import numpy as np
import scipy.stats as si
import math

def show():  # ✅ Corrected show() function
    st.title("📈 Options Pricing Models")

    # 📌 Sidebar for Inputs
    st.sidebar.header("Enter Option Parameters")
    S = st.sidebar.number_input("Stock Price (S)", value=100.0, step=1.0)
    K = st.sidebar.number_input("Strike Price (K)", value=100.0, step=1.0)
    T = st.sidebar.number_input("Time to Expiration (Years, T)", value=1.0, step=0.1)
    r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05, step=0.01)
    sigma = st.sidebar.number_input("Volatility (σ)", value=0.2, step=0.01)
    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    # 📌 Select Model
    model_choice = st.sidebar.radio("Select Pricing Model:", ["Black-Scholes", "Heston Model", "Binary Options"])

    # 📌 Black-Scholes Model
    def black_scholes(S, K, T, r, sigma, option_type):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "Call":
            price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)

        return round(price, 2)

    # 📌 Heston Model (Simplified)
    def heston_model(S, K, T, r, sigma, kappa=2, theta=0.2):
        """Simplified Heston Model Approximation"""
        v_t = sigma**2  # Initial variance
        mean_variance = theta  # Long-term variance
        price = black_scholes(S, K, T, r, np.sqrt(v_t), option_type) + (kappa * (mean_variance - v_t))
        return round(price, 2)

    # 📌 Binary Options Model
    def binary_option(S, K, T, r, sigma, option_type):
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        if option_type == "Call":
            price = np.exp(-r * T) * si.norm.cdf(d2)
        else:
            price = np.exp(-r * T) * si.norm.cdf(-d2)
        return round(price, 2)

    # 📌 Calculate & Display Result
    st.subheader("📊 Pricing Result")

    if model_choice == "Black-Scholes":
        price = black_scholes(S, K, T, r, sigma, option_type)
        st.success(f"Black-Scholes {option_type} Option Price: **${price}**")

    elif model_choice == "Heston Model":
        price = heston_model(S, K, T, r, sigma)
        st.success(f"Heston Model {option_type} Option Price: **${price}**")

    elif model_choice == "Binary Options":
        price = binary_option(S, K, T, r, sigma, option_type)
        st.success(f"Binary {option_type} Option Price: **${price}**")

    # 📌 Explanation
    st.subheader("📚 Model Explanation")
    if model_choice == "Black-Scholes":
        st.write("The **Black-Scholes Model** is used for pricing European call and put options.")
    elif model_choice == "Heston Model":
        st.write("The **Heston Model** accounts for stochastic volatility in option pricing.")
    elif model_choice == "Binary Options":
        st.write("Binary options pay a fixed amount if the option expires in-the-money.")

