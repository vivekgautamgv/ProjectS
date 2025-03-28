import streamlit as st
from models.black_scholes import black_scholes
from models.monte_carlo import monte_carlo_option

def show():
    st.title("📈 Options Pricing Models")

    # User inputs
    S = st.number_input("Stock Price (S)", min_value=1.0, value=100.0)
    K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0)
    T = st.number_input("Time to Expiration (T in years)", min_value=0.01, value=1.0)
    r = st.number_input("Risk-Free Interest Rate (r)", min_value=0.0, value=0.05)
    sigma = st.number_input("Volatility (σ)", min_value=0.01, value=0.2)
    option_type = st.radio("Option Type", ["call", "put"])
    model = st.radio("Choose Model", ["Black-Scholes", "Monte Carlo"])

    if st.button("Calculate Option Price"):
        if model == "Black-Scholes":
            price = black_scholes(S, K, T, r, sigma, option_type)
            st.success(f"Black-Scholes Price: ${price:.2f}")
        else:
            price = monte_carlo_option(S, K, T, r, sigma, num_simulations=10000, option_type=option_type)
            st.success(f"Monte Carlo Price: ${price:.2f}")
