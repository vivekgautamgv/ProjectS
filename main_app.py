# main_app.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

st.set_page_config(
    page_title="Trading Analytics Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic sidebar configuration
st.sidebar.header("Global Parameters")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.5) / 100