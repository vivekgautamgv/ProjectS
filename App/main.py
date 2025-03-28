import streamlit as st
#from app import home, stock_screen, options_pricing, option_chain
import home
import stock_screen
import options_pricing
import option_chain

# Set up page config
st.set_page_config(page_title="Stock & Options Analysis", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Stock Screener", "Options Pricing", "Option Chain Analysis"])

# Route to selected page
if page == "Home":
    home.show()
elif page == "Stock Screener":
    stock_screen.show()
elif page == "Options Pricing":
    options_pricing.show()
elif page == "Option Chain Analysis":
    option_chain.show()
