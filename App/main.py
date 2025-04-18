import streamlit as st
import home
import stock_screen
import options_pricing
import option_chain_analysis  # Ensure this file exists!


st.set_page_config(page_title="Stock & Options Analysis", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select a Page:", ["Home", "Stock Screener", "Options Pricing", "Option Chain Analysis"])

#Page Routing
if page == "Home":
    home.show()  
elif page == "Stock Screener":
    stock_screen.show() 
elif page == "Options Pricing":
    options_pricing.show() 
elif page == "Option Chain Analysis":
    option_chain_analysis.show() 
