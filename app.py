"""
app.py - Main Streamlit Application for Personal Finance Assistant
Enhanced with Groq fallback and Tavily web crawling
"""

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import os

# Load environment variables
load_dotenv()

# Local imports
from data_fetcher import fetch_all_assets, get_top_movers
from news import display_finance_news
from chat import chat_interface
from budgeting import budgeting_tool
from technical_analysis import parse_technical_indicators

# Get API keys
AV_API_KEY = os.getenv("AV_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Personal Finance Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize session state variables
if 'financial_data' not in st.session_state:
    st.session_state['financial_data'] = ''

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if 'asset_data' not in st.session_state:
    st.session_state['asset_data'] = []
    st.session_state['asset_data_timestamp'] = None

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .api-status {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    .api-ok { background-color: #d4edda; color: #155724; }
    .api-missing { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<p class="main-header">💰 Personal Finance Assistant</p>', unsafe_allow_html=True)
st.markdown("---")

# Button to load all asset data from Alpha Vantage
col1, col2 = st.columns([8, 2])
with col2:
    if st.session_state['asset_data_timestamp']:
        st.write(f"**Last Updated:**")
        st.write(f"{st.session_state['asset_data_timestamp']}")
    else:
        st.write("**Data not loaded.**")
    
    if st.button("🔄 Update Data", use_container_width=True):
        with st.spinner("Fetching assets..."):
            st.session_state['asset_data'] = fetch_all_assets()
            st.session_state['asset_data_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.success("Asset data updated!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Status
    st.subheader("API Status")
    
    if OPENAI_API_KEY:
        st.markdown('<div class="api-status api-ok">✅ OpenAI API</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-missing">❌ OpenAI API</div>', unsafe_allow_html=True)
    
    if GROQ_API_KEY:
        st.markdown('<div class="api-status api-ok">✅ Groq API (Fallback)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-missing">⚠️ Groq API (Optional)</div>', unsafe_allow_html=True)
    
    if AV_API_KEY:
        st.markdown('<div class="api-status api-ok">✅ Alpha Vantage API</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="api-status api-missing">❌ Alpha Vantage API</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Financial Data Input
    st.header("📊 Your Financial Data")
    with st.form("financial_data_form"):
        st.write("Enter financial data for personalized advice:")
        financial_data_input = st.text_area(
            "Financial Data",
            value=st.session_state['financial_data'],
            height=150,
            help="Enter income, expenses, investments, goals, etc."
        )
        submitted = st.form_submit_button("💾 Save")
        if submitted:
            st.session_state['financial_data'] = financial_data_input
            st.success("Financial data saved!")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📰 News", "📈 Assets", "💬 Chat", "🛠️ Tools"])

# 1) News Tab
with tab1:
    display_finance_news()

# 2) Assets Tab
with tab2:
    st.header("📈 Asset Data")
    if st.session_state['asset_data']:
        df = pd.DataFrame(st.session_state['asset_data'])
        st.dataframe(df, use_container_width=True)

        # Technical indicator section
        st.subheader("📊 Technical Indicators")
        if st.button("📉 Parse Tech Indicators for 1st Asset"):
            if not df.empty:
                first_ticker = df.iloc[0]['Ticker']
                st.write(f"Parsing indicators for **{first_ticker}**...")
                with st.spinner("Fetching indicators..."):
                    indicators_dict = parse_technical_indicators(first_ticker)
                    if indicators_dict:
                        st.json(indicators_dict)
                    else:
                        st.warning("No indicators available")
    else:
        st.info("📥 No asset data loaded. Click 'Update Data' at the top right to load data.")

    # Crypto top movers
    st.subheader("🚀 Top Cryptocurrency Movers (24h Change)")
    if st.button("🔄 Refresh Crypto Data"):
        with st.spinner("Fetching crypto data..."):
            top_movers = get_top_movers()
            if top_movers:
                st.dataframe(pd.DataFrame(top_movers), use_container_width=True)
            else:
                st.warning("Failed to retrieve cryptocurrency data.")

# 3) Chat Tab
with tab3:
    chat_interface()

# 4) Tools Tab
with tab4:
    budgeting_tool()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.85rem;">
        Personal Finance Assistant • Powered by OpenAI & Groq • 
        <a href="https://github.com/codeanuj2528/RAG_finance_chatbot" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
