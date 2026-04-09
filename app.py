"""
app.py - Perfection for ANUJ - Minimalist & Premium
"""

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import os
import threading
import time
import requests

# Load environment variables
load_dotenv()

# Local imports
from data_fetcher import fetch_all_assets, get_top_movers
from news import display_finance_news
from chat import chat_interface, upload_document
from budgeting import budgeting_tool
from technical_analysis import parse_technical_indicators

# Get API keys
AV_API_KEY = os.getenv("AV_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL")

# Keep-alive logic for Render
@st.cache_resource
def start_keep_alive():
    if RENDER_URL:
        def ping_loop():
            while True:
                try: requests.get(RENDER_URL)
                except Exception: pass
                time.sleep(600)
        thread = threading.Thread(target=ping_loop, daemon=True)
        thread.start()
        return True
    return False

keep_alive_active = start_keep_alive()

# Page configuration
st.set_page_config(
    page_title="ANUJ'S FinAI",
    page_icon="👑",
    layout="wide"
)

# Initialize session state variables
if 'financial_data' not in st.session_state: st.session_state['financial_data'] = ''
if 'chat_history' not in st.session_state: st.session_state['chat_history'] = []
if 'asset_data' not in st.session_state:
    st.session_state['asset_data'] = []
    st.session_state['asset_data_timestamp'] = None

# CSS - THE POWER RED & WHITE THEME
st.markdown("""
<style>
    /* 1. Reset and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@300;400;600;800&display=swap');
    
    .stApp {
        background: #ffffff !important;
        color: #1a1a1a !important;
    }

    /* Force all text to readable dark color */
    .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, [data-testid="stMarkdownContainer"] {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Fix for icons */
    [data-testid="stSidebar"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* 2. HEADER - BOLD RED */
    .hero-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 4.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(90deg, #FF0000, #8B0000) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center !important;
        margin-top: -40px !important;
        margin-bottom: 0px !important;
        letter-spacing: -2px !important;
    }
    
    .hero-subtitle {
        text-align: center !important;
        color: #FF0000 !important;
        font-size: 1rem !important;
        letter-spacing: 5px !important;
        margin-bottom: 40px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
    }

    /* 3. SIDEBAR RED DESIGN */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 2px solid #FF0000 !important;
    }
    
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h1 {
        color: #FF0000 !important;
    }
    
    /* Ensure sidebar text is readable */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #1a1a1a !important;
    }

    /* 4. CHAT BUBBLES - WHITE WITH RED BORDERS */
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border-radius: 15px !important;
        border: 1px solid #eeeeee !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02) !important;
        padding: 20px !important;
        margin: 10px 0 !important;
    }
    
    /* User message slightly different */
    [data-testid="stChatMessage"]:nth-child(even) {
        border-left: 5px solid #FF0000 !important;
    }

    /* 5. RED POWER BUTTONS */
    .stButton>button {
        width: 100% !important;
        border-radius: 8px !important;
        background: linear-gradient(90deg, #FF0000, #8B0000) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        text-transform: uppercase !important;
    }
    
    .stButton>button:hover {
        background: #000000 !important;
        color: white !important;
    }

    /* Metric refinement */
    [data-testid="stMetricValue"] {
        color: #FF0000 !important;
        font-weight: 800 !important;
    }
    
    /* Input border focus */
    .stTextArea textarea:focus {
        border-color: #FF0000 !important;
    }

    /* Hide Streamlit components */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# THE PERFECT BRANDING
st.markdown('<h1 class="hero-title">👑 ANUJ\'S FINANCE AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Mera Wala Custom Wealth Agent</p>', unsafe_allow_html=True)

# Sidebar - Clean & Proper
with st.sidebar:
    st.title("📂 Control Panel")
    st.divider()
    
    # Navigation
    menu = st.radio(
        "SELECT VIEW",
        ["📚 Document Analyst", "🌐 Market Intelligence", "📊 Market Hub", "🛠️ Wealth Tools"],
        index=0
    )
    
    st.divider()
    
    # Financial Profile
    st.subheader("👤 User Profile")
    financial_data_input = st.text_area(
        "Your Financial Summary",
        value=st.session_state['financial_data'],
        height=150,
        placeholder="E.g. Income: 1L, Savings: 20k, Goal: House..."
    )
    if st.button("💾 SAVE PROFILE", use_container_width=True):
        st.session_state['financial_data'] = financial_data_input
        st.success("Profile Locked! ✅")
    
    st.divider()
    
    # Knowledge Base (Vault) - Always available for any mode
    st.subheader("📚 Finance Vault")
    upload_document()
    
    if keep_alive_active:
        st.divider()
        st.caption("🟢 System Live (Keep-Alive)")

# Navigation Routing
if menu == "📚 Document Analyst":
    chat_interface(mode="pdf")
    
elif menu == "🌐 Market Intelligence":
    chat_interface(mode="news")
    
elif menu == "📊 Market Hub":
    st.header("📈 Market Intelligence Hub")
    
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("REFRESH DATA"):
            with st.spinner("Updating..."):
                st.session_state['asset_data'] = fetch_all_assets()
                st.session_state['asset_data_timestamp'] = datetime.now().strftime('%H:%M:%S')
    with c2:
        if st.session_state['asset_data_timestamp']:
            st.info(f"Market Sync: {st.session_state['asset_data_timestamp']}")
            
    if st.session_state['asset_data']:
        st.dataframe(pd.DataFrame(st.session_state['asset_data']), use_container_width=True)
    else:
        st.warning("Hit Refresh to populate the market terminal.")
        
    st.divider()
    
    n1, n2 = st.columns(2)
    with n1:
        st.subheader("📰 Market Headlines")
        display_finance_news()
    with n2:
        st.subheader("🚀 High Velocity Crypto")
        tm = get_top_movers()
        if tm: st.dataframe(pd.DataFrame(tm), use_container_width=True)

elif menu == "🛠️ Wealth Tools":
    budgeting_tool()

# FOOTER
st.markdown("---")
st.markdown('<div style="text-align:center; color:#555; font-weight:300;">© 2026 | CUSTOM BUILT FOR ANUJ</div>', unsafe_allow_html=True)
