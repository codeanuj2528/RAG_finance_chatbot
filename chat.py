"""
Chat Interface - 2026 RAG-Powered Personal Finance Assistant
Full integration with RAG pipeline, multi-LLM, and web search
"""

import streamlit as st
import re
import yfinance as yf
import os
from typing import Optional, List, Dict
import pandas as pd
from dotenv import load_dotenv

# Import our modules
from rag_pipeline import get_rag_pipeline, initialize_rag
from llm_handler import get_llm
from tavily_search import get_tavily_search, is_tavily_available

load_dotenv()


# Specialized System Prompts
DOC_ANALYST_PROMPT = """You are an expert Document Analysis Specialist. 
Your goal is to answer questions EXCLUSIVELY based on the provided PDF context. 
If the information is not in the documents, state that clearly. 
Cite filenames and specific details from the retrieved chunks."""

MARKET_INTEL_PROMPT = """You are a Market Intelligence Expert. 
Your goal is to analyze real-time news, stock data, and market trends. 
Provide a 'News-Based Guidance' (EME analysis) to guide the user on potential market moves. 
Focus on data from Tavily, NewsAPI, and YFinance."""

SYSTEM_PROMPT = """You are an expert Personal Finance Assistant. 
Provide accurate, helpful, and professional advice based on the available context."""


def get_financial_context() -> str:
    """Get user's financial context and current asset data from session state"""
    context = []
    
    if st.session_state.get('financial_data'):
        context.append(f"## User's Financial Profile:\n{st.session_state['financial_data']}")
        
    if st.session_state.get('asset_data'):
        assets_df = pd.DataFrame(st.session_state['asset_data'])
        assets_str = assets_df.to_string(index=False)
        context.append(f"## Current Portfolio/Market Data:\n{assets_str}")
        
    return "\n\n".join(context)


def get_rag_context(query: str) -> tuple:
    """
    Get RAG context for the query
    Returns: (context_string, sources_list)
    """
    try:
        pipeline = get_rag_pipeline()
        context, sources = pipeline.get_context_for_llm(query, top_k=5)
        return context, sources
    except Exception as e:
        print(f"RAG context error: {e}")
        return "", []


def get_web_context(query: str) -> str:
    """Get web search context for current information"""
    if not is_tavily_available():
        return ""
    
    try:
        # Check if query needs real-time info
        needs_realtime = any(word in query.lower() for word in [
            'today', 'current', 'latest', 'now', 'recent', 'news',
            'price', 'market', 'stock', 'crypto', 'bitcoin'
        ])
        
        if needs_realtime:
            tavily = get_tavily_search()
            return tavily.get_context_for_query(query, max_results=3)
        
        return ""
    except Exception as e:
        print(f"Web context error: {e}")
        return ""


def display_chart_for_asset(message: str) -> Optional[object]:
    """Extract ticker from message and return price chart data"""
    pattern = r'\b(?:price|chart|stock)\s+(?:of\s+)?([A-Za-z0-9.\-]+)\b'
    matches = re.findall(pattern, message, re.IGNORECASE)
    
    if matches:
        ticker = matches[0].upper()
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if not hist.empty:
                return hist['Close']
        except Exception as e:
            print(f"Chart error for {ticker}: {e}")
    
    return None


def generate_assistant_response(user_input: str, mode: str = "general") -> tuple:
    """
    Generate specialized response based on mode
    Returns: (response_text, sources_list, provider_used)
    """
    financial_context = get_financial_context()
    rag_context, rag_sources = "", []
    web_context = ""
    
    # Context Filtering based on Mode
    if mode == "pdf":
        rag_context, rag_sources = get_rag_context(user_input)
        sys_prompt = DOC_ANALYST_PROMPT
    elif mode == "news":
        web_context = get_web_context(user_input)
        sys_prompt = MARKET_INTEL_PROMPT
    else:
        rag_context, rag_sources = get_rag_context(user_input)
        web_context = get_web_context(user_input)
        sys_prompt = SYSTEM_PROMPT

    # Build conversation history (Mode-specific history is handled in chat_interface)
    conversation_history = ""
    history_key = f'chat_history_{mode}'
    if len(st.session_state.get(history_key, [])) > 1:
        recent = st.session_state[history_key][-6:]
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            conversation_history += f"{role}: {msg['content'][:500]}\n"
    
    # Construct the prompt
    context_parts = [f"## User's Financial Profile\n{financial_context}"]
    if rag_context: context_parts.append(f"## Retrieved Document Context\n{rag_context}")
    if web_context: context_parts.append(f"## Current Web & News Information\n{web_context}")
    if conversation_history: context_parts.append(f"## Recent Conversation\n{conversation_history}")
    
    full_context = "\n\n".join(context_parts)
    
    user_prompt = f"""Based on the following context, please answer the user's question.

{full_context}

---

**User Question**: {user_input}

Please provide a helpful, accurate response. If you use information from the retrieved documents, mention the source."""

    # Get LLM response
    llm = get_llm()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {user_input}"}
    ]
    
    response = llm.chat(messages, temperature=0.7, max_tokens=1024)
    provider = llm.last_provider or "unknown"
    return response, rag_sources, provider


def chat_interface(mode: str = "general"):
    """Main chat interface with specialized mode support"""
    header_name = "ANUJ'S Intelligence Terminal"
    if mode == "pdf": header_name = "📚 Document Analysis Expert"
    elif mode == "news": header_name = "🌐 Market Intelligence Expert"
    
    st.header(f"💬 {header_name}")
    
    # Initialize RAG on first load with error handling
    if 'rag_initialized' not in st.session_state:
        try:
            with st.spinner("🔄 Connecting to knowledge base..."):
                stats = initialize_rag()
                st.session_state['rag_initialized'] = True
                st.session_state['rag_stats'] = stats
        except Exception as e:
            st.warning(f"⚠️ RAG system loading... This may take a moment on first run.")
            st.session_state['rag_initialized'] = True
            st.session_state['rag_stats'] = {"total_documents": 0, "error": str(e)}
    
    # Separate histories by mode
    history_key = f'chat_history_{mode}'
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    # Display chat history
    for message in st.session_state[history_key]:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message.get('sources'):
                with st.expander("📖 Sources"):
                    for source in message['sources']: st.caption(f"• {source}")
            if message.get('chart_data') is not None:
                st.line_chart(message['chart_data'])
    
    # Chat input
    prompt_text = "Ask about your PDFs..." if mode == "pdf" else "Ask about market news & data..."
    user_input = st.chat_input(prompt_text)
    
    if user_input:
        st.session_state[history_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing..."):
                chart_data = display_chart_for_asset(user_input)
                response, sources, provider = generate_assistant_response(user_input, mode=mode)
                st.markdown(response)
                if sources:
                    with st.expander("📖 Sources Used"):
                        for source in sources: st.caption(f"• {source}")
                if chart_data is not None: st.line_chart(chart_data)
                st.caption(f"_Powered by {provider.upper()}_")
        
        assistant_message = {
            "role": "assistant",
            "content": response,
            "sources": sources,
            "provider": provider,
            "chart_data": chart_data
        }
        st.session_state[history_key].append(assistant_message)
        st.rerun()


def upload_document():
    """Handle document upload for RAG"""
    # Header removed for cleaner sidebar integration
    
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=['pdf'],
        help="Upload financial documents to get personalized insights",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        # Save temporarily
        os.makedirs("data", exist_ok=True)
        temp_path = f"./data/{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"📚 Processing {uploaded_file.name}..."):
            pipeline = get_rag_pipeline()
            chunks = pipeline.add_pdf(temp_path)
            
            if chunks > 0:
                st.success(f"✅ Added {uploaded_file.name} ({chunks} chunks)")
                # Update stats
                st.session_state['rag_stats'] = pipeline.get_collection_stats()
            else:
                st.error("Failed to process the document")


# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Finance Chat Test", page_icon="💬")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    chat_interface()
