"""
Chat Interface - 2026 RAG-Powered Personal Finance Assistant
Full integration with RAG pipeline, multi-LLM, and web search
"""

import streamlit as st
import re
import yfinance as yf
import os
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Import our modules
from rag_pipeline import get_rag_pipeline, initialize_rag
from llm_handler import get_llm
from tavily_search import get_tavily_search, is_tavily_available

load_dotenv()


# System prompt for the finance assistant
SYSTEM_PROMPT = """You are an expert Personal Finance Assistant powered by advanced RAG (Retrieval-Augmented Generation) technology. You provide accurate, helpful, and personalized financial advice.

## Your Capabilities:
1. **Document Knowledge**: You have access to financial books and documents uploaded by the user
2. **Real-time Data**: You can access current market data and financial news
3. **Personalized Advice**: You consider the user's financial situation when giving advice

## Guidelines:
- Always cite your sources when using retrieved information
- Be specific and actionable in your advice
- Remind users to consult professionals for major financial decisions
- If you're unsure, say so rather than making things up
- Format your responses clearly with headers and bullet points when appropriate

## Response Format:
- Start with a direct answer to the question
- Provide supporting details and context
- Include relevant citations from documents when applicable
- End with actionable next steps if appropriate"""


def get_financial_context() -> str:
    """Get user's financial context from session state"""
    if st.session_state.get('financial_data'):
        return f"User's Financial Situation:\n{st.session_state['financial_data']}"
    return ""


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


def generate_assistant_response(user_input: str) -> tuple:
    """
    Generate RAG-powered response
    Returns: (response_text, sources_list, provider_used)
    """
    # Gather all context
    financial_context = get_financial_context()
    rag_context, rag_sources = get_rag_context(user_input)
    web_context = get_web_context(user_input)
    
    # Build conversation history
    conversation_history = ""
    if len(st.session_state.get('chat_history', [])) > 1:
        recent = st.session_state['chat_history'][-6:]  # Last 6 messages
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:500]  # Truncate for context window
            conversation_history += f"{role}: {content}\n"
    
    # Construct the prompt
    context_parts = []
    
    if financial_context:
        context_parts.append(f"## User's Financial Profile\n{financial_context}")
    
    if rag_context:
        context_parts.append(f"## Retrieved Document Context\n{rag_context}")
    
    if web_context:
        context_parts.append(f"## Current Web Information\n{web_context}")
    
    if conversation_history:
        context_parts.append(f"## Recent Conversation\n{conversation_history}")
    
    full_context = "\n\n".join(context_parts) if context_parts else "No additional context available."
    
    user_prompt = f"""Based on the following context, please answer the user's question.

{full_context}

---

**User Question**: {user_input}

Please provide a helpful, accurate response. If you use information from the retrieved documents, mention the source."""

    # Get LLM response
    llm = get_llm()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm.chat(messages, temperature=0.7, max_tokens=1024)
    provider = llm.last_provider or "unknown"
    
    return response, rag_sources, provider


def chat_interface():
    """Main chat interface with RAG integration"""
    st.header("💬 Chat with Your Personal Finance Assistant")
    
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
    
    # Show RAG status in expander
    with st.expander("📊 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            stats = st.session_state.get('rag_stats', {})
            st.metric("📚 Documents Indexed", stats.get('total_documents', 0))
        
        with col2:
            llm = get_llm()
            providers = llm.get_available_providers()
            st.metric("🤖 LLM Providers", len(providers))
            st.caption(", ".join(providers) if providers else "None")
        
        with col3:
            tavily_status = "✅ Active" if is_tavily_available() else "❌ Not configured"
            st.metric("🌐 Web Search", tavily_status)
    
    # Display chat history
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            # Show sources if available
            if message.get('sources'):
                with st.expander("📖 Sources"):
                    for source in message['sources']:
                        st.caption(f"• {source}")
            
            # Show chart if available
            if message.get('chart_data') is not None:
                st.line_chart(message['chart_data'])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about personal finance...")
    
    if user_input:
        # Add user message
        st.session_state['chat_history'].append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                # Check for chart request
                chart_data = display_chart_for_asset(user_input)
                
                # Generate RAG response
                response, sources, provider = generate_assistant_response(user_input)
                
                # Display response
                st.markdown(response)
                
                # Show sources
                if sources:
                    with st.expander("📖 Sources Used"):
                        for source in sources:
                            st.caption(f"• {source}")
                
                # Show chart if applicable
                if chart_data is not None:
                    st.line_chart(chart_data)
                
                # Show provider info
                st.caption(f"_Powered by {provider.upper()}_")
        
        # Save assistant message
        assistant_message = {
            "role": "assistant",
            "content": response,
            "sources": sources,
            "provider": provider
        }
        
        if chart_data is not None:
            assistant_message['chart_data'] = chart_data
        
        st.session_state['chat_history'].append(assistant_message)
        
        # Rerun to update UI
        st.rerun()


def upload_document():
    """Handle document upload for RAG"""
    st.subheader("📄 Upload Financial Documents")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF to add to your knowledge base",
        type=['pdf'],
        help="Upload financial documents, statements, or books to get personalized insights"
    )
    
    if uploaded_file is not None:
        # Save temporarily
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
