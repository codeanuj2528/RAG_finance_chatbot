"""
chat.py - Complete RAG Implementation with OpenAI + Groq Fallback
Enhanced with web crawling via Tavily
"""

import streamlit as st
import re
import yfinance as yf
import os
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# ============================================
# API Client Setup with Fallback
# ============================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Primary: OpenAI
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Fallback: Groq
groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except ImportError:
        st.warning("Groq package not installed. Install with: pip install groq")

# Web Crawler: Tavily
tavily_client = None
if TAVILY_API_KEY:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        pass  # Tavily is optional

# ============================================
# In-Memory Vector Store (Simple Implementation)
# ============================================

# Store documents in session state
if 'document_chunks' not in st.session_state:
    st.session_state['document_chunks'] = []
if 'document_embeddings' not in st.session_state:
    st.session_state['document_embeddings'] = []

# ============================================
# PDF Processing Functions
# ============================================

def process_pdf(pdf_file):
    """Extract text from PDF and split into chunks"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text.strip():
            st.warning("PDF mein text nahi mila. Scanned PDF ho sakta hai.")
            return []

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)

        st.success(f"✅ PDF processed! Total chunks: {len(chunks)}")
        return chunks

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return []

# ============================================
# Embedding Functions
# ============================================

def get_embeddings(text_chunks):
    """Generate OpenAI embeddings for text chunks"""
    embeddings = []
    
    if not openai_client:
        st.warning("OpenAI API not configured. Using simple text matching.")
        return []
    
    try:
        progress_bar = st.progress(0)
        for i, chunk in enumerate(text_chunks):
            response = openai_client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
            progress_bar.progress((i + 1) / len(text_chunks))

        progress_bar.empty()
        st.success(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings

    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return []

def store_documents(chunks, embeddings):
    """Store chunks and embeddings in session state"""
    st.session_state['document_chunks'] = chunks
    st.session_state['document_embeddings'] = embeddings
    st.success(f"✅ Stored {len(chunks)} chunks")
    return True

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0

def similarity_search_docs(query: str, top_k: int = 3):
    """Search for similar documents using query embedding"""
    try:
        chunks = st.session_state.get('document_chunks', [])
        embeddings = st.session_state.get('document_embeddings', [])
        
        if not chunks:
            return []
        
        # If no embeddings, use simple keyword matching
        if not embeddings or not openai_client:
            query_lower = query.lower()
            scored = [(chunk, sum(1 for word in query_lower.split() if word in chunk.lower())) 
                      for chunk in chunks]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, score in scored[:top_k] if score > 0]
        
        # Use embedding similarity
        import numpy as np
        query_response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding
        
        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [chunks[i] for i in top_indices if similarities[i] > 0.5]

    except Exception as e:
        st.error(f"Error in similarity search: {e}")
        return []

# ============================================
# Web Search with Tavily
# ============================================

def search_web(query: str, max_results: int = 3):
    """Search the web using Tavily API"""
    if not tavily_client:
        return None
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=max_results
        )
        
        if response and 'results' in response:
            return "\n\n".join([
                f"**{r.get('title', 'No Title')}**\n{r.get('content', '')[:500]}"
                for r in response['results']
            ])
        return None
    except Exception as e:
        return None

# ============================================
# PDF Upload Section
# ============================================

def pdf_upload_section():
    """PDF upload and processing UI"""
    st.subheader("📄 Upload Financial Documents")

    uploaded_file = st.file_uploader(
        "Upload your financial documents (PDF)",
        type=['pdf'],
        help="Upload bank statements, investment plans, or any financial documents"
    )

    if uploaded_file is not None:
        if st.button("📥 Process Document", key="process_pdf"):
            with st.spinner("Processing PDF..."):
                chunks = process_pdf(uploaded_file)

                if chunks:
                    with st.spinner("Generating embeddings..."):
                        embeddings = get_embeddings(chunks)
                    
                    with st.spinner("Storing in database..."):
                        success = store_documents(chunks, embeddings)

                        if success:
                            st.balloons()
                            st.success("🎉 Document processed and stored!")
                            st.info("Ab aap chat mein questions puch sakte ho based on your document.")

                            if 'processed_docs' not in st.session_state:
                                st.session_state['processed_docs'] = []
                            st.session_state['processed_docs'].append(uploaded_file.name)

# ============================================
# LLM Response Generation with Fallback
# ============================================

def call_openai(messages, max_tokens=500):
    """Call OpenAI API"""
    if not openai_client:
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"OpenAI error: {e}")
        return None

def call_groq(messages, max_tokens=500):
    """Call Groq API as fallback"""
    if not groq_client:
        return None
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"Groq error: {e}")
        return None

def generate_assistant_response_with_rag(user_input: str) -> str:
    """Generate response using RAG with fallback LLMs"""
    try:
        # Get relevant docs from uploaded documents
        relevant_docs = similarity_search_docs(user_input, top_k=3)
        
        # Build document context
        document_context = ""
        if relevant_docs:
            document_context = "\n\n".join(relevant_docs)
            document_context = f"\n\nUser's Uploaded Documents Context:\n{document_context}\n"

        # Get web search results for current info
        web_context = ""
        if tavily_client and any(word in user_input.lower() for word in ['latest', 'current', 'today', 'news', 'market']):
            web_results = search_web(f"finance {user_input}")
            if web_results:
                web_context = f"\n\nLatest Web Information:\n{web_results}\n"

        # Financial data from session state
        financial_data = st.session_state.get('financial_data', '')
        financial_context = ""
        if financial_data:
            financial_context = f"\n\nUser's Financial Data:\n{financial_data}\n"

        # Conversation history
        conversation_context = ""
        if len(st.session_state['chat_history']) > 1:
            recent_messages = st.session_state['chat_history'][-4:]
            for msg in recent_messages:
                role = "Human" if msg['role'] == 'user' else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"

        # Build prompt
        prompt = f"""You are a knowledgeable personal finance assistant. Help users with financial questions, investment advice, budgeting, and market analysis.
{document_context}{web_context}{financial_context}
Previous conversation:
{conversation_context}

Current question: {user_input}

Provide helpful, accurate financial guidance based on the user's documents and data. If asked about specific stocks or investments, provide general analysis but remind users to do their own research and consider their risk tolerance."""

        messages = [
            {"role": "system", "content": "You are a helpful personal finance assistant with access to user's financial documents."},
            {"role": "user", "content": prompt}
        ]

        # Try OpenAI first, then Groq as fallback
        response = call_openai(messages)
        if response:
            return response
        
        response = call_groq(messages)
        if response:
            return f"[via Groq] {response}"
        
        # If both fail, use fallback
        return generate_fallback_response(user_input)

    except Exception as e:
        st.error(f"Error with RAG response: {e}")
        return generate_fallback_response(user_input)

# ============================================
# Chart Display
# ============================================

def display_chart_for_asset(message: str):
    """Extract ticker from message and return price chart data"""
    pattern = r'\b(?:price|chart)\s+(?:of\s+)?([A-Za-z0-9.\-]+)\b'
    matches = re.findall(pattern, message, re.IGNORECASE)

    if matches:
        ticker = matches[0].upper()
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            if not hist.empty:
                return hist['Close']
            else:
                st.write(f"No data found for ticker {ticker}")
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")

    return None

# ============================================
# Main Response Generator
# ============================================

def generate_assistant_response(user_input: str) -> str:
    """Generate response using OpenAI/Groq API or fallback"""
    if OPENAI_API_KEY or GROQ_API_KEY:
        try:
            return generate_assistant_response_with_rag(user_input)
        except Exception as e:
            st.error(f"Error with API: {e}")
            return generate_fallback_response(user_input)
    else:
        return generate_fallback_response(user_input)

def generate_fallback_response(user_input: str) -> str:
    """Generate rule-based responses when APIs are not available"""
    user_input_lower = user_input.lower()

    if any(word in user_input_lower for word in ['budget', 'budgeting', 'expense', 'spending']):
        return """Great question about budgeting! Here are some key tips:

• Follow the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings
• Track your expenses for at least a month to understand your spending patterns
• Use the budgeting tool in the 'Tools' tab to analyze your current situation
• Set up automatic transfers to savings accounts
• Review and adjust your budget monthly

Would you like me to help you with any specific aspect of budgeting?"""

    elif any(word in user_input_lower for word in ['invest', 'investment', 'stock', 'portfolio']):
        return """Here's some general investment guidance:

• Diversification is key - don't put all eggs in one basket
• Consider your risk tolerance and investment timeline
• Low-cost index funds are great for beginners
• Dollar-cost averaging can help reduce timing risk
• Emergency fund should come before investing
• Consider tax-advantaged accounts (401k, IRA)

Remember: This is general advice. Always do your own research!"""

    elif any(word in user_input_lower for word in ['save', 'saving', 'savings', 'emergency fund']):
        return """Savings tips and strategies:

• Build an emergency fund of 3-6 months of expenses first
• Automate your savings - pay yourself first
• Use high-yield savings accounts for better returns
• Set specific savings goals with deadlines
• Consider the envelope method for different categories
• Take advantage of employer 401k matching

Start small if needed - even $25/month builds the habit!"""

    elif any(word in user_input_lower for word in ['debt', 'loan', 'credit card', 'payment']):
        return """Debt management strategies:

• List all debts with balances, minimum payments, and interest rates
• Consider debt avalanche (highest rate first) or debt snowball (smallest balance first)
• Negotiate with creditors if you're struggling
• Avoid taking on new debt while paying off existing debt
• Consider debt consolidation if it lowers your rates
• Build a small emergency fund even while paying debt

The key is consistency and having a clear plan!"""

    elif any(word in user_input_lower for word in ['market', 'economy', 'recession', 'inflation']):
        return """Market and economic considerations:

• Markets are cyclical - ups and downs are normal
• Stay focused on long-term goals, not short-term volatility
• Inflation erodes purchasing power - consider assets that outpace inflation
• Don't try to time the market - consistent investing tends to win
• Keep some cash for opportunities during market downturns
• Review and rebalance your portfolio periodically

Remember: Market timing is extremely difficult even for professionals!"""

    else:
        return f"""Thanks for your question about "{user_input}". As your personal finance assistant, I'm here to help with:

• Budgeting and expense tracking
• Investment strategies and portfolio advice
• Savings goals and emergency funds
• Debt management
• Financial planning
• Market analysis and stock information

You can also:
• Check the 'Assets' tab for current market data
• Use the 'Tools' tab for budgeting calculators
• View the 'News' tab for latest financial news
• Upload your financial documents for personalized advice

Could you be more specific about what aspect of personal finance you'd like help with?"""

# ============================================
# Chat Interface
# ============================================

def chat_interface():
    st.header("💬 Chat with Your Personal Finance Assistant")

    # PDF Upload Section
    with st.expander("📄 Upload Financial Documents (RAG)", expanded=False):
        pdf_upload_section()

        # Show processed documents
        if 'processed_docs' in st.session_state and st.session_state['processed_docs']:
            st.write("**Processed Documents:**")
            for doc in st.session_state['processed_docs']:
                st.write(f"✅ {doc}")

    # Display chat messages
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'chart_data' in message:
                st.line_chart(message['chart_data'])

    # Chat input
    user_input = st.chat_input("Ask me about personal finance...")

    if user_input:
        # Add user message
        st.session_state['chat_history'].append({"role": "user", "content": user_input})

        # Check for chart request
        chart_data = display_chart_for_asset(user_input)

        # Generate response
        with st.spinner("Thinking..."):
            assistant_response = generate_assistant_response(user_input)

        assistant_message = {"role": "assistant", "content": assistant_response}
        if chart_data is not None:
            assistant_message['chart_data'] = chart_data

        st.session_state['chat_history'].append(assistant_message)

        # Rerun to display new messages
        st.rerun()
