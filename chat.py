import streamlit as st
import re
import yfinance as yf
import os
from openai import OpenAI

def chat_interface():
    st.header("Chat with Your Personal Finance Assistant")

    # Display chat messages in session state
    for message in st.session_state['chat_history']:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if 'chart_data' in message:
                st.line_chart(message['chart_data'])

    user_input = st.chat_input("You:")
    if user_input:
        # The user's message
        st.session_state['chat_history'].append({"role": "user", "content": user_input})

        # Possibly fetch chart data if user requests: e.g. "price of TSLA"
        chart_data = display_chart_for_asset(user_input)

        # Generate an assistant response
        assistant_response = generate_assistant_response(user_input)
        assistant_message = {"role": "assistant", "content": assistant_response}

        if chart_data is not None:
            assistant_message['chart_data'] = chart_data

        st.session_state['chat_history'].append(assistant_message)

        # Display last two messages
        for msg in st.session_state['chat_history'][-2:]:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
                if 'chart_data' in msg:
                    st.line_chart(msg['chart_data'])

def display_chart_for_asset(message: str):
    """Extract ticker from message and return price chart data"""
    pattern = r'\b(?:price|chart)\s+(?:of\s+)?([A-Za-z0-9.\-]+)\b'
    matches = re.findall(pattern, message, re.IGNORECASE)
    if matches:
        ticker = matches[0].upper()
        stock = yf.Ticker(ticker)
        try:
            hist = stock.history(period="1y")
            if not hist.empty:
                return hist['Close']
            else:
                st.write(f"No data found for ticker {ticker}")
        except Exception as e:
            st.write(f"Error retrieving data for {ticker}: {e}")
    return None

def generate_assistant_response(user_input: str) -> str:
    """Generate response using OpenAI API or fallback to rule-based responses"""
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key:
        try:
            client = OpenAI(api_key=openai_api_key)
            
            # Create context with user's financial data if available
            context = ""
            if st.session_state.get('financial_data'):
                context = f"User's financial context: {st.session_state['financial_data']}\n\n"
            
            # Build conversation history for context
            conversation_context = ""
            if len(st.session_state['chat_history']) > 1:
                recent_messages = st.session_state['chat_history'][-4:]  # Last 4 messages
                for msg in recent_messages:
                    role = "Human" if msg['role'] == 'user' else "Assistant"
                    conversation_context += f"{role}: {msg['content']}\n"
            
            prompt = f"""You are a knowledgeable personal finance assistant. Help users with financial questions, investment advice, budgeting, and market analysis.

{context}Previous conversation:
{conversation_context}

Current question: {user_input}

Please provide helpful, accurate financial guidance. If asked about specific stocks or investments, provide general analysis but remind users to do their own research and consider their risk tolerance."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful personal finance assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Error with OpenAI API: {e}")
            return generate_fallback_response(user_input)
    else:
        return generate_fallback_response(user_input)

def generate_fallback_response(user_input: str) -> str:
    """Generate rule-based responses when OpenAI API is not available"""
    user_input_lower = user_input.lower()
    
    # Budget-related responses
    if any(word in user_input_lower for word in ['budget', 'budgeting', 'expense', 'spending']):
        return """Great question about budgeting! Here are some key tips:

• Follow the 50/30/20 rule: 50% for needs, 30% for wants, 20% for savings
• Track your expenses for at least a month to understand your spending patterns
• Use the budgeting tool in the 'Tools' tab to analyze your current situation
• Set up automatic transfers to savings accounts
• Review and adjust your budget monthly

Would you like me to help you with any specific aspect of budgeting?"""
    
    # Investment-related responses
    elif any(word in user_input_lower for word in ['invest', 'investment', 'stock', 'portfolio']):
        return """Here's some general investment guidance:

• Diversification is key - don't put all eggs in one basket
• Consider your risk tolerance and investment timeline
• Low-cost index funds are great for beginners
• Dollar-cost averaging can help reduce timing risk
• Emergency fund should come before investing
• Consider tax-advantaged accounts (401k, IRA)

Remember: This is general advice. Always do your own research and consider consulting with a financial advisor for personalized guidance."""
    
    # Savings-related responses
    elif any(word in user_input_lower for word in ['save', 'saving', 'savings', 'emergency fund']):
        return """Savings tips and strategies:

• Build an emergency fund of 3-6 months of expenses first
• Automate your savings - pay yourself first
• Use high-yield savings accounts for better returns
• Set specific savings goals with deadlines
• Consider the envelope method for different savings categories
• Take advantage of employer 401k matching

Start small if needed - even $25/month builds the habit and adds up over time!"""
    
    # Debt-related responses
    elif any(word in user_input_lower for word in ['debt', 'loan', 'credit card', 'payment']):
        return """Debt management strategies:

• List all debts with balances, minimum payments, and interest rates
• Consider debt avalanche (pay minimums, extra to highest rate) or debt snowball (smallest balance first)
• Negotiate with creditors if you're struggling
• Avoid taking on new debt while paying off existing debt
• Consider debt consolidation if it lowers your rates
• Build a small emergency fund even while paying debt

The key is consistency and having a clear plan!"""
    
    # Market/economic responses
    elif any(word in user_input_lower for word in ['market', 'economy', 'recession', 'inflation']):
        return """Market and economic considerations:

• Markets are cyclical - ups and downs are normal
• Stay focused on long-term goals, not short-term volatility
• Inflation erodes purchasing power - consider assets that historically outpace inflation
• Don't try to time the market - consistent investing tends to win
• Keep some cash for opportunities during market downturns
• Review and rebalance your portfolio periodically

Remember: Market timing is extremely difficult even for professionals!"""
    
    # Default response
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

Could you be more specific about what aspect of personal finance you'd like help with?"""

def similarity_search_docs(query: str, top_k: int = 3):
    """Placeholder for document similarity search - returns empty for now"""
    # This would normally search through uploaded documents
    # For now, return empty list since we removed the vectorstore dependency
    return []