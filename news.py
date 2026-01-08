"""
news.py - Finance News Fetching with NewsAPI
"""

import streamlit as st
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def fetch_finance_news_newsapi(num_articles=5):
    """Fetch news using NewsAPI"""
    if not NEWS_API_KEY:
        return []
    
    try:
        today = datetime.today().strftime('%Y-%m-%d')
        last_week = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "finance OR economy OR stocks OR market",
            "from": last_week,
            "to": today,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": num_articles,
            "apiKey": NEWS_API_KEY
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") == "ok":
            articles = data.get('articles', [])
            return [{
                "title": article.get('title', 'No Title'),
                "url": article.get('url', '#'),
                "source": article.get('source', {}).get('name', 'Unknown'),
                "description": article.get('description', '')[:150] if article.get('description') else ''
            } for article in articles]
        else:
            return []
            
    except Exception as e:
        st.warning(f"NewsAPI error: {e}")
        return []


def fetch_finance_news_tavily(num_articles=5):
    """Fetch news using Tavily as fallback"""
    if not TAVILY_API_KEY:
        return []
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        
        response = client.search(
            query="latest finance market news today",
            search_depth="basic",
            max_results=num_articles
        )
        
        if response and 'results' in response:
            return [{
                "title": r.get('title', 'No Title'),
                "url": r.get('url', '#'),
                "source": "Web",
                "description": r.get('content', '')[:150] if r.get('content') else ''
            } for r in response['results']]
        return []
        
    except Exception as e:
        return []


def fetch_finance_news(num_articles=5):
    """Fetch news with fallback"""
    # Try NewsAPI first
    articles = fetch_finance_news_newsapi(num_articles)
    
    # Fallback to Tavily
    if not articles:
        articles = fetch_finance_news_tavily(num_articles)
    
    return articles


def display_finance_news():
    """Display finance news in Streamlit"""
    st.subheader("📰 Top Finance News")
    
    with st.spinner("Fetching latest news..."):
        articles = fetch_finance_news(num_articles=5)
    
    if articles:
        for i, article in enumerate(articles, 1):
            with st.container():
                st.markdown(f"**{i}. [{article['title']}]({article['url']})**")
                st.caption(f"Source: {article['source']}")
                if article.get('description'):
                    st.write(article['description'])
                st.markdown("---")
    else:
        st.info("📭 No news articles available. Check your API keys in Environment settings.")
        st.write("Required: `NEWS_API_KEY` or `TAVILY_API_KEY`")
