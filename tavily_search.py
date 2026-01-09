"""
Web Search - Real-time data augmentation with Tavily + News API fallback
Best of both: Tavily for deep web search, News API for finance news fallback
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# API Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


class WebSearch:
    """Unified web search with Tavily primary + News API fallback"""
    
    def __init__(self):
        self.tavily_key = TAVILY_API_KEY
        self.news_api_key = NEWS_API_KEY
        self.tavily_client = None
        
        # Initialize Tavily if available
        if self.tavily_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
            except Exception as e:
                print(f"Tavily init error: {e}")

    def search_tavily(self, query: str, max_results: int = 5):
        """Primary search via Tavily"""
        if not self.tavily_client:
            return None
        try:
            return self.tavily_client.search(
                query=query, 
                search_depth="advanced", 
                max_results=max_results
            )
        except Exception as e:
            print(f"Tavily search error: {e}")
            return None

    def search_news_api(self, query: str, max_results: int = 5):
        """Fallback search via News API - great for finance news"""
        if not self.news_api_key:
            return None
        try:
            # Add finance context to query
            finance_query = f"{query} finance OR stock OR market OR investment"
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": finance_query,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": max_results
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"News API error: {e}")
            return None

    def search(self, query: str, max_results: int = 5):
        """Unified search: Tavily first, News API fallback"""
        # 1. Try Tavily first
        tavily_results = self.search_tavily(query, max_results)
        if tavily_results and tavily_results.get('results'):
            return {
                'source': 'tavily',
                'results': tavily_results.get('results', [])
            }
        
        # 2. Fallback to News API
        news_results = self.search_news_api(query, max_results)
        if news_results and news_results.get('articles'):
            formatted = []
            for article in news_results['articles']:
                formatted.append({
                    'url': article.get('url', ''),
                    'title': article.get('title', ''),
                    'content': article.get('description', '') or article.get('content', '')
                })
            return {
                'source': 'newsapi',
                'results': formatted
            }
        
        return {'source': 'none', 'results': []}

    def get_context_for_query(self, query: str, max_results: int = 3):
        """Format search results as LLM-ready context string"""
        search_data = self.search(query, max_results=max_results)
        
        if not search_data['results']:
            return ""
        
        context_parts = []
        source = search_data['source'].upper()
        
        for res in search_data['results']:
            url = res.get('url', 'Unknown')
            content = res.get('content', res.get('title', ''))
            context_parts.append(f"[{source}] SOURCE: {url}\nCONTENT: {content}")
        
        return "\n\n---\n\n".join(context_parts)


# Compatibility functions for existing code
def is_tavily_available():
    """Check if any web search is available"""
    return TAVILY_API_KEY is not None or NEWS_API_KEY is not None


_web_search = None


def get_tavily_search():
    """Returns WebSearch instance (backward compatible name)"""
    global _web_search
    if _web_search is None:
        _web_search = WebSearch()
    return _web_search


# Alias for clarity
def get_web_search():
    return get_tavily_search()
