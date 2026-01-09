"""
Tavily Search - Real-time web data augmentation
"""

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

class TavilySearch:
    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None

    def search(self, query: str, max_results: int = 5):
        if not self.client:
            return []
        try:
            return self.client.search(query=query, search_depth="advanced", max_results=max_results)
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []

    def get_context_for_query(self, query: str, max_results: int = 3):
        """Format search results as a prompt-ready context string"""
        results = self.search(query, max_results=max_results)
        if not results:
            return ""
        
        context_parts = []
        for res in results.get('results', []):
            context_parts.append(f"SOURCE: {res.get('url')}\nCONTENT: {res.get('content')}")
        
        return "\n\n---\n\n".join(context_parts)

def is_tavily_available():
    return os.getenv("TAVILY_API_KEY") is not None

_tavily_search = None

def get_tavily_search():
    global _tavily_search
    if _tavily_search is None:
        _tavily_search = TavilySearch()
    return _tavily_search
