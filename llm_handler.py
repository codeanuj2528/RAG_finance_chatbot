"""
LLM Handler - Orchestrates between Groq and OpenAI
Features:
- Multi-provider support
- Failover mechanism
- Cost/Performance optimization
"""

import os
import openai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class LLMHandler:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.last_provider = None
        
        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.groq_client = Groq(api_key=self.groq_key) if self.groq_key else None

    def chat(self, messages, temperature=0.7, max_tokens=1024, model=None):
        """
        Enhanced chat with failover
        Standard order: Groq (Fast/Cheap) -> OpenAI (Robust fallback)
        """
        # 1. Try Groq first
        if self.groq_client:
            try:
                self.last_provider = "groq"
                response = self.groq_client.chat.completions.create(
                    model=model or "llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Groq error: {e}")
                
        # 2. Try OpenAI second
        if self.openai_client:
            try:
                self.last_provider = "openai"
                response = self.openai_client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error: {e}")
                
        return "Error: No LLM provider available or configured."

    def get_available_providers(self):
        providers = []
        if self.openai_key: providers.append("OpenAI")
        if self.groq_key: providers.append("Groq")
        return providers

_handler = None

def get_llm():
    global _handler
    if _handler is None:
        _handler = LLMHandler()
    return _handler
