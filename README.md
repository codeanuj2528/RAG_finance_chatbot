# 🚀 RAG Finance Chatbot (Docker & Groq Optimized)

A premium Personal Finance Assistant powered by Retrieval-Augmented Generation (RAG), Groq LLMs, and real-time market data.

## ✨ ANUJ'S SPECIALIZED EDITIONS
- **📚 Document Analyst Mode**: Query your private financial PDFs indexed in the ChromaDB vault.
- **🌐 Market Intelligence Mode**: Real-time market news (Tavily), stock charts (YFinance), and "News-Based Guidance".
- **🤖 Groq-First AI**: Powered by `llama-3.3-70b-versatile` (Ultra-fast).
- **🔴 Power Crimson Theme**: High-contrast White & Red premium interface.
- **🐳 One-Command Docker**: Fully-persistent setup for your finance data.
- **🔄 Keep-Alive Engine**: Built-in background thread to prevent Render free-tier sleep.

## 🛠️ Environment Variables

Create a `.env` file in the root directory:

| Variable | Description | Source |
|----------|-------------|--------|
| `GROQ_API_KEY` | **Primary LLM** API Key | [Groq Cloud](https://console.groq.com/) |
| `OPENAI_API_KEY`| Fallback LLM & Embeddings | [OpenAI](https://platform.openai.com/) |
| `AV_API_KEY`   | Real-time Market Data | [Alpha Vantage](https://www.alphavantage.co/) |
| `TAVILY_API_KEY`| Modern Web Search | [Tavily AI](https://tavily.com/) |
| `NEWS_API_KEY`  | Finance News | [NewsAPI](https://newsapi.org/) |
| `RENDER_EXTERNAL_URL` | App URL for Keep-Alive | [Render Dashboard](https://dashboard.render.com/) |

## 🐳 Docker Setup (Recommended)

1. **Clone & Setup**:
   ```bash
   git clone https://github.com/codeanuj2528/RAG_finance_chatbot
   cd RAG_finance_chatbot
   cp .env.example .env  # Fill in your keys
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```
   *Access at: http://localhost:8501*

## 💻 Local Development (Manual)

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Streamlit**:
   ```bash
   streamlit run app.py
   ```

## 🚀 Deployment

This application is ready for deployment on **Render**, **Railway**, or any Docker-compatible cloud provider. 
- **ChromaDB Persistence**: Ensure you mount a volume to `/app/chroma_db` in your production environment to keep your indexed PDFs.

---
*Built with ❤️ for Personal Finance Excellence.*
