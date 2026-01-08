# RAG Finance Chatbot

Personal Finance Assistant with RAG (Retrieval-Augmented Generation) capabilities.

## Features
- 💬 AI Chat with OpenAI + Groq fallback
- 📄 PDF upload and RAG-based Q&A
- 📰 Finance news with NewsAPI + Tavily fallback
- 📈 Asset tracking (stocks, forex, crypto)
- 🛠️ Budgeting tools with 50/30/20 recommendations

## Environment Variables

Set these in Render Dashboard → Environment:

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for chat & embeddings | Yes |
| `GROQ_API_KEY` | Groq API key (fallback LLM) | Optional |
| `TAVILY_API_KEY` | Tavily API for web search | Optional |
| `NEWS_API_KEY` | NewsAPI for finance news | Optional |
| `AV_API_KEY` | Alpha Vantage for stock data | Optional |

## Deploy to Render

1. Push this folder to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Add environment variables in the Environment tab
6. Deploy!

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## API Keys Setup

Your `.env` file should look like:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
NEWS_API_KEY=...
AV_API_KEY=...
```

## Troubleshooting

**Blank page on Render?**
- Make sure `Procfile` exists with correct start command
- Check that environment variables are set
- View logs in Render dashboard

**Chat not working?**
- Verify `OPENAI_API_KEY` is set correctly
- Groq will be used as fallback if OpenAI fails
