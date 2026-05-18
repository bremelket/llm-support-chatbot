# LLM Support Chatbot

A RAG-based AI support chatbot built with LangGraph, FAISS, and Claude API.

## What it does
Answers customer support questions using a knowledge base of product documentation. Built as a portfolio project mirroring real-world AI engineer work.

## Tech stack
- **LangGraph** — agent orchestration
- **FAISS** — vector database for semantic search
- **sentence-transformers** — text embeddings
- **Claude API (Anthropic)** — LLM for answer generation
- **Streamlit** — chat UI

## Architecture
1. `ingest.py` — loads docs, chunks, embeds, saves to FAISS
2. `retriever.py` — semantic search over FAISS index
3. `agent.py` — LangGraph agent: retrieve → answer
4. `tools.py` — mock tools (order status, pricing, escalation)
5. `app.py` — Streamlit chat interface

## How to run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
streamlit run src/app.py
```

## Skills demonstrated
RAG pipelines, LLM agent orchestration, vector databases, prompt engineering, evaluation frameworks