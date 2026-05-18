FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    langchain \
    langgraph \
    langchain-anthropic \
    sentence-transformers \
    faiss-cpu \
    streamlit \
    python-dotenv

RUN python src/ingest.py

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]