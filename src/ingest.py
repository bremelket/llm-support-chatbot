import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


def load_documents(data_dir: str) -> list[dict]:
    docs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r") as f:
                text = f.read()
            docs.append({"filename": filename, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_vectorstore(docs: list[dict], model: SentenceTransformer) -> None:
    all_chunks = []
    all_metadata = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"filename": doc["filename"], "text": chunk})

    print(f"Total chunks: {len(all_chunks)}")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTORSTORE_DIR, "index.faiss"))
    with open(os.path.join(VECTORSTORE_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"Vectorstore saved to {VECTORSTORE_DIR}/")


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Building vectorstore...")
    build_vectorstore(docs, model)
    print("Done!")