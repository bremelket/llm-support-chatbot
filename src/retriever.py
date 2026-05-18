import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

VECTORSTORE_DIR = "vectorstore"

def load_vectorstore():
    index = faiss.read_index(os.path.join(VECTORSTORE_DIR, "index.faiss"))
    with open(os.path.join(VECTORSTORE_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def retrieve(query: str, model: SentenceTransformer, index, metadata, top_k: int = 3) -> list[dict]:
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": metadata[idx]["text"],
            "filename": metadata[idx]["filename"],
            "distance": distances[0][i]
        })
    return results


if __name__ == "__main__":
    print("Loading vectorstore...")
    index, metadata = load_vectorstore()

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "How do I reset my password?"
    print(f"\nQuery: {query}")
    print("\nTop results:")

    results = retrieve(query, model, index, metadata)
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (distance: {result['distance']:.4f}) ---")
        print(f"Source: {result['filename']}")
        print(f"Text: {result['text'][:200]}...")