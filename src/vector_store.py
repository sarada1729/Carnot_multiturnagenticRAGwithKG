import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import DOCS_PATH, INDEX_PATH, EMBEDDING_MODEL_NAME


def load_documents():
    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    return documents


def build_vector_index():
    documents = load_documents()

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [doc["text"] for doc in documents]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Number of vectors: {index.ntotal}")
    print(f"Embedding dimension: {dim}")


def load_vector_index():
    documents = load_documents()
    index = faiss.read_index(str(INDEX_PATH))
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return documents, index, embedding_model


def vector_search(query, documents, index, embedding_model, k=3):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        doc = documents[idx].copy()
        doc["score"] = float(score)
        results.append(doc)

    return results


if __name__ == "__main__":
    build_vector_index()