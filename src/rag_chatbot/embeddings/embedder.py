import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_chatbot.core.project_root import get_project_root


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = get_project_root()
VECTOR_STORE_PATH = ROOT / "vector_store"


# -------------------------------------------------------------------
# Embeddings
# -------------------------------------------------------------------
def build_embeddings(
    docs: List[Dict[str, Any]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Generate normalized sentence embeddings for document chunks.

    Args:
        docs: List of documents with a `text` field.
        model_name: SentenceTransformer model name.

    Returns:
        A NumPy array of shape (n_docs, embedding_dim).

    Raises:
        ValueError: If docs are empty or missing text.
        RuntimeError: If embedding generation fails.
    """
    if not docs:
        raise ValueError("No documents provided for embedding.")

    texts = [d.get("text", "") for d in docs if isinstance(d.get("text"), str)]
    if not texts:
        raise ValueError("Documents contain no valid text fields.")

    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,  # required for cosine similarity
        )
        return np.asarray(embeddings)
    except Exception as exc:
        raise RuntimeError("Failed to generate embeddings.") from exc


# -------------------------------------------------------------------
# FAISS Index
# -------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using inner product (cosine similarity).

    Args:
        embeddings: Normalized embedding matrix.

    Returns:
        FAISS index with all embeddings added.

    Raises:
        ValueError: If embeddings are invalid.
    """
    if embeddings is None or embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy array.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index


# -------------------------------------------------------------------
# Persistence
# -------------------------------------------------------------------
def save_vector_store(
    index: faiss.Index,
    docs: List[Dict[str, Any]],
    path: Path = VECTOR_STORE_PATH,
) -> None:
    """
    Persist FAISS index and document metadata to disk.

    Args:
        index: FAISS index instance.
        docs: Original document chunks with metadata.
        path: Directory where vector store will be saved.

    Raises:
        RuntimeError: If saving fails.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(path / "index.faiss"))

        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(docs, f)

    except Exception as exc:
        raise RuntimeError("Failed to save vector store.") from exc
