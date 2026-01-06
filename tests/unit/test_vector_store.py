import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import faiss

from rag_chatbot.embeddings.embedder import (
    build_embeddings,
    build_faiss_index,
    save_vector_store,
)


@pytest.fixture
def sample_docs():
    return [
        {
            "text": "This is a test complaint about a credit card issue.",
            "metadata": {"complaint_id": 1, "product_category": "Credit card"},
        },
        {
            "text": "The bank failed to process my money transfer correctly.",
            "metadata": {"complaint_id": 2, "product_category": "Money transfers"},
        },
    ]


@pytest.fixture
def temp_vector_store(tmp_path):
    return tmp_path / "vector_store"


def test_build_embeddings_success(sample_docs):
    embeddings = build_embeddings(sample_docs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == len(sample_docs)

    # embeddings are normalized for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_embeddings_empty_docs():
    with pytest.raises(ValueError, match="No documents provided"):
        build_embeddings([])


def test_build_embeddings_no_valid_text():
    docs = [
        {"metadata": {"complaint_id": 1}},
        {"text": None},
    ]

    with pytest.raises(ValueError, match="no valid text"):
        build_embeddings(docs)


def test_build_embeddings_runtime_error(monkeypatch, sample_docs):
    def broken_encode(*args, **kwargs):
        raise RuntimeError("Model failure")

    from sentence_transformers import SentenceTransformer
    monkeypatch.setattr(SentenceTransformer, "encode", broken_encode)

    with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
        build_embeddings(sample_docs)


def test_build_faiss_index_success(sample_docs):
    embeddings = build_embeddings(sample_docs)
    index = build_faiss_index(embeddings)

    assert isinstance(index, faiss.Index)
    assert index.ntotal == embeddings.shape[0]


@pytest.mark.parametrize(
    "invalid_embeddings",
    [
        None,
        np.array([1, 2, 3]),
        np.array([]),
    ],
)
def test_build_faiss_index_invalid_embeddings(invalid_embeddings):
    with pytest.raises(ValueError):
        build_faiss_index(invalid_embeddings)


def test_save_vector_store_success(sample_docs, temp_vector_store):
    embeddings = build_embeddings(sample_docs)
    index = build_faiss_index(embeddings)

    save_vector_store(index, sample_docs, path=temp_vector_store)

    index_path = temp_vector_store / "index.faiss"
    metadata_path = temp_vector_store / "metadata.pkl"

    assert index_path.exists()
    assert metadata_path.exists()

    # Verify FAISS index loads
    loaded_index = faiss.read_index(str(index_path))
    assert loaded_index.ntotal == len(sample_docs)

    # Verify metadata integrity
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    assert metadata == sample_docs


def test_save_vector_store_failure(monkeypatch, sample_docs, temp_vector_store):
    embeddings = build_embeddings(sample_docs)
    index = build_faiss_index(embeddings)

    def broken_write(*args, **kwargs):
        raise IOError("Disk error")

    monkeypatch.setattr(faiss, "write_index", broken_write)

    with pytest.raises(RuntimeError, match="Failed to save vector store"):
        save_vector_store(index, sample_docs, path=temp_vector_store)
