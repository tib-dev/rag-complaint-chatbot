import pandas as pd
import pytest
from typing import List, Dict, Any

from rag_chatbot.chunking.text_splitter import chunk_documents  
from rag_chatbot.chunking.text_splitter import text_splitter

def test_chunk_documents_basic():
    df = pd.DataFrame({
        "complaint_id": [1],
        "product_category": ["Credit card"],
        "consumer_complaint_narrative": [
            "This is a test complaint. It has multiple sentences. "
            "We expect it to be chunked properly."
        ],
    })

    docs = chunk_documents(df)

    assert isinstance(docs, list)
    assert len(docs) > 0

    for doc in docs:
        assert "text" in doc
        assert "metadata" in doc
        assert doc["metadata"]["complaint_id"] == 1
        assert doc["metadata"]["product_category"] == "Credit card"
        assert isinstance(doc["metadata"]["chunk_id"], int)
        assert doc["text"].strip() != ""


def test_chunk_documents_skips_empty_narratives():
    df = pd.DataFrame({
        "complaint_id": [1, 2, 3],
        "product_category": ["Credit card"] * 3,
        "consumer_complaint_narrative": [
            "",
            None,
            "Valid narrative text here."
        ],
    })

    docs = chunk_documents(df)

    assert len(docs) > 0
    assert all(
        doc["metadata"]["complaint_id"] == 3 for doc in docs
    )


@pytest.mark.parametrize(
    "missing_col",
    [
        "consumer_complaint_narrative",
        "complaint_id",
        "product_category",
    ],
)
def test_chunk_documents_missing_required_column(missing_col):
    data = {
        "consumer_complaint_narrative": ["Some text"],
        "complaint_id": [1],
        "product_category": ["Mortgage"],
    }
    data.pop(missing_col)

    df = pd.DataFrame(data)

    with pytest.raises(ValueError) as exc:
        chunk_documents(df)

    assert "Missing required columns" in str(exc.value)


def test_chunk_id_resets_per_complaint():
    df = pd.DataFrame({
        "complaint_id": [1, 2],
        "product_category": ["Mortgage", "Mortgage"],
        "consumer_complaint_narrative": [
            "Sentence one. Sentence two. Sentence three.",
            "Another complaint with enough text to split.",
        ],
    })

    docs = chunk_documents(df)

    chunks_by_id = {}
    for doc in docs:
        cid = doc["metadata"]["complaint_id"]
        chunks_by_id.setdefault(cid, []).append(
            doc["metadata"]["chunk_id"]
        )

    for chunk_ids in chunks_by_id.values():
        assert chunk_ids == list(range(len(chunk_ids)))


def test_chunk_documents_large_input():
    long_text = "This is a sentence. " * 2000

    df = pd.DataFrame({
        "complaint_id": [99],
        "product_category": ["Student loan"],
        "consumer_complaint_narrative": [long_text],
    })

    docs = chunk_documents(df)

    assert len(docs) > 1
    assert all(len(doc["text"]) <= 500 for doc in docs)


def test_chunk_documents_runtime_error(monkeypatch):
    def broken_split(*args, **kwargs):
        raise Exception("Splitter failed")

    monkeypatch.setattr(text_splitter, "split_text", broken_split)

    df = pd.DataFrame({
        "complaint_id": [1],
        "product_category": ["Credit card"],
        "consumer_complaint_narrative": ["Some text"],
    })

    with pytest.raises(RuntimeError) as exc:
        chunk_documents(df)

    assert "Failed during document chunking" in str(exc.value)
