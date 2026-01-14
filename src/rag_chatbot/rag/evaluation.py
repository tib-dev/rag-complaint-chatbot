import re
import pandas as pd
import numpy as np
import nltk
from typing import List, Dict
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

SET_STOPWORDS = set(stopwords.words('english'))


def get_clean_tokens(text: str) -> set:
    """Helper to tokenize and remove stopwords/punctuation."""
    if not text:
        return set()
    # Tokenize words and remove non-alphanumeric characters
    tokens = re.findall(r"\w+", text.lower())
    return {t for t in tokens if t not in SET_STOPWORDS and not t.isdigit()}


def precision_at_k_semantic(
    retrieved_chunks: List[Dict],
    threshold: float = 0.7,
    k: int = 5,
    is_distance: bool = False
) -> float:
    """
    Calculates precision based on retrieval scores.
    is_distance: Set True if using L2 (lower is better).
    """
    if not retrieved_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]
    actual_k = len(top_k)

    if is_distance:
        hits = sum(1 for c in top_k if c.get("score", 1.0) <= threshold)
    else:
        hits = sum(1 for c in top_k if c.get("score", 0.0) >= threshold)

    return round(hits / actual_k, 3)


def faithfulness_score(answer: str, context: str) -> float:
    """Measures how much of the answer is derived directly from the context."""
    answer_tokens = get_clean_tokens(answer)
    context_tokens = get_clean_tokens(context)

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 3)


def answer_relevancy_score(answer: str, query: str) -> float:
    """Simple check to see if the answer shares key terms with the query."""
    answer_tokens = get_clean_tokens(answer)
    query_tokens = get_clean_tokens(query)

    if not query_tokens:
        return 1.0

    overlap = answer_tokens & query_tokens
    return round(len(overlap) / len(query_tokens), 3)


def build_evaluation_table(
    results: List[Dict],
    similarity_threshold: float = 0.7,
    is_distance: bool = False
) -> pd.DataFrame:
    """Builds a comprehensive evaluation dataframe."""
    rows = []

    for r in results:
        sources = r.get("sources", [])
        context = "\n".join(c.get("document", "") for c in sources)

        # Calculate scores
        prec = precision_at_k_semantic(
            sources, threshold=similarity_threshold, is_distance=is_distance)
        faith = faithfulness_score(r["answer"], context)
        relevancy = answer_relevancy_score(r["answer"], r["query"])

        # Determine Max Similarity (handle Distance vs Similarity)
        scores = [c.get("score", 0.0) for c in sources]
        max_sim = min(scores) if is_distance else max(scores, default=0.0)

        rows.append({
            "query": r["query"],
            "answer": r["answer"][:100] + "...",  # Preview for table
            "precision@k": prec,
            "faithfulness": faith,
            "relevancy": relevancy,
            "best_score": round(max_sim, 4),
            "num_chunks": len(sources),
        })

    return pd.DataFrame(rows)
