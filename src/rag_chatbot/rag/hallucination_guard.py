from typing import List, Dict


def should_answer(
    retrieved_chunks: List[Dict],
    min_similarity: float = 0.35,
    min_context_chars: int = 200,
) -> bool:
    if not retrieved_chunks:
        return False

    scores = [c.get("score", 0.0) for c in retrieved_chunks]
    if max(scores) < min_similarity:
        return False

    total_chars = sum(
        len(c.get("document", "") or "")
        for c in retrieved_chunks
    )

    return total_chars >= min_context_chars
