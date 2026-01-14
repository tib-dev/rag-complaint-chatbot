import numpy as np
from typing import List, Dict


def compute_confidence(retrieved_chunks: List[Dict]) -> float:
    if not retrieved_chunks:
        return 0.0

    scores = np.array(
        [float(c["score"]) for c in retrieved_chunks if "score" in c]
    )

    if len(scores) == 0:
        return 0.0

    confidence = 0.6 * scores.max() + 0.4 * scores.mean()
    return round(min(confidence, 1.0), 3)
