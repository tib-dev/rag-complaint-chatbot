from sentence_transformers import SentenceTransformer
import numpy as np


class QueryEmbedder:
    """
    Generates normalized embeddings for queries.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, query: str) -> np.ndarray:
        return self.model.encode(
            query,
            normalize_embeddings=True
        ).astype("float32")
