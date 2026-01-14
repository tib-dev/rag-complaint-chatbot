import faiss
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict


class Retriever:
    def __init__(self, index_path: Path, metadata_path: Path, k: int = 5):
        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)
        self.k = k

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Mismatch: Index has {self.index.ntotal} vectors, Metadata has {len(self.metadata)} rows.")

    def retrieve(self, query_embedding: np.ndarray) -> List[Dict]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')

        # Only normalize if your index is IndexFlatIP (Inner Product)
        # faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, self.k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            # BUG FIX: FAISS returns -1 if it can't find enough neighbors
            if idx < 0:
                continue

            try:
                row = self.metadata.iloc[int(idx)].to_dict()
                row["score"] = float(score)
                results.append(row)
            except IndexError:
                # Safety check for out-of-bounds
                continue

        return results
