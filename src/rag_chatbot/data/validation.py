
import pandas as pd


def validate_rag_ready(df: pd.DataFrame, min_words: int = 20):
    """
    Ensure cleaned narratives are usable for RAG.
    """
    if "clean_narrative" not in df.columns:
        raise ValueError("Missing 'clean_narrative' column")

    word_counts = df["clean_narrative"].str.split().apply(len)

    if word_counts.lt(min_words).all():
        raise ValueError(
            "All narratives are too short for meaningful retrieval"
        )

    return True
