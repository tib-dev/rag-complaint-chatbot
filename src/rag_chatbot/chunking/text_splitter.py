from typing import List, Dict, Any
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------------------------------
# Text splitter configuration
# ------------------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""],
)


# ------------------------------------------------------------------
# Chunking logic
# ------------------------------------------------------------------

def chunk_documents(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Split complaint narratives into overlapping text chunks suitable
    for embedding and vector indexing.

    Each chunk is stored with metadata to allow traceability back to
    the original complaint.

    Args:
        df: Cleaned DataFrame containing at least:
            - consumer_complaint_narrative
            - complaint_id
            - product_category

    Returns:
        A list of dictionaries with keys:
            - text: chunked text
            - metadata: complaint_id, product_category, chunk_id

    Raises:
        ValueError: If required columns are missing
        RuntimeError: If chunking fails unexpectedly
    """
    required_columns = {
        "consumer_complaint_narrative",
        "complaint_id",
        "product_category",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for chunking: {missing}")

    documents: List[Dict[str, Any]] = []

    try:
        for _, row in df.iterrows():
            narrative = row["consumer_complaint_narrative"]

            # Skip empty narratives defensively
            if not isinstance(narrative, str) or not narrative.strip():
                continue

            chunks = text_splitter.split_text(narrative)

            for i, chunk in enumerate(chunks):
                documents.append(
                    {
                        "text": chunk,
                        "metadata": {
                            "complaint_id": row["complaint_id"],
                            "product_category": row["product_category"],
                            "chunk_id": i,
                        },
                    }
                )

        return documents

    except Exception as e:
        raise RuntimeError(f"Failed during document chunking: {e}") from e
