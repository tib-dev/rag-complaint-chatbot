import re
import pandas as pd
from typing import Dict, Set

# ------------------------------------------------------------------------------
# Column normalization
# ------------------------------------------------------------------------------


def clean_and_select_columns(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
    required_columns: Set[str],
) -> pd.DataFrame:
    """
    Rename columns, validate schema, and select required fields.
    """
    try:
        df = df.copy()

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Validate required columns
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Preserve column order from mapping
        ordered_cols = list(column_mapping.values())
        df = df[ordered_cols]

        # Parse dates safely (only if present)
        if "date_received" in df.columns:
            df["date_received"] = pd.to_datetime(
                df["date_received"], errors="coerce"
            )

        return df

    except Exception as e:
        raise RuntimeError(f"Column cleaning failed: {e}") from e

# ------------------------------------------------------------------------------
# Narrative cleaning (minimal, embedding-safe)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------
# Safe boilerplate prefixes (exact, low-risk)
# ------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    r"\bi am filing a complaint\b",
    r"\bi am filing this complaint\b",
    r"\bi am writing to file a complaint\b",
    r"\bi am writing to formally express my concerns\b",
    r"\bi am writing to express my concerns\b",
    r"\bi am submitting this complaint\b",
    r"\bthis complaint is regarding\b",
    r"\bi would like to file a complaint\b",
    r"\bi would like to report\b",
]

# CFPB system / metadata artifacts
SYSTEM_MARKERS = [
    r"\[\[company public response\]\]",
    r"\[\[consumer consent provided\]\]",
    r"\[\[submitted via\]\]",
]


def clean_narrative_text(text: str) -> str:
    """
    RAG-safe narrative cleaning:
    - preserves meaning
    - normalizes masking
    - removes CFPB boilerplate + system artifacts
    """
    if not isinstance(text, str):
        return ""

    # Normalize case
    text = text.lower()

    # --------------------------------------------------------------
    # Normalize masked content (xxxx, xx/xx, xxxx xxxx, etc.)
    # --------------------------------------------------------------
    text = re.sub(
        r"\b[xX]{2,}([/\-\s][xX]{2,})*\b",
        "<masked>",
        text,
    )

    # --------------------------------------------------------------
    # Remove CFPB boilerplate phrases (once, safely)
    # --------------------------------------------------------------
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text, count=1)

    # Remove system artifacts
    for marker in SYSTEM_MARKERS:
        text = re.sub(marker, "", text)

    # --------------------------------------------------------------
    # Remove non-informative symbols (keep structure for embeddings)
    # --------------------------------------------------------------
    text = re.sub(r"[^a-z0-9\s\.\,\-<>]", " ", text)

    # --------------------------------------------------------------
    # Normalize whitespace
    # --------------------------------------------------------------
    text = re.sub(r"\s+", " ", text).strip()

    return text



def apply_text_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply narrative cleaning and add a clean_narrative column.
    """
    try:
        df = df.copy()

        if "consumer_complaint_narrative" not in df.columns:
            raise ValueError(
                "Required column 'consumer_complaint_narrative' not found"
            )

        df["clean_narrative"] = df[
            "consumer_complaint_narrative"
        ].apply(clean_narrative_text)

        return df

    except Exception as e:
        raise RuntimeError(f"Text cleaning failed: {e}") from e
