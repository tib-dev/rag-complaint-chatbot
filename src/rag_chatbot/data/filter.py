from typing import Dict, List
import pandas as pd
from typing import List


def normalize_and_filter_products(
    df: pd.DataFrame,
    *,
    product_column: str,
    category_mapping,
    allowed_products: List[str],
) -> pd.DataFrame:
    """
    Normalize raw CFPB product categories into canonical categories
    and filter to allowed products only.

    Supports:
    - dict[str, str]
    - list[dict[str, str]] (YAML-friendly)
    """

    if product_column not in df.columns:
        raise ValueError(f"Column '{product_column}' not found in DataFrame")

    # --- Normalize category_mapping ---
    if isinstance(category_mapping, list):
        flat_mapping: Dict[str, str] = {}
        for item in category_mapping:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(
                    "Each category mapping must be a single-key dictionary"
                )
            flat_mapping.update(item)
        category_mapping = flat_mapping

    if not isinstance(category_mapping, dict):
        raise TypeError(
            "category_mapping must be a dict or list of single-key dicts"
        )

    df = df.copy()

    # --- Normalize product labels before mapping ---
    raw_products = (
        df[product_column]
        .astype(str)
        .str.strip()
        .str.replace('"', '', regex=False)
    )

    df["product_category"] = raw_products.map(category_mapping)

    # --- Filter to allowed canonical products ---
    df = df[df["product_category"].isin(allowed_products)]

    return df





def filter_non_empty_narratives(
    df: pd.DataFrame,
    narrative_column: str = "consumer_complaint_narrative",
) -> pd.DataFrame:
    """
    Remove records with missing or empty complaint narratives.
    """
    if narrative_column not in df.columns:
        raise ValueError(f"Column '{narrative_column}' not found")

    mask = (
        df[narrative_column].notna()
        & df[narrative_column].str.strip().ne("")
    )

    return df[mask].copy()
