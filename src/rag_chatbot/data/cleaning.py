from typing import Dict
import pandas as pd

import pandas as pd
from typing import Dict, Set

COLUMN_MAP: Dict[str, str] = {
    "Complaint ID": "complaint_id",
    "Sub-product": "product_category",
    "Product": "product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "State": "state",
    "Date received": "date_received",
}

REQUIRED_COLUMNS: Set[str] = set(COLUMN_MAP.values())


def normalize_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CFPB complaint dataset column names and validate schema.
    """
    try:
        # ---------- Validate input ----------
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        missing_raw = set(COLUMN_MAP.keys()) - set(df.columns)
        if missing_raw:
            raise ValueError(f"Missing expected raw columns: {missing_raw}")

        # ---------- Rename ----------
        df = df.rename(columns=COLUMN_MAP)

        # ---------- Validate output ----------
        missing_normalized = REQUIRED_COLUMNS - set(df.columns)
        if missing_normalized:
            raise ValueError(
                f"Schema validation failed. Missing columns: {missing_normalized}"
            )

        # ---------- Type enforcement ----------
        df["date_received"] = pd.to_datetime(
            df["date_received"], errors="coerce")
        df["complaint_id"] = df["complaint_id"].astype(str)

        # ---------- Content sanity checks ----------
        if df["consumer_complaint_narrative"].isna().all():
            raise ValueError("All complaint narratives are empty")

        return df

    except Exception as e:
        raise RuntimeError(
            f"Failed during column normalization & validation: {e}"
        ) from e
