import pandas as pd
from typing import Hashable


def stratified_sample(
    df: pd.DataFrame,
    group_col: Hashable,
    total_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Perform stratified sampling while approximately preserving
    category proportions.

    Args:
        df: Input DataFrame.
        group_col: Column name used for stratification.
        total_samples: Total number of rows to sample.
        random_state: Random seed for reproducibility.

    Returns:
        A shuffled DataFrame containing the stratified sample.

    Raises:
        ValueError: If inputs are invalid.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    if total_samples <= 0:
        raise ValueError("total_samples must be a positive integer.")

    total_samples = min(total_samples, len(df))

    # Proportions per group
    proportions = df[group_col].value_counts(normalize=True)

    samples = []
    allocated = 0

    for category, ratio in proportions.items():
        n = max(1, int(round(total_samples * ratio)))
        subset = df[df[group_col] == category]

        n = min(n, len(subset))
        allocated += n

        samples.append(
            subset.sample(n=n, random_state=random_state)
        )

    result = pd.concat(samples, ignore_index=True)

    # Adjust size if rounding caused over/under sampling
    if len(result) > total_samples:
        result = result.sample(
            n=total_samples, random_state=random_state
        )
    elif len(result) < total_samples:
        remaining = df.drop(result.index)
        extra = remaining.sample(
            n=total_samples - len(result),
            random_state=random_state,
        )
        result = pd.concat([result, extra], ignore_index=True)

    return result.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
