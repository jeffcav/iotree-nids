import numpy as np
import pandas as pd

from config import FEATURES, FLOAT_FEATURES, LABEL_COL, ATTACK_COL, DATASETS

_UINT16_MAX = np.iinfo(np.uint16).max  # 65535


def to_uint16_saturated(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all feature columns of *df* to uint16 with saturation.

    Values below 0 are clamped to 0; values above 65535 are clamped to 65535.
    The label columns (LABEL_COL, ATTACK_COL) are left unchanged.

    Parameters
    ----------
    df:
        DataFrame as returned by :func:`load_dataset`.

    Returns
    -------
    pd.DataFrame
        A new DataFrame where every feature column has dtype uint16.
    """
    df = df.copy()
    df[FEATURES] = (
        df[FEATURES]
        .clip(lower=0, upper=_UINT16_MAX)
        .astype(np.uint16)
    )
    return df


def load_dataset(path: str, integer_only: bool = False) -> pd.DataFrame:
    """Load a NetFlow v2 dataset CSV into a DataFrame.

    Only the columns listed in config.FEATURES are loaded as input features.
    The label and attack-type columns are appended so callers can use them
    for supervised learning.

    Parameters
    ----------
    path:
        Absolute or relative path to the dataset CSV file.
    integer_only:
        If True, columns listed in config.FLOAT_FEATURES (e.g.
        SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES) are dropped, and
        the remaining feature columns are converted to uint16 with saturation
        (values clamped to [0, 65535] before casting).
        If False (default), all columns retain their original dtypes as parsed
        from the CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns = FEATURES + [LABEL_COL, ATTACK_COL].
    """
    usecols = FEATURES + [LABEL_COL, ATTACK_COL]
    df = pd.read_csv(path, usecols=usecols)
    # Enforce feature column order: features first, then labels
    df = df[usecols]

    if integer_only:
        df = df.drop(columns=FLOAT_FEATURES)
        df = to_uint16_saturated(df)

    return df


def print_l7_proto_values() -> None:
    """Print the unique L7_PROTO values across all datasets.

    Only L7_PROTO is read from each file for efficiency.
    """
    unique_vals = set()
    for path in DATASETS:
        df = pd.read_csv(path, usecols=["L7_PROTO"])
        unique_vals.update(df["L7_PROTO"].dropna().unique())

    print(f"Unique L7_PROTO values across all datasets ({len(unique_vals)}): {sorted(unique_vals)}")
