from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import FEATURES, LABEL_COL, ATTACK_COL, DATASETS, L7_PROTO_MAP, L7_PROTO_UNKNOWN

_UINT16_MAX = np.iinfo(np.uint16).max  # 65535
_FLOAT32_MAX = np.finfo(np.float32).max  # ~3.4e+38

# Float-only throughput columns that can contain astronomically large values
# in some datasets (e.g. NF-CSE-CIC-IDS2018-v2) and must be capped before
# writing train/test splits so that float32-based models do not receive
# values that overflow to infinity.
_THROUGHPUT_COLS = ("SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES")


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
    feature_cols = [c for c in FEATURES if c in df.columns]
    df[feature_cols] = (
        df[feature_cols]
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
    usecols = list(FEATURES) + [LABEL_COL, ATTACK_COL]
    df = pd.read_csv(path, usecols=usecols)
    # Enforce feature column order: features first, then labels
    df = df[usecols]

    if integer_only:
        # Map L7_PROTO float values to categorical uint16 IDs before saturation.
        # Unknown values (not in the map) are assigned L7_PROTO_UNKNOWN.
        df["L7_PROTO"] = (
            df["L7_PROTO"].map(L7_PROTO_MAP).fillna(L7_PROTO_UNKNOWN).astype(np.uint16)
        )
        df = to_uint16_saturated(df)

    return df


def split_datasets(
    repo_root: Path | str | None = None,
    train_size: float = 0.70,
    random_state: int = 42,
) -> None:
    """Split every dataset in config.DATASETS into train/test CSVs (70/30).

    For each dataset two variants are produced:
      - **float**   – all original feature dtypes preserved (``integer_only=False``)
      - **uint16**  – float features dropped, remaining columns cast to uint16
                      (``integer_only=True``)

    Output files are written to the ``dataset/`` directory alongside the
    source CSVs, using the naming convention::

        <stem>_float_train.csv  /  <stem>_float_test.csv
        <stem>_uint16_train.csv /  <stem>_uint16_test.csv

    The split is stratified on :data:`~config.LABEL_COL` so that the
    class-ratio is preserved in both partitions.

    Parameters
    ----------
    repo_root:
        Path to the repository root (the directory that contains the
        ``dataset/`` folder).  Defaults to the parent of this file's parent,
        i.e. the standard layout where ``src/dataset.py`` lives one level
        below the repo root.
    train_size:
        Fraction of rows to place in the training split.  Default is ``0.70``.
    random_state:
        Seed for the random splitter.  Default is ``42``.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent
    repo_root = Path(repo_root)

    for dataset_path in DATASETS:
        abs_path = repo_root / dataset_path
        stem = abs_path.stem  # e.g. "NF-BoT-IoT-v2"
        out_dir = abs_path.parent
        print(f"Splitting {stem} ...")

        for dtype_label, integer_only in (("float", False), ("uint16", True)):
            df = load_dataset(str(abs_path), integer_only=integer_only)

            # Cap throughput columns to the float32 range so that splits
            # never contain values that overflow to infinity when a model
            # casts the feature matrix to float32 internally.
            if not integer_only:
                for col in _THROUGHPUT_COLS:
                    if col in df.columns:
                        df[col] = df[col].clip(lower=0, upper=_FLOAT32_MAX)

            train_df, test_df = train_test_split(
                df,
                train_size=train_size,
                random_state=random_state,
                stratify=df[LABEL_COL],
            )

            train_path = out_dir / f"{stem}_{dtype_label}_train.csv"
            test_path = out_dir / f"{stem}_{dtype_label}_test.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            print(
                f"  [{dtype_label}] train={len(train_df):,}  test={len(test_df):,}"
                f"  -> {train_path.name} / {test_path.name}"
            ) 

            del df
            del train_df
            del test_df


if __name__ == "__main__":
    split_datasets()
