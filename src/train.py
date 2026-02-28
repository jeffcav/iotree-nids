"""Train LightGBM models for every combination of:
  - dataset   (from config.DATASETS)
  - feature mode  (float  /  integer-only)
  - num_leaves     (from config.MAX_LEAVES)
  - n_estimators   (from config.N_ESTIMATORS)

Saved layout
------------
models/
  <dataset_stem>/
    float/
      <max_leaves>/
        <n_estimators>/
          lgbm.joblib
    integer/
      <max_leaves>/
        <n_estimators>/
          lgbm.joblib
"""

import os
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
from joblib import Parallel, delayed

# Allow running directly from the src/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ATTACK_COL, DATASETS, FEATURES, LABEL_COL, MAX_LEAVES, N_ESTIMATORS
from dataset import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _train_one(
    X_train,
    y_train,
    model_type: str,
    max_leaves: int,
    n_estimators: int,
    out_dir: Path,
    mode: str,
) -> None:
    """Train a single (model_type, max_leaves, n_estimators) combination and save it.

    ``n_jobs=1`` is used for each model because parallelism is already
    provided by the outer :func:`joblib.Parallel` call.
    """
    if model_type == "lgbm":
        path = out_dir / "lgbm.joblib"
        print(
            f"  [mode={mode}] Training LGBM num_leaves={max_leaves:<5} n_estimators={n_estimators:<4} … ",
            flush=True,
        )
        clf = lgb.LGBMClassifier(
            num_leaves=max_leaves,
            n_estimators=n_estimators,
            n_jobs=1,
            verbosity=-1,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        joblib.dump(clf, path)
        print(f"  [mode={mode}] LGBM num_leaves={max_leaves:<5} n_estimators={n_estimators:<4} →  {path.relative_to(REPO_ROOT)}")


def train() -> None:
    """Train and persist all models defined by the config.

    For each (dataset, mode) pair the full grid of max_leaves values is
    trained in parallel – one job per (model_type × max_leaves) combination –
    using :func:`joblib.Parallel` with a thread-based backend so that the
    already-loaded training data is shared without copying.
    """

    for dataset_idx,dataset_path in enumerate(DATASETS):
        dataset_name = Path(dataset_path).stem  # e.g. "NF-BoT-IoT-v2"
        print(f"\n{'=' * 60}")
        print(f"Dataset : {dataset_name}")
        print(f"{'=' * 60}")

        for integer_only in (False, True):
            mode = "integer" if integer_only else "float"
            # Load only the pre-split train partition (float variant is always
            # used as the source so that integer_only processing is applied
            # consistently by load_dataset).
            dtype_label = "uint16" if integer_only else "float"
            train_path = str(
                REPO_ROOT
                / Path(dataset_path).parent
                / f"{dataset_name}_{dtype_label}_train.csv"
            )
            print(f"\n  [mode={mode}] Loading train split …")

            df = load_dataset(train_path, integer_only=integer_only)

            feature_cols = [c for c in FEATURES if c in df.columns]
            X_train = df[feature_cols].values
            y_train = df[LABEL_COL].values

            print(f"  [mode={mode}] Train samples : {len(X_train):>10,}")

            # Pre-create output directories (not thread-safe if done inside
            # _train_one, since multiple jobs share the same directory path).
            for max_leaves in MAX_LEAVES:
                for n_estimators in N_ESTIMATORS:
                    (MODELS_DIR / dataset_name / mode / str(max_leaves) / str(n_estimators)).mkdir(
                        parents=True, exist_ok=True
                    )

            # Build the full task list: every (model_type × max_leaves × n_estimators) combo.
            tasks = [
                (model_type, max_leaves, n_estimators)
                for max_leaves in MAX_LEAVES
                for n_estimators in N_ESTIMATORS
                for model_type in ("lgbm",)
            ]

            # prefer="threads" keeps X_train / y_train in shared memory and
            # avoids the serialisation cost of a process-based backend.
            # Both scikit-learn and LightGBM release the GIL during fitting,
            # so true parallel speedup is achieved.
            Parallel(n_jobs=8, prefer="threads")(
                delayed(_train_one)(
                    X_train,
                    y_train,
                    model_type,
                    max_leaves,
                    n_estimators,
                    MODELS_DIR / dataset_name / mode / str(max_leaves) / str(n_estimators),
                    mode,
                )
                for model_type, max_leaves, n_estimators in tasks
            )


if __name__ == "__main__":
    # suppress lightgbm warnings
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    train()
