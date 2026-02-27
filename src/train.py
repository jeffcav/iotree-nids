"""Train random forest models for every combination of:
  - dataset   (from config.DATASETS)
  - feature mode  (float  /  integer-only)
  - max_leaf_nodes  (from config.MAX_LEAVES)

Saved layout
------------
models/
  <dataset_stem>/
    float/
      <max_leaves>/
        model.joblib
    integer/
      <max_leaves>/
        model.joblib
"""

import os
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Allow running directly from the src/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ATTACK_COL, DATASETS, LABEL_COL, MAX_LEAVES
from dataset import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# Train/test split ratio (training fraction).
TRAIN_SIZE = 0.70


def train() -> None:
    """Train and persist all random forest models defined by the config."""

    for dataset_path in DATASETS:
        dataset_name = Path(dataset_path).stem  # e.g. "NF-BoT-IoT-v2"
        abs_path = str(REPO_ROOT / dataset_path)
        print(f"\n{'=' * 60}")
        print(f"Dataset : {dataset_name}")
        print(f"{'=' * 60}")

        for integer_only in (False, True):
            mode = "integer" if integer_only else "float"
            print(f"\n  [mode={mode}] Loading data …")

            df = load_dataset(abs_path, integer_only=integer_only)

            feature_cols = [c for c in df.columns if c not in (LABEL_COL, ATTACK_COL)]
            X = df[feature_cols].values
            y = df[LABEL_COL].values
            strat = df[ATTACK_COL].values

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=1.0 - TRAIN_SIZE,
                random_state=42,
                stratify=strat,
            )
            print(
                f"  [mode={mode}] Train samples : {len(X_train):>10,}"
                f"  |  Test samples : {len(X_test):>10,}"
            )

            for max_leaves in MAX_LEAVES:
                out_dir = MODELS_DIR / dataset_name / mode / str(max_leaves)
                out_dir.mkdir(parents=True, exist_ok=True)
                model_path = out_dir / "model.joblib"

                print(
                    f"  [mode={mode}] Training RF  max_leaf_nodes={max_leaves:<5} … ",
                    end="",
                    flush=True,
                )

                clf = RandomForestClassifier(
                    max_leaf_nodes=max_leaves,
                    n_jobs=-1,
                    random_state=42,
                )
                clf.fit(X_train, y_train)

                acc = accuracy_score(y_test, clf.predict(X_test))
                joblib.dump(clf, model_path)

                rel = model_path.relative_to(REPO_ROOT)
                print(f"acc={acc:.4f}  →  {rel}")


if __name__ == "__main__":
    train()
