"""Train random forest and LightGBM models for every combination of:
  - dataset   (from config.DATASETS)
  - feature mode  (float  /  integer-only)
  - max_leaf_nodes / num_leaves  (from config.MAX_LEAVES)

Saved layout
------------
models/
  <dataset_stem>/
    float/
      <max_leaves>/
        rf.joblib
        lgbm.joblib
    integer/
      <max_leaves>/
        rf.joblib
        lgbm.joblib
"""

import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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

                # ── Random Forest ──────────────────────────────────────────
                rf_path = out_dir / "rf.joblib"
                print(
                    f"  [mode={mode}] Training RF   max_leaf_nodes={max_leaves:<5} … ",
                    end="",
                    flush=True,
                )
                clf_rf = RandomForestClassifier(
                    max_leaf_nodes=max_leaves,
                    n_jobs=-1,
                    random_state=42,
                )
                clf_rf.fit(X_train, y_train)
                f1_rf = f1_score(y_test, clf_rf.predict(X_test), average="weighted")
                joblib.dump(clf_rf, rf_path)
                print(f"f1={f1_rf:.4f}  →  {rf_path.relative_to(REPO_ROOT)}")

                # ── LightGBM ───────────────────────────────────────────────
                lgbm_path = out_dir / "lgbm.joblib"
                print(
                    f"  [mode={mode}] Training LGBM num_leaves={max_leaves:<5}      … ",
                    end="",
                    flush=True,
                )
                clf_lgbm = lgb.LGBMClassifier(
                    num_leaves=max_leaves,
                    n_jobs=-1,
                    random_state=42,
                )
                clf_lgbm.fit(X_train, y_train)
                f1_lgbm = f1_score(y_test, clf_lgbm.predict(X_test), average="weighted")
                joblib.dump(clf_lgbm, lgbm_path)
                print(f"f1={f1_lgbm:.4f}  →  {lgbm_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    train()
