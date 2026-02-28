"""Evaluate all trained LightGBM models on the test split of each dataset.

For every combination of:
  - dataset        (from config.DATASETS)
  - feature mode   (float / integer)
  - num_leaves     (from config.MAX_LEAVES)
  - n_estimators   (from config.N_ESTIMATORS)

the script loads the corresponding model, runs inference on the held-out test
partition and records weighted precision, recall and F1-score.

Output files
------------
reports/metrics.csv   – one row per (dataset, mode, num_leaves, n_estimators)
reports/metrics.md    – Markdown tables comparing all models per dataset
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

# Allow running directly from src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ATTACK_COL, DATASETS, FEATURES, LABEL_COL, MAX_LEAVES, N_ESTIMATORS
from dataset import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "reports"

# Average strategy for sklearn metrics.  "binary" is used per round since
# the balanced draws always contain exactly 50 % of each class.
_AVG = "binary"

# Balanced-sampling evaluation parameters.
# Each model is evaluated over N_ROUNDS independent draws.  Each draw takes
# N_PER_CLASS samples from class-0 and the same number from class-1, giving
# a perfectly balanced 50/50 evaluation set.  N_PER_CLASS is capped at
# N_PER_CLASS_MAX so evaluation stays fast even for very large test splits;
# the final cap applied at runtime is min(|class0|, |class1|, N_PER_CLASS_MAX).
N_ROUNDS        = 100
N_PER_CLASS_MAX = 10_000


# ---------------------------------------------------------------------------
# Core evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_balanced(
    X0, X1,            # test rows for class 0 and class 1 respectively
    y0, y1,            # corresponding labels (all 0s and all 1s)
    clf,
    n_per_class: int,
    rng: "np.random.Generator",
) -> dict[str, float]:
    """Return averaged metrics over N_ROUNDS balanced draws.

    Each round independently samples *n_per_class* rows from each class with
    replacement (so even the smallest minority class can fill every round),
    concatenates them into a balanced set, and computes precision / recall /
    F1.  The round-level scores are averaged at the end.
    """
    precisions, recalls, f1s = [], [], []
    for _ in range(N_ROUNDS):
        idx0 = rng.choice(len(X0), size=n_per_class, replace=len(X0) < n_per_class)
        idx1 = rng.choice(len(X1), size=n_per_class, replace=len(X1) < n_per_class)

        X_sample = np.concatenate([X0[idx0], X1[idx1]], axis=0)
        y_sample = np.concatenate([y0[idx0], y1[idx1]], axis=0)

        y_pred = clf.predict(X_sample)
        precisions.append(precision_score(y_sample, y_pred, average=_AVG, zero_division=0))
        recalls.append(   recall_score(   y_sample, y_pred, average=_AVG, zero_division=0))
        f1s.append(       f1_score(       y_sample, y_pred, average=_AVG, zero_division=0))

    return {
        "precision": float(np.mean(precisions)),
        "recall":    float(np.mean(recalls)),
        "f1_score":  float(np.mean(f1s)),
    }


def _evaluate_one(
    X0, X1,
    y0, y1,
    n_per_class: int,
    model_path: Path,
    seed: int = 42,
) -> dict[str, float]:
    """Load a model and evaluate it with balanced sampling."""
    clf = joblib.load(model_path)
    rng = np.random.default_rng(seed)
    return _evaluate_balanced(X0, X1, y0, y1, clf, n_per_class, rng)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate() -> pd.DataFrame:
    """Evaluate every trained model and return a tidy results DataFrame.

    Columns
    -------
    dataset, mode, num_leaves, n_estimators, precision, recall, f1_score
    """
    rows: list[dict] = []

    for dataset_path in DATASETS:
        dataset_name = Path(dataset_path).stem  # e.g. "NF-BoT-IoT-v2"
        print(f"\n{'=' * 60}")
        print(f"Dataset : {dataset_name}")
        print(f"{'=' * 60}")

        for integer_only in (False, True):
            mode       = "integer" if integer_only else "float"
            dtype_label = "uint16"  if integer_only else "float"
            test_path  = str(
                REPO_ROOT
                / Path(dataset_path).parent
                / f"{dataset_name}_{dtype_label}_test.csv"
            )

            print(f"\n  [mode={mode}] Loading test split …")
            df = load_dataset(test_path, integer_only=integer_only)

            feature_cols = [c for c in FEATURES if c in df.columns]
            X_test = df[feature_cols].values
            y_test  = df[LABEL_COL].values
            print(f"  [mode={mode}] Test samples : {len(X_test):>10,}")

            # Split by class for balanced sampling.
            mask0 = (y_test == 0)
            X0, y0 = X_test[mask0],  y_test[mask0]
            X1, y1 = X_test[~mask0], y_test[~mask0]
            n_per_class = min(len(X0), len(X1), N_PER_CLASS_MAX)
            print(
                f"  [mode={mode}] Class 0: {len(X0):,}  Class 1: {len(X1):,}"
                f"  → {n_per_class:,} samples/class × {N_ROUNDS} rounds"
            )

            for num_leaves in MAX_LEAVES:
                for n_estimators in N_ESTIMATORS:
                    model_path = (
                        MODELS_DIR
                        / dataset_name
                        / mode
                        / str(num_leaves)
                        / str(n_estimators)
                        / "lgbm.joblib"
                    )

                    if not model_path.exists():
                        print(
                            f"  [mode={mode}] SKIP  num_leaves={num_leaves:<5}"
                            f" n_estimators={n_estimators:<4}  (model not found)"
                        )
                        continue

                    print(
                        f"  [mode={mode}] Eval  num_leaves={num_leaves:<5}"
                        f" n_estimators={n_estimators:<4} … ",
                        end="",
                        flush=True,
                    )
                    metrics = _evaluate_one(X0, X1, y0, y1, n_per_class, model_path)
                    print(
                        f"P={metrics['precision']:.4f}"
                        f"  R={metrics['recall']:.4f}"
                        f"  F1={metrics['f1_score']:.4f}"
                    )
                    rows.append(
                        {
                            "dataset":      dataset_name,
                            "mode":         mode,
                            "num_leaves":   num_leaves,
                            "n_estimators": n_estimators,
                            **metrics,
                        }
                    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Persist the results DataFrame to *path* as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    print(f"\nCSV  saved → {path.relative_to(REPO_ROOT)}")


def save_markdown(df: pd.DataFrame, path: Path) -> None:
    """Write a Markdown comparison report to *path*.

    One section (H2) per dataset; within each section one table per feature
    mode so readers can directly compare float vs integer models side-by-side.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("# Model Comparison – LightGBM on NetFlow v2 Datasets\n")
    lines.append(
        "Metrics are **weighted** averages of per-class precision, recall and F1-score"
        " on the held-out 30 % test split.\n"
    )
    lines.append(f"Average strategy: `{_AVG}`\n")

    for dataset_name in df["dataset"].unique():
        lines.append(f"\n## {dataset_name}\n")

        for mode in ("float", "integer"):
            subset = df[(df["dataset"] == dataset_name) & (df["mode"] == mode)]
            if subset.empty:
                continue

            lines.append(f"### Mode: {mode}\n")

            # Pivot: rows = num_leaves, columns = n_estimators
            for metric in ("precision", "recall", "f1_score"):
                pivot = subset.pivot_table(
                    index="num_leaves",
                    columns="n_estimators",
                    values=metric,
                )
                pivot.index.name   = "num\\_leaves \\ n\\_est"
                pivot.columns.name = None

                # Rename columns to be readable
                pivot.columns = [f"n={c}" for c in pivot.columns]

                lines.append(f"#### {metric.replace('_', ' ').title()}\n")
                lines.append(pivot.to_markdown(floatfmt=".4f"))
                lines.append("\n")

        # Summary table: best configuration per mode
        lines.append(f"### Best configuration by F1-score ({dataset_name})\n")
        best_rows = []
        for mode in ("float", "integer"):
            subset = df[(df["dataset"] == dataset_name) & (df["mode"] == mode)]
            if subset.empty:
                continue
            best = subset.loc[subset["f1_score"].idxmax()].copy()
            best["mode"] = mode
            best_rows.append(best)

        if best_rows:
            best_df = pd.DataFrame(best_rows)[
                ["mode", "num_leaves", "n_estimators", "precision", "recall", "f1_score"]
            ].reset_index(drop=True)
            lines.append(best_df.to_markdown(index=False, floatfmt=".4f"))
            lines.append("\n")

    # --- Overall best per dataset (F1) ---
    lines.append("\n## Overall Best Model per Dataset\n")
    overall_rows = []
    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]
        if subset.empty:
            continue
        best = subset.loc[subset["f1_score"].idxmax()].copy()
        overall_rows.append(best)

    if overall_rows:
        overall_df = pd.DataFrame(overall_rows)[
            ["dataset", "mode", "num_leaves", "n_estimators", "precision", "recall", "f1_score"]
        ].reset_index(drop=True)
        lines.append(overall_df.to_markdown(index=False, floatfmt=".4f"))
        lines.append("\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"MD   saved → {path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def test() -> None:
    """Run the full evaluation pipeline and write both output files."""
    results = evaluate()

    if results.empty:
        print("\nNo results collected – make sure models have been trained first.")
        return

    save_csv(results,      REPORTS_DIR / "metrics.csv")
    save_markdown(results, REPORTS_DIR / "metrics.md")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    test()
