"""Partial dependence-based feature importance (Greenwell, Boehmke & McCarthy 2018).

Reference
---------
Greenwell, B. M., Boehmke, B. C., & McCarthy, A. J. (2018).
    A Simple and Effective Model-Based Variable Importance Measure.
    arXiv:1805.04755.

For each feature j the importance is defined as the standard deviation of
the partial dependence function evaluated over a uniform quantile grid of
``grid_resolution`` points:

    VI(x_j) = SD( f̂(x_{j,1}), f̂(x_{j,2}), ..., f̂(x_{j,k}) )

where

    f̂(x_{j,l}) = (1/n) Σ_i  ĝ(x_{j,l}, x_{i, -j})

and ĝ is the model's predicted class-1 probability.

Typical usage
-------------
>>> importance_df = aggregate_pd_importance()
>>> best = top_features(importance_df, n=20)
>>> print(best)

Or run this module directly::

    python features.py
"""

import os
import sys
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

# Allow running directly from the src/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ATTACK_COL,
    DATASETS,
    LABEL_COL,
    MAX_LEAVES,
    N_ESTIMATORS,
)
from dataset import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# ---------------------------------------------------------------------------
# Core partial-dependence helpers
# ---------------------------------------------------------------------------


def _partial_dependence(
    model: LGBMClassifier,
    X: np.ndarray,
    feature_idx: int,
    grid_resolution: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 1-D partial dependence for *feature_idx*.

    For each grid point the value of the target feature column is fixed at
    that point throughout the entire sample, and the prediction is averaged
    over all rows.  This marginalises out the effect of every other feature.

    Parameters
    ----------
    model:
        A fitted LightGBM classifier (or any estimator that implements
        ``predict_proba``).
    X:
        2-D float64 array, shape ``(n_samples, n_features)``.  A copy is
        made internally so the caller's array is never mutated.
    feature_idx:
        Column index of the feature whose partial dependence is required.
    grid_resolution:
        Number of quantile-spaced grid points at which to evaluate the
        partial dependence.  Duplicate quantile values are removed, so the
        effective grid may be smaller for low-cardinality features.

    Returns
    -------
    grid : np.ndarray, shape ``(k,)``
        Feature values at which the partial dependence was evaluated.
    pd_values : np.ndarray, shape ``(k,)``
        Mean predicted class-1 probability at each grid point.
    """
    # Build a quantile grid so that sparse / heavily-skewed features are
    # covered proportionally rather than by equally-spaced raw values.
    quantile_points = np.linspace(0, 100, grid_resolution)
    grid = np.unique(np.percentile(X[:, feature_idx], quantile_points))

    X_work = X.copy()
    pd_values = np.empty(len(grid), dtype=np.float64)
    for k, val in enumerate(grid):
        X_work[:, feature_idx] = val
        pd_values[k] = model.predict_proba(X_work)[:, 1].mean()

    return grid, pd_values


def pd_feature_importance(
    model: LGBMClassifier,
    X: np.ndarray,
    feature_names: Sequence[str],
    grid_resolution: int = 20,
) -> pd.Series:
    """Compute PD-based feature importance for every feature.

    The importance of feature j is the standard deviation of its partial
    dependence function (Greenwell et al., 2018).  A larger spread means
    the model's average prediction changes more as feature j varies, hence
    the feature carries more weight.

    Parameters
    ----------
    model:
        A fitted LightGBM classifier.
    X:
        2-D float64 array, shape ``(n_samples, n_features)``.
    feature_names:
        Ordered sequence of feature names corresponding to the columns of *X*.
    grid_resolution:
        Number of quantile grid points per feature (default 20).

    Returns
    -------
    pd.Series
        PD-based importance for each feature, sorted in descending order.
        The index contains the feature names.
    """
    importances: dict[str, float] = {}
    for idx, name in enumerate(feature_names):
        _, pd_vals = _partial_dependence(model, X, idx, grid_resolution)
        importances[name] = float(pd_vals.std())

    return pd.Series(importances).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Stratified subsampling helper
# ---------------------------------------------------------------------------


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a class-balanced subsample of *X*.

    Exactly ``sample_size // 2`` rows are drawn from each class (0 and 1).
    If a class has fewer available rows than requested, all rows for that
    class are used (without replacement).

    Parameters
    ----------
    X:
        2-D float64 array, shape ``(n_samples, n_features)``.
    y:
        1-D integer label array, shape ``(n_samples,)``.
    sample_size:
        Total target sample size.  Half is allocated to each class.
    rng:
        NumPy random Generator instance.

    Returns
    -------
    np.ndarray
        Subsampled rows of *X*, shape ``(k, n_features)`` where
        ``k <= sample_size``.
    """
    half = sample_size // 2
    idx_per_class = []
    for cls in (0, 1):
        cls_idx = np.where(y == cls)[0]
        n_draw = min(half, len(cls_idx))
        idx_per_class.append(rng.choice(cls_idx, size=n_draw, replace=False))
    idx = np.concatenate(idx_per_class)
    rng.shuffle(idx)
    return X[idx]


# ---------------------------------------------------------------------------
# Aggregation across all trained models and datasets
# ---------------------------------------------------------------------------


def aggregate_pd_importance(
    modes: Sequence[str] = ("float", "integer"),
    max_leaves_list: Sequence[int] | None = None,
    n_estimators_list: Sequence[int] | None = None,
    grid_resolution: int = 20,
    sample_size: int = 2_000,
    n_rounds: int = 33,
    random_state: int = 42,
) -> pd.DataFrame:
    """Aggregate PD-based feature importance across all datasets and models.

    For every combination of (dataset, mode, max_leaves, n_estimators) the
    corresponding pre-trained LightGBM model and test split are loaded.
    PD-based importance (Greenwell et al., 2018) is computed over
    *n_rounds* independent stratified subsamples of the test data and
    averaged; the mean is stored as one row in the returned DataFrame.

    Each subsample is class-balanced: exactly ``sample_size // 2`` rows are
    drawn from class 0 and ``sample_size // 2`` from class 1 (capped at the
    available count for each class), so that the importance estimate is not
    dominated by the majority class.

    Features that do not appear in a particular mode (e.g. ``SRC_TO_DST_SECOND_BYTES``
    and ``DST_TO_SRC_SECOND_BYTES`` are absent from integer-only models) receive
    ``NaN`` for that row so that aggregation remains well-defined.

    Parameters
    ----------
    modes:
        Subset of ``{"float", "integer"}`` to include.
    max_leaves_list:
        ``num_leaves`` values to include.  Defaults to ``config.MAX_LEAVES``.
    n_estimators_list:
        ``n_estimators`` values to include.  Defaults to ``config.N_ESTIMATORS``.
    grid_resolution:
        Number of quantile grid points per feature passed to
        :func:`pd_feature_importance`.
    sample_size:
        Total rows per stratified subsample (``sample_size // 2`` per class).
        Subsampling makes the computation tractable; 2 000 rows is typically
        sufficient for stable rankings.
    n_rounds:
        Number of independent stratified subsampling rounds over which to
        average the importance scores.  More rounds reduce variance at the
        cost of proportionally more compute (default 5).
    random_state:
        Seed for the subsampling RNG.

    Returns
    -------
    pd.DataFrame
        Each row corresponds to one (dataset, mode, max_leaves, n_estimators)
        evaluation run.  The index columns are ``"dataset"``, ``"mode"``,
        ``"max_leaves"``, and ``"n_estimators"``; all remaining columns are
        feature importance scores.
    """
    if max_leaves_list is None:
        max_leaves_list = MAX_LEAVES
    if n_estimators_list is None:
        n_estimators_list = N_ESTIMATORS

    rng = np.random.default_rng(random_state)
    records: list[dict] = []

    for dataset_path in DATASETS:
        dataset_name = Path(dataset_path).stem  # e.g. "NF-BoT-IoT-v2"

        for mode in modes:
            integer_only = mode == "integer"
            dtype_label = "uint16" if integer_only else "float"
            test_path = str(
                REPO_ROOT
                / Path(dataset_path).parent
                / f"{dataset_name}_{dtype_label}_test.csv"
            )

            print(f"\n  Loading test split: {dataset_name} [{mode}] …")
            df = load_dataset(test_path, integer_only=integer_only)
            feature_cols = [c for c in df.columns if c not in (LABEL_COL, ATTACK_COL)]

            # Keep full float64 feature matrix and labels for stratified
            # per-round subsampling inside the model loop.
            X_full = df[feature_cols].values.astype(np.float64)
            y_full = df[LABEL_COL].values

            for max_leaves in max_leaves_list:
                for n_est in n_estimators_list:
                    model_path = (
                        MODELS_DIR
                        / dataset_name
                        / mode
                        / str(max_leaves)
                        / str(n_est)
                        / "lgbm.joblib"
                    )
                    if not model_path.exists():
                        print(
                            f"    [SKIP] model not found: "
                            f"{model_path.relative_to(REPO_ROOT)}"
                        )
                        continue

                    print(
                        f"    {dataset_name}/{mode}/leaves={max_leaves}"
                        f"/n_est={n_est} …",
                        end=" ",
                        flush=True,
                    )
                    clf: LGBMClassifier = joblib.load(model_path)

                    # Average importance over n_rounds stratified subsamples
                    # (50 % class-0 / 50 % class-1) to reduce sampling variance.
                    round_imps: list[pd.Series] = []
                    for _ in range(n_rounds):
                        X_sample = _stratified_subsample(
                            X_full, y_full, sample_size, rng
                        )
                        round_imps.append(
                            pd_feature_importance(
                                clf, X_sample, feature_cols, grid_resolution
                            )
                        )
                    imp = pd.concat(round_imps, axis=1).mean(axis=1)

                    record: dict = {
                        "dataset": dataset_name,
                        "mode": mode,
                        "max_leaves": max_leaves,
                        "n_estimators": n_est,
                    }
                    record.update(imp.to_dict())
                    records.append(record)
                    print(f"done ({n_rounds} rounds)")

    if not records:
        raise RuntimeError(
            "No importance records collected – are the models trained?"
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Ranking helper
# ---------------------------------------------------------------------------

_META_COLS = ("dataset", "mode", "max_leaves", "n_estimators")


def top_features(
    importance_df: pd.DataFrame,
    n: int = 20,
    meta_cols: Sequence[str] = _META_COLS,
) -> pd.Series:
    """Return the top-*n* features ranked by mean PD-based importance.

    The mean is taken over all rows in *importance_df* (i.e. all
    dataset / model combinations), with ``NaN`` values excluded from each
    per-feature average so that features absent from some modes still receive
    a meaningful score based on the runs where they were present.

    Parameters
    ----------
    importance_df:
        DataFrame as returned by :func:`aggregate_pd_importance`.
    n:
        Number of top features to return.
    meta_cols:
        Column names in *importance_df* that are *not* importance scores.

    Returns
    -------
    pd.Series
        Mean PD-based importance score for each of the top-*n* features,
        sorted in descending order.  The index contains the feature names.
    """
    score_cols = [c for c in importance_df.columns if c not in meta_cols]
    mean_scores = importance_df[score_cols].mean(numeric_only=True, skipna=True)
    return mean_scores.sort_values(ascending=False).head(n)


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def export_top_features_md(
    top: pd.Series,
    output_path: Path | str,
    modes: Sequence[str] = ("float", "integer"),
    meta_cols: Sequence[str] = _META_COLS,
    importance_df: pd.DataFrame | None = None,
) -> Path:
    """Write the top-N feature ranking to a Markdown file.

    The report includes:
    * The parameters used (modes, number of models aggregated).
    * A ranked table of features with their mean PD-based importance score.
    * Per-dataset mean importance for each top feature (when
      *importance_df* is supplied).

    Parameters
    ----------
    top:
        ``pd.Series`` as returned by :func:`top_features`, with feature
        names as the index and mean importance scores as values.
    output_path:
        Destination ``.md`` file path.  Parent directories are created
        automatically.
    modes:
        The modes that were included in the aggregation (informational).
    meta_cols:
        Columns in *importance_df* that are metadata, not scores.
    importance_df:
        Full DataFrame from :func:`aggregate_pd_importance`.  When
        supplied a per-dataset breakdown table is appended.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(top)
    lines: list[str] = []
    lines.append("# PD-based Feature Importance")
    lines.append("")
    lines.append(
        "Importance measure: **standard deviation of the partial dependence function** "
        "(Greenwell, Boehmke & McCarthy, 2018, arXiv:1805.04755)."
    )
    lines.append("")

    # --- parameters block ---
    lines.append("## Parameters")
    lines.append("")
    n_models = len(importance_df) if importance_df is not None else "N/A"
    lines.append("| Parameter | Value |")
    lines.append("|:----------|:------|")
    lines.append(f"| Modes | {', '.join(modes)} |")
    lines.append(f"| Models aggregated | {n_models} |")
    lines.append(f"| Top-N | {n} |")
    lines.append("")

    # Helper: render a ranked importance table from a pd.Series.
    def _ranking_table(series: pd.Series) -> list[str]:
        rows = ["| Rank | Feature | Mean Importance |", "|-----:|:--------|----------------:|"]
        for rank, (feat, score) in enumerate(series.items(), start=1):
            rows.append(f"| {rank} | `{feat}` | {score:.6f} |")
        return rows

    # Helper: render a rank × dataset side-by-side table for one mode slice.
    def _dataset_table(
        imp_slice: pd.DataFrame,
        score_cols: list[str],
        datasets: list[str],
        n_top: int,
    ) -> list[str]:
        ds_tops: dict[str, pd.Series] = {}
        for ds in datasets:
            mask = imp_slice["dataset"] == ds
            if not mask.any():
                ds_tops[ds] = pd.Series(dtype=float)
                continue
            ds_mean = imp_slice.loc[mask, score_cols].mean(numeric_only=True, skipna=True)
            ds_tops[ds] = ds_mean.sort_values(ascending=False).head(n_top)

        header = "| Rank | " + " | ".join(datasets) + " |"
        sep = "|-----:" + "".join([f"|:{'-' * max(len(ds), 7)}-" for ds in datasets]) + "|"
        rows = [header, sep]
        for rank in range(n_top):
            row = f"| {rank + 1} |"
            for ds in datasets:
                series = ds_tops[ds]
                if rank < len(series):
                    feat = series.index[rank]
                    score = series.iloc[rank]
                    row += f" `{feat}` ({score:.4f}) |"
                else:
                    row += " — |"
            rows.append(row)
        return rows

    # --- overall ranking — one subsection per mode ---
    lines.append("## Overall Ranking")
    lines.append("")

    if importance_df is not None:
        score_cols = [c for c in importance_df.columns if c not in meta_cols]
        present_modes = [
            m for m in modes if m in importance_df["mode"].unique()
        ]
        for mode in present_modes:
            lines.append(f"### Mode: {mode}")
            lines.append("")
            mode_slice = importance_df[importance_df["mode"] == mode]
            mode_mean = mode_slice[score_cols].mean(numeric_only=True, skipna=True)
            mode_top = mode_mean.sort_values(ascending=False).head(n)
            lines.extend(_ranking_table(mode_top))
            lines.append("")
    else:
        # Fallback: render the supplied overall top series without mode split.
        lines.extend(_ranking_table(top))
        lines.append("")

    # --- per-dataset top-N ranking — one subsection per mode ---
    if importance_df is not None:
        datasets = importance_df["dataset"].unique().tolist()
        lines.append("## Per-Dataset Top Features")
        lines.append("")
        lines.append(
            "Each column shows the top features ranked by mean PD-based importance "
            "within that dataset independently."
        )
        lines.append("")
        for mode in present_modes:
            lines.append(f"### Mode: {mode}")
            lines.append("")
            mode_slice = importance_df[importance_df["mode"] == mode]
            lines.extend(
                _dataset_table(mode_slice, score_cols, datasets, n)
            )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import warnings

    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(
        description="Compute PD-based feature importance (Greenwell et al., 2018) "
                    "across all trained LightGBM models and datasets."
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["float", "integer"],
        choices=["float", "integer"],
        help="Feature modes to include (default: both).",
    )
    parser.add_argument(
        "--max-leaves",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="num_leaves values to include (default: all from config).",
    )
    parser.add_argument(
        "--n-estimators",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="n_estimators values to include (default: all from config).",
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=20,
        metavar="K",
        help="Quantile grid points per feature (default: 20).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=2_000,
        metavar="N",
        help="Rows per stratified subsample, split 50/50 by class (default: 2000).",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=33,
        metavar="R",
        help="Stratified subsampling rounds to average per model (default: 33).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        metavar="N",
        help="Number of top features to display (default: 20).",
    )
    parser.add_argument(
        "--from-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Load a pre-computed importance CSV instead of re-running the "
            "full computation.  The file must have been produced by a previous "
            "run with --output.  When this flag is set all computation "
            "arguments (--grid-resolution, --sample-size, --n-rounds, "
            "--max-leaves, --n-estimators) are ignored."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "reports" / "pd_importance.csv"),
        metavar="PATH",
        help="Path to save the full importance DataFrame.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=str(REPO_ROOT / "reports" / "pd_importance.md"),
        metavar="PATH",
        help="Markdown report path (default: reports/pd_importance.md).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PD-based feature importance  (Greenwell et al., 2018)")
    print("=" * 60)

    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"--from-csv path not found: {csv_path}")
        print(f"\nLoading pre-computed importance table from: {csv_path}")
        imp_df = pd.read_csv(csv_path)
        # Filter to the requested modes so the report respects --modes.
        if "mode" in imp_df.columns:
            imp_df = imp_df[imp_df["mode"].isin(args.modes)]
        if imp_df.empty:
            raise ValueError(
                f"No rows remain after filtering to modes {args.modes}. "
                "Check that the CSV contains the requested modes."
            )
    else:
        imp_df = aggregate_pd_importance(
            modes=args.modes,
            max_leaves_list=args.max_leaves,
            n_estimators_list=args.n_estimators,
            grid_resolution=args.grid_resolution,
            sample_size=args.sample_size,
            n_rounds=args.n_rounds,
        )

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imp_df.to_csv(out_path, index=False)
            print(f"\nFull importance table saved to: {out_path}")

    top = top_features(imp_df, n=args.top_n)
    print(f"\nTop-{args.top_n} features by mean PD-based importance:")
    print("-" * 40)
    print(top.to_string())

    report_path = export_top_features_md(
        top,
        output_path=args.report,
        modes=args.modes,
        importance_df=imp_df,
    )
    print(f"\nMarkdown report saved to: {report_path}")
