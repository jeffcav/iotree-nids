"""Export a trained LightGBM model (.joblib) as a self-contained C file.

The generated C file contains:
  - ``#define`` constants for every feature index.
  - One static inline function per tree that walks the if-else structure and
    returns the leaf value (raw score contribution).
  - A ``lgbm_predict`` function that sums all tree outputs and applies the
    link function (sigmoid for binary classification, softmax for multiclass)
    to return the predicted class label.

Usage
-----
    python -m src.export models/NF-BoT-IoT-v2/float/64/9/lgbm.joblib

The C file is written next to the .joblib file as ``lgbm.c``.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import joblib

# Allow running directly from src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEATURES, FLOAT_FEATURES


# ---------------------------------------------------------------------------
# Internal tree-traversal helpers
# ---------------------------------------------------------------------------

def _node_to_c(node: dict[str, Any], indent: int, feature_names: list[str]) -> str:
    """Recursively convert a LightGBM tree node to a C if-else fragment.

    Parameters
    ----------
    node:
        A node dict from ``booster_.dump_model()["tree_info"][i]["tree_structure"]``.
        Leaf nodes have a ``"leaf_value"`` key; internal nodes have
        ``"split_feature"``, ``"threshold"``, ``"left_child"``, and
        ``"right_child"``.
    indent:
        Current indentation level (number of 4-space blocks).
    feature_names:
        Ordered list of feature names used by the model.

    Returns
    -------
    str
        C source fragment for this subtree.
    """
    pad = "    " * indent

    # Leaf node
    if "leaf_value" in node:
        return f"{pad}return {node['leaf_value']:.17g};\n"

    feat_idx: int = node["split_feature"]
    threshold: float = node["threshold"]
    feat_name: str = feature_names[feat_idx] if feat_idx < len(feature_names) else f"f{feat_idx}"
    decision: str = node.get("decision_type", "<=")

    # LightGBM only uses <= for numerical splits; categorical splits are
    # represented differently (split_type == "categorical").  We handle the
    # categorical case as a bitset membership check.
    split_type: str = node.get("split_type", "numerical")

    if split_type == "categorical":
        # threshold is a space-separated list of category values that go LEFT.
        cats = {int(c) for c in str(threshold).split("||")}
        cond = " || ".join(f"(int)(f[{feat_idx}]) == {c}" for c in sorted(cats))
        condition = f"({cond})"
    else:
        condition = f"f[{feat_idx}] {decision} {threshold:.17g}"

    left_c  = _node_to_c(node["left_child"],  indent + 1, feature_names)
    right_c = _node_to_c(node["right_child"], indent + 1, feature_names)

    return (
        f"{pad}/* {feat_name} {decision} {threshold:.17g} */\n"
        f"{pad}if ({condition}) {{\n"
        f"{left_c}"
        f"{pad}}} else {{\n"
        f"{right_c}"
        f"{pad}}}\n"
    )


def _tree_to_c_function(
    tree_index: int,
    tree_info: dict[str, Any],
    feature_names: list[str],
) -> str:
    """Return the full C source for one tree as a ``static double`` function."""
    body = _node_to_c(tree_info["tree_structure"], 1, feature_names)
    return (
        f"static double _tree_{tree_index}(const double *f)\n"
        f"{{\n"
        f"{body}"
        f"}}\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_lgbm_to_c(model_path: str | Path, out_path: str | Path | None = None, runnable: bool = False) -> Path:
    """Load a LightGBM model from *model_path* and write a C implementation.

    Parameters
    ----------
    model_path:
        Path to a ``lgbm.joblib`` file produced by :mod:`src.train`.
    out_path:
        Destination ``.c`` file.  Defaults to ``lgbm.c`` in the same
        directory as *model_path*.

    Returns
    -------
    Path
        Absolute path of the written C file.
    """
    model_path = Path(model_path).resolve()
    if out_path is None:
        out_path = model_path.with_name("lgbm.c")
    out_path = Path(out_path).resolve()

    # ------------------------------------------------------------------ load
    clf = joblib.load(model_path)

    # Unwrap sklearn Pipeline if needed
    try:
        from sklearn.pipeline import Pipeline as _Pipeline
        if isinstance(clf, _Pipeline):
            clf = clf.named_steps[clf.steps[-1][0]]
    except ImportError:
        pass

    booster = clf.booster_
    n_classes: int = clf.n_classes_  # 2 → binary, >2 → multiclass

    # Use the real feature names from config when the model was trained without
    # named columns (LightGBM then auto-assigns "Column_N" placeholders).
    raw_names: list[str] = list(clf.feature_name_) if hasattr(clf, "feature_name_") else []
    _generic = all(n.startswith("Column_") for n in raw_names) if raw_names else True
    if _generic:
        # For float mode all 41 features are present; for integer mode the two
        # float-only features are dropped, so we exclude them from FEATURES to
        # keep ordering consistent.
        n = clf.n_features_in_
        candidate = [f for f in FEATURES if f not in FLOAT_FEATURES] if n < len(FEATURES) else list(FEATURES)
        feature_names = candidate[:n]
    else:
        feature_names = raw_names

    dump = booster.dump_model()
    trees: list[dict] = dump["tree_info"]
    n_trees = len(trees)

    # ---------------------------------------------------------- C generation
    lines: list[str] = []

    # Header comment
    lines.append(
        textwrap.dedent(f"""\
        /*
         * Auto-generated C implementation of a LightGBM model.
         * Source : {model_path}
         * Trees  : {n_trees}
         * Classes: {n_classes}
         * Features: {len(feature_names)}
         *
         * Compile with:
         *   gcc -O2 -o lgbm lgbm.c -lm
         */
        """)
    )

    lines.append("#include <math.h>\n\n")

    # Feature index #defines
    lines.append("/* Feature indices */\n")
    for idx, name in enumerate(feature_names):
        lines.append(f"#define FEAT_{name.upper()} {idx}\n")
    lines.append("\n")

    # Number of classes
    lines.append(f"#define LGBM_NUM_CLASSES {n_classes}\n\n")

    # One function per tree
    lines.append("/* --- Tree functions --- */\n\n")
    for i, tree_info in enumerate(trees):
        lines.append(_tree_to_c_function(i, tree_info, feature_names))
        lines.append("\n")

    # Dispatch table
    lines.append(
        "/* Dispatch table – index by tree number */\n"
        "static double (*_trees[])(const double *) = {\n"
    )
    for i in range(n_trees):
        lines.append(f"    _tree_{i},\n")
    lines.append("};\n\n")

    # Predict function
    if n_classes == 2:
        lines.append(_binary_predict_fn(n_trees))
    else:
        lines.append(_multiclass_predict_fn(n_classes, n_trees))

    # Optional main() for quick smoke-test
    if runnable:
        lines.append(_main_stub(feature_names, n_classes))

    c_source = "".join(lines)
    out_path.write_text(c_source, encoding="utf-8")
    print(f"Exported C model → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Predict-function templates
# ---------------------------------------------------------------------------

def _binary_predict_fn(n_trees: int) -> str:
    return textwrap.dedent(f"""\
        /*
         * lgbm_score – raw log-odds (sum of leaf values across all trees).
         */
        double lgbm_score(const double *f)
        {{
            double s = 0.0;
            for (int i = 0; i < {n_trees}; ++i)
                s += _trees[i](f);
            return s;
        }}

        /*
         * lgbm_predict – returns 0 (benign) or 1 (attack).
         * The sigmoid maps the raw score to a probability; threshold = 0.5.
         */
        int lgbm_predict(const double *f)
        {{
            double prob = 1.0 / (1.0 + exp(-lgbm_score(f)));
            return (prob >= 0.5) ? 1 : 0;
        }}

    """)


def _multiclass_predict_fn(n_classes: int, n_trees: int) -> str:
    """Raw scores and softmax argmax for multiclass models."""
    n_iters = n_trees // n_classes
    return textwrap.dedent(f"""\
        /*
         * lgbm_scores – fills *scores* (length {n_classes}) with the raw
         * log-odds for each class.  Trees are interleaved:
         *   tree 0 → class 0, iter 0
         *   tree 1 → class 1, iter 0
         *   ...
         *   tree {n_classes} → class 0, iter 1
         */
        void lgbm_scores(const double *f, double *scores)
        {{
            for (int c = 0; c < {n_classes}; ++c) scores[c] = 0.0;
            for (int it = 0; it < {n_iters}; ++it)
                for (int c = 0; c < {n_classes}; ++c)
                    scores[c] += _trees[it * {n_classes} + c](f);
        }}

        /*
         * lgbm_predict – returns the class index with the highest softmax
         * probability.
         */
        int lgbm_predict(const double *f)
        {{
            double scores[{n_classes}];
            lgbm_scores(f, scores);

            /* softmax argmax – we only need the argmax, so we skip exp() */
            int best = 0;
            for (int c = 1; c < {n_classes}; ++c)
                if (scores[c] > scores[best]) best = c;
            return best;
        }}

    """)


def _main_stub(feature_names: list[str], n_classes: int) -> str:
    """A minimal main() that prints a prediction for an all-zero feature vector."""
    return textwrap.dedent(f"""\
        /* --- Smoke-test entry point (remove before embedding) --- */
        #include <stdio.h>

        int main(void)
        {{
            double features[{len(feature_names)}] = {{0}};
            int label = lgbm_predict(features);
            printf("predicted class: %d\\n", label);
            return 0;
        }}
    """)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a LightGBM .joblib model as an if-else C file."
    )
    parser.add_argument("model_path", help="Path to lgbm.joblib")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .c file path (default: lgbm.c next to model_path)",
    )
    parser.add_argument(
        "-r", "--runnable",
        default=False,
        action="store_true"
    )
    args = parser.parse_args()
    export_lgbm_to_c(args.model_path, args.output, args.runnable)
