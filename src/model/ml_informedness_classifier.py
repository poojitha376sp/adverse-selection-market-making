#!/usr/bin/env python3
"""
ml_informedness_classifier.py

Part 3 / Phase 4, "Now (Part 3, classical ML)" from README's AI/ML plan:
a gradient boosting classifier (scikit-learn) trained on the order-flow
features from `informedness_signal.py`, predicting whether the flow at a
given point in time is "informed" (per that module's volatility-scaled
forward-price-move label). This is the signal both the heuristic-overlay
and principled variants in `avellaneda_stoikov_adverse.py` consume.

Train/test split is CHRONOLOGICAL (first 70% of the capture by time =
train, last 30% = test) -- no shuffling. This matters more than usual
here: order-flow features are strongly autocorrelated over short windows
(a burst of one-sided flow persists for multiple consecutive snapshots),
so a random/shuffled split would leak adjacent-in-time information
between train and test and overstate accuracy. A chronological split is
the honest test of "would this have worked walking forward," which is
the whole point of backtesting this on real captured data.

Usage:
    python src/model/ml_informedness_classifier.py \
        --depth data/raw/depth_btcusdt_20260723_094653.jsonl \
        --trades data/raw/trades_btcusdt_20260723_094653.jsonl \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.informedness_signal import FEATURE_COLUMNS, build_feature_frame  # noqa: E402

from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def chronological_split(frame, train_frac: float = 0.7):
    """Split a time-sorted frame into (train, test) by row position, i.e.
    purely by time -- the first train_frac of rows (earliest in the
    capture) become train, the remaining rows (latest) become test. No
    shuffling anywhere in this pipeline."""
    labeled = frame.dropna(subset=["informed"]).reset_index(drop=True)
    n = len(labeled)
    split_idx = int(n * train_frac)
    train = labeled.iloc[:split_idx]
    test = labeled.iloc[split_idx:]
    return train, test


def train_classifier(train_df, feature_columns=FEATURE_COLUMNS, random_state: int = 42):
    """Fit a GradientBoostingClassifier. Class weighting: the informed
    label is a minority class (~8% positive on this capture -- see
    informedness_signal meta), so training rows are re-weighted inversely
    proportional to class frequency (GradientBoostingClassifier has no
    built-in class_weight param, unlike e.g. RandomForestClassifier, so
    this is done explicitly via sample_weight)."""
    X = train_df[feature_columns].to_numpy()
    y = train_df["informed"].to_numpy()

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    n_total = n_pos + n_neg
    w_pos = n_total / (2.0 * max(n_pos, 1.0))
    w_neg = n_total / (2.0 * max(n_neg, 1.0))
    sample_weight = np.where(y == 1, w_pos, w_neg)

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def evaluate_classifier(model, test_df, feature_columns=FEATURE_COLUMNS, threshold: float = 0.5):
    X = test_df[feature_columns].to_numpy()
    y_true = test_df["informed"].to_numpy()
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(float)

    metrics = {
        "n_test": len(y_true),
        "n_positive_test": int((y_true == 1).sum()),
        "positive_rate_test": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "majority_class_baseline_accuracy": float(max(np.mean(y_true == 0), np.mean(y_true == 1))),
        "decision_threshold": threshold,
    }
    return metrics, y_prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--trailing-window-sec", type=float, default=2.0)
    ap.add_argument("--horizon-sec", type=float, default=2.0)
    ap.add_argument("--k-sigma", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    frame, feat_meta = build_feature_frame(
        args.depth, args.trades,
        trailing_window_sec=args.trailing_window_sec,
        horizon_sec=args.horizon_sec,
        k_sigma=args.k_sigma,
    )
    print(f"[features] {feat_meta['n_rows']} rows, {feat_meta['n_labeled']} labeled, "
          f"informed_rate={feat_meta['informed_rate']:.4f}, "
          f"threshold=${feat_meta['threshold_dollars']:.4f}")

    train_df, test_df = chronological_split(frame, args.train_frac)
    print(f"[split] chronological: train={len(train_df)} rows (earliest "
          f"{args.train_frac:.0%}), test={len(test_df)} rows (latest "
          f"{1 - args.train_frac:.0%}) -- no shuffling")

    model = train_classifier(train_df)
    metrics, test_probs = evaluate_classifier(model, test_df)

    print("\n===== ML informedness classifier -- test-set metrics =====")
    for k, v in metrics.items():
        print(f"  {k:32s}: {v}")

    # Feature importances (gradient-boosting native importance, quick
    # sanity check on which signals the model actually leaned on).
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    print("\n[feature importances]")
    for k, v in sorted(importances.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s}: {v:.4f}")

    # Score EVERY row in the frame (train + test + unlabeled tail) so the
    # backtest harness has a p_informed(t) value at every timestamp,
    # including the last horizon_sec seconds that had no label to train
    # against.
    X_all = frame[FEATURE_COLUMNS].to_numpy()
    p_informed_all = model.predict_proba(X_all)[:, 1]
    frame["p_informed"] = p_informed_all

    scores_path = os.path.join(args.out_dir, "ml_informedness_scores.csv")
    frame[["local_ts", "mid", "informed", "p_informed"]].to_csv(scores_path, index=False)

    model_path = os.path.join(args.out_dir, "ml_informedness_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "feature_meta": feat_meta,
        }, f)

    metrics_out = {
        "feature_meta": feat_meta,
        "train_test_split": {
            "method": "chronological (first train_frac by time -> train, remainder -> test)",
            "train_frac": args.train_frac,
            "n_train": len(train_df),
            "n_test": len(test_df),
        },
        "test_metrics": metrics,
        "feature_importances": importances,
    }
    metrics_path = os.path.join(args.out_dir, "ml_informedness_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    print(f"\nWrote {scores_path}")
    print(f"Wrote {model_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
