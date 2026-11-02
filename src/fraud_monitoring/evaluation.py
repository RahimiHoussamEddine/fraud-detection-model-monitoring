"""Rare-event ranking and capacity-constrained review metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional arrays of equal length")
    if y_true.size == 0 or not np.isfinite(y_score).all():
        raise ValueError("Ranking metrics require finite scores and at least one row")
    if np.unique(y_true).size != 2:
        raise ValueError("Ranking metrics require both target classes")
    return {
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "prevalence": float(y_true.mean()),
    }


def capacity_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    alert_fraction: float,
    *,
    tie_seed: int = 2026,
) -> dict[str, Any]:
    """Evaluate the top-ranked queue at a fixed review fraction.

    Equal scores are broken with a deterministic seeded permutation so tied
    tree scores do not systematically favor earlier rows in the time series.
    """
    y_true = np.asarray(labels, dtype=np.int8)
    y_score = np.asarray(scores, dtype=np.float64)
    if y_true.shape != y_score.shape or y_true.ndim != 1:
        raise ValueError("Labels and scores must be one-dimensional arrays of equal length")
    if y_true.size == 0 or not np.isfinite(y_score).all():
        raise ValueError("Capacity metrics require finite scores and at least one row")
    if not 0 < alert_fraction <= 1:
        raise ValueError("Alert fraction must be in (0, 1]")
    positive_count = int(y_true.sum())
    if positive_count == 0:
        raise ValueError("Capacity metrics require at least one positive label")

    alerts = min(len(y_true), max(1, math.ceil(len(y_true) * alert_fraction)))
    tie_break = np.random.default_rng(tie_seed).random(len(y_score))
    order = np.lexsort((tie_break, -y_score))
    selected = order[:alerts]
    caught = int(y_true[selected].sum())
    return {
        "alert_fraction": float(alert_fraction),
        "alert_count": int(alerts),
        "alert_rate": float(alerts / len(y_true)),
        "fraud_captured": caught,
        "fraud_count": positive_count,
        "precision": float(caught / alerts),
        "recall": float(caught / positive_count),
        "minimum_alert_score": float(y_score[selected].min()),
    }
