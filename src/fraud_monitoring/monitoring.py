"""Simple, inspectable population and score drift summaries."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fraud_monitoring.data import TARGET, TIME
from fraud_monitoring.evaluation import capacity_metrics, ranking_metrics


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], bins: int = 10
) -> float:
    """Compute PSI using quantile bins fitted only on the reference sample."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI requires nonempty reference and current samples")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("PSI samples must be finite")
    if bins < 2:
        raise ValueError("PSI requires at least two bins")

    quantiles = np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1])
    edges = np.unique(np.r_[-np.inf, quantiles, np.inf])
    ref_counts = np.histogram(ref, bins=edges)[0].astype(np.float64)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(np.float64)
    epsilon = 1e-6
    ref_share = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * len(ref_counts))
    cur_share = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * len(cur_counts))
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def feature_drift(
    reference: dict[str, Sequence[float]],
    current: dict[str, Sequence[float]],
    *,
    bins: int = 10,
) -> dict[str, float]:
    """Return per-feature PSI for the exact shared feature set."""
    if set(reference) != set(current):
        raise ValueError("Reference and current feature names must match")
    return {
        name: population_stability_index(reference[name], current[name], bins=bins)
        for name in sorted(reference)
    }


def time_window_monitoring(
    reference,
    current,
    reference_scores: Sequence[float],
    current_scores: Sequence[float],
    feature_names: Sequence[str],
    *,
    window_seconds: int = 4 * 60 * 60,
    alert_fraction: float = 0.0025,
) -> list[dict]:
    """Monitor chronological batches against train-only distributions."""
    if window_seconds <= 0:
        raise ValueError("Window size must be positive")
    reference_score_array = np.asarray(reference_scores, dtype=np.float64)
    scores = np.asarray(current_scores, dtype=np.float64)
    if (
        len(scores) != len(current)
        or len(reference_score_array) != len(reference)
        or not np.isfinite(scores).all()
        or not np.isfinite(reference_score_array).all()
    ):
        raise ValueError("Current scores must be finite and align to current rows")
    start = float(current[TIME].min())
    buckets = np.floor((current[TIME].to_numpy() - start) / window_seconds).astype(int)
    report = []
    for bucket in np.unique(buckets):
        positions = np.flatnonzero(buckets == bucket)
        batch = current.iloc[positions]
        batch_scores = scores[positions]
        reference_features = {name: reference[name].to_numpy() for name in feature_names}
        current_features = {name: batch[name].to_numpy() for name in feature_names}
        drift = feature_drift(reference_features, current_features)
        drift["fraud_score"] = population_stability_index(reference_score_array, batch_scores)

        positives = int(batch[TARGET].sum())
        negatives = len(batch) - positives
        if positives and negatives:
            label_metrics = ranking_metrics(batch[TARGET], batch_scores)
            capacity = capacity_metrics(batch[TARGET], batch_scores, alert_fraction)
        else:
            label_metrics = None
            capacity = None
        ordered_drift = sorted(drift.items(), key=lambda item: (-item[1], item[0]))
        report.append(
            {
                "window_index": int(bucket),
                "time_start_seconds": float(batch[TIME].min()),
                "time_end_seconds": float(batch[TIME].max()),
                "rows": int(len(batch)),
                "fraud_count": positives,
                "fraud_rate": float(positives / len(batch)),
                "label_metrics": label_metrics,
                "capacity_metrics": capacity,
                "psi_status_thresholds": {"watch": 0.10, "alert": 0.25},
                "drift_status": {
                    name: "alert" if value >= 0.25 else "watch" if value >= 0.10 else "stable"
                    for name, value in drift.items()
                },
                "top_drift_features": [
                    {"feature": name, "psi": float(value)} for name, value in ordered_drift[:10]
                ],
                "drift_psi": {name: float(value) for name, value in sorted(drift.items())},
            }
        )
    return report
