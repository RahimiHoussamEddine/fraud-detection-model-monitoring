"""Dataset loading, validation, and leakage-safe chronological partitioning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff

TARGET = "Class"
TIME = "Time"
EXPECTED_FEATURES = [TIME, *(f"V{i}" for i in range(1, 29)), "Amount"]
EXPECTED_COLUMNS = [*EXPECTED_FEATURES, TARGET]


def load_transactions(path: str | Path) -> pd.DataFrame:
    """Read the OpenML Parquet/ARFF snapshot and enforce its expected schema."""
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".arff":
        records, _metadata = arff.loadarff(path)
        frame = pd.DataFrame(records)
    else:
        raise ValueError("Input must be an OpenML Parquet or ARFF file")

    frame.columns = [str(column) for column in frame.columns]
    if set(frame.columns) != set(EXPECTED_COLUMNS):
        missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
        raise ValueError(f"Unexpected source schema; missing={missing}, unexpected={unexpected}")

    frame = frame.loc[:, EXPECTED_COLUMNS].copy()
    frame[TARGET] = frame[TARGET].map(
        lambda value: value.decode("ascii") if isinstance(value, bytes) else value
    )
    for column in EXPECTED_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype("int8")
    frame = frame.sort_values(TIME, kind="mergesort").reset_index(drop=True)
    validate_transactions(frame)
    return frame


def validate_transactions(frame: pd.DataFrame) -> None:
    """Fail closed on schema, target, timestamp, missingness, or numeric errors."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError("Transaction columns or column order do not match the source contract")
    if frame.empty:
        raise ValueError("Transaction data are empty")
    if frame.isna().any().any():
        raise ValueError("Transaction data contain missing values")
    values = frame[EXPECTED_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Transaction features contain non-finite values")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Class must be binary with 0=legitimate and 1=fraud")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both legitimate and fraud labels must be present")
    if (frame[TIME].diff().dropna() < 0).any():
        raise ValueError("Time must be monotonically nondecreasing")
    if (frame["Amount"] < 0).any():
        raise ValueError("Amount must be nonnegative")


def audit_transactions(frame: pd.DataFrame) -> dict[str, Any]:
    """Return compact quality and prevalence evidence for a validated table."""
    validate_transactions(frame)
    fraud_count = int(frame[TARGET].sum())
    exact_duplicate_mask = frame.duplicated(keep=False)
    exact_duplicate_extras = frame.duplicated(keep="first")
    exact_duplicate_groups = int(
        frame.loc[exact_duplicate_mask].groupby(EXPECTED_COLUMNS, dropna=False).ngroups
    )
    model_features = [column for column in EXPECTED_FEATURES if column != TIME]
    model_feature_groups = frame.groupby(model_features, dropna=False, sort=False)
    model_feature_counts = model_feature_groups.size()
    model_feature_duplicate_groups = model_feature_counts[model_feature_counts > 1]
    conflicting_feature_labels = model_feature_groups[TARGET].nunique()
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "fraud_count": fraud_count,
        "legitimate_count": int(len(frame) - fraud_count),
        "fraud_rate": float(frame[TARGET].mean()),
        "exact_duplicate_rows_including_first": int(exact_duplicate_mask.sum()),
        "exact_duplicate_rows_beyond_first": int(exact_duplicate_extras.sum()),
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_extras_by_class": {
            str(int(label)): int(count)
            for label, count in frame.loc[exact_duplicate_extras, TARGET].value_counts().items()
        },
        "identical_model_feature_rows_beyond_first": int(
            (model_feature_duplicate_groups - 1).sum()
        ),
        "identical_model_feature_groups": int(len(model_feature_duplicate_groups)),
        "model_feature_groups_with_conflicting_labels": int((conflicting_feature_labels > 1).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
        "time_min_seconds": float(frame[TIME].min()),
        "time_max_seconds": float(frame[TIME].max()),
        "unique_time_values": int(frame[TIME].nunique()),
        "amount_min": float(frame["Amount"].min()),
        "amount_median": float(frame["Amount"].median()),
        "amount_max": float(frame["Amount"].max()),
    }


def temporal_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, pd.DataFrame]:
    """Split on whole elapsed-time groups, preserving strict chronology."""
    validate_transactions(frame)
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must leave a test window")

    time_counts = frame.groupby(TIME, sort=True, observed=True).size()
    cumulative = time_counts.cumsum().to_numpy()
    time_values = time_counts.index.to_numpy()
    train_position = int(np.argmin(np.abs(cumulative - len(frame) * train_fraction)))
    validation_target = len(frame) * (train_fraction + validation_fraction)
    eligible = np.arange(train_position + 1, len(cumulative))
    if len(eligible) == 0:
        raise ValueError("Not enough distinct time values for three chronological splits")
    validation_position = int(eligible[np.argmin(np.abs(cumulative[eligible] - validation_target))])
    train_end = time_values[train_position]
    validation_end = time_values[validation_position]

    splits = {
        "train": frame.loc[frame[TIME] <= train_end].copy(),
        "validation": frame.loc[(frame[TIME] > train_end) & (frame[TIME] <= validation_end)].copy(),
        "test": frame.loc[frame[TIME] > validation_end].copy(),
    }
    for name, part in splits.items():
        if part.empty:
            raise ValueError(f"Chronological {name} split must not be empty")

    if not (
        splits["train"][TIME].max() < splits["validation"][TIME].min()
        and splits["validation"][TIME].max() < splits["test"][TIME].min()
    ):
        raise AssertionError("Time values crossed a partition boundary")
    return splits


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return transaction attributes while excluding time and the label."""
    return [column for column in frame.columns if column not in {TIME, TARGET}]
