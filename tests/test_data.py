import numpy as np
import pandas as pd
import pytest

from fraud_monitoring.data import (
    EXPECTED_COLUMNS,
    audit_transactions,
    load_transactions,
    temporal_split,
    validate_transactions,
)


def _frame() -> pd.DataFrame:
    rows = []
    for time in range(30):
        for label in (0, 0, 0, 1):
            row = {column: 0.0 for column in EXPECTED_COLUMNS}
            row["Time"] = float(time)
            row["Amount"] = float(time + label)
            row["Class"] = label
            rows.append(row)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_temporal_split_keeps_time_groups_together_and_ordered() -> None:
    frame = _frame()
    splits = temporal_split(frame)
    assert list(splits) == ["train", "validation", "test"]
    assert splits["train"].Time.max() < splits["validation"].Time.min()
    assert splits["validation"].Time.max() < splits["test"].Time.min()
    assert sum(len(part) for part in splits.values()) == len(frame)
    assert all(part.Class.nunique() == 2 for part in splits.values())


def test_validation_rejects_nonfinite_values() -> None:
    frame = _frame()
    frame.loc[0, "V1"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_transactions(frame)


def test_validation_rejects_time_ordering_errors() -> None:
    frame = _frame()
    frame.loc[1, "Time"] = -1
    with pytest.raises(ValueError, match="monotonically"):
        validate_transactions(frame)


def test_audit_reports_prevalence_and_duplicates() -> None:
    frame = _frame()
    report = audit_transactions(frame)
    assert report["rows"] == 120
    assert report["fraud_count"] == 30
    assert report["fraud_rate"] == 0.25
    assert report["exact_duplicate_rows_beyond_first"] == 60


def test_load_transactions_reads_parquet_and_sorts_time(tmp_path) -> None:
    path = tmp_path / "transactions.parquet"
    _frame().sample(frac=1, random_state=3).to_parquet(path)
    loaded = load_transactions(path)
    assert loaded.Time.is_monotonic_increasing
    assert loaded.Class.dtype == np.int8


def test_load_transactions_rejects_schema_drift(tmp_path) -> None:
    path = tmp_path / "transactions.parquet"
    _frame().drop(columns="V1").to_parquet(path)
    with pytest.raises(ValueError, match="schema"):
        load_transactions(path)
