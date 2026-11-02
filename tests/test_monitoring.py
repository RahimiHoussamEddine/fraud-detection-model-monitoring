import numpy as np
import pandas as pd
import pytest

from fraud_monitoring.monitoring import (
    feature_drift,
    population_stability_index,
    time_window_monitoring,
)


def test_psi_is_near_zero_for_identical_distributions() -> None:
    sample = np.linspace(0, 1, 1000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-12)


def test_psi_detects_shift() -> None:
    reference = np.linspace(0, 1, 1000)
    shifted = np.linspace(2, 3, 1000)
    assert population_stability_index(reference, shifted) > 0.5


def test_feature_drift_requires_matching_names() -> None:
    with pytest.raises(ValueError, match="feature names"):
        feature_drift({"x": [1, 2]}, {"y": [1, 2]})


def test_time_window_monitoring_separates_unlabeled_drift_from_metrics() -> None:
    reference = pd.DataFrame(
        {"Time": np.arange(8), "V1": np.linspace(0, 1, 8), "Amount": np.arange(8)}
    )
    current = pd.DataFrame(
        {
            "Time": np.arange(20_000, 20_008),
            "V1": np.linspace(2, 3, 8),
            "Amount": np.arange(8),
            "Class": [0, 0, 1, 0, 0, 0, 0, 0],
        }
    )
    report = time_window_monitoring(
        reference,
        current,
        reference_scores=np.linspace(0, 1, 8),
        current_scores=np.linspace(0.1, 0.9, 8),
        feature_names=["V1", "Amount"],
        window_seconds=4,
    )
    assert len(report) == 2
    assert report[0]["label_metrics"] is not None
    assert report[1]["label_metrics"] is None
    assert report[0]["drift_psi"]["V1"] > 0.1
