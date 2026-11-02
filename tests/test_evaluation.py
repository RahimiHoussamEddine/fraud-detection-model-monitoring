import numpy as np
import pytest

from fraud_monitoring.evaluation import capacity_metrics, ranking_metrics


def test_ranking_metrics_report_prevalence_and_ranking_quality() -> None:
    result = ranking_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert result["average_precision"] == 1.0
    assert result["roc_auc"] == 1.0
    assert result["prevalence"] == 0.5


def test_capacity_metrics_respect_alert_budget_and_ties_are_repeatable() -> None:
    labels = [1, 0, 1, 0, 0]
    scores = [0.5, 0.5, 0.2, 0.1, 0.1]
    first = capacity_metrics(labels, scores, 0.4, tie_seed=7)
    second = capacity_metrics(labels, scores, 0.4, tie_seed=7)
    assert first == second
    assert first["alert_count"] == 2
    assert 0 <= first["recall"] <= 1


@pytest.mark.parametrize("labels,scores", [([0, 0], [0.1, 0.2]), ([0, 1], [np.nan, 0.2])])
def test_ranking_metrics_reject_invalid_inputs(labels, scores) -> None:
    with pytest.raises(ValueError):
        ranking_metrics(labels, scores)


def test_capacity_metrics_reject_zero_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        capacity_metrics([0, 1], [0.1, 0.9], 0)
