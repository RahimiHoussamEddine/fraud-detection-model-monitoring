import numpy as np

from fraud_monitoring.io import atomic_json, sha256_file
from fraud_monitoring.models import candidate_models, fit_candidate, fraud_scores


def test_atomic_json_writes_readable_payload(tmp_path) -> None:
    path = tmp_path / "nested" / "payload.json"
    atomic_json(path, {"value": 3})
    assert path.read_text() == '{\n  "value": 3\n}\n'
    assert len(sha256_file(path)) == 64


def test_candidate_models_fit_and_return_fraud_scores() -> None:
    x = np.asarray([[-2.0], [-1.0], [0.0], [0.5], [1.0], [2.0]])
    y = np.asarray([0, 0, 0, 1, 1, 1])
    candidates = candidate_models()
    logistic = fit_candidate("balanced_logistic", candidates["balanced_logistic"], x, y)
    histogram = fit_candidate(
        "weighted_hist_gradient_boosting",
        candidates["weighted_hist_gradient_boosting"],
        x,
        y,
    )
    assert fraud_scores(logistic, x).shape == (6,)
    assert fraud_scores(histogram, x).shape == (6,)
