import asyncio

import httpx
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fraud_monitoring.api import app
from fraud_monitoring.io import atomic_json, sha256_file
from fraud_monitoring.models import fraud_scores
from fraud_monitoring.service import ArtifactValidationError, FraudScoringService

FEATURES = [*(f"V{i}" for i in range(1, 29)), "Amount"]


def _artifact(directory):
    x = np.zeros((8, len(FEATURES)), dtype=np.float64)
    x[:, 0] = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]
    x[:, -1] = [2, 3, 4, 5, 6, 7, 8, 9]
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    model = LogisticRegression(random_state=1).fit(pd.DataFrame(x, columns=FEATURES), y)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)
    features = {name: 0.0 for name in FEATURES}
    features["Amount"] = 10.0
    expected = float(fraud_scores(model, pd.DataFrame([features], columns=FEATURES))[0])
    smoke_path = directory / "smoke_queries.json"
    atomic_json(
        smoke_path,
        {
            "queries": [
                {"query_id": "synthetic-1", "features": features, "expected_score": expected}
            ]
        },
    )
    atomic_json(
        directory / "manifest.json",
        {
            "status": "locked_release",
            "artifact_version": "v1",
            "selected_model": "synthetic",
            "model_file": "model.joblib",
            "model_sha256": sha256_file(model_path),
            "feature_columns": FEATURES,
            "smoke_queries_file": "smoke_queries.json",
            "smoke_queries_sha256": sha256_file(smoke_path),
        },
    )
    return features


def test_service_scores_and_verifies_synthetic_smoke(tmp_path) -> None:
    features = _artifact(tmp_path)
    service = FraudScoringService(tmp_path)
    assert 0 <= service.score(features) <= 1
    assert service.smoke_check()[0]["query_id"] == "synthetic-1"


def test_service_rejects_wrong_schema_nonfinite_values_and_negative_amount(tmp_path) -> None:
    features = _artifact(tmp_path)
    service = FraudScoringService(tmp_path)
    with pytest.raises(ValueError, match="schema mismatch"):
        service.score({"Amount": 1.0})
    invalid = dict(features, V1=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        service.score(invalid)
    invalid = dict(features, Amount=-1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        service.score(invalid)


def test_service_rejects_tampered_model(tmp_path) -> None:
    _artifact(tmp_path)
    (tmp_path / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ArtifactValidationError, match="digest"):
        FraudScoringService(tmp_path)


def test_api_health_and_scoring_use_the_validated_artifact(tmp_path, monkeypatch) -> None:
    features = _artifact(tmp_path)
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tmp_path))

    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/health?deep=true")
            scored = await client.post("/v1/score", json={"features": features})
            invalid = await client.post("/v1/score", json={"features": {"Amount": 2.0}})
            return health, scored, invalid

    health, scored, invalid = asyncio.run(exercise())
    assert health.status_code == 200
    assert health.json()["smoke_checks"][0]["query_id"] == "synthetic-1"
    assert scored.status_code == 200
    assert 0 <= scored.json()["ranking_score"] <= 1
    assert invalid.status_code == 422
