"""Validated local scoring facade for the immutable model artifact."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from fraud_monitoring.io import sha256_file
from fraud_monitoring.models import fraud_scores


class ArtifactValidationError(RuntimeError):
    """The requested model artifact is absent or inconsistent."""


class FraudScoringService:
    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactValidationError("Artifact manifest is missing")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("status") != "locked_release":
            raise ArtifactValidationError("Artifact is not a locked release")
        self.model_path = self.artifact_dir / self.manifest["model_file"]
        if not self.model_path.is_file():
            raise ArtifactValidationError("Model file is missing")
        if sha256_file(self.model_path) != self.manifest.get("model_sha256"):
            raise ArtifactValidationError("Model digest does not match the artifact manifest")
        self.model = joblib.load(self.model_path)
        self.feature_columns = list(self.manifest["feature_columns"])
        if (
            not self.feature_columns
            or "Time" in self.feature_columns
            or "Class" in self.feature_columns
        ):
            raise ArtifactValidationError("Artifact feature schema is invalid")
        expected_features = [*(f"V{i}" for i in range(1, 29)), "Amount"]
        if self.feature_columns != expected_features:
            raise ArtifactValidationError(
                "Artifact feature schema differs from the source contract"
            )
        if getattr(self.model, "n_features_in_", None) != len(self.feature_columns):
            raise ArtifactValidationError("Model input dimension differs from the feature schema")
        model_classes = getattr(self.model, "classes_", None)
        if model_classes is None or set(model_classes) != {0, 1}:
            raise ArtifactValidationError("Model must expose both binary target classes")
        self.smoke_path = self.artifact_dir / self.manifest["smoke_queries_file"]
        if not self.smoke_path.is_file():
            raise ArtifactValidationError("Artifact smoke queries are missing")
        if sha256_file(self.smoke_path) != self.manifest.get("smoke_queries_sha256"):
            raise ArtifactValidationError("Smoke query digest does not match the artifact manifest")

    def score(self, features: dict[str, float]) -> float:
        if set(features) != set(self.feature_columns):
            missing = sorted(set(self.feature_columns) - set(features))
            unexpected = sorted(set(features) - set(self.feature_columns))
            raise ValueError(f"Feature schema mismatch; missing={missing}, unexpected={unexpected}")
        values = np.asarray([features[name] for name in self.feature_columns], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("All input features must be finite numbers")
        if features["Amount"] < 0:
            raise ValueError("Amount must be nonnegative")
        return float(fraud_scores(self.model, values.reshape(1, -1))[0])

    def smoke_check(self) -> list[dict]:
        payload = json.loads(self.smoke_path.read_text())
        results = []
        for row in payload["queries"]:
            score = self.score(row["features"])
            expected = float(row["expected_score"])
            if not np.isclose(score, expected, rtol=0, atol=1e-8):
                raise ArtifactValidationError("Smoke query output differs from the release")
            results.append({"query_id": row["query_id"], "score": score})
        if not results:
            raise ArtifactValidationError("Artifact contains no smoke queries")
        return results
