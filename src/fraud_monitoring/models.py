"""Fixed candidate models used by the development selection step."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


def candidate_models() -> dict[str, Any]:
    return {
        "balanced_logistic": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=2026,
                    ),
                ),
            ]
        ),
        "weighted_hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            l2_regularization=1.0,
            max_iter=120,
            max_leaf_nodes=31,
            early_stopping=False,
            random_state=2026,
        ),
    }


def fit_candidate(name: str, model: Any, features: Any, labels: Any) -> Any:
    if name == "weighted_hist_gradient_boosting":
        sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
        model.fit(features, labels, sample_weight=sample_weight)
    else:
        model.fit(features, labels)
    return model


def fraud_scores(model: Any, features: Any):
    """Return the score for class 1 without treating it as calibrated risk."""
    classes = list(
        model.classes_ if hasattr(model, "classes_") else model.named_steps["model"].classes_
    )
    positive_position = classes.index(1)
    return model.predict_proba(features)[:, positive_position]
