from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import joblib

from fraud_monitoring.data import TARGET, feature_columns, load_transactions, temporal_split
from fraud_monitoring.evaluation import capacity_metrics, ranking_metrics
from fraud_monitoring.io import atomic_json, sha256_file
from fraud_monitoring.models import candidate_models, fit_candidate, fraud_scores

CAPACITIES = (0.001, 0.0025, 0.005)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a model using train/validation only")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/v1"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/EXPERIMENT_PROTOCOL.md"))
    args = parser.parse_args()

    frame = load_transactions(args.input)
    splits = temporal_split(frame)
    columns = feature_columns(frame)
    train = splits["train"]
    validation = splits["validation"]
    if train[TARGET].nunique() != 2 or validation[TARGET].nunique() != 2:
        raise SystemExit("Train and validation partitions must contain both target classes")
    x_train, y_train = train[columns], train[TARGET]
    x_validation, y_validation = validation[columns], validation[TARGET]

    results: dict[str, dict] = {}
    fitted = {}
    for name, candidate in candidate_models().items():
        model = fit_candidate(name, candidate, x_train, y_train)
        scores = fraud_scores(model, x_validation)
        results[name] = {
            **ranking_metrics(y_validation, scores),
            "capacity_metrics": {
                f"{fraction:.4f}": capacity_metrics(y_validation, scores, fraction)
                for fraction in CAPACITIES
            },
        }
        fitted[name] = model

    selected_name = min(results, key=lambda name: (-results[name]["average_precision"], name))
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.joblib"
    temporary_model_path = model_path.with_suffix(".joblib.tmp")
    joblib.dump(fitted[selected_name], temporary_model_path, compress=3)
    temporary_model_path.replace(model_path)

    manifest = {
        "status": "selected_on_validation",
        "selected_model": selected_name,
        "selection_metric": "validation average precision",
        "feature_columns": columns,
        "alert_capacity_fractions": list(CAPACITIES),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_time_max_seconds": float(train["Time"].max()),
        "validation_time_min_seconds": float(validation["Time"].min()),
        "validation_time_max_seconds": float(validation["Time"].max()),
        "source_sha256": sha256_file(args.input),
        "protocol_sha256": sha256_file(args.protocol),
        "repository_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "candidate_results": results,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
    }
    atomic_json(args.output / "selection_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
