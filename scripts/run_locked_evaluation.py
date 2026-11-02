from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from fraud_monitoring.data import TARGET, feature_columns, load_transactions, temporal_split
from fraud_monitoring.evaluation import capacity_metrics, ranking_metrics
from fraud_monitoring.io import atomic_json, sha256_file
from fraud_monitoring.models import fraud_scores
from fraud_monitoring.monitoring import time_window_monitoring

CAPACITIES = (0.001, 0.0025, 0.005)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected model once on the locked test"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/locked/v1"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/EXPERIMENT_PROTOCOL.md"))
    args = parser.parse_args()

    state_path = args.output / "evaluation_state.json"
    if state_path.exists():
        raise SystemExit("Locked evaluation state already exists; refusing to rerun")
    selection_path = args.artifact_dir / "selection_manifest.json"
    model_path = args.artifact_dir / "model.joblib"
    if not selection_path.is_file() or not model_path.is_file():
        raise SystemExit("Complete model selection before opening the locked test")

    selection = json.loads(selection_path.read_text())
    source_digest = sha256_file(args.input)
    if selection.get("source_sha256") != source_digest:
        raise SystemExit("Selected model and locked test source hashes do not match")
    protocol_digest = sha256_file(args.protocol)
    if selection.get("protocol_sha256") != protocol_digest:
        raise SystemExit("Experiment protocol changed after model selection")
    model_digest = sha256_file(model_path)
    if selection.get("model_sha256") != model_digest:
        raise SystemExit("Selected model digest differs from its selection manifest")

    started = {
        "status": "started",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": source_digest,
        "selection_manifest_sha256": sha256_file(selection_path),
        "protocol_sha256": protocol_digest,
        "model_sha256": model_digest,
        "selected_model": selection["selected_model"],
        "test_evaluation_attempt": 1,
    }
    atomic_json(state_path, started)

    frame = load_transactions(args.input)
    splits = temporal_split(frame)
    train = splits["train"]
    test = splits["test"]
    columns = feature_columns(frame)
    if columns != selection["feature_columns"]:
        raise SystemExit("Feature schema differs from the model selection manifest")
    model = joblib.load(model_path)
    test_scores = fraud_scores(model, test[columns])
    train_scores = fraud_scores(model, train[columns])
    test_metrics = {
        **ranking_metrics(test[TARGET], test_scores),
        "capacity_metrics": {
            f"{fraction:.4f}": capacity_metrics(test[TARGET], test_scores, fraction)
            for fraction in CAPACITIES
        },
        "test_rows": int(len(test)),
        "test_time_min_seconds": float(test["Time"].min()),
        "test_time_max_seconds": float(test["Time"].max()),
    }
    unique_mask = ~test.duplicated(keep="first")
    deduplicated_test = test.loc[unique_mask]
    deduplicated_scores = test_scores[unique_mask.to_numpy()]
    duplicate_sensitivity = {
        **ranking_metrics(deduplicated_test[TARGET], deduplicated_scores),
        "capacity_metrics": {
            f"{fraction:.4f}": capacity_metrics(
                deduplicated_test[TARGET], deduplicated_scores, fraction
            )
            for fraction in CAPACITIES
        },
        "test_rows": int(len(deduplicated_test)),
        "collapsed_exact_rows": int(len(test) - len(deduplicated_test)),
        "fraud_count": int(deduplicated_test[TARGET].sum()),
    }
    monitoring = time_window_monitoring(
        train,
        test,
        train_scores,
        test_scores,
        columns,
        window_seconds=4 * 60 * 60,
        alert_fraction=0.0025,
    )
    metrics_path = args.output / "locked_metrics.json"
    monitoring_path = args.output / "monitoring_windows.json"
    atomic_json(
        metrics_path,
        {
            "evaluation_status": "locked_test",
            "selected_model": selection["selected_model"],
            "selection_metric": selection["selection_metric"],
            "source_sha256": source_digest,
            "model_sha256": model_digest,
            "metrics": test_metrics,
            "exact_duplicate_sensitivity": duplicate_sensitivity,
        },
    )
    atomic_json(
        monitoring_path,
        {
            "reference_period": "train",
            "monitored_period": "locked_test",
            "window_seconds": 4 * 60 * 60,
            "psi_thresholds": {"watch": 0.10, "alert": 0.25},
            "threshold_note": "Heuristic review triggers, not statistical guarantees.",
            "windows": monitoring,
        },
    )
    atomic_json(
        state_path,
        {
            **started,
            "status": "completed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "locked_metrics_sha256": sha256_file(metrics_path),
            "monitoring_sha256": sha256_file(monitoring_path),
        },
    )
    print(
        json.dumps({"metrics": test_metrics, "monitoring_windows": len(monitoring)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
