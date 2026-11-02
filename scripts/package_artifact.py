from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np

from fraud_monitoring.io import atomic_json, sha256_file
from fraud_monitoring.models import fraud_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the locked v1 fraud-ranking artifact")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--locked", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=Path("data/source_manifest.json"))
    parser.add_argument("--data-audit", type=Path, default=Path("reports/data_audit.json"))
    parser.add_argument("--protocol", type=Path, default=Path("docs/EXPERIMENT_PROTOCOL.md"))
    parser.add_argument("--lockfile", type=Path, default=Path("requirements.lock"))
    args = parser.parse_args()

    artifact_dir = args.artifact_dir
    manifest_path = artifact_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit("Artifact manifest already exists; refusing to replace a release")
    selection_path = artifact_dir / "selection_manifest.json"
    model_path = artifact_dir / "model.joblib"
    state_path = args.locked / "evaluation_state.json"
    metrics_path = args.locked / "locked_metrics.json"
    monitoring_path = args.locked / "monitoring_windows.json"
    required = [
        selection_path,
        model_path,
        state_path,
        metrics_path,
        monitoring_path,
        args.source_manifest,
        args.data_audit,
        args.protocol,
        args.lockfile,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Release inputs are missing: {missing}")

    selection = json.loads(selection_path.read_text())
    state = json.loads(state_path.read_text())
    source = json.loads(args.source_manifest.read_text())
    metrics = json.loads(metrics_path.read_text())
    if state.get("status") != "completed":
        raise SystemExit("Locked test evidence is not complete")
    if state.get("locked_metrics_sha256") != sha256_file(metrics_path):
        raise SystemExit("Locked metrics hash does not match the one-shot state record")
    if state.get("monitoring_sha256") != sha256_file(monitoring_path):
        raise SystemExit("Monitoring report hash does not match the one-shot state record")
    if selection.get("model_sha256") != sha256_file(model_path):
        raise SystemExit("Selected model hash does not match the selection manifest")
    if selection.get("source_sha256") != source.get("sha256"):
        raise SystemExit("Selected model and source manifest hashes do not match")
    if selection.get("protocol_sha256") != sha256_file(args.protocol):
        raise SystemExit("Experiment protocol changed after model selection")
    if metrics.get("source_sha256") != source.get("sha256"):
        raise SystemExit("Locked metrics and source manifest hashes do not match")

    reports_dir = artifact_dir / "reports"
    lineage_dir = artifact_dir / "lineage"
    reports_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, reports_dir / "locked_metrics.json")
    shutil.copy2(monitoring_path, reports_dir / "monitoring_windows.json")
    shutil.copy2(args.data_audit, reports_dir / "data_audit.json")
    shutil.copy2(state_path, lineage_dir / "evaluation_state.json")
    shutil.copy2(args.source_manifest, lineage_dir / "source_manifest.json")

    feature_columns = selection["feature_columns"]
    model = joblib.load(model_path)
    smoke_features = []
    baseline = {name: 0.0 for name in feature_columns}
    baseline["Amount"] = 1.0
    smoke_features.append(("zero_components", baseline))
    second = dict(baseline, Amount=1250.0, V14=-8.0, V17=4.5)
    smoke_features.append(("high_amount_tail", second))
    third = dict(baseline, Amount=40.0, V10=3.0, V4=-2.0, V27=0.2)
    smoke_features.append(("mixed_components", third))
    queries = []
    for query_id, features in smoke_features:
        vector = np.asarray([[features[name] for name in feature_columns]], dtype=np.float64)
        queries.append(
            {
                "query_id": query_id,
                "features": features,
                "expected_score": float(fraud_scores(model, vector)[0]),
            }
        )
    smoke_path = artifact_dir / "smoke_queries.json"
    atomic_json(smoke_path, {"queries": queries})

    file_paths = [
        model_path,
        selection_path,
        smoke_path,
        reports_dir / "locked_metrics.json",
        reports_dir / "monitoring_windows.json",
        reports_dir / "data_audit.json",
        lineage_dir / "evaluation_state.json",
        lineage_dir / "source_manifest.json",
    ]
    manifest = {
        "status": "locked_release",
        "artifact_version": "v1",
        "selected_model": selection["selected_model"],
        "selection_metric": selection["selection_metric"],
        "repository_commit": selection["repository_commit"],
        "feature_columns": feature_columns,
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "selection_manifest_file": selection_path.name,
        "selection_manifest_sha256": sha256_file(selection_path),
        "smoke_queries_file": smoke_path.name,
        "smoke_queries_sha256": sha256_file(smoke_path),
        "source_sha256": source["sha256"],
        "source_manifest_sha256": sha256_file(lineage_dir / "source_manifest.json"),
        "data_audit_sha256": sha256_file(reports_dir / "data_audit.json"),
        "locked_metrics_sha256": sha256_file(reports_dir / "locked_metrics.json"),
        "monitoring_sha256": sha256_file(reports_dir / "monitoring_windows.json"),
        "protocol_sha256": sha256_file(args.protocol),
        "requirements_lock_sha256": sha256_file(args.lockfile),
        "evaluation_state": json.loads((lineage_dir / "evaluation_state.json").read_text()),
        "file_sha256": {
            str(path.relative_to(artifact_dir)): sha256_file(path) for path in file_paths
        },
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "artifact_version": "v1",
                "selected_model": selection["selected_model"],
                "file_count": len(file_paths),
                "locked_metrics_sha256": manifest["locked_metrics_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
