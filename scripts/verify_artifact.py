from __future__ import annotations

import argparse
import json
from pathlib import Path

from fraud_monitoring.io import sha256_file
from fraud_monitoring.service import FraudScoringService


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the packaged locked model artifact")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "locked_release" or manifest.get("artifact_version") != "v1":
        raise SystemExit("Artifact status or version is not a locked v1 release")

    for relative, expected in manifest["file_sha256"].items():
        path = args.artifact_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"Artifact file is missing or has a hash mismatch: {relative}")
    for key, relative in (
        ("selection_manifest_sha256", manifest["selection_manifest_file"]),
        ("smoke_queries_sha256", manifest["smoke_queries_file"]),
    ):
        if sha256_file(args.artifact_dir / relative) != manifest[key]:
            raise SystemExit(f"Artifact {relative} hash does not match the manifest")

    state_path = args.artifact_dir / "lineage/evaluation_state.json"
    state = json.loads(state_path.read_text())
    if state.get("status") != "completed":
        raise SystemExit("Packaged evaluation state is incomplete")
    if state.get("locked_metrics_sha256") != manifest.get("locked_metrics_sha256"):
        raise SystemExit("Packaged locked metrics differ from the one-shot state record")
    if state.get("monitoring_sha256") != manifest.get("monitoring_sha256"):
        raise SystemExit("Packaged monitoring report differs from the one-shot state record")

    service = FraudScoringService(args.artifact_dir)
    smoke_results = service.smoke_check()
    print(
        json.dumps(
            {
                "status": "ok",
                "artifact_version": manifest["artifact_version"],
                "selected_model": manifest["selected_model"],
                "source_sha256": manifest["source_sha256"],
                "locked_metrics_sha256": manifest["locked_metrics_sha256"],
                "smoke_checks": smoke_results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
