from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_monitoring.io import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".tmp/release-smoke"
FEATURES = [*(f"V{i}" for i in range(1, 29)), "Amount"]


def run(*arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    artifact = WORK / "artifacts/v1"
    locked = WORK / "reports/locked/v1"
    data_path = WORK / "synthetic.parquet"
    source_manifest = WORK / "source_manifest.json"
    data_audit = WORK / "data_audit.json"
    WORK.mkdir(parents=True)

    rng = np.random.default_rng(2026)
    row_count = 2400
    labels = (np.arange(row_count) % 17 == 0).astype(np.int8)
    values = rng.normal(size=(row_count, len(FEATURES)))
    values[:, 0] += labels * 3.0
    values[:, 3] -= labels * 2.5
    values[:, -1] = rng.lognormal(mean=2.0, sigma=1.0, size=row_count)
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "Time", np.linspace(0.0, 172_792.0, row_count))
    frame["Class"] = labels
    frame.to_parquet(data_path, index=False)

    selection_output = run(
        "scripts/select_model.py",
        "--input",
        str(data_path.relative_to(ROOT)),
        "--output",
        str(artifact.relative_to(ROOT)),
    )
    selection = json.loads(selection_output)
    run(
        "scripts/run_locked_evaluation.py",
        "--input",
        str(data_path.relative_to(ROOT)),
        "--artifact-dir",
        str(artifact.relative_to(ROOT)),
        "--output",
        str(locked.relative_to(ROOT)),
    )
    atomic_json(source_manifest, {"sha256": sha256_file(data_path), "synthetic": True})
    atomic_json(data_audit, {"rows": row_count, "synthetic": True})
    run(
        "scripts/package_artifact.py",
        "--artifact-dir",
        str(artifact.relative_to(ROOT)),
        "--locked",
        str(locked.relative_to(ROOT)),
        "--source-manifest",
        str(source_manifest.relative_to(ROOT)),
        "--data-audit",
        str(data_audit.relative_to(ROOT)),
    )
    verification = json.loads(
        run("scripts/verify_artifact.py", "--artifact-dir", str(artifact.relative_to(ROOT)))
    )
    if verification["status"] != "ok" or len(verification["smoke_checks"]) != 3:
        raise SystemExit("Synthetic release pipeline verification failed")
    print(
        json.dumps(
            {
                "status": "ok",
                "synthetic_rows": row_count,
                "selected_model": selection["selected_model"],
                "artifact_smoke_checks": len(verification["smoke_checks"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
