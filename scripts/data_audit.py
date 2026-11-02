from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from fraud_monitoring.data import audit_transactions, load_transactions
from fraud_monitoring.io import atomic_json, sha256_file

DATASET_ID = 1597
SOURCE_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
RECORD_URL = "https://www.openml.org/d/1597"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the pinned OpenML creditcard snapshot")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/source_manifest.json"))
    args = parser.parse_args()

    frame = load_transactions(args.input)
    report = audit_transactions(frame)
    if report["rows"] != EXPECTED_ROWS or report["fraud_count"] != EXPECTED_FRAUDS:
        raise SystemExit(
            f"OpenML source contract mismatch: rows={report['rows']}, "
            f"fraud_count={report['fraud_count']}"
        )

    digest = sha256_file(args.input)
    manifest = {
        "dataset_id": DATASET_ID,
        "record_url": RECORD_URL,
        "download_url": SOURCE_URL,
        "format": "OpenML Parquet mirror",
        "retrieved_on": date.today().isoformat(),
        "file_name": args.input.name,
        "file_bytes": args.input.stat().st_size,
        "sha256": digest,
        "parquet_http_etag": "1593844f40edbdabaf5bddec4649e1c4",
        "catalog_original_arff_md5": "178bcf9bb1f31a3dfe12d0e577884add",
        "audit": report,
    }
    if args.manifest.exists():
        previous = json.loads(args.manifest.read_text())
        if previous.get("sha256") != digest:
            raise SystemExit("Existing source manifest has a different digest; refusing to replace")
    atomic_json(args.manifest, manifest)
    atomic_json(args.output, report)
    print(json.dumps({"source_sha256": digest, "audit": report}, sort_keys=True))


if __name__ == "__main__":
    main()
