from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.request import Request, urlopen

SOURCE_URL = "https://data.openml.org/datasets/0000/1597/dataset_1597.pq"
EXPECTED_BYTES = 73_231_184
EXPECTED_SHA256 = "b7efcb35a428bbe22347a05d2437d9177bab07ce61e51214a17bec584ad9496d"
OUTPUT = Path("data/raw/creditcard.parquet")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if OUTPUT.stat().st_size == EXPECTED_BYTES and digest(OUTPUT) == EXPECTED_SHA256:
            print(f"Already verified {OUTPUT}: SHA-256 {EXPECTED_SHA256}")
            return
        raise SystemExit("Existing output differs from the pinned snapshot; refusing to overwrite")
    partial = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = Request(SOURCE_URL, headers=headers)
    with urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise SystemExit("OpenML returned an unexpected byte range")
        mode = "ab" if append else "wb"
        with partial.open(mode) as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
    if partial.stat().st_size != EXPECTED_BYTES or digest(partial) != EXPECTED_SHA256:
        raise SystemExit("Downloaded source does not match the pinned size/SHA-256")
    os.replace(partial, OUTPUT)
    print(f"Verified {OUTPUT}: {EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
