"""Download the trained demo checkpoint from the project's GitHub Release.

Why a Release and not the repo: the checkpoint is ~114 MB, and GitHub rejects any
single file over 100 MB in a normal push. A Release asset has no such limit and
does not bloat the git history.

Usage:
    python scripts/download_checkpoint.py            # fetch into checkpoints/
    python scripts/download_checkpoint.py --force    # re-download even if present

The SHA-256 check guarantees the file is byte-identical to the one the reported
results came from, so a truncated or swapped download is caught before use.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_NAME = "crema_d_partial_top4_best.pt"
CHECKPOINT_SHA256 = "ba2ccd328e9187f1155d25653f791806f2cfb2adfb341c8823bd8c5c5c87e2c4"
RELEASE_URL = f"https://github.com/qing123777/emotion-aware-call-routing/releases/download/v1.0/{CHECKPOINT_NAME}"

CHUNK_BYTES = 1 << 20


def sha256_of(path: Path) -> str:
    """Hash in chunks so a 100+ MB file never sits in memory at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid(path: Path, expected_sha256: str = CHECKPOINT_SHA256) -> bool:
    return path.is_file() and sha256_of(path) == expected_sha256


def download(url: str, destination: Path) -> None:
    """Stream to a .part file and rename at the end, so an interrupted
    download never leaves a half-written file under the real name."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=RELEASE_URL, help="override the download URL")
    parser.add_argument("--out", default=str(REPO_ROOT / "checkpoints" / CHECKPOINT_NAME))
    parser.add_argument("--force", action="store_true", help="download even if a valid file exists")
    args = parser.parse_args()

    destination = Path(args.out)
    if not args.force and is_valid(destination):
        print(f"Checkpoint already present and verified: {destination}")
        return 0

    print(f"Downloading {args.url}")
    download(args.url, destination)
    if not is_valid(destination):
        print("ERROR: downloaded file failed the SHA-256 check; it was not the expected checkpoint.", file=sys.stderr)
        return 1
    print(f"Saved and verified: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
