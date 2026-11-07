"""Build a clean copy of the project, ready to drag into Google Drive.

Run this instead of deleting folders by hand. The point is not tidiness -- it is
that `data` must be excluded at the top level but `src/data/` must be kept, and a
rule that matches on folder name alone gets that wrong. That mistake produces
`ModuleNotFoundError: No module named 'src.data'` in Colab, which does not look
like an upload problem at all.

    python scripts/make_upload_bundle.py
    python scripts/make_upload_bundle.py --out "D:/uccd3094_upload"
    python scripts/make_upload_bundle.py --zip

Then drag the resulting folder into MyDrive/uccd3094/.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

# Skipped wherever they appear.
EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "node_modules",
    ".idea",
    ".vscode",
    ".mypy_cache",
}

# Skipped ONLY at the project root. This is the important list: "data" belongs
# here and not in EXCLUDE_DIR_NAMES, because src/data/ must survive.
EXCLUDE_TOP_LEVEL = {"data", "artifacts", "checkpoints", "runs", "build", "upload"}

# Datasets and model weights never belong in the copy -- they live in Drive
# separately, and the submission zip is capped at 5MB.
EXCLUDE_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors",
    ".wav", ".mp3", ".mp4", ".flac",
    ".zip", ".tar", ".gz",
    ".pyc", ".pyo",
}

EXCLUDE_NAMES = {"CLAUDE.local.md", "settings.local.json", ".DS_Store", "Thumbs.db"}

# Must exist in the output. Same list the Colab notebook checks on arrival, so a
# failure is caught here rather than after the upload.
REQUIRED = [
    "app.py",
    "src/__init__.py",
    "src/inference.py",
    "src/data/__init__.py",
    "src/data/crema_d.py",
    "src/data/dataset.py",
    "src/data/labels.py",
    "src/data/audio.py",
    "src/models/model.py",
    "src/train/loop.py",
    "src/evaluation/metrics.py",
    "src/utils/seed.py",
    "scripts/train.py",
    "scripts/prepare_crema_d.py",
    "scripts/smoke_test.py",
    "configs/crema_d_baseline.yaml",
    "tests/test_inference.py",
]


def should_skip(path: Path, root: Path) -> bool:
    """Decide whether one file or folder stays out of the bundle."""
    relative = path.relative_to(root)
    parts = relative.parts

    if any(part in EXCLUDE_DIR_NAMES for part in parts):
        return True
    # Top-level only: parts[0] is the entry directly under the project root.
    if parts[0] in EXCLUDE_TOP_LEVEL:
        return True
    if path.is_file():
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            return True
        if path.name in EXCLUDE_NAMES:
            return True
    return False


def collect(root: Path) -> Tuple[List[Path], List[Path]]:
    """Split every file under root into kept and skipped."""
    kept: List[Path] = []
    skipped: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        (skipped if should_skip(path, root) else kept).append(path)
    return kept, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT.parent / f"{REPO_ROOT.name}_upload"),
        help="where to write the clean copy (default: a sibling folder next to the project)",
    )
    parser.add_argument("--zip", action="store_true", help="also produce a .zip beside the folder")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()

    # Copying into a subfolder of the source would recurse into itself.
    if out_dir == REPO_ROOT or REPO_ROOT in out_dir.parents:
        print(f"error: --out must be outside the project, got {out_dir}", file=sys.stderr)
        return 1

    kept, skipped = collect(REPO_ROOT)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    total_bytes = 0
    for source in kept:
        destination = out_dir / source.relative_to(REPO_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        total_bytes += source.stat().st_size

    missing = [name for name in REQUIRED if not (out_dir / name).exists()]
    if missing:
        print("error: the bundle is missing required files:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(f"bundle: {out_dir}")
    print(f"  kept    {len(kept):4d} files  ({total_bytes / 1e6:.2f} MB)")
    print(f"  skipped {len(skipped):4d} files")
    print(f"  verified all {len(REQUIRED)} required files present, including src/data/")

    # Report by reason, not by first path component -- a .pyc under src/ would
    # otherwise print as "src excluded", which reads as though the library was
    # dropped.
    if skipped:
        dropped_dirs = sorted(
            name for name in EXCLUDE_TOP_LEVEL | EXCLUDE_DIR_NAMES if (REPO_ROOT / name).exists()
        )
        caches = sum(1 for p in skipped if any(part in EXCLUDE_DIR_NAMES for part in p.parts))
        heavy = sum(1 for p in skipped if p.suffix.lower() in EXCLUDE_SUFFIXES)
        if dropped_dirs:
            print(f"  excluded folders    : {', '.join(dropped_dirs)}")
        print(f"  excluded cache files: {caches}")
        print(f"  excluded data/model : {heavy}")

    if args.zip:
        archive = shutil.make_archive(str(out_dir), "zip", root_dir=out_dir)
        print(f"  zip: {archive} ({Path(archive).stat().st_size / 1e6:.2f} MB)")

    print()
    print("Next: drag this folder into Google Drive at MyDrive/uccd3094/")
    print("      (replace the previous copy so Colab picks up the changes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
