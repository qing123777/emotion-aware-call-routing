"""Build the CREMA-D manifest: scan audio, derive labels, split by speaker.

Getting the data first
----------------------
CREMA-D is free and needs no licence request. Either:

  git clone https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/CREMA-D

or, on Kaggle, add the "CREMA-D" dataset to your notebook and point ``--audio-dir``
at ``/kaggle/input/cremad/AudioWAV``.

Only the ``AudioWAV/`` folder is needed; the video files are not used.

Usage
-----
    python scripts/prepare_crema_d.py --audio-dir data/raw/CREMA-D/AudioWAV
    python scripts/prepare_crema_d.py --audio-dir ... --seed 7 --out data/processed/fold7.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    assert_no_speaker_leakage,
    assign_splits,
    scan_audio_dir,
    split_actors,
    summarise,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="path to CREMA-D AudioWAV/")
    parser.add_argument("--out", default="data/processed/crema_d_manifest.csv")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337, help="controls the actor split only")
    args = parser.parse_args()

    clips = scan_audio_dir(args.audio_dir)
    if not clips:
        print(f"error: no .wav files found under {args.audio_dir}", file=sys.stderr)
        return 1
    print(f"found {len(clips)} clips from {len({c.actor_id for c in clips})} actors")

    splits = split_actors(
        (c.actor_id for c in clips),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    buckets = assign_splits(clips, splits)
    assert_no_speaker_leakage(buckets)

    print(summarise(buckets))
    out_path = write_manifest(buckets, args.out)
    print(f"\nwrote {out_path}")
    print("splits are speaker-independent (verified); seed =", args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
