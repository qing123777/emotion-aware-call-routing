"""Open every audio file once and report anything wrong with it.

The manifest is built from *filenames* alone -- no code opens the audio until
training starts. A corrupt, empty or silent clip is therefore discovered 45
minutes into a Colab run rather than here, on CPU, in a couple of minutes.

What comes out is also reportable: "all 7,442 clips verified readable, no
silent files, median duration 2.5 s" is a measurement, not an assumption.

    python scripts/audit_dataset.py --audio-dir data/raw/CREMA-D/AudioWAV
    python scripts/audit_dataset.py --audio-dir <dir> --manifest <csv> --out audit.json

Exit code is 1 when a clip failed to load or a filename failed to parse, since
those block training. Silence and duplicates are reported as warnings: they are
worth knowing about but do not necessarily mean the run should stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.audio import TARGET_SR, load_audio  # noqa: E402
from src.data.crema_d import CremaDParseError, parse_filename, read_manifest  # noqa: E402

# A clip whose loudest sample sits below this counts as silence. 1e-4 is roughly
# -80 dBFS -- far quieter than any real recording, so ordinary soft speech is
# not flagged, but an all-zero or near-dead file is.
SILENCE_PEAK = 1e-4


def inspect_clip(path: Path, target_sr: int = TARGET_SR) -> Dict[str, Any]:
    """Load one file and measure it.

    Returns a dict carrying ``error`` when the file could not be decoded at all;
    in that case no other field is present, so callers must check ``error``
    first. Any backend exception counts as a finding rather than a crash -- the
    whole point is to survey the corpus, not to stop at the first bad file.
    """
    try:
        waveform = load_audio(path, target_sr=target_sr)
    except Exception as exc:  # noqa: BLE001 - any failure here is a result
        return {"error": f"{type(exc).__name__}: {exc}"}

    if waveform.size == 0:
        return {"samples": 0, "seconds": 0.0, "peak": 0.0, "audio_hash": None}

    return {
        "samples": int(waveform.size),
        "seconds": waveform.size / float(target_sr),
        "peak": float(np.max(np.abs(waveform))),
        # Hash the decoded audio rather than the raw file bytes: two files that
        # differ only in header metadata still contain identical audio, and that
        # is the duplication worth catching.
        "audio_hash": hashlib.md5(waveform.tobytes()).hexdigest(),
    }


def audit_audio_dir(
    audio_dir: Path, target_sr: int = TARGET_SR, progress_every: int = 500
) -> Dict[str, Any]:
    """Parse and decode every ``*.wav`` under ``audio_dir``.

    Filename parsing and audio decoding are checked separately because they fail
    for different reasons: a parse failure means an unexpected naming scheme, a
    decode failure means a damaged file.
    """
    paths = sorted(audio_dir.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"no .wav files under {audio_dir}")

    unparsable: List[str] = []
    unreadable: List[Dict[str, str]] = []
    silent: List[str] = []
    durations: List[float] = []
    by_hash: Dict[str, List[str]] = defaultdict(list)

    for index, path in enumerate(paths, start=1):
        if progress_every and index % progress_every == 0:
            print(f"  ... {index}/{len(paths)}", flush=True)

        try:
            parse_filename(path.name)
        except CremaDParseError as exc:
            unparsable.append(f"{path.name}: {exc}")

        measured = inspect_clip(path, target_sr)
        if "error" in measured:
            unreadable.append({"file": path.name, "error": measured["error"]})
            continue

        durations.append(measured["seconds"])
        if measured["peak"] < SILENCE_PEAK:
            silent.append(path.name)
        if measured["audio_hash"] is not None:
            by_hash[measured["audio_hash"]].append(path.name)

    duplicates = [names for names in by_hash.values() if len(names) > 1]

    return {
        "audio_dir": str(audio_dir),
        "files_found": len(paths),
        "files_readable": len(durations),
        "unparsable_filenames": unparsable,
        "unreadable_files": unreadable,
        "silent_files": silent,
        "duplicate_groups": duplicates,
        "durations": durations,
    }


def duration_summary(durations: List[float], window_seconds: float) -> Dict[str, float]:
    """Duration statistics, plus how much of the training window is padding.

    ``padding_share`` matters because clips shorter than the window are
    right-padded with silence. If most clips are far shorter than the window,
    most of what the model pools over is silence, and a shorter window would
    likely train better.
    """
    if not durations:
        return {}

    array = np.asarray(durations, dtype=np.float64)
    padded = np.clip(window_seconds - array, 0.0, None)

    return {
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
        "shorter_than_window": int((array < window_seconds).sum()),
        "longer_than_window": int((array > window_seconds).sum()),
        "padding_share": float(padded.sum() / (len(array) * window_seconds)),
    }


def label_distribution(manifest_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Per-split counts of emotion and intensity, straight from the manifest.

    Feeds the label-distribution figure the report needs, and makes an
    accidentally lopsided split visible before it quietly skews a result.
    """
    if not read_manifest(manifest_path):
        raise ValueError(f"{manifest_path} contains no rows")

    per_split: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split_name in ("train", "val", "test"):
        split_clips = read_manifest(manifest_path, split=split_name)
        if not split_clips:
            continue
        per_split[split_name] = {
            "emotion": dict(sorted(Counter(c.emotion for c in split_clips).items())),
            "intensity": dict(sorted(Counter(c.intensity for c in split_clips).items())),
            "actors": len({c.actor_id for c in split_clips}),
            "clips": len(split_clips),
        }
    return per_split


def print_report(result: Dict[str, Any]) -> None:
    """Human-readable summary. The JSON carries the full detail."""
    print(f"\naudio dir : {result['audio_dir']}")
    print(f"files found: {result['files_found']}")
    print(f"readable   : {result['files_readable']}")

    problems = 0

    unparsable = result["unparsable_filenames"]
    print(f"\nunparsable filenames: {len(unparsable)}")
    for line in unparsable[:5]:
        print(f"  {line}")
    problems += len(unparsable)

    unreadable = result["unreadable_files"]
    print(f"unreadable files    : {len(unreadable)}")
    for entry in unreadable[:5]:
        print(f"  {entry['file']}: {entry['error']}")
    problems += len(unreadable)

    silent = result["silent_files"]
    print(f"silent files        : {len(silent)}  (peak < {SILENCE_PEAK})")
    for name in silent[:5]:
        print(f"  {name}")

    duplicates = result["duplicate_groups"]
    print(f"duplicate audio     : {len(duplicates)} group(s)")
    for group in duplicates[:3]:
        print(f"  {len(group)} identical: {group[:3]}")

    stats = result.get("duration_seconds")
    if stats:
        print("\nclip duration (seconds)")
        print(f"  min {stats['min']:.2f}   p05 {stats['p05']:.2f}   median {stats['median']:.2f}"
              f"   mean {stats['mean']:.2f}   p95 {stats['p95']:.2f}   max {stats['max']:.2f}")
        print(f"  shorter than the {result['window_seconds']:.1f}s window: "
              f"{stats['shorter_than_window']} of {result['files_readable']}")
        print(f"  share of the window that is padding: {stats['padding_share']:.1%}")

    per_split = result.get("label_distribution")
    if per_split:
        print("\nlabel distribution")
        for split_name, counts in per_split.items():
            emotions = " ".join(f"{k}:{v}" for k, v in counts["emotion"].items())
            print(f"  {split_name:<6} {counts['clips']:>5} clips  "
                  f"{counts['actors']:>3} actors   {emotions}")

    if problems:
        print(f"\nPROBLEMS FOUND — {problems} file(s) cannot be used")
    else:
        print("\nno blocking problems")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="the AudioWAV directory")
    parser.add_argument("--manifest", default=None, help="optional: also report per-split label counts")
    parser.add_argument("--out", default=None, help="optional: write the full findings as JSON")
    parser.add_argument("--window-seconds", type=float, default=4.0,
                        help="training window, for the padding calculation (default: 4.0)")
    parser.add_argument("--sample-rate", type=int, default=TARGET_SR)
    args = parser.parse_args()

    result = audit_audio_dir(Path(args.audio_dir), target_sr=args.sample_rate)

    durations = result.pop("durations")
    result["window_seconds"] = args.window_seconds
    result["duration_seconds"] = duration_summary(durations, args.window_seconds)

    if args.manifest:
        result["label_distribution"] = label_distribution(Path(args.manifest))

    print_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")

    blocking = len(result["unparsable_filenames"]) + len(result["unreadable_files"])
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
