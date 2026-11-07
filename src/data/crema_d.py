"""CREMA-D dataset: filename parsing, manifest building, speaker-independent splits.

CREMA-D contains 7,442 clips from 91 actors, each acting one of 12 fixed
sentences under one of 6 emotions at one of 4 intensity levels [1]. Files live
in ``AudioWAV/`` and are named::

    1001_DFA_ANG_XX.wav
    ^^^^ ^^^ ^^^ ^^
    |    |   |   +-- intensity: LO / MD / HI / XX (XX = unspecified)
    |    |   +------ emotion:   ANG DIS FEA HAP NEU SAD
    |    +---------- sentence:  one of 12 three-letter codes
    +--------------- actor id:  1001..1091

Because the same 12 sentences are spoken by every actor, a random split would
leak both speaker identity and lexical content across train and test. Splits
here are therefore **by actor**.

[1] H. Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset,"
    IEEE Transactions on Affective Computing, vol. 5, no. 4, pp. 377-390, 2014.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .labels import EMOTION_TO_INDEX, INTENSITY_TO_SCALE, emotion_to_arousal_valence, escalation_target

SENTENCE_CODES = (
    "IEO", "TIE", "IOM", "IWW", "TAI", "MTI",
    "IWL", "ITH", "DFA", "ITS", "TSI", "WSI",
)


@dataclass(frozen=True)
class Clip:
    """One CREMA-D utterance and every label we derive from it."""

    path: str
    actor_id: int
    sentence: str
    emotion: str
    intensity: str
    emotion_index: int
    intensity_scale: float
    arousal: float
    valence: float
    escalation: float


class CremaDParseError(ValueError):
    """Raised when a filename does not match the CREMA-D naming scheme."""


def parse_filename(filename: str) -> Dict[str, object]:
    """Parse a CREMA-D filename into its four fields.

    Accepts a bare name or a full path, with or without the ``.wav`` suffix.

    >>> parse_filename("1001_DFA_ANG_XX.wav") == {
    ...     "actor_id": 1001, "sentence": "DFA", "emotion": "ANG", "intensity": "XX"}
    True
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise CremaDParseError(f"expected 4 underscore-separated fields, got {len(parts)}: {filename!r}")

    actor_raw, sentence, emotion, intensity = parts

    # One file in the real corpus is tagged "X" rather than "XX"
    # (1040_ITH_SAD_X.wav). Treating it as unspecified keeps the clip instead of
    # discarding it over a single-character typo in the dataset.
    if intensity == "X":
        intensity = "XX"

    if not actor_raw.isdigit():
        raise CremaDParseError(f"actor id {actor_raw!r} is not numeric in {filename!r}")
    if sentence not in SENTENCE_CODES:
        raise CremaDParseError(f"unknown sentence code {sentence!r} in {filename!r}")
    if emotion not in EMOTION_TO_INDEX:
        raise CremaDParseError(f"unknown emotion code {emotion!r} in {filename!r}")
    if intensity not in INTENSITY_TO_SCALE:
        raise CremaDParseError(f"unknown intensity code {intensity!r} in {filename!r}")

    return {
        "actor_id": int(actor_raw),
        "sentence": sentence,
        "emotion": emotion,
        "intensity": intensity,
    }


def build_clip(path: Path) -> Clip:
    """Build a fully-labelled :class:`Clip` from one audio file path."""
    fields = parse_filename(path.name)
    emotion = str(fields["emotion"])
    intensity = str(fields["intensity"])
    arousal, valence = emotion_to_arousal_valence(emotion, intensity)
    return Clip(
        path=str(path),
        actor_id=int(fields["actor_id"]),  # type: ignore[arg-type]
        sentence=str(fields["sentence"]),
        emotion=emotion,
        intensity=intensity,
        emotion_index=EMOTION_TO_INDEX[emotion],
        intensity_scale=INTENSITY_TO_SCALE[intensity],
        arousal=arousal,
        valence=valence,
        escalation=escalation_target(emotion, intensity),
    )


def scan_audio_dir(audio_dir: str | Path, strict: bool = False) -> List[Clip]:
    """Scan ``AudioWAV/`` and return one :class:`Clip` per parsable file.

    With ``strict=False`` (default) unparsable filenames are skipped and counted;
    with ``strict=True`` the first bad filename raises.
    """
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(
            f"{audio_dir} does not exist. Download CREMA-D first: see scripts/prepare_crema_d.py --help"
        )

    clips: List[Clip] = []
    skipped: List[str] = []
    for path in sorted(audio_dir.glob("*.wav")):
        try:
            clips.append(build_clip(path))
        except CremaDParseError:
            if strict:
                raise
            skipped.append(path.name)

    if skipped:
        print(f"[scan_audio_dir] skipped {len(skipped)} unparsable file(s), e.g. {skipped[:3]}")
    return clips


# --- Splitting --------------------------------------------------------------


def split_actors(
    actor_ids: Iterable[int],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 1337,
) -> Dict[str, List[int]]:
    """Partition actor ids into train/val/test.

    Splitting happens at the *actor* level so no speaker appears in more than one
    split. The shuffle is seeded, so the same seed always yields the same split.
    """
    ids = sorted(set(int(a) for a in actor_ids))
    if not ids:
        raise ValueError("no actor ids given")
    if not 0.0 < val_frac + test_frac < 1.0:
        raise ValueError(f"val_frac + test_frac must be in (0, 1), got {val_frac + test_frac}")

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, round(n_total * test_frac))
    n_val = max(1, round(n_total * val_frac))
    if n_val + n_test >= n_total:
        raise ValueError(f"split leaves no training actors ({n_total} actors, {n_val} val, {n_test} test)")

    return {
        "test": sorted(shuffled[:n_test]),
        "val": sorted(shuffled[n_test : n_test + n_val]),
        "train": sorted(shuffled[n_test + n_val :]),
    }


def assign_splits(clips: Sequence[Clip], splits: Dict[str, List[int]]) -> Dict[str, List[Clip]]:
    """Bucket clips into splits according to an actor-id partition."""
    actor_to_split = {actor: name for name, actors in splits.items() for actor in actors}
    out: Dict[str, List[Clip]] = {name: [] for name in splits}
    for clip in clips:
        split_name = actor_to_split.get(clip.actor_id)
        if split_name is not None:
            out[split_name].append(clip)
    return out


def assert_no_speaker_leakage(buckets: Dict[str, List[Clip]]) -> None:
    """Raise if any actor id appears in more than one split.

    Called by the smoke test. Cheap, and catches the single most damaging silent
    bug in this kind of project.
    """
    seen: Dict[int, str] = {}
    for split_name, clips in buckets.items():
        for clip in clips:
            previous = seen.setdefault(clip.actor_id, split_name)
            if previous != split_name:
                raise AssertionError(
                    f"speaker leakage: actor {clip.actor_id} appears in both {previous!r} and {split_name!r}"
                )


# --- Manifest I/O -----------------------------------------------------------

MANIFEST_FIELDS = [*(f.name for f in Clip.__dataclass_fields__.values()), "split"]  # type: ignore[attr-defined]


def write_manifest(buckets: Dict[str, List[Clip]], out_path: str | Path) -> Path:
    """Write every clip plus its split to a single CSV manifest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for split_name, clips in buckets.items():
            for clip in clips:
                writer.writerow({**asdict(clip), "split": split_name})
    return out_path


def read_manifest(path: str | Path, split: str | None = None) -> List[Clip]:
    """Read a manifest CSV back, optionally filtering to one split."""
    rows: List[Clip] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row["split"] != split:
                continue
            rows.append(
                Clip(
                    path=row["path"],
                    actor_id=int(row["actor_id"]),
                    sentence=row["sentence"],
                    emotion=row["emotion"],
                    intensity=row["intensity"],
                    emotion_index=int(row["emotion_index"]),
                    intensity_scale=float(row["intensity_scale"]),
                    arousal=float(row["arousal"]),
                    valence=float(row["valence"]),
                    escalation=float(row["escalation"]),
                )
            )
    return rows


def summarise(buckets: Dict[str, List[Clip]]) -> str:
    """Human-readable split summary, printed by prepare_crema_d.py."""
    lines = [f"{'split':<8}{'clips':>8}{'actors':>8}   emotion counts"]
    for name in ("train", "val", "test"):
        clips = buckets.get(name, [])
        actors = {c.actor_id for c in clips}
        counts: Dict[str, int] = {}
        for clip in clips:
            counts[clip.emotion] = counts.get(clip.emotion, 0) + 1
        pretty = " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        lines.append(f"{name:<8}{len(clips):>8}{len(actors):>8}   {pretty}")
    return "\n".join(lines)
