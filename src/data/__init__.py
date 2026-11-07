"""Data loading for the speech emotion monitoring project."""

from .crema_d import (
    Clip,
    assert_no_speaker_leakage,
    assign_splits,
    build_clip,
    parse_filename,
    read_manifest,
    scan_audio_dir,
    split_actors,
    summarise,
    write_manifest,
)
from .dataset import SpeechEmotionDataset, class_weights
from .labels import EMOTIONS, emotion_to_arousal_valence, escalation_target

__all__ = [
    "Clip",
    "EMOTIONS",
    "SpeechEmotionDataset",
    "assert_no_speaker_leakage",
    "assign_splits",
    "build_clip",
    "class_weights",
    "emotion_to_arousal_valence",
    "escalation_target",
    "parse_filename",
    "read_manifest",
    "scan_audio_dir",
    "split_actors",
    "summarise",
    "write_manifest",
]
