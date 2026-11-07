"""PyTorch ``Dataset`` over a CREMA-D manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import TARGET_SR, fix_length, load_audio
from .crema_d import Clip, read_manifest
from .labels import INTENSITY_TO_INDEX


class SpeechEmotionDataset(Dataset):
    """Fixed-length audio windows with categorical, dimensional and escalation labels.

    Every item is a dict so the training loop can pick whichever heads it needs
    without the dataset knowing about the model:

        waveform    float32 (num_samples,)  mono, 16 kHz, in [-1, 1]
        num_valid   int64   ()              real audio length before right-padding
        emotion     int64   ()              class index into labels.EMOTIONS
        intensity   int64   ()              class index into labels.INTENSITIES
        arousal     float32 ()              proxy arousal in [-1, 1]
        valence     float32 ()              proxy valence in [-1, 1]
        escalation  float32 ()              combined 0..1 escalation target
        actor_id    int64   ()              kept for per-speaker error analysis

    ``num_valid`` exists so the model can ignore padding when pooling. CREMA-D
    clips are mostly shorter than the window, so without it a large fraction of
    every pooled representation would be silence.

    Parameters
    ----------
    clips
        Clips to serve. Use :func:`from_manifest` to load these from CSV.
    window_seconds
        Length of the audio window fed to the model. 4.0 s is the default because
        it matches the streaming window the deployed system will use.
    train
        When True, windows are randomly cropped (cheap augmentation) and the
        optional augmentations are applied. When False everything is
        deterministic, which is what evaluation needs.
    """

    def __init__(
        self,
        clips: Sequence[Clip],
        window_seconds: float = 4.0,
        sample_rate: int = TARGET_SR,
        train: bool = False,
        gain_jitter_db: float = 0.0,
        noise_snr_db: float | None = None,
    ) -> None:
        if not clips:
            raise ValueError("dataset is empty; check the manifest path and split name")
        self.clips: List[Clip] = list(clips)
        self.sample_rate = sample_rate
        self.num_samples = int(round(window_seconds * sample_rate))
        self.train = train
        self.gain_jitter_db = gain_jitter_db
        self.noise_snr_db = noise_snr_db

    @classmethod
    def from_manifest(
        cls, manifest_path: str | Path, split: str, **kwargs
    ) -> "SpeechEmotionDataset":
        clips = read_manifest(manifest_path, split=split)
        if not clips:
            raise ValueError(f"no clips for split {split!r} in {manifest_path}")
        return cls(clips, train=(split == "train"), **kwargs)

    def __len__(self) -> int:
        return len(self.clips)

    def _augment(self, waveform: np.ndarray) -> np.ndarray:
        if self.gain_jitter_db > 0:
            gain_db = np.random.uniform(-self.gain_jitter_db, self.gain_jitter_db)
            waveform = waveform * float(10.0 ** (gain_db / 20.0))
        if self.noise_snr_db is not None:
            signal_power = float(np.mean(waveform**2)) + 1e-12
            noise_power = signal_power / float(10.0 ** (self.noise_snr_db / 10.0))
            waveform = waveform + np.random.normal(0.0, np.sqrt(noise_power), waveform.shape)
        return np.clip(waveform, -1.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        clip = self.clips[index]
        waveform = load_audio(clip.path, target_sr=self.sample_rate)

        # Record the real length before padding, so the model can mask silence.
        num_valid = min(int(waveform.shape[-1]), self.num_samples)

        waveform = fix_length(waveform, self.num_samples, mode="random" if self.train else "center")
        if self.train:
            waveform = self._augment(waveform)

        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)),
            "num_valid": torch.tensor(num_valid, dtype=torch.long),
            "emotion": torch.tensor(clip.emotion_index, dtype=torch.long),
            "intensity": torch.tensor(INTENSITY_TO_INDEX[clip.intensity], dtype=torch.long),
            "arousal": torch.tensor(clip.arousal, dtype=torch.float32),
            "valence": torch.tensor(clip.valence, dtype=torch.float32),
            "escalation": torch.tensor(clip.escalation, dtype=torch.float32),
            "actor_id": torch.tensor(clip.actor_id, dtype=torch.long),
        }


def class_weights(clips: Sequence[Clip], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean 1.

    CREMA-D has roughly 4x fewer neutral clips than the other categories, so an
    unweighted cross-entropy quietly learns to under-predict neutral. Pass these
    to ``nn.CrossEntropyLoss(weight=...)``.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for clip in clips:
        counts[clip.emotion_index] += 1
    if (counts == 0).any():
        missing = [i for i, c in enumerate(counts.tolist()) if c == 0]
        raise ValueError(f"class(es) {missing} have no examples; cannot compute weights")
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.mean()
