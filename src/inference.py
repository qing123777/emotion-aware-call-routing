"""Checkpoint loading and one-utterance inference for the demo application.

The training pipeline stores only parameters that were trainable.  Reconstructing
an inference model therefore means loading the same pretrained backbone first and
then overlaying the checkpoint tensors.  This module keeps that detail out of the
web interface and gives tests a small, deterministic surface to exercise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.data.audio import TARGET_SR, fix_length, resample
from src.data.labels import EMOTION_NAMES, EMOTIONS
from src.models import build_model

AudioInput = Tuple[int, np.ndarray]

DEFAULT_WINDOW_SECONDS = 4.0
MINIMUM_SECONDS = 0.25
SILENCE_PEAK = 1.0e-4

# The raw CREMA-D waveforms are not distributed with the demo, so this is a
# conservative speech-level anchor rather than a claimed corpus measurement.
# It is close to ordinary recorded-speech levels, leaves useful headroom for
# expressive peaks, and is deliberately bounded below so quiet-room noise is
# never amplified without limit.
DATASET_MATCH_RMS_DBFS = -24.0
MIN_LEVEL_GAIN_DB = -30.0
MAX_LEVEL_GAIN_DB = 18.0
PEAK_LIMIT = 0.98
TRIM_FRAME_SECONDS = 0.020
TRIM_HOP_SECONDS = 0.010
TRIM_MARGIN_SECONDS = 0.120

# This is the "distress" grouping evaluated by scripts/routing_analysis.py.
# It performed better than the narrow ANG/DIS grouping on the held-out test set
# and matches the application's goal of putting distressed callers in touch with
# a person rather than deciding whether they sound hostile.
ESCALATE_EMOTIONS = frozenset({"ANG", "FEA", "SAD", "DIS"})


@dataclass(frozen=True)
class PreparedAudio:
    """Model-ready audio plus information useful to the interface."""

    waveform: np.ndarray
    num_valid: int
    original_sample_rate: int
    original_duration_seconds: float
    analysed_duration_seconds: float
    was_resampled: bool
    was_cropped: bool
    trimmed_seconds: float
    input_rms_dbfs: float
    output_rms_dbfs: float
    level_gain_db: float


@dataclass(frozen=True)
class Prediction:
    """One complete result from a caller utterance."""

    emotion_code: str
    emotion_name: str
    confidence: float
    probabilities: Dict[str, float]
    should_escalate: bool
    inference_seconds: float
    audio: PreparedAudio


def _as_float32(samples: np.ndarray) -> np.ndarray:
    """Convert decoded PCM to finite float32 samples in [-1, 1]."""
    if samples.dtype == np.bool_:
        raise ValueError("The decoded audio has an unsupported boolean sample format.")

    if np.issubdtype(samples.dtype, np.integer):
        if samples.dtype == np.uint8:
            converted = (samples.astype(np.float32) - 128.0) / 128.0
        else:
            info = np.iinfo(samples.dtype)
            converted = samples.astype(np.float32) / float(max(abs(info.min), info.max))
    elif np.issubdtype(samples.dtype, np.floating):
        converted = samples.astype(np.float32, copy=False)
    else:
        raise ValueError(f"The decoded audio sample format {samples.dtype} is not supported.")

    if not np.isfinite(converted).all():
        raise ValueError("The decoded audio contains invalid or infinite sample values.")
    return np.clip(converted, -1.0, 1.0)


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Accept mono or ordinary multi-channel audio and return one channel."""
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise ValueError("Audio must be mono or multi-channel, not a higher-dimensional array.")

    # Gradio supplies (samples, channels).  The second branch also accepts the
    # (channels, samples) convention used by several Python audio libraries.
    if samples.shape[1] <= 8:
        return samples.mean(axis=1)
    if samples.shape[0] <= 8:
        return samples.mean(axis=0)
    raise ValueError("The audio has an unsupported channel layout.")


def _amplitude_to_dbfs(amplitude: float) -> float:
    """Convert a non-negative linear amplitude to dBFS without log(0)."""
    return float(20.0 * np.log10(max(float(amplitude), 1.0e-12)))


def _speech_bounds(waveform: np.ndarray, sample_rate: int) -> tuple[int, int]:
    """Find speech-like content while retaining a short natural edge margin.

    The threshold is relative to both the clip's strongest frame and its own
    noise floor. This is more robust than a fixed amplitude threshold for
    microphones whose automatic gain control differs between browsers.
    """
    frame_length = max(1, int(round(sample_rate * TRIM_FRAME_SECONDS)))
    hop_length = max(1, int(round(sample_rate * TRIM_HOP_SECONDS)))
    if waveform.size <= frame_length:
        return 0, int(waveform.size)

    starts = np.arange(0, waveform.size - frame_length + 1, hop_length)
    frame_rms = np.array(
        [np.sqrt(np.mean(waveform[start : start + frame_length] ** 2)) for start in starts],
        dtype=np.float32,
    )
    strongest = float(frame_rms.max(initial=0.0))
    if strongest <= 0.0:
        return 0, int(waveform.size)

    noise_floor = float(np.percentile(frame_rms, 20.0))
    threshold = max(
        strongest * float(10.0 ** (-30.0 / 20.0)),
        noise_floor * float(10.0 ** (8.0 / 20.0)),
        1.0e-7,
    )
    active = np.flatnonzero(frame_rms >= threshold)
    if active.size == 0:
        return 0, int(waveform.size)

    margin = int(round(sample_rate * TRIM_MARGIN_SECONDS))
    start = max(0, int(starts[int(active[0])]) - margin)
    end = min(
        int(waveform.size),
        int(starts[int(active[-1])]) + frame_length + margin,
    )
    return start, end


def _match_dataset_level(
    waveform: np.ndarray,
    *,
    target_rms_dbfs: float = DATASET_MATCH_RMS_DBFS,
) -> tuple[np.ndarray, float, float, float]:
    """Match one trimmed utterance to a stable speech level with headroom.

    Returns the conditioned signal plus input/output RMS and applied gain in dB.
    Gain limits and a peak ceiling prevent a very quiet or high-crest-factor clip
    from being amplified into clipping.
    """
    input_rms = float(np.sqrt(np.mean(waveform**2)))
    input_rms_dbfs = _amplitude_to_dbfs(input_rms)
    requested_gain_db = float(target_rms_dbfs - input_rms_dbfs)
    bounded_gain_db = float(np.clip(requested_gain_db, MIN_LEVEL_GAIN_DB, MAX_LEVEL_GAIN_DB))
    gain = float(10.0 ** (bounded_gain_db / 20.0))

    peak = float(np.max(np.abs(waveform)))
    if peak > 0.0:
        gain = min(gain, PEAK_LIMIT / peak)

    conditioned = np.clip(waveform * gain, -PEAK_LIMIT, PEAK_LIMIT).astype(np.float32)
    output_rms = float(np.sqrt(np.mean(conditioned**2)))
    applied_gain_db = _amplitude_to_dbfs(gain)
    return conditioned, input_rms_dbfs, _amplitude_to_dbfs(output_rms), applied_gain_db


def prepare_audio(
    audio: AudioInput | None,
    *,
    target_sample_rate: int = TARGET_SR,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
) -> PreparedAudio:
    """Validate, clean, resample, and pad/crop one decoded recording.

    Gradio's ``Audio(type="numpy")`` supplies ``(sample_rate, samples)`` for
    both uploads and microphone recordings, so both input routes receive the
    exact same preprocessing.
    """
    if audio is None:
        raise ValueError("Record an utterance or upload an audio file first.")
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        raise ValueError("The audio input could not be decoded.")

    sample_rate, raw_samples = audio
    if not isinstance(sample_rate, (int, np.integer)) or int(sample_rate) <= 0:
        raise ValueError("The decoded audio has an invalid sample rate.")
    if target_sample_rate <= 0 or window_seconds <= 0:
        raise ValueError("The model audio settings must be positive.")

    samples = np.asarray(raw_samples)
    if samples.size == 0:
        raise ValueError("The selected audio file is empty.")

    mono = np.ascontiguousarray(_to_mono(_as_float32(samples)), dtype=np.float32)
    original_duration = mono.shape[0] / float(sample_rate)
    if original_duration < MINIMUM_SECONDS:
        raise ValueError(
            f"The utterance is too short ({original_duration:.2f}s). "
            f"Please provide at least {MINIMUM_SECONDS:.2f}s of speech."
        )
    if float(np.max(np.abs(mono))) < SILENCE_PEAK:
        raise ValueError("The recording is silent or too quiet to analyse.")

    # Remove constant microphone bias before detecting speech or calculating
    # loudness; a DC offset must not be mistaken for usable signal energy.
    mono = np.ascontiguousarray(mono - float(np.mean(mono)), dtype=np.float32)
    speech_start, speech_end = _speech_bounds(mono, int(sample_rate))
    trimmed = np.ascontiguousarray(mono[speech_start:speech_end], dtype=np.float32)
    if trimmed.size == 0 or float(np.max(np.abs(trimmed))) < SILENCE_PEAK:
        raise ValueError("No usable speech was found after removing edge silence.")

    level_matched, input_rms_dbfs, output_rms_dbfs, level_gain_db = _match_dataset_level(trimmed)
    cleaned = resample(level_matched, int(sample_rate), target_sample_rate)
    cleaned = np.ascontiguousarray(cleaned, dtype=np.float32)
    if cleaned.size == 0 or not np.isfinite(cleaned).all():
        raise ValueError("Audio preprocessing produced an invalid waveform.")

    window_samples = int(round(target_sample_rate * window_seconds))
    was_cropped = cleaned.shape[0] > window_samples
    num_valid = min(cleaned.shape[0], window_samples)
    fixed = fix_length(cleaned, window_samples, mode="center")

    return PreparedAudio(
        waveform=np.ascontiguousarray(fixed, dtype=np.float32),
        num_valid=num_valid,
        original_sample_rate=int(sample_rate),
        original_duration_seconds=original_duration,
        analysed_duration_seconds=num_valid / float(target_sample_rate),
        was_resampled=int(sample_rate) != target_sample_rate,
        was_cropped=was_cropped,
        trimmed_seconds=(mono.shape[0] - trimmed.shape[0]) / float(sample_rate),
        input_rms_dbfs=input_rms_dbfs,
        output_rms_dbfs=output_rms_dbfs,
        level_gain_db=level_gain_db,
    )


def resolve_device(requested: str) -> torch.device:
    """Resolve a user-facing device option and fail clearly when unavailable."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    return torch.device(requested)


class EmotionPredictor:
    """Own a loaded model and expose deterministic single-utterance prediction."""

    def __init__(
        self,
        model: nn.Module,
        emotion_labels: Sequence[str] = EMOTIONS,
        *,
        device: str | torch.device = "cpu",
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        checkpoint_name: str = "in-memory model",
    ) -> None:
        labels = tuple(emotion_labels)
        if not labels:
            raise ValueError("At least one emotion label is required.")

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.emotion_labels = labels
        self.window_seconds = float(window_seconds)
        self.checkpoint_name = checkpoint_name

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "auto",
    ) -> "EmotionPredictor":
        """Rebuild the backbone and overlay a local, project-produced checkpoint."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        selected_device = resolve_device(device)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch before weights_only was added
            payload = torch.load(path, map_location="cpu")

        if not isinstance(payload, dict):
            raise ValueError("The checkpoint is not a supported project checkpoint.")
        config = payload.get("config")
        state_dict = payload.get("state_dict")
        labels = tuple(payload.get("emotion_labels") or EMOTIONS)
        if not isinstance(config, dict) or not isinstance(state_dict, dict):
            raise ValueError("The checkpoint is missing its config or state_dict.")

        model = build_model(config, num_emotions=len(labels))
        model_keys = set(model.state_dict())
        unknown_keys = sorted(set(state_dict) - model_keys)
        if unknown_keys:
            preview = ", ".join(unknown_keys[:3])
            raise ValueError(
                "The checkpoint does not match the reconstructed model; "
                f"unexpected parameter(s): {preview}"
            )

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                "The checkpoint contains unexpected parameters: "
                + ", ".join(incompatible.unexpected_keys[:3])
            )
        if not payload.get("trainable_only", False) and incompatible.missing_keys:
            raise ValueError(
                "A full checkpoint is missing model parameters: "
                + ", ".join(incompatible.missing_keys[:3])
            )

        data_config = config.get("data", {})
        window_seconds = float(data_config.get("window_seconds", DEFAULT_WINDOW_SECONDS))
        return cls(
            model,
            labels,
            device=selected_device,
            window_seconds=window_seconds,
            checkpoint_name=path.name,
        )

    def predict(self, audio: AudioInput | None) -> Prediction:
        prepared = prepare_audio(audio, window_seconds=self.window_seconds)
        waveform = torch.from_numpy(prepared.waveform).unsqueeze(0).to(self.device)
        num_valid = torch.tensor([prepared.num_valid], dtype=torch.long, device=self.device)

        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(waveform, num_valid=num_valid)
            if "emotion" not in outputs:
                raise RuntimeError("The loaded model has no emotion output head.")
            probabilities = torch.softmax(outputs["emotion"], dim=-1)[0].detach().cpu()
        elapsed = time.perf_counter() - start

        if probabilities.numel() != len(self.emotion_labels):
            raise RuntimeError(
                f"The model returned {probabilities.numel()} scores for "
                f"{len(self.emotion_labels)} labels."
            )

        best_index = int(probabilities.argmax().item())
        code = self.emotion_labels[best_index]
        readable = EMOTION_NAMES.get(code, code.lower())
        probability_map = {
            EMOTION_NAMES.get(label, label.lower()).title(): float(probabilities[index].item())
            for index, label in enumerate(self.emotion_labels)
        }
        return Prediction(
            emotion_code=code,
            emotion_name=readable.title(),
            confidence=float(probabilities[best_index].item()),
            probabilities=probability_map,
            should_escalate=code in ESCALATE_EMOTIONS,
            inference_seconds=elapsed,
            audio=prepared,
        )
