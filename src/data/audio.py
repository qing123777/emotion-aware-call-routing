"""Audio loading that works with whatever backend is installed.

Colab and Kaggle ship ``soundfile``/``torchaudio``; the local Windows machine
currently has neither, so this module degrades to ``scipy`` and finally to the
standard library ``wave`` module. Keep it dependency-light: shared code imports
this, not a specific backend.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SR = 16_000


def _load_with_soundfile(path: Path):
    import soundfile as sf  # noqa: PLC0415 - optional backend

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


def _load_with_scipy(path: Path):
    from scipy.io import wavfile  # noqa: PLC0415 - optional backend

    sr, data = wavfile.read(str(path))
    data = _to_float32(data)
    if data.ndim == 1:
        data = data[None, :]
    else:
        data = data.T
    return data, sr


def _load_with_wave(path: Path):
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sr = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sample_width)
    if dtype is None:
        raise ValueError(f"unsupported sample width {sample_width} bytes in {path}")

    data = _to_float32(np.frombuffer(frames, dtype=dtype))
    data = data.reshape(-1, n_channels).T
    return data, sr


def _to_float32(data: np.ndarray) -> np.ndarray:
    """Convert integer PCM to float32 in [-1, 1]; pass floats through."""
    if data.dtype == np.float32:
        return data
    if data.dtype == np.float64:
        return data.astype(np.float32)
    if data.dtype == np.uint8:  # 8-bit PCM is unsigned, centred on 128
        return ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    info = np.iinfo(data.dtype)
    return (data.astype(np.float32) / float(-info.min)).astype(np.float32)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample a 1-D waveform. Uses scipy's polyphase filter when available."""
    if orig_sr == target_sr:
        return waveform
    try:
        from math import gcd  # noqa: PLC0415

        from scipy.signal import resample_poly  # noqa: PLC0415

        divisor = gcd(orig_sr, target_sr)
        return resample_poly(waveform, target_sr // divisor, orig_sr // divisor).astype(np.float32)
    except ImportError:
        # Linear interpolation fallback. Adequate for a smoke test, not for
        # producing reported results -- scipy is in requirements.txt for a reason.
        duration = waveform.shape[-1] / orig_sr
        n_target = int(round(duration * target_sr))
        src_idx = np.linspace(0.0, waveform.shape[-1] - 1, num=n_target)
        return np.interp(src_idx, np.arange(waveform.shape[-1]), waveform).astype(np.float32)


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sr``.

    Returns a 1-D array in [-1, 1]. Multi-channel input is averaged to mono.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    for loader in (_load_with_soundfile, _load_with_scipy, _load_with_wave):
        try:
            data, sr = loader(path)
            break
        except ImportError:
            continue
    else:  # pragma: no cover - only if every backend is missing
        raise RuntimeError("no usable audio backend (tried soundfile, scipy, wave)")

    mono = data.mean(axis=0) if data.ndim > 1 else data
    return resample(np.ascontiguousarray(mono, dtype=np.float32), sr, target_sr)


def fix_length(waveform: np.ndarray, num_samples: int, mode: str = "center") -> np.ndarray:
    """Crop or zero-pad a waveform to exactly ``num_samples``.

    ``mode`` is ``"random"`` for training (a random crop is cheap augmentation)
    or ``"center"`` for deterministic evaluation. Cropping honours the mode;
    padding is always added on the right.

    Right-padding is deliberate: it keeps the real audio as a prefix, so the
    model can mask the padding with a single valid-length number per clip.
    Centre-padding would put silence on both sides and need a start index too.
    This matters here because CREMA-D clips average well under the 4 s window,
    so a large share of every batch is padding.
    """
    length = waveform.shape[-1]
    if length == num_samples:
        return waveform
    if length > num_samples:
        if mode == "random":
            start = int(np.random.randint(0, length - num_samples + 1))
        else:
            start = (length - num_samples) // 2
        return waveform[start : start + num_samples]

    return np.pad(waveform, (0, num_samples - length), mode="constant")
