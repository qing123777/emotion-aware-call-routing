"""Emotion label definitions and the categorical -> arousal/valence proxy mapping.

IMPORTANT
---------
CREMA-D provides categorical emotion labels and a coarse intensity tag. It does
NOT provide measured arousal/valence ratings. The values in ``EMOTION_TO_AV``
below are a *proxy*: they place each of the six categories at a representative
point in Russell's circumplex model of affect [1] so that the regression
pipeline can be developed and debugged before a dimensionally-annotated corpus
(MSP-Podcast, IEMOCAP) becomes available.

This proxy must be reported as a limitation. It is not human-rated ground truth,
and results obtained on it are not comparable to published dimensional SER
numbers.

[1] J. A. Russell, "A circumplex model of affect," Journal of Personality and
    Social Psychology, vol. 39, no. 6, pp. 1161-1178, 1980.
"""

from __future__ import annotations

from typing import Dict, Tuple

# --- Categorical labels -----------------------------------------------------

# CREMA-D emotion codes, in a fixed order. The index is the class id used by the
# classification head; never reorder this list without retraining.
EMOTIONS: Tuple[str, ...] = ("NEU", "HAP", "SAD", "ANG", "FEA", "DIS")

EMOTION_NAMES: Dict[str, str] = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgusted",
}

EMOTION_TO_INDEX: Dict[str, int] = {code: i for i, code in enumerate(EMOTIONS)}

# --- Intensity --------------------------------------------------------------

# CREMA-D intensity tags. "XX" means the actor was not given an intensity
# instruction, so it is treated as the mid point rather than dropped.
INTENSITY_TO_SCALE: Dict[str, float] = {
    "LO": 0.6,
    "MD": 0.8,
    "HI": 1.0,
    "XX": 0.8,
}

# Fixed order for the optional intensity classification head. "XX" is kept as a
# class rather than dropped, because a large share of CREMA-D carries it and
# discarding those clips would waste most of the dataset.
INTENSITIES: Tuple[str, ...] = ("XX", "LO", "MD", "HI")
INTENSITY_TO_INDEX: Dict[str, int] = {code: i for i, code in enumerate(INTENSITIES)}

# --- Dimensional proxy ------------------------------------------------------

# (arousal, valence), each in [-1, 1].
#   arousal: -1 = calm / low energy, +1 = highly activated
#   valence: -1 = unpleasant,        +1 = pleasant
EMOTION_TO_AV: Dict[str, Tuple[float, float]] = {
    "NEU": (0.00, 0.00),
    "HAP": (0.60, 0.80),
    "SAD": (-0.50, -0.70),
    "ANG": (0.70, -0.70),
    "FEA": (0.80, -0.60),
    "DIS": (0.20, -0.70),
}


def emotion_to_arousal_valence(emotion: str, intensity: str = "XX") -> Tuple[float, float]:
    """Map a CREMA-D (emotion, intensity) pair to proxy arousal and valence.

    The intensity tag scales the *magnitude* of the point, moving weaker
    readings toward the neutral origin, while leaving its direction unchanged.

    Returns
    -------
    (arousal, valence), each clipped to [-1, 1].
    """
    if emotion not in EMOTION_TO_AV:
        raise KeyError(f"unknown emotion code {emotion!r}; expected one of {sorted(EMOTION_TO_AV)}")
    scale = INTENSITY_TO_SCALE.get(intensity, INTENSITY_TO_SCALE["XX"])
    arousal, valence = EMOTION_TO_AV[emotion]
    return (
        max(-1.0, min(1.0, arousal * scale)),
        max(-1.0, min(1.0, valence * scale)),
    )


def escalation_target(emotion: str, intensity: str = "XX") -> float:
    """A single 0..1 'how close is this to an angry customer' score.

    This is the quantity the escalation layer ultimately thresholds. It is
    defined directly from the proxy dimensions: high arousal combined with
    negative valence. Neutral speech scores near 0, shouted anger near 1.
    """
    arousal, valence = emotion_to_arousal_valence(emotion, intensity)
    # Rescale arousal from [-1, 1] to [0, 1] and negative valence likewise, then
    # take their product so BOTH conditions must hold for a high score.
    activation = (arousal + 1.0) / 2.0
    unpleasantness = (1.0 - valence) / 2.0
    return round(activation * unpleasantness, 6)
