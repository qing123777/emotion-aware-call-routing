"""The small layers that turn one clip vector into predictions."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ClassificationHead(nn.Module):
    """One hidden layer, then one score per class.

    A single linear layer would also work. The extra hidden layer costs little
    and gives the head some capacity of its own, which matters when the backbone
    is frozen or only lightly adapted.
    """

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Continuous outputs squashed to [-1, 1].

    Out of scope for the current plan: no freely available corpus has real
    arousal/valence ratings. Kept because switching it on is a config flag if
    that ever changes.
    """

    def __init__(self, input_size: int, num_outputs: int = 2, hidden_size: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_heads(
    input_size: int,
    num_emotions: int,
    num_intensities: int = 0,
    num_dimensional: int = 0,
    hidden_size: int = 256,
    dropout: float = 0.1,
) -> nn.ModuleDict:
    """Assemble whichever heads the config asked for.

    Passing 0 for a head means it is not built at all, so it costs no parameters
    and appears nowhere in the loss.
    """
    heads: Dict[str, nn.Module] = {
        "emotion": ClassificationHead(input_size, num_emotions, hidden_size, dropout)
    }
    if num_intensities:
        heads["intensity"] = ClassificationHead(input_size, num_intensities, hidden_size, dropout)
    if num_dimensional:
        heads["dimensional"] = RegressionHead(input_size, num_dimensional, hidden_size, dropout)
    return nn.ModuleDict(heads)
