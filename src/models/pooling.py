"""Turning a sequence of frames into one vector per clip."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class WeightedLayerPool(nn.Module):
    """Combine every transformer layer's output using learned weights.

    The usual approach takes only the final layer. That is a habit borrowed from
    text models. In speech models the layers specialise: lower ones keep pitch,
    energy and voice quality, while the top ones drift toward the objective the
    model was pre-trained on, which is not emotion. Learning one weight per layer
    lets the model pull from wherever the emotional signal actually lives.

    Costs one parameter per layer, so around 13 on a 12-layer backbone. The
    learned weights are also worth plotting -- they show which layers matter.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        # Start uniform: softmax of all-zeros gives every layer equal weight.
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        """hidden_states: num_layers tensors of (batch, frames, features)."""
        if len(hidden_states) != self.weights.numel():
            raise ValueError(
                f"expected {self.weights.numel()} layers, got {len(hidden_states)}. "
                "Was the pool built with the wrong num_layers?"
            )
        stacked = torch.stack(tuple(hidden_states), dim=0)
        weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def layer_weights(self) -> torch.Tensor:
        """The normalised weights, for plotting in the report."""
        return torch.softmax(self.weights.detach(), dim=0)


class LastLayerPool(nn.Module):
    """Take only the final layer. The ablation arm against WeightedLayerPool."""

    def forward(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        return hidden_states[-1]


def masked_mean(frames: torch.Tensor, frame_mask: torch.Tensor | None) -> torch.Tensor:
    """Average over time, ignoring padded frames.

    frames:     (batch, num_frames, features)
    frame_mask: (batch, num_frames) with 1 for real audio and 0 for padding

    Without the mask, a clip that is half padding has half its representation
    made of silence. CREMA-D clips average well under the 4 s window, so this is
    not a small correction.
    """
    if frame_mask is None:
        return frames.mean(dim=1)

    mask = frame_mask.unsqueeze(-1).to(frames.dtype)
    total = (frames * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)  # clamp guards against an all-zero row
    return total / count
