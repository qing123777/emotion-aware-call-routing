"""The full model: pre-trained backbone, layer pooling, prediction heads."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn

from .finetune import apply_strategy, count_parameters
from .heads import build_heads
from .pooling import LastLayerPool, WeightedLayerPool, masked_mean


def load_backbone(name: str):
    """Download (or load from cache) a pre-trained speech model.

    Kept separate from the model class so tests can inject a small stand-in
    instead of pulling ~360MB from the internet.
    """
    from transformers import AutoModel  # noqa: PLC0415 - heavy, only needed for real runs

    return AutoModel.from_pretrained(name)


def frame_mask_from_lengths(
    backbone: nn.Module, num_valid: torch.Tensor, num_frames: int
) -> torch.Tensor | None:
    """Convert per-clip sample counts into a per-frame 1/0 mask.

    The backbone's convolutional front end reduces roughly 16,000 samples per
    second down to about 50 frames per second, and only the model itself knows
    the exact ratio. Transformers exposes that as
    _get_feat_extract_output_lengths. If it is unavailable we return None and
    pooling falls back to averaging everything, which is wrong but not fatal.
    """
    to_frames = getattr(backbone, "_get_feat_extract_output_lengths", None)
    if to_frames is None:
        return None

    frame_lengths = to_frames(num_valid)
    if isinstance(frame_lengths, torch.Tensor):
        frame_lengths = frame_lengths.to(num_valid.device).long()
    else:
        frame_lengths = torch.as_tensor(frame_lengths, device=num_valid.device).long()

    positions = torch.arange(num_frames, device=num_valid.device).unsqueeze(0)
    return (positions < frame_lengths.clamp(max=num_frames).unsqueeze(1)).long()


class SpeechEmotionModel(nn.Module):
    """Backbone -> layer pooling -> average over time -> heads.

    Parameters
    ----------
    backbone
        Any wav2vec2/WavLM/HuBERT-shaped model. Must accept
        ``output_hidden_states=True`` and expose ``config.hidden_size``.
    finetune_strategy
        One of frozen / partial / full / lora. See finetune.py.
    layer_pooling
        "weighted" learns one weight per layer; "last" uses only the top layer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_emotions: int,
        num_intensities: int = 0,
        num_dimensional: int = 0,
        finetune_strategy: str = "lora",
        partial_unfreeze_layers: int = 4,
        lora_config: Dict[str, Any] | None = None,
        layer_pooling: str = "weighted",
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_size = int(backbone.config.hidden_size)
        # hidden_states includes the embedding output as well as every layer.
        num_hidden_states = int(backbone.config.num_hidden_layers) + 1

        self.backbone = apply_strategy(
            backbone,
            finetune_strategy,
            partial_unfreeze_layers=partial_unfreeze_layers,
            lora_config=lora_config,
        )
        self.finetune_strategy = finetune_strategy

        if layer_pooling == "weighted":
            self.layer_pool: nn.Module = WeightedLayerPool(num_hidden_states)
        elif layer_pooling == "last":
            self.layer_pool = LastLayerPool()
        else:
            raise ValueError(f"unknown layer_pooling {layer_pooling!r}; expected 'weighted' or 'last'")

        self.heads = build_heads(
            input_size=hidden_size,
            num_emotions=num_emotions,
            num_intensities=num_intensities,
            num_dimensional=num_dimensional,
            hidden_size=head_hidden_size,
            dropout=head_dropout,
        )

    def _base_backbone(self) -> nn.Module:
        """The real speech model, digging through any LoRA wrappers.

        peft nests two deep: PeftModel.base_model is a LoraModel, and
        LoraModel.model is the actual WavLM. Attribute forwarding usually hides
        that, but relying on it is fragile, so unwrap until we find the object
        that owns the frame-length method we need.
        """
        candidate = self.backbone
        for _ in range(4):  # depth guard; peft nests two, never four
            if hasattr(candidate, "_get_feat_extract_output_lengths"):
                return candidate
            nxt = getattr(candidate, "base_model", None) or getattr(candidate, "model", None)
            if nxt is None or nxt is candidate:
                break
            candidate = nxt
        return candidate

    def forward(
        self, waveform: torch.Tensor, num_valid: torch.Tensor | None = None
    ) -> Dict[str, torch.Tensor]:
        """waveform: (batch, samples). Returns one tensor per head."""
        outputs = self.backbone(waveform, output_hidden_states=True)
        hidden_states: Sequence[torch.Tensor] = outputs.hidden_states

        frames = self.layer_pool(hidden_states)

        mask = None
        if num_valid is not None:
            mask = frame_mask_from_lengths(self._base_backbone(), num_valid, frames.shape[1])

        clip_vector = masked_mean(frames, mask)
        return {name: head(clip_vector) for name, head in self.heads.items()}

    def parameter_summary(self) -> Dict[str, int]:
        return count_parameters(self)

    def layer_weights(self) -> torch.Tensor | None:
        """Learned per-layer weights, or None when using last-layer pooling."""
        if isinstance(self.layer_pool, WeightedLayerPool):
            return self.layer_pool.layer_weights()
        return None


def build_model(config: Dict[str, Any], num_emotions: int, num_intensities: int = 0) -> SpeechEmotionModel:
    """Build a model straight from the parsed YAML config."""
    model_cfg = config.get("model", {})
    heads_cfg = config.get("heads", {})

    backbone = load_backbone(model_cfg.get("backbone", "microsoft/wavlm-base-plus"))

    return SpeechEmotionModel(
        backbone=backbone,
        num_emotions=num_emotions,
        num_intensities=num_intensities if heads_cfg.get("intensity") else 0,
        num_dimensional=2 if heads_cfg.get("dimensional") else 0,
        finetune_strategy=model_cfg.get("finetune_strategy", "lora"),
        partial_unfreeze_layers=int(model_cfg.get("partial_unfreeze_layers", 4)),
        lora_config=model_cfg.get("lora"),
        layer_pooling=model_cfg.get("layer_pooling", "weighted"),
    )
