"""The four ways of deciding which parts of the backbone are allowed to learn.

This is the project's main ablation axis, so all four live behind one string:

  frozen   nothing in the backbone moves; only the heads learn
  partial  the top N transformer layers move
  full     the whole transformer moves
  lora     the backbone is locked, and small trainable adapters are added

The convolutional feature extractor at the very bottom is always frozen. It
learns generic sound structure that no small emotion dataset can improve on, and
unfreezing it is a known way to destabilise fine-tuning.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from torch import nn

STRATEGIES = ("frozen", "partial", "full", "lora")


class MissingPeftError(RuntimeError):
    """Raised when strategy is 'lora' but the peft library is not installed."""


def freeze_all(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_all(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def freeze_feature_extractor(backbone: nn.Module) -> bool:
    """Freeze the convolutional front end. Returns whether one was found."""
    extractor = getattr(backbone, "feature_extractor", None)
    if extractor is None:
        return False
    freeze_all(extractor)
    return True


def _encoder_layers(backbone: nn.Module) -> Sequence[nn.Module]:
    """The transformer layer list, for wav2vec2/WavLM/HuBERT-shaped models."""
    encoder = getattr(backbone, "encoder", None)
    layers = getattr(encoder, "layers", None) if encoder is not None else None
    if layers is None:
        raise AttributeError(
            "backbone has no .encoder.layers; the 'partial' strategy does not know "
            "how to find its transformer layers"
        )
    return layers


def apply_strategy(
    backbone: nn.Module,
    strategy: str,
    partial_unfreeze_layers: int = 4,
    lora_config: Dict[str, Any] | None = None,
) -> nn.Module:
    """Set requires_grad across the backbone, or wrap it with LoRA adapters.

    Returns the backbone to use from here on. Only 'lora' returns a different
    object; the others modify in place and return the same one.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")

    if strategy == "lora":
        return _apply_lora(backbone, lora_config or {})

    if strategy == "full":
        unfreeze_all(backbone)
    else:
        freeze_all(backbone)
        if strategy == "partial":
            layers = _encoder_layers(backbone)
            if partial_unfreeze_layers > len(layers):
                raise ValueError(
                    f"asked to unfreeze {partial_unfreeze_layers} layers but the backbone "
                    f"has only {len(layers)}"
                )
            for layer in layers[-partial_unfreeze_layers:]:
                unfreeze_all(layer)

    freeze_feature_extractor(backbone)
    return backbone


def _apply_lora(backbone: nn.Module, config: Dict[str, Any]) -> nn.Module:
    """Lock the backbone and attach small trainable adapters.

    peft handles the mechanics: it freezes the original weights itself and adds
    a low-rank pair beside each targeted layer.
    """
    try:
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingPeftError(
            "finetune_strategy is 'lora' but peft is not installed. Run: pip install peft"
        ) from exc

    lora = LoraConfig(
        r=int(config.get("r", 8)),
        lora_alpha=int(config.get("alpha", 16)),
        lora_dropout=float(config.get("dropout", 0.05)),
        target_modules=list(config.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
    )
    try:
        wrapped = get_peft_model(backbone, lora)
    except ImportError as exc:
        # peft checks several optional backends while attaching adapters, and
        # raises if one is installed at a version it does not support. Colab
        # ships an old torchao that trips this. None of them are used here.
        raise MissingPeftError(
            f"peft could not attach LoRA adapters: {exc}\n"
            "This is usually an unrelated optional package at an unsupported version. "
            "In Colab: !pip uninstall -q -y torchao\n"
            "Or fall back to a strategy with no extra dependencies:\n"
            "  --override model.finetune_strategy=partial"
        ) from exc

    freeze_feature_extractor(backbone)
    return wrapped


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """Trainable vs total parameter counts, for the results JSON.

    Worth recording per run: it is the number that makes the LoRA comparison
    concrete in the report.
    """
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": round(100.0 * trainable / total, 4) if total else 0.0,
    }
