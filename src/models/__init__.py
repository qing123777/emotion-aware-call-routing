"""Model components: backbone wrapper, pooling, prediction heads."""

from .finetune import STRATEGIES, apply_strategy, count_parameters
from .heads import ClassificationHead, RegressionHead, build_heads
from .model import SpeechEmotionModel, build_model, frame_mask_from_lengths, load_backbone
from .pooling import LastLayerPool, WeightedLayerPool, masked_mean

__all__ = [
    "STRATEGIES",
    "ClassificationHead",
    "LastLayerPool",
    "RegressionHead",
    "SpeechEmotionModel",
    "WeightedLayerPool",
    "apply_strategy",
    "build_heads",
    "build_model",
    "count_parameters",
    "frame_mask_from_lengths",
    "load_backbone",
    "masked_mean",
]
