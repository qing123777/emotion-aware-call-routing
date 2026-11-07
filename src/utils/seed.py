"""Seeding, so a reported number can actually be reproduced."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337, deterministic: bool = True) -> int:
    """Seed python, numpy and torch. Call this first in every entry point.

    ``deterministic=True`` also pins cuDNN into deterministic mode, which costs
    some speed but means two runs of the same config give the same number. Keep
    it on for anything you intend to report.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
