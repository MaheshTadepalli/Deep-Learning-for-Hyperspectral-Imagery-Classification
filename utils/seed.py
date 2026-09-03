"""
utils/seed.py
===============
Reproducibility helper — sets random seeds across numpy, torch (CPU + CUDA)
and Python's `random` module. Called at the start of every run in
train.py / benchmark.py for the "repeat with multiple random seeds"
requirement.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Full determinism (slower) — uncomment if bit-exact repeatability needed:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
