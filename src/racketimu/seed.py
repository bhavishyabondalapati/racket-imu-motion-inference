"""One place to seed every random number generator the project uses."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy, and PyTorch, and return a fresh NumPy Generator.

    Prefer passing the returned Generator around over using NumPy's global state.

    Example:
        >>> rng = set_seed(0)
        >>> float(rng.random()) == float(set_seed(0).random())
        True
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS and CUDA generators
    os.environ["PYTHONHASHSEED"] = str(seed)  # only affects subprocesses started after this
    return np.random.default_rng(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Pick the torch device: "auto" uses Apple Silicon's MPS when available, else CPU."""
    if preference == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(preference)
