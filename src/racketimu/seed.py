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


DEVICES = ("cpu", "mps", "auto")


def get_device(preference: str = "cpu") -> torch.device:
    """Pick the torch device from the config's `train.device` value.

    "cpu" (the default) is the only choice for reported experiments, because MPS
    results are not bit-for-bit reproducible. "mps" and "auto" are opt-in for quick
    local iteration; "auto" uses MPS when available, else CPU.
    """
    if preference not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {preference!r}")
    mps_ok = torch.backends.mps.is_available()
    if preference == "mps" and not mps_ok:
        raise RuntimeError("device 'mps' was requested but MPS is not available here")
    if preference == "mps" or (preference == "auto" and mps_ok):
        return torch.device("mps")
    return torch.device("cpu")
