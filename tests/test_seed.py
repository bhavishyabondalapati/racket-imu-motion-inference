"""Tests that seeding makes every random source reproducible."""

import random

import numpy as np
import pytest
import torch

from racketimu.config import load_config
from racketimu.seed import get_device, set_seed


def _draw_all(seed: int) -> tuple[float, float, float, float]:
    rng = set_seed(seed)
    return random.random(), float(np.random.rand()), float(rng.random()), float(torch.rand(1))


def test_same_seed_gives_same_numbers():
    assert _draw_all(42) == _draw_all(42)


def test_different_seeds_give_different_numbers():
    assert _draw_all(1) != _draw_all(2)


@pytest.mark.parametrize("bad", [-1, 1.5, "0"])
def test_invalid_seed_is_rejected(bad):
    with pytest.raises(ValueError):
        set_seed(bad)


def test_default_device_is_cpu():
    assert get_device().type == "cpu"


def test_config_default_device_is_cpu():
    assert load_config()["train"]["device"] == "cpu"


def test_auto_device_is_usable():
    device = get_device("auto")
    assert device.type in {"mps", "cpu"}
    assert torch.zeros(2, device=device).sum().item() == 0


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        get_device("cuda")
