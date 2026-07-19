"""Phase 0 smoke test + Phase 1 data-pipeline tests."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.data import (
    TwoViewTransform,
    build_eval_transform,
    build_ssl_transform,
)

IMAGE_SIZE = 96


def make_image(size: int = IMAGE_SIZE, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def test_smoke_tiny_model_forward():
    """Phase 0 smoke test: a tiny model does a forward pass on a random batch."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 16),
    )
    out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_ssl_transform_returns_two_views_of_same_image():
    image = make_image()
    transform = TwoViewTransform(build_ssl_transform(IMAGE_SIZE))
    view_a, view_b = transform(image)
    assert view_a.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert view_b.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert view_a.dtype == torch.float32


def test_ssl_transform_is_stochastic():
    image = make_image()
    transform = build_ssl_transform(IMAGE_SIZE)
    torch.manual_seed(0)
    view_a = transform(image)
    torch.manual_seed(1)
    view_b = transform(image)
    assert not torch.allclose(view_a, view_b)


def test_ssl_transform_is_reproducible_with_seed():
    image = make_image()
    transform = build_ssl_transform(IMAGE_SIZE)
    torch.manual_seed(0)
    first = transform(image)
    torch.manual_seed(0)
    second = transform(image)
    assert torch.allclose(first, second)


def test_eval_transform_is_deterministic():
    image = make_image()
    transform = build_eval_transform(IMAGE_SIZE)
    assert torch.allclose(transform(image), transform(image))


def test_loader_collates_two_views():
    dataset = datasets.FakeData(
        size=16,
        image_size=(3, IMAGE_SIZE, IMAGE_SIZE),
        transform=TwoViewTransform(build_ssl_transform(IMAGE_SIZE)),
    )
    loader = DataLoader(dataset, batch_size=8)
    (view_a, view_b), targets = next(iter(loader))
    assert view_a.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert view_b.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert targets.shape == (8,)


@pytest.mark.parametrize("config", ["configs/stl10.yaml", "configs/cifar10_debug.yaml"])
def test_configs_are_valid(config):
    cfg = load_config(config)
    assert cfg["dataset"]["name"] in {"stl10", "cifar10"}
