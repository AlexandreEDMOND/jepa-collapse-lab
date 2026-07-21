"""Dataset and DataLoader builders: unlabeled SSL pairs + labeled probe splits."""

from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .augmentations import TwoViewTransform, build_eval_transform, build_ssl_transform


def _build_datasets(
    cfg: dict[str, Any],
) -> tuple[datasets.VisionDataset, datasets.VisionDataset, datasets.VisionDataset]:
    name = cfg["dataset"]["name"]
    root = cfg["dataset"]["root"]
    image_size = cfg["dataset"]["image_size"]
    aug = cfg.get("augmentation", {})

    ssl_transform = TwoViewTransform(
        build_ssl_transform(
            image_size,
            crop_scale=tuple(aug.get("crop_scale", (0.3, 1.0))),
            jitter_strength=aug.get("jitter_strength", 0.4),
            grayscale_p=aug.get("grayscale_p", 0.2),
            blur_p=aug.get("blur_p", 0.5),
        )
    )
    eval_transform = build_eval_transform(image_size)

    if name == "stl10":
        # `unlabeled` for SSL pretraining; labeled `train`/`test` only for the probe.
        ssl = datasets.STL10(root, split="unlabeled", transform=ssl_transform, download=True)
        probe_train = datasets.STL10(root, split="train", transform=eval_transform, download=True)
        probe_test = datasets.STL10(root, split="test", transform=eval_transform, download=True)
    elif name == "cifar10":
        ssl = datasets.CIFAR10(root, train=True, transform=ssl_transform, download=True)
        probe_train = datasets.CIFAR10(root, train=True, transform=eval_transform, download=True)
        probe_test = datasets.CIFAR10(root, train=False, transform=eval_transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name!r}")
    return ssl, probe_train, probe_test


def build_loaders(cfg: dict[str, Any]) -> dict[str, DataLoader]:
    """Build the three loaders: ``ssl`` (two views), ``probe_train``, ``probe_test``."""
    ssl_dataset, probe_train_dataset, probe_test_dataset = _build_datasets(cfg)
    loader_cfg = cfg.get("loader", {})
    batch_size = loader_cfg.get("batch_size", 256)
    common: dict[str, Any] = {
        "num_workers": loader_cfg.get("num_workers", 4),
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "ssl": DataLoader(
            ssl_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **common
        ),
        "probe_train": DataLoader(probe_train_dataset, batch_size=batch_size, **common),
        "probe_test": DataLoader(probe_test_dataset, batch_size=batch_size, **common),
    }
