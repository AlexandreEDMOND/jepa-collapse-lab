"""Reproducibility and checkpoint utilities."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch for reproducible runs (incl. weight init)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save model (+ optional optimizer state, epoch and config) to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt: dict[str, Any] = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if config is not None:
        ckpt["config"] = config
    torch.save(ckpt, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into `model` (and `optimizer` if given). Returns the raw dict.

    Checkpoints are produced by this project, so `weights_only=False` is safe here.
    """
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
