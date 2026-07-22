"""Collect backbone features `h` and projections `z` on a fixed eval set."""

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> dict[str, Tensor]:
    """Run `model` on `loader` and stack embeddings.

    Returns a dict with:
    - ``h``: (N, D_h) backbone features (linear-probe space)
    - ``z``: (N, D_z) projector outputs (loss space)
    - ``y``: (N,) integer labels
    """
    model.eval()
    hs: list[Tensor] = []
    zs: list[Tensor] = []
    ys: list[Tensor] = []
    n = 0
    iterator = tqdm(loader, desc="collect", leave=False) if show_progress else loader
    for batch in iterator:
        # SSL loaders yield ((view_a, view_b), y); eval loaders yield (x, y).
        images, targets = batch
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        h, z = model(images)
        hs.append(h.cpu())
        zs.append(z.cpu())
        ys.append(targets.cpu() if isinstance(targets, Tensor) else torch.as_tensor(targets))
        n += images.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    h_all = torch.cat(hs, dim=0)
    z_all = torch.cat(zs, dim=0)
    y_all = torch.cat(ys, dim=0)
    if max_samples is not None:
        h_all = h_all[:max_samples]
        z_all = z_all[:max_samples]
        y_all = y_all[:max_samples]
    return {"h": h_all, "z": z_all, "y": y_all}


@torch.no_grad()
def collect_paired_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    max_samples: int | None = None,
    show_progress: bool = True,
) -> dict[str, Tensor]:
    """Run paired SSL views through ``model`` and stack their embeddings.

    The returned ``*_a`` and ``*_b`` tensors contain aligned augmentations of the
    same source images, suitable for cross-correlation diagnostics.
    """
    model.eval()
    embeddings: dict[str, list[Tensor]] = {key: [] for key in ("h_a", "h_b", "z_a", "z_b")}
    n = 0
    iterator = tqdm(loader, desc="collect paired", leave=False) if show_progress else loader
    for (view_a, view_b), _ in iterator:
        view_a, view_b = view_a.to(device), view_b.to(device)
        h_a, z_a = model(view_a)
        h_b, z_b = model(view_b)
        for key, value in (("h_a", h_a), ("h_b", h_b), ("z_a", z_a), ("z_b", z_b)):
            embeddings[key].append(value.cpu())
        n += view_a.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    return {
        key: (
            torch.cat(values, dim=0)[:max_samples]
            if max_samples is not None
            else torch.cat(values, dim=0)
        )
        for key, values in embeddings.items()
    }


def select_embeddings(bundle: dict[str, Tensor], space: str = "z") -> tuple[Tensor, Tensor]:
    """Return ``(embeddings, labels)`` for ``space`` in ``{'h', 'z'}``."""
    if space not in {"h", "z"}:
        raise ValueError(f"Unknown embedding space: {space!r} (expected 'h' or 'z')")
    return bundle[space], bundle["y"]


def load_model_from_checkpoint(
    path: str,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model from a training checkpoint and load weights."""
    from ..models import build_model
    from ..utils import get_device, load_checkpoint

    device = device or get_device()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise ValueError(f"Checkpoint {path} has no embedded config; cannot rebuild the model.")
    model = build_model(ckpt["config"]).to(device)
    load_checkpoint(path, model, map_location=str(device))
    model.eval()
    return model, ckpt
