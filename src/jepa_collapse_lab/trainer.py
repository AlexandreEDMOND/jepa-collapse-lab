"""Config-driven trainer shared by the three experiments.

Same backbone, same projector, same augmentations, same budget: only the loss
changes. Each epoch logs the mean loss, the per-term values, and `z_std` (mean
per-dimension std of the projections) — the quantity that collapses to zero in
experiment A.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_loaders
from .losses import build_loss
from .models import build_model
from .utils import get_device, save_checkpoint, set_seed


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0
    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    for (view_a, view_b), _ in tqdm(loader, desc="train", leave=False, total=total):
        view_a, view_b = view_a.to(device), view_b.to(device)
        _, z_a = model(view_a)
        _, z_b = model(view_b)
        loss, terms = loss_fn(z_a, z_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["loss"] = totals.get("loss", 0.0) + loss.item()
        totals["z_std"] = totals.get("z_std", 0.0) + z_a.std(dim=0, unbiased=False).mean().item()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def train(cfg: dict[str, Any]) -> Path:
    """Train the experiment described by `cfg`. Returns the run directory."""
    set_seed(cfg.get("seed", 0))
    device = get_device()
    loaders = build_loaders(cfg)
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)

    tcfg = cfg["training"]
    max_batches = tcfg.get("max_batches")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )

    run_dir = (
        Path(tcfg.get("output_dir", "checkpoints"))
        / f"{cfg['dataset']['name']}_{cfg['experiment']['name']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  run_dir={run_dir}  max_batches={max_batches}")
    history = []
    for epoch in range(1, tcfg["epochs"] + 1):
        metrics = train_epoch(
            model, loaders["ssl"], loss_fn, optimizer, device, max_batches=max_batches
        )
        history.append({"epoch": epoch, **metrics})
        printable = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"epoch {epoch:3d}/{tcfg['epochs']}  {printable}")
        save_checkpoint(run_dir / "last.pt", model, optimizer=optimizer, epoch=epoch, config=cfg)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    return run_dir
