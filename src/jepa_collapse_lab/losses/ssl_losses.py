"""The three SSL objectives compared in this lab.

All losses share the same interface: ``loss_fn(z_a, z_b) -> (loss, terms)`` where
``terms`` is a dict of per-term values for logging. Only the loss changes between
experiments A, B and C — backbone, projector, augmentations and budget are identical.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

EXPERIMENTS = ("naive", "barlow_twins", "vicreg")


def off_diagonal(x: Tensor) -> Tensor:
    """Flattened view of the off-diagonal elements of a square matrix."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def naive_invariance_loss(z_a: Tensor, z_b: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Experiment A: pure invariance. A constant output is a perfect (degenerate) solution."""
    loss = F.mse_loss(z_a, z_b)
    return loss, {"inv": loss.item()}


def barlow_twins_loss(
    z_a: Tensor,
    z_b: Tensor,
    lambda_offdiag: float = 5e-3,
    eps: float = 1e-5,
) -> tuple[Tensor, dict[str, float]]:
    """Experiment B: push the batch cross-correlation matrix toward the identity.

    On-diagonal terms enforce invariance between views, off-diagonal terms
    decorrelate the embedding dimensions (redundancy reduction).
    """
    n, _ = z_a.shape
    z_a = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0, unbiased=False) + eps)
    z_b = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0, unbiased=False) + eps)
    cross_corr = (z_a.T @ z_b) / n

    on_diag = (cross_corr.diagonal() - 1).pow(2).sum()
    off_diag = off_diagonal(cross_corr).pow(2).sum()
    loss = on_diag + lambda_offdiag * off_diag
    return loss, {"inv": on_diag.item(), "offdiag": off_diag.item()}


def _variance_term(z: Tensor, gamma: float, eps: float) -> Tensor:
    """Hinge loss keeping each dimension's std above the target gamma."""
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(gamma - std).mean()


def _covariance_term(z: Tensor) -> Tensor:
    """Squared off-diagonal covariance: dimensions must not copy each other."""
    n, d = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (n - 1)
    return off_diagonal(cov).pow(2).sum() / d


def vicreg_loss(
    z_a: Tensor,
    z_b: Tensor,
    lambda_inv: float = 25.0,
    mu_var: float = 25.0,
    nu_cov: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[Tensor, dict[str, float]]:
    """Experiment C: invariance + variance hinge + covariance penalty."""
    inv = F.mse_loss(z_a, z_b)
    var = _variance_term(z_a, gamma, eps) + _variance_term(z_b, gamma, eps)
    cov = _covariance_term(z_a) + _covariance_term(z_b)
    loss = lambda_inv * inv + mu_var * var + nu_cov * cov
    return loss, {"inv": inv.item(), "var": var.item(), "cov": cov.item()}


def build_loss(cfg: dict[str, Any]) -> Callable[[Tensor, Tensor], tuple[Tensor, dict[str, float]]]:
    """Return the loss function selected by ``cfg['experiment']['name']``."""
    name = cfg["experiment"]["name"]
    p = cfg.get("loss", {})
    if name == "naive":
        return naive_invariance_loss
    if name == "barlow_twins":
        return partial(barlow_twins_loss, lambda_offdiag=p.get("lambda_offdiag", 5e-3))
    if name == "vicreg":
        return partial(
            vicreg_loss,
            lambda_inv=p.get("lambda_inv", 25.0),
            mu_var=p.get("mu_var", 25.0),
            nu_cov=p.get("nu_cov", 1.0),
            gamma=p.get("gamma", 1.0),
        )
    raise ValueError(f"Unknown experiment: {name!r} (available: {list(EXPERIMENTS)})")
