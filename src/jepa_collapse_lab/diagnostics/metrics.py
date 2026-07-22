"""Scalar and matrix collapse metrics on an embedding matrix Z (N, D)."""

from typing import Any

import torch
from torch import Tensor

# Below this mean per-dim std the embeddings carry no signal: the SVD sees only
# numerical noise, whose near-flat spectrum makes the entropy-based effective rank
# meaningless (a collapsed run can show erank ≈ 37–110). Gate reporting on this floor.
COLLAPSE_STD_FLOOR = 1e-2


def per_dim_std(z: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-dimension standard deviation over the batch (population). Shape: (D,)."""
    return z.std(dim=0, unbiased=False).clamp_min(eps)


def mean_per_dim_std(z: Tensor) -> float:
    """Mean of per-dimension std — collapses to ~0 under naive invariance."""
    return per_dim_std(z).mean().item()


def covariance_matrix(z: Tensor) -> Tensor:
    """Empirical covariance of centered embeddings (population / N). Shape: (D, D)."""
    z_c = z - z.mean(dim=0, keepdim=True)
    n = max(z.shape[0], 1)
    return (z_c.T @ z_c) / n


def correlation_matrix(z: Tensor, eps: float = 1e-12) -> Tensor:
    """Pearson correlation matrix of embedding dimensions. Shape: (D, D)."""
    std = per_dim_std(z, eps=eps)
    cov = covariance_matrix(z)
    denom = std.unsqueeze(1) * std.unsqueeze(0)
    return cov / denom.clamp_min(eps)


def cross_correlation_matrix(z_a: Tensor, z_b: Tensor, eps: float = 1e-5) -> Tensor:
    """Cross-correlation between dimensions of paired embeddings. Shape: (D, D)."""
    if z_a.shape != z_b.shape:
        raise ValueError(
            f"Expected paired embeddings with equal shapes, got {z_a.shape} and {z_b.shape}."
        )
    n = max(z_a.shape[0], 1)
    z_a = (z_a - z_a.mean(dim=0, keepdim=True)) / (
        z_a.std(dim=0, unbiased=False, keepdim=True) + eps
    )
    z_b = (z_b - z_b.mean(dim=0, keepdim=True)) / (
        z_b.std(dim=0, unbiased=False, keepdim=True) + eps
    )
    return (z_a.T @ z_b) / n


def cross_correlation_summary(z_a: Tensor, z_b: Tensor) -> dict[str, float]:
    """Compact Barlow-Twins-style diagnostics for paired embeddings."""
    cross_corr = cross_correlation_matrix(z_a, z_b)
    diagonal_error = (cross_corr.diagonal() - 1).pow(2).mean()
    off_diagonal = cross_corr - torch.diag_embed(cross_corr.diagonal())
    return {
        "diagonal_mean": cross_corr.diagonal().mean().item(),
        "diagonal_error": diagonal_error.item(),
        "off_diagonal_mean_square": (
            off_diagonal.pow(2).sum() / max(off_diagonal.numel() - len(cross_corr), 1)
        ).item(),
        "identity_mean_square_error": (
            cross_corr
            - torch.eye(cross_corr.shape[0], device=cross_corr.device, dtype=cross_corr.dtype)
        )
        .pow(2)
        .mean()
        .item(),
    }


def singular_spectrum(z: Tensor, center: bool = True) -> Tensor:
    """Singular values of the (optionally centered) embedding matrix, descending."""
    if center:
        z = z - z.mean(dim=0, keepdim=True)
    # SVD on (N, D): cheaper via eig of Gram when N or D is large; torch.linalg.svdvals is fine.
    return torch.linalg.svdvals(z)


def effective_rank(z: Tensor, center: bool = True, eps: float = 1e-12) -> float:
    """Entropy-based effective rank: exp(H(p)) with p_i = σ_i / ∑ σ_j.

    Collapsed representations → erank ≈ 1; healthy ones → erank ≫ 1.
    """
    sigma = singular_spectrum(z, center=center).clamp_min(0)
    total = sigma.sum()
    if total <= eps:
        return 0.0
    p = sigma / total
    # Avoid 0 * log(0): mask empty mass.
    p = p[p > eps]
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy).item())


def summarize_embeddings(z: Tensor) -> dict[str, Any]:
    """Compact dict of scalar collapse diagnostics for logging / tables.

    ``effective_rank`` is ``None`` when the embedding is collapsed (mean per-dim std
    below ``COLLAPSE_STD_FLOOR``): the metric is noise-dominated there.
    """
    std = per_dim_std(z)
    mean_std = std.mean().item()
    collapsed = mean_std < COLLAPSE_STD_FLOOR
    return {
        "mean_std": mean_std,
        "min_std": std.min().item(),
        "max_std": std.max().item(),
        "effective_rank": None if collapsed else effective_rank(z),
        "collapsed": collapsed,
        "n_samples": float(z.shape[0]),
        "dim": float(z.shape[1]),
    }
