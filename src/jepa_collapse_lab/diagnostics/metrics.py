"""Scalar and matrix collapse metrics on an embedding matrix Z (N, D)."""

import torch
from torch import Tensor


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


def summarize_embeddings(z: Tensor) -> dict[str, float]:
    """Compact dict of scalar collapse diagnostics for logging / tables."""
    std = per_dim_std(z)
    return {
        "mean_std": std.mean().item(),
        "min_std": std.min().item(),
        "max_std": std.max().item(),
        "effective_rank": effective_rank(z),
        "n_samples": float(z.shape[0]),
        "dim": float(z.shape[1]),
    }
