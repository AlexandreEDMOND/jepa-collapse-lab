"""One figure per collapse diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from .metrics import (
    COLLAPSE_STD_FLOOR,
    correlation_matrix,
    effective_rank,
    mean_per_dim_std,
    per_dim_std,
    singular_spectrum,
    summarize_embeddings,
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_dim_std(z: Tensor, path: str | Path, *, title: str | None = None) -> Path:
    """Bar chart of per-dimension std — flat near zero = collapse."""
    std = per_dim_std(z).numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(std)), std, width=1.0, color="steelblue", edgecolor="none")
    ax.axhline(std.mean(), color="crimson", ls="--", lw=1, label=f"mean={std.mean():.3f}")
    ax.set_xlabel("dimension")
    ax.set_ylabel("std")
    ax.set_title(title or "Per-dimension embedding std")
    ax.legend(loc="upper right")
    ax.set_xlim(-0.5, len(std) - 0.5)
    return _save(fig, Path(path))


def plot_covariance_heatmap(
    z: Tensor,
    path: str | Path,
    *,
    title: str | None = None,
    max_dims: int = 64,
    correlation: bool = True,
) -> Path:
    """Heatmap of dim×dim correlation (default) or covariance."""
    from .metrics import covariance_matrix

    mat = correlation_matrix(z) if correlation else covariance_matrix(z)
    m = mat.numpy()
    if m.shape[0] > max_dims:
        m = m[:max_dims, :max_dims]
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = 1.0 if correlation else np.percentile(np.abs(m), 99)
    im = ax.imshow(m, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    kind = "correlation" if correlation else "covariance"
    ax.set_title(title or f"Embedding {kind} heatmap")
    ax.set_xlabel("dimension")
    ax.set_ylabel("dimension")
    return _save(fig, Path(path))


def plot_singular_spectrum(z: Tensor, path: str | Path, *, title: str | None = None) -> Path:
    """Singular values of the centered embedding matrix (log scale)."""
    sigma = singular_spectrum(z).numpy()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(np.arange(1, len(sigma) + 1), sigma + 1e-12, color="darkorange", lw=1.5)
    ax.set_xlabel("component index")
    ax.set_ylabel("singular value")
    if mean_per_dim_std(z) < COLLAPSE_STD_FLOOR:
        erank_label = "erank=n/a (collapsed)"
    else:
        erank_label = f"erank={effective_rank(z):.1f}"
    ax.set_title(title or f"Singular spectrum  ({erank_label})")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    return _save(fig, Path(path))


def plot_projection(
    z: Tensor,
    y: Tensor,
    path: str | Path,
    *,
    method: str = "pca",
    title: str | None = None,
    max_points: int = 4000,
    seed: int = 0,
) -> Path:
    """2D PCA or UMAP projection colored by true label."""
    z_np = z.numpy()
    y_np = y.numpy()
    if len(z_np) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(z_np), size=max_points, replace=False)
        z_np, y_np = z_np[idx], y_np[idx]

    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=seed).fit_transform(z_np)
        label = "PCA"
    elif method == "umap":
        import umap

        reducer = umap.UMAP(
            n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1
        )
        xy = reducer.fit_transform(z_np)
        label = "UMAP"
    else:
        raise ValueError(f"Unknown projection method: {method!r} (expected 'pca' or 'umap')")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    classes = np.unique(y_np)
    cmap = plt.get_cmap("tab10", max(len(classes), 10))
    for i, c in enumerate(classes):
        mask = y_np == c
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.7, color=cmap(i), label=str(int(c)))
    ax.set_title(title or f"{label} of embeddings (colored by label)")
    ax.set_xticks([])
    ax.set_yticks([])
    if len(classes) <= 12:
        ax.legend(markerscale=2, fontsize=7, frameon=False, loc="best")
    return _save(fig, Path(path))


def plot_training_curves(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Plot loss and z_std (and optional erank) from a training history.json."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    ax = axes[0]
    if "loss" in history[0]:
        ax.plot(epochs, [h["loss"] for h in history], color="steelblue")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    ax.grid(True, ls=":", alpha=0.5)

    ax = axes[1]
    if "z_std" in history[0]:
        ax.plot(epochs, [h["z_std"] for h in history], color="crimson", label="z_std")
    if "effective_rank" in history[0]:
        ax.plot(epochs, [h["effective_rank"] for h in history], color="seagreen", label="erank")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Collapse signals over training")
    ax.grid(True, ls=":", alpha=0.5)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False)

    fig.tight_layout()
    return _save(fig, Path(path))


def plot_metric_over_checkpoints(
    values: list[tuple[int, float]],
    path: str | Path,
    *,
    ylabel: str,
    title: str | None = None,
) -> Path:
    """Generic epoch→metric curve (e.g. erank or mean std across saved checkpoints)."""
    epochs, ys = zip(*values, strict=True) if values else ([], [])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(epochs, ys, marker="o", color="purple")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.grid(True, ls=":", alpha=0.5)
    return _save(fig, Path(path))


def run_all_diagnostics(
    z: Tensor,
    y: Tensor,
    out_dir: str | Path,
    *,
    prefix: str = "",
    history: list[dict[str, Any]] | None = None,
    make_umap: bool = True,
) -> dict[str, Any]:
    """Write every Phase-4 figure for one embedding matrix. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    paths = {
        "per_dim_std": str(plot_per_dim_std(z, out_dir / f"{p}per_dim_std.png")),
        "covariance": str(plot_covariance_heatmap(z, out_dir / f"{p}covariance.png")),
        "spectrum": str(plot_singular_spectrum(z, out_dir / f"{p}spectrum.png")),
        "pca": str(plot_projection(z, y, out_dir / f"{p}pca.png", method="pca")),
    }
    if make_umap:
        paths["umap"] = str(plot_projection(z, y, out_dir / f"{p}umap.png", method="umap"))
    if history:
        paths["training_curves"] = str(
            plot_training_curves(history, out_dir / f"{p}training_curves.png")
        )

    summary = summarize_embeddings(z)
    summary["mean_std"] = mean_per_dim_std(z)
    (out_dir / f"{p}summary.json").write_text(json.dumps(summary, indent=2))
    summary["figures"] = paths
    return summary
