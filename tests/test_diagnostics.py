"""Phase 4 diagnostics: metrics, collection, plots."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from jepa_collapse_lab.diagnostics import (
    collect_embeddings,
    correlation_matrix,
    covariance_matrix,
    effective_rank,
    mean_per_dim_std,
    per_dim_std,
    run_all_diagnostics,
    select_embeddings,
    singular_spectrum,
    summarize_embeddings,
)
from jepa_collapse_lab.diagnostics.plots import (
    plot_covariance_heatmap,
    plot_per_dim_std,
    plot_projection,
    plot_singular_spectrum,
    plot_training_curves,
)
from jepa_collapse_lab.models import build_model


def _full_rank_batch(n: int = 200, d: int = 16, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def _collapsed_batch(n: int = 200, d: int = 16) -> torch.Tensor:
    """Constant embeddings — the naive-invariance fixed point (rank 0 after centering)."""
    return torch.ones(n, d) * 0.5


def test_per_dim_std_shape_and_collapse():
    z = _full_rank_batch()
    std = per_dim_std(z)
    assert std.shape == (16,)
    assert mean_per_dim_std(_collapsed_batch()) < 0.01
    assert mean_per_dim_std(z) > 0.5


def test_covariance_and_correlation_shapes():
    z = _full_rank_batch(d=8)
    cov = covariance_matrix(z)
    corr = correlation_matrix(z)
    assert cov.shape == (8, 8)
    assert corr.shape == (8, 8)
    assert torch.allclose(corr.diag(), torch.ones(8), atol=1e-4)


def test_singular_spectrum_descending():
    z = _full_rank_batch()
    sigma = singular_spectrum(z)
    assert sigma.ndim == 1
    assert torch.all(sigma[:-1] >= sigma[1:] - 1e-5)


def test_effective_rank_collapse_vs_healthy():
    healthy = effective_rank(_full_rank_batch(n=400, d=32))
    collapsed = effective_rank(_collapsed_batch(n=400, d=32))
    # Rank-1 line in embedding space (all rows equal up to a scalar direction).
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(1, 32, generator=g)
    rank1 = torch.randn(400, 1, generator=g) @ direction
    assert healthy > 10
    assert collapsed < 1.5
    assert effective_rank(rank1) < 2.5


def test_summarize_embeddings_keys():
    s = summarize_embeddings(_full_rank_batch())
    assert {"mean_std", "effective_rank", "n_samples", "dim"} <= s.keys()


def test_summarize_embeddings_gates_erank_when_collapsed():
    healthy = summarize_embeddings(_full_rank_batch())
    assert healthy["effective_rank"] is not None
    assert healthy["collapsed"] is False
    collapsed = summarize_embeddings(_collapsed_batch())
    assert collapsed["effective_rank"] is None  # noise-dominated → not reported
    assert collapsed["collapsed"] is True


def test_select_embeddings():
    bundle = {"h": torch.randn(4, 8), "z": torch.randn(4, 4), "y": torch.arange(4)}
    z, y = select_embeddings(bundle, "z")
    assert z.shape == (4, 4)
    assert y.shape == (4,)


def test_collect_embeddings_on_fake_model():
    cfg = {
        "model": {
            "backbone": "resnet18",
            "projector_hidden": [64],
            "projector_output": 16,
        }
    }
    model = build_model(cfg).eval()
    x = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    bundle = collect_embeddings(
        model, loader, torch.device("cpu"), max_samples=12, show_progress=False
    )
    assert bundle["h"].shape == (12, 512)
    assert bundle["z"].shape == (12, 16)
    assert bundle["y"].shape == (12,)


def test_plots_write_files(tmp_path: Path):
    z = _full_rank_batch(n=100, d=16)
    y = torch.randint(0, 5, (100,))
    assert plot_per_dim_std(z, tmp_path / "std.png").is_file()
    assert plot_covariance_heatmap(z, tmp_path / "cov.png").is_file()
    assert plot_singular_spectrum(z, tmp_path / "spec.png").is_file()
    assert plot_projection(z, y, tmp_path / "pca.png", method="pca").is_file()
    history = [{"epoch": 1, "loss": 1.0, "z_std": 0.5}, {"epoch": 2, "loss": 0.5, "z_std": 0.1}]
    assert plot_training_curves(history, tmp_path / "curves.png").is_file()


def test_run_all_diagnostics(tmp_path: Path):
    z = _full_rank_batch(n=80, d=12)
    y = torch.randint(0, 3, (80,))
    summary = run_all_diagnostics(z, y, tmp_path, prefix="z", make_umap=False)
    assert (tmp_path / "z_summary.json").is_file()
    data = json.loads((tmp_path / "z_summary.json").read_text())
    assert "effective_rank" in data
    assert (tmp_path / "z_per_dim_std.png").is_file()
    assert (tmp_path / "z_pca.png").is_file()
    assert "figures" in summary
