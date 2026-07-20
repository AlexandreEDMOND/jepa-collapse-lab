"""Phase 5 linear probe tests."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jepa_collapse_lab.eval import evaluate_loaders, fit_linear_probe
from jepa_collapse_lab.models import build_model
from jepa_collapse_lab.utils import save_checkpoint, set_seed


def test_fit_linear_probe_separable():
    rng = np.random.default_rng(0)
    # 2 well-separated blobs → near-perfect accuracy.
    h0 = rng.normal(0.0, 0.3, size=(80, 8))
    h1 = rng.normal(3.0, 0.3, size=(80, 8))
    h_train = np.vstack([h0[:60], h1[:60]])
    y_train = np.array([0] * 60 + [1] * 60)
    h_test = np.vstack([h0[60:], h1[60:]])
    y_test = np.array([0] * 20 + [1] * 20)
    result = fit_linear_probe(h_train, y_train, h_test, y_test, seed=0)
    assert result["accuracy"] >= 0.95
    assert result["n_classes"] == 2
    assert result["feature_dim"] == 8
    assert "0" in result["per_class_recall"]


def test_fit_linear_probe_chance_on_noise():
    rng = np.random.default_rng(1)
    h_train = rng.normal(size=(200, 16))
    y_train = rng.integers(0, 10, size=200)
    h_test = rng.normal(size=(100, 16))
    y_test = rng.integers(0, 10, size=100)
    result = fit_linear_probe(h_train, y_train, h_test, y_test, seed=0)
    # Pure noise: should not be dramatically above chance for long.
    assert 0.0 <= result["accuracy"] <= 0.35


def test_evaluate_loaders_on_tiny_model(tmp_path: Path):
    set_seed(0)
    cfg = {
        "model": {
            "backbone": "resnet18",
            "projector_hidden": [64],
            "projector_output": 16,
        }
    }
    model = build_model(cfg).eval()
    # Synthetic labeled batches.
    n, c = 40, 5
    x = torch.randn(n, 3, 32, 32)
    y = torch.arange(n) % c
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    result = evaluate_loaders(
        model, loader, loader, torch.device("cpu"), space="h", max_samples=32, max_iter=200
    )
    assert "accuracy" in result
    assert result["feature_dim"] == 512
    assert 0.0 <= result["accuracy"] <= 1.0

    # Checkpoint round-trip path used by the CLI.
    ckpt = save_checkpoint(
        tmp_path / "last.pt",
        model,
        epoch=1,
        config={
            **cfg,
            "dataset": {"name": "cifar10", "root": "data", "image_size": 32},
            "experiment": {"name": "naive"},
            "loader": {"batch_size": 8, "num_workers": 0},
            "augmentation": {},
            "seed": 0,
        },
    )
    assert ckpt.is_file()
