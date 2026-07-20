"""Frozen-backbone linear evaluation (logistic regression on `h`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ..config import load_config
from ..data import build_loaders
from ..diagnostics.collect import collect_embeddings, load_model_from_checkpoint
from ..utils import get_device


def fit_linear_probe(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_test: np.ndarray,
    y_test: np.ndarray,
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Standardize features, fit multinomial logistic regression, score test set."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(h_train)
    x_test = scaler.transform(h_test)

    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    classes = np.unique(np.concatenate([y_train, y_test]))
    report = classification_report(
        y_test, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    per_class_recall = {
        str(int(c)): float(report[str(int(c))]["recall"])
        for c in classes
        if str(int(c)) in report
    }

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "train_accuracy": float(accuracy_score(y_train, clf.predict(x_train))),
        "per_class_recall": per_class_recall,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(classes)),
        "feature_dim": int(h_train.shape[1]),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=classes).tolist(),
    }


def _to_numpy(bundle: dict[str, torch.Tensor], space: str = "h") -> tuple[np.ndarray, np.ndarray]:
    return bundle[space].numpy(), bundle["y"].numpy()


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    config: str | None = None,
    space: str = "h",
    max_samples: int | None = None,
    max_iter: int = 1000,
    C: float = 1.0,
    seed: int = 0,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint, extract frozen features, fit a linear probe, return metrics."""
    if space not in {"h", "z"}:
        raise ValueError(f"Unknown probe space: {space!r} (expected 'h' or 'z')")

    device = get_device()
    model, ckpt = load_model_from_checkpoint(str(checkpoint), device=device)
    cfg = load_config(config) if config is not None else ckpt["config"]

    loader_cfg = {**cfg.get("loader", {})}
    loader_cfg.setdefault("num_workers", 0)
    cfg = {**cfg, "loader": loader_cfg}
    loaders = build_loaders(cfg)

    train_bundle = collect_embeddings(
        model, loaders["probe_train"], device, max_samples=max_samples
    )
    test_bundle = collect_embeddings(
        model, loaders["probe_test"], device, max_samples=max_samples
    )
    h_train, y_train = _to_numpy(train_bundle, space)
    h_test, y_test = _to_numpy(test_bundle, space)

    results = fit_linear_probe(
        h_train, y_train, h_test, y_test, max_iter=max_iter, C=C, seed=seed
    )
    results.update(
        {
            "checkpoint": str(checkpoint),
            "space": space,
            "experiment": cfg.get("experiment", {}).get("name"),
            "dataset": cfg.get("dataset", {}).get("name"),
            "device": str(device),
        }
    )

    if out_dir is None:
        out_dir = Path(checkpoint).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_{space}.json"
    out_path.write_text(json.dumps(results, indent=2))
    results["output"] = str(out_path)
    return results


def evaluate_loaders(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    space: str = "h",
    max_samples: int | None = None,
    max_iter: int = 1000,
    C: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Probe without loading a checkpoint (handy for tests)."""
    train_bundle = collect_embeddings(
        model, train_loader, device, max_samples=max_samples, show_progress=False
    )
    test_bundle = collect_embeddings(
        model, test_loader, device, max_samples=max_samples, show_progress=False
    )
    h_train, y_train = _to_numpy(train_bundle, space)
    h_test, y_test = _to_numpy(test_bundle, space)
    return fit_linear_probe(
        h_train, y_train, h_test, y_test, max_iter=max_iter, C=C, seed=seed
    )
