"""Phase 3 trainer smoke test: one short epoch on FakeData."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.data import TwoViewTransform, build_ssl_transform
from jepa_collapse_lab.losses import build_loss
from jepa_collapse_lab.models import build_model
from jepa_collapse_lab.trainer import train, train_epoch
from jepa_collapse_lab.utils import get_device, set_seed


def _tiny_cfg(tmp_path: Path, experiment: str = "naive") -> dict:
    cfg = load_config("configs/cifar10_debug.yaml")
    cfg["experiment"]["name"] = experiment
    cfg["model"]["projector_hidden"] = [64]
    cfg["model"]["projector_output"] = 16
    cfg["training"] = {
        "epochs": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "output_dir": str(tmp_path / "ckpts"),
    }
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["num_workers"] = 0
    cfg["dataset"]["image_size"] = 32
    return cfg


def _fake_ssl_loader(image_size: int = 32, batch_size: int = 8, n: int = 16) -> DataLoader:
    ds = datasets.FakeData(
        size=n,
        image_size=(3, image_size, image_size),
        transform=TwoViewTransform(build_ssl_transform(image_size)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_train_epoch_returns_metrics():
    set_seed(0)
    cfg = _tiny_cfg(Path("/tmp"))
    model = build_model(cfg)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = _fake_ssl_loader()
    metrics = train_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    assert "loss" in metrics and "z_std" in metrics and "inv" in metrics
    assert metrics["loss"] > 0
    assert metrics["z_std"] > 0


def test_train_writes_checkpoint_and_history(tmp_path):
    cfg = _tiny_cfg(tmp_path, experiment="vicreg")
    fake_loaders = {"ssl": _fake_ssl_loader()}
    with patch("jepa_collapse_lab.trainer.build_loaders", return_value=fake_loaders):
        run_dir = train(cfg)
    assert (run_dir / "last.pt").is_file()
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history) == 1
    assert history[0]["epoch"] == 1
    assert "loss" in history[0]
    ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1
    assert ckpt["config"]["experiment"]["name"] == "vicreg"
