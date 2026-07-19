"""Phase 2 model tests: backbone, projector, joint-embedding model, checkpoints."""

import torch

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.models import MLPProjector, build_backbone, build_model
from jepa_collapse_lab.utils import load_checkpoint, save_checkpoint, set_seed

CFG = load_config("configs/stl10.yaml")


def test_backbone_output_shape_stl10():
    backbone, out_dim = build_backbone("resnet18")
    h = backbone(torch.randn(2, 3, 96, 96))
    assert h.shape == (2, 512)
    assert out_dim == 512


def test_backbone_output_shape_cifar():
    backbone, _ = build_backbone("resnet18")
    h = backbone(torch.randn(2, 3, 32, 32))
    assert h.shape == (2, 512)


def test_backbone_rejects_unknown_name():
    try:
        build_backbone("resnet50")
    except ValueError as e:
        assert "resnet50" in str(e)
    else:
        raise AssertionError("build_backbone should reject unknown backbones")


def test_projector_output_shape():
    projector = MLPProjector(input_dim=512, hidden_dims=(256, 256), output_dim=128)
    z = projector(torch.randn(4, 512))
    assert z.shape == (4, 128)


def test_ssl_model_returns_features_and_projection():
    model = build_model(CFG)
    h, z = model(torch.randn(2, 3, 96, 96))
    assert h.shape == (2, 512)
    assert z.shape == (2, CFG["model"]["projector_output"])


def test_gradients_flow_to_backbone():
    model = build_model(CFG)
    _, z = model(torch.randn(2, 3, 96, 96))
    z.sum().backward()
    grad = model.backbone.conv1.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0


def test_reproducible_init():
    set_seed(0)
    model_a = build_model(CFG)
    set_seed(0)
    model_b = build_model(CFG)
    for p_a, p_b in zip(model_a.state_dict().values(), model_b.state_dict().values(), strict=True):
        assert torch.equal(p_a, p_b)
    set_seed(1)
    model_c = build_model(CFG)
    assert not torch.equal(
        model_a.state_dict()["backbone.conv1.weight"],
        model_c.state_dict()["backbone.conv1.weight"],
    )


def test_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    model = build_model(CFG).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 3, 96, 96)

    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer=optimizer, epoch=3, config=CFG)

    set_seed(1)
    model_b = build_model(CFG).eval()
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
    ckpt = load_checkpoint(path, model_b, optimizer_b)

    assert ckpt["epoch"] == 3
    assert ckpt["config"]["model"]["backbone"] == "resnet18"
    with torch.no_grad():
        h_a, z_a = model(x)
        h_b, z_b = model_b(x)
    assert torch.allclose(h_a, h_b)
    assert torch.allclose(z_a, z_b)
