"""Phase 3 loss tests: naive, Barlow Twins, VICReg, build_loss."""

import pytest
import torch

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.losses import (
    EXPERIMENTS,
    barlow_twins_loss,
    build_loss,
    naive_invariance_loss,
    off_diagonal,
    vicreg_loss,
)


def _rand_batch(n: int = 32, d: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    z_a = torch.randn(n, d, generator=g)
    z_b = z_a + 0.1 * torch.randn(n, d, generator=g)
    return z_a, z_b


def test_off_diagonal_excludes_diagonal():
    x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    # [[0,1,2],[3,4,5],[6,7,8]] → off-diag 1,2,3,5,6,7
    assert torch.equal(off_diagonal(x), torch.tensor([1.0, 2.0, 3.0, 5.0, 6.0, 7.0]))


def test_naive_invariance_zero_when_identical():
    z = torch.randn(16, 8)
    loss, terms = naive_invariance_loss(z, z.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert "inv" in terms


def test_naive_invariance_positive_when_different():
    z_a, z_b = _rand_batch()
    z_a = z_a.requires_grad_(True)
    loss, _ = naive_invariance_loss(z_a, z_b)
    assert loss.item() > 0
    assert loss.requires_grad


def test_barlow_twins_identity_is_low():
    """Two identical non-constant batches → cross-corr ≈ I → low loss."""
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, 8, generator=g)
    loss, terms = barlow_twins_loss(z, z.clone())
    assert loss.item() < 0.1
    assert set(terms) == {"inv", "offdiag"}


def test_barlow_twins_constant_is_high():
    """Constant embeddings cannot satisfy unit diagonal → high on-diag term."""
    z = torch.ones(32, 8)
    loss, terms = barlow_twins_loss(z, z.clone())
    assert not torch.isfinite(loss) or terms["inv"] > 1.0 or loss.item() > 1.0


def test_vicreg_terms_present_and_finite():
    z_a, z_b = _rand_batch(n=64, d=16)
    z_a = z_a.requires_grad_(True)
    loss, terms = vicreg_loss(z_a, z_b)
    assert torch.isfinite(loss)
    assert set(terms) == {"inv", "var", "cov"}
    assert loss.requires_grad


def test_vicreg_variance_hinge_on_collapsed():
    """Near-constant batch → variance hinge fires."""
    z = torch.ones(32, 8) + 1e-6 * torch.randn(32, 8)
    _, terms = vicreg_loss(z, z.clone(), lambda_inv=0.0, mu_var=1.0, nu_cov=0.0, gamma=1.0)
    assert terms["var"] > 0.5


def test_build_loss_selects_experiment():
    cfg = load_config("configs/cifar10_debug.yaml")
    for name in EXPERIMENTS:
        cfg["experiment"]["name"] = name
        fn = build_loss(cfg)
        z_a, z_b = _rand_batch()
        loss, terms = fn(z_a, z_b)
        assert torch.isfinite(loss)
        assert isinstance(terms, dict)


def test_build_loss_rejects_unknown():
    cfg = load_config("configs/cifar10_debug.yaml")
    cfg["experiment"]["name"] = "simclr"
    with pytest.raises(ValueError, match="simclr"):
        build_loss(cfg)
