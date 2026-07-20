"""Losses: naive invariance, Barlow Twins, VICReg."""

from .ssl_losses import (
    EXPERIMENTS,
    barlow_twins_loss,
    build_loss,
    naive_invariance_loss,
    off_diagonal,
    vicreg_loss,
)

__all__ = [
    "EXPERIMENTS",
    "barlow_twins_loss",
    "build_loss",
    "naive_invariance_loss",
    "off_diagonal",
    "vicreg_loss",
]
