"""The joint-embedding model: shared backbone + projector.

`forward(x)` returns both:
- `h`: the backbone feature (used later by the linear probe),
- `z`: the projection (used by the SSL losses).
"""

from typing import Any

from torch import Tensor, nn

from .backbone import build_backbone
from .projector import MLPProjector


class SSLModel(nn.Module):
    def __init__(self, backbone: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.backbone(x)
        z = self.projector(h)
        return h, z


def build_model(cfg: dict[str, Any]) -> SSLModel:
    """Build the SSLModel from an experiment config."""
    backbone, feature_dim = build_backbone(cfg["model"]["backbone"])
    projector = MLPProjector(
        input_dim=feature_dim,
        hidden_dims=tuple(cfg["model"]["projector_hidden"]),
        output_dim=cfg["model"]["projector_output"],
    )
    return SSLModel(backbone, projector)
