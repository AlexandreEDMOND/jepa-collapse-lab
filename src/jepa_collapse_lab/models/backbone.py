"""ResNet-18 backbone adapted for small images (STL-10 96x96, CIFAR-10 32x32).

The standard small-image variant (SimCLR / Barlow Twins style): the 7x7 stride-2
stem is replaced by a 3x3 stride-1 conv and the maxpool is removed, so the spatial
resolution is preserved. The classifier head is dropped; the backbone outputs the
512-d pooled feature `h` used by the linear probe.
"""

from torch import nn
from torchvision.models import resnet18

BACKBONE_DIMS = {"resnet18": 512}


def build_backbone(name: str = "resnet18") -> tuple[nn.Module, int]:
    """Build a backbone without its classifier head. Returns (module, feature_dim)."""
    if name != "resnet18":
        raise ValueError(f"Unknown backbone: {name!r} (available: {sorted(BACKBONE_DIMS)})")
    backbone = resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone, BACKBONE_DIMS[name]
