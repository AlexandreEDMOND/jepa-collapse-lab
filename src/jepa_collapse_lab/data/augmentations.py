"""Two-view SSL augmentations (SimCLR-style) and deterministic eval transform."""

from collections.abc import Callable

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TwoViewTransform:
    """Apply the same stochastic transform twice → two correlated views of one image."""

    def __init__(self, transform: Callable) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(image), self.transform(image)


def _gaussian_blur(image_size: int, p: float) -> transforms.RandomApply:
    # Odd kernel ≈ 10 % of the image size (9 for 96×96, 3 for 32×32).
    kernel_size = max(3, (image_size // 10) | 1)
    return transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=p
    )


def build_ssl_transform(
    image_size: int,
    crop_scale: tuple[float, float] = (0.3, 1.0),
    jitter_strength: float = 0.4,
    grayscale_p: float = 0.2,
    blur_p: float = 0.5,
) -> transforms.Compose:
    """Stochastic augmentation used to build both views during SSL pretraining."""
    s = jitter_strength
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=crop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)], p=0.8
            ),
            transforms.RandomGrayscale(p=grayscale_p),
            _gaussian_blur(image_size, blur_p),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int) -> transforms.Compose:
    """Deterministic transform for linear-probe and diagnostics (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
