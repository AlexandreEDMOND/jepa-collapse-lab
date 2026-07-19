"""Show augmented pairs side by side: original | view A | view B.

This is the Phase-1 sanity check: every downstream conclusion rests on the
augmentations being correct, so look at them before writing any loss.

Example:
    uv run scripts/visualize_pairs.py --config configs/cifar10_debug.yaml
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.data.augmentations import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_ssl_transform,
)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def get_raw_dataset(cfg: dict) -> datasets.VisionDataset:
    """Dataset without transform, to access the original PIL images."""
    name, root = cfg["dataset"]["name"], cfg["dataset"]["root"]
    if name == "stl10":
        return datasets.STL10(root, split="unlabeled", download=True)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=True, download=True)
    raise ValueError(f"Unknown dataset: {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stl10.yaml")
    parser.add_argument("--num-pairs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output path (default: results/figures/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or f"results/figures/augmented_pairs_{cfg['dataset']['name']}.png")

    transform = build_ssl_transform(
        image_size=cfg["dataset"]["image_size"],
        crop_scale=tuple(cfg["augmentation"]["crop_scale"]),
        jitter_strength=cfg["augmentation"]["jitter_strength"],
        grayscale_p=cfg["augmentation"]["grayscale_p"],
        blur_p=cfg["augmentation"]["blur_p"],
    )
    dataset = get_raw_dataset(cfg)

    torch.manual_seed(args.seed)
    indices = torch.randperm(len(dataset))[: args.num_pairs]

    fig, axes = plt.subplots(args.num_pairs, 3, figsize=(6, 2 * args.num_pairs), squeeze=False)
    for row, idx in enumerate(indices):
        image, _ = dataset[int(idx)]
        view_a, view_b = transform(image), transform(image)
        for col, (title, img) in enumerate(
            zip(("Original", "View A", "View B"), (image, view_a, view_b), strict=True)
        ):
            ax = axes[row, col]
            ax.imshow(img if col == 0 else denormalize(img).permute(1, 2, 0))
            ax.set_axis_off()
            if row == 0:
                ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {args.num_pairs} augmented pairs to {out}")


if __name__ == "__main__":
    main()
