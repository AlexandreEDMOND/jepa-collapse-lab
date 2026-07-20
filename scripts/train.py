"""Train one SSL experiment (naive / barlow_twins / vicreg).

Examples:
    uv run scripts/train.py --config configs/cifar10_debug.yaml --experiment naive --epochs 3
    uv run scripts/train.py --config configs/stl10.yaml --experiment barlow_twins
"""

import argparse

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.losses import EXPERIMENTS
from jepa_collapse_lab.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/stl10.yaml")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per epoch (smoke / fast debug)",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.experiment is not None:
        cfg["experiment"]["name"] = args.experiment
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.max_batches is not None:
        cfg["training"]["max_batches"] = args.max_batches
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir

    run_dir = train(cfg)
    print(f"Done. Checkpoint and history saved in {run_dir}")


if __name__ == "__main__":
    main()
