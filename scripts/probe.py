"""Frozen linear probe on a training checkpoint.

Examples:
    uv run scripts/probe.py --checkpoint results/checkpoints/cifar10_naive/last.pt
    uv run scripts/probe.py --all --checkpoints-root results/checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_collapse_lab.eval import evaluate_checkpoint


def _find_checkpoints(root: Path) -> list[Path]:
    return sorted(root.glob("*/last.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Probe every */last.pt under --checkpoints-root",
    )
    parser.add_argument("--checkpoints-root", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--config", default=None, help="Override embedded checkpoint config")
    parser.add_argument("--space", choices=("h", "z"), default="h")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.all:
        ckpts = _find_checkpoints(args.checkpoints_root)
        if not ckpts:
            raise SystemExit(f"No last.pt found under {args.checkpoints_root}")
        rows = []
        for ckpt in ckpts:
            print(f"\n=== {ckpt} ===")
            result = evaluate_checkpoint(
                ckpt,
                config=args.config,
                space=args.space,
                max_samples=args.max_samples,
                max_iter=args.max_iter,
                C=args.C,
                seed=args.seed,
                out_dir=args.out / ckpt.parent.name if args.out else None,
            )
            acc = result["accuracy"]
            print(f"accuracy={acc:.4f}  train={result['train_accuracy']:.4f}  → {result['output']}")
            rows.append(
                {
                    "run": ckpt.parent.name,
                    "experiment": result.get("experiment"),
                    "accuracy": acc,
                    "train_accuracy": result["train_accuracy"],
                    "space": result["space"],
                }
            )
        table_path = (args.out or args.checkpoints_root) / "probe_table.json"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(json.dumps(rows, indent=2))
        print("\n=== summary ===")
        for row in rows:
            print(f"{row['run']:30s}  acc={row['accuracy']:.4f}")
        print(f"Table → {table_path}")
        return

    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint PATH or --all")

    result = evaluate_checkpoint(
        args.checkpoint,
        config=args.config,
        space=args.space,
        max_samples=args.max_samples,
        max_iter=args.max_iter,
        C=args.C,
        seed=args.seed,
        out_dir=args.out,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_matrix"}, indent=2))
    print(f"accuracy={result['accuracy']:.4f}  → {result['output']}")


if __name__ == "__main__":
    main()
