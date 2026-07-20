"""Run collapse diagnostics on a training checkpoint.

Examples:
    uv run scripts/diagnose.py --checkpoint results/checkpoints/cifar10_naive/last.pt
    uv run scripts/diagnose.py --checkpoint results/checkpoints/cifar10_vicreg/last.pt --space h
    uv run scripts/diagnose.py --all --checkpoints-root results/checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_collapse_lab.config import load_config
from jepa_collapse_lab.data import build_loaders
from jepa_collapse_lab.diagnostics import (
    collect_embeddings,
    load_model_from_checkpoint,
    run_all_diagnostics,
    select_embeddings,
)
from jepa_collapse_lab.utils import get_device


def _resolve_out_dir(checkpoint: Path, out: str | None) -> Path:
    if out is not None:
        return Path(out)
    # results/checkpoints/{run}/last.pt → results/figures/{run}/
    run_name = checkpoint.parent.name
    return Path("results/figures") / run_name


def _load_history(checkpoint: Path) -> list[dict] | None:
    history_path = checkpoint.parent / "history.json"
    if not history_path.is_file():
        return None
    return json.loads(history_path.read_text())


def diagnose_checkpoint(
    checkpoint: Path,
    *,
    config: str | None = None,
    space: str = "z",
    max_samples: int = 2000,
    out_dir: Path | None = None,
    no_umap: bool = False,
) -> dict:
    device = get_device()
    model, ckpt = load_model_from_checkpoint(str(checkpoint), device=device)
    cfg = load_config(config) if config else ckpt["config"]
    # Eval split is labeled and deterministic — collapse metrics + label-colored plots.
    loader_cfg = {**cfg.get("loader", {})}
    loader_cfg.setdefault("num_workers", 0)
    cfg = {**cfg, "loader": loader_cfg}
    loaders = build_loaders(cfg)
    bundle = collect_embeddings(
        model,
        loaders["probe_test"],
        device,
        max_samples=max_samples,
    )
    z, y = select_embeddings(bundle, space=space)
    out = out_dir or _resolve_out_dir(checkpoint, None)
    history = _load_history(checkpoint)
    summary = run_all_diagnostics(
        z,
        y,
        out,
        prefix=space,
        history=history,
        make_umap=not no_umap,
    )
    summary["checkpoint"] = str(checkpoint)
    summary["space"] = space
    summary["device"] = str(device)
    print(json.dumps({k: v for k, v in summary.items() if k != "figures"}, indent=2))
    print(f"Figures → {out}")
    return summary


def _find_checkpoints(root: Path) -> list[Path]:
    return sorted(root.glob("*/last.pt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a .pt checkpoint")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Diagnose every */last.pt under --checkpoints-root",
    )
    parser.add_argument("--checkpoints-root", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--config", default=None, help="Override embedded checkpoint config")
    parser.add_argument("--space", choices=("z", "h"), default="z")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-umap", action="store_true", help="Skip UMAP (PCA only)")
    args = parser.parse_args()

    if args.all:
        ckpts = _find_checkpoints(args.checkpoints_root)
        if not ckpts:
            raise SystemExit(f"No last.pt found under {args.checkpoints_root}")
        for ckpt in ckpts:
            print(f"\n=== {ckpt} ===")
            diagnose_checkpoint(
                ckpt,
                config=args.config,
                space=args.space,
                max_samples=args.max_samples,
                out_dir=args.out / ckpt.parent.name if args.out else None,
                no_umap=args.no_umap,
            )
        return

    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint PATH or --all")
    diagnose_checkpoint(
        args.checkpoint,
        config=args.config,
        space=args.space,
        max_samples=args.max_samples,
        out_dir=args.out,
        no_umap=args.no_umap,
    )


if __name__ == "__main__":
    main()
