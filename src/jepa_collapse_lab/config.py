"""YAML configuration loading."""

from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = {"dataset", "augmentation", "loader", "model"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config and check that required sections exist."""
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    missing = REQUIRED_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config {path} is missing required keys: {sorted(missing)}")
    return cfg
