"""YAML config loading with dotted overrides (a.b=c)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def _parse_value(raw: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(raw)
    except Exception:  # pragma: no cover
        return raw


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for k in parts[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict:
    cfg: dict = {}
    if path is not None:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)
