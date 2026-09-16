"""Project configuration loader for OceanEmbed.

All scoping (region, time range, target depths, paths, hyper-parameters) lives in
``config.yaml`` (or an overridden ``--config path``) and is loaded through this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and validate the YAML configuration.

    An optional ``--config`` path overrides the default ``config.yaml`` (e.g. the
    short-window ``config.bootstrap.yaml`` used by the quick-start flow).
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    _apply_env_overrides(cfg)
    _validate(cfg, cfg_path)
    return cfg


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Allow environment variables to override config (used for quick short-window runs)."""
    start = os.environ.get("OCEANEMBED_START_DATE")
    if start:
        cfg["time_range"]["start"] = start


def _validate(cfg: dict[str, Any], cfg_path: Path) -> None:
    region = cfg.get("region", {})
    box = [region.get("west"), region.get("east"), region.get("south"), region.get("north")]
    if None in box:
        raise ValueError(f"region.{['west','east','south','north']} must all be set in {cfg_path}")
    if not (box[0] < box[1] and box[2] < box[3]):
        raise ValueError("Invalid region box: west<east and south<north required.")

    tr = cfg.get("time_range", {})
    if not tr.get("start") or not tr.get("end"):
        raise ValueError(f"time_range.start / time_range.end must be set in {cfg_path}")

    levels = cfg.get("depths", {}).get("target_levels")
    if not levels or sorted(levels) != levels:
        raise ValueError("depths.target_levels must be a non-empty, ascending list.")

    for key in ("raw_dir", "interim_dir", "processed_dir", "model_dir", "report_dir", "figure_dir"):
        if not cfg.get("paths", {}).get(key):
            raise ValueError(f"paths.{key} must be set.")
    if "cached_dir" not in cfg.get("paths", {}):
        cfg.setdefault("paths", {})["cached_dir"] = "data/cached"


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Return an absolute project path for one of the ``paths.*`` keys."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create all configured output directories if missing."""
    for key in ("raw_dir", "interim_dir", "processed_dir", "model_dir", "report_dir", "figure_dir", "cached_dir"):
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)