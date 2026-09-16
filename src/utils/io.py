"""Small I/O helpers shared by ingestion / collocation scripts."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PROJECT_ROOT, resolve_path

_PLACEHOLDER_TOKENS = ("your_", "xxxx", "xxx", "changeme", "enter_", "example")


def populated(value: Any) -> bool:
    """True if a config/credential value is set to something real rather than a placeholder.

    ``.env.example`` ships with placeholder tokens (``your_...``) so a blind ``cp`` never
    produces a false-positive credential check that would send bogus logins downstream.
    """
    if not value:
        return False
    v = str(value).strip()
    if not v:
        return False
    low = v.lower()
    return not any(tok in low for tok in _PLACEHOLDER_TOKENS)


def source_manifest_path(cfg: dict[str, Any], source: str) -> Path:
    return resolve_path(cfg, "raw_dir") / f"manifest_{source}.json"


def repo_relative(path: str | Path) -> str:
    """Repo-relative POSIX path string for a given path (portable across machines)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_manifest(cfg: dict[str, Any], source: str, payload: dict[str, Any]) -> Path:
    path = source_manifest_path(cfg, source)
    payload.setdefault("updated_utc", dt.datetime.utcnow().isoformat(timespec="seconds"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def read_manifest(cfg: dict[str, Any], source: str) -> dict[str, Any] | None:
    path = source_manifest_path(cfg, source)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def manifest_file_path(cfg: dict[str, Any], manifest: dict[str, Any] | None) -> Path | None:
    """Resolve the ``file`` entry of a manifest against the project root.

    Manifests store repo-relative paths (e.g. ``data/cached/cmems_ssh.nc``) so the repo
    is relocatable — a fresh `git clone` on Streamlit Cloud, another laptop, or CI can
    resolve the same file without any machine-specific absolute path. Absolute paths that
    point at this checkout are accepted as-is; paths pointing elsewhere return None.
    """
    if not manifest:
        return None
    raw = manifest.get("file")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute():
        try:
            p = p.resolve()
            if PROJECT_ROOT.resolve() in p.parents or p == PROJECT_ROOT.resolve():
                return p
        except OSError:
            pass
        return None  # absolute path outside this checkout — not usable on another machine
    return (PROJECT_ROOT / p).resolve()


def installed_surface_sources(cfg: dict[str, Any]) -> list[str]:
    """Which surface data sources are available for the current run.

    ``sst`` (OISST) is mandatory; ``ssh`` / ``sss`` are optional. They count as
    available when their manifests report a real download **or** a clearly-labelled
    demo sample (``status: "sampled"``) so the offline demo still exercises the full
    six-feature pipeline.
    """
    def _ok(source: str) -> bool:
        return (read_manifest(cfg, source) or {}).get("status") in ("downloaded", "sampled")

    srcs = []
    if _ok("oisst"):
        srcs.append("sst")
    if _ok("copernicus"):
        srcs.append("ssh")
    if _ok("smap"):
        srcs.append("sss")
    return srcs


def save_df(df: pd.DataFrame, cfg: dict[str, Any], key: str, name: str) -> Path:
    """Save a dataframe to a parquet under one of the configured data dirs."""
    out_dir = resolve_path(cfg, key)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path