"""Fetch real NOAA OISST daily sea-surface temperature for the configured region/window.

Uses the public NOAA ERDDAP griddap endpoint (no login required). Subsets the global
daily 0.25 deg analysis to (region x time range) and downloads a single NetCDF.

This is the **mandatory** surface feature source for OceanEmbed: with no SST there is
no collocated dataset, so this script fails loudly rather than degrading silently.

Output: ``data/raw/oisst_<start>_<end>.nc`` + ``data/raw/manifest_oisst.json``.
Run:  ``python -m src.ingestion.fetch_oisst [--config config.bootstrap.yaml]``
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import requests
import xarray as xr

import numpy as np  # noqa: E402 (used by verify_nc reporting)

from src.utils.config import load_config, resolve_path, ensure_dirs
from src.utils.io import repo_relative, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_oisst")

CHUNK = 1 << 20  # 1 MiB streaming chunk


def build_erddap_url(cfg: dict) -> str:
    region = cfg["region"]
    tr = cfg["time_range"]
    base = cfg["ingestion"]["oisst"]["erddap_url"].rstrip("/")
    var = cfg["ingestion"]["oisst"]["variable"]
    # ERDDAP griddap subset syntax is bracket-per-dimension in the variable's
    # dimension order. OISST v2.1 stores sst(time, zlev, latitude, longitude)
    # with a single zlev (0.0), which we constrain explicitly.
    # No stride on time -> ERDDAP returns every grid step inside the box (daily, 0.25 deg).
    box = f"{var}[({tr['start']}T00:00:00Z):({tr['end']}T00:00:00Z)]" \
          f"[(0.0):(0.0)]" \
          f"[({region['south']}):({region['north']})][({region['west']}):({region['east']})]"
    return f"{base}?{box}"


def download(url: str, dest: Path) -> None:
    log.info("downloading OISST subset:\n  %s", url)
    headers = {"Accept-Encoding": "identity"}  # ERDDAP gzips by default; NetCDF must not be gzipped
    with requests.get(url, stream=True, headers=headers, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=CHUNK)
    # defensive: ERDDAP may still send gzip if it ignores the header
    if dest.read_bytes()[:2] == b"\x1f\x8b":
        import gzip

        log.warning("server sent gzip-compressed NetCDF; gunzipping")
        with gzip.open(dest, "rb") as gz, open(dest.with_suffix(".nc.tmp"), "wb") as out:
            shutil.copyfileobj(gz, out, length=CHUNK)
        dest.with_suffix(".nc.tmp").replace(dest)
    log.info("downloaded -> %s (%.1f MiB)", dest, dest.stat().st_size / (1 << 20))


def verify_nc(path: Path, cfg: dict) -> dict:
    """Sanity-check the downloaded NetCDF and report coordinates."""
    with xr.open_dataset(path) as ds:
        var = cfg["ingestion"]["oisst"]["variable"]
        if var not in ds:
            raise RuntimeError(f"'{var}' variable missing from {path}: {list(ds)}")
        da = ds[var]
        lon = da["lon"].values if "lon" in da.coords else da["longitude"].values
        lat = da["lat"].values if "lat" in da.coords else da["latitude"].values
        t = [str(x) for x in da["time"].values]
        return {
            "variable": var,
            "n_time": int(da["time"].size),
            "shape": list(da.shape),
            "time_first": t[0],
            "time_last": t[-1],
            "lat_min": float(np.min(lat)),
            "lat_max": float(np.max(lat)),
            "lon_min": float(np.min(lon)),
            "lon_max": float(np.max(lon)),
            "units": str(getattr(da, "units", "")),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    tr = cfg["time_range"]
    out_path = resolve_path(cfg, "raw_dir") / f"oisst_{tr['start'][:10]}_{tr['end'][:10]}.nc"
    manifest_path = resolve_path(cfg, "raw_dir") / "manifest_oisst.json"

    if out_path.exists():
        log.info("OISST file already exists, skipping download: %s", out_path)
        info = verify_nc(out_path, cfg)
    else:
        url = build_erddap_url(cfg)
        download(url, out_path)
        info = verify_nc(out_path, cfg)

    write_manifest(
        cfg,
        "oisst",
        {
            "status": "downloaded",
            "source": "NOAA OISST v2.1 daily (ERDDAP griddap)",
            "file": repo_relative(out_path),
            "url": build_erddap_url(cfg),
            **info,
        },
    )
    log.info("OISST OK: %d daily frames %s..%s", info["n_time"], info["time_first"], info["time_last"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))