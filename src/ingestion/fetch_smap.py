"""Fetch real NASA SMAP sea-surface salinity (SSS) for region/window.

Uses the ``earthaccess`` library against PO.DAAC. Requires a **free NASA Earthdata
login** supplied via ``.env``:

    NASA_EARTHDATA_USERNAME=...
    NASA_EARTHDATA_PASSWORD=...

If credentials are absent the script **skips gracefully** (manifest ``status: skipped``)
and the pipeline simply omits the ``sss`` feature. Only real SMAP data ever enters the
pipeline.

``--sample`` writes a deterministic **demo-sample** SSS grid (manifest ``status: sampled``)
so the full 6-feature pipeline can run offline before credentials are configured. The
sample is clearly labelled synthetic and must be replaced with a live download for any
real-world use.

The MVP fetches the SMAP L3 8-day composite granules covering the region and stacks them
into one regional NetCDF (``smap_sss.nc``) so collocation can sample each composite.

Run:  ``python -m src.ingestion.fetch_smap [--config config.bootstrap.yaml] [--sample]``
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import xarray as xr

from src.ingestion.sample_data import write_sample_surface
from src.utils.config import load_config, resolve_path, ensure_dirs
from src.utils.io import populated, repo_relative, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_smap")

OUT_NAME = "smap_sss"


def has_credentials() -> bool:
    username = os.environ.get("NASA_EARTHDATA_USERNAME")
    password = os.environ.get("NASA_EARTHDATA_PASSWORD")
    return populated(username) and populated(password)


def _load_granule(path: Path) -> xr.Dataset:
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        sss = next((ds[k] for k in ("sss", "SSS", "sea_surface_salinity") if k in ds), None)
        if sss is None:
            raise ValueError(f"no SSS variable in {path}: {list(ds)}")
        sss = sss.rename("sss")
        lat, lon = sss["lat"], sss["lon"]
        return xr.Dataset({"sss": sss}).assign_coords(lat=lat, lon=lon)


def fetch_striped(cfg: dict, raw_dir: Path) -> list[Path]:
    """Download SMAP granules covering the region and return downloaded file paths."""
    import earthaccess

    earthaccess.login(strategy="environment")
    region = cfg["region"]
    tr = cfg["time_range"]
    about = cfg["ingestion"]["smap"]

    results = earthaccess.search_data(
        short_name=about["short_name"],
        bounding_box=[region["west"], region["south"], region["east"], region["north"]],
        temporal=[tr["start"], tr["end"]],
    )
    if not results:
        raise RuntimeError("earthaccess returned zero SMAP granules for the box/time range")
    log.info("SMAP: %d granules matched, downloading...", len(results))
    files = earthaccess.download(results, local_path=str(raw_dir))
    return [Path(f) for f in files if Path(f).suffix.lower() in (".nc", ".nc4", ".h5")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument("--sample", action="store_true",
                        help="write a deterministic demo-sample grid instead of downloading")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    raw_dir = resolve_path(cfg, "raw_dir")

    if args.sample:
        path = write_sample_surface(cfg, "sss")
        log.info("SSS (SMAP) demo sample written -> %s", path)
        return 0

    out_path = raw_dir / f"{OUT_NAME}.nc"

    if not has_credentials():
        write_manifest(
            cfg,
            "smap",
            {
                "status": "skipped",
                "reason": "NASA_EARTHDATA_USERNAME / NASA_EARTHDATA_PASSWORD not set in .env",
            },
        )
        log.warning(
            "SSS (SMAP) skipped: set NASA_EARTHDATA_USERNAME / NASA_EARTHDATA_PASSWORD "
            "in .env to enable. Proceeding without sss feature."
        )
        return 0

    if out_path.exists():
        log.info("SMAP file already exists, skipping download: %s", out_path)
    else:
        granules = fetch_striped(cfg, raw_dir)
        dsets = [g for g in map(_load_granule, granules) if g is not None]
        if not dsets:
            log.error("no usable SMAP granules parsed; refusing to create empty dataset.")
            write_manifest(cfg, "smap", {"status": "failed", "reason": "no parseable granules"})
            return 1
        merged = xr.concat(dsets, dim="time").sortby("time")
        merged.to_netcdf(out_path)
        log.info("SMAP stacked -> %s (%d time steps)", out_path, merged["time"].size)

    write_manifest(
        cfg,
        "smap",
        {
            "status": "downloaded",
            "source": "NASA SMAP L3 sea-surface salinity (PO.DAAC via earthaccess)",
            "file": repo_relative(out_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))