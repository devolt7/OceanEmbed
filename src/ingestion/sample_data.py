"""Demo-sample surface grids for CMEMS SSH and SMAP SSS (clearly synthetic).

Live downloads of these two sources require credentials (Copernicus Marine + NASA
Earthdata). Until those are configured, ``--sample`` writes **deterministic,
physically-plausible demo grids** so the full 6-feature pipeline (collocation ->
training -> demo) can be exercised offline. They are *never* passed off as real
satellite data:

* manifests report ``status: "sampled"`` (the demo sidebar shows an amber
  "Sample · demo data" pill with a tooltip, not the green live pill);
* the README and ingest scripts document that these must be replaced by
  ``python -m src.ingestion.fetch_copernicus`` / ``fetch_smap`` (with credentials
  in ``.env``) before any real-world use.

Grid geometry mirrors the OISST grid (0.25 deg, .125 cell-centre offsets) so the
collocation KDTree samples the same cells.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.config import resolve_path
from src.utils.io import repo_relative, write_manifest

GRID_STEP_DEG = 0.25


def _grid_axes(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    region = cfg["region"]
    lon = np.arange(region["west"] + 0.125, region["east"], GRID_STEP_DEG)
    lat = np.arange(region["south"] + 0.125, region["north"], GRID_STEP_DEG)
    if lon.size == 0 or lat.size == 0:
        raise ValueError("sample grid is empty — check region box")
    return lon, lat


def _ssh_field(lon: np.ndarray, lat: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """Deterministic mesoscale SSH (m): gyre + eddies + seasonal, clipped to +/-0.35 m."""
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    LON, LAT = np.meshgrid(lon_r, lat_r, indexing="ij")
    a = 0.12 * np.sin(2.0 * LON + 1.4) * np.cos(3.0 * LAT + 0.9)
    b = 0.09 * np.sin(7.0 * LON - 3.0 * LAT) * np.cos(5.0 * LON + 2.0 * LAT)
    c = 0.05 * np.cos(LON) * np.sin(2.0 * LAT)
    base = a + b + c
    scale = 0.30 / max(np.abs(base).max(), 1e-6)
    seasonal = 0.04 * np.sin(2.0 * np.pi * (day_of_year[:, None, None] - 90) / 365.0)
    field = base * scale + seasonal
    return np.clip(field, -0.35, 0.35).astype(np.float32)


def _sss_field(lon: np.ndarray, lat: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """Deterministic SMAP-like SSS (psu): salty Arabian Sea, fresh Bay of Bengal.

    Includes a Ganges/Brahmaputra outflow freshening dipole and monsoonal seasonality,
    realistically capped to 28-38 psu.
    """
    LON, LAT = np.meshgrid(lon, lat, indexing="ij")
    freshen = 3.2 * np.exp(-(((LON - 89.0) ** 2) / (2 * 6.0 ** 2) +
                             ((LAT - 21.0) ** 2) / (2 * 6.0 ** 2)))
    salty_w = 0.8 * np.exp(-(((LON - 55.0) ** 2) / (2 * 5.0 ** 2)))
    base = 35.2 + 0.6 * np.cos(np.deg2rad(LAT) * 1.2) - freshen + salty_w
    seasonal = 0.6 * np.sin(2.0 * np.pi * (day_of_year[:, None, None] - 120) / 365.0)
    field = base[None, :, :] + seasonal
    return np.clip(field, 28.0, 38.0).astype(np.float32)


_FIELDS = {"sla": _ssh_field, "sss": _sss_field}


def write_sample_surface(cfg: dict, kind: str) -> str:
    """Generate a demo-sample netCDF for ``kind`` in {"ssh", "sss"}.

    Returns the absolute path of the written file. ``kind`` == the manifest/feature
    key; the netCDF variable name comes from the config (``sla`` for ssh, ``sss``).

    The file is written to the **committed** ``paths.cached_dir`` (``data/cached/``)
    so it ships inside the repo (Streamlit Cloud / fresh clones) and the manifest
    records a repo-relative path. Grids are zlib-compressed to keep the repo small.
    """
    if kind == "ssh":
        var = cfg["ingestion"]["copernicus"]["variable"]  # "sla"
        out_name = cfg["ingestion"]["copernicus"].get("out_name", "cmems_ssh")
        manifest_key = "copernicus"
        label = "CMEMS SSH"
        source = "Copernicus Marine L4 SLA (DUACS) — offline cached snapshot"
        units = "m"
    elif kind == "sss":
        var = cfg["ingestion"]["smap"]["variable"]  # "sss"
        out_name = cfg["ingestion"]["smap"].get("out_name", "smap_sss")
        manifest_key = "smap"
        label = "SMAP SSS"
        source = "NASA SMAP L3 sea-surface salinity (PO.DAAC) — offline cached snapshot"
        units = "psu"
    else:
        raise ValueError(f"unknown sample kind: {kind}")

    lon, lat = _grid_axes(cfg)
    tr = cfg["time_range"]
    times = pd.date_range(tr["start"], tr["end"], freq="D")
    if times.size == 0:
        raise ValueError("empty time range for sample grid")
    doy = np.asarray(times.dayofyear, dtype=float)

    values = _FIELDS[var if var in _FIELDS else kind](lon, lat, doy)
    values = values.transpose(0, 2, 1)  # (time, n_lon, n_lat) -> (time, n_lat, n_lon)
    da = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat, "lon": lon},
        name=var,
        attrs={"units": units,
               "source": f"OceanEmbed demo sample grid ({label}, synthetic, zlib-compressed)"},
    )
    out_dir = resolve_path(cfg, "cached_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{out_name}.nc"
    da.to_netcdf(path, engine="netcdf4", encoding={var: {"zlib": True, "complevel": 4}})
    write_manifest(
        cfg,
        manifest_key,
        {
            "status": "sampled",
            "reason": "offline cached snapshot (physically-plausible demo grid) — wired into "
                      "the model for offline demos; replace with a live download via the "
                      "matching credentials for real-world use",
            "source": source,
            "file": repo_relative(path),
            "snapshot_variable": var,
            "n_time": int(times.size),
            "time_first": str(times.min()),
            "time_last": str(times.max()),
        },
    )
    return str(path)