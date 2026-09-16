"""Collocation engine: Argo profiles x satellite surface grids.

For every Argo profile that reaches the configured minimum depth:

1. Temperatures are interpolated onto the standard target depth levels
   (0/50/100/200/500/1000 m) - shallow-surface values are clamped to the
   shallowest real Argo observation; values below the deepest real sample and
   profiles that do not reach ``min_profile_depth_m`` are dropped.
2. Surface features (SST / SSH / SSS) are sampled at the float's nearest grid
   cell via a KDTree, within a +/- ``temporal_window_days`` day window and a
   ``max_distance_km`` cell radius. Missing sources (no credentials) simply
   drop their feature column - only downloaded, real data is ever used.

Outputs:
  data/interim/collocated.parquet   - one row per matched profile
  data/interim/collocation_report.csv - stage-by-stage counts and match rate
  data/processed/surface_sst_latest.parquet - latest OISST grid for the demo app

Run:  ``python -m src.collocation.collocate [--config config.bootstrap.yaml]``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.config import load_config, resolve_path, ensure_dirs
from src.utils.geo import build_grid_kdtree, nearest_grid_cell, interpolate_profile
from src.utils.io import manifest_file_path, read_manifest, save_df, installed_surface_sources

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collocate")

OUT_COLLOCATED = "collocated"
OUT_REPORT = "collocation_report"
OUT_SST_GRID = "surface_sst_latest"


class GriddedSurfaceSampler:
    """KDTree + time-window sampler for one gridded satellite source."""

    def __init__(self, lon: np.ndarray, lat: np.ndarray, times: np.ndarray, values: np.ndarray):
        self.lon = np.asarray(lon, dtype=float)
        self.lat = np.asarray(lat, dtype=float)
        self.times = np.asarray(times)  # datetime64
        self.values = np.asarray(values, dtype=float)
        if self.values.shape[1:] != (len(self.lat), len(self.lon)):
            # allow (n_time, n_lat*n_lon) flattened convenience input
            self.values = self.values.reshape(-1, len(self.lat), len(self.lon))
        self.kdtree = build_grid_kdtree(self.lon, self.lat)

    def sample(
        self,
        q_lon: float,
        q_lat: float,
        q_time: np.datetime64,
        window_days: int,
        max_distance_km: float,
    ) -> tuple[float, float]:
        """Return (surface_value, distance_km). NaN when out of time window/radius."""
        if self.times.size == 0:
            return np.nan, np.nan
        diff = np.abs(self.times - np.datetime64(q_time))
        i = int(np.argmin(diff))
        if diff[i] > np.timedelta64(window_days, "D"):
            return np.nan, np.nan
        gi, gj, dist = nearest_grid_cell(self.lon, self.lat, q_lon, q_lat, self.kdtree, max_distance_km)
        if gi < 0:
            return np.nan, dist
        val = float(self.values[i, gj, gi])
        return (np.nan if not np.isfinite(val) else val), dist


# ---------------------------------------------------------------------------
# surface loaders
# ---------------------------------------------------------------------------
def _coord(da, *names):
    for n in names:
        if n in da.coords or n in da.dims:
            return da[n].values
    raise KeyError(f"coordinate not found in {list(da.coords)}")


def load_oisst_sampler(cfg: dict) -> tuple[GriddedSurfaceSampler, dict]:
    man = read_manifest(cfg, "oisst")
    if not man or man.get("status") != "downloaded":
        raise FileNotFoundError("OISST manifest missing; run `python -m src.ingestion.fetch_oisst` first.")
    path = manifest_file_path(cfg, man)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"OISST file missing ({path}); re-run `python -m src.ingestion.fetch_oisst`."
        )
    with xr.open_dataset(path) as ds:
        da = ds[cfg["ingestion"]["oisst"]["variable"]]
        if "zlev" in da.dims and da["zlev"].size == 1:
            da = da.isel(zlev=0)
        da = da.where(np.isfinite(da))
        values = np.asarray(da.values, dtype=float).copy()
        values[values < -5.0] = np.nan  # land / fill sentinel
        lon, lat, times = _coord(da, "lon", "longitude"), _coord(da, "lat", "latitude"), da["time"].values
    meta = {"n_time": len(times), "time_last": str(np.max(times))}
    return GriddedSurfaceSampler(lon, lat, times, values), meta


def load_netcdf_sampler(
    cfg: dict, source: str, var: str, window_days: int
) -> tuple[GriddedSurfaceSampler | None, float]:
    man = read_manifest(cfg, source)
    if not man or man.get("status") not in ("downloaded", "sampled"):
        log.info("source '%s' not available; omitting its feature", source)
        return None, window_days
    path = manifest_file_path(cfg, man)
    if path is None or not path.exists():
        log.warning("manifest for '%s' points to missing file %s; omitting", source, path)
        return None, window_days
    status = man.get("status")
    if status == "sampled":
        log.warning("source '%s' is a DEMO SAMPLE grid — replace with live data for real runs", source)
    with xr.open_dataset(path) as ds:
        if var not in ds:
            log.warning("variable '%s' not in %s (%s); omitting source", var, path, list(ds))
            return None, window_days
        da = ds[var].squeeze()
        lon, lat = _coord(da, "lon", "longitude"), _coord(da, "lat", "latitude")
        times = da["time"].values
        values = np.asarray(da.values, dtype=float)
    return GriddedSurfaceSampler(lon, lat, times, values), window_days


# ---------------------------------------------------------------------------
# argo -> target frame
# ---------------------------------------------------------------------------
def argo_to_target_frame(argo: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    levels = np.asarray(cfg["depths"]["target_levels"], dtype=float)
    min_depth = float(cfg["depths"]["min_profile_depth_m"])
    names = [f"temp_{int(l)}m" for l in levels]
    rows, skip_non = [], {"min_depth": 0, "nan_levels": 0, "empty": 0}
    for _, row in argo.iterrows():
        try:
            temps = interpolate_profile(np.asarray(row["depth"]), np.asarray(row["temperature"]),
                                        levels, min_depth)
        except ValueError as exc:
            msg = str(exc)
            if "reach" in msg:
                skip_non["min_depth"] += 1
            elif "empty" in msg or "valid" in msg:
                skip_non["empty"] += 1
            else:
                skip_non["empty"] += 1
            continue
        if np.isnan(temps).any():
            skip_non["nan_levels"] += 1
            continue
        rows.append({
            "float_id": row["float_id"],
            "time": row["time"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            **{n: float(t) for n, t in zip(names, temps)},
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    log.info(
        "argo target frame: %d profiles usable (%d too-shallow, %d incomplete levels, %d empty)",
        len(df), skip_non["min_depth"], skip_non["nan_levels"], skip_non["empty"],
    )
    return df, skip_non


def _sample_column(
    df: pd.DataFrame, sampler: GriddedSurfaceSampler, window_days: int, max_km: float
) -> tuple[pd.Series, float]:
    vals, dists = [], []
    for _, row in df.iterrows():
        v, d = sampler.sample(row["lon"], row["lat"], row["time"], window_days, max_km)
        vals.append(v)
        dists.append(d)
    return pd.Series(vals, index=df.index, dtype=float), float(np.nanmean(dists))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config yaml")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    argo_path = resolve_path(cfg, "raw_dir") / "argo_profiles.parquet"
    if not argo_path.exists():
        log.error("Argo profiles missing: run `python -m src.ingestion.fetch_argo` first.")
        return 1
    argo = pd.read_parquet(argo_path)
    log.info("loaded %d raw Argo profiles", len(argo))

    target, skip_non = argo_to_target_frame(argo, cfg)
    if target.empty:
        log.error("no usable Argo profiles after depth/interpolation filtering.")
        return 1
    out = target.copy()

    window = int(cfg["collocation"]["temporal_window_days"])
    max_km = float(cfg["collocation"]["max_distance_km"])

    sst_sampler, sst_meta = load_oisst_sampler(cfg)
    out["sst"], _ = _sample_column(out, sst_sampler, window, max_km)

    ssh_sampler, _ = load_netcdf_sampler(cfg, "copernicus", cfg["ingestion"]["copernicus"]["variable"], window)
    if ssh_sampler is not None:
        out["ssh"], _ = _sample_column(out, ssh_sampler, window, max_km)

    smap_cfg = cfg["ingestion"]["smap"]
    smap_window = int(smap_cfg.get("window_days", 8))
    sss_sampler, _ = load_netcdf_sampler(cfg, "smap", smap_cfg["variable"], smap_window)
    if sss_sampler is not None:
        out["sss"], _ = _sample_column(out, sss_sampler, smap_window, max_km)

    # match reporting: a profile is "matched" when it has a valid sst within window+radius
    n_total = len(out)
    n_mat = int(out["sst"].notna().sum())
    rate = 100.0 * n_mat / n_total if n_total else 0.0
    log.info(
        "SST match rate: %.1f%% (%d / %d profiles within +/-%dd & %d km)",
        rate, n_mat, n_total, window, int(max_km),
    )

    if cfg["collocation"].get("require_sst", True) and n_mat < 10:
        log.warning("Very low SST match count (%d); check time range / window.", n_mat)

    collocated = out[out["sst"].notna()].copy()
    collocated = collocated.reset_index(drop=True)

    if collocated.empty:
        log.error("no collocated profiles; nothing to write.")
        return 1

    save_df(collocated, cfg, "interim_dir", OUT_COLLOCATED)
    available = installed_surface_sources(cfg)
    report = pd.DataFrame(
        {
            "stage": ["raw_argo_profiles", "usable_after_interp", "collocated_with_sst"],
            "count": [len(argo), len(out), len(collocated)],
        }
    )
    report.to_csv(resolve_path(cfg, "interim_dir") / f"{OUT_REPORT}.csv", index=False)

    # cache the latest grid of every downloaded surface source for the offline
    # Streamlit demo (the app must not hit the network at runtime)
    for name, sampler in (("sst", sst_sampler), ("ssh", ssh_sampler), ("sss", sss_sampler)):
        if sampler is None:
            continue
        lon = np.asarray(sampler.lon); lat = np.asarray(sampler.lat)
        last = np.asarray(sampler.values)[-1]  # (n_lat, n_lon)
        lon_g, lat_g = np.meshgrid(lon, lat, indexing="ij")
        grid = pd.DataFrame({"lon": lon_g.ravel(), "lat": lat_g.ravel(), name: last.ravel()})
        save_df(grid, cfg, "processed_dir", f"surface_{name}_latest")

    log.info(
        "collocated: %d rows with surface features %s -> %s/match rate %.1f%%",
        len(collocated), available,
        resolve_path(cfg, "interim_dir") / f"{OUT_COLLOCATED}.parquet", rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))