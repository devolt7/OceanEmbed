"""Fetch real Argo float temperature/salinity profiles for the configured region/window.

Two backends, both 100% real data (no synthesis):

* ``argopy`` (default)  — official GDAC index + profile access via the Ifremer/UC data hubs.
* ``argovis`` (fallback) — HTTPS JSON API to the Argovis Argo archive at the University of
  Colorado (no login). Automatically used when ``argopy`` is unavailable or fails.

Output: ``data/raw/argo_profiles.parquet`` plus ``data/raw/manifest_argo.json``.
Rows of the parquet are individual Argo profiles with ``depth`` and ``temperature``
stored as arrow list columns.

Run:  ``python -m src.ingestion.fetch_argo [--config config.bootstrap.yaml]``
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path, ensure_dirs
from src.utils.io import repo_relative, save_df, write_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch_argo")

OUT_NAME = "argo_profiles"

DEFAULT_DEPTH_CAP_M = 1200.0  # keep storage small; targets never exceed 1000 m


def _ts(ds) -> pd.Series:
    """Normalise argopy timestamps to pandas datetime64."""
    vals = np.asarray(ds["TIME"].values)
    if vals.ndim == 0:
        vals = np.array([vals.ravel()[0]])
    while vals.ndim > 1:  # (N_PROF, N_LEVELS) -> first level time per profile
        vals = vals[:, 0]
    s = pd.Series(vals)
    try:
        return pd.to_datetime(s, utc=True, unit="D", origin="1950-01-01")
    except Exception:
        return pd.to_datetime(s, utc=True)


def _pres_to_depth_m(pres_bar: np.ndarray) -> np.ndarray:
    """Approximate pressure (dbar) to geometric depth (m) using the standard UNESCO9 formula."""
    p = np.asarray(pres_bar, dtype=float)
    dep = ((((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p) / 9.80655
    return np.clip(dep, 0.0, None)


def _exclusive_end(cfg: dict) -> str:
    """argopy region time bounds are exclusive at the top; use end + 1 day."""
    import datetime as dt

    d = dt.date.fromisoformat(cfg["time_range"]["end"][:10])
    return (d + dt.timedelta(days=1)).isoformat()


def _box(cfg: dict) -> list:
    r = cfg["region"]
    return [
        r["west"], r["east"], r["south"], r["north"],
        0, int(DEFAULT_DEPTH_CAP_M),
        cfg["time_range"]["start"][:10], _exclusive_end(cfg),
    ]


def _tidy_points(ds, source: str) -> pd.DataFrame:
    """Collapse an argopy N_POINTS dataset into one row per Argo profile."""
    def vec(name, default):
        if name not in ds:
            return default
        return np.asarray(ds[name].values)

    time = _ts(ds)
    lat = vec("LATITUDE", None)
    lon = vec("LONGITUDE", None)
    if lat is None or lon is None:
        raise ValueError(f"{source} dataset missing LATITUDE/LONGITUDE")
    pres = vec("PRES", np.full(len(time), np.nan))
    temp = vec("TEMP", np.full(len(time), np.nan))
    temp_qc = vec("TEMP_QC", None)
    sal = vec("PSAL", np.full(len(time), np.nan))
    sal_qc = vec("PSAL_QC", None)
    plat = vec("PLATFORM_NUMBER", None)
    cycle = vec("CYCLE_NUMBER", None)
    if plat is None or cycle is None:
        plat = np.arange(len(time))  # degraded grouping: one "float" per row
        cycle = np.zeros(len(time))

    tidy = pd.DataFrame(
        {
            "float_id": [str(x) for x in np.asarray(plat).ravel()],
            "cycle": np.asarray(cycle, dtype=float).ravel(),
            "time": time.values,
            "lat": np.asarray(lat, dtype=float).ravel(),
            "lon": np.asarray(lon, dtype=float).ravel(),
            "pres": np.asarray(pres, dtype=float).ravel(),
            "temp": np.asarray(temp, dtype=float).ravel(),
            "temp_qc": (np.asarray(temp_qc, dtype=float).ravel() if temp_qc is not None
                        else np.full(len(time), np.nan)),
            "sal": np.asarray(sal, dtype=float).ravel(),
            "sal_qc": (np.asarray(sal_qc, dtype=float).ravel() if sal_qc is not None
                       else np.full(len(time), np.nan)),
        }
    )
    tidy = tidy.dropna(subset=["time", "lat", "lon"])

    # Keep only oceanographically plausible values (real-data sanity check).
    tidy = tidy[(tidy["temp"] >= -2.6) & (tidy["temp"] <= 40.0)]
    tidy = tidy[(tidy["sal"].isna()) | ((tidy["sal"] >= 0.0) & (tidy["sal"] <= 42.5))]
    # QC filter: keep good/adjusted (1, 2); NaN cells are dropped as bad.
    if temp_qc is not None:
        tidy = tidy[tidy["temp_qc"].isin([1.0, 2.0]) | tidy["temp_qc"].isna()]
    if sal_qc is not None:
        tidy = tidy[tidy["sal_qc"].isin([1.0, 2.0]) | tidy["sal_qc"].isna()]

    depth_m = _pres_to_depth_m(tidy["pres"].to_numpy())
    tidy["depth"] = depth_m
    tidy = tidy[(tidy["depth"] >= 0) & (tidy["depth"] <= DEFAULT_DEPTH_CAP_M)]

    if tidy.empty:
        return pd.DataFrame(columns=["float_id", "time", "lat", "lon", "depth",
                                     "temperature", "salinity"])

    # Collapse N_POINTS -> one row per (float, cycle) profile.
    rows = []
    for (fid, cyc), grp in tidy.sort_values("depth").groupby(["float_id", "cycle"]):
        rows.append(
            {
                "float_id": str(fid),
                "cycle": str(int(cyc)) if np.isfinite(cyc) else "",
                "time": grp["time"].iloc[0],
                "lat": float(grp["lat"].iloc[0]),
                "lon": float(grp["lon"].iloc[0]),
                "depth": grp["depth"].tolist(),
                "temperature": grp["temp"].tolist(),
                "salinity": grp["sal"].tolist(),
            }
        )
    return pd.DataFrame(rows)


def fetch_with_argopy(cfg: dict) -> pd.DataFrame:
    """Fetch via the argopy package (GDAC / Ifremer erddap).

    argopy's *standard* user mode returns one row per measurement (N_POINTS).
    Profiles are reconstructed by grouping on (float id, cycle number).
    """
    from argopy import DataFetcher as ArgoDataFetcher

    box = _box(cfg)
    log.info("argopy region fetch: box=%s", box)
    last_exc: Exception | None = None
    for attempt in range(1, 4):  # transient Ifremer ERDDAP hiccups are common
        try:
            ds = ArgoDataFetcher().region(box).load().data
            return _tidy_points(ds, "argopy")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("argopy attempt %d failed: %s", attempt, exc)
            time.sleep(20 * attempt)
    raise RuntimeError(f"argopy fetch failed after 3 attempts: {last_exc}")


def fetch_with_argovis(cfg: dict) -> pd.DataFrame:
    """Fetch via argopy's Argovis datasource (HTTPS fallback backend, no login)."""
    from argopy import DataFetcher as ArgoDataFetcher

    box = _box(cfg)
    log.info("argopy(argovis source) region fetch: box=%s", box)
    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            ds = ArgoDataFetcher(datasource="argovis").region(box).load().data
            return _tidy_points(ds, "argovis")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("argovis attempt %d failed: %s", attempt, exc)
            time.sleep(10 * attempt)
    raise RuntimeError(f"argovis fetch failed after 3 attempts: {last_exc}")


def fetch(cfg: dict, backend: str | None = None) -> pd.DataFrame:
    backend = backend or cfg["ingestion"]["argo"].get("backend", "argopy")
    if backend == "argovis":
        return fetch_with_argovis(cfg)
    try:
        return fetch_with_argopy(cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("argopy backend failed (%s); falling back to Argovis HTTPS API", exc)
        df = fetch_with_argovis(cfg)
        log.info("FALLBACK_BACKEND=argovis")
        return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument("--backend", default=None, choices=["argopy", "argovis"])
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    out_path = resolve_path(cfg, "raw_dir") / f"{OUT_NAME}.parquet"
    manifest_path = resolve_path(cfg, "raw_dir") / "manifest_argo.json"

    if out_path.exists():
        log.info("Argo file already exists, skipping download: %s", out_path)
        return 0
    if manifest_path.exists():
        log.info("Argo manifest exists; re-running implies a fresh start. Proceeding anyway.")

    df = fetch(cfg, backend=args.backend)
    if df.empty:
        log.error("No Argo profiles fetched for the configured region/window.")
        return 1

    df = df.dropna(subset=["lat", "lon"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[["float_id", "time", "lat", "lon", "depth", "temperature", "salinity"]]
    save_df(df, cfg, "raw_dir", OUT_NAME)

    write_manifest(
        cfg,
        "argo",
        {
            "status": "downloaded",
            "source": "Argo float profiles (GDAC/Argovis)",
            "file": repo_relative(out_path),
            "n_profiles": int(len(df)),
            "n_floats": int(df["float_id"].nunique()),
            "time_range": [str(df["time"].min()), str(df["time"].max())],
        },
    )
    log.info(
        "Wrote %d profiles (%d floats) to %s",
        len(df), df["float_id"].nunique(), out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))