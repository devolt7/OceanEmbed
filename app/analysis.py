"""Shared offline analysis powering the demo.

Everything here runs from committed / cached artifacts (parquet splits, netCDF grid
archives, the trained model) so the app never needs the network at runtime:

- monthly climatology (mean + std per depth) from collocated Argo profiles,
- a simple multi-output linear-regression baseline on the same features,
- an RMSE comparison of MLP vs baselines against held-out Argo profiles,
- thermocline depth (steepest temperature drop between standard levels),
- per-location monthly replay predictions from the cached satellite grid series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import resolve_path
from src.utils.io import manifest_file_path, read_manifest

BASELINE_MLP = "OceanEmbed MLP"
BASELINE_LINEAR = "Linear regression"
BASELINE_CLIM = "Climatology (monthly mean)"

MIN_MONTH_SAMPLES = 3
MIN_DAYS_PER_MONTH = 10


def depths_from_targets(meta: dict) -> list[int]:
    return [int(t.split("_")[1].replace("m", "")) for t in meta["targets"]]


def mlp_predict(meta: dict, scaler, mlp, X: np.ndarray) -> np.ndarray:
    """MLP prediction for feature rows, adding the centred-target offset back.

    The MLP is trained on (Y - train mean); inference must re-add that offset or every
    depth would be systematically biased. ``targets_mean`` lives in feature_meta.json.
    """
    pred = np.asarray(mlp.predict(scaler.transform(X), verbose=0), dtype=float)
    mean = meta.get("targets_mean")
    if mean:
        pred = pred + np.asarray(mean, dtype=float)
    return pred


# --------------------------------------------------------------------------- climatology
def monthly_climatology(collocated: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly mean / std of temperature per depth across all collocated Argo profiles.

    Months with too few samples fall back to the region-wide (all-month) statistics so
    the anomaly check and climatology baseline never encounter empty bins.
    """
    df = collocated
    if "month" not in df.columns:
        df = df.assign(month=pd.to_datetime(df["time"], utc=True).dt.month)
    g = df[["month", *targets]].copy()
    overall_mean = g[targets].mean()
    overall_std = g[targets].std()

    means = g.groupby("month")[targets].mean()
    stds = g.groupby("month")[targets].std()
    counts = g["month"].value_counts()
    for m in range(1, 13):
        if m not in stds.index or int(counts.get(m, 0)) < MIN_MONTH_SAMPLES:
            means.loc[m] = overall_mean
            stds.loc[m] = overall_std
    return means.sort_index(), stds.sort_index()


# --------------------------------------------------------------------------- baselines
def _read_processed(cfg: dict, name: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_path(cfg, "processed_dir") / name)


def fit_linear_baseline(cfg: dict, meta: dict):
    """Multi-output linear regression on the same feature set as the MLP (train split)."""
    from sklearn.linear_model import LinearRegression

    tr = _read_processed(cfg, "train.parquet")
    X = tr[meta["features"]].to_numpy(dtype=float)
    y = tr[meta["targets"]].to_numpy(dtype=float)
    return LinearRegression().fit(X, y)


def baseline_profiles(meta: dict, scaler, mlp, linear, clim_mean: pd.DataFrame,
                      X: np.ndarray, months: np.ndarray) -> dict[str, np.ndarray]:
    """Predicted profiles (depth columns) for test rows across all three models."""
    return {
        BASELINE_MLP: mlp_predict(meta, scaler, mlp, X),
        BASELINE_LINEAR: linear.predict(X),
        BASELINE_CLIM: clim_mean.loc[months, meta["targets"]].to_numpy(),
    }


def evaluate_baselines(cfg: dict, meta: dict, scaler, mlp, linear=None,
                       collocated: pd.DataFrame | None = None) -> pd.DataFrame:
    """RMSE of each method against held-out real Argo profiles (test split).

    Returns a compact comparison table: one row per method, per-depth RMSE plus the
    mean across depths — the naive methods are expected to do visibly worse than the MLP.
    """
    feat_cols = meta["features"]
    targets = meta["targets"]
    depths = depths_from_targets(meta)
    test = _read_processed(cfg, "test.parquet")
    X = test[feat_cols].to_numpy(dtype=float)
    y = test[targets].to_numpy(dtype=float)
    months = test["month"].to_numpy(dtype=int)

    if collocated is None:
        raise ValueError("collocated profiles required for the climatology baseline")
    clim_mean, _ = monthly_climatology(collocated, targets)
    linear = linear if linear is not None else fit_linear_baseline(cfg, meta)

    preds = baseline_profiles(meta, scaler, mlp, linear, clim_mean, X, months)
    rows = []
    for name, p in preds.items():
        per_depth = np.sqrt(np.nanmean((p - y) ** 2.0, axis=0))
        global_rmse = float(np.sqrt(np.nanmean((p - y) ** 2.0)))
        row = {"Method": name, "RMSE (°C)": round(global_rmse, 2)}
        for d, r in zip(depths, per_depth):
            row[f"{d} m"] = round(float(r), 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Method")


# --------------------------------------------------------------------------- thermocline
def thermocline_depth(temps, depths) -> float | None:
    """Depth of the steepest temperature drop between consecutive standard levels."""
    t = np.asarray(temps, dtype=float)
    d = np.asarray(depths, dtype=float)
    if t.size < 2 or d.size < 2:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        slopes = np.diff(t) / np.diff(d)  # negative -> cooling with depth
    finite = np.isfinite(slopes)
    if not finite.any():
        return None
    i = int(np.flatnonzero(finite)[np.nanargmin(slopes[finite])])
    return float((d[i] + d[i + 1]) / 2.0)


# --------------------------------------------------------------------------- replay
_VAR_ALIAS = {"sst": "sst", "ssh": "sla", "sss": "sss"}


def cached_surface_file(cfg: dict, feat: str) -> Path | None:
    """Find the committed / cached netCDF grid archive for one surface feature."""
    cached = resolve_path(cfg, "cached_dir")
    if feat == "sst":
        cands: list[Path | None] = [cached / "oisst_sst.nc"]
        man = read_manifest(cfg, "oisst")
        if man:
            cands.append(manifest_file_path(cfg, man))
    elif feat == "ssh":
        cands = [cached / "cmems_ssh.nc"]
    elif feat == "sss":
        cands = [cached / "smap_sss.nc"]
    else:
        cands = []
    for p in cands:
        if p is not None and p.exists():
            return p
    return None


def surface_monthly_series(path: Path, var: str, lat: float, lon: float,
                           min_days: int = MIN_DAYS_PER_MONTH) -> pd.Series | None:
    """Monthly-mean surface value at (lat, lon) from a cached grid netCDF.

    Returns a `pd.Series` indexed by month-start timestamps (UTC). Months with fewer
    than ``min_days`` valid daily samples are dropped (they read as data gaps).
    """
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        da = ds[var]
        latdim = "latitude" if "latitude" in da.dims else "lat"
        londim = "longitude" if "longitude" in da.dims else "lon"
        tdim = "time"
        ts = da.sel({latdim: float(lat), londim: float(lon)}, method="nearest")
        for d in list(ts.dims):
            if d != tdim and ts.sizes[d] == 1:
                ts = ts.isel({d: 0}, drop=True)
        idx = pd.to_datetime(np.asarray(ts.coords[tdim].values))
        s = pd.Series(np.asarray(ts.to_numpy()).ravel(), index=idx)
        s = s.dropna()
        s = s[~s.index.duplicated(keep="first")].sort_index()
        if s.empty:
            return None
        monthly = s.resample("ME").agg(["mean", "count"])
        monthly = monthly["mean"][monthly["count"] >= min_days].dropna()
        monthly.index = monthly.index.to_period("M").to_timestamp()
        return monthly
    finally:
        ds.close()


def replay_monthly(cfg: dict, meta: dict, scaler, mlp, lat: float, lon: float,
                   n_last: int = 12) -> dict | None:
    """Predicted 0–1000 m profiles for the last ``n_last`` months at (lat, lon).

    Reads each surface feature's full daily series from the cached netCDF archives,
    aggregates to monthly means, and predicts one profile per available month.

    Returns ``{"months", "pred", "gaps"}`` — ``months`` are month-start timestamps,
    ``pred`` is (n_months, n_targets) and ``gaps`` lists months present in the archive
    window but without enough valid data (nothing is ever faked). Returns None if any
    required surface feature has no cached archive or the clicked point is land.
    """
    feat_cols = meta["features"]
    surface_feats = [f for f in feat_cols if f not in ("lat", "lon", "month")]

    series: dict[str, pd.Series] = {}
    for feat in surface_feats:
        path = cached_surface_file(cfg, feat)
        if path is None:
            continue
        try:
            m = surface_monthly_series(path, _VAR_ALIAS.get(feat, feat), lat, lon)
        except Exception:
            m = None
        if m is not None and not m.empty:
            series[feat] = m

    if len(series) != len(surface_feats):
        return None

    all_months = pd.DatetimeIndex([], freq=None)
    for m in series.values():
        all_months = all_months.union(m.index)
    if all_months.empty:
        return None
    all_months = all_months.sort_values()

    frame = pd.DataFrame(series).reindex(all_months)
    complete = frame.dropna()
    gaps = [pd.Timestamp(t) for t in all_months if t not in complete.index]
    frame = complete.sort_index().tail(n_last)
    if frame.empty:
        return None

    pos = {f: i for i, f in enumerate(feat_cols)}
    X = np.full((len(frame), len(feat_cols)), np.nan, dtype=float)
    for feat in series:
        X[:, pos[feat]] = frame[feat].to_numpy(dtype=float)
    X[:, pos["lat"]] = float(lat)
    X[:, pos["lon"]] = float(lon)
    X[:, pos["month"]] = frame.index.month.to_numpy(dtype=float)

    pred = mlp_predict(meta, scaler, mlp, X)
    return {"months": frame.index, "pred": pred, "gaps": gaps, "frame": frame}