"""Internet-first live data layer for the deployed demo.

Tries the real, current satellite / Argo feeds first and automatically falls back to
the committed offline snapshot whenever a feed is unreachable, times out, or lacks
credentials. Nothing here ever fakes data: a source is either "live" (this session
fetched or validated it against the internet) or "cached" (the committed snapshot is
being used as a fallback).

Modes per source
----------------
* ``argo``    — live via the Argovis HTTPS API (no login) / argopy GDAC when reachable.
* ``oisst``   — live via NOAA ERDDAP griddap (no login): the NRT product first
  (current to ~1-2 days), then the Final product; each mirror is probed first so a
  blocked host can never stall the pass.
* ``copernicus`` (SSH) — live only when Copernicus Marine credentials exist
  (``COPERNICUSMARINE_USERNAME/PASSWORD``); otherwise cached.
* ``smap``    (SSS) — live only when NASA Earthdata credentials exist
  (``NASA_EARTHDATA_USERNAME/PASSWORD``); otherwise cached.

Every live fetch honours a hard timeout so the demo never hangs on a dead feed.
"""

from __future__ import annotations

import copy
import datetime as dt
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from src.utils.config import load_config, resolve_path
from src.utils.io import read_manifest

LIVE = "live"
CACHED = "cached"

SOURCES = ("argo", "oisst", "copernicus", "smap")
DEFAULT_WINDOW_DAYS = 400      # ~13 months so the last-12-months replay stays full
LIVE_ARGO_WINDOW_DAYS = 90     # the live Argo overlay only needs recent profiles (small + fast)
_DOWNLOAD_TIMEOUT_S = 90       # per-mirror read cap (connect cap is separate: _CONNECT_TIMEOUT_S)
_CONNECT_TIMEOUT_S = 10        # unreachable hosts fail in ~10s, not ~150s
_FETCH_TIMEOUT_S = 120         # hard cap on any one live fetch (argo/cmems/smap)
_PROBE_TIMEOUT_S = (4, 6)      # (connect, read) — the "are we online at all?" check

LIVE_DIR_HINT = "_live"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def rolling_window(now: dt.datetime | None = None, days: int = DEFAULT_WINDOW_DAYS) -> tuple[str, str]:
    """Inclusive (start, end) ISO dates ending today, so charts stay relative to now."""
    now = now or _now()
    start = now - dt.timedelta(days=days)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def probe_online(timeout=_PROBE_TIMEOUT_S) -> bool:
    """Cheap reachability probe against the two no-login live endpoints."""
    for url in ("https://argovis-api.colorado.edu/ping",
                "https://upwell.pfeg.noaa.gov/erddap/index.html"):
        try:
            with requests.get(url, timeout=timeout, stream=True) as r:
                if r.status_code < 500:
                    return True
        except Exception:
            pass
    return False


def _run_with_timeout(fn, timeout_s: int, *args, **kwargs):
    """Run ``fn`` on a daemon worker, raising TimeoutError if it exceeds ``timeout_s``.

    The worker is deliberately *not* joined afterwards: a genuinely stuck network call
    must not hold the demo UI hostage. Daemon threads are used so a refused call never
    blocks interpreter shutdown either (plain ``ThreadPoolExecutor`` registers an
    atexit join and would stall CI/CLI exits, which is preferable to avoid).
    """
    out: list = []

    def _runner() -> None:
        try:
            out.append(("ok", fn(*args, **kwargs)))
        except BaseException as exc:  # noqa: BLE001
            out.append(("err", exc))

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', 'live_fetch')} exceeded {timeout_s}s")
    status, value = out[0]
    if status == "err":
        raise value
    return value


# --------------------------------------------------------------------------- OISST (SST)
def _oisst_mirrors(cfg: dict) -> list[str]:
    """OISST ERDDAP griddap bases, tried in order (NRT first, then Final product).

    NCEI is deliberately absent: its public ERDDAP does not host these gridded ids
    (``info NCEI griddap/ncdcOisst21Agg`` is a 404), and upwell transparently
    redirects griddap *data* to coastwatch, so the direct coastwatch entries give
    every network its best chance.
    """
    default = [
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg.nc",
        "https://upwell.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg.nc",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.nc",
        "https://upwell.pfeg.noaa.gov/erddap/griddap/ncdcOisst21Agg.nc",
    ]
    cfgd = cfg.get("ingestion", {}).get("oisst", {}).get("mirrors")
    cfgd = [m.rstrip("/") for m in cfgd] if cfgd else []
    return cfgd or default


def _erddap_info_url(base: str) -> str:
    """The cheap ``/erddap/info/<dataset>/index.json`` probe URL for a griddap base.

    e.g. ``.../griddap/ncdcOisst21NrtAgg.nc`` -> ``.../info/ncdcOisst21NrtAgg/index.json``
    """
    dsid = base.rstrip("/").rsplit("/", 1)[-1].removesuffix(".nc")
    host = base.rstrip("/").rsplit("/erddap/", 1)[0] + "/erddap"
    return f"{host}/info/{dsid}/index.json"


def _probe_erddap_coverage(base: str, timeout=_PROBE_TIMEOUT_S) -> tuple[str | None, str]:
    """Fast reachability + coverage check for one ERDDAP mirror before the big download.

    Returns ``(coverage_end_iso, descriptor)`` where ``coverage_end_iso`` is the
    dataset's ``time_coverage_end`` (YYYY-MM-DD) or None when unknown, and
    ``descriptor`` is a short human reason for refusal/absence ("unreachable",
    "404 dataset not hosted", "no coverage attr", etc.).
    """
    info = _erddap_info_url(base)
    try:
        with requests.get(info, timeout=timeout,
                          headers={"Accept-Encoding": "identity"}) as r:
            if r.status_code == 404:
                return None, "dataset not hosted (404)"
            if r.status_code >= 500:
                return None, f"server error {r.status_code}"
            if r.status_code != 200:
                return None, f"http {r.status_code}"
            payload = r.json()
    except requests.exceptions.ConnectTimeout:
        return None, "unreachable (connect timeout)"
    except requests.exceptions.Timeout:
        return None, "unreachable (timeout)"
    except requests.exceptions.ConnectionError:
        return None, "unreachable (connection failed)"
    except ValueError:
        return None, "bad info response"
    except Exception:  # noqa: BLE001
        return None, "unreachable (probe error)"
    end = None
    for row in payload.get("table", {}).get("rows", []):
        if len(row) >= 5 and row[0] == "attribute" and row[1] == "NC_GLOBAL" \
                and row[2] == "time_coverage_end":
            # Layout: [..., 'Data Type', 'Value'] (post-ERDDAP 2.x) or older
            # [..., 'Value', 'Data Type']. Pick whichever cell looks like a date.
            end = next((c for c in (row[4], row[3]) if str(c).count("-") >= 2), None)
            if end:
                break
    if not end:
        return None, "no coverage attribute"
    return str(end)[:10], "ok"


def build_erddap_url(cfg: dict, base: str, start: str, end: str) -> str:
    region = cfg["region"]
    var = cfg["ingestion"]["oisst"]["variable"]
    box = (f"{var}[({start}T00:00:00Z):({end}T00:00:00Z)]"
           f"[(0.0):(0.0)]"
           f"[({region['south']}):({region['north']})][({region['west']}):({region['east']})]")
    return f"{base}?{box}"


def _download_stream(url: str, dest: Path, timeout: int, connect: int = _CONNECT_TIMEOUT_S) -> None:
    headers = {"Accept-Encoding": "identity"}
    # (connect, read) tuple: an unreachable host fails fast instead of eating the whole
    # read budget, so a dead mirror can't stall the live pass for the other sources.
    with requests.get(url, stream=True, headers=headers, timeout=(connect, timeout)) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=1 << 20)
    if dest.read_bytes()[:2] == b"\x1f\x8b":  # defensive: server may still gzip
        import gzip

        with gzip.open(dest, "rb") as gz, open(dest.with_suffix(".tmp"), "wb") as out:
            shutil.copyfileobj(gz, out, length=1 << 20)
        dest.with_suffix(".tmp").replace(dest)


def refresh_oisst(cfg: dict, live_dir: Path, start: str, end: str,
                  timeout: int = _DOWNLOAD_TIMEOUT_S) -> Path | None:
    """Download the rolling OISST window to ``live_dir/oisst_sst.nc``.

    Each mirror is first probed with its cheap ``info`` JSON (fast fail on down /
    absent hosts, and to read ``time_coverage_end``), then the request's ``end`` is
    clamped to that coverage so a current NRT product always resolves inside its own
    time axis. Tries every mirror and keeps a distinct reason for each; the final
    error is a one-line summary the side panel can show honestly.
    """
    var = cfg["ingestion"]["oisst"]["variable"]
    dest = live_dir / "oisst_sst.nc"
    reasons: list[str] = []
    for base in _oisst_mirrors(cfg):
        rest = base.rstrip("/").rsplit("/erddap/", 1)[-1]  # griddap/<dsid>.nc
        host = base.rstrip("/").rsplit("/erddap/", 1)[0].split("//")[-1]
        label = f"{host} {rest.removeprefix('griddap/').removesuffix('.nc')}"
        cov_end, why = _probe_erddap_coverage(base)
        if cov_end is None:
            reasons.append(f"{label}: {why}")
            continue
        mirror_end = min(end, cov_end)
        if mirror_end < start:
            reasons.append(f"{label}: ends {cov_end}")
            continue
        tmp = dest.with_suffix(".nc.part")
        try:
            _download_stream(build_erddap_url(cfg, base, start, mirror_end), tmp, timeout)
            with xr.open_dataset(tmp) as ds:
                if var not in ds or int(ds[var]["time"].size) < 30:
                    raise ValueError("archive too small / variable missing")
                if int(np.nanmean(ds[var].values)) <= 0:
                    raise ValueError("archive values look empty")
            tmp.replace(dest)
            return dest
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"{label}: {type(exc).__name__}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    summary = "; ".join(reasons)
    if len(summary) > 220:
        summary = summary[:217] + "…"
    raise RuntimeError(f"OISST live unavailable — {summary}")


# --------------------------------------------------------------------------- Argo
def refresh_argo(cfg: dict, start: str, end: str,
                 timeout: int = _FETCH_TIMEOUT_S) -> pd.DataFrame | None:
    """Fetch recent Argo profiles via the Argovis HTTPS API (no login).

    Uses a short rolling window (``LIVE_ARGO_WINDOW_DAYS``) on purpose: the live
    overlay only needs *recent* profiles, and a ~3-month window keeps the response
    small (a few MB) so it completes well inside the overall session budget. The
    committed snapshot already covers the historical record.
    """
    from src.ingestion.fetch_argo import fetch_profiles_argovis_api

    argo_start, argo_end = rolling_window(days=LIVE_ARGO_WINDOW_DAYS)
    live_cfg = copy.deepcopy(cfg)
    live_cfg["time_range"] = {"start": argo_start, "end": argo_end}

    def _fetch() -> pd.DataFrame:
        last_exc: Exception | None = None
        for _attempt in range(2):  # Argovis is occasionally flaky mid-stream — retry once
            try:
                df = fetch_profiles_argovis_api(live_cfg, argo_start, argo_end,
                                                timeout_s=60)
                if df.empty:
                    raise RuntimeError("Argovis returned zero profiles for the window")
                return df
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(f"Argovis gave up after 2 attempts: {last_exc}")

    return _run_with_timeout(_fetch, timeout)


# --------------------------------------------------------------------------- SSH / SSS
def _copernicus_has_credentials(cfg: dict) -> bool:
    from src.ingestion.fetch_copernicus import has_credentials

    return bool(has_credentials())


def refresh_copernicus(cfg: dict, live_dir: Path, start: str, end: str,
                       timeout: int = _FETCH_TIMEOUT_S) -> Path | None:
    """Subset recent CMEMS SLA into ``live_dir/cmems_ssh.nc`` (requires credentials)."""
    try:
        if not _copernicus_has_credentials(cfg):
            return None
        import copernicusmarine as cm
    except (ImportError, Exception):  # noqa: BLE001
        return None

    dest = live_dir / "cmems_ssh.nc"
    about = cfg["ingestion"]["copernicus"]
    region = cfg["region"]

    def _fetch() -> Path:
        tmp = dest.with_suffix(".nc.part")
        cm.subset(
            dataset_id=about["dataset_id"],
            variables=[about["variable"]],
            minimum_longitude=region["west"], maximum_longitude=region["east"],
            minimum_latitude=region["south"], maximum_latitude=region["north"],
            start_datetime=f"{start}T00:00:00", end_datetime=f"{end}T23:59:59",
            output_filename=tmp.name, output_directory=str(live_dir),
            force_download=True,
        )
        if not tmp.exists():
            raise RuntimeError("copernicusmarine produced no file")
        with xr.open_dataset(tmp) as ds:
            if int(ds[about["variable"]]["time"].size) < 1:
                raise ValueError("empty SSH archive")
        tmp.replace(dest)
        return dest

    try:
        return _run_with_timeout(_fetch, timeout)
    except Exception:  # noqa: BLE001
        return None


def refresh_smap(cfg: dict, live_dir: Path, start: str, end: str,
                 timeout: int = _FETCH_TIMEOUT_S) -> Path | None:
    """Stack recent SMAP granules into ``live_dir/smap_sss.nc`` (requires credentials)."""
    try:
        from src.ingestion.fetch_smap import has_credentials, fetch_striped, _load_granule

        if not has_credentials():
            return None
    except Exception:  # noqa: BLE001
        return None

    dest = live_dir / "smap_sss.nc"
    region = cfg["region"]

    def _fetch() -> Path:
        import xarray as _xr

        granules = fetch_striped(cfg_live_region(cfg, start, end), live_dir)
        dsets = [g for g in map(_load_granule, granules) if g is not None]
        if not dsets:
            raise RuntimeError("no usable SMAP granules")
        merged = _xr.concat(dsets, dim="time").sortby("time")
        merged.to_netcdf(dest)
        return dest

    try:
        return _run_with_timeout(_fetch, timeout)
    except Exception:  # noqa: BLE001
        return None


def cfg_live_region(cfg: dict, start: str, end: str) -> dict:
    """A deep copy of ``cfg`` with ``time_range`` set to the rolling window."""
    live_cfg = copy.deepcopy(cfg)
    live_cfg["time_range"] = {"start": start, "end": end}
    return live_cfg


# --------------------------------------------------------------------------- grids + collocation
def _sampler_from_netcdf(path: Path, var: str):
    """Build a collocation sampler from an arbitrary netCDF archive (live or cached)."""
    from src.collocation.collocate import GriddedSurfaceSampler

    with xr.open_dataset(path) as ds:
        da = ds[var]
        if "zlev" in da.dims and da.sizes["zlev"] == 1:
            da = da.isel(zlev=0)
        da = da.squeeze()
        lon = da["lon"].values if "lon" in da.coords else da["longitude"].values
        lat = da["lat"].values if "lat" in da.coords else da["latitude"].values
        times = da["time"].values
        values = np.asarray(da.values, dtype=float).copy()
    return GriddedSurfaceSampler(lon, lat, times, values)


def build_live_grid(path: Path, var: str, feat: str) -> pd.DataFrame:
    """Latest-frame surface grid DataFrame (lon, lat, feat) from a live archive."""
    with xr.open_dataset(path) as ds:
        da = ds[var].squeeze().isel(time=-1)
        lon = da["lon"].values if "lon" in da.coords else da["longitude"].values
        lat = da["lat"].values if "lat" in da.coords else da["latitude"].values
        vals = np.asarray(da.values, dtype=float)
    lon_g, lat_g = np.meshgrid(lon, lat, indexing="ij")
    df = pd.DataFrame({"lon": lon_g.ravel(), "lat": lat_g.ravel()})
    df[feat] = vals.ravel()
    return df


def _netcdf_vintage(path: Path, var: str) -> dict:
    with xr.open_dataset(path) as ds:
        times = pd.to_datetime(np.asarray(ds[var]["time"].values))
    return {"first": times.min().isoformat(), "last": times.max().isoformat(),
            "n": int(times.size)}


def collocate_live(cfg: dict, argo: pd.DataFrame, sst_path: Path | None,
                   extra_paths: dict[str, Path | None]) -> pd.DataFrame | None:
    """Collocate freshly-fetched Argo against active surface archives.

    ``extra_paths`` maps feat ("ssh"/"sss") -> its active (live or cached) archive.
    Returns the collocated DataFrame, or None when too few matches.
    """
    from src.collocation.collocate import (_sample_column, argo_to_target_frame,
                                          load_netcdf_sampler)

    window = int(cfg["collocation"]["temporal_window_days"])
    max_km = float(cfg["collocation"]["max_distance_km"])
    target, _ = argo_to_target_frame(argo, cfg)
    if target.empty:
        return None

    sst_sampler = _sampler_from_netcdf(sst_path, "sst") if sst_path else None
    if sst_sampler is None:
        return None
    out = target.copy()
    out["sst"], _ = _sample_column(out, sst_sampler, window, max_km)

    for feat, src in (("ssh", "copernicus"), ("sss", "smap")):
        path = extra_paths.get(feat)
        var = cfg["ingestion"][src]["variable"]
        if path is not None:
            sampler = _sampler_from_netcdf(path, var)
        else:
            sampler, _ = load_netcdf_sampler(cfg, src, var, window)
        if sampler is not None:
            w = int(cfg["ingestion"].get(src, {}).get("window_days", window))
            out[feat], _ = _sample_column(out, sampler, w, max_km)

    collocated = out[out["sst"].notna()].reset_index(drop=True)
    return collocated if not collocated.empty else None


# --------------------------------------------------------------------------- state
@dataclass
class LiveState:
    online: bool
    status: dict[str, str] = field(default_factory=dict)
    grids: dict[str, pd.DataFrame] = field(default_factory=dict)       # live grids present
    collocated: pd.DataFrame | None = None                             # live collocated, if any
    replay_cached_dir: str | None = None                               # override when archives renewed
    vintage: dict[str, dict] = field(default_factory=dict)             # per-source date coverage
    refreshed_utc: str = ""
    errors: dict[str, str] = field(default_factory=dict)               # last error per source

    def all_live(self) -> bool:
        return self.online and all(self.status.get(s) == LIVE for s in SOURCES)


def build_live_state(cfg: dict, now: dt.datetime | None = None,
                     include_argo: bool = True, budget_s: int = 120) -> LiveState:
    """Fetch what the internet allows right now; anything else stays cached fallback.

    All four sources are attempted **concurrently** under a single hard wall-clock
    ``budget_s`` so startup is bounded no matter how slow the slowest feed is.
    """
    now = now or _now()
    start, end = rolling_window(now)
    state = LiveState(online=False)
    state.refreshed_utc = now.isoformat(timespec="seconds")

    state.online = probe_online()
    if not state.online:
        state.status = {s: CACHED for s in SOURCES}
        for s in SOURCES:
            state.errors[s] = "no internet (probe failed)"
        return state

    live_dir = resolve_path(cfg, "cached_dir") / LIVE_DIR_HINT
    live_dir.mkdir(parents=True, exist_ok=True)

    # per-source workers; each returns (ok_summary) or raises TimeoutError/other
    def _oisst():
        p = refresh_oisst(cfg, live_dir, start, end)
        return {
            "sst": build_live_grid(p, "sst", "sst"),
            "vintage": _netcdf_vintage(p, "sst"),
            "path": p,
        }

    def _argo():
        df = refresh_argo(cfg, start, end)
        return {
            "df": df,
            "vintage": {
                "first": str(pd.to_datetime(df["time"].min(), utc=True)),
                "last": str(pd.to_datetime(df["time"].max(), utc=True)),
                "n": int(len(df)),
            },
        }

    def _copernicus():
        p = refresh_copernicus(cfg, live_dir, start, end)
        if p is None:
            raise RuntimeError("live CMEMS needs Copernicus Marine credentials")
        var = cfg["ingestion"]["copernicus"]["variable"]
        return {"ssh": build_live_grid(p, var, "ssh"),
                "vintage": _netcdf_vintage(p, var), "path": p}

    def _smap():
        p = refresh_smap(cfg, live_dir, start, end)
        if p is None:
            raise RuntimeError("live SMAP needs NASA/PODAAC credentials")
        var = cfg["ingestion"]["smap"]["variable"]
        return {"sss": build_live_grid(p, var, "sss"),
                "vintage": _netcdf_vintage(p, var), "path": p}

    jobs = {"oisst": _oisst}
    if include_argo:
        jobs["argo"] = _argo
    jobs["copernicus"] = _copernicus
    jobs["smap"] = _smap

    results: dict[str, dict] = {}

    # Launch every source concurrently on daemon workers (never joined at exit), then
    # harvest results under a single hard wall-clock deadline so a slow feed cannot
    # stall the demo: whatever has not finished by ``budget_s`` is recorded as timed out
    # and that source simply stays cached fallback.
    outbox: dict[str, tuple] = {}
    threads: dict[str, threading.Thread] = {}

    def _worker(name: str, fn) -> None:
        try:
            outbox[name] = ("ok", fn())
        except BaseException as exc:  # noqa: BLE001
            outbox[name] = ("err", str(exc)[:200])

    for name, fn in jobs.items():
        t = threading.Thread(target=_worker, args=(name, fn), daemon=True)
        t.start()
        threads[name] = t

    deadline = now + dt.timedelta(seconds=budget_s)
    for name, t in threads.items():
        remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining > 0:
            t.join(remaining)
        if t.is_alive():
            state.errors[name] = f"timed out after {budget_s}s"
            continue
        # The worker finished (possibly before the deadline ran out for this slot) —
        # honour its real result instead of assuming the whole budget was consumed.
        status, value = outbox[name]
        if status == "ok":
            results[name] = value
        else:
            state.errors[name] = value

    for name in SOURCES:
        ok = name in results
        state.status[name] = LIVE if ok else CACHED
        if ok:
            state.vintage[name] = results[name].get("vintage")

    if "oisst" in results:
        state.grids["sst"] = results["oisst"]["sst"]
        state.replay_cached_dir = str(live_dir)
    if "copernicus" in results:
        state.grids["ssh"] = results["copernicus"]["ssh"]
        state.replay_cached_dir = str(live_dir)
    if "smap" in results:
        state.grids["sss"] = results["smap"]["sss"]
        state.replay_cached_dir = str(live_dir)

    if "argo" in results and "oisst" in results:
        try:
            sst_path = results["oisst"]["path"]
            extras = {
                "ssh": results["copernicus"]["path"] if "copernicus" in results else None,
                "sss": results["smap"]["path"] if "smap" in results else None,
            }
            collocated = collocate_live(cfg, results["argo"]["df"], sst_path, extras)
            if collocated is not None and len(collocated) >= 5:
                state.collocated = collocated
        except Exception as exc:  # noqa: BLE001
            state.errors["collocation"] = str(exc)[:200]

    for s in SOURCES:
        state.status.setdefault(s, CACHED)
    return state


if __name__ == "__main__":  # quick CLI probe for debugging
    import sys

    cfg = load_config()
    st = build_live_state(cfg)
    print("online:", st.online)
    for s in SOURCES:
        print(f"  {s:10s} {st.status.get(s, '?'):<11s} vintage={st.vintage.get(s)} err={st.errors.get(s, '')[:120]}")
    print("live grids:", list(st.grids), "| live collocated rows:",
          None if st.collocated is None else len(st.collocated),
          "| replay override:", st.replay_cached_dir)
    sys.exit(0)