"""Geospatial helpers: haversine distance, KDTree grid lookup, profile interpolation.

All functions are pure and dependency-light (numpy/scipy only) so they can be
unit-tested without any network access.
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0


def _radians(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    return np.deg2rad([lon, lat])


def project_xy(lon: np.ndarray | float, lat: np.ndarray | float) -> np.ndarray:
    """Project (lon, lat) in degrees to local planar (x, y) in km.

    Uses an equirectangular projection anchored at latitude 0; adequate for the
    small regional boxes OceanEmbed targets. Returns shape (2, N) in km.
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    x = EARTH_RADIUS_KM * np.deg2rad(lon) * np.cos(np.deg2rad(lat))
    y = EARTH_RADIUS_KM * np.deg2rad(lat)
    return np.vstack([x, y])


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two points (degrees)."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def build_grid_kdtree(lon: np.ndarray, lat: np.ndarray) -> cKDTree:
    """Build a KDTree over grid-cell centres from 1-D longitude/latitude vectors.

    The tree is built on projected (x, y) km coordinates so nearest-neighbour
    queries return cells by physical proximity, not raw degrees.
    """
    lon_g, lat_g = np.meshgrid(lon, lat, indexing="ij")
    pts = project_xy(lon_g.ravel(), lat_g.ravel()).T
    return cKDTree(pts)


def nearest_grid_cell(
    lon: np.ndarray,
    lat: np.ndarray,
    query_lon: float,
    query_lat: float,
    kdtree: cKDTree | None = None,
    max_distance_km: float = np.inf,
) -> tuple[int, int, float]:
    """Nearest grid cell for a query point.

    Returns ``(i, j, distance_km)`` where ``i`` indexes ``lon`` and ``j`` indexes
    ``lat``. ``i``/``j`` are ``-1`` when no cell is within ``max_distance_km``.
    """
    if kdtree is None:
        kdtree = build_grid_kdtree(lon, lat)
    q = project_xy(query_lon, query_lat).reshape(1, 2)
    dist, idx = kdtree.query(q, k=1)
    if dist > max_distance_km:
        return -1, -1, float(dist)
    # tree index layout: idx = i * len(lat) + j  (i indexes lon, j indexes lat)
    i, j = np.unravel_index(idx, (len(lon), len(lat)))
    return int(i), int(j), float(dist)


def interpolate_profile(
    depth: np.ndarray,
    temperature: np.ndarray,
    target_levels: np.ndarray,
    min_profile_depth_m: float,
    top_clamp_m: float = 30.0,
) -> np.ndarray:
    """Interpolate an Argo (depth, temperature) profile onto target levels.

    - Raises ``ValueError`` if the profile does not reach ``min_profile_depth_m``.
    - Target levels **below** the profile's deepest sample become NaN (no invented
      deep values).
    - Target levels **above** the profile's shallowest sample are clamped to the
      shallowest real observation when within ``top_clamp_m`` of the surface
      (standard Argo practice — the shallowest Argo level is typically ~5 m);
      otherwise they become NaN.
    """
    depth = np.asarray(depth, dtype=float).ravel()
    temperature = np.asarray(temperature, dtype=float).ravel()

    if len(depth) == 0:
        raise ValueError("empty profile")
    if np.isnan(temperature).all():
        raise ValueError("profile has no valid temperature data")

    finite = np.isfinite(depth) & np.isfinite(temperature) & (depth >= 0)
    if finite.sum() < 2:
        raise ValueError("profile has fewer than 2 valid (depth, temperature) points")

    depth = depth[finite]
    temperature = temperature[finite]

    if depth[-1] < min_profile_depth_m:
        raise ValueError(
            f"profile max depth {depth[-1]:.0f} m < required {min_profile_depth_m:.0f} m"
        )

    if np.any(np.diff(depth) < 0):  # Argo depths are ascending; normalise defensively
        order = np.argsort(depth)
        depth, temperature = depth[order], temperature[order]

    levels = np.asarray(target_levels, dtype=float)
    interp = interpolate.interp1d(depth, temperature, kind="linear", bounds_error=False)
    out = np.asarray(interp(levels), dtype=float)
    for k, level in enumerate(levels):
        if level > depth[-1]:
            out[k] = np.nan
        elif level < depth[0]:
            out[k] = temperature[0] if (depth[0] - level) <= top_clamp_m else np.nan
    return out


def colocate_time_2d(
    profile_times: np.ndarray,
    surface_times: np.ndarray,
    window_days: int,
) -> np.ndarray:
    """For each profile time, index of the surface snapshot within +/- window days.

    Returns int array of indices into ``surface_times`` (or -1 when out of window).
    Times are numpy datetime64. The snapshot with the smallest temporal offset wins.
    """
    profile_times = np.atleast_1d(profile_times)
    surface_times = np.asarray(surface_times)
    if surface_times.size == 0 or profile_times.size == 0:
        return np.full(profile_times.size, -1, dtype=int)

    window = np.timedelta64(window_days, "D")
    out = []
    for pt in profile_times:
        diff = np.abs(surface_times - pt)
        within = diff <= window
        if not within.any():
            out.append(-1)
        else:
            out.append(int(np.argmin(diff)))
    return np.asarray(out, dtype=int)


def resample_to_daily(values: np.ndarray, times: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Mean of ``values`` grouped by calendar date (used to collapse sub-daily grids)."""
    out = np.full(len(days), np.nan)
    day_u = np.unique(days)
    for d in day_u:
        mask = days == d
        with np.errstate(all="ignore"):
            out[days == d] = np.nanmean(values[mask])
    return out