"""Unit tests for the collocation engine (KDTree + depth interpolation).

These tests use small in-memory fixtures and never touch the network.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.geo import (
    build_grid_kdtree,
    colocate_time_2d,
    haversine_km,
    interpolate_profile,
    nearest_grid_cell,
    project_xy,
)


def test_haversine_known_distance():
    # ~111.2 km per degree along the equator (slight formula rounding)
    d = haversine_km(0.0, 0.0, 1.0, 0.0)
    assert 110.0 < d < 112.0


def test_kdtree_nearest_cell_returns_exact_grid_index():
    lon = np.array([-10.0, 0.0, 10.0])
    lat = np.array([-10.0, 0.0, 10.0])
    kd = build_grid_kdtree(lon, lat)
    i, j, dist = nearest_grid_cell(lon, lat, 9.9, 10.1, kd, max_distance_km=np.inf)
    assert (i, j) == (2, 2)
    assert dist < 30.0


def test_nearest_grid_cell_rejects_out_of_radius():
    lon = np.array([50.0, 51.0])
    lat = np.array([10.0, 11.0])
    kd = build_grid_kdtree(lon, lat)
    # 70E is ~19 deg (~1900 km) from 51E at 10N -> clearly beyond 1000 km
    i, j, dist = nearest_grid_cell(lon, lat, 70.0, 10.0, kd, max_distance_km=1000.0)
    assert i == -1 and j == -1
    assert dist > 1000.0


def test_project_xy_returns_shaped_array():
    xy = project_xy(np.array([0.0, 10.0]), np.array([0.0, 10.0]))
    assert xy.shape == (2, 2)
    assert np.all(np.isfinite(xy))


def test_interpolate_profile_at_target_levels():
    depth = np.array([0, 50, 100, 150, 200, 500, 1000])
    temp = np.array([28.0, 26.0, 22.0, 18.0, 15.0, 10.0, 5.0])
    out = interpolate_profile(depth, temp, np.array([0, 50, 100, 200, 500, 1000]), min_profile_depth_m=500)
    assert np.allclose(out, [28.0, 26.0, 22.0, 15.0, 10.0, 5.0], atol=1e-6)


def test_interpolate_profile_rejects_too_shallow():
    depth = np.array([0, 50, 100])
    temp = np.array([28, 26, 22])
    with pytest.raises(ValueError):
        interpolate_profile(depth, temp, np.array([0, 50, 100]), min_profile_depth_m=500)


def test_interpolate_profile_no_invented_deep_values():
    depth = np.array([0, 50, 500])
    temp = np.array([28, 26, 10])
    out = interpolate_profile(depth, temp, np.array([0, 1000]), min_profile_depth_m=400)
    assert out[0] == 28.0
    assert np.isnan(out[1])  # 1000 m is below the profile -> NaN, never invented


def test_interpolate_profile_shallow_clamp_within_30m():
    depth = np.array([5, 50, 500])
    temp = np.array([27.5, 26.0, 10.0])
    out = interpolate_profile(depth, temp, np.array([0, 50]), min_profile_depth_m=400)
    assert out[0] == pytest.approx(27.5)  # clamped to the real shallowest observation


def test_colocate_time_2d_selects_within_window():
    profiles = np.array(["2021-01-01", "2021-01-10", "2021-02-01", "2021-02-10"],
                        dtype="datetime64[D]")
    surface = np.array(["2021-01-02", "2021-01-09", "2021-01-31"], dtype="datetime64[D]")
    idx = colocate_time_2d(profiles, surface, window_days=2)
    # 2021-01-01->0, 2021-01-10->1, 2021-02-01->2 (1 day from Jan 31), 2021-02-10 out of window
    assert idx.tolist() == [0, 1, 2, -1]


def test_colocate_time_2d_empty():
    assert colocate_time_2d(np.array([], dtype="datetime64[D]"),
                            np.array(["2021-01-01"], dtype="datetime64[D]"),
                            2).size == 0