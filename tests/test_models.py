"""Unit tests for the model layer (features, split, input/output shapes).

Uses small in-memory fixtures only — no network, no real data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.datasets import build_feature_target_arrays, split_by_float, target_columns


def _fake_profiles(n_float: int = 4, per_float: int = 20):
    rows = []
    for f in range(n_float):
        for k in range(per_float):
            rows.append(
                {
                    "float_id": f"WMO{5900000 + f}",
                    "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=k * 9),
                    "lat": 10.0 + f,
                    "lon": 60.0 + f,
                    "month": 1 + (k % 12),
                    "sst": 27.0 + 0.5 * np.sin(k) + f,
                    "temp_0m": 27.0,
                    "temp_50m": 24.0,
                    "temp_100m": 21.0,
                    "temp_200m": 16.0,
                    "temp_500m": 11.0,
                    "temp_1000m": 6.0,
                }
            )
    return pd.DataFrame(rows)


def test_split_by_float_has_no_leakage():
    df = _fake_profiles(n_float=6, per_float=20)
    train, val, test = split_by_float(df, "float_id", 0.2, 0.2, random_state=7)
    train_ids = set(train["float_id"])
    val_ids = set(val["float_id"])
    test_ids = set(test["float_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(df["float_id"].unique())
    # every profile of a given float must live in exactly one partition
    for fid in df["float_id"].unique():
        n = len(df[df["float_id"] == fid])
        n_parts = sum(
            len(part[part["float_id"] == fid]) for part in (train, val, test)
        )
        assert n_parts == n


def test_target_columns_from_cfg():
    class CFG:
        pass

    cfg = {"depths": {"target_levels": [0, 50, 100, 200, 500, 1000]}}
    assert target_columns(cfg) == [
        "temp_0m", "temp_50m", "temp_100m", "temp_200m", "temp_500m", "temp_1000m",
    ]


def test_feature_target_arrays_shape_and_na_dropping():
    df = _fake_profiles()
    df.loc[df.index[3], "sst"] = np.nan
    df.loc[df.index[7], "temp_200m"] = np.nan
    X, Y, dropped = build_feature_target_arrays(
        df,
        ["lat", "lon", "month", "sst"],
        target_columns({"depths": {"target_levels": [0, 50, 100, 200, 500, 1000]}}),
    )
    assert dropped == 2
    assert X.shape[0] == len(df) - 2
    assert X.shape[1] == 4
    assert Y.shape[1] == 6
    assert np.isfinite(X).all() and np.isfinite(Y).all()


def test_feature_target_arrays_rejects_missing_columns():
    df = _fake_profiles()
    with pytest.raises(KeyError):
        build_feature_target_arrays(df, ["ssh", "sst"], ["temp_0m"])


def test_train_mlp_output_shape():
    """Tiny MLP train/predict round-trip: 6 targets out of a shared head."""
    cfg = {
        "model": {
            "random_state": 0,
            "mlp": {
                "hidden_units": [8, 4],
                "activation": "relu",
                "dropout": 0.0,
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.01,
                "patience": 2,
            },
        }
    }
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    Y = np.stack([X.mean(axis=1)] * 6, axis=1) + rng.normal(scale=0.05, size=(60, 6))

    from src.models.deep_model import build_mlp, fit_model

    model = build_mlp(4, 6, cfg)
    fit_model(model, X, Y, X[:20], Y[:20], cfg, "/tmp/opencode/_mlp_test.keras")
    pred = model.predict(X[:5])
    assert pred.shape == (5, 6)
    assert np.isfinite(pred).all()


def test_baselines_predict_6_targets():
    cfg = {
        "model": {
            "random_state": 0,
            "rf": {"n_estimators": 5, "max_depth": 3, "min_samples_leaf": 1, "n_jobs": 1},
            "xgb": {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3,
                    "subsample": 1.0, "colsample_bytree": 1.0},
        }
    }
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 4))
    Y = np.stack([X.sum(axis=1)] * 6, axis=1) + rng.normal(scale=0.05, size=(80, 6))

    from src.models.baseline import train_random_forest, train_xgboost

    rf = train_random_forest(X, Y, cfg)
    xgb = train_xgboost(X, Y, cfg)
    assert rf.predict(X[:5]).shape == (5, 6)
    assert xgb.predict(X[:5]).shape == (5, 6)