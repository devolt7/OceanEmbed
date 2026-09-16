"""Dataset splitting utilities.

The scientifically correct split for this problem is **by Argo float ID** (WMO
platform number): every profile of a given float goes into a single partition so
the same physical float never appears in both train and test. Random row splits
would leak information because successive profiles of the same float are highly
autocorrelated. This splitter is unit-tested in ``tests/test_models.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import load_config


def build_feature_target_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and extract feature / target matrices with NaN rows dropped.

    The optional satellite features are skipped when their columns don't exist
    (degraded/collocation-less mode), which is why the caller passes the column
    list ``feature_cols`` explicitly.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns missing from dataframe: {missing}")
    missing_t = [c for c in target_cols if c not in df.columns]
    if missing_t:
        raise KeyError(f"target columns missing from dataframe: {missing_t}")

    X = df[feature_cols].to_numpy(dtype=float)
    Y = df[target_cols].to_numpy(dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    n_dropped = int((~valid).sum())
    return X[valid], Y[valid], n_dropped


def split_by_float(
    df: pd.DataFrame,
    float_id_col: str = "float_id",
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a profile dataframe into train/val/test by float ID (no leakage).

    Returns (train, val, test) dataframes. Every float appears in exactly one
    partition. If fewer than 3 distinct floats are present, falls back to keep
    one float per partition.
    """
    rng = np.random.default_rng(random_state)
    float_ids = np.asarray(sorted(df[float_id_col].unique()))
    rng.shuffle(float_ids)

    n_val = max(1, int(round(len(float_ids) * val_fraction)))
    n_test = max(1, int(round(len(float_ids) * test_fraction)))
    n_train = max(0, len(float_ids) - n_val - n_test)

    if len(float_ids) >= 3:
        val_ids = set(float_ids[:n_val])
        test_ids = set(float_ids[n_val:n_val + n_test])
        train_ids = set(float_ids[n_val + n_test:])
    else:
        val_ids = {float_ids[0]} if len(float_ids) >= 1 else set()
        test_ids = {float_ids[1]} if len(float_ids) >= 2 else set()
        train_ids = set(float_ids) - val_ids - test_ids

    train = df[df[float_id_col].isin(train_ids)].copy()
    val = df[df[float_id_col].isin(val_ids)].copy()
    test = df[df[float_id_col].isin(test_ids)].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("split_by_float produced an empty partition; need >= 3 floats")

    return train, val, test


def feature_columns(cfg: dict[str, Any], available_surface: list[str]) -> list[str]:
    """Effective feature columns for this run.

    Base features come from ``cfg.model.features``; if the optional satellite
    sources (ssh/sss) were ingested they are appended automatically.
    """
    base = list(cfg["model"]["features"])
    features = list(base)
    for col in cfg["model"].get("features_optional", []):
        if col in available_surface and col not in features:
            features.append(col)
    return features


def target_columns(cfg: dict[str, Any]) -> list[str]:
    levels = cfg["depths"]["target_levels"]
    return [f"temp_{int(l)}m" for l in levels]