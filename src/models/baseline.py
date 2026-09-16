"""Baseline regressors: Random Forest and XGBoost.

Design decision (documented in README): each baseline predicts **all six depth
targets jointly** with a single multi-output model.

* Random Forest uses ``sklearn.multioutput.MultiOutputRegressor`` wrapping a single
  ``RandomForestRegressor`` (the forest structure is shared; only leaves differ).
* Gradient boosting uses ``xgboost.XGBRegressor`` with ``objective=reg:squarederror``
  and ``multi_strategy=multi_output_tree``, XGBoost's native multi-target mode.

A single multi-output model per family (rather than six independent models) keeps
training fast on free-tier compute and lets the deep MLP be compared apples-to-apples
(same input -> same six outputs in one forward pass).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.utils.config import load_config


def train_random_forest(X: np.ndarray, Y: np.ndarray, cfg: dict) -> MultiOutputRegressor:
    rf_cfg = cfg["model"]["rf"]
    base = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_leaf=rf_cfg.get("min_samples_leaf", 1),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=cfg["model"]["random_state"],
    )
    model = MultiOutputRegressor(base, n_jobs=1)
    model.fit(X, Y)
    return model


def train_xgboost(X: np.ndarray, Y: np.ndarray, cfg: dict):
    import xgboost as xgb

    xgb_cfg = cfg["model"]["xgb"]
    model = xgb.XGBRegressor(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        objective="reg:squarederror",
        multi_strategy="multi_output_tree",
        tree_method="hist",
        n_jobs=xgb_cfg.get("n_jobs", -1),
        random_state=cfg["model"]["random_state"],
        verbosity=0,
    )
    model.fit(X, Y)
    return model


def train_baselines(X: np.ndarray, Y: np.ndarray, cfg: dict) -> dict[str, object]:
    """Train the Random Forest + XGBoost baselines; returns {"rf": ..., "xgb": ...}."""
    return {"rf": train_random_forest(X, Y, cfg), "xgb": train_xgboost(X, Y, cfg)}