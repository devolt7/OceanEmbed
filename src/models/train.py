"""Training CLI — trains baselines (RF, XGBoost) and the deep MLP end-to-end.

Flow:
1. Load ``data/interim/collocated.parquet``.
2. Build features (month derived from float time; ssh/sss appended only when those
   sources were downloaded, read from the raw-source manifests).
3. Split **by Argo float ID** into train/val/test (no leakage — see README), persist
   the splits to ``data/processed/*.parquet`` for the evaluation stage.
4. Train Random Forest + XGBoost (multi-output baselines) and the Keras MLP.
5. Save model artifacts, the fitted input scaler, feature metadata and a config
   snapshot to ``models/``.

Run:   ``python -m src.models.train [--config config.bootstrap.yaml] [--fast]``
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import baseline as bl
from src.models import deep_model as dm
from src.utils.config import load_config, resolve_path, ensure_dirs
from src.utils.datasets import build_feature_target_arrays, split_by_float, target_columns
from src.utils.io import installed_surface_sources, save_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def prepare_frame(collocated: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = collocated.copy()
    if "month" not in df.columns:
        df["month"] = df["time"].dt.month.astype(int)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=None, help="path to config yaml")
    parser.add_argument("--fast", action="store_true",
                        help="tiny hyper-parameters for quick smoke runs (tests/CI)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.fast:
        cfg["model"]["rf"] = {"n_estimators": 30, "max_depth": 6, "min_samples_leaf": 2, "n_jobs": -1}
        cfg["model"]["xgb"] = {"n_estimators": 30, "max_depth": 4, "learning_rate": 0.2,
                               "subsample": 1.0, "colsample_bytree": 1.0}
        cfg["model"]["mlp"]["hidden_units"] = [16, 8]
        cfg["model"]["mlp"]["epochs"] = 20
        cfg["model"]["mlp"]["patience"] = 5
    ensure_dirs(cfg)

    collocated_path = resolve_path(cfg, "interim_dir") / "collocated.parquet"
    if not collocated_path.exists():
        log.error("collocated.parquet missing; run `python -m src.collocation.collocate` first.")
        return 1
    df = prepare_frame(pd.read_parquet(collocated_path), cfg)
    log.info("loaded %d collocated profiles", len(df))

    available = installed_surface_sources(cfg)
    features = list(cfg["model"]["features"])
    for col in cfg["model"].get("features_optional", []):
        if col in available and col not in features:
            features.append(col)
    targets = target_columns(cfg)
    log.info("features=%s targets=%s", features, targets)

    train, val, test = split_by_float(
        df, val_fraction=cfg["model"]["val_fraction"],
        test_fraction=cfg["model"]["test_fraction"],
        random_state=cfg["model"]["random_state"],
    )
    save_df(train, cfg, "processed_dir", "train")
    save_df(val, cfg, "processed_dir", "val")
    save_df(test, cfg, "processed_dir", "test")
    log.info(
        "split by float ID -> train=%d (%d floats), val=%d, test=%d",
        len(train), train["float_id"].nunique(), len(val), len(test),
    )

    X_train, Y_train, dropped_tr = build_feature_target_arrays(train, features, targets)
    X_val, Y_val, dropped_va = build_feature_target_arrays(val, features, targets)
    X_test, Y_test, dropped_te = build_feature_target_arrays(test, features, targets)
    log.info("rows dropped for NaN features/targets: train=%d val=%d test=%d",
             dropped_tr, dropped_va, dropped_te)
    if len(X_train) < 30:
        log.error("only %d training rows — insufficient. Enlarge region/time range.", len(X_train))
        return 1

    model_dir = resolve_path(cfg, "model_dir")
    meta = {
        "features": features,
        "targets": targets,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "surface_sources": available,
        "trained_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
    }
    (model_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2))

    # --- baselines ---
    log.info("training Random Forest baseline...")
    rf = bl.train_random_forest(X_train, Y_train, cfg)
    joblib.dump(rf, model_dir / "baseline_rf.joblib")
    log.info("training XGBoost baseline...")
    xgb = bl.train_xgboost(X_train, Y_train, cfg)
    joblib.dump(xgb, model_dir / "baseline_xgb.joblib")

    # --- deep MLP ---
    log.info("training deep MLP...")
    mlp_path = str(model_dir / "deep_mlp.keras")
    tf_log = logging.getLogger("tensorflow")
    tf_log.setLevel(logging.ERROR)
    model, scaler, target_mean = dm.train_deep_mlp(X_train, Y_train, X_val, Y_val, cfg, mlp_path)
    joblib.dump(scaler, model_dir / "scaler_input.joblib")
    meta["targets_mean"] = [float(round(float(m), 4)) for m in target_mean]
    (model_dir / "feature_meta.json").write_text(json.dumps(meta, indent=2))

    model.save(mlp_path)  # ensure best checkpoint actually exists
    log.info("model artifacts saved to %s", model_dir)

    # quick sanity scores on the held-out test set
    pred_rf = rf.predict(X_test)
    pred_xgb = xgb.predict(X_test)
    pred_mlp = model.predict(scaler.transform(X_test), verbose=0) + target_mean

    def mae_row(name, pred):
        err = np.abs(pred - Y_test).mean(axis=0)
        return " ".join(f"{t}:{m:.2f}°C" for t, m in zip(targets, err))

    log.info("test MAE  RF : %s", mae_row("rf", pred_rf))
    log.info("test MAE  XGB: %s", mae_row("xgb", pred_xgb))
    log.info("test MAE  MLP: %s", mae_row("mlp", pred_mlp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))