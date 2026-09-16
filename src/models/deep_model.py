"""Deep learning model for OceanEmbed: multi-output MLP (required) + optional CNN-LSTM.

The required model is a fully-connected MLP that receives the surface feature vector
-- ``[lat, lon, month, sst, (ssh), (sss)]`` -- and predicts all six depth temperatures
in a **single forward pass** through a shared multi-output regression head.

A stretch-goal CNN-LSTM variant is also implemented: given a small space-time patch of
SST around each point, it runs a CNN encoder followed by a bidirectional LSTM over the
temporal axis and merges the patch embedding with the point features before the shared
regression head. It is enabled only when ``cfg.model.mlp.use_cnn_lstm`` is true and the
collocated dataset provides ``sst_patch`` arrays (default off in the MVP).

TensorFlow / Keras 3 backend (``tensorflow-cpu``), MSE loss, Adam optimiser, early
stopping on validation loss, best-checkpoint saving to ``models/deep_mlp.keras``.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_config


def build_mlp(n_features: int, n_targets: int, cfg: dict) -> tf.keras.Model:
    mlp_cfg = cfg["model"]["mlp"]
    inputs = tf.keras.Input(shape=(n_features,), name="surface_features")
    x = inputs
    for units in mlp_cfg["hidden_units"]:
        x = tf.keras.layers.Dense(units, activation=mlp_cfg["activation"])(x)
        if mlp_cfg.get("dropout", 0) > 0:
            x = tf.keras.layers.Dropout(mlp_cfg["dropout"])(x)
    outputs = tf.keras.layers.Dense(n_targets, name="depth_temperatures")(x)
    model = tf.keras.Model(inputs, outputs, name="deep_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=mlp_cfg["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def build_cnn_lstm(
    n_features: int, n_targets: int, patch_shape: tuple[int, int, int], cfg: dict
) -> tf.keras.Model:
    """Optional spatial-patch model: CNN over SST patch + BiLSTM over time + MLP head.

    ``patch_shape`` is ``(n_time_steps, patch_height, patch_width)`` sampled around
    the float's grid cell from the surface SST stack. Each time step's spatial patch
    is encoded by a small per-timestep CNN (TimeDistributed), the per-timestep
    embeddings are run through a bidirectional LSTM, and the resulting sequence
    embedding is concatenated with the point feature vector before the shared head.
    """
    f_cfg = cfg["model"]["mlp"]
    T, H, W = patch_shape
    features_in = tf.keras.Input(shape=(n_features,), name="surface_features")
    patch_in = tf.keras.Input(shape=(T, H, W, 1), name="sst_patch")

    def cnn_block():
        inp = tf.keras.Input(shape=(H, W, 1))
        z = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same")(inp)
        z = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(z)
        z = tf.keras.layers.GlobalAveragePooling2D()(z)
        return tf.keras.Model(inp, z)

    patch_enc = tf.keras.layers.TimeDistributed(cnn_block(), name="cnn_time_encoder")(patch_in)
    seq = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16), name="bi_lstm")(patch_enc)

    merged = tf.keras.layers.Concatenate()([features_in, seq])
    for units in f_cfg["hidden_units"]:
        merged = tf.keras.layers.Dense(units, activation=f_cfg["activation"])(merged)
        if f_cfg.get("dropout", 0) > 0:
            merged = tf.keras.layers.Dropout(f_cfg["dropout"])(merged)
    outputs = tf.keras.layers.Dense(n_targets, name="depth_temperatures")(merged)
    model = tf.keras.Model(inputs=[features_in, patch_in], outputs=outputs, name="cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=f_cfg["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: dict,
    checkpoint_path: str,
) -> tf.keras.callbacks.History:
    mlp_cfg = cfg["model"]["mlp"]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=mlp_cfg["patience"], restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=mlp_cfg["epochs"],
        batch_size=mlp_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )
    return history


def train_deep_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    cfg: dict,
    out_path: str,
) -> tuple[tf.keras.Model, StandardScaler, np.ndarray]:
    """Train+persist the MLP; returns the best model, input scaler and target mean.

    The network predicts **centred targets** (Y - train mean). The caller must add the
    returned ``target_mean`` back to the predictions at inference time — this sharpens
    the fit on the residual (surface-driven) signal, which is what the "MLP beats a
    flat climatology" demo relies on.
    """
    target_mean = Y_train.mean(axis=0)
    scaler = StandardScaler().fit(np.vstack([X_train, X_val]))
    model = build_mlp(X_train.shape[1], Y_train.shape[1], cfg)
    fit_model(model, scaler.transform(X_train), Y_train - target_mean,
              scaler.transform(X_val), Y_val - target_mean, cfg, out_path)
    best = tf.keras.models.load_model(out_path)
    return best, scaler, target_mean