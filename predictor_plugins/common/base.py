#!/usr/bin/env python
"""Base predictor plugin abstractions.

Provides documented base classes to eliminate duplication across concrete
predictor plugins (ANN, CNN, LSTM, Transformer, N-BEATS).

Interface expected by pipeline:
    build_model(input_shape, x_train, config)
    train(x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config)
    predict_with_uncertainty(x_test, mc_samples=...)
    save(path) / load(path)
    calculate_mae(y_true, y_pred)
    calculate_r2(y_true, y_pred)
    set_params / get_debug_info / add_debug_info

Design:
  - BasePredictorPlugin: parameter handling + generic metrics.
  - BaseKerasPredictor: adds generic Keras save/load + common callbacks & train loop.
  - BaseBayesianKerasPredictor: adds KL weight variable + MC uncertainty.
  - BaseDeterministicKerasPredictor: zero-uncertainty implementation.

Concrete plugins only implement:
  * plugin_params (class attr)
  * plugin_debug_vars (class attr)
  * build_model (sets self.model & self.output_names)

All other behaviors inherited, drastically reducing code duplication.
"""
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LambdaCallback
import tensorflow.keras.backend as K

from .losses import (
    mae_magnitude, r2_metric, composite_loss_multihead as composite_loss,
    compute_mmd
)
from .callbacks import (
    ReduceLROnPlateauWithCounter, EarlyStoppingWithPatienceCounter
)
from .bayesian import build_kl_anneal_callback, predict_mc_welford

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------
class BasePredictorPlugin:
    """Holds parameter management + generic metrics shared by all plugins."""
    plugin_params: Dict[str, Any] = {"predicted_horizons": [1]}
    plugin_debug_vars: List[str] = ["predicted_horizons"]

    def __init__(self, config: Dict[str, Any] | None = None):
        self.params = self.plugin_params.copy()
        if config:
            self.params.update(config)
        self.model: Model | None = None
        self.output_names: List[str] = []

    # --- Param / debug API ---
    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            self.params[k] = v
    def get_debug_info(self) -> Dict[str, Any]:
        return {k: self.params.get(k) for k in self.plugin_debug_vars}
    def add_debug_info(self, debug_info: Dict[str, Any]):
        debug_info.update(self.get_debug_info())

    # --- Metrics (magnitude = first column) ---
    def _ensure_two_cols(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            arr = arr.reshape(-1, 1)
            arr = np.concatenate([arr, np.zeros_like(arr)], axis=1)
        return arr
    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = self._ensure_two_cols(y_true)
        y_pred = self._ensure_two_cols(y_pred)
        mae = float(np.mean(np.abs(y_true[:, 0] - y_pred[:, 0])))
        print(f"MAE (magnitude): {mae}")
        return mae
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = self._ensure_two_cols(y_true)
        y_pred = self._ensure_two_cols(y_pred)
        ss_res = np.sum((y_true[:, 0] - y_pred[:, 0]) ** 2)
        ss_tot = np.sum((y_true[:, 0] - np.mean(y_true[:, 0])) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-12))
        print(f"R2 (magnitude): {r2}")
        return r2

    # --- Abstract placeholders (implemented in subclasses) ---
    def build_model(self, input_shape: Tuple[int, ...], x_train, config: Dict[str, Any]):  # noqa: D401
        raise NotImplementedError
    def predict_with_uncertainty(self, x_test, mc_samples: int = 50):  # noqa: D401
        raise NotImplementedError
    def save(self, file_path: str):  # noqa: D401
        raise NotImplementedError
    def load(self, file_path: str):  # noqa: D401
        raise NotImplementedError


class BaseKerasPredictor(BasePredictorPlugin):
    """Adds shared Keras training loop, callbacks, save/load.

    Subclasses must implement build_model to populate self.model & self.output_names.
    """
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)

    # --- Custom objects reused across all models ---
    def get_custom_objects(self):
        return {
            'composite_loss': composite_loss,
            'compute_mmd': compute_mmd,
            'r2_metric': r2_metric,
            'mae_magnitude': mae_magnitude,
        }

    # --- Callbacks factory (Bayesian variant will extend) ---
    def _build_callbacks(self):
        return [
            EarlyStoppingWithPatienceCounter(
                monitor='val_loss',
                patience=self.params.get('early_patience', 10),
                restore_best_weights=True,
                min_delta=1e-8,
                verbose=1
            ),
            ReduceLROnPlateauWithCounter(
                monitor='val_loss',
                factor=0.3,
                min_delta=1e-8,
                patience=max(1, self.params.get('early_patience', 10) // 4),
                verbose=1
            ),
            LambdaCallback(on_epoch_end=lambda e, l: print(
                f"Epoch {e+1}: LR={K.get_value(self.model.optimizer.learning_rate):.6f}")),
        ]

    # --- Generic train loop ---
    def train(self, x_train, y_train, epochs, batch_size, threshold_error, x_val, y_val, config):
        if config:
            self.params.update(config)
        if 'predicted_horizons' not in self.params or 'plotted_horizon' not in self.params:
            raise ValueError("Config must contain 'predicted_horizons' and 'plotted_horizon'.")
        ph = self.params['predicted_horizons']
        plotted = self.params['plotted_horizon']
        if plotted not in ph:
            raise ValueError('plotted_horizon must be one of predicted_horizons')
        plotted_index = ph.index(plotted)
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
            raise TypeError('y_train/y_val must be dicts mapping output names -> arrays')
        callbacks = self._build_callbacks()
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1,
        )
        mc = self.params.get('mc_samples', 50)
        train_preds, train_unc = self.predict_with_uncertainty(x_train, mc)
        val_preds, val_unc = self.predict_with_uncertainty(x_val, mc)
        try:
            self.calculate_mae(y_train[self.output_names[plotted_index]], train_preds[plotted_index])
            self.calculate_r2(y_train[self.output_names[plotted_index]], train_preds[plotted_index])
        except Exception as e:  # pragma: no cover (defensive)
            print(f'Metric calculation error: {e}')
        return history, train_preds, train_unc, val_preds, val_unc

    # --- Persistence ---
    def save(self, file_path: str):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")
    def load(self, file_path: str):
        self.model = load_model(file_path, custom_objects=self.get_custom_objects())
        print(f"Predictor model loaded from {file_path}")


class BaseBayesianKerasPredictor(BaseKerasPredictor):
    """Adds KL annealing variable + MC uncertainty to Keras base predictor."""
    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="kl_weight_var")
        # Patch DenseFlipout add_variable once (TFP quirk for some versions)
        if not hasattr(tfp.layers.DenseFlipout, "_already_patched_add_variable"):
            def _patched_add_variable(layer_instance, name, shape, dtype, initializer, trainable, **kwargs):
                return layer_instance.add_weight(name=name, shape=shape, dtype=dtype, initializer=initializer, trainable=trainable, **kwargs)
            tfp.layers.DenseFlipout.add_variable = _patched_add_variable  # type: ignore
            tfp.layers.DenseFlipout._already_patched_add_variable = True  # type: ignore

    def _build_callbacks(self):
        base = super()._build_callbacks()
        base.append(build_kl_anneal_callback(self, self.params.get('kl_weight', 1e-3), self.params.get('kl_anneal_epochs', 10)))
        return base

    def predict_with_uncertainty(self, x_test, mc_samples: int = 50):
        return predict_mc_welford(self.model, x_test, mc_samples)


class BaseDeterministicKerasPredictor(BaseKerasPredictor):
    """Deterministic variant returning zero uncertainties."""
    def predict_with_uncertainty(self, x_test, mc_samples: int = 1):
        preds = self.model.predict(x_test, verbose=0)
        preds = [preds] if isinstance(preds, np.ndarray) else preds
        zeros = [np.zeros_like(p) for p in preds]
        return preds, zeros

__all__ = [
    'BasePredictorPlugin', 'BaseKerasPredictor', 'BaseBayesianKerasPredictor', 'BaseDeterministicKerasPredictor'
]
