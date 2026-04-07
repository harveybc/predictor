"""Binary classification mixin for predictor plugins.

Provides train/save/load overrides for binary classification models.
Must be placed BEFORE the regression parent in MRO so that its ``train()``
takes precedence over ``BaseKerasPredictor.train()``::

    class Plugin(BinaryMixin, ANNPlugin):
        ...
"""
from __future__ import annotations

import json
import numpy as np

VALID_SIGNAL_TYPES = ("buy_entry", "sell_entry", "buy_exit", "sell_exit")

FEATURE_COLUMNS = [
    "typical_price", "hod_sin", "hod_cos", "dow_sin", "dow_cos",
    "dom_sin", "dom_cos", "moy_sin", "moy_cos",
    "rolling_std_24", "rolling_ema_24", "price_minus_ema",
]


def _as_bool(v, default=False):
    """Safely cast various types to bool (handles string/int/float/None)."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off", ""):
            return False
        return default
    return bool(v)


class BinaryMixin:
    """Overrides train/save/load for binary classification plugins.

    The base ``BaseKerasPredictor.train()`` requires ``predicted_horizons``
    and ``plotted_horizon`` in config and computes MAE/R² (regression
    metrics).  This mixin replaces that with binary-appropriate logic
    (accuracy, positive rate) and drops the horizon requirement.
    """

    def train(self, x_train, y_train, epochs, batch_size, threshold_error,
              x_val, y_val, config):
        if config:
            self.params.update(config)
        if not isinstance(y_train, dict) or not isinstance(y_val, dict):
            raise TypeError(
                "y_train/y_val must be dicts mapping output names -> arrays"
            )

        # For single-output binary models, extract the array from the dict
        # to avoid optree/Keras3 tree_map issues with single-key dicts.
        if len(y_train) == 1:
            y_train_fit = next(iter(y_train.values()))
            y_val_fit = next(iter(y_val.values()))
        else:
            y_train_fit = y_train
            y_val_fit = y_val

        callbacks = self._build_callbacks()

        # Resource snapshot at fit start (same as base class).
        try:
            from ..common.callbacks import capture_resource_snapshot
            snap = capture_resource_snapshot(
                include_gpu=bool(self.params.get("memory_log_gpu", True)),
                include_gc=bool(self.params.get("memory_log_gc", False)),
            )
            print(
                f"[RESOURCE] fit_start ts={snap.ts:.3f} "
                f"VmRSS_kB={snap.rss_kb} VmHWM_kB={snap.hwm_kb} "
                f"gpu_current_B={snap.gpu_current_bytes} "
                f"gpu_peak_B={snap.gpu_peak_bytes} gc={snap.gc_counts}"
            )
        except Exception as e:
            print(f"[RESOURCE] fit_start snapshot failed: {e}")

        _quiet = self.params.get("quiet", False) or self.params.get("quiet_mode", False)
        fit_verbose = 0 if _quiet else 1
        history = self.model.fit(
            x_train,
            y_train_fit,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val_fit),
            callbacks=callbacks,
            verbose=fit_verbose,
            shuffle=False,
        )

        # Post-fit predictions (identical to base except metrics at the end).
        disable_postfit = bool(
            self.params.get("disable_postfit_uncertainty", False)
        )
        pred_bs = int(self.params.get("predict_batch_size", 0) or 0)
        if pred_bs <= 0:
            pred_bs = (
                int(batch_size)
                if isinstance(batch_size, int) and batch_size > 0
                else 256
            )

        if disable_postfit:
            train_preds = self.model.predict(
                x_train, batch_size=pred_bs, verbose=0
            )
            val_preds = self.model.predict(
                x_val, batch_size=pred_bs, verbose=0
            )
            train_preds = (
                [train_preds]
                if isinstance(train_preds, np.ndarray)
                else train_preds
            )
            val_preds = (
                [val_preds]
                if isinstance(val_preds, np.ndarray)
                else val_preds
            )
            train_unc = [np.zeros_like(p) for p in train_preds]
            val_unc = [np.zeros_like(p) for p in val_preds]
        else:
            mc = int(self.params.get("mc_samples", 50))
            train_preds, train_unc = self.predict_with_uncertainty(
                x_train, mc
            )
            val_preds, val_unc = self.predict_with_uncertainty(x_val, mc)

        # Binary classification metrics (replaces MAE/R²).
        try:
            out_name = self.output_names[0]
            y_true_arr = y_train[out_name].flatten()
            y_prob = train_preds[0].flatten()
            y_hat = (y_prob >= 0.5).astype(int)
            acc = float(np.mean(y_hat == y_true_arr))
            pos_rate = float(np.mean(y_true_arr))
            pred_pos = float(np.mean(y_hat))
            print(
                f"Train Accuracy ({out_name}): {acc:.4f}  "
                f"pos_rate={pos_rate:.3f}  pred_pos={pred_pos:.3f}"
            )
        except Exception as e:
            print(f"Binary metric calculation error: {e}")

        return history, train_preds, train_unc, val_preds, val_unc

    # -- Persistence with metadata -------------------------------------------

    def save(self, file_path):
        super().save(file_path)
        metadata = {
            "model_type": "binary",
            "signal_type": self.params.get("signal_type", "buy_entry"),
            "window_size": self.params.get("window_size"),
            "output_names": self.output_names,
            "feature_columns": FEATURE_COLUMNS,
        }
        meta_path = str(file_path).rsplit(".", 1)[0] + "_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {meta_path}")

    def load(self, file_path):
        super().load(file_path)
        meta_path = str(file_path).rsplit(".", 1)[0] + "_metadata.json"
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            self.output_names = metadata.get(
                "output_names", self.output_names
            )
            if "signal_type" in metadata:
                self.params["signal_type"] = metadata["signal_type"]
            if "window_size" in metadata:
                self.params["window_size"] = metadata["window_size"]
            print(f"Metadata loaded from {meta_path}")
        except FileNotFoundError:
            print(f"No metadata file found at {meta_path}")
