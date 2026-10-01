"""Opt-in hierarchical modular predictor behind the legacy ``predictor.plugins`` facade.

Selected only by ``predictor_plugin: "modular_temporal"``. No legacy plugin name is
routed here and no legacy plugin is modified. The model is assembled by
``predictor_plugins.modular_temporal.build_modular`` from independent branch,
fusion, core and head components resolved through the ``modular.*`` entry-point
groups; preprocessing stays in the preprocessor plugins.

Configuration (flat run config, as every other predictor plugin receives it):

``modular``            the nested, versioned modular config (schema
                       ``predictor.modular.v1``), or
``modular_config_file`` a path to a JSON file holding it (exactly one of the two).
flat overrides         any key in the optimizer namespace ``modular.*``,
                       ``branches.<name>.*``, ``core.*``, ``fusion.*``, ``head.*``
                       (see ``predictor_plugins.modular_config``).
``modular_training``   optional explicit fit settings for the shared early-stopping
                       loop (max_epochs, patience, min_delta, batch_size,
                       learning_rate, weight_decay, loss, huber_delta, seed,
                       max_updates, max_seconds). When absent they are derived from
                       the legacy keys epochs / early_patience / batch_size /
                       learning_rate; values outside the loop's bounds FAIL rather
                       than being clamped.

Legacy consistency is checked, never guessed: ``window_size`` must equal the
modular window, ``predicted_horizons`` must equal the modular horizons (they are
copied only when the modular config omits them), the input channel count must
equal ``len(feature_names)``, and a run config ``feature_names`` must list the
same features in the same order. All of this, every donor check and every
plugin/parameter check happens in ``build_model`` -- before any fit.

The Keras model exposed as ``self.model`` has one output per horizon named
``output_horizon_{h}`` with shape (batch, target_count), sharing weights with the
modular forecast model, so the existing pipelines, plots and result writers work
unchanged. After ``train`` the attribute ``training_receipt`` records observed
optimizer updates, stop reason and, per component, its regime, trainable
parameter count and whether its weights actually changed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from predictor_plugins import modular_config
from predictor_plugins import modular_temporal as mt

keras = mt.keras


@keras.utils.register_keras_serializable(package="predictor_plugin_modular")
class HorizonSlice(keras.layers.Layer):
    """Select one horizon of the (batch, H, T) forecast as a (batch, T) output."""

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[:, self.index, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return {**super().get_config(), "index": self.index}


_TRAINING_KEYS = {"max_epochs", "patience", "min_delta", "batch_size", "learning_rate",
                  "weight_decay", "loss", "huber_delta", "seed", "max_updates", "max_seconds"}


class Plugin:
    plugin_params = {
        "predicted_horizons": [1],
        "modular": None,
        "modular_config_file": None,
        "modular_training": None,
        # provenance written into the saved bundle (item 6): conditioning_contract / learned_corpus /
        # reconstruction. None -> every field UNKNOWN, explicitly. Bundles saved before this field
        # existed stay UNKNOWN until re-exported with a declared provenance; nothing is inferred.
        "modular_provenance": None,
        "predict_batch_size": 256,
    }
    plugin_debug_vars = ["predicted_horizons", "modular_config_sha256", "modular_regime"]

    def __init__(self, config=None):
        self.params = copy.deepcopy(self.plugin_params)
        if config:
            self.params.update(config)
        self.model = None
        self.bundle = None
        self.output_names = []
        self.modular_config = None
        self.applied_overrides = {}
        self.training_receipt = None

    # ------------------------------------------------------------------ params
    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def get_debug_info(self):
        return {"predicted_horizons": self.params.get("predicted_horizons"),
                "modular_config_sha256": mt.config_digest(self.modular_config)
                if self.modular_config else None,
                "modular_regime": mt.regime_summary(self.modular_config)["common"]
                if self.modular_config else None}

    def add_debug_info(self, debug_info):
        debug_info.update(self.get_debug_info())

    # ------------------------------------------------------------ config/build
    def resolve_modular_config(self, config=None):
        """Nested config + flat overrides + legacy consistency, normalized; fails closed."""
        run = dict(self.params)
        run.update(config or {})
        nested, path = run.get("modular"), run.get("modular_config_file")
        if (nested is None) == (path is None):
            raise ValueError("modular_temporal requires exactly one of 'modular' or 'modular_config_file'")
        if path is not None:
            nested = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(nested, dict):
            raise ValueError("The modular config must be a JSON object")
        nested = copy.deepcopy(nested)
        legacy_h = run.get("predicted_horizons")
        if "horizons" not in nested and legacy_h is not None:
            nested["horizons"] = list(legacy_h)
        resolved, applied = modular_config.apply_flat_overrides(nested, run)
        if legacy_h is not None and list(legacy_h) != resolved["horizons"]:
            raise ValueError("predicted_horizons disagrees with the modular horizons")
        window = run.get("window_size")
        if window is not None and int(window) != resolved["window"]:
            raise ValueError("window_size disagrees with the modular window")
        names = run.get("feature_names")
        if names is not None and list(names) != resolved["feature_names"]:
            raise ValueError("Run feature_names differ from the modular feature order")
        return resolved, applied

    def build_model(self, input_shape, x_train=None, config=None):
        if config:
            self.params.update(config)
        resolved, applied = self.resolve_modular_config()
        shape = tuple(input_shape) if isinstance(input_shape, (tuple, list)) else (int(input_shape),)
        if x_train is not None:
            shape = tuple(np.shape(x_train)[1:])
        if shape != (resolved["window"], len(resolved["feature_names"])):
            raise ValueError(f"Input shape {shape} does not match modular (window, features) "
                             f"({resolved['window']}, {len(resolved['feature_names'])})")
        self.bundle = mt.build_modular(resolved)
        self.modular_config, self.applied_overrides = self.bundle.config, applied
        self.params["predicted_horizons"] = list(resolved["horizons"])
        self._wrap()
        self.training_receipt = None
        return self.model

    def _wrap(self):
        forecast = self.bundle.forecast_model
        horizons = self.bundle.config["horizons"]
        outputs = [HorizonSlice(i, name=f"output_horizon_{h}")(forecast.output)
                   for i, h in enumerate(horizons)]
        self.output_names = [f"output_horizon_{h}" for h in horizons]
        self.model = keras.Model(forecast.inputs, outputs, name="modular_temporal_predictor")

    # --------------------------------------------------------------- training
    def training_settings(self, epochs=None, batch_size=None):
        explicit = self.params.get("modular_training")
        if explicit is not None:
            if not isinstance(explicit, dict) or set(explicit) - _TRAINING_KEYS:
                raise ValueError(f"modular_training keys must be within {sorted(_TRAINING_KEYS)}")
            return dict(explicit)
        settings = {"max_epochs": int(epochs if epochs is not None else self.params.get("epochs", 20)),
                    "patience": int(self.params.get("early_patience", 5)),
                    "batch_size": int(batch_size if batch_size is not None
                                      else self.params.get("batch_size", 32)),
                    "learning_rate": float(self.params.get("learning_rate", 1e-3))}
        if "min_delta" in self.params:
            settings["min_delta"] = float(self.params["min_delta"])
        return settings

    def _targets(self, y):
        horizons, count = self.bundle.config["horizons"], self.bundle.config["target_count"]
        if isinstance(y, dict):
            missing = [n for n in self.output_names if n not in y]
            if missing:
                raise ValueError(f"Targets missing outputs {missing}")
            parts = [np.asarray(y[n], dtype="float32") for n in self.output_names]
        elif isinstance(y, (list, tuple)):
            parts = [np.asarray(v, dtype="float32") for v in y]
        else:
            array = np.asarray(y, dtype="float32")
            if array.ndim == 2 and count == 1:
                array = array[:, :, None]
            if array.shape[1:] != (len(horizons), count):
                raise ValueError("Array targets must be (N, horizons, target_count)")
            return array
        if len(parts) != len(horizons):
            raise ValueError("One target array per horizon is required")
        parts = [p.reshape(len(p), -1) for p in parts]
        if any(p.shape[1] != count for p in parts):
            raise ValueError("Each horizon target must have target_count columns")
        return np.stack(parts, axis=1)

    def _components(self):
        items = [("branch:" + n, m, s["regime"]) for (n, m), s in
                 zip(self.bundle.branch_models.items(), self.bundle.config["branches"])]
        items.append(("core", self.bundle.core_model, self.bundle.config["core"]["regime"]))
        items.append(("head", self.bundle.forecast_model.get_layer("forecast_head"), "R0"))
        return items

    def train(self, x_train, y_train, epochs=None, batch_size=None, threshold_error=None,
              x_val=None, y_val=None, config=None):
        if self.bundle is None:
            raise ValueError("build_model must run before train")
        if config:
            self.params.update(config)
        if x_val is None or y_val is None:
            raise ValueError("A validation population is required for early stopping")
        from tools.modular_candidate_evaluator import fit_with_early_stopping
        settings = self.training_settings(epochs, batch_size)
        x, vx = np.asarray(x_train, dtype="float32"), np.asarray(x_val, dtype="float32")
        y, vy = self._targets(y_train), self._targets(y_val)
        before = {name: mt.weights_hash(m) for name, m, _ in self._components()}
        if any(spec["regime"] == "R3" for spec in [*self.bundle.config["branches"], self.bundle.config["core"]]):
            from predictor_plugins.modular_temporal.warm import fit_warm
            result = fit_warm(self.bundle, x, y, vx, vy, settings)          # R3: frozen, then unfrozen
            result["history"] = [*(result["history"]["phase_1"] or []), *(result["history"]["phase_2"] or [])]
        else:
            result = fit_with_early_stopping(self.bundle.forecast_model, x, y, vx, vy, settings)
        components = {}
        for name, model, regime in self._components():
            after = mt.weights_hash(model)
            components[name] = {
                "regime": regime,
                "trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                "weights_sha256_before": before[name], "weights_sha256_after": after,
                "weights_changed": after != before[name]}
        self.training_receipt = {
            "schema": "predictor.modular.training.v1",
            "config_sha256": mt.config_digest(self.modular_config),
            "regimes": mt.regime_summary(self.modular_config),
            "settings": settings, "applied_overrides": self.applied_overrides,
            "components": components,
            **{k: v for k, v in result.items() if k != "history"}}
        history = SimpleNamespace(history={
            "loss": [float(e["train_loss"]) for e in result["history"]],
            "val_loss": [float(e["validation_loss"]) for e in result["history"]]})
        train_preds, train_unc = self.predict_with_uncertainty(x)
        val_preds, val_unc = self.predict_with_uncertainty(vx)
        return history, train_preds, train_unc, val_preds, val_unc

    # ------------------------------------------------------------- prediction
    def predict_with_uncertainty(self, x_test, mc_samples=1):
        """Deterministic point forecasts, one (N, target_count) array per horizon; zero spread."""
        batch = int(self.params.get("predict_batch_size") or 256)
        preds = self.model.predict(np.asarray(x_test, dtype="float32"), batch_size=batch, verbose=0)
        preds = [preds] if isinstance(preds, np.ndarray) else list(preds)
        return preds, [np.zeros_like(p) for p in preds]

    # ------------------------------------------------------------ persistence
    def save(self, file_path):
        """``file_path`` (.keras) holds the facade graph; ``<file_path>.bundle/`` the modular bundle."""
        path = Path(file_path)
        self.model.save(path)
        document = mt.save_bundle(self.bundle, path.with_name(path.name + ".bundle"),
                                  provenance=self.params.get("modular_provenance"))
        if self.training_receipt is not None:
            (path.with_name(path.name + ".bundle") / "training_receipt.json").write_text(
                json.dumps(self.training_receipt, sort_keys=True, indent=2, default=str) + "\n",
                encoding="utf-8")
        return document

    def load(self, file_path):
        path = Path(file_path)
        bundle_dir = path.with_name(path.name + ".bundle")
        self.bundle, document = mt.load_bundle(bundle_dir)
        self.modular_config = self.bundle.config
        self.params["predicted_horizons"] = list(self.modular_config["horizons"])
        self._wrap()
        facade = keras.models.load_model(path, compile=False, safe_mode=True)
        probe = np.random.default_rng(1).normal(
            size=(2, *self.model.input_shape[1:])).astype("float32")
        mine, theirs = self.model(probe), facade(probe)
        mine = mine if isinstance(mine, (list, tuple)) else [mine]
        theirs = theirs if isinstance(theirs, (list, tuple)) else [theirs]
        if any(not np.allclose(a, b, atol=1e-6) for a, b in zip(mine, theirs)):
            raise ValueError("Facade archive and modular bundle disagree")
        return document
