#!/usr/bin/env python
"""Quantile head: one dense trunk over the window, a monotone set of declared quantiles per horizon.

WP07 of the M5PHET work plan (2026-09-24). Every forecast bundle this stack has exported so far is a point model: one
number per target and horizon, no predictive distribution, so the forecast provider refuses `interval` by name for all
of them. This plugin is the first graph here whose output *is* a set of quantiles, fitted with the pinball loss, so an
interval can be read off a fitted pair instead of manufactured from a residual nobody recorded.

Two properties are built into the graph rather than hoped for:

* **the quantiles cannot cross.** The head emits the lowest quantile with a plain linear unit and every following one as
  the previous one plus a strictly non-negative increment (`softplus`). A crossing (a 5th percentile above the 95th) is
  then not merely unlikely, it is unreachable, so an interval read off this graph is always an interval;
* **nothing custom is needed to LOAD it.** The monotone head is built from `Dense`, `Add` and `Concatenate` only, so a
  saved graph reloads with `compile=False` in a process that has never imported this module -- which is exactly the
  situation of the export in `prediction_provider`. The pinball loss is a custom object, and it is needed only to
  continue training, never to serve.

Declared configuration (everything else is the common training configuration inherited from ``common.base``):

``quantiles``
    the quantiles to fit, ascending, each strictly inside (0, 1), at least two. `[0.05, 0.5, 0.95]` is the default and
    the set WP07 declares: the median is the point forecast, and the pair (0.05, 0.95) is the one fitted interval,
    whose nominal level is 0.90 -- **not** 0.95. A level with no fitted pair is refused by the provider by name
    (`CONFIDENCE_LEVEL_NOT_FITTED`); this plugin does not invent the pair by scaling another one.
``trunk_units`` / ``trunk_activation`` / ``dropout_rate``
    the dense trunk over the flattened window.

Refusals, by name, nothing guessed:

``QUANTILES_NOT_ASCENDING``  a quantile list that is not strictly ascending, or holds a duplicate.
``QUANTILE_OUT_OF_RANGE``    a quantile outside (0, 1).
``TOO_FEW_QUANTILES``        fewer than two quantiles: one quantile is a point forecast, and this is not that plugin.
``NOT_MULTI_BRANCH``         a per-branch input declaration: this core takes ONE window, and says so instead of
                             silently using the first branch.
"""
from __future__ import annotations

from typing import Any, Dict, List

import tensorflow as tf
from tensorflow.keras.layers import Add, Concatenate, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW

from .common.base import BaseDeterministicKerasPredictor
from .common.losses import mae_magnitude

QUANTILES_NOT_ASCENDING = "QUANTILES_NOT_ASCENDING"
QUANTILE_OUT_OF_RANGE = "QUANTILE_OUT_OF_RANGE"
TOO_FEW_QUANTILES = "TOO_FEW_QUANTILES"
NOT_MULTI_BRANCH = "NOT_MULTI_BRANCH"


def declared_quantiles(values) -> List[float]:
    """The quantile list, checked. A quantile set that is not a quantile set is refused, never sorted into shape."""
    quantiles = [float(q) for q in (values or ())]
    if len(quantiles) < 2:
        raise ValueError(f"{TOO_FEW_QUANTILES}: {quantiles!r}; one quantile is a point forecast, not a distribution")
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"{QUANTILE_OUT_OF_RANGE}: {q!r} is not strictly inside (0, 1)")
    if any(b <= a for a, b in zip(quantiles, quantiles[1:])):
        raise ValueError(f"{QUANTILES_NOT_ASCENDING}: {quantiles!r} must be strictly ascending; a quantile set given "
                         f"out of order would be silently reordered and the head would fit a different question")
    return quantiles


def pinball_loss(quantiles):
    """The check (pinball) loss, averaged over the declared quantiles.

    ``y_true`` is one realised value per row and ``y_pred`` one value per quantile; the subtraction broadcasts, which is
    what makes the loss per quantile and the mean over them the loss of the head.
    """
    levels = tf.constant([float(q) for q in quantiles], dtype=tf.float32)

    def loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        error = tf.cast(tf.reshape(y_true, (-1, 1)), tf.float32) - y_pred
        return tf.reduce_mean(tf.maximum(levels * error, (levels - 1.0) * error), axis=-1)

    loss.__name__ = "pinball_loss"
    return loss


class Plugin(BaseDeterministicKerasPredictor):
    """Dense trunk over one window, a monotone quantile head per horizon, fitted with the pinball loss."""

    plugin_params = {
        "batch_size": 32,
        "quantiles": [0.05, 0.5, 0.95],
        "trunk_units": [64, 32],
        "trunk_activation": "relu",
        "dropout_rate": 0.0,
        "learning_rate": 0.001,
        "predicted_horizons": [1],
        "early_patience": 10,
        "mc_samples": 1,
    }
    plugin_debug_vars = [
        "batch_size", "quantiles", "trunk_units", "trunk_activation", "dropout_rate",
        "learning_rate", "predicted_horizons", "early_patience",
    ]

    def get_custom_objects(self):
        """What a graph of this family needs to be LOADED WITH ITS LOSS. Serving needs none of it: the head is built
        from stock layers, so `compile=False` is enough to run it."""
        objects = dict(super().get_custom_objects())
        objects["pinball_loss"] = pinball_loss(self.params.get("quantiles") or [0.05, 0.5, 0.95])
        objects["mae_magnitude"] = mae_magnitude
        return objects

    def _quantile_head(self, trunk, horizon: int, quantiles: List[float]):
        """The lowest quantile, then each following one as the previous plus a non-negative increment.

        Built from `Dense`, `Add` and `Concatenate` deliberately: a `Lambda` or a custom layer here would make every
        reader of the saved graph -- the exporter above all -- depend on importing this module.
        """
        suffix = f"_h{horizon}"
        outputs = [Dense(1, activation="linear", name=f"q{0}{suffix}")(trunk)]
        for position in range(1, len(quantiles)):
            step = Dense(1, activation="softplus", name=f"q{position}_step{suffix}")(trunk)
            outputs.append(Add(name=f"q{position}{suffix}")([outputs[-1], step]))
        return Concatenate(name=f"output_horizon_{horizon}")(outputs)

    def build_model(self, input_shape, x_train, config: Dict[str, Any]):
        if config:
            self.params.update(config)
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2
                and all(isinstance(part, int) for part in input_shape)):
            raise ValueError(f"{NOT_MULTI_BRANCH}: this core takes one (window, channels) input, not {input_shape!r}; "
                             f"a grouped input needs a fusing core such as 'fused_branches'")
        quantiles = declared_quantiles(self.params.get("quantiles"))
        self.params["quantiles"] = quantiles

        window, channels = int(input_shape[0]), int(input_shape[1])
        inputs = Input(shape=(window, channels), name="input_layer")
        trunk = Flatten(name="flatten")(inputs)
        dropout = float(self.params.get("dropout_rate", 0.0) or 0.0)
        for layer, units in enumerate(self.params.get("trunk_units") or [64]):
            trunk = Dense(int(units), activation=self.params.get("trunk_activation", "relu"),
                          name=f"trunk_{layer}")(trunk)
            if dropout > 0:
                trunk = Dropout(dropout, name=f"trunk_dropout_{layer}")(trunk)

        horizons = list(self.params["predicted_horizons"])
        outputs, self.output_names = [], []
        for horizon in horizons:
            outputs.append(self._quantile_head(trunk, int(horizon), quantiles))
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs,
                           name=f"QuantileANN_{len(quantiles)}Q_{len(horizons)}H")
        loss = pinball_loss(quantiles)
        self.model.compile(optimizer=AdamW(learning_rate=self.params.get("learning_rate", 1e-3)),
                           loss={name: loss for name in self.output_names},
                           metrics={name: [mae_magnitude] for name in self.output_names})
        if not self.params.get("quiet", False):
            self.model.summary(line_length=140)
        return self.model


if __name__ == "__main__":  # pragma: no cover
    plug = Plugin({"predicted_horizons": [60], "plotted_horizon": 60, "quiet": True})
    plug.build_model((60, 7), None, {})
    print("outputs:", plug.output_names, "shape:", plug.model.output_shape)
