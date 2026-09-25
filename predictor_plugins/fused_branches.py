#!/usr/bin/env python
"""Multi-branch core: one encoder per feature group, fused.

WP24 of the M5PHET work plan (2026-09-24, revision 2). The WP18 probe
(``tests/test_wp18_branch_capability.py``) found that no ``predictor.plugins``
entry point accepted several input branches, so the pipeline WP18 describes --
one extractor per feature GROUP, the branches fused, one core over the fusion --
had no core to name. This plugin is that core, and nothing more: it builds the
graph, it says nothing about whether the graph is any good.

Declared configuration (everything else is the common training configuration the
other plugins take, inherited from ``common.base``):

``branches``
    ``[{"name": str, "columns": [str|int], "encoder": "cnn"|"lstm"|"rnn"|"dense"|"tcn"}]``.
    One branch per feature group. ``columns`` are column NAMES resolved against
    ``feature_names`` (the preprocessor's own key for the channel order), or raw
    channel indices. Omitted, the branches are derived from the input shapes.
``fusion``
    ``"concat"`` (declared, the default) or ``"sum"`` (optional; every encoder
    emits ``encoder_units`` features, so the sum is defined).
``head``
    ``{"units": [..], "activation": ...}`` -- the dense trunk over the fusion,
    before the per-horizon heads. The heads are the family's Bayesian heads
    (``bayesian_head: true``, the default) or plain linear ones.

``build_model`` accepts EITHER of the two input declarations, and they describe
the same model:

* the framework's single ``(window, channels)`` pair -- the pipeline is
  unchanged, and each branch's channels are gathered out of that one tensor
  inside the graph (a ``Lambda`` over ``tf.gather``);
* a list, tuple or mapping of per-branch ``(window, channels_g)`` shapes -- one
  Keras ``Input`` per branch, which is what the WP18 probe hands a core.

Refusals are by name, and nothing is guessed:

``BRANCH_COLUMN_UNKNOWN``      a branch names a column that the input does not have.
``UNKNOWN_ENCODER``            a branch names an encoder this plugin does not implement.
``UNKNOWN_FUSION``             a fusion this plugin does not implement.
``FEATURE_NAMES_NOT_DECLARED`` columns given by name with no ``feature_names`` to resolve them.
``BRANCH_SHAPE_MISMATCH``      a per-branch shape that contradicts the branch's column count.
``NO_BRANCHES``                an empty branch list.

The encoders are implemented here with Keras layers only; nothing is imported
from feature-extractor. They are not that repository's encoders and are not
claimed to be.
"""
from __future__ import annotations

from typing import Optional, Any, Dict, List, Sequence

import keras
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import (
    Add, Concatenate, Conv1D, Dense, Dropout, Flatten, GlobalAveragePooling1D,
    Input, GRU, LSTM, Lambda,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW

from .common.base import BaseBayesianKerasPredictor
from .common.bayesian import posterior_mean_field, prior_fn
from .common.losses import mae_magnitude, random_normal_initializer_44

#: the encoders this plugin implements, with the one-line label of each
ENCODERS: Dict[str, str] = {
    "cnn": "causal Conv1D stack, average-pooled over time",
    "lstm": "stacked LSTM, last state",
    "rnn": "stacked GRU, last state (the feature-extractor 'rnn' family)",
    "dense": "flattened window through a dense stack",
    "tcn": "dilated causal residual Conv1D blocks, average-pooled over time",
}
#: feature-extractor's registered encoder keys, declared here as the inline family each one IS -- a person's
#: declaration reviewed in this repository, never a runtime guess. WP18 (2026-09-25) chose `default` and `ann` for
#: two groups and the pipeline could not map them because the keys differ between repositories although the
#: architectures are the same family. `None` means "no inline family implements it": that branch stays NOT_MAPPED.
EXTRACTOR_FAMILIES: Dict[str, Optional[str]] = {
    "default": "dense",       # per-channel Dense branches over the window -> dense
    "ann": "dense",           # per-channel Dense branches over the window -> dense
    "cnn": "cnn",             # two strided Conv1D layers -> cnn
    "vae": "cnn",             # two strided Conv1D layers, no sampling step (WP25 finding) -> cnn
    "lstm": "lstm",           # attention + two BiLSTM; the recurrent family -> lstm (attention not reproduced)
    "rnn": "rnn",             # two recurrent layers (SimpleRNN/GRU) -> rnn
    "transformer": None,      # attention + strided Conv1D: no inline attention encoder here
    "vae_small": None,        # per-step CVAE inference network: not a window encoder
}
#: the fusions this plugin implements
FUSIONS: Dict[str, str] = {
    "concat": "the per-branch vectors concatenated",
    "sum": "the per-branch vectors added (every encoder emits encoder_units features)",
}

BRANCH_COLUMN_UNKNOWN = "BRANCH_COLUMN_UNKNOWN"
UNKNOWN_ENCODER = "UNKNOWN_ENCODER"
UNKNOWN_FUSION = "UNKNOWN_FUSION"
FEATURE_NAMES_NOT_DECLARED = "FEATURE_NAMES_NOT_DECLARED"
BRANCH_SHAPE_MISMATCH = "BRANCH_SHAPE_MISMATCH"
NO_BRANCHES = "NO_BRANCHES"


def _is_shape_pair(value: Any) -> bool:
    """True for a single ``(window, channels)`` declaration, false for a collection of them."""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(part, int) for part in value))


@keras.saving.register_keras_serializable(package="predictor_plugins.fused_branches")
class GatherColumns(keras.layers.Layer):
    """The channel-axis slice one branch reads, carrying its own indices into the saved graph.

    This was a ``Lambda`` over a closure until 2026-09-25, and a closure does not survive being written to disk: Keras
    serialises such a layer as the bare NAME of the inner function, so every saved graph whose branch read a SUBSET of
    the columns could be fitted, scored and then never loaded again -- ``Could not locate function '_slice'`` -- by
    anything but the process that built it. That made those models unservable and unreviewable, which is the same
    defect twice. A registered layer that writes its ``indices`` into its own config is loadable by anybody, and the
    indices are read from the file instead of being re-derived by the reader from the column names, which would look
    exactly like the original and could gather other columns.

    Graphs saved before this change still carry the old ``Lambda``; nothing here can repair them, and a fit is cheap.
    """

    def __init__(self, indices: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        self.indices = tuple(int(index) for index in indices)

    def call(self, tensor):
        return tf.gather(tensor, tf.constant(self.indices, dtype=tf.int32), axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (len(self.indices),)

    def get_config(self):
        config = super().get_config()
        config["indices"] = list(self.indices)
        return config


class Plugin(BaseBayesianKerasPredictor):
    """Multi-branch core: one encoder per feature group, fused."""

    plugin_params = {
        "batch_size": 32,
        "branches": None,
        "fusion": "concat",
        "head": None,
        "feature_names": None,
        "default_encoder": "cnn",
        "encoder_units": 32,
        "encoder_layers": 1,
        "encoder_kernel_size": 3,
        "encoder_dropout": 0.0,
        "activation": "relu",
        "head_units": [64],
        "head_activation": "relu",
        "dropout_rate": 0.0,
        "bayesian_head": True,
        "learning_rate": 0.001,
        "predicted_horizons": [1],
        "early_patience": 10,
        "kl_weight": 1e-3,
        "kl_anneal_epochs": 10,
        "mc_samples": 50,
    }
    plugin_debug_vars = [
        "batch_size", "branches", "fusion", "head", "default_encoder", "encoder_units",
        "encoder_layers", "encoder_kernel_size", "activation", "head_units", "head_activation",
        "dropout_rate", "bayesian_head", "learning_rate", "predicted_horizons", "early_patience",
        "kl_weight", "mc_samples",
    ]

    # --- the declared configuration, read and refused by name -------------------------------------

    def _declared_branch_shapes(self, input_shape):
        """The per-branch ``(window, channels)`` pairs, or ``None`` for the single-tensor path."""
        if _is_shape_pair(input_shape):
            return None
        if isinstance(input_shape, dict):
            return [(str(name), tuple(shape)) for name, shape in input_shape.items()]
        if isinstance(input_shape, (list, tuple)):
            return [(None, tuple(shape)) for shape in input_shape]
        raise ValueError(f"BRANCH_SHAPE_MISMATCH: input_shape {input_shape!r} is neither a "
                         f"(window, channels) pair nor a collection of them")

    def _resolve_columns(self, columns, channels, branch_name):
        """Channel indices for one branch, refusing by name anything the input does not have."""
        if columns is None:
            return list(range(channels))
        names = self.params.get("feature_names")
        indices: List[int] = []
        for column in columns:
            if isinstance(column, bool):
                raise ValueError(f"{BRANCH_COLUMN_UNKNOWN}: branch {branch_name!r} names {column!r}, "
                                 f"which is not a column name nor a channel index")
            if isinstance(column, int):
                index = column
            else:
                if not names:
                    raise ValueError(f"{FEATURE_NAMES_NOT_DECLARED}: branch {branch_name!r} names the "
                                     f"column {column!r}, and no feature_names were declared to resolve it")
                if column not in names:
                    raise ValueError(f"{BRANCH_COLUMN_UNKNOWN}: branch {branch_name!r} names the column "
                                     f"{column!r}, which is not among the declared feature_names "
                                     f"{list(names)!r}")
                index = list(names).index(column)
            if index < 0 or index >= channels:
                raise ValueError(f"{BRANCH_COLUMN_UNKNOWN}: branch {branch_name!r} names the column "
                                 f"{column!r} at channel index {index}, and the input has {channels} "
                                 f"channel(s)")
            indices.append(index)
        if not indices:
            raise ValueError(f"{BRANCH_COLUMN_UNKNOWN}: branch {branch_name!r} declares no column")
        return indices

    def _branch_plan(self, input_shape):
        """One record per branch: name, encoder, channel indices, and its own input shape if it has one."""
        declared = self.params.get("branches")
        shapes = self._declared_branch_shapes(input_shape)

        if declared is None:
            if shapes is None:
                window, channels = input_shape
                declared = [{"name": "all", "columns": None, "encoder": self.params["default_encoder"]}]
            else:
                declared = [{"name": name or f"branch_{position}", "columns": None,
                             "encoder": self.params["default_encoder"]}
                            for position, (name, _shape) in enumerate(shapes)]
        if not declared:
            raise ValueError(f"{NO_BRANCHES}: the branches list is empty; a fusing core needs at least one branch")

        if shapes is not None and len(shapes) != len(declared):
            raise ValueError(f"{BRANCH_SHAPE_MISMATCH}: {len(shapes)} input shape(s) were given for "
                             f"{len(declared)} declared branch(es)")

        total_channels = None if shapes is not None else int(input_shape[1])
        window = int(shapes[0][1][0]) if shapes is not None else int(input_shape[0])

        plan = []
        for position, branch in enumerate(declared):
            name = str(branch.get("name") or f"branch_{position}")
            encoder = str(branch.get("encoder") or self.params["default_encoder"])
            if encoder not in ENCODERS:
                raise ValueError(f"{UNKNOWN_ENCODER}: branch {name!r} names the encoder {encoder!r}; "
                                 f"this plugin implements {sorted(ENCODERS)}")
            columns = branch.get("columns")
            if shapes is None:
                indices = self._resolve_columns(columns, total_channels, name)
                plan.append({"name": name, "encoder": encoder, "indices": indices, "shape": None})
            else:
                branch_window, branch_channels = int(shapes[position][1][0]), int(shapes[position][1][1])
                if branch_window != window:
                    raise ValueError(f"{BRANCH_SHAPE_MISMATCH}: branch {name!r} declares a window of "
                                     f"{branch_window} while branch 0 declares {window}")
                # the columns are still checked against the declared names, so the same branch list
                # is refused the same way on both input paths
                if columns is not None:
                    names = self.params.get("feature_names")
                    if names:
                        self._resolve_columns(columns, len(names), name)
                    if len(columns) != branch_channels:
                        raise ValueError(f"{BRANCH_SHAPE_MISMATCH}: branch {name!r} declares "
                                         f"{len(columns)} column(s) and an input shape with "
                                         f"{branch_channels} channel(s)")
                plan.append({"name": name, "encoder": encoder, "indices": None,
                             "shape": (branch_window, branch_channels)})
        fusion = str(self.params.get("fusion", "concat") or "concat")
        if fusion not in FUSIONS:
            raise ValueError(f"{UNKNOWN_FUSION}: {fusion!r}; this plugin implements {sorted(FUSIONS)}")
        return plan, fusion, window

    # --- the encoders (Keras layers only; nothing imported from feature-extractor) ------------------

    def _encode(self, tensor, kind: str, name: str):
        units = int(self.params.get("encoder_units", 32))
        depth = max(1, int(self.params.get("encoder_layers", 1)))
        kernel = max(1, int(self.params.get("encoder_kernel_size", 3)))
        activation = self.params.get("activation", "relu")
        dropout = float(self.params.get("encoder_dropout", 0.0) or 0.0)

        x = tensor
        if kind == "dense":
            x = Flatten(name=f"{name}_flatten")(x)
            for layer in range(depth):
                x = Dense(units, activation=activation, name=f"{name}_dense_{layer}")(x)
        elif kind == "cnn":
            for layer in range(depth):
                x = Conv1D(units, kernel, padding="causal", activation=activation,
                           name=f"{name}_conv_{layer}")(x)
            x = GlobalAveragePooling1D(name=f"{name}_pool")(x)
        elif kind == "lstm":
            for layer in range(depth):
                last = layer == depth - 1
                x = LSTM(units, return_sequences=not last, name=f"{name}_lstm_{layer}")(x)
        elif kind == "rnn":
            for layer in range(depth):
                last = layer == depth - 1
                x = GRU(units, return_sequences=not last, name=f"{name}_gru_{layer}")(x)
        elif kind == "tcn":
            x = Conv1D(units, 1, padding="same", name=f"{name}_tcn_project")(x)
            for layer in range(depth):
                residual = Conv1D(units, kernel, padding="causal", dilation_rate=2 ** layer,
                                  activation=activation, name=f"{name}_tcn_conv_{layer}")(x)
                x = Add(name=f"{name}_tcn_residual_{layer}")([x, residual])
            x = GlobalAveragePooling1D(name=f"{name}_tcn_pool")(x)
        else:  # pragma: no cover -- _branch_plan refuses an unknown encoder before this point
            raise ValueError(f"{UNKNOWN_ENCODER}: {kind!r}")
        if dropout > 0:
            x = Dropout(dropout, name=f"{name}_dropout")(x)
        return x

    # --- the model ----------------------------------------------------------------------------------

    def build_model(self, input_shape, x_train, config):
        if config:
            self.params.update(config)
        head = self.params.get("head") or {}
        if head:
            if "units" in head:
                self.params["head_units"] = head["units"]
            if "activation" in head:
                self.params["head_activation"] = head["activation"]

        plan, fusion, window = self._branch_plan(input_shape)

        if plan[0]["shape"] is None:
            channels = int(input_shape[1])
            shared = Input(shape=(window, channels), name="input_layer")
            inputs = shared
            branch_tensors = []
            for branch in plan:
                indices = branch["indices"]
                if list(indices) == list(range(channels)):
                    sliced = shared
                else:
                    sliced = GatherColumns(indices, name=f"{branch['name']}_columns")(shared)
                branch_tensors.append(sliced)
        else:
            inputs = [Input(shape=branch["shape"], name=f"{branch['name']}_input") for branch in plan]
            branch_tensors = list(inputs)

        encoded = [self._encode(tensor, branch["encoder"], branch["name"])
                   for tensor, branch in zip(branch_tensors, plan)]
        if len(encoded) == 1:
            fused = encoded[0]
        elif fusion == "sum":
            fused = Add(name="fusion_sum")(encoded)
        else:
            fused = Concatenate(name="fusion_concat")(encoded)

        trunk = fused
        head_units = self.params.get("head_units") or [64]
        head_activation = self.params.get("head_activation", "relu")
        dropout = float(self.params.get("dropout_rate", 0.0) or 0.0)
        for layer, units in enumerate(head_units):
            trunk = Dense(int(units), activation=head_activation, name=f"head_dense_{layer}")(trunk)
            if dropout > 0:
                trunk = Dropout(dropout, name=f"head_dropout_{layer}")(trunk)

        horizons = list(self.params["predicted_horizons"])
        bayesian = bool(self.params.get("bayesian_head", True))
        last_units = int(head_units[-1]) if head_units else int(self.params.get("encoder_units", 32))
        DenseFlipout = tfp.layers.DenseFlipout
        kl_weight = self.kl_weight_var

        outputs = []
        self.output_names = []
        for horizon in horizons:
            suffix = f"_h{horizon}"
            branch_head = Dense(max(8, last_units // 2), activation=head_activation,
                                name=f"horizon_dense{suffix}")(trunk)
            if bayesian:
                flip_name = f"flipout{suffix}"
                flip_layer = DenseFlipout(
                    units=1,
                    activation="linear",
                    kernel_posterior_fn=lambda dt, sh, bs, tr, nm=flip_name: posterior_mean_field(dt, sh, bs, tr, nm),
                    kernel_prior_fn=lambda dt, sh, bs, tr, nm=flip_name: prior_fn(dt, sh, bs, tr, nm),
                    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) * kl_weight,
                    name=flip_name,
                )
                bayes = Lambda(lambda t, fl=flip_layer: fl(t), name=f"bayes_out{suffix}")(branch_head)
                bias = Dense(1, activation="linear", kernel_initializer=random_normal_initializer_44,
                             name=f"bias{suffix}")(branch_head)
                out = Add(name=f"output_horizon_{horizon}")([bayes, bias])
            else:
                out = Dense(1, activation="linear", name=f"output_horizon_{horizon}")(branch_head)
            outputs.append(out)
            self.output_names.append(f"output_horizon_{horizon}")

        self.model = Model(inputs=inputs, outputs=outputs,
                           name=f"FusedBranches_{len(plan)}B_{len(horizons)}H")
        optimizer = AdamW(learning_rate=self.params.get("learning_rate", 1e-3))
        huber = Huber()
        loss_dict = {name: huber for name in self.output_names}
        metrics_dict = {name: [mae_magnitude] for name in self.output_names}
        self.model.compile(optimizer=optimizer, loss=loss_dict, metrics=metrics_dict)
        if not self.params.get("quiet", False):
            self.model.summary(line_length=140)
        return self.model


if __name__ == "__main__":  # pragma: no cover
    plug = Plugin({"predicted_horizons": [1, 3], "plotted_horizon": 1, "quiet": True,
                   "feature_names": ["open", "high", "low", "close"],
                   "branches": [{"name": "prices", "columns": ["open", "close"], "encoder": "cnn"},
                                {"name": "range", "columns": ["high", "low"], "encoder": "lstm"}]})
    plug.build_model((24, 4), None, {})
    print("inputs:", len(plug.model.inputs), "outputs:", plug.output_names)
