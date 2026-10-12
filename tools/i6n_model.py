"""Bounded I6N candidate construction; no fitting, data access or device fallback.

Pass an explicit complete config, usually a copy of DEFAULT_CONFIG. Entrance
is ``conv`` (zero Transformer blocks), ``transformer`` or ``conv_transformer``
(one/two blocks). Widths are integers in 1..128; Transformer width must be even.
Heads are fixed at two and FFN expansion at 2*width, not configurable genes.
One to three reducer stages have independent filter widths and factors 1/2.
``adapter_width`` must be None for ``none``, otherwise a positive bounded width.
Only the last-tick forecast head collapses time. Input layout and physical-time
encoding are inherited from I6M: feature/source-mask pairs then market presence.

Importing/normalizing configs does not import TensorFlow. Loading a saved model
in a fresh process requires importing tools.i6e_masked_model and
tools.i6n_contracts.temporal_ops to register the existing serializable layers.
This is candidate engineering, not campaign-readiness evidence.
"""

from collections.abc import Mapping


DEFAULT_CONFIG = {
    "entrance": "conv",
    "width": 16,
    "transformer_blocks": 0,
    "reducer_filters": [8, 4],
    "reducer_strides": [1, 1],
    "adapter": "none",
    "adapter_width": None,
    "n_horizons": 1,
}
_CONFIG_KEYS = frozenset(DEFAULT_CONFIG)
_MAX_WIDTH = 128


def _integer(value, name, minimum=1, maximum=None):
    # Exact ints exclude booleans, floating-point aliases, NaN and infinity.
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum or 'unbounded'}")
    return value


def normalize_config(config):
    """Validate the explicit schema and return a detached JSON-compatible dict.

    Missing/unknown keys and inactive genes are errors, not silent defaults.
    No upper sample limit is imposed; callers own runtime resource admission.
    """
    if not isinstance(config, Mapping):
        raise ValueError("config must be an explicit mapping")
    if set(config) != _CONFIG_KEYS:
        missing = sorted(_CONFIG_KEYS - set(config))
        unknown = sorted(set(config) - _CONFIG_KEYS, key=str)
        raise ValueError(f"config schema mismatch: missing={missing}, unknown={unknown}")
    cfg = dict(config)
    entrance = cfg["entrance"]
    if not isinstance(entrance, str) or entrance not in (
        "conv", "transformer", "conv_transformer"
    ):
        raise ValueError("entrance must be conv, transformer or conv_transformer")
    width = _integer(cfg["width"], "width", maximum=_MAX_WIDTH)
    blocks = _integer(cfg["transformer_blocks"], "transformer_blocks", 0, 2)
    if (entrance == "conv") != (blocks == 0):
        raise ValueError("conv requires zero blocks; transformer entrances require 1 or 2")
    if blocks and width % 2:
        raise ValueError("Transformer width must be divisible by two heads")
    for key in ("reducer_filters", "reducer_strides"):
        if not isinstance(cfg[key], (list, tuple)) or not 1 <= len(cfg[key]) <= 3:
            raise ValueError(f"{key} must contain 1..3 stage values")
        cfg[key] = list(cfg[key])
    if len(cfg["reducer_filters"]) != len(cfg["reducer_strides"]):
        raise ValueError("each reducer must have an explicit independent stride")
    for index, filters in enumerate(cfg["reducer_filters"]):
        _integer(filters, f"reducer_filters[{index}]", maximum=_MAX_WIDTH)
    for index, stride in enumerate(cfg["reducer_strides"]):
        _integer(stride, f"reducer_strides[{index}]", maximum=2)
    adapter = cfg["adapter"]
    if not isinstance(adapter, str) or adapter not in ("none", "causalconv", "lstm"):
        raise ValueError("adapter must be none, causalconv or lstm")
    if adapter == "none":
        if cfg["adapter_width"] is not None:
            raise ValueError("adapter_width is inactive for adapter=none; use None")
    else:
        _integer(cfg["adapter_width"], "adapter_width", maximum=_MAX_WIDTH)
    _integer(cfg["n_horizons"], "n_horizons")
    return cfg


def build_model(samples, n_features, spacing_hours, config):
    """Initialize an uncompiled Keras model, without training or device selection.

    samples, n_features and spacing_hours are positive integers (bool rejected).
    Output is (batch, n_horizons). A missing final market tick zeroes head input,
    not the Dense head's trainable bias. All dimensions remain temporal until
    ``head_input``; ``core_output`` is available for synthetic diagnostic probes.
    """
    _integer(samples, "samples")
    _integer(n_features, "n_features")
    _integer(spacing_hours, "spacing_hours")
    cfg = normalize_config(config)

    import tensorflow as tf
    from predictor_plugins.modular_temporal.components import causal_conv1d
    from tools.i6e_masked_model import ClockView
    from tools.i6n_contracts.temporal_ops import (
        I6NAttentionMask, RightAnchoredSubsample, reduction_plan)

    # Reject, never repair, a cascade that would lose t or empty the sequence.
    reduction_plan(samples, cfg["reducer_strides"])
    K = tf.keras
    inp = K.Input((samples, 2 * n_features + 1), name="clock_input")
    mask = ClockView("mask", name="market_presence")(inp)
    clean = ClockView("clean", name="ignore_placeholders")(inp)
    grid = tuple((i - samples + 1) * spacing_hours for i in range(samples))
    parts = []
    for index in range(n_features):
        feature = ClockView("feature", index=index, name=f"feature_{index}")(clean)
        branch = causal_conv1d(
            input_shape=(samples, 2), time_grid=grid, output_steps=samples,
            name=f"branch_{index}", params={"channels": 16, "kernel_size": 3},
        )
        parts.append(ClockView("gate", name=f"branch_mask_{index}")(
            [branch.model(feature), mask]))
    fused = (K.layers.Concatenate(axis=-1, name="fusion")(parts)
             if len(parts) > 1 else K.layers.Activation("linear", name="fusion")(parts[0]))
    x = ClockView("position", spacing_hours=float(spacing_hours),
                  name="physical_position")(fused)
    # Position codes at absent slots must not leak into a later causal kernel.
    x = ClockView("gate", name="position_mask")([x, mask])
    width = cfg["width"]
    x = K.layers.Conv1D(
        width, 1 if cfg["entrance"] == "transformer" else 3,
        padding="causal", activation=None if cfg["entrance"] == "transformer" else "gelu",
        name="core_projection",
    )(x)
    x = ClockView("gate", name="core_entry_mask")([x, mask])
    if cfg["transformer_blocks"]:
        attention_mask = I6NAttentionMask(name="causal_market_attention")(mask)
    for block in range(cfg["transformer_blocks"]):
        attention = K.layers.MultiHeadAttention(
            num_heads=2, key_dim=width // 2, dropout=0., name=f"attention_{block}",
        )(x, x, attention_mask=attention_mask)
        x = K.layers.LayerNormalization(name=f"attention_norm_{block}")(
            K.layers.Add(name=f"attention_add_{block}")([x, attention]))
        x = ClockView("gate", name=f"attention_mask_{block}")([x, mask])
        ff = K.layers.Dense(2 * width, activation="gelu", name=f"ff_expand_{block}")(x)
        ff = K.layers.Dense(width, name=f"ff_project_{block}")(ff)
        x = K.layers.LayerNormalization(name=f"ff_norm_{block}")(
            K.layers.Add(name=f"ff_add_{block}")([x, ff]))
        x = ClockView("gate", name=f"transformer_mask_{block}")([x, mask])
    for stage, (filters, factor) in enumerate(zip(
        cfg["reducer_filters"], cfg["reducer_strides"]
    )):
        x = K.layers.Conv1D(filters, 3, strides=1, padding="causal",
                            activation="gelu", name=f"reducer_conv_{stage}")(x)
        x = ClockView("gate", name=f"reducer_gate_{stage}")([x, mask])
        x = RightAnchoredSubsample(factor, name=f"reducer_subsample_{stage}")(x)
        mask = RightAnchoredSubsample(factor, dtype="bool",
                                      name=f"reducer_mask_{stage}")(mask)
        x = ClockView("gate", name=f"reducer_output_{stage}")([x, mask])
    if cfg["adapter"] == "causalconv":
        x = K.layers.Conv1D(cfg["adapter_width"], 3, padding="causal",
                            activation="gelu", name="final_adapter")(x)
    elif cfg["adapter"] == "lstm":
        # Interior gaps violate cuDNN's right-padding requirement. Choose the
        # portable kernel explicitly on CPU and GPU, never an automatic retry.
        x = K.layers.LSTM(cfg["adapter_width"], return_sequences=True,
                          use_cudnn=False, name="final_adapter")(
            x, mask=K.ops.squeeze(mask, axis=-1))
    x = ClockView("gate", name="core_output")([x, mask])
    last = ClockView("last", name="head_input")(x)
    output = K.layers.Dense(cfg["n_horizons"], name="forecast_head")(last)
    return K.Model(inp, output, name="I6N_masked_candidate")


def verify_placement(model, placement, probe):
    """Strict device contract for a built model; no fallback, no repair.

    Requires soft device placement disabled, every variable (attention
    included) on ``placement`` and a probe forward pass whose output is on
    ``placement``. Variables are checked before any op runs, so a GPU request
    on a host without that GPU fails as PLACEMENT_MISMATCH rather than being
    executed elsewhere. Returns a JSON-safe report for the candidate receipt.
    """
    import tensorflow as tf

    if not isinstance(placement, str) or placement not in ("/CPU:0", "/GPU:0"):
        raise ValueError("placement must be /CPU:0 or /GPU:0")
    if tf.config.get_soft_device_placement():
        raise ValueError("SOFT_PLACEMENT_FORBIDDEN: disable soft device placement first")
    suffix = "/device:" + placement[1:]

    def on_device(variables):
        return all(v.value.handle.device.endswith(suffix) for v in variables)

    attention = [v for layer in model.layers
                 if isinstance(layer, tf.keras.layers.MultiHeadAttention)
                 for v in layer.variables]
    if not on_device(model.variables) or not on_device(attention):
        raise ValueError(f"PLACEMENT_MISMATCH: model variables are not all on {placement}")
    with tf.device(placement):
        output = model(probe, training=False)
    if not output.device.endswith(suffix):
        raise ValueError(f"PLACEMENT_MISMATCH: output on {output.device}, not {placement}")
    return dict(soft_device_placement=False, variables_on_device=True,
                attention_variables_on_device=True if attention else None,
                output_device=output.device)
