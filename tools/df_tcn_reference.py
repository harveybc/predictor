#!/usr/bin/env python3
"""RP61: an INDEPENDENT TCN comparator, ported from the reference implementation, adaptations declared.

Source of the specification, read at its origin (raw.githubusercontent.com/locuslab/TCN, `TCN/tcn.py`,
the implementation accompanying Bai, Kolter & Koltun 2018, arXiv:1803.01271):

    TemporalBlock(n_in, n_out, k, stride, dilation, padding, dropout=0.2):
        weight_norm(Conv1d(n_in,  n_out, k, dilation=d, padding=(k-1)*d)) -> Chomp1d -> ReLU -> Dropout
        weight_norm(Conv1d(n_out, n_out, k, dilation=d, padding=(k-1)*d)) -> Chomp1d -> ReLU -> Dropout
        residual: x, or Conv1d(n_in, n_out, 1) when the channel count changes
        output: ReLU(net(x) + residual)
        init: every conv weight ~ N(0, 0.01)
    TemporalConvNet(num_inputs, num_channels, kernel_size=2, dropout=0.2):
        one TemporalBlock per level, dilation 2**i

Our own core is NOT this block, and this module exists so the difference can be measured instead of
assumed away. Ours is one Conv1D with ELU and a residual sum: no weight normalisation, no dropout,
no second convolution, and no activation after the sum.

WHAT THIS PORT CHANGES, AND WHY — every item is a departure from the source, declared:

  1. framework. The reference is PyTorch; this is Keras, to run inside the existing route. Causal
     padding replaces `padding=(k-1)*d` followed by `Chomp1d`: the two are the same operation.
  2. weight normalisation is implemented here (w = g * v/||v||), because Keras ships none. It is
     applied per output filter, as `torch.nn.utils.weight_norm` does by default (dim=0).
  3. head. The reference's sequence tasks put a linear layer on the last position; this does the
     same, Dense(1) on the last step, to predict one value h steps ahead.
  4. size. num_channels defaults to [20] x 5 and kernel_size to 2 (the reference's own default), so
     the receptive field is 1 + 2*(k-1)*sum(2**i) = 63 >= the 60-step window. The reference's papers
     use 25..150 channels depending on the task; the width here was chosen by a stated rule — the
     value whose trainable-parameter count is closest to our own core's — so the contrast is about
     the BLOCK and not about capacity: 8 061 against our 8 127. Widths 12/16/20/24/32 give
     3 109/5 297/8 061/11 401/19 809, and 20 is the closest. Both counts travel with every result.
  5. training. Optimiser, learning rate, batch, budget, early stopping and the loss are taken from
     the run being compared against, NOT from the reference's recipe (which uses gradient clipping
     and its own schedules). Holding them fixed is what makes the architecture the only thing that
     differs; it also means this is not a reproduction of the paper's numbers.

None of the above promises the paper's performance on this data. A TCN is not a result; it is a
block, and this one is here to be measured on the same rows as everything else.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = {
    "paper": "Bai, Kolter & Koltun 2018, An Empirical Evaluation of Generic Convolutional and "
             "Recurrent Networks for Sequence Modeling (arXiv:1803.01271)",
    "implementation": "https://github.com/locuslab/TCN/blob/master/TCN/tcn.py",
    "block": "weight_norm(Conv1d) -> chomp -> ReLU -> Dropout, twice, + 1x1 residual, then ReLU",
    "defaults_in_the_source": {"kernel_size": 2, "dropout": 0.2, "dilation": "2**i per level",
                               "init": "conv weights ~ N(0, 0.01)"},
    "ours_for_contrast": "one Conv1D(k=3, ELU) + 1x1 projection skip, no weight norm, no dropout, "
                         "no activation after the sum",
}


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tf():
    return _module("df_mod_e0")._tf()


def weight_normalised_conv(tf):
    """Conv1D whose kernel is g * v/||v|| per output filter, as torch's weight_norm(dim=0) does."""

    class WeightNormConv1D(tf.keras.layers.Layer):
        def __init__(self, filters, kernel_size, dilation_rate=1, **kw):
            super().__init__(**kw)
            self.filters, self.kernel_size, self.dilation_rate = filters, kernel_size, dilation_rate

        def build(self, shape):
            in_channels = int(shape[-1])
            init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)   # the source's init
            self.v = self.add_weight(name="v", shape=(self.kernel_size, in_channels, self.filters),
                                     initializer=init, trainable=True)
            self.g = self.add_weight(name="g", shape=(self.filters,),
                                     initializer=tf.keras.initializers.Ones(), trainable=True)
            self.b = self.add_weight(name="b", shape=(self.filters,),
                                     initializer=tf.keras.initializers.Zeros(), trainable=True)
            super().build(shape)

        def call(self, x):
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=[0, 1]) + 1e-12)
            kernel = self.v * (self.g / norm)
            # keras.ops.conv, the operation Conv1D itself uses: tf.nn.conv1d has no CPU gradient for
            # a dilation above one, and this port must run on CPU like everything else in the round
            y = tf.keras.ops.conv(x, kernel, strides=1, padding="valid",
                                  dilation_rate=self.dilation_rate)
            return y + self.b

        def get_config(self):
            return {**super().get_config(), "filters": self.filters,
                    "kernel_size": self.kernel_size, "dilation_rate": self.dilation_rate}

    return WeightNormConv1D


def temporal_block(tf, x, filters, kernel_size, dilation, dropout, name):
    """The reference block, in order: two normalised causal convolutions with ReLU and dropout."""
    WN = weight_normalised_conv(tf)
    pad = (kernel_size-1)*dilation
    out = x
    for i in (1, 2):
        out = tf.keras.layers.ZeroPadding1D((pad, 0), name=f"{name}_pad{i}")(out)   # causal: pad left
        out = WN(filters, kernel_size, dilation_rate=dilation, name=f"{name}_wnconv{i}")(out)
        out = tf.keras.layers.ReLU(name=f"{name}_relu{i}")(out)
        out = tf.keras.layers.Dropout(dropout, name=f"{name}_drop{i}")(out)
    if int(x.shape[-1]) != filters:
        res = tf.keras.layers.Conv1D(filters, 1, name=f"{name}_downsample",
                                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.01))(x)
    else:
        res = x
    out = tf.keras.layers.Add(name=f"{name}_add")([out, res])
    return tf.keras.layers.ReLU(name=f"{name}_out")(out)


def build(window: int, channels: int, *, num_channels=(20, 20, 20, 20, 20), kernel_size: int = 2,
          dropout: float = 0.2, seed: int = 1):
    """The reference TCN over a (window, channels) input, one value out."""
    tf = _tf()
    tf.keras.utils.set_random_seed(int(seed))
    inp = tf.keras.Input(shape=(window, channels), name="window")
    x = inp
    for i, filters in enumerate(num_channels):
        x = temporal_block(tf, x, filters, kernel_size, 2**i, dropout, name=f"block{i}")
    last = tf.keras.layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)
    out = tf.keras.layers.Dense(1, name="readout")(last)
    model = tf.keras.Model(inp, out, name="tcn_reference")
    return model


def receptive_field(num_channels, kernel_size: int) -> int:
    return 1 + 2*(kernel_size-1)*sum(2**i for i in range(len(num_channels)))


def description(window: int, channels: int, **kw) -> dict:
    model = build(window, channels, **kw)
    num_channels = kw.get("num_channels", (20, 20, 20, 20, 20))
    kernel_size = kw.get("kernel_size", 2)
    return {"reference": REFERENCE, "levels": len(num_channels), "num_channels": list(num_channels),
            "kernel_size": kernel_size, "dropout": kw.get("dropout", 0.2),
            "receptive_field_steps": receptive_field(num_channels, kernel_size),
            "window_steps": window,
            "covers_the_window": receptive_field(num_channels, kernel_size) >= window,
            "trainable_parameters": int(sum(int(w.shape.num_elements()) for w in model.trainable_weights)),
            "head": "Dense(1) on the last position"}


if __name__ == "__main__":
    import json
    print(json.dumps(description(60, 7), indent=1))
