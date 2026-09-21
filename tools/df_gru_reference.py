#!/usr/bin/env python3
"""RP69: the literature comparator, ADAPTED from Gasparin, Lukovic & Alippi (2019) and measured on our task.

Primary source, read at its origin (arXiv:1907.09207 v1; CAAI Trans. Intell. Technol. 7(1), 2022):
Table 4, IHEPC column, GRU-MIMO: L = 1 recurrent layer, n_H = 50 units, L2 lambda = 0.0005, dropout 0.
Section 6: the MIMO strategy emits the whole horizon vector from the last hidden state through a
dense readout. Section 7.2: the household task is the 15-minute resampled load, 384-step window,
96-step horizon, "using only historical load values". Section 7.1: metrics; eq. 21: MSE loss.

What is taken from the article, what is adapted here, what the article does not say — kept apart:

  ARTICLE FACT       one GRU layer, 50 units, L2 0.0005, dropout 0.0, dense linear readout of the last
                     hidden state (MIMO); MSE training loss; grid search over Table 4's space; 10 repeats.
  ADAPTATION         our one 60-minute-ahead target (h = 60 at 1-minute resolution) instead of the 96-step
                     day-ahead vector at 15 minutes: the readout is Dense(1); the SAME permitted inputs as
                     the modular arm (7 channels, target history included) instead of "historical load
                     only", so the two architectures see identical information; the household CONTINUITY
                     recipe (MAE loss, Adam, lr 3e-3, batch 64, validation in observed updates, restore
                     best) instead of the article's MSE + unstated optimizer, so training is a held
                     factor and the architecture is the contrast. A native-recipe variant (MSE) is a
                     separately costed factor, not run in this block.
  UNKNOWN            the optimizer, learning rate, batch size, epochs, early stopping, initialization,
                     the regularized tensors (kernel, recurrent kernel, bias?), the exact split dates,
                     the imputation rule's population, seeds and code. None is invented: every choice
                     below is declared as ours.

Declared here (ours, not the article's): L2 is applied to the GRU kernel AND recurrent kernel (bias
excluded), and to the readout kernel; initialization is Keras' default (glorot_uniform kernel,
orthogonal recurrent kernel, zero bias; reset_after=True); the readout is linear with no persistence
skip (the article describes none). The reach of a GRU is the whole window by construction and is
MEASURED by gradient and perturbation like every other receiver (tools/df_e1_receiver.py).

    parameters(p, units)   the analytic count, checked against the built graph in tests
    build(W, p, j, seed)   the model; the target channel index j is recorded in the name only — the
                           readout is trained on the target's own scaled series, as the modular arm is
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SOURCE = {
    "citation": "Gasparin, Lukovic, Alippi. Deep Learning for Time Series Forecasting: The Electric Load Case. "
                "arXiv:1907.09207 (2019); CAAI Trans. Intell. Technol. 7(1), 2022, doi:10.1049/cit2.12060",
    "read_at": "arXiv PDF v1: Sec. 6 (MIMO), 7.1 (metrics), 7.2 (IHEPC), Table 4 (best configurations), Table 5 (results)",
    "table4_gru_mimo_ihepc": {"L": 1, "n_H": 50, "lambda_l2": 0.0005, "dropout": 0.0},
    "code": "NOT PUBLIC",
}
ADAPTATION = {
    "target": "one value, 60 minutes ahead at 1-minute resolution (ours) instead of the 96-step day-ahead vector at 15 min",
    "inputs": "the modular arm's 7 permitted channels, target history included (ours) instead of historical load only",
    "readout": "Dense(1, linear) on the last hidden state; no persistence skip (the article describes none)",
    "regularized_tensors": "L2 0.0005 on the GRU kernel, the GRU recurrent kernel and the readout kernel; biases excluded (ours)",
    "initialization": "Keras default: glorot_uniform kernel, orthogonal recurrent kernel, zeros bias, reset_after=True (ours)",
    "training": "the household continuity recipe (MAE, Adam 3e-3, batch 64, validation in observed updates, restore best) "
                "instead of the article's MSE with an unstated optimizer; a native MSE variant is a separate factor",
    "unknown_in_the_article": ["optimizer", "learning rate", "batch", "epochs/early stopping", "initialization",
                               "which tensors L2 covers", "exact split dates", "imputation population", "seeds", "code"],
    "what_this_is": "an adapted, literature-derived receiver measured under OUR contract; NOT a reproduction of Table 5",
}


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parameters(p: int, units: int = 50) -> int:
    """Keras GRU (reset_after=True): kernel p x 3u, recurrent u x 3u, bias 2 x 3u; readout u + 1."""
    return 3*units*p + 3*units*units + 2*3*units + units + 1


def build(W: int, p: int, j: int, seed: int, *, units: int = 50, l2: float = 0.0005, dropout: float = 0.0):
    tf = _module("df_mod_e0")._tf()
    tf.keras.utils.set_random_seed(int(seed))
    reg = tf.keras.regularizers.L2(l2) if l2 else None
    inp = tf.keras.Input(shape=(W, p), name="x")
    h = tf.keras.layers.GRU(units, dropout=dropout, recurrent_dropout=0.0, return_sequences=False,
                            kernel_regularizer=reg, recurrent_regularizer=reg, bias_regularizer=None,
                            name="gru")(inp)
    out = tf.keras.layers.Dense(1, activation=None, kernel_regularizer=reg, name="readout")(h)
    return tf.keras.Model(inp, out, name=f"gru_mimo_adapted_target{j}")


def declaration(W: int, p: int, units: int = 50, l2: float = 0.0005) -> dict:
    return {"source": SOURCE, "adaptation": ADAPTATION,
            "layers": {"gru": f"GRU({units}), tanh state / sigmoid gates, reset_after=True, return last state",
                       "readout": "Dense(1), linear"},
            "regularization": {"l2": l2, "on": ["gru/kernel", "gru/recurrent_kernel", "readout/kernel"]},
            "dropout": 0.0, "window": W, "channels": p, "parameters_analytic": parameters(p, units)}
