#!/usr/bin/env python3
"""RP58: what a whole-series wavelet leak DOES to a forecast, measured on the affected family's data.

The published phase-3 tables beat the naive by 96-97% at one to six hours on an FX price. No
forecast does that. But a suspicious score is not a finding, and the order is explicit: an
extraordinary improvement triggers the checks, it does not conclude them. So this does not classify
any historical run. It calibrates the SIGNATURE, on the same kind of series, with a mechanism that
is already a named negative control of this repository's causal battery
(`full_series_dwt_as_time_row`, detected by PREFIX_ALL_T).

Three representations of the same series, the same rows, the same labels, the same model, the same
budget and the same seeds:

  RAW              the price history alone
  TRAILING_WAVELET a wavelet decomposition computed from the PAST ONLY at each row: at row t the
                   transform sees x[:t+1] and nothing else
  WHOLE_SERIES_DWT the decomposition computed ONCE over the WHOLE series and then read row by row —
                   the classical leak: row t's feature was built with knowledge of x[t+1:]

What comes out is the size of the effect, in skill against the naive, for a mechanism that is known
and deliberate. It says what such a leak looks like; it does not say that any particular past run
had one.

    python tools/df_leak_signature.py --series CSV --column CLOSE --out REPORT.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def _series(path: Path, column: str, limit: int | None = None) -> np.ndarray:
    import csv
    values = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                values.append(float(row[column]))
            except (TypeError, ValueError, KeyError):
                continue
            if limit and len(values) >= limit:
                break
    return np.asarray(values, dtype=np.float64)


def trailing_wavelet(x: np.ndarray, *, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """At every row, the decomposition of the past only: x[:t+1] and nothing after it."""
    import pywt
    out = np.zeros((x.size, level+1))
    support = pywt.Wavelet(wavelet).dec_len * (2**level)
    for t in range(x.size):
        window = x[max(0, t-support+1):t+1]
        if window.size < support:
            out[t] = np.nan
            continue
        coeffs = pywt.wavedec(window, wavelet, level=level, mode="periodization")
        out[t] = [float(c[-1]) for c in coeffs]        # the last coefficient of each band
    return out


def whole_series_dwt(x: np.ndarray, *, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """The decomposition of the WHOLE series, read row by row. Row t carries x[t+1:] in it."""
    import pywt
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    bands = []
    for c in coeffs:
        up = np.repeat(c, int(np.ceil(x.size/c.size)))[:x.size]
        bands.append(up)
    return np.stack(bands, axis=1)


def whole_series_reconstruction(x: np.ndarray, *, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """The denoised signal, reconstructed from the WHOLE series at FULL resolution.

    This is what a "wavelet denoising" feature usually is in a pipeline: decompose everything,
    threshold the detail bands, rebuild, and hand the result to the model row by row. Every row of
    the output is a function of the entire series, so row t carries information from x[t+1:] at the
    same resolution as the series itself. The coarse band-repeat variant above understates it.
    """
    import pywt
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    sigma = float(np.median(np.abs(coeffs[-1]-np.median(coeffs[-1])))/0.6745) if coeffs[-1].size else 0.0
    threshold = sigma*np.sqrt(2*np.log(max(2, x.size)))
    kept = [coeffs[0]] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
    rebuilt = pywt.waverec(kept, wavelet, mode="periodization")[:x.size]
    return np.stack([rebuilt, x-rebuilt], axis=1)          # the denoised level and its residual


def centred_smoother(x: np.ndarray, *, half: int = 6) -> np.ndarray:
    """A centred moving average: row t averages x[t-half : t+half+1].

    Another declared non-causal class of the battery (`centered_rolling_window`). It is included
    because the wavelet variants above do NOT reproduce the magnitude seen in the published tables,
    and the question of WHICH family of mechanisms can is worth answering with a declared control
    rather than by tuning a wavelet until it matches a suspicion.
    """
    kernel = np.ones(2*half+1)/(2*half+1)
    padded = np.pad(x, half, mode="edge")
    smooth = np.convolve(padded, kernel, mode="valid")[:x.size]
    return np.stack([smooth, x-smooth], axis=1)


def _windows(features: np.ndarray, target: np.ndarray, origins: np.ndarray, w: int) -> tuple:
    X = np.stack([features[o-w+1:o+1] for o in origins])
    return X, target[origins]


def fit_and_score(name: str, features: np.ndarray, x: np.ndarray, *, window: int, horizon: int,
                  seed: int, updates: int, split: float = 0.8) -> dict:
    """The same model on each representation, predicting the CHANGE over the horizon.

    Two things this had to get right, and did not at first:

      * the head predicts y(t+h) - y(t) and the last observation is added back. Against a
        random-walk-like series the naive IS predicting a zero change, so a model that predicts the
        level from a normalised window starts far behind it and nothing can be read. With a change
        head, zero output reproduces the naive exactly and any skill is information.
      * a row is usable only when the WHOLE window is finite. A trailing transform is undefined over
        its warm-up, and letting those rows into a window turns the error into NaN.
    """
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("df_mod_e0", Path(__file__).resolve().parent/"df_mod_e0.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["df_mod_e0"] = mod
    spec.loader.exec_module(mod)
    tf = mod._tf()
    tf.keras.utils.set_random_seed(int(seed))
    n = x.size
    finite_row = np.isfinite(features).all(axis=1)
    whole_window = np.array([finite_row[max(0, o-window+1):o+1].all() for o in range(n)])
    usable = np.arange(window-1, n-horizon)
    usable = usable[whole_window[usable]]
    cut = int(len(usable)*split)
    tr, ev = usable[:cut], usable[cut:]
    mean, sd = features[tr].mean(axis=0), features[tr].std(axis=0)
    sd[sd == 0] = 1.0
    scaled = (features-mean)/sd
    change_tr = x[tr+horizon]-x[tr]
    cm, cs = float(change_tr.mean()), float(change_tr.std())
    cs = cs if cs > 0 else 1.0
    Xtr, _ = _windows(scaled, x, tr, window)
    Xev, _ = _windows(scaled, x, ev, window)
    ytr = ((change_tr-cm)/cs)[:, None]
    yev = x[ev+horizon]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window, features.shape[1])),
        tf.keras.layers.Conv1D(16, 3, activation="elu", padding="causal"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="elu"),
        # zero-initialised, so an untrained model predicts no change: exactly the naive
        tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    batch = 64
    epochs = max(1, int(np.ceil(updates/max(1, len(Xtr)//batch))))
    model.fit(Xtr, ytr, epochs=epochs, batch_size=batch, verbose=0)
    predicted_change = model.predict(Xev, verbose=0).reshape(-1)*cs + cm
    pred = x[ev] + predicted_change
    naive = x[ev]
    mae = float(np.mean(np.abs(pred-yev)))
    naive_mae = float(np.mean(np.abs(naive-yev)))
    return {"representation": name, "channels": int(features.shape[1]),
            "train_rows": int(tr.size), "evaluation_rows": int(ev.size),
            "head": "change over the horizon, zero-initialised: no change = the naive",
            "mae": mae, "naive_mae": naive_mae,
            "mae_skill_percent_vs_naive": 100*(1-mae/naive_mae) if naive_mae > 0 else None}


def report(series_path: Path, column: str, *, window: int = 24, horizon: int = 6, seed: int = 1,
           updates: int = 600, limit: int | None = 20000) -> dict:
    x = _series(series_path, column, limit)
    raw = x[:, None]
    trail = np.concatenate([raw, trailing_wavelet(x)], axis=1)
    whole = np.concatenate([raw, whole_series_dwt(x)], axis=1)
    rebuilt = np.concatenate([raw, whole_series_reconstruction(x)], axis=1)
    rows = [fit_and_score("RAW", raw, x, window=window, horizon=horizon, seed=seed, updates=updates),
            fit_and_score("TRAILING_WAVELET", trail, x, window=window, horizon=horizon, seed=seed,
                          updates=updates),
            fit_and_score("WHOLE_SERIES_DWT", whole, x, window=window, horizon=horizon, seed=seed,
                          updates=updates),
            fit_and_score("WHOLE_SERIES_RECONSTRUCTION", rebuilt, x, window=window, horizon=horizon,
                          seed=seed, updates=updates),
            fit_and_score("CENTRED_SMOOTHER", np.concatenate([raw, centred_smoother(x, half=horizon)], axis=1),
                          x, window=window, horizon=horizon, seed=seed, updates=updates)]
    leak = next(r for r in rows if r["representation"] == "WHOLE_SERIES_RECONSTRUCTION")
    causal = next(r for r in rows if r["representation"] == "TRAILING_WAVELET")
    return {"schema": "df_leak_signature.v1",
            "at": datetime.utcnow().isoformat(timespec="seconds")+"Z",
            "series": str(series_path), "column": column, "rows_used": int(x.size),
            "window": window, "horizon_steps": horizon, "seed": seed, "updates": updates,
            "results": rows,
            "signature": {"leaking_minus_causal_skill_points":
                          (leak["mae_skill_percent_vs_naive"]-causal["mae_skill_percent_vs_naive"]),
                          "leaking_skill": leak["mae_skill_percent_vs_naive"],
                          "causal_skill": causal["mae_skill_percent_vs_naive"]},
            "scope": "one series, one seed, one model: this CALIBRATES what a known leak does, and "
                     "classifies no historical run. The mechanism used here is a named negative "
                     "control of tools/df_causal_battery.py (full_series_dwt_as_time_row), which "
                     "detects it by prefix equality."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", type=Path, required=True)
    ap.add_argument("--column", default="CLOSE")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--updates", type=int, default=600)
    a = ap.parse_args(argv)
    doc = report(a.series, a.column, window=a.window, horizon=a.horizon, updates=a.updates)
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({"results": doc["results"], "signature": doc["signature"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
