#!/usr/bin/env python3
"""The nine D3 operators of design 07 §2 and their non-causal twins (J3).

Each operator carries the v2 declaration of `df_d3_contract` and is measured by
`df_d3_acceptance` against the sealed amendment `df_d3_design.D3_AMENDMENT_V1`. Established
numerical libraries do the transforms — numpy, scipy.signal, pywt — and every declaration
records their versions.

    quantization / compression   uniform_decile_quantizer, sax_paa_trailing, delta_run_length
    time-frequency               stft_trailing, wavelet_trailing, butterworth_causal
    detectors                    cusum_causal, mad_extremes_trailing, variance_regime_trailing

Twins (deliberately non-causal, recorded and never promoted): sax_paa_centred, stft_centred,
wavelet_centred, butterworth_filtfilt, cusum_lookahead, mad_extremes_centred,
variance_regime_centred. The two pointwise codecs declare NOT_APPLICABLE with the design
reason of the amendment §5.

Conventions shared by every operator, so none invents its own clock or its own missingness:

* the input is a `df_d3_contract` input (values, timestamps, available_at, period_seconds);
* `emitted_at` comes from `df_d3_contract.emission_times` with the declared lookback and
  emission delay;
* a window that contains a missing value yields an UNAVAILABLE output, never a number; a
  recursive operator skips a missing input and leaves its state where it was;
* the raw branch is the input, untouched;
* `fit` sees the train prefix only; `transform` never mutates the state it is handed.

Nothing here scores, selects or promotes anything.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    import sys

    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("df_d3_contract")

FAMILIES_ALL = ["sinusoid", "multiband", "trend_linear", "trend_piecewise", "seasonal", "chirp",
                "impulses", "steps", "bumps", "motif", "regime_mean", "regime_variance",
                "regime_frequency", "multivariate", "null", "toy_price", "toy_ohlc",
                "toy_features"]


def library_versions() -> dict:
    out = {"numpy": np.__version__}
    try:
        import scipy

        out["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        out["scipy"] = "ABSENT"
    try:
        import pywt

        out["pywt"] = pywt.__version__
    except Exception:  # pragma: no cover
        out["pywt"] = "ABSENT"
    return out


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _window_ok(values, start: int, stop: int) -> bool:
    return all(_finite(v) for v in values[start:stop])


def _as_array(values) -> np.ndarray:
    out = np.asarray([v if _finite(v) else np.nan for v in values], dtype=np.float64)
    return out


def _trailing_windows(v: np.ndarray, w: int):
    """(n - w + 1, w) view of every trailing window, and the mask of windows with no NaN."""
    from numpy.lib.stride_tricks import sliding_window_view

    if v.size < w:
        return np.empty((0, w)), np.zeros(0, dtype=bool)
    windows = sliding_window_view(v, w)
    return windows, ~np.isnan(windows).any(axis=1)


def _centred_windows(v: np.ndarray, w: int):
    """Windows [i - w//2, i - w//2 + w) for every i that has one: the twin's leak."""
    windows, ok = _trailing_windows(v, w)
    return windows, ok


def _trailing_apply(v: np.ndarray, w: int, fn):
    """Apply `fn(windows) -> (m,)` to every complete trailing window; earlier rows unavailable."""
    n = v.size
    values = np.zeros(n)
    available = np.zeros(n, dtype=bool)
    windows, ok = _trailing_windows(v, w)
    if windows.shape[0]:
        result = np.zeros(windows.shape[0])
        if ok.any():
            result[ok] = fn(windows[ok])
        values[w - 1:] = result
        available[w - 1:] = ok
    return values, available


def _centred_apply(v: np.ndarray, w: int, fn):
    """Apply `fn` to centred windows: output i uses [i - w//2, i - w//2 + w)."""
    n = v.size
    half = w // 2
    values = np.zeros(n)
    available = np.zeros(n, dtype=bool)
    windows, ok = _trailing_windows(v, w)
    if windows.shape[0]:
        result = np.zeros(windows.shape[0])
        if ok.any():
            result[ok] = fn(windows[ok])
        # window k covers [k, k + w) and is centred on i = k + half
        values[half:half + windows.shape[0]] = result
        available[half:half + windows.shape[0]] = ok
    return values, available


# --- base -----------------------------------------------------------------------------------

class Operator:
    KIND = ""
    GROUP = ""
    FIT_SCOPE = "NONE"
    CHUNK_RESTART = "IDEMPOTENT"
    APPLICABILITY = list(FAMILIES_ALL)
    DEFAULT_PARAMS: dict = {}
    #: measured by the cost pilot; declared here as an upper bound the battery re-measures
    COST_CPU_SECONDS_PER_1000 = 2.0

    def __init__(self, **params):
        self.params = dict(self.DEFAULT_PARAMS, **params)
        for name in params:
            if name not in self.DEFAULT_PARAMS:
                raise contract.SpecRefusal(f"{self.KIND}: unknown parameter {name!r}")

    # declaration ---------------------------------------------------------------------------
    def lookback(self) -> int:
        raise NotImplementedError

    def warm_up(self) -> int:
        return self.lookback()

    def delay(self) -> int:
        return 0

    def probe(self) -> dict:
        raise NotImplementedError

    def reach_right(self, n: int) -> int:
        """How many samples AFTER i output i consumes (L2). A causal operator: 0."""
        return 0

    def probe_resolution(self, state, *, baseline: float, scale: float, sigma: float) -> dict:
        """The smallest excitation, in input units at `baseline`, this operator guarantees moves
        its impact-sample output — from its training fit only (K2). The quiet branch carries
        noise of `sigma` (3 sd band); a fitted codec must clear the first edge ABOVE that band.
        A linear or unbounded response moves for any excitation of the fitted scale;
        UNIDENTIFIED with a reason when nothing is guaranteed."""
        return {"amplitude": float(scale),
                "reason": "the response moves for any excitation of the fitted scale"}

    def twin(self) -> dict:
        raise NotImplementedError

    def support(self) -> dict:
        raise NotImplementedError

    def bytes_state(self) -> int:
        return 0

    def describe(self) -> dict:
        return {"schema": contract.SPEC_SCHEMA, "kind": self.KIND, "params": dict(self.params),
                "bytes_state": self.bytes_state(), "fit_scope": self.FIT_SCOPE,
                "lookback_samples": self.lookback(), "warm_up_samples": self.warm_up(),
                "delay_samples": self.delay(),
                "output_availability": "t" if not self.delay() else f"t + {self.delay()}",
                "response_probe": self.probe(), "non_causal_twin": self.twin(),
                "support": self.support(),
                "cost_cpu_seconds_per_1000": float(self.COST_CPU_SECONDS_PER_1000),
                "applicability": list(self.APPLICABILITY),
                "chunk_restart": self.CHUNK_RESTART,
                "library_versions": library_versions()}

    # behaviour -----------------------------------------------------------------------------
    def fit(self, train):
        return {}

    def _emit(self, x) -> list:
        return contract.emission_times(x, lookback=self.lookback(), delay=self.delay())

    def _pack(self, x, values, available) -> dict:
        return {"values": [float(v) for v in values], "available": [bool(a) for a in available],
                "emitted_at": self._emit(x), "raw": list(x["values"])}

    def transform(self, x, state):
        raise NotImplementedError

    def apply_to_family(self, family, x, state):
        if family not in self.APPLICABILITY:
            return contract.NOT_APPLICABLE
        return self.transform(x, state)


def _na_twin(reason: str) -> dict:
    return {"not_applicable": True, "reason": reason}


# --- quantization / compression -------------------------------------------------------------

class UniformDecileQuantizer(Operator):
    """Levels from the train prefix's quantiles; each sample maps to its level. Pointwise."""

    KIND = "uniform_decile_quantizer"
    GROUP = "quantization_compression"
    FIT_SCOPE = "TRAIN_PREFIX_ONLY"
    DEFAULT_PARAMS = {"levels": 10}

    def lookback(self):
        return 0

    def bytes_state(self):
        return 8 * (int(self.params["levels"]) + 1)

    def probe(self):
        return {"kind": "step", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return _na_twin("a stateless pointwise codec has no window to centre")

    def support(self):
        return {"kind": "POINTWISE", "samples": 1, "derivation": "one sample, no window",
                "boundary_mode": None}

    def fit(self, train):
        finite = [v for v in train["values"] if _finite(v)]
        if len(finite) < 2:
            raise contract.SpecRefusal("the train prefix has fewer than two finite samples")
        k = int(self.params["levels"])
        qs = np.quantile(np.asarray(finite, dtype=np.float64), np.linspace(0, 1, k + 1))
        return {"edges": [float(q) for q in qs[1:-1]]}

    def probe_resolution(self, state, *, baseline, scale, sigma):
        # a step moves the level at the impact sample iff it crosses an edge the quiet sample
        # has not crossed: the first edge above the noise band (baseline + 3 sigma), cleared by
        # another 3 sigma; the baseline of a decile codec fitted on the same train IS an edge
        ceiling = baseline + 3.0 * sigma
        above = [e for e in state["edges"] if e > ceiling]
        if not above:
            return {"amplitude": contract.UNIDENTIFIED,
                    "reason": "saturation: no fitted edge lies above the baseline's noise band"}
        return {"amplitude": (above[0] - baseline) + 3.0 * sigma,
                "reason": f"first fitted edge above the noise band at {above[0] - baseline:.6g} "
                          "over the baseline"}

    def transform(self, x, state):
        edges = np.asarray(state["edges"], dtype=np.float64)
        v = _as_array(x["values"])
        ok = ~np.isnan(v)
        values = np.zeros(v.size)
        values[ok] = np.searchsorted(edges, v[ok], side="right").astype(np.float64)
        return self._pack(x, values, ok)


_SAX_BREAKPOINTS = {3: [-0.43, 0.43], 4: [-0.67, 0.0, 0.67], 5: [-0.84, -0.25, 0.25, 0.84],
                    6: [-0.97, -0.43, 0.0, 0.43, 0.97]}


class SaxPaaTrailing(Operator):
    """Trailing PAA over the last `segment` samples, z-scored with train statistics, mapped to
    a symbol by Gaussian breakpoints. The window is the last `segment` samples, never centred."""

    KIND = "sax_paa_trailing"
    GROUP = "quantization_compression"
    FIT_SCOPE = "TRAIN_PREFIX_ONLY"
    DEFAULT_PARAMS = {"segment": 8, "alphabet": 4}

    def lookback(self):
        return int(self.params["segment"]) - 1

    def bytes_state(self):
        return 16 + 8 * (int(self.params["alphabet"]) - 1)

    def probe(self):
        return {"kind": "step", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return {"kind": "sax_paa_centred"}

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["segment"]),
                "derivation": "trailing segment of `segment` samples", "boundary_mode": None}

    def fit(self, train):
        finite = np.asarray([v for v in train["values"] if _finite(v)], dtype=np.float64)
        if finite.size < 2:
            raise contract.SpecRefusal("the train prefix has fewer than two finite samples")
        sd = float(finite.std()) or 1.0
        a = int(self.params["alphabet"])
        if a not in _SAX_BREAKPOINTS:
            raise contract.SpecRefusal(f"alphabet {a} has no declared breakpoints")
        return {"mean": float(finite.mean()), "sd": sd, "breakpoints": list(_SAX_BREAKPOINTS[a])}

    def probe_resolution(self, state, *, baseline, scale, sigma):
        # the impact window holds s-1 quiet samples and one excited sample: its PAA rises by
        # amplitude/s; the symbol moves iff that crosses a breakpoint above the quiet window's
        # own z band (noise of the window mean is sigma/sqrt(s); 3 sd cleared on both sides)
        seg = int(self.params["segment"])
        band = 3.0 * sigma / (seg ** 0.5)
        z_ceiling = (baseline + band - state["mean"]) / state["sd"]
        above = [b for b in state["breakpoints"] if b > z_ceiling]
        if not above:
            return {"amplitude": contract.UNIDENTIFIED,
                    "reason": "saturation: the baseline's PAA symbol is already the top one"}
        rise = above[0] * state["sd"] + state["mean"] - baseline + 2.0 * band
        return {"amplitude": seg * rise,
                "reason": f"first breakpoint above the quiet window's z band at {above[0]}, over "
                          f"a segment of {seg}"}

    def _segment(self, values, i):
        s = int(self.params["segment"])
        return values[i + 1 - s:i + 1]

    def _symbols(self, state):
        bps = np.asarray(state["breakpoints"], dtype=np.float64)
        mean, sd = state["mean"], state["sd"]

        def fn(windows):
            paa = (windows.mean(axis=1) - mean) / sd
            return np.searchsorted(bps, paa, side="right").astype(np.float64)
        return fn

    def transform(self, x, state):
        values, available = _trailing_apply(_as_array(x["values"]), int(self.params["segment"]),
                                            self._symbols(state))
        return self._pack(x, values, available)


class SaxPaaCentred(SaxPaaTrailing):
    """Twin: the same segment, centred on t. Reads the future by construction."""

    KIND = "sax_paa_centred"

    def lookback(self):
        return int(self.params["segment"]) // 2

    def warm_up(self):
        return int(self.params["segment"]) // 2

    def twin(self):
        return _na_twin("it is itself a twin")

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["segment"]) // 2 + 1,
                "derivation": "centred segment; the future half is the leak",
                "boundary_mode": None}

    def reach_right(self, n):
        seg = int(self.params["segment"])
        return seg - seg // 2 - 1

    def describe(self):
        return dict(super().describe(), non_causal_control_of="sax_paa_trailing")

    def transform(self, x, state):
        values, available = _centred_apply(_as_array(x["values"]), int(self.params["segment"]),
                                           self._symbols(state))
        return self._pack(x, values, available)


class DeltaRunLength(Operator):
    """x[t] - x[t-1]; the run length of the delta's sign travels as an auxiliary output."""

    KIND = "delta_run_length"
    GROUP = "quantization_compression"
    DEFAULT_PARAMS = {}

    def lookback(self):
        return 1

    def probe(self):
        return {"kind": "impulse", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return _na_twin("delta of one sample and its run length have no window to centre; "
                        "the only past consumed is x[t-1]")

    def support(self):
        return {"kind": "FINITE", "samples": 2, "derivation": "x[t] and x[t-1]",
                "boundary_mode": None}

    def transform(self, x, state):
        v = x["values"]
        values, available, runs = [], [], []
        run, last_sign = 0, 0
        for i in range(len(v)):
            if i == 0 or not _finite(v[i]) or not _finite(v[i - 1]):
                values.append(0.0)
                available.append(False)
                runs.append(0)
                run, last_sign = 0, 0
                continue
            d = v[i] - v[i - 1]
            sign = (d > 0) - (d < 0)
            run = run + 1 if sign == last_sign else 1
            last_sign = sign
            values.append(d)
            available.append(True)
            runs.append(run)
        out = self._pack(x, values, available)
        out["aux"] = {"run_length": runs}
        return out


# --- time-frequency --------------------------------------------------------------------------

class StftTrailing(Operator):
    """Spectral centroid of a Hann-windowed rFFT over the last `w` samples. Hop is one sample
    so every t has its own trailing spectrum.

    Measured, not assumed: a Hann window weights its newest sample by exactly zero, so the
    output for t does not see x[t] at all and an impulse first moves it one sample later.
    The probe declares that onset of 1. Reshaping the window to make it 0 would be tuning
    the operator against the evaluation fixture, which the order forbids."""

    KIND = "stft_trailing"
    GROUP = "time_frequency"
    DEFAULT_PARAMS = {"w": 16}
    COST_CPU_SECONDS_PER_1000 = 4.0

    def lookback(self):
        return int(self.params["w"]) - 1

    def probe(self):
        return {"kind": "impulse", "expected_onset_samples": 1, "scale": contract.PROBE_SCALE}

    def twin(self):
        return {"kind": "stft_centred"}

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["w"]),
                "derivation": "trailing Hann window of w samples", "boundary_mode": None}

    def _centroids(self, windows):
        w = np.hanning(windows.shape[1])
        spectrum = np.abs(np.fft.rfft(windows * w, axis=1))
        total = spectrum.sum(axis=1)
        freqs = np.arange(spectrum.shape[1], dtype=np.float64)
        out = np.zeros(windows.shape[0])
        nz = total > 0.0
        out[nz] = (spectrum[nz] * freqs).sum(axis=1) / total[nz]
        return out

    def transform(self, x, state):
        values, available = _trailing_apply(_as_array(x["values"]), int(self.params["w"]),
                                            self._centroids)
        return self._pack(x, values, available)


class StftCentred(StftTrailing):
    KIND = "stft_centred"

    def lookback(self):
        return int(self.params["w"]) // 2

    def warm_up(self):
        return int(self.params["w"]) // 2

    def twin(self):
        return _na_twin("it is itself a twin")

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["w"]) // 2 + 1,
                "derivation": "centred window; the future half is the leak", "boundary_mode": None}

    def reach_right(self, n):
        w = int(self.params["w"])
        return w - w // 2 - 1

    def describe(self):
        return dict(super().describe(), non_causal_control_of="stft_trailing")

    def transform(self, x, state):
        values, available = _centred_apply(_as_array(x["values"]), int(self.params["w"]),
                                           self._centroids)
        return self._pack(x, values, available)


def wavelet_support(wavelet: str, levels: int) -> int:
    """(dec_len - 1) * (2^L - 1) + 1, from the library. 2^L describes Haar alone."""
    import pywt

    dec_len = int(pywt.Wavelet(wavelet).dec_len)
    return (dec_len - 1) * (2 ** int(levels) - 1) + 1


class WaveletTrailing(Operator):
    """The last level-L approximation coefficient of a wavedec over the trailing support, with
    zero boundary mode. Support is derived from the library's filter length, not assumed."""

    KIND = "wavelet_trailing"
    GROUP = "time_frequency"
    DEFAULT_PARAMS = {"wavelet": "db4", "levels": 3}
    COST_CPU_SECONDS_PER_1000 = 6.0

    def _support(self) -> int:
        return wavelet_support(self.params["wavelet"], int(self.params["levels"]))

    def lookback(self):
        return self._support() - 1

    def probe(self):
        return {"kind": "impulse", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return {"kind": "wavelet_centred"}

    def support(self):
        import pywt

        return {"kind": "FINITE", "samples": self._support(),
                "derivation": f"pywt.Wavelet({self.params['wavelet']!r}).dec_len="
                              f"{pywt.Wavelet(self.params['wavelet']).dec_len}, "
                              f"(dec_len-1)*(2^{self.params['levels']}-1)+1",
                "boundary_mode": "zero"}

    def _coef(self, seg):
        import warnings

        import pywt

        with warnings.catch_warnings():
            # The support is sized exactly to the level, so every coefficient sits at the
            # boundary by construction; the boundary mode is declared, not hidden.
            warnings.simplefilter("ignore", category=UserWarning)
            coeffs = pywt.wavedec(np.asarray(seg, dtype=np.float64), self.params["wavelet"],
                                  mode="zero", level=int(self.params["levels"]))
        return float(coeffs[0][-1])

    def _coefs(self, windows):
        import warnings

        import pywt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            coeffs = pywt.wavedec(windows, self.params["wavelet"], mode="zero",
                                  level=int(self.params["levels"]), axis=1)
        return coeffs[0][:, -1].astype(np.float64)

    def transform(self, x, state):
        values, available = _trailing_apply(_as_array(x["values"]), self._support(), self._coefs)
        return self._pack(x, values, available)


class WaveletCentred(WaveletTrailing):
    KIND = "wavelet_centred"

    def lookback(self):
        return self._support() // 2

    def warm_up(self):
        return self._support() // 2

    def twin(self):
        return _na_twin("it is itself a twin")

    def support(self):
        base = super().support()
        return dict(base, samples=self._support() // 2 + 1,
                    derivation=base["derivation"] + "; centred, the future half is the leak")

    def reach_right(self, n):
        w = self._support()
        return w - w // 2 - 1

    def describe(self):
        return dict(super().describe(), non_causal_control_of="wavelet_trailing")

    def transform(self, x, state):
        values, available = _centred_apply(_as_array(x["values"]), self._support(), self._coefs)
        return self._pack(x, values, available)


class ButterworthCausal(Operator):
    """scipy.signal.butter + lfilter with explicit zi: the state is a recursive dependency,
    not a finite memory of order p. Checkpoint/resume carry zi and the sample count."""

    KIND = "butterworth_causal"
    GROUP = "time_frequency"
    CHUNK_RESTART = "STATEFUL_WITH_CHECKPOINT"
    DEFAULT_PARAMS = {"order": 2, "cutoff": 0.1}
    LOOKBACK_ALL_PAST = 1 << 40

    def __init__(self, **params):
        super().__init__(**params)
        self._zi = None
        self._seen = 0

    def lookback(self):
        return self.LOOKBACK_ALL_PAST

    def warm_up(self):
        return 0

    def probe(self):
        return {"kind": "impulse", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return {"kind": "butterworth_filtfilt"}

    def support(self):
        return {"kind": "RECURSIVE", "samples": None,
                "derivation": f"butter(order={self.params['order']}, cutoff={self.params['cutoff']})"
                              " with lfilter state zi", "boundary_mode": None}

    def _ba(self):
        from scipy.signal import butter

        return butter(int(self.params["order"]), float(self.params["cutoff"]))

    def fit(self, train):
        return {"zi": None, "seen": 0}

    def transform(self, x, state):
        from scipy.signal import lfilter, lfiltic

        b, a = self._ba()
        v = _as_array(x["values"])
        zi = None if state.get("zi") is None else np.asarray(state["zi"], dtype=np.float64)
        seen = int(state.get("seen") or 0)
        values = np.zeros(v.size)
        available = ~np.isnan(v)
        # Runs of finite samples are filtered in one call each, the state carried across; a
        # missing sample leaves the state where it was, exactly as the sample-by-sample loop.
        i = 0
        while i < v.size:
            if not available[i]:
                i += 1
                continue
            j = i
            while j < v.size and available[j]:
                j += 1
            if zi is None:
                zi = lfiltic(b, a, y=[v[i]] * max(len(a) - 1, 1), x=[v[i]] * max(len(b) - 1, 1))
            y, zi = lfilter(b, a, v[i:j], zi=zi)
            values[i:j] = y
            seen += j - i
            i = j
        self._zi = None if zi is None else [float(z) for z in zi]
        self._seen = seen
        return self._pack(x, values, available)

    def checkpoint(self):
        return json.dumps({"zi": self._zi, "seen": self._seen})

    def resume(self, blob):
        doc = json.loads(blob)
        return {"zi": doc["zi"], "seen": doc["seen"]}


class ButterworthFiltfilt(ButterworthCausal):
    """Twin: zero-phase filtfilt over the whole series. Every output reads the future."""

    KIND = "butterworth_filtfilt"
    CHUNK_RESTART = "IDEMPOTENT"

    def twin(self):
        return _na_twin("it is itself a twin")

    def reach_right(self, n):
        return int(n)                      # zero-phase over the whole series

    def describe(self):
        return dict(super().describe(), non_causal_control_of="butterworth_causal")

    def transform(self, x, state):
        from scipy.signal import filtfilt

        b, a = self._ba()
        v = np.asarray(x["values"], dtype=np.float64)
        ok = np.isfinite(v)
        values = np.zeros_like(v)
        if ok.sum() > 3 * max(len(a), len(b)):
            values[ok] = filtfilt(b, a, v[ok])
        else:
            ok = np.zeros_like(ok)
        return self._pack(x, values.tolist(), ok.tolist())


# --- detectors ---------------------------------------------------------------------------------

class CusumCausal(Operator):
    """One-sided CUSUM of the positive shift, with drift k and threshold h from the train
    prefix (k = k_sd * sd, h = h_sd * sd). The value is the statistic; a detection is the
    statistic exceeding h. Recursive state: the running sums."""

    KIND = "cusum_causal"
    GROUP = "detectors"
    FIT_SCOPE = "TRAIN_PREFIX_ONLY"
    CHUNK_RESTART = "STATEFUL_WITH_CHECKPOINT"
    DEFAULT_PARAMS = {"k_sd": 0.5, "h_sd": 5.0}
    LOOKBACK_ALL_PAST = 1 << 40

    def __init__(self, **params):
        super().__init__(**params)
        self._sums = None

    def lookback(self):
        return self.LOOKBACK_ALL_PAST

    def warm_up(self):
        return 0

    def bytes_state(self):
        return 32

    def probe(self):
        return {"kind": "level_shift", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return {"kind": "cusum_lookahead"}

    def support(self):
        return {"kind": "RECURSIVE", "samples": None,
                "derivation": "running sums S+ and S-", "boundary_mode": None}

    def fit(self, train):
        finite = np.asarray([v for v in train["values"] if _finite(v)], dtype=np.float64)
        if finite.size < 2:
            raise contract.SpecRefusal("the train prefix has fewer than two finite samples")
        sd = float(finite.std()) or 1.0
        return {"mean": float(finite.mean()), "k": float(self.params["k_sd"]) * sd,
                "h": float(self.params["h_sd"]) * sd, "s_pos": 0.0, "s_neg": 0.0}

    def probe_resolution(self, state, *, baseline, scale, sigma):
        # the statistic at the impact sample changes iff the excited increment differs from
        # the quiet one after clamping: reaching above mean + k guarantees S+ moves
        need = (state["mean"] + state["k"]) - baseline
        return {"amplitude": max(need, 0.0) + 3.0 * sigma + 0.05 * scale,
                "reason": f"reach mean + k from the baseline ({need:.6g}) plus margin"}

    def transform(self, x, state):
        s_pos, s_neg = float(state["s_pos"]), float(state["s_neg"])
        mean, k = float(state["mean"]), float(state["k"])
        v = _as_array(x["values"])
        available = ~np.isnan(v)
        values = np.zeros(v.size)
        for i in np.flatnonzero(available):
            d = v[i] - mean
            s_pos = max(0.0, s_pos + d - k)
            s_neg = max(0.0, s_neg - d - k)
            values[i] = s_pos if s_pos >= s_neg else s_neg
        self._sums = {"s_pos": s_pos, "s_neg": s_neg, "mean": mean, "k": k, "h": state["h"]}
        return self._pack(x, values, available)

    def checkpoint(self):
        return json.dumps(self._sums)

    def resume(self, blob):
        return json.loads(blob)


class CusumLookahead(CusumCausal):
    """Twin: the statistic at t also sees the next `ahead` samples."""

    KIND = "cusum_lookahead"
    CHUNK_RESTART = "IDEMPOTENT"
    DEFAULT_PARAMS = {"k_sd": 0.5, "h_sd": 5.0, "ahead": 4}

    def twin(self):
        return _na_twin("it is itself a twin")

    def reach_right(self, n):
        return int(self.params["ahead"])

    def describe(self):
        return dict(super().describe(), non_causal_control_of="cusum_causal")

    def transform(self, x, state):
        # One causal pass of S+ over the whole series, then every output i reports the
        # statistic as it stands `ahead` samples LATER: the leak is the shift, not a
        # recomputation, and the twin costs the same as the operator it shadows.
        v = _as_array(x["values"])
        ahead = int(self.params["ahead"])
        mean, k = float(state["mean"]), float(state["k"])
        n = v.size
        finite = ~np.isnan(v)
        running = np.zeros(n)
        s_pos = 0.0
        for i in range(n):
            if finite[i]:
                s_pos = max(0.0, s_pos + (v[i] - mean) - k)
            running[i] = s_pos
        values = np.zeros(n)
        available = np.zeros(n, dtype=bool)
        if n > ahead:
            values[:n - ahead] = running[ahead:]
            windows, ok = _trailing_windows(v, ahead + 1)
            available[:n - ahead] = ok
        return self._pack(x, values, available)


class MadExtremesTrailing(Operator):
    """|x[t] - median(last w)| / MAD_train: an extreme score with a trailing median and a
    train-frozen scale."""

    KIND = "mad_extremes_trailing"
    GROUP = "detectors"
    FIT_SCOPE = "TRAIN_PREFIX_ONLY"
    DEFAULT_PARAMS = {"w": 16}

    def lookback(self):
        return int(self.params["w"]) - 1

    def bytes_state(self):
        return 16

    def probe(self):
        return {"kind": "impulse", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def twin(self):
        return {"kind": "mad_extremes_centred"}

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["w"]),
                "derivation": "trailing window of w samples", "boundary_mode": None}

    def fit(self, train):
        finite = np.asarray([v for v in train["values"] if _finite(v)], dtype=np.float64)
        if finite.size < 2:
            raise contract.SpecRefusal("the train prefix has fewer than two finite samples")
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med))) or 1e-12
        return {"median": med, "mad": mad}

    def _scores(self, state):
        mad = float(state["mad"])

        def fn(windows):
            return np.abs(windows[:, -1] - np.median(windows, axis=1)) / mad
        return fn

    def transform(self, x, state):
        values, available = _trailing_apply(_as_array(x["values"]), int(self.params["w"]),
                                            self._scores(state))
        return self._pack(x, values, available)


class MadExtremesCentred(MadExtremesTrailing):
    KIND = "mad_extremes_centred"

    def lookback(self):
        return int(self.params["w"]) // 2

    def warm_up(self):
        return int(self.params["w"]) // 2

    def twin(self):
        return _na_twin("it is itself a twin")

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["w"]) // 2 + 1,
                "derivation": "centred window; the future half is the leak", "boundary_mode": None}

    def reach_right(self, n):
        w = int(self.params["w"])
        return w - w // 2 - 1

    def describe(self):
        return dict(super().describe(), non_causal_control_of="mad_extremes_trailing")

    def _scores(self, state):
        mad = float(state["mad"])
        half = int(self.params["w"]) // 2

        def fn(windows):
            return np.abs(windows[:, half] - np.median(windows, axis=1)) / mad
        return fn

    def transform(self, x, state):
        values, available = _centred_apply(_as_array(x["values"]), int(self.params["w"]),
                                           self._scores(state))
        return self._pack(x, values, available)


class VarianceRegimeTrailing(Operator):
    """Variance of the last w samples."""

    KIND = "variance_regime_trailing"
    GROUP = "detectors"
    DEFAULT_PARAMS = {"w": 32}

    def lookback(self):
        return int(self.params["w"]) - 1

    def probe(self):
        return {"kind": "variance_shift", "expected_onset_samples": 0, "scale": contract.PROBE_SCALE}

    def probe_resolution(self, state, *, baseline, scale, sigma):
        return {"amplitude": 8.0, "gain": True,
                "reason": "a variance shift is a gain on the quiet deviations; 8x moves the "
                          "trailing variance at the impact sample"}

    def twin(self):
        return {"kind": "variance_regime_centred"}

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["w"]),
                "derivation": "trailing window of w samples", "boundary_mode": None}

    def transform(self, x, state):
        values, available = _trailing_apply(_as_array(x["values"]), int(self.params["w"]),
                                            lambda windows: windows.var(axis=1))
        return self._pack(x, values, available)


class VarianceRegimeCentred(VarianceRegimeTrailing):
    KIND = "variance_regime_centred"

    def lookback(self):
        return int(self.params["w"]) // 2

    def warm_up(self):
        return int(self.params["w"]) // 2

    def twin(self):
        return _na_twin("it is itself a twin")

    def support(self):
        return {"kind": "FINITE", "samples": int(self.params["w"]) // 2 + 1,
                "derivation": "centred window; the future half is the leak", "boundary_mode": None}

    def reach_right(self, n):
        w = int(self.params["w"])
        return w - w // 2 - 1

    def describe(self):
        return dict(super().describe(), non_causal_control_of="variance_regime_trailing")

    def transform(self, x, state):
        values, available = _centred_apply(_as_array(x["values"]), int(self.params["w"]),
                                           lambda windows: windows.var(axis=1))
        return self._pack(x, values, available)


# --- registry ----------------------------------------------------------------------------------

OPERATORS = {cls.KIND: cls for cls in (
    UniformDecileQuantizer, SaxPaaTrailing, DeltaRunLength, StftTrailing, WaveletTrailing,
    ButterworthCausal, CusumCausal, MadExtremesTrailing, VarianceRegimeTrailing)}
TWINS = {cls.KIND: cls for cls in (
    SaxPaaCentred, StftCentred, WaveletCentred, ButterworthFiltfilt, CusumLookahead,
    MadExtremesCentred, VarianceRegimeCentred)}


def build(kind: str, **params) -> Operator:
    if kind in OPERATORS:
        return OPERATORS[kind](**params)
    if kind in TWINS:
        return TWINS[kind](**params)
    raise KeyError(kind)


def twin_of(operator: Operator):
    """The declared twin instance, with the operator's own parameters, or None."""
    declared = operator.twin()
    if declared.get("not_applicable"):
        return None
    cls = TWINS[declared["kind"]]
    params = {k: v for k, v in operator.params.items() if k in cls.DEFAULT_PARAMS}
    return cls(**params)


def bank() -> list:
    """The nine operators with their default parameters, in the order of the design."""
    return [OPERATORS[kind]() for kind in OPERATORS]


def code_sha256() -> str:
    import hashlib

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = ["Operator", "OPERATORS", "TWINS", "build", "twin_of", "bank", "wavelet_support",
           "library_versions", "code_sha256"]
