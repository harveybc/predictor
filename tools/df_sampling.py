#!/usr/bin/env python3
"""C133 (order 2026-09-12): sampling and aliasing diagnostics, without targets.

Per partition, never mixing partitions:

- nominal period (from the contract) against observed timestamp differences:
  median, distribution, jitter (MAD), non-positive differences, gaps, coverage,
  maximal regular segments;
- a Nyquist limit only when regular segments exist;
- per variable, the Welch energy fraction near Nyquist on regular finite runs;
- downsampling sensitivity: decimation by 2 with and without an anti-alias filter;
- an aliasing assessment that can only be claimed with a control: a
  higher-frequency source of the same quantity, or (for method validation) a
  synthetic sine of known frequency. With one sampling rate and no control the
  answer is ALIASING_NOT_IDENTIFIABLE_WITHOUT_CONTROL, whatever the spectrum
  looks like. A spectral peak alone never identifies aliasing.

Filters are causal (scipy.signal.decimate with zero_phase=False); nothing is
aligned with future samples.

Entry point: run_sampling(contract, X, timestamps=None, controls=None) -> list of rows.
Method check: synthetic_aliasing_control(...) -> list of rows.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import time
from pathlib import Path

import numpy as np

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARTITIONS = ("train", "calibration", "confirmation")
STATUSES = ("COMPLETED", "UNAVAILABLE", "INCONCLUSIVE", "FAILED", "NOT_RUN")

GAP_MULTIPLE = 1.5
REGULAR_TOLERANCE = 0.01
MIN_SPECTRAL_SEGMENT = 256
WELCH_NPERSEG = 256
NEAR_NYQUIST_FRACTION = 0.8
DECIMATE_FACTOR = 2
DECIMATE_FTYPE = "fir"
DECIMATE_TRANSIENT_DROP = 21
CONTROL_FOLDED_MIN = 0.05
CONTROL_EXCESS_MIN = 0.05
SYNTH_BAND_BINS = 2
SYNTH_FOLDED_MIN = 0.5
SYNTH_SEED = 20260915
NOT_IDENTIFIABLE = "ALIASING_NOT_IDENTIFIABLE_WITHOUT_CONTROL"


# ------------------------------------------------------------------ rows
def make_row(dataset_id, key, partition, metric, estimator, value, status="COMPLETED", reason="", cpu=0.0):
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if status == "COMPLETED":
        if value is None or not math.isfinite(float(value)):
            status, value, reason = "INCONCLUSIVE", None, "NON_FINITE_RESULT"
        else:
            value = float(value)
    else:
        value = None
        if not reason:
            raise ValueError("a row that is not COMPLETED needs a reason")
    (k, v), = key.items()
    return {"dataset_id": dataset_id, k: v, "partition": partition, "metric": metric,
            "estimator": estimator, "value": value, "status": status, "reason": reason,
            "code_sha256": CODE_SHA256, "cpu_seconds": round(float(cpu), 6)}


def est(name, params=None, assumptions=()):
    return {"name": name, "params": dict(params or {}), "assumptions": list(assumptions)}


def contract_layout(contract, X, timestamps):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a (T, V) array")
    T, V = X.shape
    b = contract["partitions"]["boundaries"]
    parts = [(p, int(b[p][0]), int(b[p][1])) for p in PARTITIONS]
    if parts[0][1] != 0 or parts[0][2] != parts[1][1] or parts[1][2] != parts[2][1] or parts[2][2] != T \
            or any(s >= e for _, s, e in parts):
        raise ValueError("partition boundaries must be contiguous [start, end) blocks covering X")
    vids = [v["variable_id"] for v in contract["variables"]]
    if len(vids) != V:
        raise ValueError(f"contract declares {len(vids)} variables, X has {V} columns")
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        if timestamps.shape != (T,) or timestamps.dtype != np.int64:
            raise ValueError("timestamps must be int64 nanoseconds of shape (T,)")
        if contract["time"]["timestamp_meaning"] == "SAMPLE_INDEX":
            raise ValueError("the contract says SAMPLE_INDEX; timestamps must be None")
    return X, parts, vids, timestamps


def blas_single_thread():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1, user_api="blas")
    except ImportError:  # pragma: no cover
        return contextlib.nullcontext()


# --------------------------------------------------------------- helpers
def runs(valid, regular_diff):
    """Maximal [start, end) runs: sample i continues the run of i-1 when both are
    valid and the difference between them is regular."""
    n = valid.size
    if n == 0:
        return []
    valid = np.asarray(valid, dtype=bool)
    cont = np.zeros(n + 1, dtype=bool)
    cont[1:n] = valid[1:] & valid[:-1] & np.asarray(regular_diff, dtype=bool)
    starts = np.flatnonzero(valid & ~cont[:n])
    ends = np.flatnonzero(valid & ~cont[1:]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def near_nyquist_fraction(segments, fs):
    """Energy above NEAR_NYQUIST_FRACTION * fs/2 over energy above DC, Welch
    spectra averaged over segments weighted by segment length."""
    from scipy.signal import welch
    acc, weight = None, 0
    for seg in segments:
        f, p = welch(seg, fs=fs, nperseg=WELCH_NPERSEG, detrend="constant")
        acc = p * seg.size if acc is None else acc + p * seg.size
        weight += seg.size
    if acc is None:
        return None
    band = f > 0
    tot = acc[band].sum()
    if tot <= 0:
        return None
    return float(acc[f > NEAR_NYQUIST_FRACTION * fs / 2].sum() / tot)


def decimation_sensitivity(x, fs):
    """(variance ratio naive/filtered, relative L1 spectral difference)."""
    from scipy.signal import decimate, welch
    naive = x[::DECIMATE_FACTOR]
    filt = decimate(x, DECIMATE_FACTOR, ftype=DECIMATE_FTYPE, zero_phase=False)
    k = min(naive.size, filt.size)
    naive, filt = naive[DECIMATE_TRANSIENT_DROP:k], filt[DECIMATE_TRANSIENT_DROP:k]
    vf = float(filt.var())
    if vf <= 0 or naive.size < WELCH_NPERSEG // 2:
        return None
    nps = min(WELCH_NPERSEG // 2, naive.size)
    _, pn = welch(naive, fs=fs / DECIMATE_FACTOR, nperseg=nps, detrend="constant")
    _, pf = welch(filt, fs=fs / DECIMATE_FACTOR, nperseg=nps, detrend="constant")
    return float(naive.var()) / vf, float(np.abs(pn - pf).sum() / pn.sum())


def folded_energy_against_source(x_low, x_high, factor, fs_low):
    """Control with a higher-frequency source of the same quantity.
    folded_fraction: share of the source's energy above the low-rate Nyquist;
    excess: share of the observed low-rate spectrum not present in a causal
    anti-aliased decimation of the source."""
    from scipy.signal import decimate, welch
    fs_high = fs_low * factor
    nps_h = min(WELCH_NPERSEG * factor, x_high.size)
    fh, ph = welch(x_high, fs=fs_high, nperseg=nps_h, detrend="constant")
    tot_h = ph[fh > 0].sum()
    folded = float(ph[fh > fs_low / 2].sum() / tot_h) if tot_h > 0 else None
    ref = decimate(x_high, factor, ftype=DECIMATE_FTYPE, zero_phase=False)
    k = min(ref.size, x_low.size)
    obs, ref = x_low[DECIMATE_TRANSIENT_DROP:k], ref[DECIMATE_TRANSIENT_DROP:k]
    nps = min(WELCH_NPERSEG, obs.size)
    f, po = welch(obs, fs=fs_low, nperseg=nps, detrend="constant")
    _, pr = welch(ref, fs=fs_low, nperseg=nps, detrend="constant")
    band = f > 0
    tot = po[band].sum()
    excess = float(np.maximum(po[band] - pr[band], 0).sum() / tot) if tot > 0 else None
    return folded, excess


# ------------------------------------------------------------ estimators
_TS = ["timestamps are int64 nanoseconds; differences within the partition only"]
E_NOM = est("nominal_period_from_contract", {"field": "time.frequency_nominal_seconds"}, [])
E_MED = est("median_timestamp_difference_seconds", {}, _TS)
E_DQ = est("timestamp_difference_quantile_seconds", {"levels": [0.01, 0.05, 0.25, 0.75, 0.95, 0.99],
                                                     "method": "linear_type7"}, _TS)
E_DMIN = est("minimum_timestamp_difference_seconds", {}, _TS)
E_DMAX = est("maximum_timestamp_difference_seconds", {}, _TS)
E_JIT = est("jitter_mad_of_differences_seconds", {"scale": 1.0}, _TS)
E_NONPOS = est("non_positive_difference_count", {}, _TS + ["duplicates and out-of-order timestamps"])
_PER = ["reference period is the contract's nominal period; when the contract has no numeric period the "
        "observed median difference is used and the row reason says so"]
E_GAPS = est("gap_count", {"gap_multiple_of_period": GAP_MULTIPLE}, _TS + _PER)
E_GAPMAX = est("longest_gap_seconds", {"gap_multiple_of_period": GAP_MULTIPLE}, _TS + _PER)
E_COV = est("coverage_vs_reference_grid", {"definition": "n / (span / period + 1)"}, _TS + _PER)
_REG_P = {"tolerance_fraction_of_period": REGULAR_TOLERANCE, "definition": "maximal runs of consecutive "
          "differences within tolerance of the period; a segment of k differences has k + 1 samples"}
E_SEGN = est("regular_segment_count", {**_REG_P, "min_samples": 2}, _TS + _PER)
E_SEGL = est("longest_regular_segment_samples", _REG_P, _TS + _PER)
E_SEGF = est("fraction_of_samples_in_regular_segments", {**_REG_P, "min_samples": 2}, _TS + _PER)
E_NYQ = est("nyquist_frequency_hz_regular_segments", {"definition": "1 / (2 * period)"},
            _PER + ["only defined when at least one regular segment exists; not valid across gaps"])
_IDX = ["SAMPLE_INDEX data are treated as regular with unit spacing; frequencies in cycles per sample"]
_SPEC_P = {"welch_nperseg": WELCH_NPERSEG, "window": "hann", "detrend": "constant",
           "min_segment_samples": MIN_SPECTRAL_SEGMENT, "near_nyquist_above_fraction_of_nyquist": NEAR_NYQUIST_FRACTION}
_SPEC_A = ["computed on maximal runs that are both regular in time and finite in value",
           "spectra of runs averaged weighted by run length", "the DC bin is excluded from the total",
           "high near-Nyquist energy is not evidence of aliasing"]
E_NN = est("near_nyquist_energy_fraction", _SPEC_P, _SPEC_A)
_DEC_P = {"factor": DECIMATE_FACTOR, "filtered": "scipy.signal.decimate ftype=fir zero_phase=False (causal)",
          "unfiltered": "x[::factor]", "transient_samples_dropped": DECIMATE_TRANSIENT_DROP,
          "welch_nperseg": WELCH_NPERSEG // 2}
_DEC_A = ["computed on the longest regular finite run",
          "a sensitivity descriptor: the difference mixes removed high-band energy with folded energy, "
          "so it is not an aliasing measure"]
E_DVAR = est("decimation_variance_ratio_unfiltered_over_filtered", _DEC_P, _DEC_A)
E_DSPEC = est("decimation_relative_l1_spectral_difference", {**_DEC_P,
              "definition": "sum |P_unfiltered - P_filtered| / sum P_unfiltered"}, _DEC_A)
_AL_P = {"rule": "aliasing is claimed only with a control; a single sampling rate without a control is "
                 "not identifiable", "control_kinds": ["HIGHER_FREQUENCY_SOURCE", "SYNTHETIC_KNOWN_FREQUENCY"]}
E_AL_NONE = est("aliasing_assessment", {**_AL_P, "control": None},
                ["a spectral peak or high near-Nyquist energy alone never identifies aliasing"])
E_AL_SRC = est("aliasing_assessment", {**_AL_P, "control": "HIGHER_FREQUENCY_SOURCE",
                                       "folded_fraction_min": CONTROL_FOLDED_MIN, "excess_min": CONTROL_EXCESS_MIN,
                                       "value": "excess share of the observed spectrum absent from the causal "
                                                "anti-aliased decimation of the source",
                                       "partition_mapping": "source rows [start * factor, end * factor)"},
               ["x_low[i] is assumed to correspond to x_high[i * factor]; an observed series that is an "
                "aggregate (e.g. a mean) of the source already includes a filter",
                "computed on the longest regular finite run of the observed series"])
E_AL_FOLD = est("control_source_energy_above_low_rate_nyquist", E_AL_SRC["params"], E_AL_SRC["assumptions"])


# ------------------------------------------------------------------- run
RESOURCE_REASON = "NOT_RUN_RESOURCE_BOUND"
_TS_METRICS = (("observed_median_period_seconds", E_MED),) + tuple(
    (f"diff_quantile_{q}", E_DQ) for q in E_DQ["params"]["levels"]) + (
    ("diff_min_seconds", E_DMIN), ("diff_max_seconds", E_DMAX), ("jitter_mad_seconds", E_JIT),
    ("non_positive_diff_count", E_NONPOS), ("gap_count", E_GAPS), ("longest_gap_seconds", E_GAPMAX),
    ("coverage_vs_grid", E_COV), ("regular_segment_count", E_SEGN), ("longest_regular_segment_samples", E_SEGL),
    ("regular_segment_sample_fraction", E_SEGF), ("nyquist_frequency", E_NYQ))


def admit_all(group, key=None, partition=None, **sizes):
    """The default gate: every group runs exactly (C147)."""
    return {"decision": "RUN_EXACT", "window": None}


def run_sampling(contract, X, timestamps=None, controls=None, gate=None):
    """controls: optional {variable_id: {"x_high": 1-D array of T * factor samples of
    the same quantity at factor times the rate, "factor": int}}."""
    with blas_single_thread():
        return _run_sampling(contract, X, timestamps, controls or {}, gate)


def _period(contract):
    p = contract["time"]["frequency_nominal_seconds"]
    return float(p) if type(p) in (int, float) and p > 0 else None


def _run_sampling(contract, X, timestamps, controls, gate=None):
    X, parts, vids, timestamps = contract_layout(contract, X, timestamps)
    rows = list(timestamp_rows(contract, parts, timestamps, gate))
    for pname, s, e in parts:
        reg, fs = partition_regularity(contract, s, e, timestamps)
        for j, vid in enumerate(vids):
            rows.extend(variable_partition_rows(contract, vid, X[s:e, j], pname, s, e, reg, fs, timestamps is None,
                                                controls.get(vid), X.shape[0], gate))
    return rows


def timestamp_rows(contract, parts, timestamps, gate=None):
    gate = gate or admit_all
    ds = contract["dataset_id"]
    meaning = contract["time"]["timestamp_meaning"]
    nominal = _period(contract)
    rows = []
    g = {"group_id": "dataset_timestamps"}

    for pname, s, e in parts:
        add = lambda metric, estimator, value, status="COMPLETED", reason="", cpu=0.0: rows.append(
            make_row(ds, g, pname, metric, estimator, value, status, reason, cpu))
        if nominal is None:
            add("nominal_period_seconds", E_NOM, None, "UNAVAILABLE", "CONTRACT_PERIOD_NOT_NUMERIC")
        else:
            add("nominal_period_seconds", E_NOM, nominal)

        t0 = time.process_time()
        if timestamps is not None and e - s >= 2 and \
                gate("timestamps", None, pname, n=e - s)["decision"] == "NOT_RUN_RESOURCE_BOUND":
            for metric, estimator in _TS_METRICS:
                add(metric, estimator, None, "NOT_RUN", RESOURCE_REASON)
            continue
        if timestamps is None:
            why = "SAMPLE_INDEX_HAS_NO_TIMESTAMPS" if meaning == "SAMPLE_INDEX" else "TIMESTAMPS_NOT_PROVIDED"
            for metric, estimator in (("observed_median_period_seconds", E_MED), ("diff_min_seconds", E_DMIN),
                                      ("diff_max_seconds", E_DMAX), ("jitter_mad_seconds", E_JIT),
                                      ("non_positive_diff_count", E_NONPOS), ("gap_count", E_GAPS),
                                      ("longest_gap_seconds", E_GAPMAX), ("coverage_vs_grid", E_COV)) + \
                    tuple((f"diff_quantile_{q}", E_DQ) for q in E_DQ["params"]["levels"]):
                add(metric, estimator, None, "UNAVAILABLE", why)
            if meaning == "SAMPLE_INDEX":
                c = time.process_time() - t0
                add("regular_segment_count", E_SEGN, 1, reason="SAMPLE_INDEX_REGULAR_BY_CONSTRUCTION", cpu=c)
                add("longest_regular_segment_samples", E_SEGL, e - s, reason="SAMPLE_INDEX_REGULAR_BY_CONSTRUCTION",
                    cpu=c)
                add("regular_segment_sample_fraction", E_SEGF, 1.0, reason="SAMPLE_INDEX_REGULAR_BY_CONSTRUCTION",
                    cpu=c)
                add("nyquist_frequency", E_NYQ, 0.5, reason="CYCLES_PER_SAMPLE", cpu=c)
            else:
                for metric, estimator in (("regular_segment_count", E_SEGN),
                                          ("longest_regular_segment_samples", E_SEGL),
                                          ("regular_segment_sample_fraction", E_SEGF), ("nyquist_frequency", E_NYQ)):
                    add(metric, estimator, None, "UNAVAILABLE", why)
            continue

        ts = timestamps[s:e]
        n = ts.size
        if n < 2:
            add("observed_median_period_seconds", E_MED, None, "INCONCLUSIVE", "FEWER_THAN_TWO_TIMESTAMPS")
            continue
        d = np.diff(ts).astype(np.float64) / 1e9
        med = float(np.median(d))
        ref, ref_reason = (nominal, "") if nominal is not None else (med, "NOMINAL_UNKNOWN_OBSERVED_MEDIAN_USED")
        c = time.process_time() - t0
        add("observed_median_period_seconds", E_MED, med, cpu=c)
        for q in E_DQ["params"]["levels"]:
            add(f"diff_quantile_{q}", E_DQ, np.quantile(d, q), cpu=c)
        add("diff_min_seconds", E_DMIN, d.min(), cpu=c)
        add("diff_max_seconds", E_DMAX, d.max(), cpu=c)
        add("jitter_mad_seconds", E_JIT, np.median(np.abs(d - med)), cpu=c)
        add("non_positive_diff_count", E_NONPOS, int((d <= 0).sum()), cpu=c)
        if ref is None or ref <= 0:
            for metric, estimator in (("gap_count", E_GAPS), ("longest_gap_seconds", E_GAPMAX),
                                      ("coverage_vs_grid", E_COV), ("regular_segment_count", E_SEGN),
                                      ("longest_regular_segment_samples", E_SEGL),
                                      ("regular_segment_sample_fraction", E_SEGF), ("nyquist_frequency", E_NYQ)):
                add(metric, estimator, None, "INCONCLUSIVE", "NO_POSITIVE_REFERENCE_PERIOD")
            continue
        gaps = d > GAP_MULTIPLE * ref
        add("gap_count", E_GAPS, int(gaps.sum()), reason=ref_reason, cpu=c)
        add("longest_gap_seconds", E_GAPMAX, float(d[gaps].max()) if gaps.any() else 0.0, reason=ref_reason, cpu=c)
        span = float(ts[-1] - ts[0]) / 1e9
        add("coverage_vs_grid", E_COV, n / (span / ref + 1) if span > 0 else 0.0, reason=ref_reason, cpu=c)
        reg = np.abs(d - ref) <= REGULAR_TOLERANCE * ref
        segs = [r for r in runs(np.ones(n, dtype=bool), reg) if r[1] - r[0] >= 2]
        c = time.process_time() - t0
        add("regular_segment_count", E_SEGN, len(segs), reason=ref_reason, cpu=c)
        add("longest_regular_segment_samples", E_SEGL, max((b - a for a, b in segs), default=0),
            reason=ref_reason, cpu=c)
        add("regular_segment_sample_fraction", E_SEGF, sum(b - a for a, b in segs) / n, reason=ref_reason, cpu=c)
        if segs:
            add("nyquist_frequency", E_NYQ, 1.0 / (2.0 * ref), reason=ref_reason or "HZ", cpu=c)
        else:
            add("nyquist_frequency", E_NYQ, None, "INCONCLUSIVE", "NO_REGULAR_SEGMENT", cpu=c)

    return rows


def partition_regularity(contract, s, e, timestamps):
    """(regular-difference mask, sampling frequency) of partition [s, e), or (None, None)."""
    meaning = contract["time"]["timestamp_meaning"]
    nominal = _period(contract)
    if timestamps is None and meaning != "SAMPLE_INDEX":
        return None, None
    if timestamps is None:
        return np.ones(e - s - 1, dtype=bool), 1.0
    d = np.diff(timestamps[s:e]).astype(np.float64) / 1e9
    ref = nominal if nominal is not None else (float(np.median(d)) if d.size else None)
    if ref is None or ref <= 0:
        return None, None
    return np.abs(d - ref) <= REGULAR_TOLERANCE * ref, 1.0 / ref


def variable_partition_rows(contract, vid, x, pname, s, e, reg, fs, no_timestamps, ctl, T, gate=None):
    """The rows of one variable in one partition; `x` is that partition's slice."""
    gate = gate or admit_all
    ds = contract["dataset_id"]
    rows = []
    if True:
        if True:
            key = {"variable_id": vid}
            add = lambda metric, estimator, value, status="COMPLETED", reason="", cpu=0.0: rows.append(
                make_row(ds, key, pname, metric, estimator, value, status, reason, cpu))
            if reg is None:
                why = "TIMESTAMPS_NOT_PROVIDED" if no_timestamps else "NO_POSITIVE_REFERENCE_PERIOD"
                for metric, estimator in (("near_nyquist_energy_fraction", E_NN),
                                          ("decimation_variance_ratio", E_DVAR),
                                          ("decimation_spectral_difference", E_DSPEC)):
                    add(metric, estimator, None, "UNAVAILABLE", why)
            elif gate("spectral_decimation", vid, pname, n=e - s)["decision"] == "NOT_RUN_RESOURCE_BOUND":
                for metric, estimator in (("near_nyquist_energy_fraction", E_NN),
                                          ("decimation_variance_ratio", E_DVAR),
                                          ("decimation_spectral_difference", E_DSPEC)):
                    add(metric, estimator, None, "NOT_RUN", RESOURCE_REASON)
            else:
                t0 = time.process_time()
                rr = [(a, b) for a, b in runs(np.isfinite(x), reg) if b - a >= MIN_SPECTRAL_SEGMENT]
                segs = [x[a:b] for a, b in rr]
                nn = near_nyquist_fraction(segs, fs) if segs else None
                c = time.process_time() - t0
                if not segs:
                    add("near_nyquist_energy_fraction", E_NN, None, "INCONCLUSIVE", "NO_REGULAR_FINITE_RUN", cpu=c)
                elif nn is None:
                    add("near_nyquist_energy_fraction", E_NN, None, "INCONCLUSIVE", "ZERO_POWER", cpu=c)
                else:
                    add("near_nyquist_energy_fraction", E_NN, nn, cpu=c)
                t0 = time.process_time()
                longest = max(segs, key=lambda z: z.size) if segs else None
                ds_res = decimation_sensitivity(longest, fs) if longest is not None else None
                c = time.process_time() - t0
                if ds_res is None:
                    why = "NO_REGULAR_FINITE_RUN" if longest is None else "ZERO_VARIANCE_OR_SHORT_RUN"
                    add("decimation_variance_ratio", E_DVAR, None, "INCONCLUSIVE", why, cpu=c)
                    add("decimation_spectral_difference", E_DSPEC, None, "INCONCLUSIVE", why, cpu=c)
                else:
                    add("decimation_variance_ratio", E_DVAR, ds_res[0], cpu=c)
                    add("decimation_spectral_difference", E_DSPEC, ds_res[1], cpu=c)

            # aliasing assessment
            if ctl is None:
                add("aliasing_assessment", E_AL_NONE, None, "INCONCLUSIVE", NOT_IDENTIFIABLE)
                return rows
            t0 = time.process_time()
            factor = int(ctl["factor"])
            xh = np.asarray(ctl["x_high"], dtype=float)
            if factor < 2 or xh.shape != (T * factor,):
                add("aliasing_assessment", E_AL_SRC, None, "FAILED", "CONTROL_SHAPE_OR_FACTOR_INVALID")
                return rows
            if reg is None:
                add("aliasing_assessment", E_AL_SRC, None, "INCONCLUSIVE", "NO_REGULAR_REFERENCE_FOR_CONTROL")
                return rows
            if gate("aliasing_control", vid, pname, n=e - s, factor=factor)["decision"] == "NOT_RUN_RESOURCE_BOUND":
                add("aliasing_assessment", E_AL_SRC, None, "NOT_RUN", RESOURCE_REASON)
                add("control_folded_energy_fraction", E_AL_FOLD, None, "NOT_RUN", RESOURCE_REASON)
                return rows
            xhp = xh[s * factor:e * factor]
            rr = [(a, b) for a, b in runs(np.isfinite(x), reg) if b - a >= MIN_SPECTRAL_SEGMENT]
            rr = [(a, b) for a, b in rr if np.all(np.isfinite(xhp[a * factor:b * factor]))]
            if not rr:
                add("aliasing_assessment", E_AL_SRC, None, "INCONCLUSIVE", "NO_REGULAR_FINITE_RUN_WITH_CONTROL")
                add("control_folded_energy_fraction", E_AL_FOLD, None, "INCONCLUSIVE",
                    "NO_REGULAR_FINITE_RUN_WITH_CONTROL")
                return rows
            a, b = max(rr, key=lambda r: r[1] - r[0])
            folded, excess = folded_energy_against_source(x[a:b], xhp[a * factor:b * factor], factor, fs)
            c = time.process_time() - t0
            if folded is None or excess is None:
                add("aliasing_assessment", E_AL_SRC, None, "INCONCLUSIVE", "ZERO_POWER", cpu=c)
                add("control_folded_energy_fraction", E_AL_FOLD, None, "INCONCLUSIVE", "ZERO_POWER", cpu=c)
                return rows
            claim = folded >= CONTROL_FOLDED_MIN and excess >= CONTROL_EXCESS_MIN
            add("aliasing_assessment", E_AL_SRC, excess,
                reason="ALIASING_CONSISTENT_WITH_CONTROL" if claim else "NO_FOLDED_ENERGY_DETECTED_WITH_CONTROL", cpu=c)
            add("control_folded_energy_fraction", E_AL_FOLD, folded, cpu=c)
    return rows


def synthetic_aliasing_control(dataset_id="synthetic_control", f0_fraction_of_fs=0.45, n=16384,
                               factor=DECIMATE_FACTOR, noise_std=0.01, seed=SYNTH_SEED):
    """Method validation with a sine of known frequency: decimating without a
    filter must put the energy at the predicted folded frequency; the causal
    anti-aliased decimation must not. Says nothing about any real variable."""
    from scipy.signal import decimate, welch
    t0 = time.process_time()
    rng = np.random.default_rng(seed)
    fs = 1.0
    t = np.arange(n)
    x = np.sin(2 * np.pi * f0_fraction_of_fs * fs * t) + noise_std * rng.standard_normal(n)
    fs_new = fs / factor
    f0 = f0_fraction_of_fs * fs
    f_alias = abs(f0 - round(f0 / fs_new) * fs_new)
    nps = WELCH_NPERSEG
    naive = x[::factor][DECIMATE_TRANSIENT_DROP:]
    filt = decimate(x, factor, ftype=DECIMATE_FTYPE, zero_phase=False)[DECIMATE_TRANSIENT_DROP:]
    f, pn = welch(naive, fs=fs_new, nperseg=nps, detrend="constant")
    _, pf = welch(filt, fs=fs_new, nperseg=nps, detrend="constant")
    k = int(np.argmin(np.abs(f - f_alias)))
    lo, hi = max(k - SYNTH_BAND_BINS, 1), k + SYNTH_BAND_BINS + 1
    frac_naive = float(pn[lo:hi].sum() / pn[1:].sum())
    ratio_filtered = float(pf[lo:hi].sum() / pn[lo:hi].sum())
    params = {"control": "SYNTHETIC_KNOWN_FREQUENCY", "f0_fraction_of_fs": f0_fraction_of_fs, "n": n,
              "factor": factor, "noise_std": noise_std, "seed": seed, "predicted_alias_fraction_of_new_fs":
              f_alias / fs_new, "band_bins_each_side": SYNTH_BAND_BINS, "welch_nperseg": nps,
              "folded_fraction_min": SYNTH_FOLDED_MIN}
    assume = ["method validation on a generated signal; it does not identify aliasing in any dataset variable",
              "filtered decimation is causal (zero_phase=False)"]
    c = time.process_time() - t0
    key = {"group_id": f"synthetic_sine_{f0_fraction_of_fs}fs_decimate_{factor}"}
    detected = frac_naive >= SYNTH_FOLDED_MIN and ratio_filtered < 0.01
    return [
        make_row(dataset_id, key, "synthetic_control", "folded_energy_fraction_unfiltered",
                 est("energy_fraction_at_predicted_alias_frequency", params, assume), frac_naive,
                 reason="FOLDED_ENERGY_AT_PREDICTED_FREQUENCY" if detected else "FOLDED_ENERGY_NOT_DETECTED", cpu=c),
        make_row(dataset_id, key, "synthetic_control", "filtered_over_unfiltered_alias_band_energy",
                 est("alias_band_energy_ratio_filtered_over_unfiltered", params, assume), ratio_filtered, cpu=c),
    ]
