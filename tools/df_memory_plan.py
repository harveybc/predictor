#!/usr/bin/env python3
"""C147 (order 2026-09-13): memory planner for the D0-D2 profile modules, from metadata only.

Before any array of a cell is allocated, the planner estimates the peak
resident memory of the profiling process during that cell (dataset x variable
x partition x metric group) and decides:

* RUN_EXACT                - the exact computation fits the task budget;
* RUN_BOUNDED              - the exact one does not fit (or the declared
                             statistical policy asks for it) and a predeclared
                             bounded variant fits: the C148 unit-root block, or
                             a contiguous suffix of train rows whose length is
                             the largest rung of BOUNDED_ROW_LADDER that fits;
* NOT_RUN_RESOURCE_BOUND   - nothing declared fits: the library is never
                             invoked and the module emits NOT_RUN rows.

Memory is never probed by exhaustion. Every estimate is

    estimated_peak_bytes = BASE_PROCESS_BYTES + SERIALIZATION_BYTES
                           + held_bytes(module, context) + ceil(FACTOR[group] * derived_bytes(group, sizes))

where `derived_bytes` is written from reading the module code (every float64
copy, boolean mask, sort copy, sliding-window copy, FFT buffer at its padded
length in complex128, OLS design, surrogate stack, bootstrap replicate and
Python object list the code creates, counted at its simultaneous maximum) and
FACTOR is a calibrated multiplier >= 1. The calibration
(tools/df_memory_calibrate.py, fixture tests/fixtures/df_memory_calibration.v1.json)
runs every group on small synthetic inputs in memory-capped children and
records the getrusage maxrss delta; FACTOR is set to at least 1.15 times the
largest measured/derived ratio, so every calibration case is under its estimate
with a margin. The ADF factor is declared (5.5) above the C146 PRE measurement
(5.04) and confirmed by the calibration.

The module is import-light (no numpy) and pure: the same metadata and budget
always give the same rows. Module constants that change memory (ACF lag,
Welch nperseg, permutation orders, pair caps, surrogate and bootstrap counts,
FFT padding) are mirrored here and checked against the modules by a test.

Rows: df_fact_resource_estimate (RESOURCE_ESTIMATE_KEYS). A row is written for
every gate call, so the estimate, its formula and its parameters exist on disk
before the computation they license.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DECISIONS = ("RUN_EXACT", "RUN_BOUNDED", "NOT_RUN_RESOURCE_BOUND")
MODULES = ("df_profile_univariate", "df_profile_information", "df_profile_multivariate", "df_sampling",
           "df_profile_run")
RESOURCE_ESTIMATE_KEYS = ("run_id", "bank", "dataset_id", "variable_id", "partition", "module", "metric", "estimator",
                          "estimated_peak_bytes", "formula", "params", "budget_bytes", "decision", "code_sha256")
MiB = 1 << 20
GiB = 1 << 30

# --------------------------------------------------- mirrored module constants
UNIT_ROOT_EXACT_MAX_N = 200_000          # df_profile_univariate
ADF_PEAK_FACTOR = 5.5                     # declared >= 5.04 measured in the C146 PRE
PE_ORDERS = (3, 4, 5)                     # df_profile_information
SPEC_WINDOW, SPEC_HOP, WELCH_NPERSEG_INFO = 256, 128, 64
MAX_PAIRS, MAX_VARIABLES_MATRIX = 1225, 400   # df_profile_multivariate
MAX_LAG, COH_NPERSEG, COH_SURROGATES, MI_SHUFFLES, CLUSTER_BOOTSTRAPS = 20, 256, 39, 20, 30
WELCH_NPERSEG_SAMPLING = 256              # df_sampling
LZMA_PRESET6_ENCODER_BYTES = 100 * MiB    # liblzma preset 6 encoder, about 94 MiB
ZLIB_LEVEL9_BYTES = 1 * MiB

# ---------------------------------------------------------- calibrated constants
BASE_PROCESS_BYTES = 420 * MiB            # interpreter + numpy/scipy/statsmodels/pyarrow/pandas, warmed
SERIALIZATION_BYTES = 16 * MiB            # rows of one cell as dicts + JSON line + write buffer
PYOBJ_BYTES_PER_RUN = 160                 # two int lists and a list of (start, end) tuples per run
FACTOR = {
    "df_profile_run.reader_column": 1.5,   # calibration: text labels needed 1.30 with margin
    "df_profile_univariate.timestamps_duplicates": 2.0,   # calibration: np.unique on int64 needs ~1.69x derived
    "df_profile_univariate.train_reference": 1.3,
    "df_profile_univariate.counts": 1.3,
    "df_profile_univariate.quantiles": 1.3,
    "df_profile_univariate.acf": 1.3,
    "df_profile_univariate.unit_root_adf": 1.1,     # on top of ADF_PEAK_FACTOR 5.5: calibration needed 1.04 with margin
    "df_profile_univariate.unit_root_kpss": 1.3,
    "df_profile_univariate.shift_ks_psi": 1.3,
    "df_profile_univariate.robust_z": 1.3,
    "df_profile_information.entropy_redundancy": 1.3,
    "df_profile_information.permutation_entropy": 1.3,
    "df_profile_information.spectral_entropy": 1.3,
    "df_profile_information.compression_raw_float64": 1.3,
    "df_profile_information.compression_symbols": 1.3,
    "df_profile_information.effective_rank": 1.3,
    "df_profile_multivariate.pair_block": 1.3,
    "df_profile_multivariate.pair_correlations": 1.3,
    "df_profile_multivariate.pair_mutual_information": 1.3,
    "df_profile_multivariate.pair_lead_lag": 1.3,
    "df_profile_multivariate.pair_coherence": 1.3,
    "df_profile_multivariate.matrix_diagnostics": 1.3,
    "df_profile_multivariate.cluster_bootstrap": 1.4,   # calibration needed 1.17 with margin
    "df_sampling.timestamps": 1.3,
    "df_sampling.spectral_decimation": 1.3,
    "df_sampling.aliasing_control": 1.3,
}
# Groups with a derived formula but no calibration case: the profile runner never invokes them
# (it passes no aliasing controls); only direct module callers with a control source do.
UNCALIBRATED_GROUPS = ("df_sampling.aliasing_control",)
# Composite groups are bounded by calibrated groups: pair_block admits the pair-block window only when
# the block plus the calibrated pair_correlations formula plus the reader fit; every pair group is then
# gated again with its own calibrated formula.
COMPOSITE_GROUPS = {"df_profile_multivariate.pair_block": ("df_profile_multivariate.pair_correlations",
                                                           "df_profile_run.reader_column")}
BOUNDED_ROW_LADDER =tuple(1 << k for k in range(23, 15, -1))   # 8,388,608 ... 65,536 rows
LADDER_GROUPS = {"df_profile_information.effective_rank": "rows", "df_profile_multivariate.pair_block": "rows",
                 "df_profile_multivariate.matrix_diagnostics": "rows", "df_profile_multivariate.pair_coherence": "L"}
DATASET_LEVEL_GROUPS = {"df_profile_run.reader_column", "df_profile_information.effective_rank",
                        "df_profile_multivariate.pair_block", "df_profile_multivariate.matrix_diagnostics",
                        "df_profile_multivariate.cluster_bootstrap"}


class PlanRefusal(ValueError):
    pass


def schwert_lag(m: int) -> int:
    lag = int(math.floor(12 * (m / 100.0) ** 0.25))
    return max(0, min(lag, m // 2 - 2))


def nfft_for(n: int) -> int:
    return 1 << int(math.ceil(math.log2(2 * n))) if n > 0 else 1


def _family(group: str) -> str:
    if group.startswith("permutation_entropy_order_"):
        return "permutation_entropy"
    if group.startswith("compression_raw_float64_"):
        return "compression_raw_float64"
    if group.startswith("compression_symbols_"):
        return "compression_symbols"
    return group


# ------------------------------------------------------------- derived formulas
def derived(module: str, group: str, s: dict, ctx: dict):
    """-> (bytes, formula) for one group at sizes `s` under dataset context `ctx`.
    Bytes are float64 = 8, int64 = 8, bool = 1, complex128 = 16 per element."""
    fam = _family(group)
    nt = int(ctx.get("n_train", 0))
    n = int(s.get("n", 0))
    if module == "df_profile_run" and fam == "reader_column":
        rg, cols = int(s["rg_rows"]), int(s.get("columns_held", 1))
        text = int(s.get("text_timestamp", 0))
        b = 8 * int(s["T"]) * cols + 32 * rg + 160 * rg * text
        return b, ("8*T*columns_held (preallocated float64/int64 outputs) + 32*rg_rows (arrow decode buffer, "
                   "chunk to_numpy, cast) + 160*rg_rows*text_timestamp (python str labels + parsed datetimes)")
    if module == "df_profile_univariate":
        if fam == "timestamps_duplicates":
            return 17 * n, "8*n (np.unique sorted copy) + 8*n (flag/diff) + 1*n (mask)"
        if fam == "train_reference":
            return 49 * n, ("1*n (isfinite) + 8*n (finite copy) + 8*n (np.sort copy) + 8*n (np.quantile copy) "
                            "+ 16*n (tf - median, abs) + 8*n (np.median copy)")
        if fam == "counts":
            return 8 * nt + 15 * n, "8*n_train (held sorted train reference) + 8*n (finite copy xf) + 7*n (masks)"
        if fam == "quantiles":
            return 8 * nt + 48 * n, ("8*n_train (held tf) + 8*n (xf) + 8*n (np.sort copy) + 8*n (np.diff) "
                                     "+ 16*n (sx - q50, abs) + 8*n (np.median copy)")
        if fam == "acf":
            nf = nfft_for(n)
            return 8 * nt + 57 * n + 64 * nf, (
                "8*n_train (tf) + 8*n (xf) + 49*n (x[m], centred, squared, np.where, mask as float, masks) "
                "+ 64*nfft (rfft padded input 8*nfft, F/conj/product complex128 3*16*(nfft/2+1), "
                "irfft complex work 16*nfft and output 8*nfft); nfft = 2^ceil(log2(2n))")
        if fam == "unit_root_adf":
            m, lag = int(s["n_run"]), int(s["lag"])
            return int(16 * nt + ADF_PEAK_FACTOR * m * (lag + 2) * 8), (
                "16*n_train (held tf and xf upper bound) + ADF_PEAK_FACTOR * n_run * (lag + 2) * 8 "
                "(statsmodels OLS design and lagmat copies; factor declared >= 5.04 measured)")
        if fam == "unit_root_kpss":
            m = int(s["n_run"])
            return 16 * nt + 12 * 8 * m, ("16*n_train (tf, xf) + 96*n_run (asarray copy, OLS residuals, cumsum, "
                                          "lagged products and their temporaries)")
        if fam == "shift_ks_psi":
            n1, n2 = int(s["n_train"]), n
            return 80 * (n1 + n2), ("8*(n1+n2) (held tf, xf) + 64*(n1+n2) (ks_2samp sorted copies, concatenate, "
                                    "two searchsorted int64, cdf floats, differences) + 8*(n1+n2) (PSI searchsorted)")
        if fam == "robust_z":
            return 16 * nt + 25 * n, "16*n_train (tf, xf upper bound) + 24*n (xf - median, abs, divide) + 1*n (mask)"
    if module == "df_profile_information":
        if fam == "entropy_redundancy":
            return 17 * nt + 55 * n, ("17*n_train (frozen edges: mask, finite copy, quantile copy) + 18*n (quantize: "
                                      "mask, symbols, finite copy) + 8*n (searchsorted int64) + 29*n (pair masks, "
                                      "int64 casts a*k+b)")
        if fam == "permutation_entropy":
            mo = int(s["order"])
            return 17 * n * mo + 11 * n, ("1*n*m (isfinite of windows) + 8*n*m (finite windows copy) + 8*n*m "
                                          "(argsort int64) + 8*n (codes int64) + 3*n (mask, symbols)")
        if fam == "spectral_entropy":
            nw = max(0, (n - SPEC_WINDOW) // SPEC_HOP + 1)
            return 26000 * nw + n, ("per trailing window of 256 (nw = (n-256)//128+1): 257 (mask) + 2048 (finite "
                                    "copy) + 7168 (Welch detrended and windowed segments 7x64 float64) + 11088 "
                                    "(rfft complex128, conj, product 7x33) + 1848 (real) + ~3600 (normalisation "
                                    "temporaries) -> 26000*nw, + 1*n symbols")
        if fam == "compression_raw_float64":
            const = LZMA_PRESET6_ENCODER_BYTES if group.endswith("lzma6") else ZLIB_LEVEL9_BYTES
            return 41 * n + const, ("8*n (contiguous copy) + 8*n (permutation int64) + 8*n (shuffled copy) + 8*n "
                                    "(tobytes) + 8*n (compressed output bound) + 1*n (symbols) + encoder state")
        if fam == "compression_symbols":
            const = LZMA_PRESET6_ENCODER_BYTES if group.endswith("lzma6") else ZLIB_LEVEL9_BYTES
            return 14 * n + const, ("1*n (symbols) + 8*n (permutation int64) + 1*n (shuffled) + 1*n (tobytes) "
                                    "+ 2*n (compressed output bound) + 1*n (ordered bytes) + encoder state")
        if fam == "effective_rank":
            R, V = int(s["rows"]), int(s["V"])
            rg = int(ctx.get("rg_rows", ctx.get("T", R)))
            return 57 * R * V + 16 * R + 8 * int(ctx.get("T", R)) + 32 * rg, (
                "8*R*V (train block) + 1*R*V (mask) + 8*R*V (complete-row copy) + 16*R*V (std temporaries) + 8*R*V "
                "(non-constant columns copy) + 16*R*V (z-score) + 8*R*V (LAPACK copy) + 16*R + reader column 8*T "
                "+ 32*rg_rows")
    if module == "df_profile_multivariate":
        k = int(ctx.get("k_pairs", 2))
        if fam == "pair_block":
            R = int(s["rows"])
            rg = int(ctx.get("rg_rows", ctx.get("T", R)))
            return 8 * R * k + 140 * R + 8 * int(ctx.get("T", R)) + 32 * rg, (
                "8*rows*k (pair block) + 140*rows (cheapest pair group) + reader 8*T + 32*rg_rows")
        if fam == "pair_correlations":
            return 8 * n * k + 140 * n, ("8*n*k (held pair block) + 3*n (masks) + 16*n (common copies) + max(spearman "
                                         "rankdata 2*49*n with results, biweight 105*n + weighted results 16*n)")
        if fam == "pair_mutual_information":
            return 8 * n * k + 88 * n, ("8*n*k (block) + 16*n (common copies) + 16*n (symbols int64) + 24*n "
                                        "(quantile copy, searchsorted, cast) + 32*n (per shuffle: permutation, "
                                        "shuffled symbols, joint code)")
        if fam == "pair_lead_lag":
            return 8 * n * k + 35 * n, "8*n*k (block) + 3*n (lag masks) + 16*n (lagged common copies) + 16*n (centred)"
        if fam == "pair_coherence":
            L = int(s["L"])
            return 8 * int(ctx.get("pair_rows", L)) * k + 4800 * L, (
                "8*pair_rows*k (block) + 80*L (observed coherence segments) + 39 surrogates: 156*L phases, "
                "3*312*L complex128 exp/product/irfft input, 312*L output, + 39-row coherence: per side 624*L "
                "detrended, 624*L windowed, 628*L complex128 spectra, 628*L cross product -> 4800*L")
        if fam == "matrix_diagnostics":
            R, V = int(s["rows"]), int(s["V"])
            rg = int(ctx.get("rg_rows", ctx.get("T", R)))
            return 42 * R * V + 160 * V * V + 8 * int(ctx.get("T", R)) + 32 * rg, (
                "8*R*V (matrix block) + 34*R*V (pairwise Pearson: mask float, where, square, matmul copies, bools) "
                "+ 160*V^2 (moment matrices, correlation, eigenvectors) + reader 8*T + 32*rg_rows")
        if fam == "cluster_bootstrap":
            R, V = int(s["rows"]), int(s["V"])
            return 50 * R * V + 24 * R + 200 * V * V, (
                "8*R*V (block) + 8*R*V (bootstrap resample copy) + 34*R*V (pairwise Pearson on it) + 24*R "
                "(block indices int64 and ravel) + 200*V^2 (two correlation matrices, linkage)")
    if module == "df_sampling":
        T = int(ctx.get("T", n))
        has_ts = int(bool(ctx.get("has_ts")))
        if fam == "timestamps":
            return 8 * T * has_ts + 64 * n + PYOBJ_BYTES_PER_RUN * n, (
                "8*T (held timestamps) + 64*n (diff int64, float cast, scaled, median/quantile copies, jitter "
                "temporaries, masks) + 160*n (runs() python lists at one run per sample, worst case)")
        if fam == "spectral_decimation":
            return 8 * T * has_ts + 330 * n + PYOBJ_BYTES_PER_RUN * n, (
                "8*T (held timestamps) + 8*n (partition column) + 25*n (regularity: diff, cast, abs, mask) + 160*n "
                "(runs() python lists, worst case) + 80*n (Welch on the longest run: detrended, windowed, "
                "complex128 spectra) + 60*n (causal FIR decimation output and Welch of both halves) "
                "+ 157*n (finite mask, segment list, accumulators; margin)")
        if fam == "aliasing_control":
            f = int(s["factor"])
            return 330 * n * f + PYOBJ_BYTES_PER_RUN * n, "330*n*factor (Welch and decimation of the control source) + 160*n (runs)"
    raise PlanRefusal(f"no memory formula for {module}.{group}")


def held_bytes(module: str, group: str, ctx: dict) -> tuple[int, str]:
    """Arrays the runner holds while the group runs, outside the group's own formula."""
    if f"{module}.{_family(group)}" in DATASET_LEVEL_GROUPS or module == "df_profile_multivariate":
        return 0, "0 (dataset-level group: its block and reader are in its own formula)"
    T = int(ctx.get("T", 0))
    has_ts = int(bool(ctx.get("has_ts")))
    if module == "df_profile_univariate":
        return 8 * T + 8 * T * has_ts, "8*T (variable column) + 8*T*has_ts (timestamps)"
    if module == "df_profile_information":
        return 8 * T, "8*T (variable column)"
    if module == "df_sampling":
        return 8 * T, "8*T (partition column upper bound)"
    return 0, "0"


def estimate(module: str, group: str, sizes: dict, ctx: dict) -> tuple[int, str, dict]:
    key = f"{module}.{_family(group)}"
    if key not in FACTOR:
        raise PlanRefusal(f"no calibrated factor for {key}")
    d, formula = derived(module, group, sizes, ctx)
    h, hform = held_bytes(module, group, ctx)
    total = BASE_PROCESS_BYTES + SERIALIZATION_BYTES + h + int(math.ceil(FACTOR[key] * d))
    text = (f"BASE_PROCESS_BYTES + SERIALIZATION_BYTES + held[{hform}] + ceil(FACTOR[{key}] * derived), "
            f"derived = {formula}")
    params = {"sizes": dict(sizes), "context": dict(ctx), "factor": FACTOR[key], "derived_bytes": int(d),
              "held_bytes": int(h), "base_process_bytes": BASE_PROCESS_BYTES,
              "serialization_bytes": SERIALIZATION_BYTES}
    return total, text, params


def constants_sha256() -> str:
    doc = {"FACTOR": FACTOR, "BASE_PROCESS_BYTES": BASE_PROCESS_BYTES, "SERIALIZATION_BYTES": SERIALIZATION_BYTES,
           "ADF_PEAK_FACTOR": ADF_PEAK_FACTOR, "PYOBJ_BYTES_PER_RUN": PYOBJ_BYTES_PER_RUN,
           "BOUNDED_ROW_LADDER": list(BOUNDED_ROW_LADDER), "UNIT_ROOT_EXACT_MAX_N": UNIT_ROOT_EXACT_MAX_N}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------- planner
class Planner:
    """Gate factory for one dataset task. `sink(row)` receives every
    df_fact_resource_estimate row before the gate returns."""

    def __init__(self, *, run_id: str, bank: str, dataset_id: str, budget_bytes: int, code_sha256: str,
                 context: dict, sink=None, stage: str = "IN_TASK_BEFORE_ALLOCATION"):
        if type(budget_bytes) is not int or budget_bytes <= 0:
            raise PlanRefusal("budget_bytes must be a positive integer")
        self.run_id, self.bank, self.dataset_id = run_id, bank, dataset_id
        self.budget, self.code_sha256, self.ctx, self.stage = budget_bytes, code_sha256, dict(context), stage
        self.sink = sink or (lambda row: None)
        self.peak_admitted = 0
        self.decisions = {d: 0 for d in DECISIONS}

    def for_module(self, module: str):
        return lambda group, key=None, partition=None, **sizes: self.gate(module, group, key, partition, **sizes)

    def _row(self, module, group, key, partition, estimator, est_bytes, formula, params, decision):
        if isinstance(key, (list, tuple)):
            vid = "PAIR:" + "|".join(str(k) for k in key)
        else:
            vid = key
        row = {"run_id": self.run_id, "bank": self.bank, "dataset_id": self.dataset_id, "variable_id": vid,
               "partition": partition, "module": module, "metric": group, "estimator": estimator,
               "estimated_peak_bytes": int(est_bytes), "formula": formula,
               "params": dict(params, stage=self.stage, constants_sha256=constants_sha256()),
               "budget_bytes": self.budget, "decision": decision, "code_sha256": self.code_sha256}
        problems = validate_resource_row(row)
        if problems:
            raise PlanRefusal("; ".join(problems))
        self.decisions[decision] += 1
        if decision != "NOT_RUN_RESOURCE_BOUND":
            self.peak_admitted = max(self.peak_admitted, int(est_bytes))
        self.sink(row)
        return row

    def gate(self, module, group, key=None, partition=None, **sizes):
        fam_key = f"{module}.{_family(group)}"
        variant = sizes.pop("variant", "EXACT")
        est_bytes, formula, params = estimate(module, group, sizes, self.ctx)
        if fam_key in LADDER_GROUPS:
            axis = LADDER_GROUPS[fam_key]
            total = int(sizes[axis])
            if est_bytes <= self.budget:
                self._row(module, group, key, partition, "EXACT", est_bytes, formula,
                          dict(params, window=[0, total]), "RUN_EXACT")
                return {"decision": "RUN_EXACT", "window": [0, total]}
            smallest = (est_bytes, formula, params)
            for rung in BOUNDED_ROW_LADDER:
                if rung >= total:
                    continue
                b_bytes, b_form, b_params = estimate(module, group, dict(sizes, **{axis: rung}), self.ctx)
                smallest = (b_bytes, b_form, b_params)
                if b_bytes <= self.budget:
                    window = [total - rung, total]
                    self._row(module, group, key, partition, "LADDER_SUFFIX", b_bytes, b_form,
                              dict(b_params, window=window, exact_estimated_peak_bytes=est_bytes, rung=rung,
                                   ladder=list(BOUNDED_ROW_LADDER)), "RUN_BOUNDED")
                    return {"decision": "RUN_BOUNDED", "window": window}
            self._row(module, group, key, partition, "LADDER_SUFFIX", smallest[0], smallest[1],
                      dict(smallest[2], exact_estimated_peak_bytes=est_bytes, ladder=list(BOUNDED_ROW_LADDER)),
                      "NOT_RUN_RESOURCE_BOUND")
            return {"decision": "NOT_RUN_RESOURCE_BOUND", "window": None}
        fits = est_bytes <= self.budget
        decision = ("RUN_EXACT" if variant == "EXACT" else "RUN_BOUNDED") if fits else "NOT_RUN_RESOURCE_BOUND"
        self._row(module, group, key, partition, variant, est_bytes, formula, dict(params, variant=variant), decision)
        return {"decision": decision, "window": None}


def validate_resource_row(row: dict) -> list[str]:
    p = []
    if not isinstance(row, dict) or tuple(sorted(row)) != tuple(sorted(RESOURCE_ESTIMATE_KEYS)):
        return [f"keys differ: expected {sorted(RESOURCE_ESTIMATE_KEYS)}"]
    for c in ("run_id", "bank", "dataset_id", "module", "metric", "estimator", "formula", "code_sha256"):
        if not isinstance(row[c], str) or not row[c]:
            p.append(f"{c}: expected non-empty text")
    for c in ("variable_id", "partition"):
        if row[c] is not None and not isinstance(row[c], str):
            p.append(f"{c}: expected text or null")
    for c in ("estimated_peak_bytes", "budget_bytes"):
        if type(row[c]) is not int or row[c] < 0:
            p.append(f"{c}: expected a non-negative integer")
    if not isinstance(row["params"], dict):
        p.append("params: expected an object")
    if row["decision"] not in DECISIONS:
        p.append(f"decision: {row['decision']!r} not in {list(DECISIONS)}")
    if row["decision"] != "NOT_RUN_RESOURCE_BOUND" and type(row["estimated_peak_bytes"]) is int \
            and type(row["budget_bytes"]) is int and row["estimated_peak_bytes"] > row["budget_bytes"]:
        p.append("an admitted cell cannot exceed its budget")
    if isinstance(row["code_sha256"], str) and not (len(row["code_sha256"]) == 64
                                                   and all(ch in "0123456789abcdef" for ch in row["code_sha256"])):
        p.append("code_sha256: expected a sha256 hex digest")
    try:
        json.dumps(row, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        p.append(f"not strict JSON: {exc}")
    return p


# ----------------------------------------------------------- metadata preflight
def pair_counts(V: int) -> tuple[int, int]:
    k = int(math.floor((1 + math.sqrt(1 + 8 * MAX_PAIRS)) / 2))
    while k * (k - 1) // 2 > MAX_PAIRS:
        k -= 1
    return min(k, V), min(MAX_VARIABLES_MATRIX, V)


def preflight(meta: dict, planner: Planner) -> dict:
    """Walk every cell of a dataset from metadata alone, with upper-bound sizes
    (a finite run is at most its partition, a coherence run at most its rows).
    meta: {"T", "variables": [variable_id...], "partitions": {name: [s, e]},
           "has_ts", "rg_rows", "text_timestamp"}.
    Returns {"planned_peak_bytes", "reader_decision", "decisions"}."""
    T, vids, parts = int(meta["T"]), list(meta["variables"]), meta["partitions"]
    order = [(p, int(parts[p][0]), int(parts[p][1])) for p in ("train", "calibration", "confirmation")]
    n_train = order[0][2] - order[0][1]
    reader = planner.gate("df_profile_run", "reader_column", None, None, T=T, rg_rows=int(meta["rg_rows"]),
                          columns_held=1 + int(bool(meta.get("has_ts"))),
                          text_timestamp=int(bool(meta.get("text_timestamp"))))
    uni, inf, mv, sm = (planner.for_module(m) for m in MODULES[:4])
    if meta.get("has_ts"):
        for p, s, e in order:
            uni("timestamps_duplicates", None, p, n=e - s)
            if e - s >= 2:
                sm("timestamps", None, p, n=e - s)
    for vid in vids:
        uni("train_reference", vid, "train", n=n_train)
        for p, s, e in order:
            n = e - s
            for g in ("counts", "quantiles", "acf", "robust_z"):
                uni(g, vid, p, n=n)
            m = min(n, UNIT_ROOT_EXACT_MAX_N)
            variant = "EXACT" if n <= UNIT_ROOT_EXACT_MAX_N else "BLOCK_APPROX"
            uni("unit_root_adf", vid, p, n_run=m, lag=schwert_lag(m), variant=variant)
            uni("unit_root_kpss", vid, p, n_run=m, variant=variant)
            if p != "train":
                uni("shift_ks_psi", vid, p, n=n, n_train=n_train)
            inf("entropy_redundancy", vid, p, n=n)
            for o in PE_ORDERS:
                inf(f"permutation_entropy_order_{o}", vid, p, n=n, order=o)
            inf("spectral_entropy", vid, p, n=n)
            for sname in ("symbols", "raw_float64"):
                for cname in ("zlib9", "lzma6"):
                    inf(f"compression_{sname}_{cname}", vid, p, n=n)
            sm("spectral_decimation", vid, p, n=n)
    V = len(vids)
    if V >= 2:
        inf("effective_rank", None, "train", rows=n_train, V=V)
        k, vm = pair_counts(V)
        d = mv("pair_block", None, "train", rows=n_train, k=k)
        if d["decision"] != "NOT_RUN_RESOURCE_BOUND":
            R = d["window"][1] - d["window"][0]
            planner.ctx["pair_rows"] = R
            key = [vids[0], vids[1]]
            for g in ("pair_correlations", "pair_mutual_information", "pair_lead_lag"):
                mv(g, key, "train", n=R)
            mv("pair_coherence", key, "train", L=R)
        dm = mv("matrix_diagnostics", None, "train", rows=n_train, V=vm)
        if dm["decision"] != "NOT_RUN_RESOURCE_BOUND":
            mv("cluster_bootstrap", None, "train", rows=dm["window"][1] - dm["window"][0], V=vm)
    return {"planned_peak_bytes": planner.peak_admitted, "reader_decision": reader["decision"],
            "decisions": dict(planner.decisions)}
