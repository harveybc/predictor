#!/usr/bin/env python3
"""C156-C160 (order 2026-09-13): the executable causal battery of the D2
operator bank, and its OLAP rows.

For every causal operator use (spec x declared fit mode) and, individually,
every level of trailing_haar_threshold:

PREFIX_ALL_T              running full X against X[:t+1]: every t on small
                          series, stratified cuts (every 2^j-1, 2^j, 2^j+1,
                          window, warm-up and period boundary plus seeded
                          random cuts) on the large one;
BATCH_STEP_CHUNK_RESTART  batch against sample-by-sample step, against
                          fragmented streams of every small chunk size, and
                          against save/load restarts right before and after
                          every 2^j boundary and the warm-up boundary;
SUFFIX_ADVERSARIAL        X[t+1:] replaced by zeros, a large constant, noise
                          of another seed, the reversed series, NaN blocks, an
                          impulse at t+1, a step, a chirp and a regime change;
REFERENCE_EQUALITY        against tests/df_causal_reference.py, which sees only
                          X[:t+1];

on cases univariate, multivariate, isolated NaN, NaN blocks, gaps, constant,
1e-12 and 1e12 scales (length 100, so every length 2^j-1..2^j+1 <= 65 and every
warm-up is a prefix) and a large case (length 600). Equality is bitwise with
NaN == NaN, on output values, availability and reason (reason carries the
status).

NEGATIVE_CONTROL  one frozen control per forbidden class, each detected.
GUARD_MUTATION    every guard switched off in turn; its probe must stop refusing
                  for the expected reason.
SNAPSHOT_REFUSAL  what fit/transform/step must refuse.
FIT_MODE          the temporal fit modes, including the PRE seasonal case.

CLI: ``python tools/df_causal_battery.py --out DIR`` writes, write-once,
df_fact_causal_test.jsonl, df_fact_naming_isolation_decision.jsonl and
CAUSAL_BATTERY_SUMMARY.json.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str, directory: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OPS = _load("df_operators")
SNAP = _load("df_snapshot")
SYNC = _load("df_synthetic_contract")
REF = _load("df_causal_reference", ROOT / "tests")

SEED = 20260913
SMALL_N = 100
LARGE_N = 600
SMALL_CHUNKS = tuple(range(1, 9))
LARGE_CHUNKS = (5, 64)
SEVEN_CUTS = (0, 5, 19, 23, 60, 150, 238)
SUFFIXES = ("zeros", "large_constant", "other_seed_noise", "reversed", "nan_blocks", "impulse_at_t_plus_1",
            "step", "chirp", "regime_change")
BATTERY_GUARDS = {"exhaustive_cuts": True}
TEST_CLASSES = ("PREFIX_ALL_T", "BATCH_STEP_CHUNK_RESTART", "SUFFIX_ADVERSARIAL", "REFERENCE_EQUALITY",
                "NEGATIVE_CONTROL", "GUARD_MUTATION", "SNAPSHOT_REFUSAL", "FIT_MODE")
REFUSALS = (OPS.OperatorRefusal, SNAP.SnapshotRefusal)


def _file_sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_sha256s() -> dict:
    return {"tools/df_operators.py": _file_sha(HERE / "df_operators.py"),
            "tools/df_snapshot.py": _file_sha(HERE / "df_snapshot.py"),
            "tools/df_causal_battery.py": _file_sha(Path(__file__)),
            "tests/df_causal_reference.py": _file_sha(ROOT / "tests" / "df_causal_reference.py"),
            "tools/df_synthetic_contract.py": _file_sha(HERE / "df_synthetic_contract.py"),
            "tools/df_snr.py": _file_sha(HERE / "df_snr.py")}


def code_sha256() -> str:
    return hashlib.sha256(json.dumps(code_sha256s(), sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ cases
def base_series(n: int, V: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    return (np.cumsum(r.normal(scale=0.3, size=(n, V)), axis=0) + 0.8 * np.sin(2 * np.pi * t / 24)
            + r.normal(size=(n, V)))


_CASES: dict = {}


def cases() -> dict:
    """case_id -> (X, scale); generated once, never mutated by the checks."""
    if not _CASES:
        _CASES.update(_make_cases())
        for x, _ in _CASES.values():
            x.flags.writeable = False
    return _CASES


def _make_cases() -> dict:
    nan = np.nan
    out = {"univariate": (base_series(SMALL_N, 1, 1), 1.0), "multivariate": (base_series(SMALL_N, 3, 2), 1.0)}
    x = base_series(SMALL_N, 2, 3)
    x[7, 0] = x[31, 1] = x[64, 0] = x[65, 1] = nan
    out["nan_isolated"] = (x, 1.0)
    x = base_series(SMALL_N, 2, 4)
    x[15:19, 0] = nan
    x[40:48, 1] = nan
    x[62:66, :] = nan
    out["nan_blocks"] = (x, 1.0)
    x = base_series(SMALL_N, 2, 5)
    x[20:50, 0] = nan
    x[70:75, :] = nan
    out["gaps"] = (x, 1.0)
    x = np.full((SMALL_N, 2), 3.25)
    x[:, 1] = 0.0
    out["constant"] = (x, 1.0)
    out["scale_1e-12"] = (base_series(SMALL_N, 2, 6) * 1e-12, 1e-12)
    out["scale_1e12"] = (base_series(SMALL_N, 2, 7) * 1e12, 1e12)
    out["large"] = (base_series(LARGE_N, 2, 8), 1.0)
    return out


SMALL_CASES = ("univariate", "multivariate", "nan_isolated", "nan_blocks", "gaps", "constant", "scale_1e-12",
               "scale_1e12")
ALL_CASES = SMALL_CASES + ("large",)


def operator_uses() -> list:
    """(spec, mode): every causal spec under each temporal mode it is used in."""
    uses = []
    for spec in OPS.bank_specs():
        kind = spec["kind"]
        if kind not in OPS.CAUSAL_KINDS:
            continue
        modes = ((OPS.EXPANDING_PREFIX, OPS.FROZEN_PREVIOUS_PARTITION) if kind == "causal_decomposition"
                 else (OPS.KIND_FIT_MODES[kind][0],))
        uses.extend((spec, m) for m in modes)
    return uses


def use_id(spec: dict, mode: str) -> str:
    return spec["kind"] + "-" + "-".join(f"{k}{v}" for k, v in sorted(spec["params"].items())) + "-" + mode


_FIT_CACHE: dict = {}


def fitted_for(spec: dict, mode: str, V: int, scale: float) -> dict:
    key = (json.dumps(spec, sort_keys=True), mode, V, scale)
    if key not in _FIT_CACHE:
        f = OPS._fit_kernel(spec, base_series(300, V, 99) * scale, mode)
        if f["status"] != "FITTED":
            f = OPS._fit_kernel(spec, base_series(300, V, 99), mode)
        if f["status"] != "FITTED":
            raise RuntimeError(f"{use_id(spec, mode)} abstains on the battery train series: {f['abstain_reason']}")
        _FIT_CACHE[key] = f
    return _FIT_CACHE[key]


def boundary_cuts(spec: dict, mode: str, n: int, n_random: int, seed: int) -> list:
    kind, p = spec["kind"], spec["params"]
    marks = set()
    j = 0
    while 2 ** j <= n:
        marks.update({2 ** j - 1, 2 ** j, 2 ** j + 1})
        j += 1
    w = OPS.warmup_length(kind, p, mode)
    marks.update({w - 1, w, w + 1})
    if kind == "causal_decomposition":
        for k in (1, 2, 3):
            marks.update({k * p["period"] - 1, k * p["period"], k * p["period"] + 1})
    marks.update({0, n - 2, n - 1})
    rng = np.random.default_rng(seed)
    marks.update(int(v) for v in rng.integers(0, n, size=n_random))
    return sorted(m for m in marks if 0 <= m < n)


def restart_points(spec: dict, mode: str, n: int) -> list:
    kind, p = spec["kind"], spec["params"]
    pts = set()
    j = 0
    while 2 ** j <= n:
        pts.update({2 ** j - 1, 2 ** j, 2 ** j + 1})
        j += 1
    w = OPS.warmup_length(kind, p, mode)
    pts.update({w - 1, w, w + 1})
    return sorted(k for k in pts if 0 <= k <= n)


# ---------------------------------------------------------------- runners
def kernel_runner(fitted: dict):
    haar = fitted["spec"]["kind"] == "trailing_haar_threshold"

    def run(X):
        outs, avail, reason = OPS._transform_kernel_components(fitted, X)
        vals = dict(outs)
        if haar:
            for j, lv in enumerate(OPS._transform_levels_kernel(fitted, X), 1):
                vals[f"L{j}.detail"], vals[f"L{j}.approx"] = lv["detail"], lv["approx"]
        return vals, avail, reason
    return run


def _level_of(name: str):
    m = re.match(r"L(\d+)\.", name)
    return int(m.group(1)) if m else None


def _groups(names) -> dict:
    g = {None: sorted(names)}
    for n in names:
        lv = _level_of(n)
        if lv is not None:
            g.setdefault(lv, []).append(n)
    return g


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)


def prefix_all_t(run, X: np.ndarray, cuts) -> dict:
    """group -> first failure message; empty when every cut holds."""
    fv, fa, fr = run(X)
    groups = _groups(fv)
    fails: dict = {}
    for t in cuts:
        v, a, r = run(X[:t + 1])
        common = _same(fa[:t + 1], a) and _same(fr[:t + 1], r)
        for g, names in groups.items():
            if g in fails:
                continue
            if not (common and all(_same(fv[n][:t + 1], v[n]) for n in names)):
                bad = [n for n in names if not _same(fv[n][:t + 1], v[n])]
                fails[g] = f"prefix differs at t={t} (outputs {bad}, availability/reason equal={common})"
        if len(fails) == len(groups):
            break
    return fails


EXHAUSTIVE_MAX_N = 256


def cuts_for_prefix(n: int, spec: dict, mode: str) -> list:
    """Every t up to EXHAUSTIVE_MAX_N rows; stratified cuts beyond. The mutation
    `exhaustive_cuts` replaces this with the seven historical cuts."""
    if not BATTERY_GUARDS["exhaustive_cuts"]:
        return [t for t in SEVEN_CUTS if t < n]
    return list(range(n)) if n <= EXHAUSTIVE_MAX_N else boundary_cuts(spec, mode, n, 24, SEED)


def suffix(kind: str, X: np.ndarray, t: int, scale: float) -> np.ndarray:
    n, V = X.shape
    m = n - t - 1
    tail = X[t + 1:].copy()
    i = np.arange(m, dtype=float)[:, None]
    if kind == "zeros":
        return np.zeros((m, V))
    if kind == "large_constant":
        return np.full((m, V), 1e6 * scale)
    if kind == "other_seed_noise":
        return np.random.default_rng(SEED + 7919 + t).normal(size=(m, V)) * 5.0 * scale
    if kind == "reversed":
        return X[::-1][t + 1:].copy()
    if kind == "nan_blocks":
        tail[((np.arange(m) // 3) % 2) == 0] = np.nan
        return tail
    if kind == "impulse_at_t_plus_1":
        tail[0] = np.where(np.isnan(tail[0]), 1e3 * scale, tail[0] + 1e3 * scale)
        return tail
    if kind == "step":
        return tail + 50.0 * scale
    if kind == "chirp":
        return 10.0 * scale * np.sin(2 * np.pi * (0.01 + 0.2 * i / max(m, 1)) * i) * np.ones((1, V))
    if kind == "regime_change":
        return tail * 10.0 + 100.0 * scale
    raise ValueError(kind)


def suffix_adversarial(run, X: np.ndarray, cuts, scale: float) -> dict:
    fv, fa, fr = run(X)
    groups = _groups(fv)
    fails: dict = {}
    for t in cuts:
        if t >= X.shape[0] - 1:
            continue
        for sk in SUFFIXES:
            X2 = X.copy()
            X2[t + 1:] = suffix(sk, X, t, scale)
            v, a, r = run(X2)
            common = _same(fa[:t + 1], a[:t + 1]) and _same(fr[:t + 1], r[:t + 1])
            for g, names in groups.items():
                if g not in fails and not (common and all(_same(fv[n][:t + 1], v[n][:t + 1]) for n in names)):
                    fails[g] = f"suffix {sk} at t={t} changed rows <= t"
    return fails


def batch_step_chunk_restart(fitted: dict, X: np.ndarray, chunks, restarts) -> dict:
    run = kernel_runner(fitted)
    fv, fa, fr = run(X)
    groups = _groups(fv)
    comps = fitted["meta"]["outputs"]
    primary = comps[0]
    haar = fitted["spec"]["kind"] == "trailing_haar_threshold"
    n = X.shape[0]
    fails: dict = {}

    def fail(g, msg):
        fails.setdefault(g, msg)

    st = OPS._kernel_init_state(fitted)
    blobs = {}
    rows = {k: [] for k in fv}
    ra, rr = [], []
    for i in range(n):
        if i in restarts:
            blobs[i] = OPS.save_state(st)
        o, a, r, st = OPS._kernel_step_components(fitted, st, X[i])
        for k in comps:
            rows[k].append(o[k].copy())
        if haar:
            for j, lv in enumerate(OPS._kernel_state_levels(fitted, st), 1):
                rows[f"L{j}.detail"].append(lv["detail"])
                rows[f"L{j}.approx"].append(lv["approx"])
        ra.append(a.copy())
        rr.append(r.copy())
    if n in restarts:
        blobs[n] = OPS.save_state(st)
    common = _same(fa, np.array(ra)) and _same(fr, np.array(rr))
    for g, names in groups.items():
        if not (common and all(_same(fv[k], np.array(rows[k])) for k in names)):
            fail(g, "batch differs from sample-by-sample step")
    for c in chunks:
        st = OPS._kernel_init_state(fitted)
        ys, as_, rs = [], [], []
        for s in range(0, n, c):
            y, a, r, st = OPS._transform_chunk_kernel(fitted, st, X[s:s + c])
            ys.append(y)
            as_.append(a)
            rs.append(r)
        if not (_same(fv[primary], np.concatenate(ys)) and _same(fa, np.concatenate(as_))
                and _same(fr, np.concatenate(rs))):
            fail(None, f"batch differs from a stream fragmented in chunks of {c}")
    for k, blob in sorted(blobs.items()):
        st = OPS.load_state(blob, fitted)
        for i in range(k, n):
            o, a, r, st = OPS._kernel_step_components(fitted, st, X[i])
            ok = _same(fa[i], a) and _same(fr[i], r) and all(_same(fv[c][i], o[c]) for c in comps)
            if not ok:
                fail(None, f"save/load restart at row {k} differs at row {i}")
                break
    return fails


def reference_equality(fitted: dict, X: np.ndarray, cuts) -> dict:
    spec, mode = fitted["spec"], fitted["fit_mode"]
    fv, fa, fr = kernel_runner(fitted)(X)
    groups = _groups(fv)
    bitwise = REF.REFERENCE_TOLERANCE[spec["kind"]][0] == REF.BITWISE
    fails: dict = {}

    def eq(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if bitwise:
            return np.array_equal(a, b, equal_nan=True)
        both = np.isnan(a) & np.isnan(b)
        tol = 1e-12 * np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
        return bool(np.all(both | (np.abs(a - b) <= tol)))

    for t in cuts:
        ref = REF.outputs_at(spec, fitted["fitted"], mode, X[:t + 1].tolist(), t0=0)
        rr = np.array(ref["reason"])
        common = _same(fr[t], rr) and _same(fa[t], rr == "AVAILABLE")
        vals = dict(ref["values"])
        for j, per_col in enumerate(zip(*ref.get("levels", [])), 1):
            vals[f"L{j}.detail"] = [lv["detail"] for lv in per_col]
            vals[f"L{j}.approx"] = [lv["approx"] for lv in per_col]
        for g, names in groups.items():
            if g in fails:
                continue
            if not (common and all(n in vals and eq(fv[n][t], vals[n]) for n in names)):
                bad = [n for n in names if n not in vals or not eq(fv[n][t], vals[n])]
                fails[g] = f"production differs from the prefix-only reference at t={t} (outputs {bad}, " \
                           f"reason equal={common})"
    return fails


def operator_rows(spec: dict, mode: str, case_id: str, run_id: str, code: str,
                  classes=("PREFIX_ALL_T", "BATCH_STEP_CHUNK_RESTART", "SUFFIX_ADVERSARIAL",
                           "REFERENCE_EQUALITY")) -> list:
    X, scale = cases()[case_id]
    n, V = X.shape
    fitted = fitted_for(spec, mode, V, scale)
    run = kernel_runner(fitted)
    small = n <= SMALL_N
    levels = [None] + (list(range(1, spec["params"]["levels"] + 1))
                       if spec["kind"] == "trailing_haar_threshold" else [])
    rows = []
    for cls in classes:
        if cls == "PREFIX_ALL_T":
            cuts = cuts_for_prefix(n, spec, mode)
            fails = prefix_all_t(run, X, cuts)
        elif cls == "BATCH_STEP_CHUNK_RESTART":
            cuts = restart_points(spec, mode, n)
            fails = batch_step_chunk_restart(fitted, X, SMALL_CHUNKS if small else LARGE_CHUNKS, set(cuts))
        elif cls == "SUFFIX_ADVERSARIAL":
            cuts = (list(range(n - 1)) if case_id == "univariate"
                    else boundary_cuts(spec, mode, n, 8 if small else 16, SEED + 1))
            fails = suffix_adversarial(run, X, cuts, scale)
        else:
            cuts = list(range(n)) if small else boundary_cuts(spec, mode, n, 16, SEED + 2)
            fails = reference_equality(fitted, X, cuts)
        for lv in levels:
            msg = fails.get(lv)
            rows.append(_row(run_id, spec["kind"], dict(spec["params"], fit_mode=mode), lv, cls,
                             f"{case_id}", n, len(cuts), "FAIL" if msg else "PASS", msg or "", code))
    return rows


def _row(run_id, kind, params, level, cls, case_id, n, cuts, outcome, reason, code) -> dict:
    return {"run_id": run_id, "operator_kind": kind, "operator_params": params, "level": level,
            "test_class": cls, "case_id": case_id, "n": int(n), "cuts_tested": int(cuts), "outcome": outcome,
            "reason": reason, "code_sha256": code}


# ------------------------------------------------------- negative controls
def _trailing_mean_np(X, w):
    out = np.full(X.shape, np.nan)
    for t in range(w - 1, X.shape[0]):
        out[t] = X[t - w + 1:t + 1].mean(axis=0)
    return out


def _control_runner(fn):
    def run(X):
        Y = np.asarray(fn(np.asarray(X, dtype=float)), dtype=float)
        avail = np.isfinite(Y)
        return {"y": Y}, avail, np.where(avail, "AVAILABLE", "UNAVAILABLE")
    return run


def _centered(X):
    out = np.full(X.shape, np.nan)
    for t in range(2, X.shape[0] - 2):
        out[t] = X[t - 2:t + 3].mean(axis=0)
    return out


def _filtfilt(X):
    from scipy import signal
    b, a = signal.butter(2, 0.2)
    if X.shape[0] <= 3 * max(len(a), len(b)):
        return np.full(X.shape, np.nan)
    return signal.filtfilt(b, a, X, axis=0)


def _shift_minus_k(X, k=2):
    out = np.full(X.shape, np.nan)
    out[:-k] = X[k:]
    return out


def _same_convolution(X):
    h = np.ones(5) / 5
    return np.stack([np.convolve(X[:, j], h, mode="same") for j in range(X.shape[1])], axis=1)


def _full_dwt_row(X):
    import pywt
    cols = []
    for j in range(X.shape[1]):
        _, d = pywt.dwt(X[:, j], "db4", mode="periodization")
        cols.append(np.repeat(d, 2)[:X.shape[0]])
    return np.stack(cols, axis=1)


def _full_fft_feature(X):
    cols = []
    for j in range(X.shape[1]):
        F = np.fft.rfft(X[:, j])
        F[max(1, len(F) // 8):] = 0
        cols.append(np.fft.irfft(F, n=X.shape[0]))
    return np.stack(cols, axis=1)


def _phase_compensated(X, w=9):
    M = _trailing_mean_np(X, w)
    out = np.full(X.shape, np.nan)
    k = (w - 1) // 2
    out[:-k] = M[k:]
    return out


OUTPUT_CONTROLS = {
    "centered_rolling_window": _centered,
    "filtfilt": _filtfilt,
    "shift_minus_k": _shift_minus_k,
    "same_convolution_or_right_padding": _same_convolution,
    "full_series_dwt_as_time_row": _full_dwt_row,
    "full_series_fft_reconstruction_feature": _full_fft_feature,
    "phase_compensation_shift_back": _phase_compensated,
}


def fixture(n=100, V=2, seed=0, **kw):
    X = base_series(n, V, seed)
    c, L = SYNC.in_memory_contract(X, name=f"fx{seed}", **kw)
    return X, c, L


def _attempt(fn):
    try:
        fn()
    except REFUSALS as exc:
        return str(exc)
    return None


EWMA = {"kind": "ewma", "params": {"alpha": 0.3}}
FROZEN, EXPANDING, OFFLINE = OPS.FROZEN_PREVIOUS_PARTITION, OPS.EXPANDING_PREFIX, OPS.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL


def _control_fit_with_later_rows():
    _, c, L = fixture(seed=11)
    a = _attempt(lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L, end=100), FROZEN))
    b = _attempt(lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "CONFIRMATION", L), FROZEN))
    ok = a is not None and "not inside the TRAIN partition" in a and b is not None and "not allowed" in b
    return ok, f"train range over calibration/confirmation: {a}; confirmation role: {b}"


def _control_reorder_by_later_timestamp():
    X = base_series(100, 2, 12)
    ts = np.arange(100, dtype=np.int64) * 60
    order = np.arange(100)
    order[[70, 71]] = [71, 70]                   # a row sorted after one with a later timestamp
    c, L = SYNC.in_memory_contract(X[order], name="fx_reorder", timestamps=ts[order])
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    a = _attempt(lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100)))
    _, c2, L2 = fixture(seed=13)
    f2 = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c2, "TRAIN", L2), FROZEN)
    snap = SNAP.TransformSnapshot.from_contract(c2, L2, start=60, end=100)
    m = snap.matrix.copy()
    m[[10, 11]] = m[[11, 10]]
    m.flags.writeable = False
    object.__setattr__(snap, "matrix", m)        # reordered after materialization
    b = _attempt(lambda: OPS.transform_batch(f2, snap))
    ok = a is not None and "strictly increasing" in a and b is not None and "matrix digest" in b
    return ok, f"source order: {a}; after materialization: {b}"


def _control_state_from_other_series():
    _, ca, La = fixture(seed=14)
    _, cb, Lb = fixture(seed=15)
    fa = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(ca, "TRAIN", La), FROZEN)
    sa = SNAP.TransformSnapshot.from_contract(ca, La, start=60, end=100)
    sb = SNAP.TransformSnapshot.from_contract(cb, Lb, start=60, end=100)
    st = OPS.init_state(fa, sa)
    for i in range(5):
        _, _, _, st = OPS.step(fa, st, sa.row(i))
    msg = _attempt(lambda: OPS.step(fa, st, sb.row(5)))
    return msg is not None and "another series" in msg, f"state of series A stepped on series B: {msg}"


SNAPSHOT_CONTROLS = {
    "fit_with_calibration_or_confirmation_rows": _control_fit_with_later_rows,
    "reorder_by_timestamp_later_than_materialized": _control_reorder_by_later_timestamp,
    "state_reused_from_another_series": _control_state_from_other_series,
}


def negative_control_detection(name: str) -> tuple:
    """-> (detected, detector message)."""
    if name in OUTPUT_CONTROLS:
        run = _control_runner(OUTPUT_CONTROLS[name])
        X = base_series(SMALL_N, 2, 21)
        fails = prefix_all_t(run, X, range(SMALL_N))
        if fails:
            return True, f"PREFIX_ALL_T: {fails[None]}"
        sfails = suffix_adversarial(run, X, range(SMALL_N - 1), 1.0)
        return bool(sfails), (f"SUFFIX_ADVERSARIAL: {sfails[None]}" if sfails else "not detected")
    return SNAPSHOT_CONTROLS[name]()


def negative_control_rows(run_id: str, code: str) -> list:
    rows = []
    for name in list(OUTPUT_CONTROLS) + list(SNAPSHOT_CONTROLS):
        detected, msg = negative_control_detection(name)
        n = SMALL_N if name in OUTPUT_CONTROLS else 100
        rows.append(_row(run_id, "NEGATIVE_CONTROL", {"class": name}, None, "NEGATIVE_CONTROL", name, n,
                         SMALL_N if name in OUTPUT_CONTROLS else 1, "DETECTED" if detected else "NOT_DETECTED",
                         msg, code))
    return rows


# --------------------------------------------------------- guard mutation
def _redigest(snap):
    for arr, key in (("matrix", "matrix_sha256"), ("timestamps", "timestamps_sha256"),
                     ("available_at", "availability_sha256")):
        object.__setattr__(snap, key, SNAP.array_sha256(getattr(snap, arr)))
    object.__setattr__(snap, "snapshot_sha256", SNAP.C.sha_obj(snap.facts()))
    return snap


def _probe_contract_digest():
    _, c, L = fixture(seed=31)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    other = dict(c, original_fields={"forged": True})
    other = SNAP.C.seal(other)
    object.__setattr__(s, "contract_json", SNAP.C.canonical(other))
    return lambda: OPS.fit(EWMA, s, FROZEN)


def _probe_matrix_digest():
    _, c, L = fixture(seed=32)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    m = s.matrix.copy()
    m[0, 0] += 1.0
    m.flags.writeable = False
    object.__setattr__(s, "matrix", m)
    return lambda: OPS.fit(EWMA, s, FROZEN)


def _probe_source_rederive():
    _, c, L = fixture(seed=33)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    m = s.matrix.copy()
    m[0, 0] += 1.0
    m.flags.writeable = False
    object.__setattr__(s, "matrix", m)
    _redigest(s)
    return lambda: OPS.fit(EWMA, s, FROZEN)


def _probe_snapshot_digest():
    _, c, L = fixture(seed=34)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    object.__setattr__(s, "timestamp_meaning", "INSTANT")
    return lambda: OPS.fit(EWMA, s, FROZEN)


def _probe_monotonic_batch():
    X = base_series(100, 2, 35)
    ts = np.arange(100, dtype=np.int64) * 60
    ts[70] = ts[69]
    c, L = SYNC.in_memory_contract(X, name="fx_dup", timestamps=ts)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100))


def _probe_monotonic_step():
    _, c, L = fixture(seed=36)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    s = SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100)
    st = OPS.init_state(f, s)
    return lambda: OPS.step(f, st, s.row(1))


def _probe_range_in_partition():
    _, c, L = fixture(seed=37)
    return lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "CALIBRATION", L, start=0, end=60), FROZEN)


def _probe_role_allowed():
    _, c, L = fixture(n=300, seed=38)            # 60 confirmation rows: only the role guard can refuse
    return lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "CONFIRMATION", L), FROZEN)


def _probe_later_exclusion():
    _, c, L = fixture(seed=39)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    object.__setattr__(s, "excluded_partitions", ())
    _redigest(s)
    return lambda: OPS.fit(EWMA, s, FROZEN)


def _probe_availability():
    X = base_series(100, 2, 40)
    c, L = SYNC.in_memory_contract(X, name="fx_delay", timestamps=np.arange(100) * 60, availability_delay=30)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100))


def _probe_artifact_bound():
    X, c, L = fixture(seed=41)
    f = OPS._fit_kernel(EWMA, X[:60], FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100))


def _probe_dataset_binding():
    _, ca, La = fixture(seed=42)
    _, cb, Lb = fixture(seed=43)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(ca, "TRAIN", La), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(cb, Lb, start=60, end=100))


def _probe_column_identity():
    _, c, L = fixture(seed=44)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L, columns=["v0", "v1"]), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100,
                                                                              columns=["v1", "v0"]))


def _probe_partition_license():
    _, c, L = fixture(seed=45)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=0, end=100))


HAAR = {"kind": "trailing_haar_threshold", "params": {"levels": 2, "threshold_k": 3.0}}


def _probe_fit_mode():
    _, c, L = fixture(seed=46)
    return lambda: OPS.transform_batch(OPS.fit(HAAR, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), EXPANDING),
                                       SNAP.TransformSnapshot.from_contract(c, L, start=0, end=60))


def _probe_stream_binding():
    _, ca, La = fixture(seed=47)
    _, cb, Lb = fixture(seed=48)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(ca, "TRAIN", La), FROZEN)
    st = OPS.init_state(f, SNAP.TransformSnapshot.from_contract(ca, La, start=60, end=100))
    sb = SNAP.TransformSnapshot.from_contract(cb, Lb, start=60, end=100)
    return lambda: OPS.step(f, st, sb.row(0))


def _probe_exhaustive_battery():
    """A one-sample leak at t=100: the exhaustive prefix check must detect it."""
    X = base_series(240, 2, 49)
    f = OPS._fit_kernel(EWMA, base_series(300, 2, 99), FROZEN)

    def leaking(Z):
        y, a, r = OPS._transform_kernel(f, Z)
        y = y.copy()
        if Z.shape[0] > 101:
            y[100] = Z[101]
        return {"y": y}, a, r

    def check():
        fails = prefix_all_t(leaking, X, cuts_for_prefix(240, EWMA, FROZEN))
        if fails:
            raise OPS.OperatorRefusal(f"battery detected a future leak: {fails[None]}")
    return check


GUARD_PROBES = (
    ("contract_digest", SNAP.GUARDS, _probe_contract_digest, "contract digest does not re-derive"),
    ("matrix_digest", SNAP.GUARDS, _probe_matrix_digest, "matrix digest does not re-derive"),
    ("source_rederive", SNAP.GUARDS, _probe_source_rederive, "source bytes do not re-derive"),
    ("snapshot_digest", SNAP.GUARDS, _probe_snapshot_digest, "snapshot digest does not re-derive"),
    ("monotonic_timestamps", SNAP.GUARDS, _probe_monotonic_batch, "not strictly increasing"),
    ("monotonic_timestamps.step", SNAP.GUARDS, _probe_monotonic_step, "is not the next row"),
    ("range_in_partition", SNAP.GUARDS, _probe_range_in_partition, "is not inside the CALIBRATION partition"),
    ("role_allowed", SNAP.GUARDS, _probe_role_allowed, "is not allowed by the design"),
    ("later_partition_exclusion", SNAP.GUARDS, _probe_later_exclusion, "later partitions are not excluded"),
    ("availability", SNAP.GUARDS, _probe_availability, "available after its decision instant"),
    ("artifact_bound", OPS.GUARDS, _probe_artifact_bound, "not bound to a FitSnapshot"),
    ("dataset_binding", OPS.GUARDS, _probe_dataset_binding, "another dataset or contract"),
    ("column_identity", OPS.GUARDS, _probe_column_identity, "columns differ from the fitted columns"),
    ("transform_partition_license", OPS.GUARDS, _probe_partition_license, "is not licensed"),
    ("fit_mode_enforcement", OPS.GUARDS, _probe_fit_mode, "is not implemented for kind"),
    ("stream_binding", OPS.GUARDS, _probe_stream_binding, "another series"),
    ("exhaustive_cuts", BATTERY_GUARDS, _probe_exhaustive_battery, "battery detected a future leak: prefix differs at t=100"),
)
_PROBE_OPERATOR = {"fit_mode_enforcement": HAAR}


@contextlib.contextmanager
def guard_disabled(table: dict, flag: str):
    old = table[flag]
    table[flag] = False
    try:
        yield
    finally:
        table[flag] = old


def guard_mutation(name: str) -> dict:
    """-> {"on": message, "off": message or None, "detected": bool}."""
    _, table, make, expected = next(p for p in GUARD_PROBES if p[0] == name)
    flag = name.split(".")[0]
    on = _attempt(make())
    with guard_disabled(table, flag):
        off = _attempt(make())
    detected = on is not None and expected in on and (off is None or expected not in off)
    return {"on": on, "off": off, "expected": expected, "detected": detected}


def fit_mode_leak_under_mutation() -> str:
    """With fit_mode_enforcement off, an EXPANDING trailing Haar reuses thresholds from later train
    rows on earlier rows: outputs at t <= 30 change when train rows 31..59 change."""
    X = base_series(100, 2, 50)
    X2 = X.copy()
    X2[31:60] = X2[31:60] * 5.0 + 3.0
    outs = []
    with guard_disabled(OPS.GUARDS, "fit_mode_enforcement"):
        for Z, nm in ((X, "a"), (X2, "b")):
            c, L = SYNC.in_memory_contract(Z, name=f"fx_leak_{nm}")
            f = OPS.fit(HAAR, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), EXPANDING)
            y, _, _ = OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=0, end=31))
            outs.append(y)
    moved = not _same(outs[0], outs[1])
    return f"outputs at t<=30 moved by later train rows: {moved}"


def guard_mutation_rows(run_id: str, code: str) -> list:
    rows = []
    for name, _, _, _ in GUARD_PROBES:
        res = guard_mutation(name)
        spec = _PROBE_OPERATOR.get(name, EWMA)
        extra = f"; {fit_mode_leak_under_mutation()}" if name == "fit_mode_enforcement" else ""
        rows.append(_row(run_id, spec["kind"], dict(spec["params"], guard=name), None, "GUARD_MUTATION", name,
                         240 if name == "exhaustive_cuts" else 100, 240 if name == "exhaustive_cuts" else 1,
                         "DETECTED" if res["detected"] else "NOT_DETECTED",
                         f"guard on: {res['on']}; guard off: {res['off'] or 'no refusal'}{extra}", code))
    return rows


# ------------------------------------------------------ snapshot refusals
def _refusal_cases():
    X, c, L = fixture(seed=60)
    fit_snap = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    f = OPS.fit(EWMA, fit_snap, FROZEN)
    tsnap = SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100)

    def unordered():
        ts = np.arange(100, dtype=np.int64) * 60
        ts[[70, 71]] = ts[[71, 70]]
        c2, L2 = SYNC.in_memory_contract(X, name="fx_unordered", timestamps=ts)
        f2 = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c2, "TRAIN", L2), FROZEN)
        OPS.transform_batch(f2, SNAP.TransformSnapshot.from_contract(c2, L2, start=60, end=100))

    def range_change():
        s = SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100)
        object.__setattr__(s, "start", 61)
        OPS.transform_batch(f, s)

    def role_change():
        s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
        object.__setattr__(s, "role", "CALIBRATION")
        OPS.fit(EWMA, s, FROZEN)

    def bytes_after():
        blobs = dict(L)
        s = SNAP.TransformSnapshot.from_contract(c, blobs, start=60, end=100)
        arr = np.load(__import__("io").BytesIO(blobs["observed_signal.npy"]))
        arr[70, 0] += 1.0
        buf = __import__("io").BytesIO()
        np.save(buf, arr)
        blobs["observed_signal.npy"] = buf.getvalue()
        OPS.transform_batch(f, s)

    def bytes_at_construction():
        blobs = dict(L)
        blobs["observed_signal.npy"] = blobs["observed_signal.npy"][:-8] + b"\0" * 8
        SNAP.FitSnapshot.from_contract(c, "TRAIN", blobs)

    def availability():
        _probe_availability()()

    return (
        ("bare_array_fit", lambda: OPS.fit(EWMA, X[:60], FROZEN), "requires a FitSnapshot"),
        ("role_string_fit", lambda: OPS.fit(EWMA, "train", FROZEN), "requires a FitSnapshot"),
        ("bare_array_transform", lambda: OPS.transform_batch(f, X[60:]), "requires a TransformSnapshot"),
        ("bare_row_step", lambda: OPS.step(f, OPS.init_state(f, tsnap), X[60]), "requires a TransformRow"),
        ("fit_includes_calibration_rows",
         lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L, end=70), FROZEN),
         "not inside the TRAIN partition"),
        ("fit_on_confirmation", lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "CONFIRMATION", L), FROZEN),
         "not allowed by the design"),
        ("unordered_rows", unordered, "not strictly increasing"),
        ("duplicated_rows", lambda: _probe_monotonic_batch()(), "not strictly increasing"),
        ("available_after_decision", availability, "available after its decision instant"),
        ("column_change", lambda: _probe_column_identity()(), "columns differ"),
        ("range_change", range_change, "do not re-derive"),
        ("role_change", role_change, "does not re-derive"),
        ("bytes_changed_after_materialization", bytes_after, "source bytes"),
        ("bytes_mismatch_at_construction", bytes_at_construction, "do not match the contract digest"),
        ("partition_not_licensed", lambda: _probe_partition_license()(), "is not licensed"),
        ("step_row_of_unlicensed_partition",
         lambda: OPS.step(f, OPS.init_state(f, tsnap),
                          SNAP.TransformSnapshot.from_contract(c, L, start=0, end=100).row(0)),
         "is not licensed"),
    )


def snapshot_refusal_rows(run_id: str, code: str) -> list:
    rows = []
    for case_id, fn, expected in _refusal_cases():
        msg = _attempt(fn)
        ok = msg is not None and expected in msg
        rows.append(_row(run_id, "ewma", dict(EWMA["params"]), None, "SNAPSHOT_REFUSAL", case_id, 100, 1,
                         "PASS" if ok else "FAIL", "" if ok else f"expected refusal containing {expected!r}, got {msg}",
                         code))
    return rows


# -------------------------------------------------------------- fit modes
DECOMP = {"kind": "causal_decomposition", "params": {"period": 24, "season_alpha": 0.1, "trend_alpha": 0.1}}


def _pre_seasonal_case() -> tuple:
    """The PRE's case (C154): training rows after t=100 change. EXPANDING outputs at t<=100 must not move;
    FROZEN parameters move, and the in-sample FROZEN transform refuses."""
    X = base_series(500, 2, 70)
    X2 = X.copy()
    X2[101:] += 10.0 * np.sin(np.arange(399) / 3.0)[:, None]
    ys, frozen, refused = [], [], []
    for Z, nm in ((X, "a"), (X2, "b")):
        c, L = SYNC.in_memory_contract(Z, name=f"fx_pre_{nm}")
        fs = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
        fe = OPS.fit(DECOMP, fs, EXPANDING)
        comps, _, r = OPS.transform_batch_components(fe, SNAP.TransformSnapshot.from_contract(c, L, start=0, end=101))
        ys.append((comps, r))
        ff = OPS.fit(DECOMP, fs, FROZEN)
        frozen.append(ff["fitted"]["seasonal_init"])
        refused.append(_attempt(lambda: OPS.transform_batch(
            ff, SNAP.TransformSnapshot.from_contract(c, L, start=0, end=101))))
    same = all(_same(ys[0][0][k], ys[1][0][k]) for k in ys[0][0]) and _same(ys[0][1], ys[1][1])
    return same, frozen[0] != frozen[1], all(m and "is not licensed" in m for m in refused)


def fit_mode_rows(run_id: str, code: str) -> list:
    rows = []
    same, frozen_moved, refused = _pre_seasonal_case()
    rows.append(_row(run_id, "causal_decomposition", dict(DECOMP["params"], fit_mode=EXPANDING), None, "FIT_MODE",
                     "pre_seasonal_case_expanding_prefix_unchanged", 500, 101, "PASS" if same else "FAIL",
                     "" if same else "EXPANDING outputs at t<=100 moved when later training rows changed", code))
    ok = frozen_moved and refused
    rows.append(_row(run_id, "causal_decomposition", dict(DECOMP["params"], fit_mode=FROZEN), None, "FIT_MODE",
                     "pre_seasonal_case_frozen_in_sample_refused", 500, 101, "PASS" if ok else "FAIL",
                     "" if ok else f"frozen parameters moved={frozen_moved}, in-sample refused={refused}", code))
    _, c, L = fixture(seed=71)
    fs = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    train_rows = SNAP.TransformSnapshot.from_contract(c, L, start=0, end=60)
    for spec in OPS.bank_specs():
        kind = spec["kind"]
        if kind in OPS.NON_CAUSAL_KINDS or spec != next(s for s in OPS.bank_specs() if s["kind"] == kind):
            continue
        f = OPS.fit(spec, fs, FROZEN)
        if f["status"] == "FITTED":
            msg = _attempt(lambda: OPS.transform_batch(f, train_rows))
            ok = msg is not None and "is not licensed" in msg
            rows.append(_row(run_id, kind, dict(spec["params"], fit_mode=FROZEN), None, "FIT_MODE",
                             "frozen_transform_of_fit_partition_refused", 100, 1, "PASS" if ok else "FAIL",
                             "" if ok else f"got {msg}", code))
        if EXPANDING not in OPS.KIND_FIT_MODES[kind]:
            msg = _attempt(lambda: OPS.fit(spec, fs, EXPANDING))
            ok = msg is not None and "is not implemented for kind" in msg
            rows.append(_row(run_id, kind, dict(spec["params"], fit_mode=EXPANDING), None, "FIT_MODE",
                             "expanding_prefix_not_implemented_refused", 100, 1, "PASS" if ok else "FAIL",
                             "" if ok else f"got {msg}", code))
        fo = OPS.fit(spec, fs, OFFLINE)
        msgs = [_attempt(lambda: OPS.transform_batch(fo, train_rows)),
                _attempt(lambda: OPS.init_state(fo, train_rows)),
                _attempt(lambda: OPS.probe_transform(fo, OPS.probe_signal("STEP", 50, 2)))]
        ok = all(m is not None and ("never emits per-timestamp values" in m or "ABSTAIN" in m) for m in msgs)
        rows.append(_row(run_id, kind, dict(spec["params"], fit_mode=OFFLINE), None, "FIT_MODE",
                         "offline_analysis_never_emits", 100, 3, "PASS" if ok else "FAIL",
                         "" if ok else f"got {msgs}", code))
    return rows


# ----------------------------------------------------- C160 call graph
SNR_KERNEL_IDS = {"_offline_wavelet_mad_kernel", "_nv_wavelet_mad", "_KERNELS", "estimate_offline_train_diagnostic",
                  "_OFFLINE_AGGREGATE"}
SNR_MODULE_NAMES = {"df_snr", "df_snr.py"}
FORBIDDEN_CONSUMER = re.compile(r"operator|router|selector|agent|feature|transform|lab_eval|causal|matrix")
LOADER_FUNCS = {"spec_from_file_location", "import_module", "__import__", "_load", "load", "tool",
                "_load_sibling", "_load_operators", "run_path"}


def snr_isolation_evidence(tools_dir: Path = HERE) -> dict:
    """AST over tools/**/*.py (tests are not under tools): which modules load df_snr, which reference its
    offline kernel identifiers, and how the kernel is reachable inside df_snr."""
    loaders, refs = [], []
    for p in sorted(Path(tools_dir).rglob("*.py")):
        if p.name == "df_snr.py" or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(Path(tools_dir).parent))
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            hit = None
            if isinstance(node, ast.Import):
                hit = any(a.name.split(".")[-1] == "df_snr" for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                hit = (node.module or "").split(".")[-1] == "df_snr" or any(a.name == "df_snr" for a in node.names)
            elif isinstance(node, ast.Call):
                f = node.func
                fname = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if fname in LOADER_FUNCS:
                    hit = any(isinstance(a, ast.Constant) and isinstance(a.value, str)
                              and (a.value in SNR_MODULE_NAMES or a.value.endswith("/df_snr.py"))
                              for arg in list(node.args) + [k.value for k in node.keywords] for a in ast.walk(arg))
            if hit:
                loaders.append(rel)
            if isinstance(node, ast.Name) and node.id in SNR_KERNEL_IDS:
                refs.append(f"{rel}:{node.lineno}:{node.id}")
            if isinstance(node, ast.Attribute) and node.attr in SNR_KERNEL_IDS:
                refs.append(f"{rel}:{node.lineno}:{node.attr}")
    loaders = sorted(set(loaders))
    forbidden = [m for m in loaders if FORBIDDEN_CONSUMER.search(Path(m).name)]
    snr = ast.parse((Path(tools_dir) / "df_snr.py").read_text())
    kernel_users, kernels_users, estimator_callables, state = set(), set(), [], None
    for fn in [n for n in snr.body if isinstance(n, (ast.FunctionDef, ast.Assign, ast.AnnAssign))]:
        label = fn.name if isinstance(fn, ast.FunctionDef) else ast.unparse(
            fn.targets[0] if isinstance(fn, ast.Assign) else fn.target)
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == "_offline_wavelet_mad_kernel":
                kernel_users.add(label)
            if isinstance(node, ast.Name) and node.id == "_KERNELS":
                kernels_users.add(label)
        target = label
        if target == "ESTIMATORS":
            for k, v in zip(fn.value.keys, fn.value.values):
                if isinstance(k, ast.Constant) and k.value == "wavelet_mad":
                    for kk, vv in zip(v.keys, v.values):
                        if isinstance(vv, (ast.Name, ast.Lambda, ast.Attribute)) and not (
                                isinstance(vv, ast.Name) and vv.id.isupper()):
                            estimator_callables.append(kk.value)
                        if isinstance(kk, ast.Constant) and kk.value == "contract_state":
                            state = ast.unparse(vv)
    passed = (not refs and not forbidden and not estimator_callables and kernel_users == {"_KERNELS"}
              and state == "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL")
    return {"passed": passed, "modules_loading_df_snr": loaders, "forbidden_consumers": forbidden,
            "kernel_identifier_references_outside_df_snr": refs,
            "df_snr_kernel_referenced_by": sorted(kernel_users), "df_snr_KERNELS_referenced_by": sorted(kernels_users),
            "wavelet_mad_estimator_entry_callables": estimator_callables, "wavelet_mad_contract_state": state}


def naming_rows(run_id: str, code: str) -> list:
    rows = [dict(run_id=run_id, code_sha256=code, **d) for d in OPS.NAMING_DECISIONS]
    ev = snr_isolation_evidence()
    rows.append({"run_id": run_id, "subject": "wavelet_mad", "subject_kind": "ESTIMATOR", "previous_name": None,
                 "decision": "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL",
                 "evidence": "C160: " + json.dumps(ev, sort_keys=True), "code_sha256": code})
    return rows


# -------------------------------------------------------------------- run
def run_battery(progress=None) -> tuple:
    code = code_sha256()
    run_id = "c156_" + code[:24]
    tests = []
    for spec, mode in operator_uses():
        for case_id in ALL_CASES:
            tests.extend(operator_rows(spec, mode, case_id, run_id, code))
        if progress:
            progress(f"{use_id(spec, mode)} done")
    tests.extend(negative_control_rows(run_id, code))
    tests.extend(guard_mutation_rows(run_id, code))
    tests.extend(snapshot_refusal_rows(run_id, code))
    tests.extend(fit_mode_rows(run_id, code))
    return run_id, tests, naming_rows(run_id, code)


def summarize(run_id: str, tests: list, naming: list, wall: float) -> dict:
    counts: dict = {}
    for r in tests:
        counts.setdefault(r["test_class"], {}).setdefault(r["outcome"], 0)
        counts[r["test_class"]][r["outcome"]] += 1
    bad = [r for r in tests if r["outcome"] in ("FAIL", "NOT_DETECTED")]
    return {"schema": "crispdm.data_foundation.causal_battery_summary.v1", "run_id": run_id,
            "code_sha256": code_sha256(), "code_sha256s": code_sha256s(),
            "counts_per_test_class_and_outcome": counts, "rows": {"df_fact_causal_test": len(tests),
                                                               "df_fact_naming_isolation_decision": len(naming)},
            "failures": [{k: r[k] for k in ("operator_kind", "level", "test_class", "case_id", "reason")}
                         for r in bad],
            "all_pass": not bad, "wall_seconds": round(wall, 1),
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            "reference_tolerance": {k: list(v) for k, v in REF.REFERENCE_TOLERANCE.items()},
            "cases": {k: list(v[0].shape) for k, v in cases().items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    out = a.out.expanduser().resolve()
    local = (Path.home() / ".local").resolve()
    if out == local or local in out.parents:
        print("REFUSED: the battery never writes under ~/.local", file=sys.stderr)
        return 2
    if out.exists():
        print(f"REFUSED: {out.name} exists; battery outputs are write-once", file=sys.stderr)
        return 2
    t0 = time.time()
    run_id, tests, naming = run_battery(progress=lambda m: print(m, file=sys.stderr, flush=True))
    summary = summarize(run_id, tests, naming, time.time() - t0)
    text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False)
    if str(Path.home()) in text:
        print("REFUSED: absolute home path in the summary", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    for table, rows in (("df_fact_causal_test", tests), ("df_fact_naming_isolation_decision", naming)):
        with open(out / f"{table}.jsonl", "x") as fh:
            fh.write("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    with open(out / "CAUSAL_BATTERY_SUMMARY.json", "x") as fh:
        fh.write(text + "\n")
    print(json.dumps({k: summary[k] for k in ("run_id", "counts_per_test_class_and_outcome", "all_pass",
                                              "wall_seconds", "peak_rss_bytes")}, indent=1))
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
