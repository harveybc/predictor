#!/usr/bin/env python3
"""C172, C174, C177 (order 2026-09-13): one D2 unit, end to end, in its own process.

The worker evaluates ONE unit (one dataset) of either stratum

* ``HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY``: a unit of the intact
  C128 bank, re-run under the current code (never grants consumption);
* ``FRESH_CONFIRMATION``: a unit of a fresh reserve root whose seed is on the
  sealed tape of the sealed design;

through ONLY the public productive API

    FitSnapshot.from_contract -> df_operators.fit -> TransformSnapshot.from_contract -> transform_batch

(and, for the wavelet audit, the public ``init_state``/``step``/``transform_chunk``/
``save_state``/``load_state``). No private kernel produces a row.

Arms: every operator of the design, with its fit mode, including the controls
(identity/raw, the previously rejected operator, and the non-causal oracle).
The oracle is transformed with ``oracle_mode=True`` only to be DETECTED by a
public-API prefix check; its rows carry ``arm_role=NON_CAUSAL_ORACLE_CONTROL``
and it never competes. A non-detected oracle invalidates the root.

Rows (``df_fact_d2_unit_denoising``), one per unit x variable x arm x partition
x branch x metric: branches RAW (X), TRANSFORMED (D(X)), RESIDUAL (X - D(X)),
COMPARISON (paired in the seed) and COST. Metrics are those of
``df_lab_evaluation.partition_metrics`` (improvement, distortion, delay, event
and extreme retention, residual signal share) plus ``extreme_retention_raw``
and cost (fit and transform CPU seconds, per 1000 samples). SNR rows
(``df_fact_d2_unit_snr``), one per unit x variable x estimator x partition
(confirmation governs; calibration is diagnostic; ``wavelet_mad`` only as one
TRAIN aggregate and NOT_APPLICABLE elsewhere).

C177 wavelet audit, for every ``trailing_haar_threshold`` arm of the evaluated
unit (``audit_trailing_haar``): every prefix t of the licensed range against
the full run (values, availability, reason); every level at every t against an
audit-local prefix-only reference on the public snapshot matrix, whose
reconstruction must equal the public output at t (no delay compensation moves
output backwards); adversarial suffixes (``df_causal_battery.suffix_adversarial``
with a public-API runner) at warm-up, power-of-two and seeded cuts; step,
chunked streams and save/load restarts at those boundaries; ``wavelet_mad``
refusing every path but one whole-TRAIN aggregate. A SINGLE difference, or a
check that cannot run, sets ``root_invalidation`` and writes a durable, write-once
``ROOT_INVALIDATED__<unit>.json`` marker in the run root: the whole root is
invalid, not the row. The non-causal oracle control must be DETECTED on every
evaluated unit, else the root is invalid too. A unit whose longest complete TRAIN
stretch is under ``MIN_FIT_ROWS`` (design ``rules.missing_data``) has every arm,
controls included, REFUSED with that one reason: it is not an evaluated unit, it
counts as abstention, and its unevaluated oracle does not invalidate the root
(``unit_evaluable`` false in the summary).

SNR facts come from ``df_snr``'s own CLI (``--d2-unit-facts``) in a child
process: this module never loads ``df_snr`` and never names its offline kernel
entry (C160 isolation, checked by ``df_causal_battery.snr_isolation_evidence``).

Memory estimate (``estimate_unit_memory_bytes``, bytes):
    2*BASE_PROCESS_BYTES (worker and df_snr child) + T*V*(8*6) + n_arms*T*V*(8*4 + 72) + haar_arms*T*V*(8*6 + 72)*3
(arrays of the unit; per arm Y, available, 72-byte reason strings and metric
copies; the audit holds a full run, a prefix run and an adversarial run at once).

Child protocol: ``--worker job.json`` writes ``heartbeat.json``, the rows into
``d2_unit_rows.jsonl`` (``{"table", "row"}`` lines) and ``result.json`` exactly as
``df_isolated_runner.classify`` expects; ``run_units`` is the parent that
launches one process per unit under hard limits and writes durable terminals.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE_PROCESS_BYTES = 450 * (1 << 20)
MIN_FIT_ROWS = 50                             # design rules.missing_data: longest complete TRAIN stretch
OUTPUT_FILE = "d2_unit_rows.jsonl"
AUDIT_FILE = "wavelet_audit.json"
DENOISING_TABLE = "df_fact_d2_unit_denoising"
SNR_TABLE = "df_fact_d2_unit_snr"
ROW_STATUSES = ("COMPLETED", "INCONCLUSIVE", "UNAVAILABLE", "NOT_APPLICABLE", "REFUSED", "FAILED")
BRANCH_OF = {"rmse_raw": "RAW", "snr_raw_db": "RAW", "impulse_retention_raw": "RAW", "motif_corr_raw": "RAW",
             "extreme_retention_raw": "RAW",
             "rmse_denoised": "TRANSFORMED", "snr_denoised_db": "TRANSFORMED", "amplitude_ratio": "TRANSFORMED",
             "distortion_ratio": "TRANSFORMED", "delay_samples": "TRANSFORMED", "delay_alignment_corr": "TRANSFORMED",
             "residual_signal_share": "RESIDUAL", "residual_excess_acf1": "RESIDUAL",
             "snr_improvement_db": "COMPARISON", "rmse_ratio": "COMPARISON", "support": "COMPARISON",
             "noise_free": "COMPARISON", "signal_free": "COMPARISON"}
RESUME_SKIP_STATUSES = ("COMPLETED", "REFUSED")


def _load(name: str, directory: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _load("df_d2_design")
OPS = _load("df_operators")
SNAP = _load("df_snapshot")
SYNC = _load("df_synthetic_contract")
LAB = _load("df_lab_evaluation")
IR = _load("df_isolated_runner")


class UnitRefusal(ValueError):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def estimate_unit_memory_bytes(T: int, V: int, n_arms: int, haar_arms: int) -> int:
    return int(2 * BASE_PROCESS_BYTES + T * V * 48 + n_arms * T * V * (32 + 72) + haar_arms * T * V * (48 + 72) * 3)


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


# ----------------------------------------------------------- public path
def fit_public(u: dict, spec: dict, mode: str, fit_sl: slice) -> dict:
    snap = SNAP.FitSnapshot.from_contract(u["contract"], "TRAIN", u["loader"], start=fit_sl.start, end=fit_sl.stop)
    return OPS.fit(spec, snap, mode)


def licensed_start(fitted: dict) -> int:
    return int(fitted["fit_binding"]["licensed_min_index"])


def transform_public(u: dict, fitted: dict, start: int, end: int, transform_batch=None) -> tuple:
    transform_batch = transform_batch or OPS.transform_batch
    tsnap = SNAP.TransformSnapshot.from_contract(u["contract"], u["loader"], start=start, end=end)
    oracle = fitted["spec"]["kind"] in OPS.NON_CAUSAL_KINDS
    return transform_batch(fitted, tsnap, oracle_mode=oracle)


def _full(u, y_part, a_part, r_part, start):
    shape = u["observed"].shape
    y = np.full(shape, np.nan)
    a = np.zeros(shape, dtype=bool)
    r = np.full(shape, "NOT_LICENSED", dtype="<U18")
    y[start:], a[start:], r[start:] = y_part, a_part, r_part
    return y, a, r


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind == "f" else bool(np.array_equal(a, b))


# ----------------------------------------------------------- oracle check
def detect_non_causal(u: dict, fitted: dict, n_cuts: int = 12, transform_batch=None) -> dict:
    """Public-API prefix check: DETECTED when a prefix run differs from the full run on rows <= t."""
    T = u["observed"].shape[0]
    s = licensed_start(fitted)
    y, a, r = transform_public(u, fitted, s, T, transform_batch)
    cuts = sorted({int(c) for c in np.linspace(s + 1, T - 2, n_cuts)})
    for t in cuts:
        yp, ap, rp = transform_public(u, fitted, s, t + 1, transform_batch)
        k = t + 1 - s
        if not (_same(y[:k], yp) and _same(a[:k], ap) and _same(r[:k], rp)):
            return {"outcome": "DETECTED", "first_cut": t, "cuts_tested": cuts.index(t) + 1}
    return {"outcome": "NOT_DETECTED", "first_cut": None, "cuts_tested": len(cuts)}


# ------------------------------------------------------------ wavelet audit
def _haar_levels_window(w: np.ndarray, lam: np.ndarray, levels: int):
    """Audit-local reference on one window (L, V) ending at t: per level soft detail and
    approximation, and y = c_J + sum soft(d_j). Only rows of the window are read."""
    c = w.T                                   # (V, L)
    acc = np.zeros(c.shape[0])
    out = []
    for j in range(levels):
        step = 2 ** j
        cn = (c[..., step:] + c[..., :-step]) * 0.5
        d = c[..., -1] - cn[..., -1]
        sd = np.sign(d) * np.maximum(np.abs(d) - lam[:, j], 0.0)
        acc = acc + sd
        out.append({"detail": sd, "approx": cn[..., -1].copy()})
        c = cn
    return c[..., -1] + acc, out


def _boundaries(s: int, T: int, L: int) -> list:
    marks = set()
    j = 0
    while 2 ** j <= T:
        for base in (0, s):
            marks.update({base + 2 ** j - 1, base + 2 ** j, base + 2 ** j + 1})
        j += 1
    marks.update({s, s + L - 2, s + L - 1, s + L, T - 2})
    return sorted(m for m in marks if s <= m <= T - 2)


def audit_trailing_haar(u: dict, fitted: dict, *, transform_batch=None, n_random_cuts: int = 8,
                        chunk_sizes=(2, 3, 7, 64), seed: int = 20260913, wavelet_mad_probe: dict | None = None) -> dict:
    """C177 on one evaluated unit. `transform_batch` defaults to the public one; tests inject a stand-in."""
    transform_batch = transform_batch or OPS.transform_batch
    BAT = _load("df_causal_battery")
    spec = fitted["spec"]
    levels = spec["params"]["levels"]
    L = 2 ** levels
    lam = np.asarray(fitted["fitted"]["thresholds"])
    T = u["observed"].shape[0]
    s = licensed_start(fitted)
    checks: dict = {}

    def record(name, passed, detail="", **extra):
        checks[name] = dict({"passed": bool(passed), "detail": detail}, **extra)

    meta = fitted["meta"]
    record("no_delay_compensation_declared", meta["future_shift_compensation"] == 0 and meta["look_ahead"] == 0
           and meta["lookback"] == L - 1, f"shift={meta['future_shift_compensation']} look_ahead={meta['look_ahead']}")
    try:
        y, a, r = transform_public(u, fitted, s, T, transform_batch)
        full = SNAP.TransformSnapshot.from_contract(u["contract"], u["loader"], start=s, end=T)
        X = np.asarray(full.matrix)
        # every prefix, every level
        first_prefix = first_level = None
        max_recon = 0.0
        for t in range(s, T):
            k = t + 1 - s
            snap_p = SNAP.TransformSnapshot.from_contract(u["contract"], u["loader"], start=s, end=t + 1)
            yp, ap, rp = transform_batch(fitted, snap_p)
            if first_prefix is None and not (_same(y[:k], yp) and _same(a[:k], ap) and _same(r[:k], rp)):
                first_prefix = t
            if k >= L:
                yr, lv_p = _haar_levels_window(np.asarray(snap_p.matrix)[-L:], lam, levels)
                _, lv_f = _haar_levels_window(X[k - L:k], lam, levels)
                for j in range(levels):
                    for part in ("detail", "approx"):
                        if first_level is None and not _same(lv_p[j][part], lv_f[j][part]):
                            first_level = (t, j + 1, part)
                ok = a[k - 1]
                if ok.any():
                    diff = np.abs(yr[ok] - y[k - 1][ok])
                    tol = 1e-12 * np.maximum(1.0, np.abs(y[k - 1][ok]))
                    max_recon = max(max_recon, float(diff.max()))
                    if first_level is None and bool(np.any(diff > tol)):
                        first_level = (t, 0, "reconstruction_vs_public_output")
                win_nan = np.isnan(np.asarray(snap_p.matrix)[-L:]).any(axis=0)
                if first_level is None and not _same(a[k - 1], ~win_nan & (k - 1 >= L - 1)):
                    first_level = (t, 0, "availability_vs_window")
        record("prefix_every_t", first_prefix is None,
               "" if first_prefix is None else f"prefix run differs from the full run at t={first_prefix}",
               prefixes=T - s)
        record("levels_every_t", first_level is None,
               "" if first_level is None else f"level check failed at {first_level}",
               levels=levels, max_reconstruction_abs_diff=max_recon)

        # adversarial suffixes through a public-API runner on in-memory contracts of the same length
        parts = {k: list(v) for k, v in u["rec"]["partitions"].items()}
        fit_sl = LAB.train_fit_slice(u["observed"], parts["train"])

        def run(X2):
            contract, blobs = SYNC.in_memory_contract(X2, name="d2_wavelet_audit")
            if contract["partitions"]["boundaries"] != parts:
                raise UnitRefusal("in-memory audit contract partitions differ from the unit")
            fsnap = SNAP.FitSnapshot.from_contract(contract, "TRAIN", blobs, start=fit_sl.start, end=fit_sl.stop)
            f2 = OPS.fit(spec, fsnap, fitted["fit_mode"])
            if f2["fitted"] != fitted["fitted"]:
                raise UnitRefusal("audit fit differs from the evaluated fit (train rows changed)")
            ts = SNAP.TransformSnapshot.from_contract(contract, blobs, start=s, end=T)
            y2, a2, r2 = transform_batch(f2, ts)
            vy = np.full(X2.shape, np.nan)
            va = np.zeros(X2.shape, dtype=bool)
            vr = np.full(X2.shape, "NOT_LICENSED", dtype="<U18")
            vy[s:], va[s:], vr[s:] = y2, a2, r2
            return {"y": vy}, va, vr

        base_vals, base_a, base_r = run(u["observed"])
        record("in_memory_baseline_equals_unit", _same(base_vals["y"][s:], y) and _same(base_a[s:], a)
               and _same(base_r[s:], r))
        rng = np.random.default_rng(seed)
        cuts = sorted(set(_boundaries(s, T, L)) | {int(v) for v in rng.integers(s, T - 1, size=n_random_cuts)})
        scale = float(np.nanstd(u["observed"])) or 1.0
        fails = BAT.suffix_adversarial(run, np.array(u["observed"]), cuts, max(1.0, scale))
        record("suffix_adversarial", not fails, "; ".join(f"{g}: {m}" for g, m in fails.items()),
               cuts=cuts, suffixes=list(BAT.SUFFIXES))

        # step, chunk and restart through the public stream API
        n = T - s
        restarts = [k for k in _boundaries(0, n + 1, L) if 0 < k < n] + [k for k in (L - 2, L - 1, L) if 0 < k < n]
        restarts = sorted(set(restarts))
        st = OPS.init_state(fitted, full)
        blobs_at = {}
        step_bad = None
        for i in range(n):
            if i in restarts:
                blobs_at[i] = OPS.save_state(st)
            yi, ai, ri, st = OPS.step(fitted, st, full.row(i))
            if step_bad is None and not (_same(yi, y[i]) and _same(ai, a[i]) and _same(ri, r[i])):
                step_bad = s + i
        record("batch_equals_step", step_bad is None, "" if step_bad is None else f"step differs at row {step_bad}")
        chunk_bad = None
        for c in chunk_sizes:
            st = OPS.init_state(fitted, full)
            ys, as_, rs = [], [], []
            for k0 in range(s, T, c):
                sn = SNAP.TransformSnapshot.from_contract(u["contract"], u["loader"], start=k0, end=min(T, k0 + c))
                yc, ac, rc, st = OPS.transform_chunk(fitted, st, sn)
                ys.append(yc), as_.append(ac), rs.append(rc)
            if not (_same(np.concatenate(ys), y) and _same(np.concatenate(as_), a) and _same(np.concatenate(rs), r)):
                chunk_bad = chunk_bad or c
        record("batch_equals_chunks", chunk_bad is None, "" if chunk_bad is None else f"chunk size {chunk_bad} differs",
               chunk_sizes=list(chunk_sizes))
        restart_bad = None
        for k, blob in sorted(blobs_at.items()):
            st = OPS.load_state(blob, fitted)
            sn = SNAP.TransformSnapshot.from_contract(u["contract"], u["loader"], start=s + k, end=T)
            yc, ac, rc, st = OPS.transform_chunk(fitted, st, sn)
            if not (_same(yc, y[k:]) and _same(ac, a[k:]) and _same(rc, r[k:])):
                restart_bad = restart_bad or s + k
        record("save_load_restart", restart_bad is None,
               "" if restart_bad is None else f"restart at row {restart_bad} differs", restarts=[s + k for k in restarts])
        warm = [s + L - 2, s + L - 1, s + L]
        record("warmup_and_power_of_two_boundaries_covered",
               all(b in cuts for b in warm if b <= T - 2) and bool(restarts),
               boundaries=_boundaries(s, T, L))
    except Exception as exc:  # noqa: BLE001 - an audit that cannot run cannot certify
        record("audit_ran", False, f"{type(exc).__name__}: {exc}"[:400])
    iso = BAT.snr_isolation_evidence(HERE)
    probe = wavelet_mad_probe or {}
    ok = bool(probe.get("passed")) and bool(iso["passed"])
    record("wavelet_mad_train_aggregate_only", ok,
           "" if ok else f"probe={probe or 'NOT_SUPPLIED'} isolation_passed={iso['passed']}", probe=probe,
           isolation={k: iso[k] for k in ("modules_loading_df_snr", "kernel_identifier_references_outside_df_snr")})
    failed = sorted(k for k, v in checks.items() if not v["passed"])
    return {"operator_kind": spec["kind"], "operator_params": spec["params"], "fit_mode": fitted["fit_mode"],
            "fitted_sha256": fitted["artifact_sha256"], "licensed_start": s, "n_rows": T, "checks": checks,
            "failed_checks": failed, "root_invalidation": bool(failed)}


def run_snr_facts(unit_dir: Path, design: dict, work_dir: Path, timeout: float = 3600.0) -> dict:
    """df_snr's own CLI in a child process (C160: this module never loads df_snr)."""
    import subprocess
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    job, out = work_dir / "snr_facts_job.json", work_dir / "snr_facts.json"
    if out.exists():
        raise UnitRefusal(f"{out.name} exists; SNR facts are write-once")
    job.write_text(json.dumps({"unit_dir": str(unit_dir), "estimators": design["snr"]["estimators"],
                               "partitions": design["snr"]["partitions"], "bootstrap": design["snr"].get("bootstrap"),
                               "out": str(out)}, sort_keys=True))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(IR.CHILD_ENV)
    p = subprocess.run([sys.executable, "-B", str(HERE / "df_snr.py"), "--d2-unit-facts", str(job)],
                       capture_output=True, text=True, timeout=timeout, env=env)
    if p.returncode != 0 or not out.is_file():
        raise RuntimeError(f"df_snr unit facts failed (exit {p.returncode}): {p.stderr[-400:]}")
    return json.loads(out.read_text())


# --------------------------------------------------------------- rows
def _base(ctx: dict, op: dict, v: int, **kw) -> dict:
    spec = op["spec"]
    row = {"run_id": ctx["run_id"], "mode": ctx["mode"], "design_sha256": ctx["design_sha256"],
           "tape_sha256": ctx["tape_sha256"], "unit_id": ctx["unit_id"], "seed": ctx["seed"],
           "content_sha256": ctx["content_sha256"], "regime": ctx["regime"], "variable_id": ctx["variable_ids"][v],
           "variable_index": v, "arm_role": op["arm_role"], "operator_kind": spec["kind"],
           "operator_params": spec["params"], "spec_sha256": op["spec_sha256"], "fit_mode": op["fit_mode"],
           "fitted_sha256": None, "partition": "", "branch": "", "metric": "", "estimator": "", "value": None,
           "status": "COMPLETED", "reason": "", "code_sha256": ctx["code_sha256"],
           "operator_code_sha256": ctx["operator_code_sha256"]}
    row.update(kw)
    return row


def _raw_extreme_retention(c, x, d, avail, lo, hi):
    cs, xs, ds, av = c[lo:hi], x[lo:hi], d[lo:hi], avail[lo:hi]
    sup = av & np.isfinite(xs) & np.isfinite(ds)
    if sup.sum() < LAB.MIN_SUPPORT:
        return None
    cc, xx = cs[sup], xs[sup]
    if float(np.var(cc)) <= 1e-15:
        return None
    med = float(np.median(cc))
    dev = np.abs(cc - med)
    sel = dev >= np.quantile(dev, 0.99)
    if dev[sel].mean() <= 1e-12:
        return None
    return float(np.mean(np.abs(xx[sel] - med)) / np.mean(dev[sel]))


def denoising_rows(ctx: dict, u: dict, op: dict, fitted, y, avail, cpu_fit: float, cpu_tr: float,
                   status: str, reason: str) -> list:
    rec = u["rec"]
    out = []
    T = u["observed"].shape[0]
    fsha = fitted["artifact_sha256"] if isinstance(fitted, dict) and "artifact_sha256" in fitted else None
    for v in range(rec["n_variables"]):
        if status != "COMPLETED":
            for p in LAB.EVAL_PARTITIONS:
                out.append(_base(ctx, op, v, fitted_sha256=fsha, partition=p, branch="COMPARISON",
                                 metric="arm_status", estimator="d2.arm_status", status=status,
                                 reason=reason or status))
            continue
        evs = [e for e in u["events"] if int(e.get("variable", 0)) == v]
        for p in LAB.EVAL_PARTITIONS:
            lo, hi = rec["partitions"][p]
            av = avail[:, v] & ~u["mask"][:, v]
            pm = LAB.partition_metrics(u["clean"][:, v], u["noise"][:, v], u["observed"][:, v], y[:, v], av, lo, hi, evs)
            if pm["status"] != "COMPLETED":
                out.append(_base(ctx, op, v, fitted_sha256=fsha, partition=p, branch="COMPARISON",
                                 metric="partition_support", estimator="available_not_missing",
                                 status="UNAVAILABLE", reason=pm["reason"]))
                continue
            pm["extreme_retention_raw"] = _raw_extreme_retention(u["clean"][:, v], u["observed"][:, v], y[:, v], av,
                                                                 lo, hi)
            for k, val in pm.items():
                if k in ("status", "reason"):
                    continue
                if isinstance(val, bool):
                    val = 1.0 if val else 0.0
                num = _num(val)
                branch = "COMPARISON" if k.endswith("__events") else BRANCH_OF.get(k, "TRANSFORMED")
                out.append(_base(ctx, op, v, fitted_sha256=fsha, partition=p, branch=branch, metric=k,
                                 estimator=f"d2v2.{k}", value=num,
                                 status="COMPLETED" if num is not None else "INCONCLUSIVE",
                                 reason="" if num is not None else "undefined for this unit"))
        for metric, val in (("cpu_seconds_fit", cpu_fit), ("cpu_seconds_transform", cpu_tr),
                            ("cpu_seconds_per_1000_samples", (cpu_fit + cpu_tr) / T * 1000.0)):
            out.append(_base(ctx, op, v, fitted_sha256=fsha, partition="all", branch="COST", metric=metric,
                             estimator="process_time", value=float(val) / rec["n_variables"]))
    return out


def snr_rows(ctx: dict, facts_doc: dict) -> list:
    out = []
    for f in facts_doc["facts"]:
        v, name, state, p = f["variable_index"], f["estimator"], f["contract_state"], f["partition"]
        if f["status"] in ("NOT_APPLICABLE", "FAILED"):
            out.append(_snr_row(ctx, v, name, state, p, None, None, None, None, f["status"], f["reason"] or f["status"]))
            continue
        res = f["result"]
        a, b = res["segment_used"]
        out.append(_snr_row(ctx, v, name, state, p, res, f["true_snr_db"], a, b,
                            "COMPLETED" if res["status"] == "ESTIMATED" else "INCONCLUSIVE", res["reason"] or ""))
    return out


def _snr_row(ctx, v, name, contract_state, partition, res, true, a, b, status, reason) -> dict:
    bsr = (res or {}).get("bootstrap") or {}
    hat = None if res is None else _num(res.get("snr_db"))
    err = None if hat is None or true is None else hat - true
    lo = -math.inf if bsr.get("lower_unbounded") else bsr.get("ci_low_db")
    hi = math.inf if bsr.get("upper_unbounded") else bsr.get("ci_high_db")
    covers = None
    if hat is not None and true is not None and bsr.get("n_replicates_used") and lo is not None and hi is not None:
        covers = 1.0 if lo <= true <= hi else 0.0
    ident = ("NOT_APPLICABLE" if status in ("NOT_APPLICABLE", "FAILED") else
             ("ESTIMATED" if res and res["status"] == "ESTIMATED" else "NOT_IDENTIFIABLE"))
    if status == "COMPLETED" and true is None:
        status, reason = "INCONCLUSIVE", "true SNR is not finite on this segment"
    return {"run_id": ctx["run_id"], "mode": ctx["mode"], "design_sha256": ctx["design_sha256"],
            "tape_sha256": ctx["tape_sha256"], "unit_id": ctx["unit_id"], "seed": ctx["seed"],
            "content_sha256": ctx["content_sha256"], "regime": ctx["regime"], "variable_index": v, "estimator": name,
            "contract_state": contract_state, "partition": partition,
            "segment_start": a, "segment_end": b, "snr_db_hat": hat, "ci_low_db": _num(bsr.get("ci_low_db")),
            "ci_high_db": _num(bsr.get("ci_high_db")), "ci_lower_unbounded": bool(bsr.get("lower_unbounded", False)),
            "ci_upper_unbounded": bool(bsr.get("upper_unbounded", False)), "true_snr_db": _num(true),
            "error_db": _num(err), "abs_error_db": None if err is None else abs(err), "ci_covers_true": covers,
            "identifiability": ident, "status": status, "reason": reason if status != "COMPLETED" else "",
            "code_sha256": ctx["code_sha256"]}


# ------------------------------------------------------------ one unit
def _unit_identity(u: dict, job: dict, design: dict) -> tuple:
    rec = u["rec"]
    mode = job["mode"]
    if mode not in D.MODES:
        raise UnitRefusal(f"unknown mode {mode!r}")
    unit_dir = Path(job["unit_dir"])
    tape_sha = None
    if mode == D.FRESH_MODE:
        manifest = json.loads((unit_dir.parent / "ROOT_MANIFEST.json").read_text())
        tape = json.loads((unit_dir.parent / "SEED_TAPE.json").read_text())
        TAPE = _load("df_seed_tape")
        problems = TAPE.verify_tape(tape, design)
        if problems or manifest.get("mode") != D.FRESH_MODE or manifest.get("tape_sha256") != tape["tape_sha256"]:
            raise UnitRefusal(f"fresh unit is not bound to a sealed tape of this design: {problems[:2]}")
        rk = D.regime_key(LAB.regime_of(rec))
        seeds = {s["seed"] for r in tape["regimes"] if r["regime_key"] == rk for s in r["seeds"]}
        if rec["seed"] not in seeds:
            raise UnitRefusal(f"seed {rec['seed']} of {rec['unit_id']} is not on the tape for its regime")
        tape_sha = tape["tape_sha256"]
    elif (unit_dir.parent / "ROOT_MANIFEST.json").is_file():
        raise UnitRefusal("a fresh reserve unit is never re-labelled as historical reanalysis")
    return mode, tape_sha


def process_unit(job: dict, writer_write, heartbeat=None, *, transform_batch=None) -> dict:
    """Evaluate one unit; rows go to writer_write({"table", "row"}). -> summary dict."""
    design = D.require_valid(job["design"])
    if design["design_sha256"] != job["design_sha256"]:
        raise UnitRefusal("job design digest differs")
    u = LAB.load_unit(Path(job["unit_dir"]))
    rec = u["rec"]
    mode, tape_sha = _unit_identity(u, job, design)
    ctx = {"run_id": job["run_id"], "mode": mode, "design_sha256": design["design_sha256"], "tape_sha256": tape_sha,
           "unit_id": rec["unit_id"], "seed": int(rec["seed"]), "content_sha256": u["content_sha256"],
           "regime": LAB.regime_of(rec), "variable_ids": u["variable_ids"], "code_sha256": D.lab_code_sha256(),
           "operator_code_sha256": OPS.code_sha256()}
    fit_sl = LAB.train_fit_slice(u["observed"], rec["partitions"]["train"])
    # The declared missing-data rule refuses every arm of a unit whose longest complete TRAIN stretch is short; such
    # a unit is not evaluated (C177 audits the units really evaluated) and counts as abstention, never as a pass.
    unit_evaluable = fit_sl.stop - fit_sl.start >= MIN_FIT_ROWS
    T = u["observed"].shape[0]
    counts = {"rows": 0, "completed": 0, "missing": 0}
    snr_doc = run_snr_facts(Path(job["unit_dir"]), design, Path(job["attempt_dir"]) / "snr")
    audits, oracle = [], []

    def emit(table, rows):
        for r in rows:
            writer_write({"table": table, "row": r})
            counts["rows"] += 1
            counts["completed" if r["status"] == "COMPLETED" else "missing"] += 1

    for op in design["operators"]:
        spec = op["spec"]
        if heartbeat:
            heartbeat.beat(module="df_d2_unit_worker", metric=spec["kind"], variable=json.dumps(spec["params"]),
                           rows_written=counts["rows"])
        t0 = time.process_time()
        fitted, y, avail, cpu_fit = None, None, None, 0.0
        try:
            if not unit_evaluable:
                raise OPS.OperatorRefusal(f"longest complete train stretch has {fit_sl.stop - fit_sl.start} rows")
            fitted = fit_public(u, spec, op["fit_mode"], fit_sl)
            cpu_fit = time.process_time() - t0
            if fitted["status"] != "FITTED":
                raise OPS.OperatorAbstain(f"ABSTAIN: {fitted['abstain_reason']}")
            s = licensed_start(fitted)
            yp, ap, rp = transform_public(u, fitted, s, T, transform_batch)
            y, avail, _ = _full(u, yp, ap, rp, s)
            status, reason = "COMPLETED", ""
        except Exception as exc:  # noqa: BLE001 - every outcome is recorded
            status = "REFUSED" if isinstance(exc, (OPS.OperatorRefusal, SNAP.SnapshotRefusal)) else "FAILED"
            reason = f"{type(exc).__name__}: {exc}"[:300]
        cpu_tr = time.process_time() - t0 - cpu_fit
        emit(DENOISING_TABLE, denoising_rows(ctx, u, op, fitted, y, avail, cpu_fit, cpu_tr, status, reason))
        if status != "COMPLETED":
            if op["arm_role"] == "NON_CAUSAL_ORACLE_CONTROL":
                oracle.append({"spec_sha256": op["spec_sha256"], "outcome": "NOT_EVALUATED", "reason": reason})
            if spec["kind"] == "trailing_haar_threshold" and status == "FAILED":
                audits.append({"operator_kind": spec["kind"], "operator_params": spec["params"],
                               "failed_checks": ["fit_or_transform_failed"], "root_invalidation": True,
                               "checks": {"fit_or_transform_failed": {"passed": False, "detail": reason}}})
            continue
        if op["arm_role"] == "NON_CAUSAL_ORACLE_CONTROL":
            det = detect_non_causal(u, fitted, transform_batch=transform_batch)
            oracle.append(dict(det, spec_sha256=op["spec_sha256"]))
        if spec["kind"] == "trailing_haar_threshold":
            audits.append(audit_trailing_haar(u, fitted, transform_batch=transform_batch,
                                              wavelet_mad_probe=snr_doc["wavelet_mad_probe"]))
    if heartbeat:
        heartbeat.beat(module="df_d2_unit_worker", metric="snr", variable=None, rows_written=counts["rows"])
    emit(SNR_TABLE, snr_rows(ctx, snr_doc))
    invalid = [a for a in audits if a["root_invalidation"]]
    reasons = [f"WAVELET_AUDIT {a['operator_params']} failed {a['failed_checks']}" for a in invalid]
    if unit_evaluable:
        reasons += [f"ORACLE_{o['outcome']}" for o in oracle if o["outcome"] != "DETECTED"]
    n_haar = sum(op["spec"]["kind"] == "trailing_haar_threshold" for op in design["operators"])
    return {"unit_id": rec["unit_id"], "seed": int(rec["seed"]), "mode": mode, "content_sha256": u["content_sha256"],
            "contract_sha256": u["contract"]["contract_sha256"], "dataset_id": u["contract"]["dataset_id"],
            "n_variables": rec["n_variables"], "rows": counts["rows"], "rows_completed": counts["completed"],
            "rows_not_completed": counts["missing"], "wavelet_audits": audits, "oracle_detection": oracle,
            "unit_evaluable": unit_evaluable, "fit_stretch_rows": fit_sl.stop - fit_sl.start,
            "root_invalidation": bool(reasons), "root_invalidation_reasons": reasons,
            "memory_estimate_bytes": estimate_unit_memory_bytes(T, rec["n_variables"], len(design["operators"]),
                                                                n_haar)}


def invalidate_root(root_dir: Path, summary: dict) -> Path:
    """A hard, durable, write-once marker: one difference invalidates the whole root."""
    path = Path(root_dir) / f"ROOT_INVALIDATED__{summary['unit_id']}.json"
    doc = {"schema": "crispdm.data_foundation.d2_root_invalidation.v1", "unit_id": summary["unit_id"],
           "mode": summary["mode"], "reasons": summary["root_invalidation_reasons"],
           "wavelet_audits": [{k: a[k] for k in ("operator_params", "failed_checks")} for a in summary["wavelet_audits"]
                              if a["root_invalidation"]],
           "oracle_detection": summary["oracle_detection"], "scope": "WHOLE_ROOT"}
    if not path.exists():
        IR.atomic_write_once(path, json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    return path


def worker_main(job_file: Path) -> int:
    PR = _load("df_profile_run")
    job = json.loads(Path(job_file).read_text())
    job["design"] = json.loads(Path(job["design_file"]).read_text())
    adir = Path(job["attempt_dir"])
    hb = PR.Heartbeat(adir / "heartbeat.json", job.get("unit_id") or "UNRESOLVED", job.get("heartbeat_seconds", 10.0))
    result = {"status": "FAILED", "reason": "", "rows_written": 0, "output_file": None, "output_sha256": None}
    writer = None
    exit_code = 2
    try:
        writer = PR.JsonlWriter(adir / OUTPUT_FILE)
        summary = process_unit(job, lambda obj: writer.write(obj), hb)
        (adir / AUDIT_FILE).write_text(json.dumps({"wavelet_audits": summary["wavelet_audits"],
                                                   "oracle_detection": summary["oracle_detection"]},
                                                  indent=1, sort_keys=True, default=str))
        if summary["root_invalidation"]:
            invalidate_root(Path(job["root_dir"]), summary)
        result.update({k: v for k, v in summary.items() if k != "wavelet_audits"})
        result["rows_written"] = writer.rows
        result["output_sha256"] = writer.close()
        result["output_file"] = OUTPUT_FILE
        result.update(status="COMPLETED", reason="")
        exit_code = 0
    except (UnitRefusal, D.DesignRefusal, _load("df_contract").ContractRefusal) as exc:
        result.update(status="REFUSED", reason=f"{type(exc).__name__}: {exc}"[:500])
        exit_code = 3
    except MemoryError as exc:
        result.update(status="FAILED", reason=f"MemoryError: {exc}"[:500], exception_type="MemoryError")
        exit_code = 4
    except Exception as exc:  # noqa: BLE001
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}"[:500], exception_type=type(exc).__name__)
    finally:
        if writer is not None and not writer.f.closed:
            writer.abandon()
        hb.stop()
    result["child_maxrss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    text = json.dumps(result, sort_keys=True, default=str).replace(str(Path.home()), "~")
    tmp = adir / "result.json.tmp"
    with open(tmp, "w") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, adir / "result.json")
    return exit_code


# ------------------------------------------------------------------ parent
def run_units(out_dir: Path, design_file: Path, unit_dirs: list, mode: str, *, host_role: str = "COORDINATOR",
              task_memory_bytes: int = 3 << 30, wall_seconds: float = 3600, cpu_seconds: int = 3600,
              heartbeat_seconds: float = 10.0, resume: bool = False, mechanism: str | None = None,
              slice_: str | None = None) -> dict:
    """One process per unit under hard limits; durable parent-written terminals; resume by identity."""
    PR = _load("df_profile_run")
    out_dir = Path(out_dir)
    design = D.require_valid(json.loads(Path(design_file).read_text()))
    code = D.lab_code_sha256()
    if host_role not in IR.HOST_ROLES:
        raise UnitRefusal(f"host_role must be one of {IR.HOST_ROLES}")
    if out_dir.exists():
        if not resume:
            raise UnitRefusal(f"{out_dir.name} exists; D2 roots are write-once")
        manifest = json.loads((out_dir / "RUN_MANIFEST.json").read_text())
        if manifest["mode"] != mode or manifest["design_sha256"] != design["design_sha256"]:
            raise UnitRefusal("resume must keep the mode and the design")
    else:
        (out_dir / "terminals").mkdir(parents=True)
        (out_dir / "attempts").mkdir()
        manifest = {"schema": "crispdm.data_foundation.d2_run_manifest.v1", "mode": mode,
                    "design_sha256": design["design_sha256"], "host_role": host_role, "code_sha256_at_creation": code,
                    "run_id": "d2v2_" + hashlib.sha256(f"{out_dir.name}|{code}|{mode}|{IR.now_iso()}".encode()).hexdigest()[:24]}
        IR.atomic_write_once(out_dir / "RUN_MANIFEST.json", json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    run_id = manifest["run_id"]
    slice_ = slice_ or IR.DEFAULT_SLICE
    mechanism = mechanism or IR.detect_mechanism(slice_)
    label = IR.mechanism_label(mechanism, slice_)
    results = []
    for ud in unit_dirs:
        ud = Path(ud)
        sname = PR.safe_name(ud.name)
        prior = PR._terminals(out_dir, sname)
        try:
            contract = SYNC.unit_contract(ud)
            ds, csha = contract["dataset_id"], contract["contract_sha256"]
        except Exception as exc:  # noqa: BLE001
            ds, csha = f"UNRESOLVED:{ud.name}", None
            refusal = f"{type(exc).__name__}: {exc}"[:400]
        else:
            refusal = None
        if prior and prior[-1][1]["status"] in RESUME_SKIP_STATUSES and prior[-1][1]["contract_sha256"] == csha \
                and prior[-1][1]["code_sha256"] == code:
            results.append({"unit": ud.name, "status": prior[-1][1]["status"], "resumed_skip": True})
            continue
        attempt = len(prior) + 1
        adir = out_dir / "attempts" / sname / f"attempt-{attempt}"
        adir.mkdir(parents=True)
        base = {k: None for k in IR.TERMINAL_KEYS}
        base.update(run_id=run_id, host_role=host_role, bank="SYNTHETIC", dataset_id=ds, contract_sha256=csha,
                    code_sha256=code, rows_written=0, variables_profiled=0, metrics_completed=0, metrics_missing=0,
                    planned_peak_bytes=0, memory_limit_bytes=IR.limits_for(task_memory_bytes)["memory_limit_bytes"],
                    limit_mechanism=label, wall_seconds=0.0, cpu_seconds=0.0, started_at=IR.now_iso(),
                    ended_at=IR.now_iso(), reason="")
        tpath = out_dir / "terminals" / f"{sname}.attempt-{attempt}.json"
        if refusal is not None:
            IR.write_terminal(tpath, dict(base, status="REFUSED", reason=refusal.replace(str(Path.home()), "~")))
            results.append({"unit": ud.name, "status": "REFUSED"})
            continue
        rec = json.loads((ud / "UNIT.json").read_text())
        n_haar = sum(o["spec"]["kind"] == "trailing_haar_threshold" for o in design["operators"])
        planned = estimate_unit_memory_bytes(rec["n_samples"], rec["n_variables"], len(design["operators"]), n_haar)
        lim = IR.limits_for(task_memory_bytes)
        if planned > lim["budget_bytes"]:
            IR.write_terminal(tpath, dict(base, status="REFUSED", reason="UNIT_MEMORY_ESTIMATE_EXCEEDS_TASK_BUDGET",
                                          planned_peak_bytes=planned))
            results.append({"unit": ud.name, "status": "REFUSED"})
            continue
        job = {"design_file": str(Path(design_file).resolve()), "design_sha256": design["design_sha256"],
               "unit_dir": str(ud.resolve()), "attempt_dir": str(adir.resolve()), "root_dir": str(out_dir.resolve()),
               "run_id": run_id, "mode": mode, "unit_id": ud.name, "heartbeat_seconds": heartbeat_seconds}
        (adir / "job.json").write_text(json.dumps(job, sort_keys=True))
        task = IR.Task(argv=[sys.executable, "-B", str(HERE / "df_d2_unit_worker.py"), "--worker",
                             str(adir / "job.json")], name=sname[:40], attempt_dir=adir,
                       assigned_bytes=task_memory_bytes, wall_seconds=wall_seconds, cpu_seconds=cpu_seconds,
                       slice_=slice_, mechanism=mechanism).start()
        o = task.wait()
        status, reason, ver = IR.classify(o, adir)
        res = o["result"] or {}
        observed = res.get("child_maxrss_bytes") or o["cgroup_memory_peak"] or o["child_maxrss_bytes"] or None
        IR.write_terminal(tpath, dict(
            base, status=status, reason=reason.replace(str(Path.home()), "~"), rows_written=ver["rows_written"],
            variables_profiled=int(res.get("n_variables") or 0), metrics_completed=int(res.get("rows_completed") or 0),
            metrics_missing=int(res.get("rows_not_completed") or 0), planned_peak_bytes=planned,
            observed_peak_rss_bytes=int(observed) if observed else None, wall_seconds=o["wall_seconds"],
            cpu_seconds=o["cpu_seconds"], output_file=(str((adir / OUTPUT_FILE).relative_to(out_dir))
                                                       if status == "COMPLETED" else None),
            output_sha256=ver["output_sha256"], started_at=o["started_at"], ended_at=o["ended_at"]))
        results.append({"unit": ud.name, "status": status, "root_invalidation": res.get("root_invalidation")})
    invalid = sorted(p.name for p in out_dir.glob("ROOT_INVALIDATED__*.json"))
    return {"run_id": run_id, "mode": mode, "units": results, "root_invalidated": bool(invalid),
            "invalidation_markers": invalid}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--design", type=Path)
    ap.add_argument("--units-root", type=Path)
    ap.add_argument("--mode", choices=list(D.MODES))
    ap.add_argument("--host-role", default="COORDINATOR")
    ap.add_argument("--task-memory", type=int, default=3 << 30)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    if a.worker:
        return worker_main(a.worker)
    units = sorted(p for p in a.units_root.iterdir() if p.is_dir() and (p / "UNIT.json").is_file())
    r = run_units(a.out, a.design, units, a.mode, host_role=a.host_role, task_memory_bytes=a.task_memory,
                  resume=a.resume)
    print(json.dumps({"units": len(r["units"]), "root_invalidated": r["root_invalidated"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
