#!/usr/bin/env python3
"""C137, C138 (order 2026-09-12): what each operator preserves and destroys,
per operator and regime, against known truth; and the only decisions the lab
may emit.

For every synthetic unit, every variable and every operator spec:

* the operator is fitted on the TRAIN partition only (the longest complete
  contiguous train stretch; no imputation) and applied causally to the
  whole observed series;
* X (observed), D(X) (operator output) and R = X - D(X) are compared with
  the clean truth on CALIBRATION and CONFIRMATION separately, only on
  samples the operator marks available and that are not missing;
* measured: RMSE and SNR (raw, denoised, change), amplitude ratio, delay
  (lag of best alignment of D with clean, D lagging), residual signal share
  and excess residual autocorrelation, distortion when the unit has no noise,
  and event preservation: impulse and bump height retention and peak shift,
  step edge delay, motif shape correlation, regime-change latency, slope
  error after trend knots, extreme retention.

Delay and cost come from `df_operator_measure.measure` on one declared
reference fit per spec, reported apart from the unit metrics.

Decisions, per operator x regime (family, perturbation, declared SNR,
length, missingness), follow DECISION_RULES, frozen and hashed before any
run. They are only LAB_CALIBRATED, REGIME_LIMITED, NOT_IDENTIFIABLE and
LAB_REJECTED; none of them is public eligibility. An RMSE improvement that
destroys events does not advance. The non-causal oracle is a negative
control and is always LAB_REJECTED. Failure regions are published for every
operator, not only its best case.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OPS = _load("df_operators")
MEASURE = _load("df_operator_measure")
LOADER = _load("load_data_foundation")
SYNC = _load("df_synthetic_contract")

EVAL_PARTITIONS = ("calibration", "confirmation")
MAX_DELAY_LAG = 32
EVENT_PRE = 10
STEP_SEARCH = 64
REGIME_WINDOW = 32
VARIANCE_WINDOW = 128
MIN_SUPPORT = 32
REFERENCE_UNIT = "sinusoid__white__snr10__none__n2048__v1__seed11"

DECISION_RULES = {
    "version": "c138.v1",
    "decision_partition": "calibration",
    "confirmation_partition": "confirmation",
    "min_units": 3,
    "improvement": {"snr_improvement_db_min": 1.0, "rmse_ratio_max": 0.9},
    "event_destruction": {"impulse_retention_min": 0.5, "bump_retention_min": 0.5,
                          "step_delay_max_samples": 8, "motif_corr_min": 0.5, "extreme_retention_min": 0.7,
                          "regime_mean_latency_max_samples": 48},
    "residual_leakage": {"signal_share_max": 0.1},
    "noise_free_false_positive": {"distortion_ratio_max": 0.1},
    "seed_dispersion": {"min_unit_snr_improvement_db": 0.0, "max_unit_rmse_ratio": 1.0},
    "precedence": [
        "NOT_IDENTIFIABLE if fewer than min_units valid unit-variable measurements in the decision partition",
        "LAB_REJECTED for a non-causal control, whatever it measures",
        "LAB_REJECTED for event destruction or residual leakage, even when RMSE improves",
        "noise-free regime: LAB_CALIBRATED if the median distortion is within its maximum, otherwise LAB_REJECTED",
        "LAB_REJECTED if the decision partition shows no improvement",
        "NOT_IDENTIFIABLE if the confirmation partition does not confirm the improvement",
        "REGIME_LIMITED if some unit-variable does not improve although the median does",
        "LAB_CALIBRATED otherwise"],
    "never": "none of these decisions equals or implies public eligibility",
}


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha_obj(obj) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


RULE_SHA256 = sha_obj(DECISION_RULES)


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


# ------------------------------------------------------------------- units
def load_unit(unit_dir: Path) -> dict:
    unit_dir = Path(unit_dir)
    rec = SYNC.verify_unit(unit_dir)
    V, T = rec["n_variables"], rec["n_samples"]

    def arr(name):
        a = np.load(unit_dir / f"{name}.npy", allow_pickle=False)
        if a.shape == (V, T) and V != T:
            a = a.T
        if a.shape != (T, V):
            raise ValueError(f"{rec['unit_id']}: {name} has shape {a.shape}, expected (T, V) or (V, T)")
        return a
    contract = SYNC.unit_contract(unit_dir)
    return {"rec": rec, "dir": unit_dir, "clean": arr("clean_signal").astype(float),
            "noise": arr("additive_noise").astype(float), "observed": arr("observed_signal").astype(float),
            "mask": arr("missing_mask").astype(bool),
            "events": json.loads((unit_dir / "events.json").read_text())["events"],
            "content_sha256": contract["content_sha256"],
            "variable_ids": [v["variable_id"] for v in contract["variables"]]}


def regime_of(rec: dict) -> dict:
    return {"family": rec["family"], "perturbation": rec["perturbation"],
            "declared_snr_db": str(rec["declared_snr_db"]), "length": int(rec["n_samples"]),
            "missingness": rec["missingness"]["kind"]}


def train_fit_slice(observed: np.ndarray, train) -> slice:
    s, e = train
    complete = ~np.isnan(observed[s:e]).any(axis=1)
    best, cur, best_start, cur_start = 0, 0, s, s
    for i, ok in enumerate(complete):
        if ok:
            if cur == 0:
                cur_start = s + i
            cur += 1
            if cur > best:
                best, best_start = cur, cur_start
        else:
            cur = 0
    return slice(best_start, best_start + best)


# ----------------------------------------------------------------- metrics
def _finite_median(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _snr_db(signal_var: float, error_power: float):
    if signal_var <= 1e-15 or error_power <= 1e-15:
        return None
    return 10.0 * math.log10(signal_var / error_power)


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _acf1(x):
    x = x[np.isfinite(x)]
    if len(x) < 16 or np.std(x) < 1e-12:
        return None
    x = x - x.mean()
    return float(np.dot(x[1:], x[:-1]) / np.dot(x, x))


def event_metrics(ev: dict, c, x, d, lo: int, hi: int) -> dict:
    i = int(ev["index"])
    out = {}
    if not (lo <= i < hi):
        return out
    t = ev["type"]
    if t == "impulse":
        pre_d, pre_x = d[max(lo, i - EVENT_PRE):i], x[max(lo, i - EVENT_PRE):i]
        h = float(ev["height"])
        if np.isfinite(d[i]) and np.isfinite(pre_d).sum() >= 3 and abs(h) > 1e-12:
            out["impulse_retention"] = float((d[i] - np.nanmedian(pre_d)) / h)
        if np.isfinite(x[i]) and np.isfinite(pre_x).sum() >= 3 and abs(h) > 1e-12:
            out["impulse_retention_raw"] = float((x[i] - np.nanmedian(pre_x)) / h)
    elif t == "bump":
        hw, h = int(ev["support_halfwidth"]), float(ev["height"])
        a, b = max(lo, i - hw), min(hi, i + hw + 1)
        seg = d[a:b]
        base = d[max(lo, a - EVENT_PRE):a]
        if np.isfinite(seg).sum() >= 3 and np.isfinite(base).sum() >= 3 and abs(h) > 1e-12:
            k = int(np.nanargmax(seg) if h > 0 else np.nanargmin(seg))
            out["bump_retention"] = float((seg[k] - np.nanmedian(base)) / h)
            out["bump_peak_shift_samples"] = float(a + k - i)
    elif t == "step":
        lb, la = float(ev["level_before"]), float(ev["level_after"])
        mid, sgn = (lb + la) / 2.0, math.copysign(1.0, la - lb)
        delay = None
        for tt in range(i, min(hi, i + STEP_SEARCH)):
            if np.isfinite(d[tt]) and sgn * (d[tt] - mid) >= 0:
                delay = tt - i
                break
        out["step_delay_samples"] = float(delay if delay is not None else STEP_SEARCH)
        out["step_delay_censored"] = 0.0 if delay is not None else 1.0
    elif t == "motif":
        L = int(ev["length"])
        if i + L <= hi:
            r_d, r_x = _corr(d[i:i + L], c[i:i + L]), _corr(x[i:i + L], c[i:i + L])
            if r_d is not None:
                out["motif_corr"] = r_d
            if r_x is not None:
                out["motif_corr_raw"] = r_x
    elif t == "regime_boundary" and ev.get("regime") == "mean":
        mid = (float(ev["mean_before"]) + float(ev["mean_after"])) / 2.0
        sgn = math.copysign(1.0, float(ev["mean_after"]) - float(ev["mean_before"]))
        lat = None
        for tt in range(i, min(hi, i + 4 * STEP_SEARCH)):
            w = d[max(lo, tt - REGIME_WINDOW + 1):tt + 1]
            if np.isfinite(w).sum() >= REGIME_WINDOW // 2 and sgn * (np.nanmean(w) - mid) >= 0:
                lat = tt - i
                break
        out["regime_mean_latency_samples"] = float(lat if lat is not None else 4 * STEP_SEARCH)
    elif t == "regime_boundary" and ev.get("regime") == "variance":
        before, after = d[max(lo, i - VARIANCE_WINDOW):i], d[i:min(hi, i + VARIANCE_WINDOW)]
        true_ratio = float(ev["amplitude_after"]) / float(ev["amplitude_before"])
        if np.isfinite(before).sum() >= 32 and np.isfinite(after).sum() >= 32 and np.nanstd(before) > 1e-12:
            out["regime_variance_ratio_error"] = float(abs(np.nanstd(after) / np.nanstd(before) - true_ratio) / true_ratio)
    elif t == "trend_knot":
        a, b = i + REGIME_WINDOW, min(hi, i + 5 * REGIME_WINDOW)
        seg = d[a:b]
        ok = np.isfinite(seg)
        if ok.sum() >= 32:
            tt = np.arange(a, b)[ok]
            slope = float(np.polyfit(tt, seg[ok], 1)[0])
            jump = abs(float(ev["slope_after"]) - float(ev["slope_before"])) + 1e-12
            out["trend_slope_error_ratio"] = abs(slope - float(ev["slope_after"])) / jump
    return out


def partition_metrics(c, n, x, d, avail, lo: int, hi: int, events: list) -> dict:
    cs, ns, xs, ds, av = c[lo:hi], n[lo:hi], x[lo:hi], d[lo:hi], avail[lo:hi]
    sup = av & np.isfinite(xs) & np.isfinite(ds)
    if sup.sum() < MIN_SUPPORT:
        return {"status": "UNAVAILABLE", "reason": f"fewer than {MIN_SUPPORT} available samples"}
    cc, nn, xx, dd = cs[sup], ns[sup], xs[sup], ds[sup]
    mse_raw, mse_den, var_c = float(np.mean((xx - cc) ** 2)), float(np.mean((dd - cc) ** 2)), float(np.var(cc))
    m = {"status": "COMPLETED", "reason": "", "support": int(sup.sum()),
         "rmse_raw": math.sqrt(mse_raw), "rmse_denoised": math.sqrt(mse_den),
         "rmse_ratio": math.sqrt(mse_den / mse_raw) if mse_raw > 1e-15 else None,
         "snr_raw_db": _snr_db(var_c, mse_raw), "snr_denoised_db": _snr_db(var_c, mse_den),
         "noise_free": bool(np.max(np.abs(nn)) < 1e-12),
         "signal_free": bool(var_c < 1e-15),
         "distortion_ratio": math.sqrt(mse_den) / math.sqrt(var_c) if var_c > 1e-15 else None,
         "amplitude_ratio": float(np.cov(dd, cc, ddof=0)[0, 1] / var_c) if var_c > 1e-15 else None}
    m["snr_improvement_db"] = (m["snr_denoised_db"] - m["snr_raw_db"]
                               if m["snr_raw_db"] is not None and m["snr_denoised_db"] is not None else None)
    best_k, best_r = None, None
    for k in range(0, MAX_DELAY_LAG + 1):
        r = _corr(ds[k:], cs[:len(cs) - k]) if k else _corr(ds, cs)
        if r is not None and (best_r is None or r > best_r):
            best_k, best_r = k, r
    m["delay_samples"] = float(best_k) if best_k is not None else None
    m["delay_alignment_corr"] = best_r
    r = xx - dd
    share = _corr(r, cc)
    m["residual_signal_share"] = share ** 2 if share is not None else None
    a_r, a_n = _acf1(np.where(sup, xs - ds, np.nan)), _acf1(ns)
    m["residual_excess_acf1"] = (a_r - a_n) if a_r is not None and a_n is not None else None
    if var_c > 1e-15:
        med = float(np.median(cc))
        dev = np.abs(cc - med)
        sel = dev >= np.quantile(dev, 0.99)
        if dev[sel].mean() > 1e-12:
            m["extreme_retention"] = float(np.mean(np.abs(dd[sel] - med)) / np.mean(dev[sel]))
    per_event: dict = {}
    for ev in events:
        for k, v in event_metrics(ev, c, x, d, lo, hi).items():
            per_event.setdefault(k, []).append(v)
    for k, vs in per_event.items():
        m[k] = float(np.median(vs))
        m[f"{k}__events"] = len(vs)
    return m


# ------------------------------------------------------------ evaluation
def spec_sha(spec: dict) -> str:
    return sha_obj({"kind": spec["kind"], "params": spec["params"]})


def evaluate_unit(args) -> list[dict]:
    unit_dir, specs = args
    u = load_unit(Path(unit_dir))
    rec = u["rec"]
    parts = rec["partitions"]
    fit_sl = train_fit_slice(u["observed"], parts["train"])
    out = []
    for spec in specs:
        t0 = time.process_time()
        base = {"unit_id": rec["unit_id"], "unit_dir": str(unit_dir), "regime": regime_of(rec), "spec": spec,
                "content_sha256": u["content_sha256"], "variable_ids": u["variable_ids"]}
        try:
            if fit_sl.stop - fit_sl.start < 50:
                raise OPS.OperatorRefusal(f"longest complete train stretch has {fit_sl.stop - fit_sl.start} rows")
            fitted = OPS.fit(spec, u["observed"][fit_sl], "train")
            if fitted["status"] != "FITTED":
                raise OPS.OperatorAbstain(f"ABSTAIN: {fitted['abstain_reason']}")
            oracle = spec["kind"] in OPS.NON_CAUSAL_KINDS
            y, avail, _ = OPS.transform_batch(fitted, u["observed"], oracle_mode=oracle)
            status, reason, fsha = "COMPLETED", "", fitted["artifact_sha256"]
        except Exception as exc:  # noqa: BLE001 - every outcome is recorded
            y = avail = None
            status = "REFUSED" if isinstance(exc, OPS.OperatorRefusal) else "FAILED"
            reason, fsha = f"{type(exc).__name__}: {exc}"[:300], None
        cpu = time.process_time() - t0
        for v in range(rec["n_variables"]):
            r = dict(base, variable_index=v, variable_id=u["variable_ids"][v], status=status, reason=reason,
                     fitted_sha256=fsha, cpu_seconds=cpu / rec["n_variables"], partitions={})
            if y is not None:
                evs = [e for e in u["events"] if int(e.get("variable", 0)) == v]
                for p in EVAL_PARTITIONS:
                    lo, hi = parts[p]
                    r["partitions"][p] = partition_metrics(u["clean"][:, v], u["noise"][:, v], u["observed"][:, v],
                                                           y[:, v], avail[:, v] & ~u["mask"][:, v], lo, hi, evs)
            out.append(r)
    return out


# ------------------------------------------------------------- decisions
def regime_key(regime: dict) -> str:
    return json.dumps(regime, sort_keys=True)


def decide(spec: dict, regime: dict, records: list[dict]) -> tuple[str, list[str], dict]:
    R = DECISION_RULES
    dp, cp = R["decision_partition"], R["confirmation_partition"]
    valid = [r for r in records if r["status"] == "COMPLETED" and r["partitions"].get(dp, {}).get("status") == "COMPLETED"]
    ev = {"unit_variables": len(records), "valid_unit_variables": len(valid),
          "refused_or_failed": sum(r["status"] != "COMPLETED" for r in records)}
    if len(valid) < R["min_units"]:
        return "NOT_IDENTIFIABLE", [f"only {len(valid)} valid unit-variable measurements"], ev

    def med(key, part=dp):
        return _finite_median([r["partitions"].get(part, {}).get(key) for r in valid])

    keys = ("snr_improvement_db", "rmse_ratio", "distortion_ratio", "impulse_retention", "bump_retention",
            "step_delay_samples", "motif_corr", "extreme_retention", "residual_signal_share",
            "regime_mean_latency_samples", "delay_samples", "amplitude_ratio")
    ev["calibration_medians"] = {k: med(k) for k in keys}
    ev["confirmation_medians"] = {k: med(k, cp) for k in keys}
    reasons = []
    if OPS.control_label(spec) == OPS.NON_CAUSAL_NEGATIVE_CONTROL or spec["kind"] in OPS.NON_CAUSAL_KINDS:
        return "LAB_REJECTED", ["NON_CAUSAL_CONTROL: uses future samples"], ev
    E = R["event_destruction"]
    checks = (("impulse_retention", E["impulse_retention_min"], "min"), ("bump_retention", E["bump_retention_min"], "min"),
              ("step_delay_samples", E["step_delay_max_samples"], "max"), ("motif_corr", E["motif_corr_min"], "min"),
              ("extreme_retention", E["extreme_retention_min"], "min"),
              ("regime_mean_latency_samples", E["regime_mean_latency_max_samples"], "max"))
    for key, bound, kind in checks:
        value = ev["calibration_medians"][key]
        if value is not None and ((kind == "min" and value < bound) or (kind == "max" and value > bound)):
            reasons.append(f"EVENT_DESTRUCTION: median {key} {value:.3f} beyond {bound}")
    leak = ev["calibration_medians"]["residual_signal_share"]
    if leak is not None and leak > R["residual_leakage"]["signal_share_max"]:
        reasons.append(f"RESIDUAL_LEAKAGE: median signal share {leak:.3f} > {R['residual_leakage']['signal_share_max']}")
    snr_i, ratio = ev["calibration_medians"]["snr_improvement_db"], ev["calibration_medians"]["rmse_ratio"]
    I = R["improvement"]
    improves = (ratio is not None and ratio <= I["rmse_ratio_max"]
                and (snr_i is None or snr_i >= I["snr_improvement_db_min"]))
    ev["improves_on_decision_partition"] = improves
    if reasons:
        return "LAB_REJECTED", reasons, ev
    if all(r["partitions"][dp].get("noise_free") for r in valid):
        dist = ev["calibration_medians"]["distortion_ratio"]
        mx = R["noise_free_false_positive"]["distortion_ratio_max"]
        if dist is not None and dist <= mx:
            return "LAB_CALIBRATED", [f"NOISE_FREE: median distortion {dist:.3f} <= {mx}"], ev
        return "LAB_REJECTED", [f"FALSE_POSITIVE_DISTORTION: median distortion {dist} > {mx}"], ev
    if not improves:
        return "LAB_REJECTED", [f"NO_IMPROVEMENT: median rmse_ratio {ratio} and snr change {snr_i} dB"], ev
    c_snr, c_ratio = ev["confirmation_medians"]["snr_improvement_db"], ev["confirmation_medians"]["rmse_ratio"]
    confirms = (c_ratio is not None and c_ratio <= I["rmse_ratio_max"]
                and (c_snr is None or c_snr >= I["snr_improvement_db_min"]))
    if not confirms:
        return "NOT_IDENTIFIABLE", [f"CONFIRMATION_DISAGREES: median rmse_ratio {c_ratio}, snr change {c_snr} dB"], ev
    D = R["seed_dispersion"]
    worst_snr = min((r["partitions"][dp].get("snr_improvement_db") for r in valid
                     if r["partitions"][dp].get("snr_improvement_db") is not None), default=None)
    worst_ratio = max((r["partitions"][dp].get("rmse_ratio") for r in valid
                       if r["partitions"][dp].get("rmse_ratio") is not None), default=None)
    ev["worst_unit_snr_improvement_db"], ev["worst_unit_rmse_ratio"] = worst_snr, worst_ratio
    if (worst_snr is not None and worst_snr < D["min_unit_snr_improvement_db"]) or \
            (worst_ratio is not None and worst_ratio > D["max_unit_rmse_ratio"]):
        return "REGIME_LIMITED", [f"SEED_OR_VARIABLE_DEPENDENT: worst unit snr change {worst_snr}, rmse_ratio {worst_ratio}"], ev
    return "LAB_CALIBRATED", ["IMPROVES_AND_PRESERVES_EVENTS_ON_BOTH_PARTITIONS"], ev


# -------------------------------------------------------------------- rows
COMPONENT = {"rmse_raw": "RAW", "snr_raw_db": "RAW", "rmse_denoised": "DENOISED", "snr_denoised_db": "DENOISED",
             "amplitude_ratio": "DENOISED", "delay_samples": "DENOISED", "delay_alignment_corr": "DENOISED",
             "residual_signal_share": "RESIDUAL", "residual_excess_acf1": "RESIDUAL"}


def _num(v):
    return float(v) if v is not None and not isinstance(v, bool) and math.isfinite(float(v)) else None


def rows_for(records: list[dict], decisions: list[dict], delay_cost: list[dict], run_id: str) -> dict:
    code = code_sha256()
    runs, metrics = [], []
    for r in records:
        spec = r["spec"]
        run_row = {"run_id": run_id, "subject_id": r["unit_id"], "content_sha256": r["content_sha256"],
                   "variable_id": r["variable_id"], "regime": r["regime"], "operator_kind": spec["kind"],
                   "operator_params": spec["params"], "spec_sha256": spec_sha(spec), "fitted_sha256": r["fitted_sha256"],
                   "status": r["status"], "reason": r["reason"] or "", "code_sha256": code,
                   "cpu_seconds": round(r["cpu_seconds"], 6), "peak_memory_bytes": None}
        if run_row["status"] != "COMPLETED" and not run_row["reason"]:
            run_row["reason"] = "not completed"
        runs.append(run_row)
        run_sha = LOADER.row_sha256("df_fact_operator_run", run_row)
        for p, pm in r["partitions"].items():
            if pm.get("status") != "COMPLETED":
                metrics.append({"run_id": run_id, "operator_run_sha256": run_sha, "component": "COMPARISON",
                                "partition": p, "metric": "partition_support", "estimator": "available_not_missing",
                                "estimator_params": {"min_support": MIN_SUPPORT}, "value": None, "value_text": None,
                                "status": "UNAVAILABLE", "reason": pm.get("reason", "unavailable"), "code_sha256": code})
                continue
            for k, v in pm.items():
                if k in ("status", "reason") or isinstance(v, bool):
                    continue
                val = _num(v)
                metrics.append({"run_id": run_id, "operator_run_sha256": run_sha,
                                "component": COMPONENT.get(k, "COMPARISON" if k in ("rmse_ratio", "snr_improvement_db", "distortion_ratio", "support") else "DENOISED"),
                                "partition": p, "metric": k, "estimator": f"c137.{k}",
                                "estimator_params": {"vs": "clean_truth", "max_delay_lag": MAX_DELAY_LAG},
                                "value": val, "value_text": None,
                                "status": "COMPLETED" if val is not None else "INCONCLUSIVE",
                                "reason": "" if val is not None else "undefined for this unit (e.g. no signal or no noise)",
                                "code_sha256": code})
    return {"df_fact_operator_run": runs, "df_fact_operator_signal_metric": metrics,
            "df_fact_operator_delay_cost": delay_cost, "df_fact_lab_decision": decisions}


def delay_cost_rows(reference: dict, specs: list[dict], run_id: str) -> list[dict]:
    code = code_sha256()
    rows = []
    fit_sl = train_fit_slice(reference["observed"], reference["rec"]["partitions"]["train"])
    for spec in specs:
        base = {"run_id": run_id, "operator_kind": spec["kind"], "operator_params": spec["params"],
                "spec_sha256": spec_sha(spec), "code_sha256": code}
        try:
            fitted = OPS.fit(spec, reference["observed"][fit_sl], "train")
            rep = MEASURE.measure(fitted)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(base, metric="measure", frequency=None, value=None, value_text=None, status="FAILED",
                             reason=f"{type(exc).__name__}: {exc}"[:300]))
            continue
        if fitted["status"] != "FITTED":
            rows.append(dict(base, metric="measure", frequency=None, value=None, value_text=None, status="REFUSED",
                             reason=f"ABSTAIN: {fitted['abstain_reason']}"))
            continue

        def walk(node, path, freq=None):
            if isinstance(node, dict):
                for k, v in node.items():
                    try:
                        f = float(k)
                    except (TypeError, ValueError):
                        f = None
                    walk(v, path if f is not None else f"{path}.{k}" if path else str(k), f if f is not None else freq)
            elif isinstance(node, bool):
                return
            elif isinstance(node, (int, float)) and math.isfinite(node):
                rows.append(dict(base, metric=path, frequency=freq, value=float(node), value_text=None,
                                 status="COMPLETED", reason=""))
            elif isinstance(node, str) and node.upper() == "UNDEFINED":
                rows.append(dict(base, metric=path, frequency=freq, value=None, value_text=None, status="UNAVAILABLE",
                                 reason="UNDEFINED for a non-linear or non-causal kind"))
        for key in ("algorithmic_lookback", "look_ahead", "group_delay", "phase_delay", "settling_99_samples",
                    "empirical", "cost", "warmup"):
            if key in rep:
                walk(rep[key], key)
    return rows


def failure_regions(decisions: list[dict]) -> dict:
    by_op: dict = {}
    for d in decisions:
        key = json.dumps({"kind": d["operator_kind"], "params": d["operator_params"]}, sort_keys=True)
        entry = by_op.setdefault(key, {"operator": json.loads(key), "counts": {}, "failure_regions": [],
                                       "calibrated_regions": [], "regime_limited_regions": []})
        entry["counts"][d["decision"]] = entry["counts"].get(d["decision"], 0) + 1
        item = {"regime": d["regime"], "reasons": d["evidence"]["reasons"]}
        if d["decision"] in ("LAB_REJECTED", "NOT_IDENTIFIABLE"):
            entry["failure_regions"].append(dict(item, decision=d["decision"]))
        elif d["decision"] == "LAB_CALIBRATED":
            entry["calibrated_regions"].append(item)
        else:
            entry["regime_limited_regions"].append(item)
    return by_op


def run(bank_root: Path, out_dir: Path, specs: list[dict] | None = None, unit_names: list[str] | None = None,
        workers: int = 4) -> dict:
    bank_root, out_dir = Path(bank_root), Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"REFUSED: {out_dir.name} exists; lab outputs are write-once")
    manifest_bytes = (bank_root / "BANK_MANIFEST.json").read_bytes()
    if not json.loads(manifest_bytes).get("complete") and unit_names is None:
        raise SystemExit("REFUSED: the bank manifest is not complete")
    specs = specs or OPS.bank_specs()
    units = sorted(p for p in bank_root.iterdir() if p.is_dir() and (unit_names is None or p.name in unit_names))
    run_id = "c137_" + sha_obj({"code": code_sha256(), "bank": hashlib.sha256(manifest_bytes).hexdigest(),
                                "specs": specs, "rules": RULE_SHA256, "units": [u.name for u in units]})[:24]
    t0 = time.time()
    records: list[dict] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk in pool.map(evaluate_unit, [(str(u), specs) for u in units], chunksize=4):
                records.extend(chunk)
    else:
        for u in units:
            records.extend(evaluate_unit((str(u), specs)))
    grouped: dict = {}
    for r in records:
        grouped.setdefault((spec_sha(r["spec"]), regime_key(r["regime"])), []).append(r)
    code = code_sha256()
    decisions = []
    for (_, rk), recs in sorted(grouped.items()):
        spec = recs[0]["spec"]
        decision, reasons, ev = decide(spec, json.loads(rk), recs)
        decisions.append({"run_id": run_id, "operator_kind": spec["kind"], "operator_params": spec["params"],
                          "spec_sha256": spec_sha(spec), "regime": json.loads(rk), "decision": decision,
                          "rule_sha256": RULE_SHA256, "evidence": dict(ev, reasons=reasons),
                          "failure_regions": [], "externally_reviewed": False, "code_sha256": code})
    ref_dir = bank_root / REFERENCE_UNIT
    reference = load_unit(ref_dir if ref_dir.is_dir() else units[0])
    tables = rows_for(records, decisions, delay_cost_rows(reference, specs, run_id), run_id)
    problems = {t: [i for i, row in enumerate(rows) if LOADER.validate_row(t, row)] for t, rows in tables.items()}
    if any(problems.values()):
        first = {t: LOADER.validate_row(t, tables[t][ix[0]]) for t, ix in problems.items() if ix}
        raise SystemExit(f"REFUSED: rows do not validate for OLAP: {first}")
    out_dir.mkdir(parents=True)
    digests = {}
    for t, rows in tables.items():
        p = out_dir / f"{t}.jsonl"
        p.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
        digests[t] = {"rows": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    counts: dict = {}
    for d in decisions:
        counts[d["decision"]] = counts.get(d["decision"], 0) + 1
    summary = {"schema": "crispdm.data_foundation.lab_evaluation_summary.v1", "run_id": run_id,
               "code_sha256": code, "rule_sha256": RULE_SHA256, "decision_rules": DECISION_RULES,
               "bank_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "reference_unit_for_delay_and_cost": reference["rec"]["unit_id"],
               "units": len(units), "specs": len(specs), "unit_variable_operator_records": len(records),
               "decision_counts": counts, "tables": digests, "wall_seconds": round(time.time() - t0, 1),
               "operators": failure_regions(decisions),
               "never": DECISION_RULES["never"]}
    text = json.dumps(summary, indent=1, sort_keys=True, allow_nan=False)
    if str(Path.home()) in text:
        raise SystemExit("REFUSED: absolute home path in the summary")
    (out_dir / "LAB_EVALUATION_SUMMARY.json").write_text(text + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    a = ap.parse_args(argv)
    s = run(a.bank, a.out, workers=a.workers)
    print(json.dumps({k: s[k] for k in ("run_id", "units", "specs", "unit_variable_operator_records",
                                         "decision_counts", "wall_seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
