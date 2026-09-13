#!/usr/bin/env python3
"""C147 calibration: the planner's estimates are upper bounds of measured peaks.

Every case runs one metric group of a profile module, the way the module runs
it, on small synthetic inputs (including adversarial ones: NaN every third
sample, fully irregular timestamps), in its own child process started with
`systemd-run --user --scope --slice=<slice> -p MemoryMax=<cap> -p MemorySwapMax=0`.
The cap is 1.5 times the case's own estimate (at least 1 GiB, at most 3 GiB),
so a correct estimate never meets it: no case probes memory by exhaustion.

Inside the child: import and warm every module on a tiny contract, collect,
read the current RSS (rss0), allocate the case's inputs, run the group, read
getrusage(RUSAGE_SELF).ru_maxrss. The measured peak is ru_maxrss - rss0 (it
includes the case's held arrays and any input-generation transient, so it can
only overstate). The estimate compared against it is the planner's
estimated_peak_bytes minus BASE_PROCESS_BYTES and SERIALIZATION_BYTES; the
absolute ru_maxrss is also compared with the full estimate. The base case
records the warmed interpreter alone.

    df_memory_calibrate.py --out tests/fixtures/df_memory_calibration.v1.json [--slice crispdm-batch.slice]

The fixture stores inputs and measurements; tests recompute the estimates from
the current planner, so a factor change cannot hide behind a stale file.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGE = os.sysconf("SC_PAGE_SIZE")
SCHEMA = "crispdm.data_foundation.memory_calibration.v1"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


P = _load("df_memory_plan")


def rss_now() -> int:
    return int(open("/proc/self/statm").read().split()[1]) * PAGE


def maxrss() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


# ------------------------------------------------------------------ cases
# (case_id, module, group, sizes, context)
def case_table():
    cases = []

    def add(cid, module, group, sizes, ctx):
        cases.append({"case": cid, "module": module, "group": group, "sizes": sizes, "context": ctx})

    for nt in (60_000, 300_000, 900_000):
        T = nt * 5 // 3
        ctx = {"T": T, "n_train": nt, "has_ts": False}
        add(f"uni.train_reference.{nt}", "df_profile_univariate", "train_reference", {"n": nt}, ctx)
        add(f"uni.counts.{nt}", "df_profile_univariate", "counts", {"n": nt}, ctx)
        add(f"uni.quantiles.{nt}", "df_profile_univariate", "quantiles", {"n": nt}, ctx)
        add(f"uni.acf.{nt}", "df_profile_univariate", "acf", {"n": nt}, ctx)
        add(f"uni.shift_ks_psi.{nt}", "df_profile_univariate", "shift_ks_psi", {"n": T - nt, "n_train": nt}, ctx)
        add(f"uni.robust_z.{nt}", "df_profile_univariate", "robust_z", {"n": nt}, ctx)
        ctx_ts = dict(ctx, has_ts=True)
        add(f"uni.timestamps_duplicates.{nt}", "df_profile_univariate", "timestamps_duplicates", {"n": nt}, ctx_ts)
        add(f"inf.entropy_redundancy.{nt}", "df_profile_information", "entropy_redundancy", {"n": nt}, ctx)
        for o in P.PE_ORDERS:
            add(f"inf.permutation_entropy_order_{o}.{nt}", "df_profile_information",
                f"permutation_entropy_order_{o}", {"n": nt, "order": o}, ctx)
        add(f"inf.spectral_entropy.{nt}", "df_profile_information", "spectral_entropy", {"n": nt}, ctx)
        for s in ("raw_float64", "symbols"):
            for c in ("zlib9", "lzma6"):
                add(f"inf.compression_{s}_{c}.{nt}", "df_profile_information", f"compression_{s}_{c}", {"n": nt}, ctx)
        add(f"smp.timestamps.irregular.{nt}", "df_sampling", "timestamps", {"n": nt}, ctx_ts)
        add(f"smp.spectral_decimation.regular.{nt}", "df_sampling", "spectral_decimation", {"n": nt}, ctx)
        add(f"smp.spectral_decimation.nan_every_3rd.{nt}", "df_sampling", "spectral_decimation", {"n": nt}, ctx)
        add(f"mv.pair_correlations.{nt}", "df_profile_multivariate", "pair_correlations", {"n": nt},
            {"T": T, "n_train": nt, "k_pairs": 2})
        add(f"mv.pair_mutual_information.{nt}", "df_profile_multivariate", "pair_mutual_information", {"n": nt},
            {"T": T, "n_train": nt, "k_pairs": 2})
        add(f"mv.pair_lead_lag.{nt}", "df_profile_multivariate", "pair_lead_lag", {"n": nt},
            {"T": T, "n_train": nt, "k_pairs": 2})
    for m in (20_000, 80_000, 200_000):
        ctx = {"T": m * 5 // 3, "n_train": m, "has_ts": False}
        add(f"uni.unit_root_adf.{m}", "df_profile_univariate", "unit_root_adf", {"n_run": m, "lag": P.schwert_lag(m)}, ctx)
        add(f"uni.unit_root_kpss.{m}", "df_profile_univariate", "unit_root_kpss", {"n_run": m}, ctx)
    for L in (20_000, 60_000, 120_000):
        add(f"mv.pair_coherence.{L}", "df_profile_multivariate", "pair_coherence", {"L": L},
            {"T": L * 5 // 3, "n_train": L, "k_pairs": 2, "pair_rows": L})
    for R, V in ((20_000, 10), (100_000, 40), (40_000, 200)):
        ctx = {"T": R * 5 // 3, "n_train": R, "rg_rows": R * 5 // 3}
        add(f"inf.effective_rank.{R}x{V}", "df_profile_information", "effective_rank", {"rows": R, "V": V}, ctx)
        add(f"mv.matrix_diagnostics.{R}x{V}", "df_profile_multivariate", "matrix_diagnostics", {"rows": R, "V": V}, ctx)
        add(f"mv.cluster_bootstrap.{R}x{V}", "df_profile_multivariate", "cluster_bootstrap", {"rows": R, "V": V}, ctx)
    for T, rg in ((1_000_000, 250_000), (3_000_000, 1_000_000)):
        add(f"run.reader_numeric.{T}", "df_profile_run", "reader_column",
            {"T": T, "rg_rows": rg, "columns_held": 1, "text_timestamp": 0}, {"T": T})
        add(f"run.reader_timestamp.{T}", "df_profile_run", "reader_column",
            {"T": T, "rg_rows": rg, "columns_held": 1, "text_timestamp": 0}, {"T": T})
    add("run.reader_text_timestamp.400000", "df_profile_run", "reader_column",
        {"T": 400_000, "rg_rows": 400_000, "columns_held": 1, "text_timestamp": 1}, {"T": 400_000})
    return cases


def only(*groups):
    return lambda group, key=None, partition=None, **s: {
        "decision": "RUN_EXACT" if group in groups else "NOT_RUN_RESOURCE_BOUND", "window": None}


def _contract(T, V, meaning="SAMPLE_INDEX", period="NOT_APPLICABLE"):
    C = _load("df_contract")
    ds = "synthetic.c147.calibration.v1"
    doc = {"schema": C.DATASET_SCHEMA, "dataset_id": ds, "version": "1", "bank": "SYNTHETIC", "files": [],
           "content_sha256": "", "contract_sha256": "",
           "source": {"provider": "generator", "official_url": C.UNKNOWN, "citation": C.UNKNOWN, "doi": C.UNKNOWN,
                      "upstream_owner": C.UNKNOWN},
           "license": {"state": "NOT_APPLICABLE_GENERATED", "id": C.UNKNOWN, "url": C.UNKNOWN,
                       "text_sha256": "UNAVAILABLE", "attribution_required": C.UNKNOWN,
                       "redistribution": C.UNKNOWN, "derivatives": C.UNKNOWN, "evidence": []},
           "time": {"frequency_nominal_seconds": period, "timezone": "UTC", "timestamp_meaning": meaning,
                    "range_start": "0", "range_end": str(T), "availability_rule": "SAMPLE_INDEX",
                    "availability_delay_seconds": 0},
           "panel": {"aligned_common_grid": True, "n_series": V, "alignment_rule": "generated on one grid"},
           "partitions": {"scheme": "CHRONOLOGICAL_FRACTIONS",
                          "fractions": {"train": 0.6, "calibration": 0.2, "confirmation": 0.2},
                          "boundaries": C.chronological_partitions(T), "sealed_periods_excluded": [],
                          "frozen_before_profile": True},
           "dependence": [],
           "variables": [C.variable(ds, f"v{i}", license_state="NOT_APPLICABLE_GENERATED") for i in range(V)],
           "original_fields": {}}
    return C.seal(doc)


def warm():
    import numpy as np
    U, I, M, S = (_load(m) for m in ("df_profile_univariate", "df_profile_information", "df_profile_multivariate",
                                     "df_sampling"))
    _load("df_profile_run")
    rng = np.random.default_rng(0)
    T = 2400
    X = rng.standard_normal((T, 3))
    ct = _contract(T, 3)
    U.run_univariate(ct, X)
    I.run_information(ct, X)
    M.run_multivariate(ct, X)
    S.run_sampling(ct, X)
    ts = np.arange(T, dtype=np.int64) * 60_000_000_000
    ct2 = _contract(T, 3, meaning="PERIOD_START", period=60)
    S.run_sampling(ct2, X, ts)
    import pyarrow  # noqa: F401
    import pandas  # noqa: F401
    gc.collect()


def run_case(case: dict, tmp: Path) -> dict:
    """In the child: returns the measurement of one case."""
    import numpy as np
    U, I, M, S = (_load(m) for m in ("df_profile_univariate", "df_profile_information", "df_profile_multivariate",
                                     "df_sampling"))
    R = _load("df_profile_run")
    group, sizes, ctx = case["group"], case["sizes"], case["context"]
    cid = case["case"]
    prepared = {}
    if cid.startswith("run.reader"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        T, rg = sizes["T"], sizes["rg_rows"]
        path = tmp / "reader.parquet"
        if "text" in cid:
            labels = pa.array([f"2020-01-01 00:{(i // 60) % 60:02d}:{i % 60:02d}" for i in range(T)])
            pq.write_table(pa.table({"timestamp_label": labels}), path, row_group_size=rg)
        elif "timestamp" in cid:
            ts = pa.array(np.arange(T, dtype=np.int64) * 60_000_000_000, type=pa.timestamp("ns", tz="UTC"))
            pq.write_table(pa.table({"timestamp": ts}), path, row_group_size=rg)
        else:
            pq.write_table(pa.table({"value": pa.array(np.arange(T, dtype=np.int64))}), path, row_group_size=rg)
        del_objs = []
        gc.collect()
    gc.collect()
    rss0, hwm0 = rss_now(), maxrss()
    t0 = time.time()
    rng = np.random.default_rng(7)

    if cid.startswith("uni."):
        T, nt = ctx["T"], ctx["n_train"]
        if group == "timestamps_duplicates":
            ts = np.arange(T, dtype=np.int64) * 1_000_000_000
            parts = [("train", 0, nt), ("calibration", nt, nt + (T - nt) // 2), ("confirmation", nt + (T - nt) // 2, T)]
            list(U.timestamp_rows("d", parts, ts, "PERIOD_START", only(group)))
        elif group in ("unit_root_adf", "unit_root_kpss"):
            m = sizes["n_run"]
            x = np.cumsum(rng.standard_normal(m)) * 0.01 + rng.standard_normal(m)
            U.unit_root_rows("d", {"variable_id": "v"}, "train", x, only(group))
        else:
            x = rng.standard_normal(T)
            x[rng.integers(0, T, T // 20)] = np.nan
            parts = [("train", 0, nt), ("calibration", nt, nt + (T - nt) // 2), ("confirmation", nt + (T - nt) // 2, T)]
            admitted = {"train_reference": ("train_reference",), "counts": ("counts",),
                        "quantiles": ("counts", "quantiles"), "acf": ("counts", "quantiles", "acf"),
                        "shift_ks_psi": ("train_reference", "counts", "shift_ks_psi"),
                        "robust_z": ("train_reference", "counts", "robust_z")}[group]
            list(U.variable_rows("d", "v", x, parts, only(*admitted)))
    elif cid.startswith("inf."):
        if group == "effective_rank":
            Rr, V = sizes["rows"], sizes["V"]
            X = rng.standard_normal((Rr, V))
            I.matrix_rows("d", V, lambda s, e: X[s:e].copy(), Rr, only(group))
        else:
            T, nt = ctx["T"], ctx["n_train"]
            x = rng.standard_normal(T)
            x[rng.integers(0, T, T // 20)] = np.nan
            parts = [("train", 0, nt), ("calibration", nt, nt + (T - nt) // 2), ("confirmation", nt + (T - nt) // 2, T)]
            I.variable_rows("d", "v", x, parts, only(group))
    elif cid.startswith("mv."):
        if group in ("matrix_diagnostics", "cluster_bootstrap"):
            Rr, V = sizes["rows"], sizes["V"]
            f = rng.standard_normal((Rr, 3))
            X = np.repeat(f, -(-V // 3), axis=1)[:, :V] + 0.5 * rng.standard_normal((Rr, V))
            del f
            ids = [f"v{i:03d}" for i in range(V)]
            M._matrix_block_rows("d", ids, list(range(V)), lambda cols, s, e: X[s:e][:, cols], Rr,
                                 only(*(("matrix_diagnostics", "cluster_bootstrap") if group == "cluster_bootstrap"
                                        else ("matrix_diagnostics",))))
        else:
            n = sizes.get("n", sizes.get("L"))
            a = rng.standard_normal(n)
            block = np.empty((n, 2))
            block[:, 0] = a
            block[:, 1] = np.roll(a, 3) + 0.5 * rng.standard_normal(n)
            del a
            M._pair_rows("d", ["a", "b"], block[:, 0], block[:, 1], only(group))
    elif cid.startswith("smp."):
        T, nt = ctx["T"], ctx["n_train"]
        if group == "timestamps":
            steps = rng.choice(np.array([31, 47, 83, 130], dtype=np.int64), size=T)
            ts = np.cumsum(steps) * 1_000_000_000
            del steps
            ct = _contract(T, 1, meaning="INSTANT", period=60)
            parts = [(p, *ct["partitions"]["boundaries"][p]) for p in ("train", "calibration", "confirmation")]
            S.timestamp_rows(ct, parts, ts, only(group))
        else:
            ct = _contract(T, 1)
            s, e = ct["partitions"]["boundaries"]["train"]
            x = rng.standard_normal(e - s)
            if "nan_every_3rd" in cid:
                x[::3] = np.nan
            reg, fs = S.partition_regularity(ct, s, e, None)
            S.variable_partition_rows(ct, "v", x, "train", s, e, reg, fs, True, None, T, only(group))
    elif cid.startswith("run.reader"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(tmp / "reader.parquet")
        T = sizes["T"]
        if "text" in cid:
            R.read_label_timestamps(pf, "timestamp_label", "%Y-%m-%d %H:%M:%S", T)
        elif "timestamp" in cid:
            R.read_timestamp_column(pf, "timestamp", T)
        else:
            R.read_numeric_column(pf, "value", T)
    else:
        raise ValueError(cid)
    peak = maxrss()
    est_total, _, params = P.estimate(case["module"], group, sizes, ctx)
    base = P.BASE_PROCESS_BYTES + P.SERIALIZATION_BYTES
    measured = peak - rss0
    return {**case, "rss0_bytes": rss0, "setup_overhang_bytes": max(0, hwm0 - rss0),
            "measured_peak_delta_bytes": measured, "absolute_maxrss_bytes": peak,
            "estimated_peak_bytes": est_total, "estimate_minus_base_bytes": est_total - base,
            "ratio_incremental": round(measured / (est_total - base), 4),
            "ratio_absolute": round(peak / est_total, 4),
            "derived_bytes": params["derived_bytes"], "held_bytes": params["held_bytes"],
            "factor_needed_for_margin_1_15": round(1.15 * max(0, measured - params["held_bytes"])
                                                   / max(1, params["derived_bytes"]), 3),
            "wall_seconds": round(time.time() - t0, 2)}


def child_main(case_id: str) -> int:
    warm()
    base = maxrss()
    if case_id == "base":
        print(json.dumps({"case": "base", "absolute_maxrss_bytes": base, "rss_after_warm_bytes": rss_now(),
                          "BASE_PROCESS_BYTES": P.BASE_PROCESS_BYTES,
                          "ratio_absolute": round(base / P.BASE_PROCESS_BYTES, 4)}))
        return 0
    case = next(c for c in case_table() if c["case"] == case_id)
    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps(run_case(case, Path(tmp))))
    return 0


def launch(case_id: str, cap: int, slice_: str) -> dict:
    env_cmd = ["env", "-u", "PYTHONPATH", "CUDA_VISIBLE_DEVICES=", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
               "MKL_NUM_THREADS=1"]
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--slice={slice_}",
           "-p", f"MemoryMax={cap}", "-p", "MemorySwapMax=0", *env_cmd,
           sys.executable, "-B", str(Path(__file__).resolve()), "--child", case_id]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return {"case": case_id, "error": f"exit {r.returncode}", "stderr_tail": r.stderr[-800:], "cap_bytes": cap}
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out["cap_bytes"] = cap
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--slice", default="crispdm-batch.slice")
    ap.add_argument("--only", default="", help="comma-separated case-id prefixes")
    a = ap.parse_args(argv)
    if a.child:
        return child_main(a.child)
    results = [launch("base", 1 << 30, a.slice)]
    prefixes = [p for p in a.only.split(",") if p]
    for case in case_table():
        if prefixes and not any(case["case"].startswith(p) for p in prefixes):
            continue
        est_total, _, _ = P.estimate(case["module"], case["group"], case["sizes"], case["context"])
        cap = int(min(3 << 30, max(1 << 30, 1.5 * est_total)))
        res = launch(case["case"], cap, a.slice)
        results.append(res)
        print(json.dumps({k: res.get(k) for k in ("case", "ratio_incremental", "ratio_absolute",
                                                  "factor_needed_for_margin_1_15", "error")}), flush=True)
    doc = {"schema": SCHEMA, "planner_constants_sha256": P.constants_sha256(),
           "planner_code_sha256": P.CODE_SHA256,
           "method": "child per case under a memory cap of 1.5x its estimate; measured = ru_maxrss - rss0",
           "python": sys.version.split()[0], "cases": results}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    bad = [r for r in results if r.get("error")]
    print(json.dumps({"cases": len(results), "errors": len(bad)}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
