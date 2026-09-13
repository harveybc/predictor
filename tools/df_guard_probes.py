#!/usr/bin/env python3
"""C167-C168 (order 2026-09-13): the probe of every causal check of the D2
boundary, runnable in a fresh process against whatever source tree this file
sits in.

Each probe builds, through the public APIs, the smallest input that only the
named check can refuse, and returns a thunk that consumes it. The battery uses
some of them as SNAPSHOT_REFUSAL cases, in process, against the productive
modules. tools/df_structural_mutation.py copies this file together with the
modules it needs into a temporary directory, removes exactly one check from the
copy, and runs

    python -B <copy>/tools/df_guard_probes.py --guard NAME

in a new interpreter, once on the intact copy and once on the mutant. The
result is one line ``STRUCTURAL_PROBE_RESULT {json}`` on stdout. Nothing here
switches a check off: there is nothing to switch.

Every module is loaded from this file's own directory, so a copy never reads
the productive tree, and the result lists the file and sha256 of every module
it loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT_MARKER = "STRUCTURAL_PROBE_RESULT "


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

FROZEN, EXPANDING, OFFLINE = OPS.FROZEN_PREVIOUS_PARTITION, OPS.EXPANDING_PREFIX, OPS.OFFLINE_ANALYSIS_ONLY_NON_CAUSAL
EWMA = {"kind": "ewma", "params": {"alpha": 0.3}}
HAAR = {"kind": "trailing_haar_threshold", "params": {"levels": 2, "threshold_k": 3.0}}
REFUSALS = (OPS.OperatorRefusal, SNAP.SnapshotRefusal)


def base_series(n: int, V: int, seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    return (np.cumsum(r.normal(scale=0.3, size=(n, V)), axis=0) + 0.8 * np.sin(2 * np.pi * t / 24)
            + r.normal(size=(n, V)))


def fixture(n=100, V=2, seed=0, **kw):
    X = base_series(n, V, seed)
    c, L = SYNC.in_memory_contract(X, name=f"fx{seed}", **kw)
    return X, c, L


def _redigest(snap):
    for arr, key in (("matrix", "matrix_sha256"), ("timestamps", "timestamps_sha256"),
                     ("available_at", "availability_sha256")):
        object.__setattr__(snap, key, SNAP.array_sha256(getattr(snap, arr)))
    object.__setattr__(snap, "snapshot_sha256", SNAP.C.sha_obj(snap.facts()))
    return snap


# ------------------------------------------------------------------ probes
def probe_contract_digest():
    _, c, L = fixture(seed=31)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    other = SNAP.C.seal(dict(c, original_fields={"forged": True}))
    object.__setattr__(s, "contract_json", SNAP.C.canonical(other))
    return lambda: OPS.fit(EWMA, s, FROZEN)


def probe_matrix_digest():
    """The recorded matrix digest is forged and the snapshot re-sealed; the bytes still re-derive from the
    source, so only the matrix digest check can refuse (a fit would bind a digest of bytes that never existed)."""
    _, c, L = fixture(seed=32)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    object.__setattr__(s, "matrix_sha256", "0" * 64)
    object.__setattr__(s, "snapshot_sha256", SNAP.C.sha_obj(s.facts()))
    return lambda: OPS.fit(EWMA, s, FROZEN)


def probe_source_rederive():
    _, c, L = fixture(seed=33)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    m = s.matrix.copy()
    m[0, 0] += 1.0
    m.flags.writeable = False
    object.__setattr__(s, "matrix", m)
    _redigest(s)
    return lambda: OPS.fit(EWMA, s, FROZEN)


def probe_snapshot_digest():
    _, c, L = fixture(seed=34)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    object.__setattr__(s, "timestamp_meaning", "INSTANT")
    return lambda: OPS.fit(EWMA, s, FROZEN)


def probe_monotonic_batch():
    X = base_series(100, 2, 35)
    ts = np.arange(100, dtype=np.int64) * 60
    ts[70] = ts[69]
    c, L = SYNC.in_memory_contract(X, name="fx_dup", timestamps=ts)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100))


def probe_monotonic_step():
    _, c, L = fixture(seed=36)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    s = SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100)
    st = OPS.init_state(f, s)
    return lambda: OPS.step(f, st, s.row(1))


def probe_range_in_partition():
    _, c, L = fixture(seed=37)
    return lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "CALIBRATION", L, start=0, end=60), FROZEN)


def probe_role_allowed():
    _, c, L = fixture(n=300, seed=38)            # 60 confirmation rows: only the role check can refuse
    return lambda: OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "CONFIRMATION", L), FROZEN)


def probe_later_exclusion():
    _, c, L = fixture(seed=39)
    s = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    object.__setattr__(s, "excluded_partitions", ())
    _redigest(s)
    return lambda: OPS.fit(EWMA, s, FROZEN)


def probe_availability():
    X = base_series(100, 2, 40)
    c, L = SYNC.in_memory_contract(X, name="fx_delay", timestamps=np.arange(100) * 60, availability_delay=30)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100))


def probe_artifact_bound():
    X, c, L = fixture(seed=41)
    f = OPS._fit_kernel(EWMA, X[:60], FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100))


def _repartitioned(seed: int):
    """The same bytes under two contracts: identical dataset id and column ids (both derive from the bytes),
    different partitions, so different contract digests. B's CALIBRATION [50, 75) overlaps A's TRAIN [0, 60)."""
    X = base_series(100, 2, seed)
    ca, La = SYNC.in_memory_contract(X, name="fx_repartitioned")
    cb, Lb = SYNC.in_memory_contract(X, name="fx_repartitioned", fractions=(0.5, 0.25, 0.25))
    return ca, La, cb, Lb


def probe_dataset_binding():
    ca, La, cb, Lb = _repartitioned(42)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(ca, "TRAIN", La), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(cb, Lb, start=60, end=100))


def probe_column_identity():
    _, c, L = fixture(seed=44)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L, columns=["v0", "v1"]), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100,
                                                                              columns=["v1", "v0"]))


def probe_partition_license():
    _, c, L = fixture(seed=45)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), FROZEN)
    return lambda: OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=0, end=100))


def probe_fit_mode():
    _, c, L = fixture(seed=46)
    return lambda: OPS.transform_batch(OPS.fit(HAAR, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), EXPANDING),
                                       SNAP.TransformSnapshot.from_contract(c, L, start=0, end=60))


def probe_stream_binding():
    ca, La, cb, Lb = _repartitioned(47)
    f = OPS.fit(EWMA, SNAP.FitSnapshot.from_contract(ca, "TRAIN", La), FROZEN)
    st = OPS.init_state(f, SNAP.TransformSnapshot.from_contract(ca, La, start=60, end=100))
    sb = SNAP.TransformSnapshot.from_contract(cb, Lb, start=60, end=100)
    return lambda: OPS.step(f, st, sb.row(0))


def probe_exhaustive_battery():
    """A one-sample leak at t=100: the battery's declared prefix cuts must detect it."""
    BAT = _load("df_causal_battery")
    X = base_series(240, 2, 49)
    f = OPS._fit_kernel(EWMA, base_series(300, 2, 99), FROZEN)

    def leaking(Z):
        y, a, r = OPS._transform_kernel(f, Z)
        y = y.copy()
        if Z.shape[0] > 101:
            y[100] = Z[101]
        return {"y": y}, a, r

    def check():
        fails = BAT.prefix_all_t(leaking, X, BAT.cuts_for_prefix(240, EWMA, FROZEN))
        if fails:
            raise OPS.OperatorRefusal(f"battery detected a future leak: {fails[None]}")
    return check


def wavelet_later_train_rows_move_early_outputs() -> dict:
    """The C166.3 case: fit trailing_haar_threshold EXPANDING on TRAIN of X and of X with train rows 31..59
    rescaled (x5+3), transform rows 0..30 of each. Productive code refuses the fit; a mutant that accepts it
    lets later training rows move outputs at t <= 30."""
    X = base_series(100, 2, 50)
    X2 = X.copy()
    X2[31:60] = X2[31:60] * 5.0 + 3.0
    outs = []
    for Z, nm in ((X, "a"), (X2, "b")):
        c, L = SYNC.in_memory_contract(Z, name=f"fx_leak_{nm}")
        try:
            f = OPS.fit(HAAR, SNAP.FitSnapshot.from_contract(c, "TRAIN", L), EXPANDING)
            y, _, _ = OPS.transform_batch(f, SNAP.TransformSnapshot.from_contract(c, L, start=0, end=31))
        except REFUSALS as exc:
            return {"outcome": "REFUSED", "message": str(exc), "outputs_t_le_30_moved": None}
        outs.append(y)
    moved = not np.array_equal(outs[0], outs[1], equal_nan=True)
    return {"outcome": "ACCEPTED", "message": None, "outputs_t_le_30_moved": bool(moved)}


# guard -> (probe, operator spec, expected refusal text)
PROBES = {
    "contract_digest": (probe_contract_digest, EWMA, "contract digest does not re-derive"),
    "matrix_digest": (probe_matrix_digest, EWMA, "matrix digest does not re-derive"),
    "source_rederive": (probe_source_rederive, EWMA, "source bytes do not re-derive"),
    "snapshot_digest": (probe_snapshot_digest, EWMA, "snapshot digest does not re-derive"),
    "monotonic_timestamps": (probe_monotonic_batch, EWMA, "not strictly increasing"),
    "monotonic_timestamps.step": (probe_monotonic_step, EWMA, "is not the next row"),
    "range_in_partition": (probe_range_in_partition, EWMA, "is not inside the CALIBRATION partition"),
    "role_allowed": (probe_role_allowed, EWMA, "is not allowed by the design"),
    "later_partition_exclusion": (probe_later_exclusion, EWMA, "later partitions are not excluded"),
    "availability": (probe_availability, EWMA, "available after its decision instant"),
    "artifact_bound": (probe_artifact_bound, EWMA, "not bound to a FitSnapshot"),
    "dataset_binding": (probe_dataset_binding, EWMA, "another dataset or contract"),
    "column_identity": (probe_column_identity, EWMA, "columns differ from the fitted columns"),
    "transform_partition_license": (probe_partition_license, EWMA, "is not licensed"),
    "fit_mode_enforcement": (probe_fit_mode, HAAR, "is not implemented for kind"),
    "stream_binding": (probe_stream_binding, EWMA, "another series"),
    "exhaustive_cuts": (probe_exhaustive_battery, EWMA, "battery detected a future leak: prefix differs at t=100"),
}


def attempt(thunk) -> dict:
    """-> {"outcome": REFUSED | ACCEPTED | EXCEPTION, "message"}. Only a typed refusal counts as REFUSED."""
    try:
        thunk()
    except REFUSALS as exc:
        return {"outcome": "REFUSED", "message": str(exc)}
    except Exception as exc:  # noqa: BLE001 - any other failure is recorded, never counted as a refusal
        return {"outcome": "EXCEPTION", "message": f"{type(exc).__name__}: {exc}"}
    return {"outcome": "ACCEPTED", "message": None}


def run_probe(guard: str) -> dict:
    make, _, expected = PROBES[guard]
    res = attempt(make())
    res.update(guard=guard, expected=expected,
               expected_refusal=res["outcome"] == "REFUSED" and expected in (res["message"] or ""))
    if guard == "fit_mode_enforcement":
        res["wavelet"] = wavelet_later_train_rows_move_early_outputs()
    return res


def _loaded_files() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and Path(f).resolve().is_relative_to(ROOT):
            p = Path(f).resolve()
            out[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    me = Path(__file__).resolve()
    out[str(me.relative_to(ROOT))] = hashlib.sha256(me.read_bytes()).hexdigest()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--guard", required=True, choices=sorted(PROBES))
    a = ap.parse_args(argv)
    res = run_probe(a.guard)
    res["loaded_files"] = _loaded_files()
    res["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    print(RESULT_MARKER + json.dumps(res, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
