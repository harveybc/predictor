#!/usr/bin/env python3
"""Lane H: parity, restart and future-leak proof of the Kalman family on a real or synthetic panel. DEVELOPMENT.

For each kind: fit on the first ``n_train`` rows (TRAIN only), then over ALL rows compare
  batch == tick-by-tick == restart-from-a-durable-state-blob-in-a-SEPARATE-PROCESS == chunked continuation
by SHA-256 of the output arrays; probe future leakage (perturb every row after t, outputs up to t must not move, and the
non-causal smoother must move so the probe is shown able to fail); check the smoother is refused as an input.
Digests are order- and environment-independent by construction; the same command on two workers must print the same
``digest_summary``.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


kf = _load("df_kalman_family")

_CHILD = r"""
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("df_kalman_family", sys.argv[1]); kf = importlib.util.module_from_spec(spec)
sys.modules["df_kalman_family"] = kf; spec.loader.exec_module(kf)
art = json.load(open(sys.argv[2])); blob = open(sys.argv[3], "rb").read(); X = np.load(sys.argv[4])
state = kf.load_state(blob, art)
out, state = kf.transform_chunk(art, state, X)
np.savez(sys.argv[5], **{k: v for k, v in out.arrays.items()}, reasons=out.reasons)
open(sys.argv[6], "wb").write(kf.save_state(state))
"""


def _digest_arrays(arrays: dict, reasons) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        h.update(k.encode())
        h.update(np.ascontiguousarray(arrays[k], dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(reasons).tobytes())
    return h.hexdigest()


def run_proof(Z, n_train, names, split_row, workdir, spec_kinds=None, leak_rows=(60, 400, None)):
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    res = {"schema": "lane_h_kalman_parity_proof.v1", "label": "DEVELOPMENT", "rows": int(Z.shape[0]),
           "columns": int(Z.shape[1]), "n_train": int(n_train), "split_row": int(split_row),
           "environment": kf.environment_record(os.environ.get("H1_HOST_ROLE")), "kinds": {}, "artifact_digests": {}}
    summary = {}
    for kind in spec_kinds or list(kf.CAUSAL_KINDS):
        art = kf.fit(kf.default_spec(kind), Z[:n_train], {"dataset_id": "proof", "role": "TRAIN", "row_range": [0, n_train],
                                                       "column_ids": list(names)}, host_role=os.environ.get("H1_HOST_ROLE"))
        res["artifact_digests"][kind] = art["artifact_sha256"]
        t0, c0 = time.time(), time.process_time()
        batch, st_batch = kf.transform_batch(art, Z, _return_state=True)
        t_batch = time.process_time() - c0
        bd = _digest_arrays(batch.arrays, batch.reasons)
        # tick by tick
        st = kf.init_state(art)
        names_out = kf.OUTPUTS[kind]
        tick = {n: np.empty_like(batch.arrays[n]) for n in names_out}
        reasons = np.empty_like(batch.reasons)
        c1 = time.process_time()
        for t in range(Z.shape[0]):
            o, why, st = kf.step(art, st, Z[t])
            for j in range(Z.shape[1]):
                for k, n in enumerate(names_out):
                    tick[n][t, j] = o[j][k]
                reasons[t, j] = why[j]
        t_tick = time.process_time() - c1
        td = _digest_arrays(tick, reasons)
        # restart: first part in this process, durable blob to disk, second part in a SEPARATE process
        first, st_first = kf.transform_batch(art, Z[:split_row], _return_state=True)
        blob_path = workdir / f"state_{kind}.json"
        blob_path.write_bytes(kf.save_state(st_first))
        art_path = workdir / f"artifact_{kind}.json"
        art_path.write_text(json.dumps(art))
        np.save(workdir / "tail.npy", Z[split_row:])
        out_npz, end_blob = workdir / f"tail_{kind}.npz", workdir / f"end_{kind}.json"
        subprocess.run([sys.executable, "-c", _CHILD, str(HERE / "df_kalman_family.py"), str(art_path), str(blob_path),
                        str(workdir / "tail.npy"), str(out_npz), str(end_blob)], check=True, env=dict(os.environ))
        with np.load(out_npz) as z:
            tail = {n: z[n] for n in names_out}
            tail_reasons = z["reasons"]
        joined = {n: np.vstack([first.arrays[n], tail[n]]) for n in names_out}
        rd = _digest_arrays(joined, np.vstack([first.reasons, tail_reasons]))
        # chunked continuation inside one process (three chunks)
        st_c = kf.init_state(art)
        parts, preasons = [], []
        for lo, hi in ((0, split_row // 2), (split_row // 2, split_row + 100), (split_row + 100, Z.shape[0])):
            o, st_c = kf.transform_chunk(art, st_c, Z[lo:hi])
            parts.append(o.arrays)
            preasons.append(o.reasons)
        cd = _digest_arrays({n: np.vstack([p[n] for p in parts]) for n in names_out}, np.vstack(preasons))
        # future-leak probe
        moves, checked = 0.0, 0
        rng = np.random.RandomState(0)
        for t in [r for r in leak_rows if r is not None] + [Z.shape[0] // 2, Z.shape[0] - 2]:
            for mode in ("shift", "noise", "nan"):
                Zp = Z.copy()
                if mode == "shift":
                    Zp[t + 1:] += 7.0
                elif mode == "noise":
                    Zp[t + 1:] = rng.standard_normal(Zp[t + 1:].shape) * 100
                else:
                    Zp[t + 1:] = np.nan
                o = kf.transform_batch(art, Zp)
                for n in names_out:
                    a, b = o.arrays[n][:t + 1], batch.arrays[n][:t + 1]
                    same = np.array_equal(a.view("<i8"), b.view("<i8"))
                    moves = max(moves, 0.0 if same else float(np.nanmax(np.abs(a - b))) or 1.0)
                checked += 1
        sm0 = kf.smoother_control(art, Z)
        Zp = Z.copy(); Zp[Z.shape[0] // 2:] += 7.0
        sm1 = kf.smoother_control(art, Zp)
        oracle = not np.array_equal(sm0.arrays["level"][:Z.shape[0] // 2], sm1.arrays["level"][:Z.shape[0] // 2])
        refused = False
        try:
            kf.eligible_matrix(sm0)
        except kf.OperatorRefusal:
            refused = True
        res["kinds"][kind] = {"batch_digest": bd, "tick_digest": td, "restart_digest": rd, "chunk_digest": cd,
                              "restart_used_separate_process": True, "state_digest_end": kf.state_digest(st_batch),
                              "end_state_after_restart_equals": kf.state_digest(kf.load_state(end_blob.read_bytes(), art)) == kf.state_digest(st_batch),
                              "leak_probe": {"rows_checked": checked, "max_abs_move_before_t": moves, "oracle_detects_leak": bool(oracle)},
                              "smoother_refused_as_input": refused,
                              "cpu_seconds": {"batch": t_batch, "tick_by_tick": t_tick},
                              "reason_counts": batch.reason_counts()}
        summary[kind] = {"artifact": art["artifact_sha256"], "batch": bd, "tick": td, "restart": rd, "chunk": cd,
                         "end_state": kf.state_digest(st_batch)}
    res["digest_summary"] = summary
    res["all_equal"] = all(r["batch_digest"] == r["tick_digest"] == r["restart_digest"] == r["chunk_digest"] and
                           r["end_state_after_restart_equals"] and r["leak_probe"]["max_abs_move_before_t"] == 0.0 and
                           r["leak_probe"]["oracle_detects_leak"] and r["smoother_refused_as_input"] for r in res["kinds"].values())
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--view", required=True)
    ap.add_argument("--feature-manifest", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", required=True)
    a = ap.parse_args()
    os.environ["H1_HOST_ROLE"] = a.role
    sys.path.insert(0, str(HERE.parent))
    eth_run = _load("h_kalman_eth_run")
    fm = json.loads(Path(a.feature_manifest).read_text())
    features = list(fm["features"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = eth_run.load_data(a.view, features, out / "data", expected_sha=eth_run.eth.VIEW_SHA256)
    cands = json.loads(Path(a.candidates).read_text())
    ds = [i for i, x in enumerate(cands["datasets"]) if x["manifest_canonical"].startswith("30b78078")][0]
    groups = eth_run.groups_from_candidates(cands, ds, features)
    end = d["val_rows"][1]
    results = {}
    for g, kind in (("local_level", kf.LOCAL_LEVEL), ("local_linear_trend", kf.LOCAL_LINEAR_TREND)):
        idx = [features.index(c) for c in groups[g]]
        r = run_proof(d["Z"][:end][:, idx], n_train=d["train_rows"][1], names=groups[g], split_row=9000, workdir=out / f"work_{g}",
                      spec_kinds=[kind])
        results[g] = r
    doc = {"schema": "lane_h_kalman_parity_proof_eth.v1", "role": a.role, "results": results,
           "digest_summary": {g: r["digest_summary"] for g, r in results.items()},
           "all_equal": all(r["all_equal"] for r in results.values())}
    (out / "PARITY_PROOF.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"all_equal": doc["all_equal"], "digest_summary": doc["digest_summary"]}))


if __name__ == "__main__":
    main()
