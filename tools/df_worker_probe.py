#!/usr/bin/env python3
"""RP78: one BOUNDED synthetic governed child on the host that runs this — under that host's own actor key.

What it proves, per host: the deployed governance path from THIS machine (campaign registered before any byte,
authenticated delivery of a SYNTHETIC resource with its content hash, verification of the bytes on disk), a
bounded compute inside the accounted wrapper (the block runner's own update loop on synthetic windows: a real
fit, tiny, no scientific meaning), the measured cost (CPU of this process AND its children, peak RSS) in a
terminal that reaches the accounting through the outbox, and the campaign's reconciliation. The receipt names
the code identity (a clean checkout is required, as for every governed unit) and the environment.

    python tools/df_worker_probe.py --gov-url URL --api-key-file KEY --lake governance_smoke \\
        --resource synthetic_features_4h_train.csv --out RECEIPT.json [--updates 200]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _module(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bounded_fit(seed: int, updates: int) -> dict:
    """The block runner's own loop on synthetic windows: y(t+h) = 2 x(t) on the target channel; tiny and real."""
    K = _module("df_e1_block")
    tf = _module("df_mod_e0")._tf()
    rng = np.random.default_rng(seed)
    n, h, W = 3000, 5, 30
    Xs = rng.normal(size=(n+h+1, 3)).astype(np.float32)
    Y = np.zeros(n+h+1)
    Y[h:] = 2.0*Xs[:-h, 2]
    o = np.arange(W, n)
    tr = K.Batches(Xs, Y, o[:2200], W, h, 2, 64, mean=0.0, sd=1.0, shuffle=True, seed=seed)
    va = K.Batches(Xs, Y, o[2200:], W, h, 2, 64, mean=0.0, sd=1.0, shuffle=False, seed=seed)
    model = K.build_modular([0, 0, 0], W, 3, 2, seed)
    before = K.evaluate_mae(model, va)
    r = K.fit_by_updates(model, tr, va, max_updates=updates, validate_every=max(1, updates//4), patience=3, lr=0.003, seed=seed)
    return {"parameters": K.n_params(model), "val_mae_before": before, "val_mae_after": r["restored_val_mae_scaled"],
            "updates": r["updates"], "optimizer_iterations": r["optimizer_iterations"], "stop_reason": r["stop_reason"],
            "cpu": r["cpu"], "learned": bool(r["restored_val_mae_scaled"] < 0.8*before)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gov-url", required=True)
    ap.add_argument("--api-key-file", type=Path, required=True)
    ap.add_argument("--lake", default="governance_smoke")
    ap.add_argument("--resource", default="synthetic_features_4h_train.csv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    host = socket.gethostname()
    run_id = a.run_id or f"rp78-worker-probe-{host}-{int(time.time())}"
    root = Path(a.root or (Path.home()/".local/state/crispdm-data-foundation"/f"rp78_probe_{host}"/run_id))
    root.mkdir(parents=True, exist_ok=True)
    G = _module("df_e1_governed")
    G._load("governed_run")
    G._load("df_e1_receipts")
    U = _module("df_utility_run")
    started = U._z(U.now_iso())
    unit = f"{host}-bounded-child"
    wall0, cpu0 = time.monotonic(), time.process_time()
    design_sha = hashlib.sha256(json.dumps({"schema": "rp78_worker_probe.v1", "host": host, "updates": a.updates, "seed": a.seed}, sort_keys=True).encode()).hexdigest()
    doc = G.acquire(run_id=run_id, root=root, lake=a.lake, resource=a.resource, unit_id=unit, role="synthetic", gov_url=a.gov_url,
                    api_key_file=a.api_key_file, design_sha256=design_sha, cache_dir=root/"cache")
    delivered = doc["units"][unit]
    on_disk = sha_file(Path(delivered["path"]))
    if on_disk != delivered["sha256"]:
        raise SystemExit("REFUSED: the delivered bytes on disk are not the ones the stream declared")
    fit = bounded_fit(a.seed, a.updates)
    ru_self, ru_child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    cost = {"wall_seconds": time.monotonic()-wall0, "cpu_seconds": (ru_self.ru_utime+ru_self.ru_stime+ru_child.ru_utime+ru_child.ru_stime),
            "cpu_seconds_self": ru_self.ru_utime+ru_self.ru_stime, "cpu_seconds_children": ru_child.ru_utime+ru_child.ru_stime,
            "peak_rss_bytes": int(max(ru_self.ru_maxrss, ru_child.ru_maxrss))*1024}
    terminal = U._terminal(status="COMPLETED", reason=None, cost=cost,
                           metrics=[U._metric("rp78.probe.val_mae_after", fit["val_mae_after"], "scaled", split="synthetic", horizon=5),
                                    U._metric("rp78.probe.updates", fit["updates"], "updates", split="synthetic", horizon=5),
                                    U._metric("rp78.probe.peak_rss_bytes", cost["peak_rss_bytes"], "bytes", split="synthetic", horizon=5)],
                           started=started, finished=U._z(U.now_iso()),
                           tags={"purpose": "RP78_WORKER_ACCEPTANCE", "classification": "NON_GOVERNING", "phase": "DEVELOPMENT", "unit": unit,
                                 "host": host, "design_sha256": design_sha, "scope": os.environ.get("CRISPDM_SCOPE", "unknown")})
    terminal["artifacts"] = [{"role": "delivered_bytes", "sha256": on_disk, "bytes": Path(delivered["path"]).stat().st_size}]
    reported = G.report_terminal(root, unit, terminal, gov_url=a.gov_url, api_key_file=a.api_key_file, outbox_dir=str(root/"outbox"), started_at=started)
    receipt = {"schema": "rp78_worker_probe_receipt.v1", "host": host, "run_id": run_id, "unit": unit, "root": str(root),
               "code_identity": delivered.get("code_identity"), "campaign_sha256": delivered["campaign_sha256"],
               "delivery": {k: delivered.get(k) for k in ("delivery_id", "sha256", "bytes", "cached", "verification_state", "availability_use")},
               "fit": fit, "cost": cost, "terminal_sha256": (reported.get("receipt") or {}).get("terminal_sha256"),
               "flushed": reported["flushed"], "reconciliation": reported["reconciliation"],
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "tensorflow": _module("df_mod_e0")._tf().__version__, "machine": platform.machine(), "kernel": platform.release()},
               "wrapper_scope_hint": "run under crispdm-run: the journal's scope line (Consumed CPU / memory peak) is the wrapper's accounting"}
    a.out.write_text(json.dumps(receipt, indent=1, default=str))
    print(json.dumps({k: receipt[k] for k in ("host", "campaign_sha256", "terminal_sha256", "cost", "reconciliation")}, indent=1, default=str))
    return 0 if reported["flushed"]["sent"] and not reported["reconciliation"].get("missing_units") else 1


if __name__ == "__main__":
    raise SystemExit(main())
