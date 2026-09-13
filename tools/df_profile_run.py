#!/usr/bin/env python3
"""C130-C133 run: raw profiles, without targets, over the three banks.

For every dataset contract:

* public: each canonical panel of the C126 v2 root (panel.parquet with its
  sealed CONTRACT.json), timestamps parsed from the panel's labels with the
  parser's declared format;
* synthetic: each unit of the C128 bank, profiled on OBSERVED only, on the
  sample index. The clean and noise truth arrays are read only to verify the
  unit's recorded digests; they are never passed to a profile module;
* financial: each file of the C127 first batch, read from the financial-data
  checkout after its bytes are re-hashed against the contract, timestamps from
  its `timestamp` column.

the four modules run on the dataset's numeric variables: univariate
(df_profile_univariate), information (df_profile_information), multivariate
(df_profile_multivariate, train only, with its declared pair cap) and sampling
(df_sampling). Text variables are not profiled numerically and are listed as
such. Every module row is written unchanged, tagged with its module, one JSONL
file per dataset, in a write-once root with a receipt that binds the code, the
contracts and every output file by digest. A dataset that fails is recorded
with its error and never stops the others.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
MODULES = ("df_profile_univariate", "df_profile_information", "df_profile_multivariate", "df_sampling")
RUNNERS = {"df_profile_univariate": "run_univariate", "df_profile_information": "run_information",
           "df_profile_multivariate": "run_multivariate", "df_sampling": "run_sampling"}


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_name(dataset_id: str) -> str:
    return hashlib.sha256(dataset_id.encode()).hexdigest()[:20] + "__" + "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_id)[-80:]


# ----------------------------------------------------------------- inputs
def public_jobs(panel_root: Path) -> list[dict]:
    return [{"bank": "PUBLIC", "dir": str(d)} for d in sorted(Path(panel_root).iterdir())
            if d.is_dir() and (d / "CONTRACT.json").is_file()]


def synthetic_jobs(bank_root: Path) -> list[dict]:
    return [{"bank": "SYNTHETIC", "dir": str(d)} for d in sorted(Path(bank_root).iterdir()) if d.is_dir()]


def financial_jobs(contracts_file: Path, financial_root: Path) -> list[dict]:
    doc = json.loads(Path(contracts_file).read_text())
    return [{"bank": "FINANCIAL", "index": i, "contracts_file": str(contracts_file), "root": str(financial_root)}
            for i in range(len(doc["contracts"]))]


def load_job(job: dict):
    """-> (contract, X (T, V) float with NaN, timestamps int64 ns or None, skipped text variables)."""
    if job["bank"] == "PUBLIC":
        d = Path(job["dir"])
        contract = json.loads((d / "CONTRACT.json").read_text())
        panel = next(f for f in contract["files"] if f["role"] == "DERIVED_CANONICAL_PANEL")
        if _sha_file(d / "panel.parquet") != panel["sha256"]:
            raise ValueError("panel bytes differ from the contract")
        table = pq.read_table(d / "panel.parquet")
        parser = contract["original_fields"]["parse_receipt"]["parser"]
        fmt = _load("df_public_contract").TIMESTAMP_FORMATS[parser]
        ts = pd.to_datetime(pd.Series(table.column("timestamp_label").to_pylist()), format=fmt).to_numpy("datetime64[ns]").astype("int64")
        get = lambda name: table.column(name).to_numpy(zero_copy_only=False)  # noqa: E731
    elif job["bank"] == "SYNTHETIC":
        d = Path(job["dir"])
        sync = _load("df_synthetic_contract")
        contract = sync.unit_contract(d)
        obs = np.load(d / "observed_signal.npy", allow_pickle=False)
        rec = contract["original_fields"]["unit_record"]
        if obs.shape == (rec["n_variables"], rec["n_samples"]):
            obs = obs.T
        cols = {v["name"]: obs[:, i] for i, v in enumerate(contract["variables"])}
        ts = None
        get = cols.__getitem__
    else:
        doc = json.loads(Path(job["contracts_file"]).read_text())
        contract = doc["contracts"][job["index"]]
        f = contract["files"][0]
        path = Path(job["root"]) / f["name"]
        if _sha_file(path) != f["sha256"]:
            raise ValueError("financial file bytes differ from the contract")
        table = pq.read_table(path)
        ts = None
        if "timestamp" in table.column_names:
            ts = pd.to_datetime(pd.Series(table.column("timestamp").to_pylist()), utc=True).to_numpy("datetime64[ns]").astype("int64")
        get = lambda name: table.column(name).to_numpy(zero_copy_only=False)  # noqa: E731
    numeric, skipped, columns = [], [], []
    for v in contract["variables"]:
        if v["name"] == "timestamp" or v["role"] == "TIMESTAMP":
            continue
        try:
            arr = np.asarray(get(v["name"]))
        except (KeyError, Exception):  # noqa: BLE001
            skipped.append({"variable_id": v["variable_id"], "name": v["name"], "reason": "COLUMN_NOT_READABLE"})
            continue
        if arr.dtype.kind not in "fiu":
            skipped.append({"variable_id": v["variable_id"], "name": v["name"], "reason": "NON_NUMERIC_NOT_PROFILED"})
            continue
        numeric.append(v)
        columns.append(np.asarray(arr, dtype="float64"))
    X = np.column_stack(columns) if columns else np.empty((0, 0))
    sub = dict(contract, variables=numeric)
    return sub, X, ts, skipped


def run_job(job: dict) -> dict:
    t0 = time.time()
    try:
        contract, X, ts, skipped = load_job(job)
        rows = []
        if X.size:
            for m in MODULES:
                mod = _load(m)
                for r in getattr(mod, RUNNERS[m])(contract, X, ts):
                    rows.append({"module": m, "row": r})
        return {"job": job, "status": "COMPLETED", "dataset_id": contract["dataset_id"],
                "contract_sha256": contract["contract_sha256"], "numeric_variables": len(contract["variables"]),
                "skipped_variables": skipped, "rows": rows, "wall_seconds": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001 - every outcome is recorded
        return {"job": job, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:500],
                "rows": [], "wall_seconds": round(time.time() - t0, 2)}


def run(out_dir: Path, jobs: list[dict], workers: int = 4) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise SystemExit(f"REFUSED: {out_dir.name} exists; profile outputs are write-once")
    out_dir.mkdir(parents=True)
    results = []

    def keep(res):
        entry = {k: v for k, v in res.items() if k != "rows"}
        job = entry.pop("job")
        entry["bank"] = job["bank"]
        if res["status"] == "COMPLETED":
            p = out_dir / job["bank"].lower() / f"{safe_name(res['dataset_id'])}.jsonl"
            p.parent.mkdir(exist_ok=True)
            p.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in res["rows"]))
            entry.update(file=str(p.relative_to(out_dir)), row_count=len(res["rows"]), sha256=_sha_file(p))
        results.append(entry)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for res in pool.map(run_job, jobs, chunksize=1):
                keep(res)
    else:
        for job in jobs:
            keep(run_job(job))
    counts = {s: sum(r["status"] == s for r in results) for s in ("COMPLETED", "FAILED")}
    receipt = {"schema": "crispdm.data_foundation.profile_run_receipt.v1",
               "code_sha256": {m: _sha_file(HERE / f"{m}.py") for m in MODULES + ("df_profile_run",)},
               "datasets": results, "counts": counts,
               "rows_total": sum(r.get("row_count", 0) for r in results),
               "rule": "profiles read observed data only; no target; fits and frozen bins on train only"}
    text = json.dumps(receipt, indent=1, sort_keys=True).replace(str(Path.home()), "~")
    (out_dir / "PROFILE_RUN_RECEIPT.json").write_text(text + "\n")
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--public-panels", type=Path)
    ap.add_argument("--synthetic-bank", type=Path)
    ap.add_argument("--financial-contracts", type=Path)
    ap.add_argument("--financial-root", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="per bank, for smoke runs")
    a = ap.parse_args(argv)
    jobs = []
    if a.public_panels:
        jobs += public_jobs(a.public_panels)[:a.limit]
    if a.synthetic_bank:
        jobs += synthetic_jobs(a.synthetic_bank)[:a.limit]
    if a.financial_contracts:
        jobs += financial_jobs(a.financial_contracts, a.financial_root)[:a.limit]
    r = run(a.out, jobs, a.workers)
    print(json.dumps({"counts": r["counts"], "rows_total": r["rows_total"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
