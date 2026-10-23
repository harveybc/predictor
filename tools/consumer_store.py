#!/usr/bin/env python3
"""Consumer store after the fold-scope correction (Musashi's final joint audit A1, 2026-10-23): the bound merged pool
with the CONSUMER scope's fold-local columns substituted (learned producers fitted on 2020-01-01 .. 2023-07-01 by
availability; the inner block 2023 S2, 2024 and TEST never fitted on; the activity model cross-fitted). Every other
column is copied byte for byte. The manifest records the parent pool, the scope report and the changed columns."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, help="merged pool parquet (MANIFEST.json beside it)")
    ap.add_argument("--scope-parquet", required=True)
    ap.add_argument("--scope-report", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    pool_p, man0 = Path(a.pool), json.loads((Path(a.pool).parent / "MANIFEST.json").read_text())
    if sha(pool_p) != man0["file_sha256"]:
        raise SystemExit("POOL_IDENTITY")
    rep = json.loads(Path(a.scope_report).read_text())
    if rep.get("sha256") != sha(a.scope_parquet) or rep["scope"]["kind"] != "consumer":
        raise SystemExit("SCOPE_IDENTITY")
    pool = pd.read_parquet(pool_p)
    sc = pd.read_parquet(a.scope_parquet)
    if not pd.DatetimeIndex(pd.to_datetime(sc["timestamp"], utc=True)).equals(pd.DatetimeIndex(pd.to_datetime(pool["timestamp"], utc=True))):
        raise SystemExit("SCOPE_GRID_MISMATCH")
    changed = [c for c in sc.columns if c != "timestamp"]
    missing = [c for c in changed if c not in pool.columns]
    if missing:
        raise SystemExit(f"SCOPE_COLUMNS_NOT_IN_POOL:{missing}")
    diffs = {}
    for c in changed:
        g, v = pool[c].to_numpy(float), sc[c].to_numpy(float)
        if not np.array_equal(np.isfinite(g), np.isfinite(v)):
            raise SystemExit(f"CONSUMER_SUPPORT_MOVED:{c}")
        both = np.isfinite(g)
        diffs[c] = float(np.max(np.abs(g[both] - v[both]))) if both.any() else 0.0
        pool[c] = v
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / pool_p.name
    pool.to_parquet(f, index=False)
    man = {k: v for k, v in man0.items() if k not in ("file_sha256",)}
    man.update(file_sha256=sha(f), features=[c for c in pool.columns if c != "timestamp"],
               parent_pool={"file": pool_p.name, "sha256": man0["file_sha256"]},
               consumer_scope={"name": rep["scope"]["name"], "fit_window": rep["scope"]["fit_window"], "report_sha256": sha(a.scope_report),
                               "parquet_sha256": rep["sha256"], "changed_columns": changed, "max_abs_change": diffs,
                               "rule": "learned producers fitted on the consumer scope only; decisions unchanged (decided on 2020-2022, inside the fit window)"})
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({"file": f.name, "sha256": man["file_sha256"], "changed": len(changed)}))


if __name__ == "__main__":
    main()
