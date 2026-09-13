#!/usr/bin/env python3
"""C169 (order 2026-09-13): the same linear-algebra fixture on every role and interpreter.

Run on each host (COORDINATOR, WORKER_A, WORKER_B) with each interpreter, then
compare the reports:

    crispdm-run -m 1G -t 10m -n parity -- env -u PYTHONPATH CUDA_VISIBLE_DEVICES= \\
        <python> tools/df_parity_fixture.py --interpreter <python> --role COORDINATOR --out <dir>/coord_base.json
    <python> tools/df_parity_fixture.py --compare <dir>/*.json

The fixture matrix is built from integer arithmetic only (a +-1 random walk
from a 64-bit LCG, scaled by powers of two), so its float64 bytes are identical
everywhere; the report binds their digest. The child (the given interpreter)
runs the production matrix descriptors of df_profile_multivariate
(`_matrix_rows`) and df_profile_information (`matrix_rows`) on that matrix in
C and Fortran memory layout, and reports every row with its raw value, its
canonical value, numpy version and linear-algebra backends.

--compare checks, across every report and layout: identical fixture digest
and tolerance declaration; tolerance-declared metrics agree within the
declared tolerance and have identical canonical values; every other value is
exactly equal. Exit 0 only when all of it holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "crispdm.data_foundation.c169_linalg_parity_fixture.v1"
T, V = 3000, 7


def _load(name):
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fixture_values() -> list[list[float]]:
    """(T, V) values from integer arithmetic only: exact dyadic floats, NaN at fixed positions."""
    state = 0x9E3779B97F4A7C15
    walk = 0
    rows = []
    for t in range(T):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        walk += 1 if (state >> 33) & 1 else -1
        row = []
        for j in range(V):
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            noise = ((state >> 40) - (1 << 23)) / float(1 << 23)           # [-1, 1), dyadic
            v = (j + 1) * walk / 64.0 + noise * (0.25 + j / 8.0)
            if j == 3:
                v = -v                                                    # a negative loading
            if (t * 7 + j * 13) % 211 == 0:
                v = float("nan")
            row.append(v)
        rows.append(row)
    return rows


def child() -> dict:
    import numpy as np
    LP = _load("df_linalg_parity")
    M = _load("df_profile_multivariate")
    I = _load("df_profile_information")
    X = np.array(fixture_values(), dtype="<f8")
    ids = [f"fixture.v{j}" for j in range(V)]
    layouts = {}
    for name, arr in (("C", np.ascontiguousarray(X)), ("F", np.asfortranarray(X))):
        with M.blas_single_thread():
            rows = [("df_profile_multivariate", r) for r in M._matrix_rows("fixture", ids, arr)]
            rows += [("df_profile_information", r) for r in I.matrix_rows("fixture", V, lambda s, e: arr[s:e], T)]
        layouts[name] = [{"module": m, "key": r.get("variable_id") or r.get("group_id"), "metric": r["metric"],
                          "status": r["status"], "value": r["value"], "value_canonical": r.get("value_canonical")}
                         for m, r in rows]
    prov = LP.linalg_provenance()
    return {"schema": SCHEMA, "python": sys.version.split()[0], "env": Path(sys.prefix).name,
            "numpy_version": prov["numpy_version"], "linalg_backends": prov["linalg_backends"],
            "tolerance_sha256": LP.TOLERANCES_SHA256,
            "fixture_sha256": hashlib.sha256(X.tobytes()).hexdigest(), "shape": [T, V], "layouts": layouts,
            "portable_digest": portable_digest(layouts["C"], LP)}


def portable_digest(rows, LP) -> str:
    """Digest of canonical values for tolerance metrics and exact values for the rest."""
    items = [[r["module"], r["key"], r["metric"], r["status"],
              r["value_canonical"] if LP.tolerance_for(r["module"], r["metric"]) else r["value"]] for r in rows]
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def compare(reports: list[dict]) -> dict:
    LP = _load("df_linalg_parity")
    problems = []
    base = reports[0]
    for rep in reports:
        for k in ("fixture_sha256", "tolerance_sha256"):
            if rep[k] != base[k]:
                problems.append(f"{rep.get('label')}: {k} differs")
        if rep["tolerance_sha256"] != LP.TOLERANCES_SHA256:
            problems.append(f"{rep.get('label')}: tolerance declaration differs from this checkout")
    ref = base["layouts"]["C"]
    max_abs = {}
    compared = tolerated = 0
    for rep in reports:
        for layout, rows in rep["layouts"].items():
            where = f"{rep.get('label')}[{layout}]"
            if len(rows) != len(ref):
                problems.append(f"{where}: {len(rows)} rows vs {len(ref)}")
                continue
            for a, b in zip(ref, rows):
                compared += 1
                if (a["module"], a["key"], a["metric"], a["status"]) != (b["module"], b["key"], b["metric"], b["status"]):
                    problems.append(f"{where}: row identity differs at {a['metric']}")
                    continue
                if LP.tolerance_for(a["module"], a["metric"]) and a["status"] == "COMPLETED":
                    tolerated += 1
                    d = abs(a["value"] - b["value"])
                    key = f"{a['module']}.{a['metric']}"
                    max_abs[key] = max(max_abs.get(key, 0.0), d)
                    if not LP.agree(a["module"], a["metric"], a["value"], b["value"]):
                        problems.append(f"{where}: {a['metric']} outside tolerance ({a['value']!r} vs {b['value']!r})")
                    if a["value_canonical"] != b["value_canonical"]:
                        problems.append(f"{where}: {a['metric']} canonical differs")
                elif a["value"] != b["value"]:
                    problems.append(f"{where}: {a['metric']} exact value differs ({a['value']!r} vs {b['value']!r})")
    stacks = sorted({(r["numpy_version"], json.dumps(r["linalg_backends"], sort_keys=True), r.get("role", "?"))
                     for r in reports})
    return {"reports": len(reports), "labels": [r.get("label") for r in reports], "rows_compared": compared,
            "tolerance_rows_compared": tolerated, "max_abs_difference_by_metric": max_abs,
            "portable_digests": sorted({r["portable_digest"] for r in reports}),
            "stacks": [{"numpy": n, "backends": json.loads(b), "role": role} for n, b, role in stacks],
            "problems": problems, "parity": not problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--interpreter", help="python to run the fixture under")
    ap.add_argument("--role", default="UNSPECIFIED", help="COORDINATOR, WORKER_A or WORKER_B")
    ap.add_argument("--label", help="report label (default: role and interpreter environment)")
    ap.add_argument("--out", type=Path, help="write-once report path")
    ap.add_argument("--compare", type=Path, nargs="+", help="reports to compare")
    a = ap.parse_args(argv)
    if a.child:
        print(json.dumps(child(), sort_keys=True))
        return 0
    if a.compare:
        result = compare([json.loads(p.read_text()) for p in a.compare])
        print(json.dumps(result, indent=1, sort_keys=True))
        return 0 if result["parity"] else 1
    if not a.interpreter or not a.out:
        ap.error("--interpreter and --out are required unless --compare is given")
    if a.out.exists():
        raise SystemExit(f"REFUSED: {a.out.name} exists; reports are write-once")
    r = subprocess.run([a.interpreter, "-B", str(Path(__file__).resolve()), "--child"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-2000:])
        return r.returncode
    report = json.loads(r.stdout.strip().splitlines()[-1])
    report["role"] = a.role
    report["label"] = a.label or f"{a.role}:{report['env']}"
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("label", "python", "numpy_version", "linalg_backends", "fixture_sha256",
                                            "tolerance_sha256", "portable_digest")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
