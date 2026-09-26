#!/usr/bin/env python3
"""RP159: close the retained nine-cell ECL modular contrast through the real parent closing path, with NO inference.

What this does and does not do
  does      assemble a run directory from RETAINED evidence only -- the accepted CONTRAST.json of the RP155 run, the design
            retained beside it, and the child records a previous scoring pass published -- and call the real
            `df_ecl_modular.close_retained_run` over it, which validates, reconciles and aggregates.
  does not  load a checkpoint, build a dataset or start a process. `subprocess.run/Popen/check_output/call/check_call`,
            `score_cell` and `author_datasets` are all replaced by doubles that RAISE, so a closure that reached inference
            fails here instead of silently costing nine model replays.

The cost recorded below is this process's own measured CPU and wall time: the price of the closure, not of the contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("df_ecl_modular", args.repo / "tools/df_ecl_modular.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules["df_ecl_modular"] = M
    spec.loader.exec_module(M)

    evidence = args.repo / "docs/audits/evidence/d3_k5_20260917"
    sources = {"CONTRAST.json": evidence / "RP155/CONTRAST.json",
               "DESIGN.json": evidence / "RP159/DESIGN.json",
               "SCORING.json": evidence / "RP158_SCORING.json"}
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        (run_dir / name).write_bytes(source.read_bytes())

    def forbidden(*a, **k):
        raise AssertionError("REFUSED: this closure reached inference; it must read the retained records and nothing else")

    before_cpu = resource.getrusage(resource.RUSAGE_SELF)
    before_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.time()
    with patch.object(subprocess, "run", forbidden), patch.object(subprocess, "Popen", forbidden), \
            patch.object(subprocess, "check_output", forbidden), patch.object(subprocess, "call", forbidden), \
            patch.object(subprocess, "check_call", forbidden), patch.object(M, "score_cell", forbidden), \
            patch.object(M, "author_datasets", forbidden):
        closed = M.close_retained_run(run_dir)
    wall = time.time() - started
    after_cpu = resource.getrusage(resource.RUSAGE_SELF)
    after_children = resource.getrusage(resource.RUSAGE_CHILDREN)

    record = {
        "schema": "df_ecl_modular_closure_publication.v1",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": ("an offline validation-and-aggregation pass over RETAINED child evidence. No model was loaded, no dataset "
                  "was built and no process was started: every route to inference was a raising double for the whole call. "
                  "This re-closes numbers that already exist; it does not measure a model again"),
        "run_dir": str(run_dir),
        "inputs": {name: {"source": str(source.relative_to(args.repo)), "sha256": _digest(source)}
                   for name, source in sources.items()},
        "inference_calls_blocked": ["subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.call",
                                    "subprocess.check_call", "df_ecl_modular.score_cell",
                                    "df_ecl_modular.author_datasets"],
        "cost": {"closure_cpu_seconds": round((after_cpu.ru_utime - before_cpu.ru_utime)
                                              + (after_cpu.ru_stime - before_cpu.ru_stime), 3),
                 "child_cpu_seconds": round((after_children.ru_utime - before_children.ru_utime)
                                            + (after_children.ru_stime - before_children.ru_stime), 3),
                 "closure_wall_seconds": round(wall, 3),
                 "model_replays": 0, "gpu_seconds": 0.0,
                 "reading": "the cost of the closure itself; the contrast's own fit cost stands in CONTRAST.json unchanged"},
        "status": closed["status"],
        "authority": closed["authority"],
        "design_authentication": closed["design_authentication"],
        "child_binding": closed["child_binding"],
        "reconciliation": closed["reconciliation"],
        "problems": closed["problems"],
        "by_regime": closed["by_regime"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n")
    (args.out.parent / "CLOSURE.json").write_bytes((run_dir / "CLOSURE.json").read_bytes())
    print(json.dumps({k: record[k] for k in ("status", "cost", "reconciliation")}, indent=1))
    return 0 if closed["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
