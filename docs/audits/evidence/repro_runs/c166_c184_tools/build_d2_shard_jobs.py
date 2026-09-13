#!/usr/bin/env python3
"""Build D2 shard directories and a dispatcher jobs file (C172 historical reanalysis, C174 fresh confirmation).

Each shard is a directory of symlinks to unit directories, created identically on every role (the unit roots live
at the same state path on all hosts, verified by digest in C161), so a shard can run wherever the dispatcher places
it. One dispatcher job = one shard = one `df_d2_unit_worker.py` run that processes its units one at a time (each unit
in its own hard-limited child). The job's memory reservation covers the shard parent plus one child at a time.

Outputs go to <state>/<out_root>/{role}/shard_NN: the dispatcher substitutes {role} with the placed role, so each
role writes its own write-once shard roots and terminals carry the true role.

Usage (from the predictor checkout):
  build_d2_shard_jobs.py --units-root <dir of units> --mode HISTORICAL_MIGRATION_REANALYSIS_NON_CONFIRMATORY \
      --design <sealed design file> --shards-root <state>/d2_shards_c172_v1 --out-root <state>/d2_reanalysis_c172_v1 \
      --n-shards 18 --jobs-file <path>
Shard directories are created locally only; the caller replicates them to the workers (same paths) before dispatch.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

GIB = 1 << 30
UNIT_SECONDS_OBSERVED_MAX = 55.0      # C172 timing: 37-55 s per unit, single core
WALL_SAFETY = 3.0
PARENT_BYTES = 512 << 20               # shard parent (run_units) resident memory, generous
CHILD_TASK_BYTES = 2 * GIB             # assigned to each unit child (planned peak ~0.95 GiB, observed 0.13 GiB)
PY = "anaconda3/envs/trading-stack/bin/python"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-root", type=Path, required=True)
    ap.add_argument("--units-list", type=Path, help="optional file of unit directory names to include")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--shards-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--jobs-file", type=Path, required=True)
    ap.add_argument("--home", type=Path, default=Path.home())
    a = ap.parse_args(argv)
    if a.shards_root.exists() or a.jobs_file.exists():
        raise SystemExit("REFUSED: shards root or jobs file exists; both are write-once")
    units = sorted(p.name for p in a.units_root.iterdir() if (p / "UNIT.json").is_file())
    if a.units_list:
        wanted = {ln.strip() for ln in a.units_list.read_text().splitlines() if ln.strip()}
        units = [u for u in units if u in wanted]
        if len(units) != len(wanted):
            raise SystemExit(f"REFUSED: {len(wanted) - len(units)} listed units are missing")
    n = max(1, min(a.n_shards, len(units)))
    shards = [units[i::n] for i in range(n)]                       # round-robin: similar regime mix per shard
    a.shards_root.mkdir(parents=True)
    jobs = []
    home = str(a.home)
    for i, members in enumerate(shards):
        sd = a.shards_root / f"shard_{i:02d}"
        sd.mkdir()
        for u in members:
            (sd / u).symlink_to(a.units_root / u)
        (sd / "MEMBERS.txt").write_text("".join(u + "\n" for u in members))
        rel = lambda p: "~/" + str(p)[len(home) + 1:] if str(p).startswith(home + "/") else str(p)  # noqa: E731
        # the dispatcher expands only arguments that START with "~/" (to "$HOME"/...); everything else is quoted
        argv = ["env", "-u", "PYTHONPATH", f"~/{PY}", "-B", "tools/df_d2_unit_worker.py",
                "--out", rel(a.out_root) + f"/{{role}}/shard_{i:02d}", "--design", rel(a.design),
                "--units-root", rel(sd), "--mode", a.mode, "--host-role", "{role}",
                "--task-memory", str(CHILD_TASK_BYTES)]
        jobs.append({"job_id": f"{a.out_root.name}-shard{i:02d}", "argv": argv,
                     "cpu_bytes": PARENT_BYTES + CHILD_TASK_BYTES, "gpu_bytes": 0, "cpus": 1,
                     "wall": int(math.ceil(len(members) * UNIT_SECONDS_OBSERVED_MAX * WALL_SAFETY)) + 600,
                     "roles": ["WORKER_A", "WORKER_B", "COORDINATOR"], "gpu_index": None, "split": None})
    a.jobs_file.write_text(json.dumps(jobs, indent=1) + "\n")
    print(json.dumps({"units": len(units), "shards": n, "units_per_shard": [len(s) for s in shards],
                      "max_wall_seconds": max(j["wall"] for j in jobs), "jobs_file": a.jobs_file.name}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
