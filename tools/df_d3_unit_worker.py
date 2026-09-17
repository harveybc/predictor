#!/usr/bin/env python3
"""One D3 mechanical unit, end to end, in its own process (J3).

The worker evaluates ONE unit — a synthetic bank unit directory or a materialised toy resource
directory — against every declared operator of `df_d3_operators.bank()`, with each operator's
twin and the unit's resource availability contract, through the amended battery
`df_d3_acceptance.run_battery`. One JSONL row per unit x variable x operator x test, plus one
verdict row per unit x variable x operator. Rows carry the amendment digest, the operator
spec digest, the state digest, the code digests and the unit's contract digest.

Child protocol (the same one `df_isolated_runner` verifies for D2): the parent writes a job
file, the child writes `heartbeat.json` while it works and `result.json` when it ends, with
`status`, `reason`, `output_file`, `output_sha256` and `rows_written`; the parent re-hashes
and recounts the output before it believes the status.

    python tools/df_d3_unit_worker.py --worker JOB.json          (one unit, child)
    python tools/df_d3_unit_worker.py --pilot UNIT_DIR ...       (cost pilot, in-process)

Runs are NON_GOVERNING mechanical evidence: no utility ranking, no promotion. An operator that
fails causality is recorded and excluded, never tuned against the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load("df_d3_contract")
battery = _load("df_d3_acceptance")
design = _load("df_d3_design")
ops = _load("df_d3_operators")

ROW_SCHEMA = "df_fact_d3_mechanics.v1"
#: The family names a unit may carry; toy resources carry their own.
TOY_FAMILIES = {"toy_price", "toy_ohlc", "toy_features"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha256s() -> dict:
    return {name: sha_file(HERE / f"{name}.py")
            for name in ("df_d3_contract", "df_d3_acceptance", "df_d3_design",
                         "df_d3_operators", "df_d3_unit_worker")}


# --- units --------------------------------------------------------------------------------

def load_unit(unit_dir: Path) -> dict:
    """A unit as the battery consumes it: per-variable inputs, family, contract, availability.

    Synthetic bank units (UNIT.json + observed_signal.npy + missing_mask.npy) are SAMPLE_INDEX:
    timestamps are indices, availability is immediate, the period is one sample and the
    resource contract is a real '0s' lag. Toy units (TOY.json + OBSERVED.npy + TIMESTAMPS.npy)
    carry the lake's availability block and its declared frequency.
    """
    unit_dir = Path(unit_dir)
    if (unit_dir / "UNIT.json").is_file():
        rec = json.loads((unit_dir / "UNIT.json").read_text(encoding="utf-8"))
        observed = np.load(unit_dir / "observed_signal.npy", allow_pickle=False)
        mask = np.load(unit_dir / "missing_mask.npy", allow_pickle=False)
        if observed.ndim == 1:
            observed, mask = observed[None, :], mask[None, :]
        values = observed.astype(np.float64).copy()
        values[mask.astype(bool)] = np.nan
        n = values.shape[1]
        train_end = int(rec["partitions"]["train"][1])
        return {"unit_id": rec["unit_id"], "family": rec["family"], "bank": "SYNTHETIC",
                "dataset_id": f"synthetic.{rec['generator']['version']}.{rec['unit_id']}",
                "contract_sha256": rec["digests"]["observed_signal"],
                "n_samples": n, "train_end": train_end,
                "variables": [f"v{i}" for i in range(values.shape[0])],
                "inputs": [battery.make_input(values[i].tolist()) for i in range(values.shape[0])],
                "resource_contract": {"frequency": "1s", "availability": {
                    "label": "WINDOW_START", "completion_lag_max": "0s",
                    "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "LIVE_EQUIVALENT"}},
                "timestamp_meaning": "SAMPLE_INDEX"}
    if (unit_dir / "TOY.json").is_file():
        rec = json.loads((unit_dir / "TOY.json").read_text(encoding="utf-8"))
        observed = np.load(unit_dir / "OBSERVED.npy", allow_pickle=False)
        timestamps = np.load(unit_dir / "TIMESTAMPS.npy", allow_pickle=False)
        if observed.ndim == 1:
            observed = observed[:, None]
        lag = battery.parse_duration_seconds(rec["resource_contract"]["availability"]
                                             .get("completion_lag_max"))
        period = battery.parse_duration_seconds(rec["resource_contract"].get("frequency"))
        ts = [int(t) for t in timestamps.tolist()]
        available = [t + (lag or 0.0) for t in ts] if lag is not None else list(ts)
        inputs = [battery.make_input(observed[:, j].astype(np.float64).tolist(),
                                     timestamps=ts, available_at=available,
                                     period_seconds=period)
                  for j in range(observed.shape[1])]
        n = observed.shape[0]
        return {"unit_id": rec["unit_id"], "family": rec["family"], "bank": "TOY",
                "dataset_id": rec["dataset_id"], "contract_sha256": rec["contract_sha256"],
                "n_samples": n, "train_end": int(rec["train_end"]),
                "variables": list(rec["variables"]), "inputs": inputs,
                "resource_contract": rec["resource_contract"],
                "timestamp_meaning": rec["timestamp_meaning"]}
    raise SystemExit(f"REFUSED: {unit_dir} is neither a synthetic unit nor a toy unit")


# --- rows ---------------------------------------------------------------------------------

def _base(job: dict, unit: dict, variable: str, op, spec_sha: str) -> dict:
    return {"schema": ROW_SCHEMA, "run_id": job["run_id"], "host_role": job["host_role"],
            "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
            "bank": unit["bank"], "unit_id": unit["unit_id"], "family": unit["family"],
            "dataset_id": unit["dataset_id"], "contract_sha256": unit["contract_sha256"],
            "variable": variable, "operator_kind": op.KIND, "operator_group": op.GROUP,
            "operator_params": json.dumps(op.params, sort_keys=True),
            "spec_sha256": spec_sha, "fit_scope": op.FIT_SCOPE,
            "n_samples": unit["n_samples"], "timestamp_meaning": unit["timestamp_meaning"],
            "code_sha256": job["code_sha256"], "result_class": "MECHANICAL",
            "classification": "NON_GOVERNING"}


def rows_for(job: dict, unit: dict) -> list:
    rows = []
    for vi, variable in enumerate(unit["variables"]):
        x = unit["inputs"][vi]
        train = battery.prefix(x, max(2, min(unit["train_end"], unit["n_samples"] - 1)))
        only_ops = set(job.get("operators") or [])
        tests = job.get("tests") or None
        for op in ops.bank():
            if only_ops and op.KIND not in only_ops:
                continue
            spec_sha = contract.spec_sha256(op.describe())
            base = _base(job, unit, variable, op, spec_sha)
            started = time.process_time()
            try:
                report = battery.run_battery(op, x, train=train, twin=ops.twin_of(op),
                                             resource_contract=unit["resource_contract"],
                                             inapplicable_family="__undeclared__", tests=tests)
            except contract.SpecRefusal as exc:
                rows.append(dict(base, test="battery", outcome="REFUSED",
                                 detail=str(exc)[:300], value=None,
                                 cpu_seconds=round(time.process_time() - started, 4)))
                rows.append(dict(base, test="verdict", outcome="REFUSED", detail=str(exc)[:300],
                                 value=None, cpu_seconds=None))
                continue
            for test, res in report["results"].items():
                outcome = ("PASSED" if res["passed"] is True else
                           "FAILED" if res["passed"] is False else
                           "SCOPED" if res.get("scoped") else
                           str(res.get("outcome") or "UNDECIDED"))
                value = None
                for key in ("observed", "measured_cpu_seconds_per_1000", "compared", "checked",
                            "lag_samples"):
                    if key in res and isinstance(res[key], (int, float)):
                        value = float(res[key])
                        break
                rows.append(dict(base, test=test, outcome=outcome, value=value,
                                 detail=(res.get("detail") or res.get("reason") or "")[:300],
                                 cpu_seconds=None))
            rows.append(dict(base, test="verdict", outcome=report["verdict"],
                             value=1.0 if report["review_ready"] else 0.0,
                             detail=json.dumps({"failed": report["failed"],
                                                "scoped": report["scoped"],
                                                "undecided": report["undecided"],
                                                "scope": report["scope"]}),
                             cpu_seconds=round(time.process_time() - started, 4)))
    return rows


# --- child ----------------------------------------------------------------------------------

def _heartbeat(path: Path, **state) -> None:
    doc = dict(state, at=time.time(), pid=os.getpid())
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def worker_main(job_file: Path) -> int:
    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    adir = Path(job["attempt_dir"])
    adir.mkdir(parents=True, exist_ok=True)
    hb = adir / "heartbeat.json"
    result = {"status": "FAILED", "reason": "", "rows_written": 0, "output_file": None,
              "output_sha256": None}
    _heartbeat(hb, unit_id=job.get("unit_id"), phase="loading")
    try:
        unit = load_unit(Path(job["unit_dir"]))
        _heartbeat(hb, unit_id=unit["unit_id"], phase="battery")
        rows = rows_for(job, unit)
        out = adir / "rows.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        verdicts = [r["outcome"] for r in rows if r["test"] == "verdict"]
        result.update(status="COMPLETED", rows_written=len(rows), output_file="rows.jsonl",
                      output_sha256=sha_file(out),
                      verdicts={v: verdicts.count(v) for v in sorted(set(verdicts))})
    except MemoryError as exc:
        result.update(status="FAILED", reason=f"MemoryError: {exc}"[:500],
                      exception_type="MemoryError")
    except Exception as exc:  # noqa: BLE001 - the parent classifies from the record
        result.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}"[:500],
                      exception_type=type(exc).__name__)
    _heartbeat(hb, unit_id=job.get("unit_id"), phase="done", status=result["status"])
    tmp = adir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, adir / "result.json")
    return 0 if result["status"] == "COMPLETED" else 1


def variables_of(unit_dir: Path) -> int:
    """How many variables a unit carries, from its own record; the budget scales with it."""
    unit_dir = Path(unit_dir)
    if (unit_dir / "UNIT.json").is_file():
        return int(json.loads((unit_dir / "UNIT.json").read_text(encoding="utf-8"))["n_variables"])
    if (unit_dir / "TOY.json").is_file():
        return len(json.loads((unit_dir / "TOY.json").read_text(encoding="utf-8"))["variables"])
    return 1


# --- shard runner (one process per unit, under the isolated runner) ------------------------

def run_shard(units_root: Path, out_dir: Path, *, run_id: str, host_role: str, tests=None,
              operators=None,
              task_memory_bytes: int, wall_seconds: float, cpu_seconds: int) -> dict:
    IR = _load("df_isolated_runner")
    out_dir.mkdir(parents=True, exist_ok=True)
    mechanism = IR.detect_mechanism()
    code = code_sha256s()["df_d3_operators"]
    results = []
    unit_dirs = sorted(p for p in Path(units_root).iterdir() if p.is_dir())
    for ud in unit_dirs:
        sname = ud.name
        prior_dirs = sorted((out_dir / "attempts" / sname).glob("attempt-*")) \
            if (out_dir / "attempts" / sname).is_dir() else []
        completed_before = [p for p in prior_dirs if (p / "result.json").is_file()
                            and json.loads((p / "result.json").read_text()).get("status")
                            == "COMPLETED"]
        if completed_before:
            prior = json.loads((completed_before[-1] / "result.json").read_text(encoding="utf-8"))
            results.append({"unit": sname, "status": prior["status"], "resumed_skip": True})
            continue
        # A killed attempt left no result: it stays on disk as evidence and the retry gets the
        # next attempt number, never the same directory.
        attempt = len(prior_dirs) + 1
        adir = out_dir / "attempts" / sname / f"attempt-{attempt}"
        adir.mkdir(parents=True, exist_ok=False)
        job = {"unit_dir": str(ud.resolve()), "attempt_dir": str(adir.resolve()),
               "run_id": run_id, "host_role": host_role, "unit_id": sname,
               "code_sha256": code, "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
               "tests": list(tests) if tests else None,
               "operators": list(operators) if operators else None}
        job_file = adir / "job.json"
        job_file.write_text(json.dumps(job, indent=1), encoding="utf-8")
        argv = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(job_file)]
        # The frozen budget is PER VARIABLE: a unit with 26 variables earns 26 times the wall
        # and CPU of a univariate one. The first run gave every unit the flat figure and the
        # three 26-variable toy units died at WALL_TIME_LIMIT; that attempt is kept as evidence.
        scale = max(1, variables_of(ud))
        task = IR.Task(argv=argv, name=f"d3-{sname}", attempt_dir=adir,
                       assigned_bytes=task_memory_bytes, wall_seconds=wall_seconds * scale,
                       cpu_seconds=int(cpu_seconds * scale), mechanism=mechanism,
                       extra_env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                  "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
        task.start()
        task.wait()
        status, reason, verified = IR.classify(task.outcome, adir)
        terminal = {k: None for k in IR.TERMINAL_KEYS}
        terminal.update(run_id=run_id, host_role=host_role, bank="D3_MECHANICS",
                        dataset_id=sname, contract_sha256=None, code_sha256=code,
                        status=status, reason=reason.replace(str(Path.home()), "~"),
                        rows_written=verified["rows_written"], variables_profiled=0,
                        metrics_completed=0, metrics_missing=0, planned_peak_bytes=0,
                        observed_peak_rss_bytes=task.outcome.get("child_maxrss_bytes"),
                        memory_limit_bytes=IR.limits_for(task_memory_bytes)["memory_limit_bytes"],
                        limit_mechanism=IR.mechanism_label(mechanism, IR.DEFAULT_SLICE),
                        wall_seconds=task.outcome["wall_seconds"],
                        cpu_seconds=task.outcome["cpu_seconds"],
                        output_file="rows.jsonl" if status == "COMPLETED" else None,
                        output_sha256=verified["output_sha256"],
                        started_at=task.outcome["started_at"], ended_at=task.outcome["ended_at"])
        (out_dir / "terminals").mkdir(exist_ok=True)
        IR.write_terminal(out_dir / "terminals" / f"{sname}.attempt-{attempt}.json", terminal)
        results.append({"unit": sname, "status": status, "reason": reason})
    manifest = {"schema": "d3_shard_run.v1", "run_id": run_id, "host_role": host_role,
                "units": results, "code_sha256s": code_sha256s(),
                "design_sha256": design.D3_DESIGN_CURRENT["design_sha256"],
                "finished_utc": now_iso()}
    (out_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                               encoding="utf-8")
    return manifest


# --- cost pilot -------------------------------------------------------------------------------

def pilot(unit_dirs: list) -> dict:
    """Single-thread CPU seconds per operator on the named units, measured now."""
    measured = {}
    for ud in unit_dirs:
        unit = load_unit(Path(ud))
        x = unit["inputs"][0]
        train = battery.prefix(x, max(2, min(unit["train_end"], unit["n_samples"] - 1)))
        for op in ops.bank():
            started = time.process_time()
            state = op.fit(train)
            op.transform(x, state)
            seconds = time.process_time() - started
            entry = measured.setdefault(op.KIND, {"per_1000": [], "declared":
                                                  op.COST_CPU_SECONDS_PER_1000})
            entry["per_1000"].append(round(seconds * 1000.0 / max(1, unit["n_samples"]), 5))
    for kind, entry in measured.items():
        entry["max_per_1000"] = max(entry["per_1000"])
        entry["within_declaration"] = entry["max_per_1000"] <= entry["declared"]
    return {"schema": "d3_cost_pilot.v1", "generated_utc": now_iso(), "threads": 1,
            "units": [str(u) for u in unit_dirs], "operators": measured}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pilot", type=Path, nargs="+", help="unit directories to time")
    parser.add_argument("--units-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--host-role", default="COORDINATOR")
    parser.add_argument("--task-memory", type=int, default=2 << 30)
    parser.add_argument("--wall-seconds", type=float, default=1800.0)
    parser.add_argument("--cpu-seconds", type=int, default=1800)
    parser.add_argument("--tests", nargs="+", help="measure only these tests (composite replay)")
    parser.add_argument("--operators", nargs="+", help="measure only these operators")
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    if args.pilot:
        print(json.dumps(pilot(args.pilot), indent=1))
        return 0
    if not (args.units_root and args.out and args.run_id):
        parser.error("--units-root, --out and --run-id are required to run a shard")
    manifest = run_shard(args.units_root, args.out, run_id=args.run_id,
                         host_role=args.host_role, task_memory_bytes=args.task_memory,
                         wall_seconds=args.wall_seconds, cpu_seconds=args.cpu_seconds, tests=args.tests, operators=args.operators)
    statuses = [u["status"] for u in manifest["units"]]
    print(json.dumps({"run_id": args.run_id, "units": len(statuses),
                      "by_status": {s: statuses.count(s) for s in sorted(set(statuses))}}))
    return 0 if all(s == "COMPLETED" for s in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
