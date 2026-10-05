#!/usr/bin/env python3
"""ETA publisher for the PS3-R GPU queues (lane FS-GPU, order §3.5).

For every queue declared in the config (alternatives on worker_a, baselines batch 001/003
on worker_a, baselines batch_002 shared by worker_b and the worker_a stealer) it computes,
from the coordinator's mirror of authenticated terminals and the relayed driver logs:

* done / total over the declared denominator, failures, running and pending cells;
* per-role median and nearest-rank p90 cell duration (``wall_seconds`` of COMPLETED
  manifests whose results digest verifies);
* active workers = roles with a BEGIN line and no END line in their driver log (a waiter
  never counts: no running cell, no worker);
* an ABSOLUTE UTC finish estimate: remaining work (pending cells plus the unfinished
  fraction of running cells) divided by the sum of the active workers' rates, in a median
  and a p90 scenario. A queue with no active worker and a ``starts_after`` queue inherits
  that queue's finish time plus its own remaining work at a provisional median, flagged.

Writes ``ps3r_eta.json`` (atomic) and appends one row per queue to ``ps3r_progress.csv``.
All paths and roles come from the config; no host names appear in the outputs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "ps3r_eta.v1"
STALE_RUNNING_SECONDS = 12 * 3600
UTC = _dt.timezone.utc
TS = "%Y-%m-%dT%H:%M:%SZ"
LOG_LINE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+(?P<verb>BEGIN|END)\s+(?P<rest>.+)$")


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(UTC)


def fmt(when: Optional[_dt.datetime]) -> Optional[str]:
    return when.strftime(TS) if when else None


def parse_ts(value: str) -> _dt.datetime:
    return _dt.datetime.strptime(value, TS).replace(tzinfo=UTC)


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return float(ordered[rank - 1])


# ------------------------------------------------------------------------------ plans


def read_plan(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        return [(row["cell_id"].strip(), row["result_dir"].strip()) for row in reader if row.get("cell_id")]


def cell_id_from_log(rest: str) -> Optional[str]:
    """``cap=.. rc=.. stage batch [family] feature`` -> ``batch::feature::family|baseline``."""

    tokens = [t for t in rest.split() if not (t.startswith("cap=") or t.startswith("rc="))]
    if len(tokens) == 4:
        stage, batch, family, feature = tokens
    elif len(tokens) == 3:
        stage, batch, feature = tokens
        family = "baseline"
    else:
        return None
    if not batch.startswith("batch_"):
        return None
    return f"{batch}::{feature}::{family}"


def parse_log(lines: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Latest BEGIN/END per cell. A cell with a BEGIN after its last END is running."""

    cells: Dict[str, Dict[str, Any]] = {}
    for line in lines:
        match = LOG_LINE.match(line.strip())
        if not match:
            continue
        cell = cell_id_from_log(match.group("rest"))
        if cell is None:
            continue
        entry = cells.setdefault(cell, {"begin": None, "end": None, "rc": None})
        when = parse_ts(match.group("ts"))
        if match.group("verb") == "BEGIN":
            entry["begin"], entry["end"], entry["rc"] = when, None, None
        else:
            entry["end"] = when
            rc = re.search(r"rc=(\d+)", match.group("rest"))
            entry["rc"] = int(rc.group(1)) if rc else None
    return cells


def running_from_logs(cells: Dict[str, Dict[str, Any]], now: _dt.datetime) -> List[Dict[str, Any]]:
    running = []
    for cell, entry in cells.items():
        if entry["begin"] is not None and entry["end"] is None:
            elapsed = (now - entry["begin"]).total_seconds()
            running.append({"cell_id": cell, "started_at_utc": fmt(entry["begin"]), "elapsed_seconds": int(elapsed), "stale": elapsed > STALE_RUNNING_SECONDS})
    return sorted(running, key=lambda item: item["cell_id"])


# --------------------------------------------------------------------------- terminals


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_terminal(directory: Path) -> Optional[Dict[str, Any]]:
    manifest_path = directory / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return {"state": "failed", "error": f"INVALID_MANIFEST: {error}"}
        if manifest.get("status") == "COMPLETED":
            results = directory / "results.jsonl"
            expected = str(manifest.get("results_sha256", "")).lower()
            if not results.is_file() or len(expected) != 64 or _sha256(results) != expected:
                return {"state": "failed", "error": "ARTIFACT_DIGEST_MISMATCH"}
            wall = manifest.get("wall_seconds")
            return {
                "state": "done",
                "wall_seconds": float(wall) if isinstance(wall, (int, float)) else None,
                "finished_at_utc": fmt(_dt.datetime.fromtimestamp(manifest_path.stat().st_mtime, UTC)),
                "results_sha256": expected,
                "cgroup_peak_bytes": manifest.get("cgroup_peak_bytes"),
            }
        return {"state": "failed", "error": f"STATUS_{manifest.get('status')}"}
    for name in ("FAILED.codex.json", "FAILED.json"):
        if (directory / name).is_file():
            try:
                reason = json.loads((directory / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                reason = {}
            return {"state": "failed", "error": f"FAILED_MARKER rc={reason.get('rc')}", "marker": str(directory / name)}
    return None


# ---------------------------------------------------------------------------- estimate


def role_stats(durations: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    stats = {}
    for role, values in durations.items():
        clean = [v for v in values if v is not None and math.isfinite(v) and v >= 0]
        stats[role] = {
            "n": len(clean),
            "median_seconds": round(statistics.median(clean), 1) if clean else None,
            "p90_seconds": round(nearest_rank(clean, 0.9), 1) if clean else None,
        }
    return stats


def estimate(
    pending: int,
    running: List[Dict[str, Any]],
    active_roles: Dict[str, Optional[float]],
    now: _dt.datetime,
) -> Dict[str, Any]:
    """Finish time from remaining work / summed worker rates.

    ``active_roles`` maps each active role to its per-cell duration in this scenario
    (None = unknown). ``running`` items carry ``role`` and ``elapsed_seconds``. Returns
    seconds and absolute UTC, or None with a reason when no active worker has a duration.
    """

    rates = {role: 1.0 / seconds for role, seconds in active_roles.items() if seconds and seconds > 0}
    if not rates:
        return {"seconds": None, "finish_utc": None, "reason": "NO_ACTIVE_WORKER_WITH_DURATION" if active_roles else "NO_ACTIVE_WORKER"}
    work = float(pending)
    for item in running:
        seconds = active_roles.get(item.get("role"))
        if seconds and seconds > 0:
            work += max(0.0, 1.0 - item.get("elapsed_seconds", 0) / seconds)
        else:
            work += 1.0
    seconds_total = work / sum(rates.values())
    return {"seconds": int(seconds_total), "finish_utc": fmt(now + _dt.timedelta(seconds=seconds_total)), "reason": None}


# ------------------------------------------------------------------------------- queue


def summarize_queue(queue: Dict[str, Any], now: _dt.datetime, base: Path, fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    plan = read_plan(_expand(queue["plan"], base))
    roots = [(item["role"], _expand(item["path"], base)) for item in queue.get("roots", [])]
    logs = [(item["role"], _expand(item["path"], base)) for item in queue.get("logs", [])]

    durations: Dict[str, List[float]] = {role: [] for role, _ in roots}
    done_by_role: Dict[str, int] = {role: 0 for role, _ in roots}
    cells: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    done = failed = 0
    done_ids = set()
    for cell_id, result_dir in plan:
        hits = []
        for role, root in roots:
            info = inspect_terminal(root / result_dir)
            if info:
                hits.append(dict(info, role=role, path=str(root / result_dir)))
        completed = [h for h in hits if h["state"] == "done"]
        if completed:
            done += 1
            done_ids.add(cell_id)
            first = completed[0]
            done_by_role[first["role"]] += 1
            if first["wall_seconds"] is not None:
                durations[first["role"]].append(first["wall_seconds"])
            cells.append({"cell_id": cell_id, "state": "done", "role": first["role"], "wall_seconds": first["wall_seconds"], "finished_at_utc": first["finished_at_utc"], "results_sha256": first["results_sha256"]})
            if len(completed) > 1:
                duplicates.append({"cell_id": cell_id, "roles": [h["role"] for h in completed], "digests_equal": len({h["results_sha256"] for h in completed}) == 1})
        elif hits:
            failed += 1
            failures.append({"cell_id": cell_id, "role": hits[0]["role"], "error": hits[0]["error"], "receipt": hits[0].get("marker") or hits[0]["path"]})
            cells.append({"cell_id": cell_id, "state": "failed", "role": hits[0]["role"], "error": hits[0]["error"]})
        else:
            cells.append({"cell_id": cell_id, "state": "pending"})

    plan_ids = {cell_id for cell_id, _ in plan}
    running: List[Dict[str, Any]] = []
    for role, log_path in logs:
        if not log_path.is_file():
            continue
        parsed = parse_log(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
        for item in running_from_logs(parsed, now):
            if item["cell_id"] in plan_ids and item["cell_id"] not in done_ids and not item["stale"]:
                running.append(dict(item, role=role, log=str(log_path)))
    running_ids = {item["cell_id"] for item in running}
    for cell in cells:
        if cell["state"] == "pending" and cell["cell_id"] in running_ids:
            cell["state"] = "running"
            cell["role"] = next(item["role"] for item in running if item["cell_id"] == cell["cell_id"])
    pending = sum(1 for cell in cells if cell["state"] == "pending")

    stats = role_stats(durations)
    active = sorted({item["role"] for item in running})
    provisional = False
    basis = {}
    for scenario in ("median_seconds", "p90_seconds"):
        per_role = {}
        for role in active:
            value = stats.get(role, {}).get(scenario)
            if value is None and fallback:
                value = fallback.get(scenario)
                provisional = True
            per_role[role] = value
        basis[scenario] = per_role
    eta_median = estimate(pending, running, basis["median_seconds"], now)
    eta_p90 = estimate(pending, running, basis["p90_seconds"], now)

    all_durations = [v for values in durations.values() for v in values]
    return {
        "name": queue["name"],
        "denominator": queue.get("denominator", len(plan)),
        "plan_cells": len(plan),
        "counts": {"done": done, "failed": failed, "running": len(running), "pending": pending},
        "by_role": {role: {"done": done_by_role[role], **stats[role]} for role, _ in roots},
        "durations_seconds": {"n": len(all_durations), "median": round(statistics.median(all_durations), 1) if all_durations else None, "p90": round(nearest_rank(all_durations, 0.9), 1) if all_durations else None},
        "active_workers": len(active),
        "active_roles": active,
        "running": running,
        "eta": {
            "finish_utc_median": eta_median["finish_utc"],
            "finish_utc_p90": eta_p90["finish_utc"],
            "seconds_median": eta_median["seconds"],
            "seconds_p90": eta_p90["seconds"],
            "provisional": provisional,
            "reason": eta_median["reason"],
            "basis": basis,
        },
        "duplicates": duplicates,
        "failures": failures,
        "cells": cells,
    }


def chain_waiting_queue(summary: Dict[str, Any], upstream: Dict[str, Any], fallback: Optional[Dict[str, Any]], now: _dt.datetime) -> None:
    """A queue with no active worker that starts after ``upstream`` inherits its finish."""

    if summary["active_workers"] or summary["counts"]["pending"] + summary["counts"]["running"] == 0:
        return
    median = summary["durations_seconds"]["median"] or (fallback or {}).get("median_seconds")
    p90 = summary["durations_seconds"]["p90"] or (fallback or {}).get("p90_seconds")
    start_med = upstream["eta"]["finish_utc_median"]
    start_p90 = upstream["eta"]["finish_utc_p90"]
    remaining = summary["counts"]["pending"] + summary["counts"]["running"]
    eta = summary["eta"]
    eta["waiting_on"] = upstream["name"]
    eta["provisional"] = True
    eta["reason"] = f"WAITING_ON_{upstream['name']}"
    if start_med and median:
        start = max(parse_ts(start_med), now)
        eta["seconds_median"] = int((start - now).total_seconds() + remaining * median)
        eta["finish_utc_median"] = fmt(start + _dt.timedelta(seconds=remaining * median))
    if start_p90 and p90:
        start = max(parse_ts(start_p90), now)
        eta["seconds_p90"] = int((start - now).total_seconds() + remaining * p90)
        eta["finish_utc_p90"] = fmt(start + _dt.timedelta(seconds=remaining * p90))


# ------------------------------------------------------------------------------- hosts


def read_host_probes(probes: Dict[str, Dict[str, str]], base: Path) -> Dict[str, Dict[str, Any]]:
    hosts = {}
    for role, files in probes.items():
        entry: Dict[str, Any] = {}
        gpu = _expand(files.get("gpu_csv", ""), base) if files.get("gpu_csv") else None
        if gpu and gpu.is_file():
            entry["gpus"] = [line.strip() for line in gpu.read_text().splitlines() if line.strip()]
            entry["probed_at_utc"] = fmt(_dt.datetime.fromtimestamp(gpu.stat().st_mtime, UTC))
        mem = _expand(files.get("mem_csv", ""), base) if files.get("mem_csv") else None
        if mem and mem.is_file():
            entry["memory"] = mem.read_text().strip()
        hosts[role] = entry
    return hosts


# ---------------------------------------------------------------------------- publish


def _expand(value: str, base: Path) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else base / path


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


PROGRESS_COLUMNS = ["generated_at_utc", "queue", "done", "denominator", "failed", "running", "pending", "active_workers", "median_seconds", "p90_seconds", "finish_utc_median", "finish_utc_p90", "provisional"]


def append_progress(path: Path, generated_at: str, queues: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_COLUMNS)
        if new:
            writer.writeheader()
        for name, q in queues.items():
            writer.writerow({
                "generated_at_utc": generated_at,
                "queue": name,
                "done": q["counts"]["done"],
                "denominator": q["denominator"],
                "failed": q["counts"]["failed"],
                "running": q["counts"]["running"],
                "pending": q["counts"]["pending"],
                "active_workers": q["active_workers"],
                "median_seconds": q["durations_seconds"]["median"],
                "p90_seconds": q["durations_seconds"]["p90"],
                "finish_utc_median": q["eta"]["finish_utc_median"],
                "finish_utc_p90": q["eta"]["finish_utc_p90"],
                "provisional": q["eta"]["provisional"],
            })


def build_report(config: Dict[str, Any], base: Path, now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    now = now or utc_now()
    queues: Dict[str, Dict[str, Any]] = {}
    fallbacks: Dict[str, Dict[str, Any]] = {}
    for queue in config["queues"]:
        fallback = None
        source = queue.get("fallback_median_from")
        if source and source in queues:
            fallback = {"median_seconds": queues[source]["durations_seconds"]["median"], "p90_seconds": queues[source]["durations_seconds"]["p90"]}
            fallbacks[queue["name"]] = fallback
        queues[queue["name"]] = summarize_queue(queue, now, base, fallback)
    for queue in config["queues"]:
        upstream = queue.get("starts_after")
        if upstream and upstream in queues:
            chain_waiting_queue(queues[queue["name"]], queues[upstream], fallbacks.get(queue["name"]), now)

    totals = {}
    for name, members in config.get("totals", {}).items():
        totals[name] = {
            "done": sum(queues[m]["counts"]["done"] for m in members["queues"] if m in queues),
            "failed": sum(queues[m]["counts"]["failed"] for m in members["queues"] if m in queues),
            "denominator": members["denominator"],
        }
    hosts = read_host_probes(config.get("host_probes", {}), base)
    for role in hosts:
        hosts[role]["running"] = [
            {"queue": name, "cell_id": item["cell_id"], "started_at_utc": item["started_at_utc"], "elapsed_seconds": item["elapsed_seconds"]}
            for name, q in queues.items() for item in q["running"] if item["role"] == role
        ]
    return {
        "schema": SCHEMA,
        "generated_at_utc": fmt(now),
        "rule": "active worker = role with a BEGIN and no END in its driver log; a waiter is not GPU work; ETA = remaining work / sum of active rates; absolute UTC",
        "totals": totals,
        "queues": {name: {k: v for k, v in q.items() if k != "cells"} for name, q in queues.items()},
        "hosts": hosts,
        "cells": {name: q["cells"] for name, q in queues.items()},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base", type=Path, help="base for relative config paths (default: config's repo root guess = cwd)")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = args.base or Path.cwd()
    report = build_report(config, base)
    if args.out_json:
        atomic_write(args.out_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out_csv:
        append_progress(args.out_csv, report["generated_at_utc"], report["queues"])
    if not args.quiet:
        brief = {name: {"done": q["counts"]["done"], "of": q["denominator"], "failed": q["counts"]["failed"], "running": q["counts"]["running"], "active_workers": q["active_workers"], "finish_utc_median": q["eta"]["finish_utc_median"], "finish_utc_p90": q["eta"]["finish_utc_p90"]} for name, q in report["queues"].items()}
        print(json.dumps({"totals": report["totals"], "queues": brief}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
