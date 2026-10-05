#!/usr/bin/env python3
"""Closure matrix per heavy candidate and automatic PS3-R ingest (lane FS-GPU, order §3.4).

For each of the 137 heavy candidates (the union of the two committed baseline plans) the
matrix gives one cell per required family -- raw/identity, random, AE, DAE, masked temporal
AE, past-to-current -- with a state and a receipt:

    DONE          results_sha256 + manifest path + host role
    RUNNING       role + start time (from the relayed driver logs)
    CLAIMED       role + claimed_at (shared claim, not yet running)
    FAILED        rc + marker path + reason excerpt (FAILED.reason.txt relayed from the worker)
    PENDING       planned, not started
    NOT_APPLICABLE  the family is not planned for this candidate (reason given)

identity, random, AE and DAE come from one baseline cell (families identity,random,ae,dae);
MTAE and P2C come from the two alternative cells of the same feature. The row is CLOSED
when every family is DONE, FAILED or NOT_APPLICABLE.

``--ingest`` additionally stages every authenticated terminal into a flat ingest root
(``<root>/<role>/<feature>/``) and runs the committed ``ps3r_manifest_ingestor.discover``
against a live config, so every terminal is adopted or refused by the same code the
readiness report uses. The decisions, not the 250 KB result files, are published.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ps3r_eta as E  # noqa: E402

SCHEMA = "ps3r_closure_matrix.v1"
FAMILIES = ["identity", "random", "ae", "dae", "masked_temporal_ae", "past_to_current_siamese"]
BASELINE_FAMILIES = ("identity", "random", "ae", "dae")
UTC = _dt.timezone.utc


def _expand(value: str, base: Path) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else base / path


def _reason(directory: Path) -> Optional[str]:
    path = directory / "FAILED.reason.txt"
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return text[-1][:300] if text else None
    return None


def family_cells(feature: str, batch: str, plans: Dict[str, set]) -> Dict[str, Optional[str]]:
    """Which plan cell serves each family for this candidate (None = not planned)."""

    cells: Dict[str, Optional[str]] = {}
    baseline = f"{batch}::{feature}::baseline"
    for family in BASELINE_FAMILIES:
        cells[family] = baseline if baseline in plans["baseline"] else None
    for family in ("masked_temporal_ae", "past_to_current_siamese"):
        cell = f"{batch}::{feature}::{family}"
        cells[family] = cell if cell in plans["alternative"] else None
    return cells


def cell_state(cell_id: str, result_dir: str, roots: List[Tuple[str, Path]], running: Dict[str, Dict[str, Any]], claims: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    completed = []
    failed = None
    for role, root in roots:
        info = E.inspect_terminal(root / result_dir)
        if not info:
            continue
        if info["state"] == "done":
            completed.append((role, root / result_dir, info))
        elif failed is None:
            failed = (role, root / result_dir, info)
    if completed:
        role, path, info = completed[0]
        state = {"state": "DONE", "role": role, "results_sha256": info["results_sha256"], "receipt": str(path / "run_manifest.json"), "wall_seconds": info["wall_seconds"], "finished_at_utc": info["finished_at_utc"]}
        if len(completed) > 1:
            state["duplicate_roles"] = [c[0] for c in completed]
            state["digests_equal"] = len({c[2]["results_sha256"] for c in completed}) == 1
        return state
    if failed:
        role, path, info = failed
        return {"state": "FAILED", "role": role, "receipt": info.get("marker") or str(path), "rc": info["error"], "reason": _reason(path) or "see stdout on the worker (reason file not relayed yet)"}
    if cell_id in running:
        item = running[cell_id]
        return {"state": "RUNNING", "role": item["role"], "started_at_utc": item["started_at_utc"], "elapsed_seconds": item["elapsed_seconds"]}
    feature = cell_id.split("::")[1]
    claim = claims.get(feature)
    if claim and cell_id.endswith("::baseline"):
        return {"state": "CLAIMED", "role": claim["role"], "claimed_at_utc": claim["claimed_at_utc"], "claim_state": claim["state"]}
    return {"state": "PENDING"}


def live_claims(ledger_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not ledger_path or not ledger_path.is_file():
        return {}
    try:
        book = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for key, cell in book.get("cells", {}).items():
        winner = cell.get("winner")
        claim = (cell.get("claims") or {}).get(winner) if winner else None
        if claim and claim.get("state") in ("CLAIMED", "RUNNING"):
            out[cell["feature_id"]] = {"role": winner, "claimed_at_utc": claim.get("claimed_at_utc"), "state": claim.get("state")}
    return out


def build_matrix(config: Dict[str, Any], base: Path, now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    now = now or _dt.datetime.now(UTC)
    plans = {"baseline": {}, "alternative": {}}
    for kind in plans:
        for plan in config["plans"][kind]:
            for cell_id, result_dir in E.read_plan(_expand(plan, base)):
                plans[kind][cell_id] = result_dir
    roots = {kind: [(r["role"], _expand(r["path"], base)) for r in config["roots"][kind]] for kind in ("baseline", "alternative")}
    running: Dict[str, Dict[str, Any]] = {}
    for item in config.get("logs", []):
        path = _expand(item["path"], base)
        if path.is_file():
            for run in E.running_from_logs(E.parse_log(path.read_text(encoding="utf-8", errors="replace").splitlines()), now):
                if not run["stale"]:
                    running[run["cell_id"]] = dict(run, role=item["role"])
    claims = live_claims(_expand(config["claims_ledger"], base) if config.get("claims_ledger") else None)

    candidates: Dict[str, str] = {}
    for cell_id in plans["baseline"]:
        batch, feature, _ = cell_id.split("::")
        candidates[feature] = batch
    rows = []
    counts = {"closed": 0, "open": 0}
    family_counts = {f: {"DONE": 0, "FAILED": 0, "RUNNING": 0, "CLAIMED": 0, "PENDING": 0, "NOT_APPLICABLE": 0} for f in FAMILIES}
    plan_sets = {"baseline": set(plans["baseline"]), "alternative": set(plans["alternative"])}
    for feature in sorted(candidates):
        batch = candidates[feature]
        cells = family_cells(feature, batch, plan_sets)
        row: Dict[str, Any] = {"feature_id": feature, "batch_id": batch, "families": {}}
        closed = True
        for family in FAMILIES:
            cell_id = cells[family]
            if cell_id is None:
                state = {"state": "NOT_APPLICABLE", "reason": f"{family} not in any committed plan for {feature}"}
            else:
                kind = "baseline" if family in BASELINE_FAMILIES else "alternative"
                state = cell_state(cell_id, plans[kind][cell_id], roots[kind], running, claims)
                state["cell_id"] = cell_id
            row["families"][family] = state
            family_counts[family][state["state"]] += 1
            if state["state"] not in ("DONE", "FAILED", "NOT_APPLICABLE"):
                closed = False
        row["closed"] = closed
        counts["closed" if closed else "open"] += 1
        rows.append(row)
    return {
        "schema": SCHEMA,
        "generated_at_utc": E.fmt(now),
        "heavy_candidates": len(rows),
        "denominator": config.get("denominator", 137),
        "counts": counts,
        "family_counts": family_counts,
        "rows": rows,
    }


def write_matrix_csv(path: Path, matrix: Dict[str, Any]) -> None:
    columns = ["feature_id", "batch_id", "closed"] + [f"{f}_state" for f in FAMILIES] + [f"{f}_receipt" for f in FAMILIES]
    lines = []
    for row in matrix["rows"]:
        record = {"feature_id": row["feature_id"], "batch_id": row["batch_id"], "closed": row["closed"]}
        for family in FAMILIES:
            state = row["families"][family]
            record[f"{family}_state"] = state["state"]
            record[f"{family}_receipt"] = state.get("results_sha256") or state.get("receipt") or state.get("reason") or state.get("role") or ""
        lines.append(record)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix="." + path.name + ".")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(lines)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------ ingest


def planned_dirs(config: Dict[str, Any], base: Path, kind: str, family_suffix: Optional[str]) -> List[str]:
    """Result directories of the committed plans for one ingest role (never anything else)."""

    dirs = []
    for plan in config["plans"][kind]:
        for cell_id, result_dir in E.read_plan(_expand(plan, base)):
            if family_suffix is None or cell_id.endswith("::" + family_suffix):
                dirs.append(result_dir)
    return dirs


def stage_terminals(config: Dict[str, Any], base: Path, ingest_root: Path) -> Dict[str, Any]:
    """Copy every authenticated PLANNED terminal to ``ingest_root/<role>/<feature>/``.

    Only directories named by a committed plan are staged: pilot, probe or sealed
    directories that live next to the plan cells on a worker are never adopted.
    Idempotent: an already staged terminal with the same digest is left untouched.
    """

    staged = {"copied": 0, "unchanged": 0, "skipped_unverified": 0, "roles": {}}
    for role_cfg in config["ingest"]["roles"]:
        role = role_cfg["name"]
        destination_root = ingest_root / role
        destination_root.mkdir(parents=True, exist_ok=True)
        wanted = planned_dirs(config, base, role_cfg["plan_kind"], role_cfg.get("family"))
        count = 0
        for result_dir in wanted:
            for root_cfg in role_cfg["sources"]:
                root = _expand(root_cfg["path"], base)
                directory = root / result_dir
                info = E.inspect_terminal(directory) if directory.is_dir() else None
                if not info:
                    continue
                if info["state"] != "done":
                    staged["skipped_unverified"] += 1
                    continue
                feature = directory.parent.name if directory.name == "pass_p2c" else directory.name
                target = destination_root / feature
                target_manifest = target / "run_manifest.json"
                if target_manifest.is_file() and json.loads(target_manifest.read_text()).get("results_sha256") == info["results_sha256"]:
                    staged["unchanged"] += 1
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    for name in ("run_manifest.json", "results.jsonl"):
                        shutil.copy2(directory / name, target / name)
                    staged["copied"] += 1
                count += 1
                break  # first root holding a verified terminal wins; duplicates are reported by the matrix
        staged["roles"][role] = count
    return staged


def run_ingestor(ingestor_path: Path, ingest_root: Path, live_config: Path) -> List[Dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("ps3r_manifest_ingestor", ingestor_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.discover(ingest_root, live_config)


def ingest_report(decisions: List[Dict[str, Any]], ingestor_path: Path, live_config: Path, staged: Dict[str, Any]) -> Dict[str, Any]:
    counts: Dict[str, Dict[str, int]] = {}
    rejected = []
    for item in decisions:
        role = item["role"]
        bucket = counts.setdefault(role, {})
        key = item["disposition"] if item["disposition"] == "ADOPTED" else f"{item['disposition']}:{item['reason']}"
        bucket[key] = bucket.get(key, 0) + 1
        if item["disposition"] != "ADOPTED":
            rejected.append({"role": role, "feature_id": item["feature_id"], "reason": item["reason"], "results_sha256": item.get("results_sha256")})
    return {
        "schema": "ps3r_ingest_live.v1",
        "generated_at_utc": E.fmt(_dt.datetime.now(UTC)),
        "ingestor_sha256": hashlib.sha256(ingestor_path.read_bytes()).hexdigest(),
        "live_config_sha256": hashlib.sha256(live_config.read_bytes()).hexdigest(),
        "staged": staged,
        "counts": counts,
        "not_adopted": rejected,
        "adopted": [{"role": i["role"], "feature_id": i["feature_id"], "results_sha256": i["results_sha256"], "utility": i["utility"], "code_commit": i["code_commit"], "input_digest": i["input_digest"]} for i in decisions if i["disposition"] == "ADOPTED"],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base", type=Path, default=Path.cwd())
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--ingest", action="store_true", help="stage terminals and run the committed ingestor")
    parser.add_argument("--ingest-out", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    matrix = build_matrix(config, args.base)
    if args.out_json:
        E.atomic_write(args.out_json, json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_matrix_csv(args.out_csv, matrix)
    if args.ingest:
        ingest_root = _expand(config["ingest"]["root"], args.base)
        staged = stage_terminals(config, args.base, ingest_root)
        live_config = _expand(config["ingest"]["live_config"], args.base)
        ingestor = _expand(config["ingest"]["ingestor"], args.base)
        report = ingest_report(run_ingestor(ingestor, ingest_root, live_config), ingestor, live_config, staged)
        if args.ingest_out:
            E.atomic_write(args.ingest_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
        if not args.quiet:
            print(json.dumps({"staged": staged, "counts": report["counts"], "not_adopted": len(report["not_adopted"])}, indent=2, sort_keys=True))
    if not args.quiet:
        print(json.dumps({"heavy_candidates": matrix["heavy_candidates"], "counts": matrix["counts"], "family_counts": matrix["family_counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
