#!/usr/bin/env python3
"""One idempotent coordinator tick: scoped FS4 closure, seals, weekly RAW queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools import fs3_preliminary_status as progress
from tools import fs4_closure as full
from tools import fs4_preliminary_close as partial
from tools import fs4_wave_frontier as frontier
from tools import fs4_weekly_campaign as weekly


def tick(*, queue: Path, manifest_path: Path, receipt_root: Path, report_path: Path,
         consolidated_paths: list[Path], out_dir: Path, weekly_db: Path,
         alignment_cert_path: Path | None = None) -> dict:
    """Never read VALIDATION/TEST rows; only initialise their sealed task identities."""
    state = progress.summarize(queue, manifest_path)
    status = {"schema": "fs4.wave_successor_status.v1", "wave_status": state,
              "stage": "WAITING_FOR_WAVE", "final_selection": False}
    if any(state["counts"][arm]["PENDING"] + state["counts"][arm]["LEASED"] for arm in partial.ARMS):
        full.atomic_json(out_dir / "STATUS.json", status)
        return status
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    report = json.loads(report_path.read_text())
    if manifest.get("target_sets") != report.get("target_sets") or report.get("coverage_gaps"):
        raise partial.Refusal("TARGET_COVERAGE_REPORT_MISMATCH_OR_GAP")
    store = full.TaskStore(queue)
    try:
        parent_sha, _parent_plan = store.plan()
        tasks = store.tasks()
    finally:
        store.close()
    receipts = full.ReceiptDir(receipt_root).all_verified()
    close = partial.close_wave(tasks, manifest, parent_sha, hashlib.sha256(manifest_bytes).hexdigest(), receipts)
    close_path = out_dir / "EXTRACTIBILITY_PARTIAL_COMPLETE.json"
    if close_path.is_file() and json.loads(close_path.read_text()) != close:
        raise partial.Refusal("SCOPED_CLOSURE_CHANGED")
    full.atomic_json(close_path, close)
    consolidated = [json.loads(path.read_text()) for path in consolidated_paths]
    seals = []
    try:
        for cons in consolidated:
            pop = cons["population_id"]
            seal = frontier.build_seal(cons, close, report["target_sets"][pop])
            seal_path = out_dir / pop / "FRONTIER_SEAL.json"
            if seal_path.is_file() and json.loads(seal_path.read_text()) != seal:
                raise frontier.Refusal("WAVE_FRONTIER_CHANGED")
            full.atomic_json(seal_path, seal)
            seals.append(seal_path)
    except frontier.Refusal as exc:
        status["stage"] = "COVERAGE_GAP"
        status["reason"] = str(exc)
        status["closure_sha256"] = close["closure_sha256"]
        full.atomic_json(out_dir / "STATUS.json", status)
        return status
    result = weekly.initialize(weekly_db, consolidated_paths, seals, close_path,
                               out_dir=out_dir / "weekly", validation_year=2024,
                               input_modes=("RAW",), alignment_cert_path=alignment_cert_path)
    status.update({"stage": "WEEKLY_RAW_READY", "closure_sha256": close["closure_sha256"],
                   "weekly_plan_sha256": result["plan_sha256"], "weekly_tasks": result["tasks"],
                   "weekly_sets": result["sets"], "weeks": result["weeks"]})
    full.atomic_json(out_dir / "STATUS.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--consolidated", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weekly-db", type=Path, required=True)
    parser.add_argument("--alignment-cert", type=Path)
    args = parser.parse_args()
    result = tick(queue=args.queue, manifest_path=args.manifest, receipt_root=args.receipt_root,
                  report_path=args.coverage_report, consolidated_paths=args.consolidated,
                  out_dir=args.out_dir, weekly_db=args.weekly_db,
                  alignment_cert_path=args.alignment_cert)
    print(json.dumps({k: v for k, v in result.items() if k != "wave_status"}, sort_keys=True))


if __name__ == "__main__":
    main()
