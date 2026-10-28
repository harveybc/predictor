"""Durable compact-resolution worker, closure and status; no agent polling needed.

The three arms each refit all 52 validation weeks. Completed cells are checked
and skipped, never re-trained. A failed cell stops its worker with the reason
retained. Statistical ARIMA controls and price-unit ETL are separate, not
silently claimed by this raw-log-return neural campaign.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from tools.i6e_compact_run import atomic_json, code_identity, file_hash

ARMS = (3, 5, 9)


def initialize(root, parent_path):
    """Freeze the population and configuration before the first outer fit."""
    from tools.i7_weekly_branch_regimes import _seal, _verify_seal
    parent = json.loads(parent_path.read_text())
    _verify_seal(parent, "design_sha256")
    if len(parent["weeks"]) != 52 or parent["test_read"] is not False:
        raise ValueError("52-week parent with unopened TEST required")
    design = _seal(dict(schema="predictor.i6e.compact.v1", expected_cells=156,
        samples=list(ARMS), seed=6102026, horizon_hours=72, context_hours=144,
        code_sha256=code_identity(), parent=parent, test_read=False,
        model=dict(branch_channels=16, branch_kernel=3, transformer_blocks=2,
                   d_model=32, heads=2, ff_dim=64, core_channels=[24,16,8], strides=[1,1,1]),
        metric_scope="raw log return only; price-unit analysis pending separate price join",
        preprocessing="TRAIN-fit scaling, no smoothing, explicit missing-market masks"), "design_sha256")
    path = root / "DESIGN.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError("existing design conflict")
    atomic_json(path, design)
    return design


def read_design(root):
    """Do not admit terminals from a changed engine or caller-supplied population."""
    from tools.i7_weekly_branch_regimes import _verify_seal
    design = json.loads((root / "DESIGN.json").read_text())
    _verify_seal(design, "design_sha256")
    if design["code_sha256"] != code_identity() or design["expected_cells"] != 156:
        raise ValueError("campaign engine or population mismatch")
    return design


def cell_dir(root, week, samples):
    """Stable identity independent of the host running a cell."""
    return root / "cells" / f"week_{week:03d}_samples_{samples}"


def validate(root, week, samples):
    """Verify a terminal before skipping fitting or accepting it into aggregation."""
    from tools.i7_weekly_branch_regimes import _verify_seal
    directory = cell_dir(root, week, samples)
    cell = json.loads((directory / "CELL.json").read_text())
    _verify_seal(cell, "cell_sha256")
    design = read_design(root)
    if (cell["status"] != "COMPLETED" or cell["week_ordinal"] != week
            or cell["samples"] != samples or cell["seed"] != 6102026
            or cell["horizon_hours"] != 72 or cell["test_read"] is not False
            or cell["code_sha256"] != code_identity()):
        raise ValueError("cell contract mismatch")
    if (cell["week"] != design["parent"]["weeks"][week]
            or cell["parent_design_sha256"] != design["parent"]["design_sha256"]
            or cell["members"] != design["parent"]["members"]):
        raise ValueError("cell is not from registered campaign design")
    for name, key in (("MODEL.keras", "model_sha256"), ("PREDICTIONS.npz", "predictions_sha256")):
        if file_hash(directory / name) != cell[key]:
            raise ValueError(f"artifact mismatch: {name}")
    return cell


def status(root):
    """Expose progress and measured-throughput projection, not a promised finish time."""
    cells, issues = [], []
    for week in range(52):
        for samples in ARMS:
            if (cell_dir(root, week, samples) / "CELL.json").exists():
                try:
                    cells.append(validate(root, week, samples))
                except Exception as exc:
                    issues.append(dict(week=week, samples=samples, error=str(exc)))
    measured = [c["elapsed_seconds"] for c in cells if not c["reused_pilot_fit"]]
    report = dict(status="COMPLETE" if len(cells) == 156 and not issues else "IN_PROGRESS",
                  completed=len(cells), expected=156, issues=issues, test_read=False,
                  by_samples={str(n): sum(c["samples"] == n for c in cells) for n in ARMS},
                  mean_cell_seconds=None if not measured else sum(measured) / len(measured),
                  remaining_serial_seconds=None if not measured else
                      (156 - len(cells)) * sum(measured) / len(measured),
                  eta_note="serial compute projection; parallel wall ETA requires live worker counts")
    atomic_json(root / "CAMPAIGN_STATUS.json", report)
    return report


def close(root):
    """Reject incomplete evidence and require identical weekly populations across arms."""
    report = status(root)
    if report["status"] != "COMPLETE":
        raise ValueError("incomplete evidence; no aggregate")
    from tools.i7_multi_horizon import pool_records
    cells = {n: [validate(root, w, n) for w in range(52)] for n in ARMS}
    for week in range(52):
        reference = cells[3][week]
        for n in (5, 9):
            cell = cells[n][week]
            for key in ("parent_design_sha256", "input_digests", "members", "scaler_sha256",
                        "target_center", "target_scale", "fit_rows", "inner_rows", "parameters"):
                if cell[key] != reference[key]:
                    raise ValueError(f"unequal contrast identity: {key}")
            if cell["metrics"]["rows_sha256"] != reference["metrics"]["rows_sha256"]:
                raise ValueError("unpaired scoring populations")
    result = dict(report, metrics_by_samples={str(n): pool_records([c["metrics"] for c in cells[n]])
                                              for n in ARMS},
                  scope="neural compact R0, raw log return; not ARIMA controls or price-unit closure",
                  olap_status="LOCAL_METRIC_TERMINALS_ONLY_NOT_YET_LIVE_WAREHOUSE")
    atomic_json(root / "NEURAL_CLOSURE.json", result)
    return result


def worker(args):
    """Sequential isolated child fits; gamma owns two arms, dragon one."""
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    read_design(root)
    lease = root / f"WORKER_{args.slot}.claim"
    with lease.open("x") as stream:
        stream.write(str(time.time()))
    try:
        for week in range(52):
            for index, samples in enumerate(ARMS):
                if (index == 0) != (args.slot == "secondary"):
                    continue
                directory = cell_dir(root, week, samples)
                if (directory / "CELL.json").exists():
                    validate(root, week, samples)
                    continue
                command = [str(args.admission), "-m", str(args.memory_bytes), "-t", "7200",
                           "-n", f"i6e72-w{week:02d}-n{samples}", "--", sys.executable,
                           "-m", "tools.i6e_compact_run", "cell", "--output", str(directory),
                           "--week", str(week), "--samples", str(samples),
                           "--parent-design", str(args.parent_design),
                           "--target-parquet", str(args.target_parquet),
                           "--validation-target-parquet", str(args.validation_target_parquet)]
                for flag, paths in (("--feature-parquet", args.feature_parquet),
                                    ("--validation-feature-parquet", args.validation_feature_parquet)):
                    for path in paths:
                        command.extend([flag, str(path)])
                if args.pilot_root:
                    command.extend(["--pilot-root", str(args.pilot_root)])
                atomic_json(root / f"WORKER_{args.slot}.json", dict(status="RUNNING", week=week,
                            samples=samples, started_epoch=time.time()))
                subprocess.run(command, check=True)
                validate(root, week, samples)
                status(root)
        atomic_json(root / f"WORKER_{args.slot}.json", dict(status="COMPLETED"))
    except Exception as exc:
        atomic_json(root / f"WORKER_{args.slot}.json", dict(status="FAILED", error=str(exc)))
        raise
    finally:
        lease.unlink(missing_ok=True)


def main():
    """Launch or inspect without training in the status/closure code paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "worker", "status", "close"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slot", choices=["preferred", "secondary"])
    parser.add_argument("--memory-bytes", type=int)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--parent-design", type=Path)
    parser.add_argument("--target-parquet", type=Path)
    parser.add_argument("--validation-target-parquet", type=Path)
    parser.add_argument("--feature-parquet", type=Path, action="append")
    parser.add_argument("--validation-feature-parquet", type=Path, action="append")
    parser.add_argument("--pilot-root", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        if not args.parent_design:
            parser.error("init requires parent design")
        print(json.dumps(initialize(args.output, args.parent_design), indent=2))
    elif args.command == "worker":
        if not all((args.slot, args.memory_bytes, args.admission, args.parent_design,
                    args.target_parquet, args.validation_target_parquet,
                    args.feature_parquet, args.validation_feature_parquet)):
            parser.error("worker requires all data, admission and slot arguments")
        worker(args)
    else:
        print(json.dumps(close(args.output) if args.command == "close" else status(args.output), indent=2))


if __name__ == "__main__":
    main()
