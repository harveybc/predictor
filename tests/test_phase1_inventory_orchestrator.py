from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import pytest

from tools.phase1_inventory_orchestrator import (
    Phase1Refusal,
    build_plan,
    finalize_if_ready,
    load_config,
    load_inventory,
    require_phase1_complete,
    run_host_once,
)
from tools.phase1_inventory_status import build_status


def _write_inventory(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "feature_id",
        "role",
        "admissibility",
        "train_rows",
        "source_bytes",
        "input_state",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _worker(path: Path) -> None:
    path.write_text(
        """\
import argparse, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--row', required=True)
p.add_argument('--output', required=True)
p.add_argument('--calls', required=True)
a = p.parse_args()
row = json.loads(Path(a.row).read_text())
with Path(a.calls).open('a') as stream:
    stream.write(row['feature_id'] + '\\n')
if row.get('input_state') != 'AVAILABLE':
    raise SystemExit(66)
result = {
    'schema': 'phase1.column_result.v1',
    'feature_id': row['feature_id'],
    'state': 'COMPLETED',
    'cpu_only_observed': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
    'envelope': {'schema': 'feature_selection_envelope.v1',
                 'feature_id': row['feature_id']},
}
Path(a.output).write_text(json.dumps(result, sort_keys=True))
""",
        encoding="utf-8",
    )


def _finalizer(path: Path) -> None:
    path.write_text(
        """\
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--terminals', required=True)
a = p.parse_args()
Path(a.output).write_text(json.dumps({
    'schema': 'phase1.finalizer_result.v1',
    'state': 'PHASE_1_COMPLETE',
    'terminal_count': len(list(Path(a.terminals).glob('*.json'))),
}, sort_keys=True))
""",
        encoding="utf-8",
    )


def _row(
    feature_id: str,
    rows: int,
    source_bytes: int,
    input_state: str = "AVAILABLE",
) -> dict[str, str]:
    return {
        "feature_id": feature_id,
        "role": "feature",
        "admissibility": "ADMISSIBLE",
        "train_rows": str(rows),
        "source_bytes": str(source_bytes),
        "input_state": input_state,
    }


def _config(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    expected_total: int | None = None,
    stale_after_seconds: float = 60.0,
) -> Path:
    inventory = tmp_path / "inventory.csv"
    worker = tmp_path / "worker.py"
    finalizer = tmp_path / "finalizer.py"
    calls = tmp_path / "calls.txt"
    _write_inventory(inventory, rows)
    _worker(worker)
    _finalizer(finalizer)
    document = {
        "schema": "phase1.inventory_orchestrator.v1",
        "phase": "PHASE_1",
        "state_root": str(tmp_path / "state"),
        "inventory": {
            "files": [str(inventory)],
            "id_field": "feature_id",
            "row_count_field": "train_rows",
            "bytes_field": "source_bytes",
            "availability_field": "input_state",
            "available_values": ["AVAILABLE"],
            "include": {"role": "feature", "admissibility": "ADMISSIBLE"},
            "expected_total": expected_total if expected_total is not None else len(rows),
            "bytes_per_row": 8,
            "bytes_per_cost_unit": 8,
        },
        "assignment": {"smallest_fraction": 0.34},
        "hosts": [
            {"id": "small", "size_class": "small", "command_prefix": []},
            {"id": "large-a", "size_class": "large", "command_prefix": []},
            {"id": "large-b", "size_class": "large", "command_prefix": []},
        ],
        "worker": {
            "command": [
                "python",
                str(worker),
                "--row",
                "{inventory_row_path}",
                "--output",
                "{worker_output_path}",
                "--calls",
                str(calls),
            ],
            "timeout_seconds": 10,
            "unavailable_exit_codes": [66],
            "environment": {},
        },
        "warehouse": {
            "url": "http://warehouse.invalid/api/v2/feature-selection-envelopes",
            "timeout_seconds": 1,
        },
        "finalizer": {
            "command": [
                "python",
                str(finalizer),
                "--output",
                "{finalizer_output_path}",
                "--terminals",
                "{terminals_dir}",
            ],
            "timeout_seconds": 10,
        },
        "claims": {"stale_after_seconds": stale_after_seconds},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _accepted_submit(document: dict, _warehouse: dict) -> dict:
    return {"accepted": True, "feature_id": document["feature_id"]}


def test_assignment_is_deterministic_and_keeps_smallest_on_small_host(
    tmp_path: Path,
) -> None:
    rows = [
        _row("largest", 1000, 8000),
        _row("tiny-b", 10, 80),
        _row("middle", 100, 800),
        _row("tiny-a", 10, 40),
        _row("large", 900, 7200),
        _row("medium", 200, 1600),
    ]
    path = _config(tmp_path, rows)
    config = load_config(path)
    first = build_plan(config)
    second = build_plan(config)

    assert first == second
    assigned = {row["feature_id"]: row["host_id"] for row in first["items"]}
    assert assigned["tiny-a"] == "small"
    assert assigned["tiny-b"] == "small"
    assert {assigned["largest"], assigned["large"]} <= {"large-a", "large-b"}
    assert first["inventory_total"] == 6


def test_denominator_mismatch_is_refused(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("a", 10, 80)], expected_total=2)
    config = load_config(path)

    with pytest.raises(Phase1Refusal, match="expected 2.*observed 1"):
        load_inventory(config)


def test_canonical_inventory_filter_resolves_exactly_366_features(tmp_path: Path) -> None:
    canonical = Path(__file__).parents[1] / "docs/audits/evidence/canonical_20261003/laneA"
    path = _config(tmp_path, [_row("placeholder", 1, 8)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["inventory"].update(
        files=[
            str(canonical / batch / "admissible_features.csv")
            for batch in ("batch_001", "batch_002", "batch_003")
        ],
        expected_total=366,
        bytes_field=None,
        availability_field=None,
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    items = load_inventory(load_config(path))

    assert len(items) == 366
    assert len({item.feature_id for item in items}) == 366


def test_exclusive_claim_prevents_duplicate_execution(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("only", 10, 80)])
    config = load_config(path)
    build_plan(config)
    results: list[dict] = []

    def execute() -> None:
        results.append(run_host_once(config, "small", submit=_accepted_submit))

    a = threading.Thread(target=execute)
    b = threading.Thread(target=execute)
    a.start()
    b.start()
    a.join()
    b.join()

    calls = (tmp_path / "calls.txt").read_text(encoding="utf-8").splitlines()
    assert calls == ["only"]
    assert sorted(result["action"] for result in results) == ["IDLE", "TERMINAL_WRITTEN"]


def test_stale_claim_is_archived_and_work_is_recovered(tmp_path: Path) -> None:
    path = _config(
        tmp_path, [_row("recover", 10, 80)], stale_after_seconds=0.01
    )
    config = load_config(path)
    plan = build_plan(config)
    item = plan["items"][0]
    claim = Path(config["state_root"]) / "claims" / f"{item['key']}.claim"
    claim.mkdir(parents=True)
    (claim / "claim.json").write_text(
        json.dumps({"feature_id": "recover", "host_id": "small", "started_epoch": 0}),
        encoding="utf-8",
    )

    result = run_host_once(config, "small", submit=_accepted_submit)

    assert result["action"] == "TERMINAL_WRITTEN"
    assert any((Path(config["state_root"]) / "stale_claims").iterdir())


def test_unavailable_input_gets_explicit_terminal(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("missing", 10, 80, "NOT_AVAILABLE")])
    config = load_config(path)
    build_plan(config)

    result = run_host_once(config, "small", submit=_accepted_submit)
    status = build_status(config)

    assert result["terminal_state"] == "UNAVAILABLE"
    assert status["unavailable"] == 1
    assert status["pending"] == 0
    assert status["total"] == 1


def test_worker_is_cpu_only_and_envelope_is_submitted(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("cpu", 10, 80)])
    config = load_config(path)
    plan = build_plan(config)
    submitted: list[dict] = []

    def submit(document: dict, _warehouse: dict) -> dict:
        submitted.append(document)
        return {"accepted": True}

    result = run_host_once(config, "small", submit=submit)
    terminal_path = Path(config["state_root"]) / "terminals" / f"{plan['items'][0]['key']}.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))

    assert result["terminal_state"] == "COMPLETED"
    assert terminal["result"]["cpu_only_observed"] is True
    assert submitted == [terminal["result"]["envelope"]]
    assert len(list((Path(config["state_root"]) / "warehouse_receipts").glob("*.json"))) == 1


def test_finalizer_and_phase2_gate_require_complete_denominator(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("a", 10, 80), _row("b", 100, 800)])
    config = load_config(path)
    build_plan(config)

    with pytest.raises(Phase1Refusal, match="PHASE_1_COMPLETE"):
        require_phase1_complete(config)
    assert finalize_if_ready(config)["action"] == "WAITING_FOR_TERMINALS"

    while build_status(config)["pending"]:
        for host in ("small", "large-a", "large-b"):
            run_host_once(config, host, submit=_accepted_submit)

    final = finalize_if_ready(config)
    gate = require_phase1_complete(config)
    status = build_status(config)

    assert final["action"] == "PHASE_1_COMPLETE"
    assert gate["state"] == "PHASE_1_COMPLETE"
    assert status["phase_state"] == "PHASE_1_COMPLETE"
    assert status["total"] == 2
    assert status["completed"] == 2
    assert status["current"] == {"large-a": None, "large-b": None, "small": None}
    assert status["eta_seconds"] == 0.0


def test_finalizer_does_not_run_while_any_item_lacks_a_terminal(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("a", 10, 80), _row("b", 100, 800)])
    config = load_config(path)
    build_plan(config)
    run_host_once(config, "small", submit=_accepted_submit)

    before = list((Path(config["state_root"]) / "finalizer").glob("*"))
    outcome = finalize_if_ready(config)
    after = list((Path(config["state_root"]) / "finalizer").glob("*"))

    assert outcome["action"] == "WAITING_FOR_TERMINALS"
    assert before == after == []
