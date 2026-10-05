from __future__ import annotations

import csv
import hashlib
import json
import threading
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


def _canonical_sha(document: object) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--calls', required=True)
p.add_argument('--behavior', required=True)
a = p.parse_args()
request = json.load(sys.stdin)
with Path(a.calls).open('a') as stream:
    stream.write(request['feature_id'] + '\\n')
behavior = Path(a.behavior).read_text().strip() if Path(a.behavior).exists() else 'complete'
if behavior == 'rc75':
    raise SystemExit(75)
if behavior == 'oom':
    raise SystemExit(137)
if behavior == 'admission':
    print(json.dumps({
        'schema': 'phase1.column_result.v1',
        'feature_id': request['feature_id'],
        'state': 'FAILED',
        'failure_class': 'ADMISSION',
        'reason': 'host headroom unavailable',
    }, sort_keys=True))
    raise SystemExit(1)
if behavior == 'failed':
    print(json.dumps({
        'schema': 'phase1.column_result.v1',
        'feature_id': request['feature_id'],
        'state': 'FAILED',
        'reason': 'scientific worker failure',
    }, sort_keys=True))
    raise SystemExit(1)
result = {
    'schema': 'phase1.column_result.v1',
    'feature_id': request['feature_id'],
    'state': 'COMPLETED',
    'cpu_only_observed': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
    'request_sha256': request['request_sha256'],
    'envelope': {
        'schema': 'feature_selection_envelope.v1',
        'feature_id': request['feature_id'],
    },
}
print(json.dumps(result, sort_keys=True))
""",
        encoding="utf-8",
    )


def _remote_prefix(path: Path) -> None:
    path.write_text(
        """\
import json, subprocess, sys
from pathlib import Path
p = Path(sys.argv[1])
command = sys.argv[2:]
request = sys.stdin.read()
p.write_text(json.dumps({'argv': command, 'request': json.loads(request)}, sort_keys=True))
child = subprocess.run(command, input=request, capture_output=True, text=True)
sys.stdout.write(child.stdout)
sys.stderr.write(child.stderr)
raise SystemExit(child.returncode)
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
    'envelope': {
        'schema_version': 'feature_selection_envelope.v1',
        'run': {'run_id': 'final-fixture'},
        'rows': {},
        'envelope_sha256': 'fixture-final-envelope',
    },
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
    max_attempts: int = 3,
) -> Path:
    inventory = tmp_path / "inventory.csv"
    worker = tmp_path / "worker.py"
    finalizer = tmp_path / "finalizer.py"
    calls = tmp_path / "calls.txt"
    behavior = tmp_path / "behavior.txt"
    behavior.write_text("complete", encoding="utf-8")
    _write_inventory(inventory, rows)
    _worker(worker)
    _finalizer(finalizer)
    document = {
        "schema": "phase1.inventory_orchestrator.v2",
        "phase": "PHASE_1",
        "population": {"id": "EURUSD", "target_pack": "eurusd-short-long-v1"},
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
            "transport": "stdio-json-v1",
            "command": [
                "python",
                str(worker),
                "--calls",
                str(calls),
                "--behavior",
                str(behavior),
            ],
            "timeout_seconds": 10,
            "environment": {},
        },
        "retries": {
            "max_attempts": max_attempts,
            "retryable_exit_codes": [75, 137, -9],
            "retryable_result_states": ["FAILED"],
            "retryable_failure_classes": ["TIMEOUT", "ADMISSION", "OOM"],
        },
        "warehouse": {
            "submit_url": "http://warehouse.invalid/api/v2/feature-selection-envelopes",
            "reconcile_url": "http://warehouse.invalid/api/v2/feature-selection-reconcile",
            "timeout_seconds": 1,
            "token_env": "TEST_PHASE1_WAREHOUSE_TOKEN",
            "auth_profile": "test-service-token",
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
    return {"accepted": True, "feature_id": document.get("feature_id")}


def _stored_submit(_document: dict, _warehouse: dict) -> dict:
    return {"status": 201, "body": {"stored": True, "already_stored": False}}


def _accepted_reconcile(request: dict, _warehouse: dict) -> dict:
    response = {
        "schema": "phase1.warehouse_reconciliation.v1",
        "state": "RECONCILED",
        "plan_sha256": request["plan_sha256"],
        "expected_count": request["expected_count"],
        "identities_sha256": request["identities_sha256"],
        "authentication_profile": request["authentication_profile"],
    }
    response["reconciliation_sha256"] = _canonical_sha(response)
    return response


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
    config = load_config(_config(tmp_path, rows))
    first = build_plan(config)
    second = build_plan(config)

    assert first == second
    assigned = {row["feature_id"]: row["host_id"] for row in first["items"]}
    assert assigned["tiny-a"] == "small"
    assert assigned["tiny-b"] == "small"
    assert {assigned["largest"], assigned["large"]} <= {"large-a", "large-b"}
    assert first["population_id"] == "EURUSD"


def test_profile_only_plan_seals_mode_and_campaign(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("a", 10, 80)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["population"].update(
        mode="PROFILE_ONLY", campaign_sha256="a" * 64
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    plan = build_plan(load_config(path))

    assert plan["mode"] == "PROFILE_ONLY"
    assert plan["campaign_sha256"] == "a" * 64


def test_denominator_mismatch_is_refused(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("a", 10, 80)], expected_total=2))
    with pytest.raises(Phase1Refusal, match="expected 2.*observed 1"):
        load_inventory(config)


def test_canonical_eurusd_inventory_resolves_exactly_366_features(tmp_path: Path) -> None:
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
    assert len(items) == len({item.feature_id for item in items}) == 366


def test_population_templates_are_independent() -> None:
    root = Path(__file__).parents[1] / "docs/phase1_inventory"
    eurusd = json.loads((root / "canonical_config.template.json").read_text())
    eth = json.loads((root / "eth_config.template.json").read_text())

    assert eurusd["population"]["id"] == "EURUSD"
    assert eurusd["inventory"]["expected_total"] == 366
    assert eth["population"]["id"] == "ETH"
    assert eth["inventory"]["expected_total"] == 83
    assert eurusd["state_root"] != eth["state_root"]
    assert eurusd["population"]["target_pack"] != eth["population"]["target_pack"]


def test_remote_prefix_uses_stdio_and_never_coordinator_paths(tmp_path: Path) -> None:
    config_path = _config(tmp_path, [_row("remote", 10, 80)])
    document = json.loads(config_path.read_text(encoding="utf-8"))
    prefix = tmp_path / "remote_prefix.py"
    observation = tmp_path / "remote_observation.json"
    _remote_prefix(prefix)
    document["hosts"][0]["command_prefix"] = [
        "python",
        str(prefix),
        str(observation),
    ]
    config_path.write_text(json.dumps(document), encoding="utf-8")
    config = load_config(config_path)
    plan = build_plan(config)

    result = run_host_once(config, "small", submit=_accepted_submit)
    observed = json.loads(observation.read_text(encoding="utf-8"))

    assert result["terminal_state"] == "COMPLETED"
    assert observed["request"]["inventory_row"]["feature_id"] == "remote"
    assert observed["request"]["schema"] == "phase1.column_request.v1"
    assert not any("inventory_row_path" in token or "worker_output_path" in token for token in observed["argv"])
    assert not any(str(Path(config["state_root"])) in token for token in observed["argv"])
    terminal = json.loads(
        (
            Path(config["state_root"])
            / "terminals"
            / f"{plan['items'][0]['key']}.json"
        ).read_text()
    )
    assert terminal["result"]["cpu_only_observed"] is True


def test_exclusive_claim_prevents_duplicate_execution(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("only", 10, 80)]))
    build_plan(config)
    results: list[dict] = []

    def execute() -> None:
        results.append(run_host_once(config, "small", submit=_accepted_submit))

    first = threading.Thread(target=execute)
    second = threading.Thread(target=execute)
    first.start()
    second.start()
    first.join()
    second.join()

    calls = (tmp_path / "calls.txt").read_text(encoding="utf-8").splitlines()
    assert calls == ["only"]
    assert sorted(result["action"] for result in results) == ["IDLE", "TERMINAL_WRITTEN"]


def test_stale_claim_is_archived_and_work_is_recovered(tmp_path: Path) -> None:
    config = load_config(
        _config(tmp_path, [_row("recover", 10, 80)], stale_after_seconds=0.01)
    )
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


def test_unavailable_input_gets_explicit_closure_terminal(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("missing", 10, 80, "NOT_AVAILABLE")]))
    build_plan(config)
    result = run_host_once(config, "small", submit=_accepted_submit)
    status = build_status(config)

    assert result["terminal_state"] == "UNAVAILABLE"
    assert status["unavailable"] == 1
    assert status["pending"] == 0


@pytest.mark.parametrize("behavior", ["failed", "rc75", "oom", "admission"])
def test_failures_retry_boundedly_and_never_satisfy_closure(
    tmp_path: Path, behavior: str
) -> None:
    config = load_config(_config(tmp_path, [_row("bad", 10, 80)], max_attempts=2))
    (tmp_path / "behavior.txt").write_text(behavior, encoding="utf-8")
    plan = build_plan(config)

    first = run_host_once(config, "small", submit=_accepted_submit)
    second = run_host_once(config, "small", submit=_accepted_submit)
    third = run_host_once(config, "small", submit=_accepted_submit)
    status = build_status(config)
    final = finalize_if_ready(
        config, submit=_accepted_submit, reconcile=_accepted_reconcile
    )

    assert first["action"] == "RETRY_SCHEDULED"
    assert second["action"] == "RETRY_EXHAUSTED"
    assert third["action"] == "IDLE"
    attempt_dir = Path(config["state_root"]) / "attempts" / plan["items"][0]["key"]
    assert len(list(attempt_dir.glob("*.json"))) == 2
    assert not list((Path(config["state_root"]) / "terminals").glob("*.json"))
    assert status["failed"] == 1
    assert status["closure_eligible"] == 0
    assert final["action"] == "BLOCKED_BY_FAILURES"
    with pytest.raises(Phase1Refusal, match="PHASE_1_COMPLETE"):
        require_phase1_complete(config)


def test_retry_contract_cannot_make_admission_or_oom_terminal(tmp_path: Path) -> None:
    path = _config(tmp_path, [_row("a", 10, 80)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["retries"]["retryable_exit_codes"] = [75]
    document["retries"]["retryable_failure_classes"] = ["TIMEOUT"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(Phase1Refusal, match="admission/OOM/rc75"):
        load_config(path)


def test_finalization_waits_for_every_receipt_and_authenticated_readback(
    tmp_path: Path,
) -> None:
    config = load_config(_config(tmp_path, [_row("a", 10, 80)]))
    build_plan(config)

    def rejected_submit(_document: dict, _warehouse: dict) -> dict:
        raise OSError("warehouse offline")

    run_host_once(config, "small", submit=rejected_submit)
    waiting = finalize_if_ready(config, submit=rejected_submit, reconcile=_accepted_reconcile)
    assert waiting["action"] == "WAITING_FOR_WAREHOUSE_RECEIPTS"

    reconciliations: list[dict] = []

    def reconcile(request: dict, warehouse: dict) -> dict:
        reconciliations.append(request)
        return _accepted_reconcile(request, warehouse)

    submissions: list[dict] = []

    def submit(document: dict, warehouse: dict) -> dict:
        submissions.append(document)
        return _accepted_submit(document, warehouse)

    final = finalize_if_ready(config, submit=submit, reconcile=reconcile)
    gate = require_phase1_complete(config)

    assert final["action"] == "PHASE_1_COMPLETE"
    assert len(submissions) == 2
    assert submissions[-1]["run"]["run_id"] == "final-fixture"
    assert gate["final_envelope_sha256"] == _canonical_sha(submissions[-1])
    assert gate["final_warehouse_receipt_sha256"]
    assert reconciliations[0]["expected_count"] == 1
    stored = json.loads(
        (Path(config["state_root"]) / "warehouse_reconciliation.json").read_text()
    )
    assert gate["warehouse_reconciliation_sha256"] == _canonical_sha(stored)


def test_live_warehouse_stored_receipt_is_accepted(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("a", 10, 80)]))
    build_plan(config)

    result = run_host_once(config, "small", submit=_stored_submit)

    assert result["action"] == "TERMINAL_WRITTEN"
    assert len(list((Path(config["state_root"]) / "warehouse_receipts").glob("*.json"))) == 1


def test_rejected_warehouse_submission_is_not_a_receipt(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("a", 10, 80)]))
    build_plan(config)

    run_host_once(
        config,
        "small",
        submit=lambda _document, _warehouse: {"accepted": False},
    )
    result = finalize_if_ready(
        config,
        submit=lambda _document, _warehouse: {"accepted": False},
        reconcile=_accepted_reconcile,
    )

    assert result["action"] == "WAITING_FOR_WAREHOUSE_RECEIPTS"
    assert not list((Path(config["state_root"]) / "warehouse_receipts").glob("*.json"))


def test_forged_or_incomplete_readback_cannot_open_gate(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("a", 10, 80)]))
    build_plan(config)
    run_host_once(config, "small", submit=_accepted_submit)

    def forged(request: dict, _warehouse: dict) -> dict:
        return {
            "schema": "phase1.warehouse_reconciliation.v1",
            "state": "RECONCILED",
            "plan_sha256": request["plan_sha256"],
            "expected_count": 0,
            "identities_sha256": request["identities_sha256"],
            "authentication_profile": request["authentication_profile"],
            "reconciliation_sha256": "0" * 64,
        }

    with pytest.raises(Phase1Refusal, match="reconciliation"):
        finalize_if_ready(config, reconcile=forged)
    with pytest.raises(Phase1Refusal, match="PHASE_1_COMPLETE"):
        require_phase1_complete(config)


def test_phase2_gate_requires_complete_population(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path, [_row("a", 10, 80), _row("b", 100, 800)]))
    build_plan(config)
    with pytest.raises(Phase1Refusal, match="PHASE_1_COMPLETE"):
        require_phase1_complete(config)
    assert finalize_if_ready(config, reconcile=_accepted_reconcile)["action"] == "WAITING_FOR_TERMINALS"

    while build_status(config)["pending"]:
        for host in ("small", "large-a", "large-b"):
            run_host_once(config, host, submit=_accepted_submit)

    final = finalize_if_ready(
        config, submit=_accepted_submit, reconcile=_accepted_reconcile
    )
    status = build_status(config)
    assert final["action"] == "PHASE_1_COMPLETE"
    assert status["phase_state"] == "PHASE_1_COMPLETE"
    assert status["closure_eligible"] == status["total"] == 2
    assert status["eta_seconds"] == 0.0
