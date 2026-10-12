"""Contract tests for the unattended H4 -> H72 island successor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools import i6n_island_successor as successor


COMPONENTS = {
    "doin-node": "node-commit",
    "doin-core": "core-commit",
    "doin-plugins": "plugins-commit",
    "predictor": "predictor-commit",
}
H4_CONFIG = "4" * 64
H72_CONFIG = "7" * 64


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _write_envelope(path: Path, schema: str, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema": schema, "payload": payload, "payload_sha256": _digest(payload)}
    path.write_text(json.dumps(document, sort_keys=True) + "\n")
    return document["payload_sha256"]


def _terminal(island_id: str, **changes) -> dict:
    payload = {
        "status": "COMPLETE",
        "island_id": island_id,
        "horizon": 4,
        "seed": 6102026,
        "scientific_config_sha256": H4_CONFIG,
        "network_id": "network",
        "genesis_hash": "genesis",
        "chain_height": 120,
        "tip_hash": "tip-120",
        "finalized_height": 114,
        "finalized_hash": "finalized-114",
        "component_versions": dict(COMPONENTS),
        "stop_reason": "patience",
        "terminal_generation": 41,
        "champion_sha256": hashlib.sha256(f"champion-{island_id}".encode()).hexdigest(),
        "champion_fitness": -0.01,
        "plan_sha256": "a" * 64,
        "contract_sha256": "b" * 64,
        "checkpoint_identity_sha256": "c" * 64,
        "final_population_sha256": hashlib.sha256(
            f"population-{island_id}".encode()
        ).hexdigest(),
        "candidate_ledger_sha256": hashlib.sha256(
            f"ledger-{island_id}".encode()
        ).hexdigest(),
        "evaluated_candidate_count": 600,
    }
    payload.update(changes)
    return payload


def _spec(tmp_path: Path) -> tuple[Path, dict]:
    terminal_paths = [tmp_path / "terminals" / f"island-{index}.json" for index in range(4)]
    spec = {
        "schema": successor.SPEC_SCHEMA,
        "state_dir": str(tmp_path / "state"),
        "poll_interval_seconds": 60,
        "launch_retry_seconds": 300,
        "terminal_receipts": [str(path) for path in terminal_paths],
        "expected": {
            "seed": 6102026,
            "h4_scientific_config_sha256": H4_CONFIG,
            "h72_scientific_config_sha256": H72_CONFIG,
            "network_id": "network",
            "genesis_hash": "genesis",
            "component_versions": dict(COMPONENTS),
        },
        "archive_closure_receipt": str(tmp_path / "archive" / "closure.json"),
        "h72_start": {
            "command": ["start-h72", "--private-spec", str(tmp_path / "h72.json")],
            "receipt": str(tmp_path / "h72" / "START.json"),
            "working_directory": str(tmp_path),
            "environment": {"PRIVATE_SETTING": "from-spec"},
            "verification_timeout_seconds": 2.0,
            "nodes": [
                {"node_id": f"island-{index}", "base_url": f"http://node-{index}.invalid"}
                for index in range(4)
            ],
        },
    }
    path = tmp_path / "private-successor.json"
    path.write_text(json.dumps(spec))
    return path, spec


def _write_terminals(spec: dict, per_island_changes: dict[int, dict] | None = None) -> list[str]:
    per_island_changes = per_island_changes or {}
    digests = []
    for index, raw_path in enumerate(spec["terminal_receipts"]):
        changes = per_island_changes.get(index, {})
        digests.append(_write_envelope(
            Path(raw_path), successor.TERMINAL_SCHEMA, _terminal(f"island-{index}", **changes)))
    return digests


def _write_closure(spec: dict, terminal_digests: list[str], **changes) -> str:
    payload = {
        "status": "GREEN",
        "horizon": 4,
        "scientific_config_sha256": H4_CONFIG,
        "terminal_receipt_sha256s": sorted(terminal_digests),
        "network_id": "network",
        "genesis_hash": "genesis",
        "chain_height": 120,
        "tip_hash": "tip-120",
        "finalized_height": 114,
        "finalized_hash": "finalized-114",
        "archive": {"items": 80, "green": 80, "missing": [], "submitted_unfinalized": 0},
        "olap": {"status": "GREEN", "chain_verified": True, "readback_verified": True},
    }
    payload.update(changes)
    return _write_envelope(
        Path(spec["archive_closure_receipt"]), successor.CLOSURE_SCHEMA, payload)


def _write_start_receipt(spec: dict, transition_id: str) -> str:
    return _write_envelope(Path(spec["h72_start"]["receipt"]), successor.START_SCHEMA, {
        "status": "STARTED",
        "horizon": 72,
        "scientific_config_sha256": H72_CONFIG,
        "transition_id": transition_id,
    })


def _fleet_fetcher(spec: dict, *, mutations: dict[tuple[int, str, str], object] | None = None,
                   unavailable: bool = False):
    mutations = mutations or {}
    responses = {}
    for index, node in enumerate(spec["h72_start"]["nodes"]):
        chain = {
            "chain_height": 200, "tip_hash": "h72-tip", "finalized_height": 194,
            "finalized_hash": "h72-finalized", "network_id": "network",
            "genesis_hash": "genesis", "component_versions": dict(COMPONENTS),
        }
        status = {
            "status": "healthy", "network_id": "network", "genesis_hash": "genesis",
            "optimization": {"seed": 6102026, "config_digest": H72_CONFIG},
        }
        for endpoint, payload in (("chain/status", chain), ("status", status)):
            for (wanted_index, wanted_endpoint, field), value in mutations.items():
                if wanted_index == index and wanted_endpoint == endpoint:
                    payload[field] = value
            responses[f"{node['base_url']}/{endpoint}"] = payload

    def fetch(url, timeout):
        assert timeout == 2.0
        if unavailable:
            raise OSError("not listening yet")
        return responses[url]

    return fetch


class RecordingLauncher:
    def __init__(self, spec, *, write_receipt=False, returncode=0):
        self.spec = spec
        self.write_receipt = write_receipt
        self.returncode = returncode
        self.calls = []
        self.effects = set()

    def __call__(self, command, *, cwd, env, check=False):
        self.calls.append((command, cwd, env, check))
        transition_id = env["I6N_TRANSITION_ID"]
        if self.write_receipt and transition_id not in self.effects:
            self.effects.add(transition_id)
            _write_start_receipt(self.spec, transition_id)

        class Result:
            pass

        result = Result()
        result.returncode = self.returncode
        return result


def test_waits_for_exactly_four_terminals_and_status_is_read_only(tmp_path):
    spec_path, spec = _spec(tmp_path)
    for index in range(3):
        _write_envelope(Path(spec["terminal_receipts"][index]), successor.TERMINAL_SCHEMA,
                        _terminal(f"island-{index}"))
    launcher = RecordingLauncher(spec)

    result = successor.tick(spec_path, run_command=launcher, now=lambda: 1000.0)
    status_path = Path(spec["state_dir"]) / "STATUS.json"
    before = status_path.read_bytes()
    observed = successor.status(spec_path, now=lambda: 1001.0)

    assert result["state"] == "WAIT_TERMINALS"
    assert observed["progress"]["terminals"] == {"found": 3, "required": 4, "valid": 3}
    assert observed["eta"]["kind"] == "unknown"
    assert status_path.read_bytes() == before
    assert launcher.calls == []


def test_identical_copied_terminal_is_refused(tmp_path):
    spec_path, spec = _spec(tmp_path)
    first = Path(spec["terminal_receipts"][0])
    _write_envelope(first, successor.TERMINAL_SCHEMA, _terminal("island-0"))
    for raw_path in spec["terminal_receipts"][1:]:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(first.read_bytes())

    with pytest.raises(successor.Refusal, match="DUPLICATE_TERMINAL"):
        successor.tick(spec_path, run_command=RecordingLauncher(spec))


def test_failed_terminal_is_never_promoted(tmp_path):
    spec_path, spec = _spec(tmp_path)
    _write_terminals(spec, {2: {"status": "FAILED", "stop_reason": "WALL_LIMIT"}})

    with pytest.raises(successor.Refusal, match="TERMINAL_NOT_COMPLETE"):
        successor.tick(spec_path, run_command=RecordingLauncher(spec))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", 99, "SEED_MISMATCH"),
        ("scientific_config_sha256", "other", "SCIENTIFIC_CONFIG_MISMATCH"),
        ("network_id", "other", "NETWORK_MISMATCH"),
        ("genesis_hash", "other", "GENESIS_MISMATCH"),
        ("chain_height", 121, "CHAIN_HEIGHT_MISMATCH"),
        ("tip_hash", "other", "TIP_HASH_MISMATCH"),
    ],
)
def test_terminal_fleet_must_have_one_exact_identity(tmp_path, field, value, message):
    spec_path, spec = _spec(tmp_path)
    _write_terminals(spec, {3: {field: value}})

    with pytest.raises(successor.Refusal, match=message):
        successor.tick(spec_path, run_command=RecordingLauncher(spec))


def test_terminal_requires_exact_four_component_commits(tmp_path):
    spec_path, spec = _spec(tmp_path)
    bad = dict(COMPONENTS)
    bad["predictor"] = "wrong"
    _write_terminals(spec, {1: {"component_versions": bad}})

    with pytest.raises(successor.Refusal, match="COMPONENT_VERSION_MISMATCH"):
        successor.tick(spec_path, run_command=RecordingLauncher(spec))


def test_archive_and_olap_must_be_green_and_bound_to_the_terminals(tmp_path):
    spec_path, spec = _spec(tmp_path)
    terminal_digests = _write_terminals(spec)
    launcher = RecordingLauncher(spec)

    waiting = successor.tick(spec_path, run_command=launcher)
    assert waiting["state"] == "WAIT_ARCHIVE"
    assert launcher.calls == []

    _write_closure(spec, terminal_digests, olap={
        "status": "GREEN", "chain_verified": True, "readback_verified": False})
    with pytest.raises(successor.Refusal, match="OLAP_NOT_GREEN"):
        successor.tick(spec_path, run_command=launcher)

    _write_closure(spec, terminal_digests[:-1])
    with pytest.raises(successor.Refusal, match="CLOSURE_TERMINALS_MISMATCH"):
        successor.tick(spec_path, run_command=launcher)


def test_transition_runs_once_and_repeated_ticks_adopt_final_receipt(tmp_path):
    spec_path, spec = _spec(tmp_path)
    terminals = _write_terminals(spec)
    _write_closure(spec, terminals)
    launcher = RecordingLauncher(spec)

    fetch = _fleet_fetcher(spec)
    first = successor.tick(spec_path, run_command=launcher, fetch_json=fetch, now=lambda: 1000.0)
    second = successor.tick(spec_path, run_command=launcher, fetch_json=fetch, now=lambda: 1001.0)

    assert first["state"] == second["state"] == "COMPLETE"
    assert first["transition_id"] == second["transition_id"]
    assert len(launcher.calls) == 1
    assert launcher.effects == set()  # the successor, not the launcher, owns the live-fleet receipt
    assert (Path(spec["state_dir"]) / "TRANSITION_RECEIPT.json").is_file()
    start = json.loads(Path(spec["h72_start"]["receipt"]).read_text())["payload"]
    assert start["status"] == "LIVE_FLEET_VERIFIED"
    assert start["seed"] == 6102026 and start["scientific_config_sha256"] == H72_CONFIG
    assert start["node_ids"] == [f"island-{index}" for index in range(4)]
    assert start["chain_height"] == 200 and start["tip_hash"] == "h72-tip"
    assert start["finalized_height"] == 194 and start["finalized_hash"] == "h72-finalized"
    assert start["component_versions"] == COMPONENTS


def test_crash_after_start_is_adopted_without_second_launch(tmp_path):
    spec_path, spec = _spec(tmp_path)
    terminals = _write_terminals(spec)
    _write_closure(spec, terminals)
    launcher = RecordingLauncher(spec)

    def crash():
        raise RuntimeError("simulated process loss")

    with pytest.raises(RuntimeError, match="simulated"):
        successor.tick(spec_path, run_command=launcher, fetch_json=_fleet_fetcher(spec), now=lambda: 1000.0,
                       after_launch=crash)
    recovered = successor.tick(spec_path, run_command=launcher, fetch_json=_fleet_fetcher(spec),
                               now=lambda: 1001.0)

    assert recovered["state"] == "COMPLETE"
    assert len(launcher.calls) == 1
    assert launcher.effects == set()


def test_unreachable_h72_fleet_waits_then_retries_same_idempotency_key(tmp_path):
    spec_path, spec = _spec(tmp_path)
    terminals = _write_terminals(spec)
    _write_closure(spec, terminals)
    launcher = RecordingLauncher(spec, write_receipt=False)

    fetch = _fleet_fetcher(spec, unavailable=True)
    first = successor.tick(spec_path, run_command=launcher, fetch_json=fetch, now=lambda: 1000.0)
    early = successor.tick(spec_path, run_command=launcher, fetch_json=fetch, now=lambda: 1100.0)
    late = successor.tick(spec_path, run_command=launcher, fetch_json=fetch, now=lambda: 1400.0)

    assert first["state"] == early["state"] == late["state"] == "WAIT_H72_FLEET"
    assert len(launcher.calls) == 2
    assert launcher.calls[0][2]["I6N_TRANSITION_ID"] == launcher.calls[1][2]["I6N_TRANSITION_ID"]
    assert early["eta"]["seconds"] == 200


def test_existing_wrong_start_receipt_is_refused(tmp_path):
    spec_path, spec = _spec(tmp_path)
    terminals = _write_terminals(spec)
    _write_closure(spec, terminals)
    _write_start_receipt(spec, "wrong-transition")

    with pytest.raises(successor.Refusal, match="START_RECEIPT_MISMATCH"):
        successor.tick(spec_path, run_command=RecordingLauncher(spec))


@pytest.mark.parametrize(
    ("endpoint", "field", "value", "message"),
    [
        ("status", "optimization", {"seed": 9, "config_digest": H72_CONFIG}, "SEED_MISMATCH"),
        ("status", "optimization", {"seed": 6102026, "config_digest": "8" * 64},
         "CONFIG_DIGEST_MISMATCH"),
        ("chain/status", "tip_hash", "divergent", "TIP_HASH_MISMATCH"),
        ("chain/status", "chain_height", 199, "CHAIN_HEIGHT_MISMATCH"),
        ("chain/status", "network_id", "divergent", "NETWORK_MISMATCH"),
        ("chain/status", "genesis_hash", "divergent", "GENESIS_MISMATCH"),
        ("chain/status", "finalized_hash", "divergent", "FINALIZED_CHECKPOINT_MISMATCH"),
        ("chain/status", "component_versions", dict(COMPONENTS, predictor="wrong"),
         "COMPONENT_VERSION_MISMATCH"),
    ],
)
def test_h72_live_fleet_mismatch_never_generates_start_receipt(
        tmp_path, endpoint, field, value, message):
    spec_path, spec = _spec(tmp_path)
    terminals = _write_terminals(spec)
    _write_closure(spec, terminals)
    fetch = _fleet_fetcher(spec, mutations={(3, endpoint, field): value})

    with pytest.raises(successor.Refusal, match=message):
        successor.tick(spec_path, run_command=RecordingLauncher(spec), fetch_json=fetch)

    assert not Path(spec["h72_start"]["receipt"]).exists()


def test_cli_refusal_is_json_and_status_mode_never_persists_it(tmp_path, capsys):
    spec_path, spec = _spec(tmp_path)
    _write_terminals(spec, {0: {"status": "FAILED"}})

    assert successor.main(["--spec", str(spec_path), "--status"]) == 2
    status_path = Path(spec["state_dir"]) / "STATUS.json"
    assert not status_path.exists()
    readonly = json.loads(capsys.readouterr().out)
    assert readonly["state"] == "REFUSED"

    assert successor.main(["--spec", str(spec_path)]) == 2
    persisted = json.loads(status_path.read_text())
    assert persisted["state"] == "REFUSED"
    assert "TERMINAL_NOT_COMPLETE" in persisted["detail"]


def test_private_spec_inside_repository_is_refused(tmp_path):
    repo_spec = Path(successor.__file__).resolve().parents[1] / "private-successor.json"
    repo_spec.write_text(json.dumps({"schema": successor.SPEC_SCHEMA}))
    try:
        with pytest.raises(successor.SpecError, match="outside the repository"):
            successor.load_spec(repo_spec)
    finally:
        repo_spec.unlink()


def test_systemd_templates_are_unattended_and_contain_no_private_hosts():
    root = Path(successor.__file__).resolve().parent
    service = (root / "i6n-island-successor.service").read_text()
    timer = (root / "i6n-island-successor.timer").read_text()

    assert "Type=oneshot" in service
    assert "--spec %h/.config/i6n/island-successor.json" in service
    assert "Persistent=true" in timer
    assert "OnUnitInactiveSec=60s" in timer
    for private_fragment in ("gamma", "dragon", "omega", "192.168.", "10.0."):
        assert private_fragment not in service + timer
