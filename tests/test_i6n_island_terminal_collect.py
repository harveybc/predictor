"""Contract tests for atomic collection of four live island terminals."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from tools import i6n_island_terminal_collect as collector
from tools import i6n_island_successor as successor


DIGEST = "a" * 64
COMPONENTS = {
    "doin-node": "node-commit",
    "doin-core": "core-commit",
    "doin-plugins": "plugins-commit",
    "predictor": "predictor-commit",
}


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _spec(tmp_path: Path) -> tuple[Path, dict]:
    spec = {
        "schema": collector.SPEC_SCHEMA,
        "state_dir": str(tmp_path / "state"),
        "output_dir": str(tmp_path / "terminals"),
        "poll_interval_seconds": 60,
        "timeout_seconds": 5,
        "expected": {
            "horizon": 4,
            "seed": 6102026,
            "scientific_config_sha256": DIGEST,
            "component_versions": dict(COMPONENTS),
        },
        "nodes": [
            {"node_id": f"node-{index}", "base_url": f"https://node-{index}.invalid"}
            for index in range(4)
        ],
    }
    path = tmp_path / "private-collector.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path, spec


def _responses(spec: dict) -> dict[str, dict]:
    responses = {}
    for index, node in enumerate(spec["nodes"]):
        base = node["base_url"]
        responses[f"{base}/chain/status"] = {
            "chain_height": 73,
            "tip_hash": "tip-73",
            "finalized_height": 70,
            "finalized_hash": "finalized-70",
            "network_id": "i6n-production",
            "genesis_hash": "genesis-i6n",
            "component_versions": dict(COMPONENTS),
        }
        responses[f"{base}/status"] = {
            "status": "healthy",
            "chain_height": 73,
            "optimization": {
                "seed": 6102026,
                "config_digest": DIGEST,
                "terminal": {
                    "status": "COMPLETE",
                    "island_id": f"island-{index}",
                    "horizon": 4,
                    "seed": 6102026,
                    "scientific_config_sha256": DIGEST,
                    "stop_reason": "patience",
                    "terminal_generation": 41 + index,
                    "champion_sha256": hashlib.sha256(
                        f"champion-{index}".encode()
                    ).hexdigest(),
                    "champion_fitness": -0.01 - index / 1000,
                    "plan_sha256": "b" * 64,
                    "contract_sha256": "c" * 64,
                    "checkpoint_identity_sha256": "d" * 64,
                    "final_population_sha256": hashlib.sha256(
                        f"population-{index}".encode()
                    ).hexdigest(),
                    "candidate_ledger_sha256": hashlib.sha256(
                        f"ledger-{index}".encode()
                    ).hexdigest(),
                    "evaluated_candidate_count": 600 + index,
                },
            },
            "network_id": "i6n-production",
            "genesis_hash": "genesis-i6n",
        }
    return responses


class Fetcher:
    def __init__(self, responses, barrier: threading.Barrier | None = None):
        self.responses = responses
        self.barrier = barrier
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, timeout):
        assert timeout > 0
        with self.lock:
            self.calls.append(url)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


def _documents(spec: dict) -> list[dict]:
    output = Path(spec["output_dir"])
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(output.glob("island-*.json"))]


def test_one_concurrent_observation_atomically_publishes_four_successor_envelopes(tmp_path):
    spec_path, spec = _spec(tmp_path)
    fetcher = Fetcher(_responses(spec), threading.Barrier(8))

    result = collector.tick(spec_path, fetch_json=fetcher, now_ns=lambda: 123456789)
    documents = _documents(spec)

    assert result["state"] == "COMPLETE"
    assert result["progress"] == {"observed": 4, "required": 4, "published": 4}
    assert len(fetcher.calls) == 8
    assert len(set(fetcher.calls)) == 8
    assert len(documents) == 4
    assert len({_digest(document) for document in documents}) == 4
    observation_ids = set()
    for document in documents:
        assert set(document) == {"schema", "payload", "payload_sha256"}
        assert document["schema"] == "i6n.island.terminal.v1"
        assert document["payload_sha256"] == _digest(document["payload"])
        payload = document["payload"]
        assert payload["status"] == "COMPLETE"
        assert payload["chain_height"] == 73
        assert payload["tip_hash"] == "tip-73"
        assert payload["finalized_height"] == 70
        assert payload["finalized_hash"] == "finalized-70"
        assert payload["component_versions"] == COMPONENTS
        assert payload["evaluated_candidate_count"] >= 600
        assert len(payload["candidate_ledger_sha256"]) == 64
        observation_ids.add(payload["observation_sha256"])
    assert len(observation_ids) == 1

    successor_expected = {
        "seed": 6102026,
        "h4_scientific_config_sha256": DIGEST,
        "network_id": "i6n-production",
        "genesis_hash": "genesis-i6n",
        "component_versions": COMPONENTS,
    }
    for index, path in enumerate(sorted(Path(spec["output_dir"]).glob("*.json"))):
        payload, _receipt_sha = successor._read_envelope(
            path, successor.TERMINAL_SCHEMA, f"terminal_receipts[{index}]"
        )
        successor._validate_terminal(
            payload, successor_expected, f"terminal_receipts[{index}]"
        )


def test_unreachable_node_or_incomplete_terminal_never_publishes_a_subset(tmp_path):
    spec_path, spec = _spec(tmp_path)
    responses = _responses(spec)
    responses[f"{spec['nodes'][3]['base_url']}/status"] = OSError("offline")

    result = collector.tick(spec_path, fetch_json=Fetcher(responses))

    assert result["state"] == "WAIT_OBSERVATION"
    assert not Path(spec["output_dir"]).exists()

    responses = _responses(spec)
    responses[f"{spec['nodes'][2]['base_url']}/status"]["optimization"]["terminal"]["status"] = "RUNNING"
    result = collector.tick(spec_path, fetch_json=Fetcher(responses))
    assert result["state"] == "WAIT_TERMINALS"
    assert not Path(spec["output_dir"]).exists()


def test_failed_or_duplicate_island_terminal_is_refused_without_output(tmp_path):
    spec_path, spec = _spec(tmp_path)
    responses = _responses(spec)
    responses[f"{spec['nodes'][0]['base_url']}/status"]["optimization"]["terminal"]["status"] = "FAILED"
    with pytest.raises(collector.Refusal, match="TERMINAL_FAILED"):
        collector.tick(spec_path, fetch_json=Fetcher(responses))
    assert not Path(spec["output_dir"]).exists()

    responses = _responses(spec)
    duplicate = responses[f"{spec['nodes'][0]['base_url']}/status"]["optimization"]["terminal"]["island_id"]
    responses[f"{spec['nodes'][1]['base_url']}/status"]["optimization"]["terminal"]["island_id"] = duplicate
    with pytest.raises(collector.Refusal, match="DUPLICATE_ISLAND"):
        collector.tick(spec_path, fetch_json=Fetcher(responses))
    assert not Path(spec["output_dir"]).exists()


@pytest.mark.parametrize(
    ("endpoint", "field", "value", "message"),
    [
        ("status", "seed", 7, "SEED_MISMATCH"),
        ("status", "config_digest", "b" * 64, "CONFIG_DIGEST_MISMATCH"),
        ("chain/status", "network_id", "other", "NETWORK_MISMATCH"),
        ("chain/status", "genesis_hash", "other", "GENESIS_MISMATCH"),
        ("chain/status", "chain_height", 72, "CHAIN_HEIGHT_MISMATCH"),
        ("chain/status", "tip_hash", "other", "TIP_HASH_MISMATCH"),
        ("chain/status", "finalized_height", 69, "FINALIZED_CHECKPOINT_MISMATCH"),
        ("chain/status", "finalized_hash", "other", "FINALIZED_CHECKPOINT_MISMATCH"),
    ],
)
def test_fleet_identity_divergence_refuses_the_whole_observation(
    tmp_path, endpoint, field, value, message
):
    spec_path, spec = _spec(tmp_path)
    responses = _responses(spec)
    payload = responses[f"{spec['nodes'][3]['base_url']}/{endpoint}"]
    if endpoint == "status" and field in {"seed", "config_digest"}:
        payload = payload["optimization"]
    payload[field] = value

    with pytest.raises(collector.Refusal, match=message):
        collector.tick(spec_path, fetch_json=Fetcher(responses))
    assert not Path(spec["output_dir"]).exists()


def test_status_and_chain_endpoint_height_must_belong_to_the_same_observation(tmp_path):
    spec_path, spec = _spec(tmp_path)
    responses = _responses(spec)
    responses[f"{spec['nodes'][1]['base_url']}/status"]["chain_height"] = 72

    with pytest.raises(collector.Refusal, match="ENDPOINT_OBSERVATION_MISMATCH"):
        collector.tick(spec_path, fetch_json=Fetcher(responses))
    assert not Path(spec["output_dir"]).exists()


def test_exact_component_commits_are_required_by_the_collector(tmp_path):
    spec_path, spec = _spec(tmp_path)
    responses = _responses(spec)
    responses[f"{spec['nodes'][2]['base_url']}/chain/status"]["component_versions"]["predictor"] = "wrong"

    with pytest.raises(collector.Refusal, match="COMPONENT_VERSION_MISMATCH"):
        collector.tick(spec_path, fetch_json=Fetcher(responses))
    assert not Path(spec["output_dir"]).exists()


def test_crashes_before_and_after_directory_publish_are_idempotent(tmp_path):
    spec_path, spec = _spec(tmp_path)
    responses = _responses(spec)

    def crash_before():
        raise RuntimeError("before publish")

    with pytest.raises(RuntimeError, match="before publish"):
        collector.tick(spec_path, fetch_json=Fetcher(responses), before_publish=crash_before)
    assert not Path(spec["output_dir"]).exists()

    def crash_after():
        raise RuntimeError("after publish")

    with pytest.raises(RuntimeError, match="after publish"):
        collector.tick(spec_path, fetch_json=Fetcher(responses), after_publish=crash_after)
    before = {path.name: path.read_bytes() for path in Path(spec["output_dir"]).iterdir()}
    recovered = collector.tick(spec_path, fetch_json=Fetcher(responses))
    after = {path.name: path.read_bytes() for path in Path(spec["output_dir"]).iterdir()}

    assert recovered["state"] == "COMPLETE"
    assert before == after


def test_conflicting_existing_receipt_set_is_never_replaced(tmp_path):
    spec_path, spec = _spec(tmp_path)
    collector.tick(spec_path, fetch_json=Fetcher(_responses(spec)))
    first = Path(spec["output_dir"]) / "island-0.json"
    document = json.loads(first.read_text())
    document["payload"]["champion_sha256"] = "tampered"
    first.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(collector.Refusal, match="EXISTING_OUTPUT_MISMATCH"):
        collector.tick(spec_path, fetch_json=Fetcher(_responses(spec)))


def test_coherently_resealed_but_false_existing_observation_is_refused(tmp_path):
    spec_path, spec = _spec(tmp_path)
    collector.tick(spec_path, fetch_json=Fetcher(_responses(spec)))
    for path in Path(spec["output_dir"]).glob("*.json"):
        document = json.loads(path.read_text())
        document["payload"]["chain_height"] = 999
        document["payload"]["tip_hash"] = "false-tip"
        document["payload_sha256"] = _digest(document["payload"])
        path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(collector.Refusal, match="EXISTING_OUTPUT_MISMATCH"):
        collector.status(spec_path)


def test_status_is_strictly_read_only_and_performs_no_network_io(tmp_path):
    spec_path, spec = _spec(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    report = collector.status(spec_path)

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert report["state"] == "NOT_COLLECTED"
    assert report["progress"] == {"observed": 0, "required": 4, "published": 0}
    assert before == after
    assert not Path(spec["state_dir"]).exists()
    assert not Path(spec["output_dir"]).exists()


def test_private_spec_inside_repository_is_refused():
    repo_spec = Path(collector.__file__).resolve().parents[1] / "private-terminal-collector.json"
    repo_spec.write_text(json.dumps({"schema": collector.SPEC_SCHEMA}), encoding="utf-8")
    try:
        with pytest.raises(collector.SpecError, match="outside the repository"):
            collector.load_spec(repo_spec)
    finally:
        repo_spec.unlink()


def test_systemd_templates_are_unattended_and_contain_no_private_hosts():
    root = Path(collector.__file__).resolve().parent
    service = (root / "i6n-island-terminal-collect.service").read_text()
    timer = (root / "i6n-island-terminal-collect.timer").read_text()

    assert "Type=oneshot" in service
    assert "--spec %h/.config/i6n/island-terminal-collect.json" in service
    assert "Persistent=true" in timer
    assert "OnUnitInactiveSec=60s" in timer
    for private_fragment in ("gamma", "dragon", "omega", "192.168.", "10.0."):
        assert private_fragment not in service + timer
