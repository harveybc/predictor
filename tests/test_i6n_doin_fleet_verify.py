import json
from pathlib import Path

import pytest

from tools.i6n_doin_fleet_verify import (
    COMPONENTS,
    FleetSpecError,
    load_spec,
    main,
    verify_fleet,
)


DIGEST = "a" * 64
VERSIONS = {
    "doin-node": "1111111",
    "doin-core": "2222222",
    "doin-plugins": "3333333",
    "predictor": "4444444",
}


def _spec() -> dict:
    return {
        "schema": "i6n.doin_fleet_verify.spec.v1",
        "expected": {
            "seed": 6102026,
            "config_digest": DIGEST,
            "component_versions": dict(VERSIONS),
        },
        "nodes": [
            {"node_id": f"node-{index}", "base_url": f"https://<private-node-{index}>"}
            for index in range(4)
        ],
    }


def _responses(spec: dict) -> dict[str, dict]:
    responses: dict[str, dict] = {}
    for node in spec["nodes"]:
        base = node["base_url"]
        responses[f"{base}/chain/status"] = {
            "chain_height": 73,
            "tip_hash": "tip-73",
            "finalized_height": 70,
            "finalized_hash": "finalized-70",
            "network_id": "i6n-production",
            "genesis_hash": "genesis-i6n",
            "component_versions": dict(VERSIONS),
        }
        responses[f"{base}/status"] = {
            "status": "healthy",
            "optimization": {
                "seed": 6102026,
                "config_digest": DIGEST,
            },
            "network_id": "i6n-production",
            "genesis_hash": "genesis-i6n",
        }
    return responses


def _fetcher(responses: dict[str, dict], calls: list[str] | None = None):
    def fetch(url: str, timeout: float) -> dict:
        assert timeout > 0
        if calls is not None:
            calls.append(url)
        value = responses[url]
        if isinstance(value, Exception):
            raise value
        return value

    return fetch


def test_matching_four_node_fleet_is_ready_and_queries_both_endpoints() -> None:
    spec = _spec()
    responses = _responses(spec)
    calls: list[str] = []

    report = verify_fleet(spec, fetch_json=_fetcher(responses, calls))

    assert report["status"] == "READY"
    assert report["ready"] is True


    assert report["fleet"] == {
        "expected_nodes": 4,
        "responding_nodes": 4,
        "network_id": "i6n-production",
        "genesis_hash": "genesis-i6n",
        "chain_height": 73,
        "tip_hash": "tip-73",
        "finalized_checkpoint": {"height": 70, "hash": "finalized-70"},
    }
    assert len(calls) == 8
    assert all(
        url.endswith(("/chain/status", "/status"))
        for url in calls
    )
    assert all("base_url" not in node for node in report["nodes"])


def test_matching_unfinalized_fleet_preserves_the_absence_of_a_checkpoint():
    spec = _spec()
    responses = _responses(spec)
    for url, value in responses.items():
        if url.endswith('/chain/status'):
            value.update(finalized_height=-1, finalized_hash='')
    report = verify_fleet(spec, fetch_json=_fetcher(responses))
    assert report['ready'] is True
    assert report['fleet']['finalized_checkpoint'] == {'height': -1, 'hash': ''}
    responses[spec['nodes'][0]['base_url'] + '/chain/status']['finalized_hash'] = 'invented'
    assert verify_fleet(spec, fetch_json=_fetcher(responses))['ready'] is False


@pytest.mark.parametrize("node_count", [0, 3, 5])
def test_spec_requires_exactly_four_nodes(node_count: int) -> None:
    spec = _spec()
    spec["nodes"] = spec["nodes"][:node_count]
    if node_count == 5:
        spec["nodes"].append(
            {"node_id": "node-4", "base_url": "https://<private-node-4>"}
        )

    with pytest.raises(FleetSpecError, match="exactly four nodes"):
        verify_fleet(spec, fetch_json=lambda _url, _timeout: {})


def test_spec_requires_unique_nodes_and_exact_component_set() -> None:
    duplicate = _spec()
    duplicate["nodes"][1]["node_id"] = duplicate["nodes"][0]["node_id"]
    with pytest.raises(FleetSpecError, match="duplicate node_id"):
        verify_fleet(duplicate, fetch_json=lambda _url, _timeout: {})

    missing_component = _spec()
    del missing_component["expected"]["component_versions"]["predictor"]
    with pytest.raises(FleetSpecError, match="component_versions must contain exactly"):
        verify_fleet(missing_component, fetch_json=lambda _url, _timeout: {})

    assert set(COMPONENTS) == {
        "doin-node",
        "doin-core",
        "doin-plugins",
        "predictor",
    }


def test_unreachable_node_refuses_fleet_without_leaking_its_url() -> None:
    spec = _spec()
    responses = _responses(spec)
    failed_base = spec["nodes"][2]["base_url"]
    responses[f"{failed_base}/chain/status"] = OSError("connection refused")

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["status"] == "REFUSED"
    assert report["fleet"]["responding_nodes"] == 3
    assert any(error["code"] == "NODE_UNREACHABLE" for error in report["errors"])
    assert failed_base not in json.dumps(report)


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        ("seed", "SEED_MISMATCH"),
        ("config_digest", "CONFIG_DIGEST_MISMATCH"),
        ("version", "COMPONENT_VERSION_MISMATCH"),
    ],
)
def test_expected_identity_mismatch_refuses_fleet(
    mutation: str, error_code: str
) -> None:
    spec = _spec()
    responses = _responses(spec)
    base = spec["nodes"][1]["base_url"]
    if mutation == "seed":
        responses[f"{base}/status"]["optimization"]["seed"] += 1
    elif mutation == "config_digest":
        responses[f"{base}/status"]["optimization"]["config_digest"] = "b" * 64
    else:
        responses[f"{base}/chain/status"]["component_versions"]["predictor"] = "different"

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["ready"] is False
    assert any(error["code"] == error_code for error in report["errors"])


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("network_id", "other-network", "NETWORK_MISMATCH"),
        ("genesis_hash", "other-genesis", "GENESIS_MISMATCH"),
        ("chain_height", 72, "CHAIN_HEIGHT_MISMATCH"),
        ("tip_hash", "other-tip", "TIP_HASH_MISMATCH"),
        ("finalized_height", 69, "FINALIZED_CHECKPOINT_MISMATCH"),
        ("finalized_hash", "other-finalized", "FINALIZED_CHECKPOINT_MISMATCH"),
    ],
)
def test_fleet_consensus_mismatch_is_refused(
    field: str, value: object, error_code: str
) -> None:
    spec = _spec()
    responses = _responses(spec)
    base = spec["nodes"][3]["base_url"]
    responses[f"{base}/chain/status"][field] = value
    if field in {"network_id", "genesis_hash"}:
        responses[f"{base}/status"][field] = value

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["ready"] is False
    assert any(error["code"] == error_code for error in report["errors"])


def test_missing_required_runtime_field_is_refused() -> None:
    spec = _spec()
    responses = _responses(spec)
    base = spec["nodes"][0]["base_url"]
    del responses[f"{base}/chain/status"]["finalized_height"]

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["status"] == "REFUSED"
    assert any(error["code"] == "MISSING_FIELD" for error in report["errors"])


def test_network_identity_must_agree_between_a_nodes_two_endpoints() -> None:
    spec = _spec()
    responses = _responses(spec)
    base = spec["nodes"][0]["base_url"]
    responses[f"{base}/status"]["genesis_hash"] = "contradiction"

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["ready"] is False
    assert any(error["code"] == "ENDPOINT_IDENTITY_MISMATCH" for error in report["errors"])


def test_top_level_run_identity_and_chain_only_network_identity_are_supported() -> None:
    spec = _spec()
    responses = _responses(spec)
    for node in spec["nodes"]:
        base = node["base_url"]
        status = responses[f"{base}/status"]
        status.update(status.pop("optimization"))
        del status["network_id"]
        del status["genesis_hash"]
        del responses[f"{base}/chain/status"]["finalized_hash"]

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["ready"] is True
    assert report["fleet"]["finalized_checkpoint"] == {"height": 70}


def test_conflicting_top_level_and_nested_run_identity_is_refused() -> None:
    spec = _spec()
    responses = _responses(spec)
    base = spec["nodes"][0]["base_url"]
    responses[f"{base}/status"]["seed"] = 9
    responses[f"{base}/status"]["config_digest"] = "b" * 64

    report = verify_fleet(spec, fetch_json=_fetcher(responses))

    assert report["ready"] is False
    assert sum(
        error["code"] == "ENDPOINT_IDENTITY_MISMATCH"
        for error in report["errors"]
    ) == 2


def test_load_spec_and_cli_emit_machine_readable_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "private.json"
    path.write_text(json.dumps(_spec()), encoding="utf-8")
    assert load_spec(path)["expected"]["seed"] == 6102026

    responses = _responses(_spec())
    base = _spec()["nodes"][0]["base_url"]
    responses[f"{base}/status"]["optimization"]["seed"] = 7

    exit_code = main(
        ["--spec", str(path), "--timeout", "2.5"],
        fetch_json=_fetcher(responses),
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["status"] == "REFUSED"
    assert report["errors"][0]["code"] == "SEED_MISMATCH"


def test_cli_reports_invalid_private_spec_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "private.json"
    path.write_text("{}", encoding="utf-8")

    exit_code = main(["--spec", str(path)])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert report["status"] == "REFUSED"
    assert report["errors"][0]["code"] == "INVALID_SPEC"
