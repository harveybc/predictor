"""Tests for snapshotting a live SQLite-backed DOIN chain over its API."""

from __future__ import annotations

from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import pytest

from doin_core.models import Block
from tools.i6n_archive import __main__ as archive_main
from tools.i6n_archive.__main__ import (
    ClosurePending,
    _load_chain_blocks,
    _load_chain_snapshot,
    produce_archive_closure,
)


def _status(height=2, tip="tip-1"):
    return {
        "chain_height": height,
        "tip_hash": tip,
        "genesis_hash": "genesis",
        "network_id": "network",
    }


def test_live_chain_snapshot_is_bound_to_status_before_and_after(monkeypatch):
    calls = []

    def fetch(url):
        calls.append(url)
        if url.endswith("/chain/status"):
            return _status()
        assert "from=0" in url and "to=1" in url
        return {"blocks": [{"hash": "genesis"}, {"hash": "tip-1"}], "has_more": False}

    monkeypatch.setattr(
        Block, "model_validate", staticmethod(lambda value: SimpleNamespace(hash=value["hash"]))
    )

    blocks = _load_chain_blocks({"url": "http://node.invalid"}, fetch_json=fetch)

    assert [block.hash for block in blocks] == ["genesis", "tip-1"]
    assert calls[0] == calls[-1] == "http://node.invalid/chain/status"


def test_live_chain_snapshot_rejects_a_moving_tip_without_projecting(monkeypatch):
    statuses = iter([_status(), _status(tip="other")])

    def fetch(url):
        if url.endswith("/chain/status"):
            return next(statuses)
        return {"blocks": [{"hash": "genesis"}, {"hash": "tip-1"}], "has_more": False}

    monkeypatch.setattr(
        Block, "model_validate", staticmethod(lambda value: SimpleNamespace(hash=value["hash"]))
    )

    with pytest.raises(RuntimeError, match="CHAIN_MOVED_DURING_SNAPSHOT"):
        _load_chain_blocks({"url": "http://node.invalid"}, fetch_json=fetch)


def test_live_chain_snapshot_rejects_moving_finality(monkeypatch):
    statuses = iter([
        _status() | {"finalized_height": 0, "finalized_hash": "genesis"},
        _status() | {"finalized_height": 1, "finalized_hash": "tip-1"},
    ])

    def fetch(url):
        if url.endswith("/chain/status"):
            return next(statuses)
        return {"blocks": [{"hash": "genesis"}, {"hash": "tip-1"}], "has_more": False}

    monkeypatch.setattr(
        Block, "model_validate", staticmethod(lambda value: SimpleNamespace(hash=value["hash"]))
    )
    with pytest.raises(RuntimeError, match="CHAIN_MOVED_DURING_SNAPSHOT"):
        _load_chain_snapshot({"url": "http://node.invalid"}, fetch_json=fetch)


def _sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _closure_fixture(tmp_path, monkeypatch, *, bad_metric=False, unresolved=False,
                     missing_final=False, blocker=False):
    report = {
        "report_sha256": "report-1",
        "tags": {"row_key_sha256": "row-1"},
        "metrics": [{"metric": "relative_error_vs_naive", "value": -0.1,
                     "unit": "ratio", "split": "INNER", "horizon": 4}],
    }
    resolved = SimpleNamespace(
        body_digest="body", manifest_digest="manifest",
        payload_schema="i6n.archive.island_candidate.v1", payload={"horizon_hours": 4},
    )
    reference = {"archive_mode": "external_payload", "locator": "candidate"}
    ref = SimpleNamespace(
        finality=archive_main.FINALIZED, tx_id="tx-1", reference=reference,
        block_hash="block-1", height=1, depth=6,
    )
    monkeypatch.setattr(
        archive_main, "canonical_references",
        lambda _blocks, confirmation_depth: [] if missing_final else [ref],
    )
    monkeypatch.setattr(archive_main.records, "units_for", lambda _schema, _payload: [{"report": report}])

    class Archive:
        def resolve(self, _reference):
            if unresolved:
                raise OSError("missing object")
            return resolved

    class Outbox:
        def items(self):
            return [{
                "item_id": "candidate-1", "body_digest": "body", "manifest_digest": "manifest",
                "reference": reference, "reference_tx_id": "tx-1",
                "states": {name: {"ok": True} for name in (
                    "LOCAL_COMPLETE", "OBJECT_DURABLE", "CHAIN_FINALIZED", "OLAP_READBACK_VERIFIED")},
            }]

        def projections(self):
            return [{"state": "PROJECTED"}]

    class Publisher:
        outbox = Outbox()

        @staticmethod
        def status():
            return {"items": 1, "green": 1, "missing": [], "submitted_unfinalized": 0}

    class Sink:
        @staticmethod
        def read(_query):
            metric = dict(report["metrics"][0])
            metric["report_sha256"] = "wrong" if bad_metric else "report-1"
            return {
                "projections": [{"state": "PROJECTED", "row_key_sha256": "row-1",
                                 "report_sha256": "report-1"}],
                "metrics": [metric],
            }

    terminals = []
    for index in range(4):
        payload = {
            "status": "COMPLETE", "island_id": f"island-{index}", "horizon": 4,
            "scientific_config_sha256": "4" * 64, "network_id": "network",
            "genesis_hash": "genesis", "chain_height": 2, "tip_hash": "tip-1",
            "finalized_height": 1, "finalized_hash": "tip-1",
        }
        envelope = {"schema": "i6n.island.terminal.v1", "payload": payload,
                    "payload_sha256": _sha(payload)}
        path = tmp_path / f"terminal-{index}.json"
        path.write_text(json.dumps(envelope))
        terminals.append(str(path))
    blocker_dir = tmp_path / "quarantine"
    blocker_dir.mkdir()
    if blocker:
        (blocker_dir / "bad.json").write_text("{}")
    config = {
        "confirmation_depth": 1,
        "closure": {
            "receipt": str(tmp_path / "CLOSURE.json"), "terminal_receipts": terminals,
            "expected": {"horizon": 4, "scientific_config_sha256": "4" * 64,
                         "network_id": "network", "genesis_hash": "genesis"},
            "blocker_paths": {"quarantine": [str(blocker_dir)]},
        },
    }
    chain_status = {
        "network_id": "network", "genesis_hash": "genesis", "chain_height": 2,
        "tip_hash": "tip-1", "finalized_height": 1, "finalized_hash": "tip-1",
    }
    return config, Publisher(), Archive(), Sink(), [SimpleNamespace()], chain_status


def test_archive_closure_is_generated_once_after_exact_candidate_readback(tmp_path, monkeypatch):
    config, publisher, archive, sink, blocks, status = _closure_fixture(tmp_path, monkeypatch)
    etl = {"chain_verified": True, "errors": []}
    pump = {"errors": []}

    first = produce_archive_closure(config, publisher, archive, sink, blocks, status, etl, pump)
    second = produce_archive_closure(config, publisher, archive, sink, blocks, status, etl, pump)

    assert first == second
    assert first["schema"] == "i6n.island.archive_closure.v1"
    assert first["payload"]["archive"]["candidate_items"] == 1
    assert first["payload"]["archive"]["candidate_metric_rows"] == 1
    assert Path(config["closure"]["receipt"]).read_text().endswith("\n")


@pytest.mark.parametrize("mutation", ["bad_metric", "unresolved", "missing_final", "blocker"])
def test_archive_closure_refuses_incomplete_or_mismatched_evidence(tmp_path, monkeypatch, mutation):
    kwargs = {mutation: True}
    config, publisher, archive, sink, blocks, status = _closure_fixture(
        tmp_path, monkeypatch, **kwargs)

    with pytest.raises(ClosurePending):
        produce_archive_closure(
            config, publisher, archive, sink, blocks, status,
            {"chain_verified": True, "errors": []}, {"errors": []},
        )
    assert not Path(config["closure"]["receipt"]).exists()


def test_archive_close_systemd_timer_is_autonomous_and_contains_no_private_host():
    root = Path(archive_main.__file__).resolve().parents[1]
    service = (root / "i6n-island-archive-close.service").read_text()
    timer = (root / "i6n-island-archive-close.timer").read_text()

    assert "Type=oneshot" in service
    assert "-m tools.i6n_archive close --config" in service
    assert "SuccessExitStatus=4" in service
    assert "Persistent=true" in timer and "OnUnitInactiveSec=60s" in timer
    for private_fragment in ("gamma", "dragon", "omega", "192.168.", "10.0."):
        assert private_fragment not in service + timer
