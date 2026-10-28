from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from optimizer_plugins.i6n_optimizer import _digest
from tests.test_i6n_island_external_archive import _candidate, _config
from tools.i6n_archive.publisher import ArchivePublisher
from tools.i6n_archive.__main__ import _verify_outbox_seals
from tools.i6n_outbox_snapshot import SnapshotRefusal, snapshot


def _terminal(path: Path, candidate: dict, island_id: str) -> Path:
    summary = {
        key: candidate.get(source)
        for key, source in (
            ("fitness", "fitness"), ("hyper_dict", "parameters"),
            ("genotype_canonical_sha256", "genotype_canonical_sha256"),
            ("train_mae", "train_mae"), ("train_naive_mae", "train_naive_mae"),
            ("val_mae", "val_mae"), ("val_naive_mae", "val_naive_mae"),
            ("metrics", "metrics"),
        )
    }
    payload = {
        "status": "COMPLETE", "island_id": island_id, "horizon": 4,
        "seed": 6102026, "plan_sha256": candidate["plan_sha256"],
        "contract_sha256": candidate["contract_sha256"],
        "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
        "evaluated_candidate_count": 1,
        "candidate_ledger_sha256": _digest({candidate["genotype_canonical_sha256"]: summary}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    envelope = {"schema": "i6n.island.terminal.v1", "payload": payload,
                "payload_sha256": hashlib.sha256(canonical).hexdigest()}
    path.write_text(json.dumps(envelope), encoding="utf-8")
    return path


def _source(tmp_path: Path, island_id="gamma-5090", peer="peer-gamma"):
    candidate = _candidate()
    candidate["island_id"] = island_id
    root = tmp_path / island_id
    publisher = ArchivePublisher(
        root, store_id="store", domain_id="i6n-island-h4", producer_peer_id=peer,
    )
    publisher.record_island_candidate(candidate, _config())
    terminal = _terminal(tmp_path / f"{island_id}.terminal.json", candidate, island_id)
    return publisher, terminal


def test_live_wal_outbox_is_snapshotted_and_bound_to_exact_terminal_ledger(tmp_path):
    publisher, terminal = _source(tmp_path)
    destination = tmp_path / "snapshot" / "outbox.sqlite"
    receipt = tmp_path / "snapshot" / "SEAL.json"

    evidence = snapshot(publisher.outbox.path, destination,
                        terminal_receipt=terminal, receipt=receipt)
    retained = snapshot(publisher.outbox.path, destination,
                        terminal_receipt=terminal, receipt=receipt)

    assert evidence["state"] == "CREATED"
    assert retained["state"] == "EXISTING_VERIFIED"
    assert evidence["candidate_ledger_sha256"] == retained["candidate_ledger_sha256"]
    assert json.loads(receipt.read_text())["payload"]["island_id"] == "gamma-5090"


def test_snapshot_rejects_same_count_with_substituted_candidate(tmp_path):
    publisher, terminal = _source(tmp_path)
    with publisher.outbox._tx():
        row = publisher.outbox._conn.execute("SELECT item_id, body FROM items").fetchone()
        body = bytearray(row["body"])
        body[-2] = ord(" ")
        publisher.outbox._conn.execute(
            "UPDATE items SET body = ? WHERE item_id = ?", (bytes(body), row["item_id"]),
        )

    with pytest.raises(SnapshotRefusal, match="OUTBOX_ITEM_DIGEST_MISMATCH"):
        snapshot(publisher.outbox.path, tmp_path / "bad.sqlite",
                 terminal_receipt=terminal, receipt=tmp_path / "bad.json")


def test_snapshot_rejects_a_copied_outbox_as_another_island(tmp_path):
    publisher, _terminal_a = _source(tmp_path, "gamma-5090", "peer-a")
    candidate = _candidate()
    candidate["island_id"] = "dragon-4090"
    terminal_b = _terminal(tmp_path / "dragon.terminal.json", candidate, "dragon-4090")

    with pytest.raises(SnapshotRefusal, match="OUTBOX_TERMINAL_ISLAND_ID_MISMATCH"):
        snapshot(publisher.outbox.path, tmp_path / "copy.sqlite",
                 terminal_receipt=terminal_b, receipt=tmp_path / "copy.json")


def test_existing_snapshot_rejects_a_tampered_receipt(tmp_path):
    publisher, terminal = _source(tmp_path)
    destination = tmp_path / "snapshot.sqlite"
    receipt = tmp_path / "seal.json"
    snapshot(publisher.outbox.path, destination, terminal_receipt=terminal, receipt=receipt)
    receipt.chmod(0o600)
    altered = json.loads(receipt.read_text())
    altered["payload"]["items_sha256"] = "0" * 64
    receipt.write_text(json.dumps(altered))

    with pytest.raises(SnapshotRefusal, match="SNAPSHOT_RECEIPT_DIGEST_MISMATCH"):
        snapshot(publisher.outbox.path, destination,
                 terminal_receipt=terminal, receipt=receipt)


def test_delivery_state_changes_do_not_invalidate_candidate_seal(tmp_path):
    publisher, terminal = _source(tmp_path)
    destination = tmp_path / "snapshot.sqlite"
    receipt = tmp_path / "seal.json"
    first = snapshot(publisher.outbox.path, destination, terminal_receipt=terminal, receipt=receipt)
    from tools.i6n_archive.outbox import Outbox
    retained = Outbox(destination)
    item_id = retained.items()[0]["item_id"]
    retained.set_state(item_id, "OBJECT_DURABLE", {"verified": True})
    retained.close()

    second = snapshot(destination, destination, terminal_receipt=terminal, receipt=receipt)

    assert second["state"] == "EXISTING_VERIFIED"
    assert second["sqlite_sha256"] == first["sqlite_sha256"]
    assert second["items_sha256"] == first["items_sha256"]


def test_four_independent_outboxes_are_bound_to_their_four_terminals(tmp_path):
    publishers, terminals, terminal_digests, seals = {}, {}, {}, []
    for index in range(4):
        island = f"island-{index}"
        live, terminal_path = _source(tmp_path, island, f"peer-{index}")
        snapshot_path = tmp_path / "snapshots" / island / "outbox.sqlite"
        seal_path = tmp_path / "snapshots" / island / "SEAL.json"
        snapshot(live.outbox.path, snapshot_path,
                 terminal_receipt=terminal_path, receipt=seal_path)
        terminal_document = json.loads(terminal_path.read_text())
        terminals[island] = terminal_document["payload"]
        terminal_digests[island] = terminal_document["payload_sha256"]
        publishers[island] = ArchivePublisher.from_outbox(snapshot_path)
        seals.append(str(seal_path))

    verified = _verify_outbox_seals(seals, terminals, terminal_digests, publishers)

    assert set(verified) == set(terminals)
    assert len({item["producer_peer_id"] for item in verified.values()}) == 4
    assert len({item["items_sha256"] for item in verified.values()}) == 4
