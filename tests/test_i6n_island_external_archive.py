"""Hash-only archive contract for I6N local-island candidate events."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "olap/store/src"))

from doin_core.archive import ArchiveRefusal  # noqa: E402
from doin_core.crypto.hashing import compute_merkle_root  # noqa: E402
from doin_core.models import Block, BlockHeader  # noqa: E402
from doin_node.archive.external import ExternalPayloadArchive  # noqa: E402
from doin_node.unified import DomainRole, UnifiedNode, UnifiedNodeConfig  # noqa: E402
from predictor_olap_store.query import Plugin  # noqa: E402
from tools.i6n_archive import records  # noqa: E402
from tools.i6n_archive.etl import ArchiveETL  # noqa: E402
from tools.i6n_archive.publisher import ArchivePublisher  # noqa: E402
from tools.i6n_archive.sink import DirectSink  # noqa: E402


PLAN = "9" * 64
FOLDS = ("2023Q1", "2023Q2", "2023Q3", "2023Q4")


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.tamper = False

    def put_bytes(self, content):
        digest = hashlib.sha256(content).hexdigest()
        self.objects.setdefault(digest, bytes(content))
        return digest

    def get_bytes(self, digest):
        data = self.objects[digest]
        return data + b"not-json" if self.tamper else data


def _config():
    def support(index):
        return {
            "counts": {"train": 10 + index, "inner": 5 + index},
            "naive_sums": {"train": 10.0 + index, "inner": 5.0 + index},
            "support_digest": {
                "train": str(index + 1) * 64,
                "inner": str(index + 5) * 64,
            },
        }

    return {
        "i6n_island_mode": True,
        "i6n_horizon": 4,
        "i6n_plan": {"plan_sha256": PLAN},
        "i6n_folds": {
            fold: {
                "i6n_support": support(index),
                "i6n_origins_sha256": support(index)["support_digest"],
                "i6n_evaluation_context": {"fold": fold, "calendar": "business-v1"},
            }
            for index, fold in enumerate(FOLDS)
        },
        "seed": 6102026,
    }


def _fold_receipt(fold, index, ratio, genotype_sha, horizon=4):
    support = _config()["i6n_folds"][fold]["i6n_support"]
    return {
        "status": "ok",
        "counts": support["counts"],
        "error_sums": {
            split: support["naive_sums"][split] * ratio
            for split in ("train", "inner")
        },
        "naive_sums": support["naive_sums"],
        "support_digest": support["support_digest"],
        "metadata": {
            "evidence_kind": "archive-contract-test",
            "candidate_receipt": {
                "genotype_canonical_sha256": genotype_sha,
                "fold": fold,
                "horizon_hours": horizon,
                "rows": [
                    {
                        "split": split.upper(),
                        "head_offset_hours": horizon,
                        "head_role": "ENDPOINT",
                        "comparison": True,
                        "support_sha256": support["support_digest"][split],
                    }
                    for split in ("train", "inner")
                ],
            },
        },
    }


def _candidate(ratio=0.8):
    parameters = {
        "horizon": 4,
        "multiplier": 6,
        "delta_x": 1,
        "input_transform": "identity",
        "output_mode": "endpoint",
        "delta_y": None,
        "core": {"entrance": "conv", "width": 8, "transformer_blocks": 0,
                 "reducer_filters": [4], "reducer_strides": [1],
                 "adapter": "none", "adapter_width": None},
        "history_hours": 24,
        "samples": 25,
        "final_steps": 25,
        "input_offsets": list(range(-24, 1)),
        "output_offsets": [4],
    }
    effective_sha = records.e_config_sha256(parameters)
    genotype_sha = hashlib.sha256(records.canonical({
        "schema": "i6n-search-v1", "config": parameters,
    })).hexdigest()
    receipts = {
        fold: _fold_receipt(fold, index, ratio, genotype_sha)
        for index, fold in enumerate(FOLDS)
    }
    origins = {
        fold: _config()["i6n_folds"][fold]["i6n_origins_sha256"]
        for fold in FOLDS
    }
    contract_sha = "8" * 64
    checkpoint_sha = "7" * 64
    return {
        "generation": 2,
        "candidate_in_gen": 5,
        "total_eval": 45,
        "fitness": ratio - 1.0,
        "is_champion": True,
        "parameters": parameters,
        "genotype_canonical_sha256": genotype_sha,
        "effective_config_sha256": effective_sha,
        "plan_sha256": PLAN,
        "contract_sha256": contract_sha,
        "checkpoint_identity_sha256": checkpoint_sha,
        "train_mae": ratio,
        "train_naive_mae": 1.0,
        "val_mae": ratio,
        "val_naive_mae": 1.0,
        "metrics": {
            "relative_error_vs_naive": ratio - 1.0,
            "per_fold": {fold: ratio for fold in FOLDS},
            "receipts": receipts,
            "origins_binding": "VERIFIED",
            "origins_sha256": origins,
            "aggregation": "mean_four_folds",
            "genotype_canonical_sha256": genotype_sha,
            "effective_config_sha256": effective_sha,
            "plan_sha256": PLAN,
            "contract_sha256": contract_sha,
            "checkpoint_identity_sha256": checkpoint_sha,
        },
        "training_seed": 6102026,
        "island_id": "gamma-5090",
    }


def test_external_chain_transaction_contains_only_reference_and_retry_is_idempotent(tmp_path):
    publisher = ArchivePublisher(tmp_path / "archive", store_id="cas",
                                 domain_id="i6n-island-h4", producer_peer_id="peer-a")
    recorded = publisher.record_island_candidate(_candidate(), _config())
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    submitted = []

    first = publisher.pump(archive, submitted.append)
    second = publisher.pump(archive, submitted.append)

    assert first == {"stored": 1, "submitted": 1, "errors": []}
    assert second == {"stored": 0, "submitted": 0, "errors": []}
    assert len(submitted) == 1
    reference = submitted[0].payload
    assert set(reference) == {
        "reference_schema", "archive_mode", "body_digest", "manifest_digest",
        "body_size", "payload_schema", "domain_id", "locator",
    }
    encoded_reference = json.dumps(reference, sort_keys=True)
    assert "genotype" not in encoded_reference and "metrics" not in encoded_reference
    resolved = archive.resolve(reference)
    assert resolved.body_digest == recorded["body_digest"]
    assert resolved.payload["effective_config"] == _candidate()["parameters"]
    assert [row["fold"] for row in resolved.payload["per_fold"]] == list(FOLDS)
    for row in resolved.payload["per_fold"]:
        receipt = row["receipt"]
        recomputed = sum(
            receipt["error_sums"][split] / receipt["naive_sums"][split]
            for split in ("train", "inner")
        ) / 2.0
        assert recomputed == pytest.approx(row["ratio_to_naive"], abs=1e-15)
        assert row["receipt_sha256"] == hashlib.sha256(
            records.canonical(receipt)
        ).hexdigest()
        assert row["origins_binding"] == "VERIFIED"
        assert row["origins_sha256"] == receipt["support_digest"]


def test_unified_node_callback_uses_real_outbox_and_spool_not_consensus(tmp_path, monkeypatch):
    role = DomainRole(
        domain_id="i6n-island-h4",
        optimize=True,
        optimization_config=_config(),
    )
    spool = tmp_path / "spool"
    node = UnifiedNode(UnifiedNodeConfig(
        port=0,
        data_dir=str(tmp_path / "node"),
        domains=[role],
        candidate_archive_mode="external_payload",
        candidate_archive_root=str(tmp_path / "archive"),
        candidate_archive_store_id="cas",
        candidate_archive_store_url="http://lake.invalid",
        candidate_archive_token_env="TEST_LAKE_TOKEN",
        archive_reference_spool=str(spool),
    ))
    store = MemoryStore()
    node._candidate_external_archive = ExternalPayloadArchive(
        {"cas": store}, write_store_id="cas"
    )
    consensus_calls = []
    monkeypatch.setattr(node.consensus, "record_transaction", consensus_calls.append)

    result = node._publish_candidate_evaluation(
        role.domain_id, _candidate(), loop=None
    )

    assert result["pump"] == {"stored": 1, "submitted": 1, "errors": []}
    assert consensus_calls == []
    files = list(spool.glob("*.json"))
    assert len(files) == 1
    encoded = files[0].read_text()
    assert "doin.external_payload_ref.v1" in encoded
    assert "genotype" not in encoded and "per_fold" not in encoded
    for publisher in node._candidate_archive_publishers.values():
        publisher.outbox.close()


def test_candidate_or_metric_mutation_changes_external_body_hash(tmp_path):
    publisher = ArchivePublisher(tmp_path / "archive", store_id="cas",
                                 domain_id="i6n-island-h4", producer_peer_id="peer-a")
    original = records.island_candidate_payload("i6n-island-h4", _candidate(), _config())
    mutated = records.island_candidate_payload("i6n-island-h4", _candidate(0.81), _config())

    from doin_core.archive.external import build_external_body

    first = build_external_body(domain_id="i6n-island-h4", producer_peer_id="peer-a",
                                payload_schema=records.ISLAND_CANDIDATE_SCHEMA, payload=original)
    second = build_external_body(domain_id="i6n-island-h4", producer_peer_id="peer-a",
                                 payload_schema=records.ISLAND_CANDIDATE_SCHEMA, payload=mutated)
    assert first.body_digest != second.body_digest

    changed_candidate = _candidate()
    changed_candidate["parameters"] = dict(changed_candidate["parameters"], multiplier=4)
    changed_candidate["parameters"]["history_hours"] = 24
    effective_sha = records.e_config_sha256(changed_candidate["parameters"])
    genotype_sha = hashlib.sha256(records.canonical({
        "schema": "i6n-search-v1", "config": changed_candidate["parameters"],
    })).hexdigest()
    changed_candidate["effective_config_sha256"] = effective_sha
    changed_candidate["genotype_canonical_sha256"] = genotype_sha
    changed_candidate["metrics"]["effective_config_sha256"] = effective_sha
    changed_candidate["metrics"]["genotype_canonical_sha256"] = genotype_sha
    for receipt in changed_candidate["metrics"]["receipts"].values():
        receipt["metadata"]["candidate_receipt"]["genotype_canonical_sha256"] = genotype_sha
    third_payload = records.island_candidate_payload(
        "i6n-island-h4", changed_candidate, _config()
    )
    third = build_external_body(
        domain_id="i6n-island-h4", producer_peer_id="peer-a",
        payload_schema=records.ISLAND_CANDIDATE_SCHEMA, payload=third_payload,
    )
    assert third.body_digest != first.body_digest


def test_missing_or_unreadable_external_payload_is_rejected(tmp_path):
    publisher = ArchivePublisher(tmp_path / "archive", store_id="cas",
                                 domain_id="i6n-island-h4", producer_peer_id="peer-a")
    publisher.record_island_candidate(_candidate(), _config())
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    submitted = []
    publisher.pump(archive, submitted.append)
    reference = submitted[0].payload

    body = store.objects.pop(reference["body_digest"])
    with pytest.raises((ArchiveRefusal, KeyError), match="MISSING|[0-9a-f]{64}"):
        archive.resolve(reference)
    store.objects[reference["body_digest"]] = body
    store.tamper = True
    with pytest.raises(ArchiveRefusal, match="DIGEST_MISMATCH"):
        archive.resolve(reference)


def test_mixed_callback_identity_is_rejected_before_outbox(tmp_path):
    publisher = ArchivePublisher(tmp_path / "archive", store_id="cas",
                                 domain_id="i6n-island-h4", producer_peer_id="peer-a")
    candidate = _candidate()
    candidate["metrics"] = dict(candidate["metrics"], contract_sha256="6" * 64)

    with pytest.raises(records.ArchiveRecordError, match="CONTRACT_SHA256_MISMATCH"):
        publisher.record_island_candidate(candidate, _config())
    assert publisher.outbox.items() == []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda candidate: candidate["metrics"]["per_fold"].__setitem__(FOLDS[0], 0.7),
            "ISLAND_FOLD_RATIO_EVIDENCE",
        ),
        (
            lambda candidate: candidate["metrics"]["receipts"][FOLDS[0]]["error_sums"].__setitem__(
                "train", 999.0
            ),
            "ISLAND_FOLD_RATIO_EVIDENCE",
        ),
        (
            lambda candidate: candidate["metrics"]["receipts"][FOLDS[0]]["counts"].__setitem__(
                "train", 999
            ),
            "ISLAND_RECEIPT_SUPPORT_MISMATCH",
        ),
        (
            lambda candidate: candidate["metrics"]["receipts"][FOLDS[0]]["naive_sums"].__setitem__(
                "train", 999.0
            ),
            "ISLAND_RECEIPT_SUPPORT_MISMATCH",
        ),
        (
            lambda candidate: candidate["metrics"]["receipts"][FOLDS[0]]["metadata"]
            ["candidate_receipt"].__setitem__("genotype_canonical_sha256", "0" * 64),
            "ISLAND_RECEIPT_CANDIDATE_MISMATCH",
        ),
        (
            lambda candidate: candidate["metrics"]["receipts"][FOLDS[0]]["metadata"]
            ["candidate_receipt"]["rows"][0].__setitem__("support_sha256", "0" * 64),
            "ISLAND_RECEIPT_ORIGINS_MISMATCH",
        ),
        (
            lambda candidate: candidate["metrics"]["origins_sha256"][FOLDS[0]].__setitem__(
                "train", "0" * 64
            ),
            "ISLAND_ORIGINS_CONFIG_MISMATCH",
        ),
        (
            lambda candidate: candidate.__setitem__("train_mae", 999.0),
            "ISLAND_AGGREGATE_TRAIN_MAE",
        ),
    ],
)
def test_actual_optimizer_fold_evidence_rejects_inconsistent_or_tampered_input(
    mutate, message,
):
    candidate = _candidate()
    mutate(candidate)

    with pytest.raises(records.ArchiveRecordError, match=message):
        records.island_candidate_payload("i6n-island-h4", candidate, _config())


def test_stored_island_payload_rejects_tampered_receipt_even_if_ratio_is_rewritten():
    payload = records.island_candidate_payload("i6n-island-h4", _candidate(), _config())
    payload["per_fold"][0]["receipt"]["error_sums"]["train"] *= 0.5
    receipt = payload["per_fold"][0]["receipt"]
    rewritten_ratio = sum(
        receipt["error_sums"][split] / receipt["naive_sums"][split]
        for split in ("train", "inner")
    ) / 2.0
    payload["per_fold"][0]["ratio_to_naive"] = rewritten_ratio
    payload["per_fold"][0]["relative_error_vs_naive"] = rewritten_ratio - 1.0

    with pytest.raises(records.ArchiveRecordError, match="ISLAND_RECEIPT_DIGEST"):
        records.validate_island_candidate_payload(payload)


def test_island_candidate_projects_identity_genome_and_each_fold_to_olap():
    payload = records.island_candidate_payload("i6n-island-h4", _candidate(), _config())
    units = records.units_for(records.ISLAND_CANDIDATE_SCHEMA, payload)

    folds = [unit for unit in units if unit["identity"]["fold"] != "ALL_FOLDS"]
    candidate = [unit for unit in units if unit["identity"]["fold"] == "ALL_FOLDS"]
    assert len(folds) == 4 and len(candidate) == 1
    assert {unit["identity"]["fold"] for unit in folds} == set(FOLDS)
    assert all(any(metric["metric"] == "fold_ratio_to_naive" for metric in unit["report"]["metrics"])
               for unit in folds)
    assert all(any(metric["metric"] == "sum_abs_error" and metric["split"] == "TRAIN"
                   for metric in unit["report"]["metrics"])
               for unit in folds)
    assert all("receipt_sha256" in unit["report"]["tags"] for unit in folds)
    assert all("origin_identity_sha256" in unit["report"]["tags"] for unit in folds)
    tags = candidate[0]["report"]["tags"]
    assert json.loads(tags["genotype_json"])["effective"] == _candidate()["parameters"]
    assert tags["search_candidate_sha256"] == payload["search_candidate_sha256"]
    assert tags["archive_event_sha256"] == payload["archive_event_sha256"]


def test_island_candidate_reaches_olap_with_exact_readback(tmp_path):
    publisher = ArchivePublisher(tmp_path / "archive", store_id="cas",
                                 domain_id="i6n-island-h4", producer_peer_id="peer-a")
    publisher.record_island_candidate(_candidate(), _config())
    store = MemoryStore()
    archive = ExternalPayloadArchive({"cas": store}, write_store_id="cas")
    submitted = []
    assert publisher.pump(archive, submitted.append)["submitted"] == 1

    blocks = [Block.genesis()]
    timestamp = datetime(2026, 10, 12, tzinfo=timezone.utc)
    for index in range(1, 8):
        transactions = submitted if index == 1 else []
        previous = blocks[-1]
        header = BlockHeader(
            index=index,
            previous_hash=previous.hash,
            timestamp=timestamp + timedelta(seconds=index),
            merkle_root=compute_merkle_root([tx.id for tx in transactions]),
            generator_id="peer-a",
            weighted_performance_sum=1.0,
            threshold=0.5,
        )
        blocks.append(Block(header=header, transactions=list(transactions)))

    warehouse = Plugin()
    warehouse.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    result = ArchiveETL(
        publisher.outbox, archive, DirectSink(warehouse), confirmation_depth=6
    ).run(blocks)

    assert result["errors"] == []
    assert result["projected_reports"] == 5
    with warehouse.engine().connect() as connection:
        roles = connection.exec_driver_sql(
            "SELECT role, COUNT(*) FROM gov_doin_projection_current GROUP BY role ORDER BY role"
        ).fetchall()
        metrics = connection.exec_driver_sql(
            "SELECT derived_metric, COUNT(*) FROM gov_doin_metric_canonical "
            "GROUP BY derived_metric ORDER BY derived_metric"
        ).fetchall()
    assert [tuple(row) for row in roles] == [("CANDIDATE", 1), ("CANDIDATE_FOLD", 4)]
    assert ("fold_ratio_to_naive", 4) in [tuple(row) for row in metrics]
    assert ("fold_relative_error_vs_naive", 4) in [tuple(row) for row in metrics]
