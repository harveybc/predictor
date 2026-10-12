"""Lane D: external_payload archive -> chain -> OLAP, end to end on disposable instances.

Everything here is a throwaway: a temporary file CAS, an in-process chain, SQLite or a DuckDB
file in tmp_path. Nothing certifies a lake, chain or warehouse deployment.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "olap/store/src"), str(ROOT / "olap/duckdb_store/src")]

from doin_core.archive import ArchiveRefusal  # noqa: E402
from doin_core.crypto.hashing import compute_merkle_root  # noqa: E402
from doin_core.models import Block, BlockHeader  # noqa: E402
from doin_node.archive.external import (ExternalPayloadArchive, GrantedByteStore,  # noqa: E402
                                        HttpByteStore)
from data_lake_service.byte_store import HOST_BEARER_GRANT, FileByteStore  # noqa: E402
from predictor_olap_store.query import Plugin  # noqa: E402

from tools.i6n_archive import records  # noqa: E402
from tools.i6n_archive.etl import ArchiveETL  # noqa: E402
from tools.i6n_archive.publisher import ArchivePublisher  # noqa: E402
from tools.i6n_archive.sink import DirectSink, HttpSink  # noqa: E402
from tools.i6n_archive.snapshot import take_snapshot, verify_snapshot  # noqa: E402

TS = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
STORE_ID = "doin_archive_cas"
PLAN = "9" * 64


# --------------------------------------------------------------------------- fixtures
def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


def candidate(index=0, disposition="COMPLETE", fitness=0.95, won=False, h=12):
    genotype = {"schema": "i6n.genotype.v2", "horizon_hours": h, "history_hours": 24, "delta_x_hours": 3,
                "width": 8, "reducer_filters": [4], "entrance": "conv", "index": index}
    config = {"horizon": h, "delta_x": 3, "core": {"width": 8}, "index": index}
    gsha = _sha({"genotype": genotype})
    return {
        "schema": records.CANDIDATE_SCHEMA, "experiment": "i6n-swarm-test", "horizon_hours": h,
        "plan_sha256": PLAN, "plan_metadata": {"class_id": "S1", "stage": "screening"},
        "genotype": genotype, "genotype_canonical_sha256": gsha,
        "search_candidate_sha256": records.search_candidate_sha256(PLAN, h, gsha),
        "effective_config": config, "effective_config_sha256": records.e_config_sha256(config),
        "disposition": disposition, "fitness": fitness, "seed": 6102026,
        "lineage": {"generation": 0, "population_index": index, "operator": "seed", "parents": []},
        "attempts": [{"fold": "2023Q1", "index": 1, "state": "COMPLETED" if fitness is not None else "FAILED",
                      "parent_status_sha256": None, "fencing_token": "tok-1", "reason": None}],
        "costs": {"elapsed_seconds": 12.5, "peak_rss_bytes": 1024, "fit_seconds": 10.0},
        "provenance": {"code_commit": "0" * 40},
    }


def receipt(cand, fold="2023Q1", scale=1.0):
    rows = []
    for split, n in (("TRAIN", 100), ("INNER", 20)):
        for offset, role in ((6, "INTERMEDIATE"), (12, "ENDPOINT")):
            rows.append({
                "schema_version": "i6n.candidate_receipt.v1", "experiment": cand["experiment"],
                "candidate_identity_sha256": hashlib.sha256((cand["search_candidate_sha256"] + fold).encode()).hexdigest(),
                "fold": fold, "horizon_hours": 12, "head_offset_hours": offset, "head_role": role,
                "comparison": role == "ENDPOINT", "role": "MANAGEMENT", "c2_role": "management_origin",
                "population": "S", "support_sha256": hashlib.sha256((fold + split).encode()).hexdigest(),
                "split": split, "metric": "ABS_AND_SQUARED_ERROR_SUMS", "unit": "PRICE",
                "scale": "ORIGINAL_PRICE", "n": n, "sum_abs_error": 0.05 * n * scale,
                "sum_sq_error": 0.0001 * n * scale, "naive_sum_abs_error": 0.06 * n,
                "naive_sum_sq_error": 0.00012 * n, "c2_receipt_sha256": "c" * 64, "status": "SCORED"})
    if fold.startswith("W"):  # weekly refit unit: outer-week VALIDATION rows (never in fitness)
        rows.append(dict(rows[-1], split="VALIDATION", n=7, sum_abs_error=0.4, sum_sq_error=0.001,
                         naive_sum_abs_error=0.35, naive_sum_sq_error=0.0009, comparison=False))
    rows.append(dict(rows[0], head_offset_hours=3, n=0, sum_abs_error=None, sum_sq_error=None,
                     naive_sum_abs_error=None, naive_sum_sq_error=None, c2_receipt_sha256=None,
                     status="NO_SPLIT_SUPPORT"))
    return {"schema_version": "i6n.candidate_receipt.v1", "experiment": cand["experiment"], "fold": fold,
            "candidate_identity_sha256": rows[0]["candidate_identity_sha256"],
            "genotype_canonical_sha256": cand["genotype_canonical_sha256"],
            "effective_config_sha256": cand["effective_config_sha256"], "horizon_hours": 12,
            "rows": rows, "seed": 6102026, "attempt": {"index": 1, "parent_status_sha256": None,
                                                      "fencing_token": "tok-1"},
            "cost": {"elapsed_seconds": 11.0}}


class LocalChain:
    """In-process chain double: deduplicates by tx id, mints linked blocks, can fork."""

    def __init__(self):
        self.blocks = [Block.genesis()]
        self.pending = []
        self.submitted = []

    def submit(self, tx):
        self.submitted.append(tx.id)
        known = {t.id for b in self.blocks for t in b.transactions} | {t.id for t in self.pending}
        if tx.id not in known:
            self.pending.append(tx)

    def mint(self, empty_after=6, gen="peer-1", include=True):
        txs = self.pending if include else []
        self.blocks.append(_block(self.blocks[-1], txs, gen))
        if include:
            self.pending = []
        for _ in range(empty_after):
            self.blocks.append(_block(self.blocks[-1], [], gen))
        return self.blocks


def _block(prev, txs, gen, salt=0):
    header = BlockHeader(index=prev.header.index + 1, previous_hash=prev.hash,
                         timestamp=TS + timedelta(seconds=prev.header.index + 1 + salt),
                         merkle_root=compute_merkle_root([t.id for t in txs]), generator_id=gen,
                         weighted_performance_sum=1.0, threshold=0.5)
    return Block(header=header, transactions=list(txs))


def _store(tmp_path, name="cas"):
    backend = FileByteStore()
    backend.set_params(root=str(tmp_path / name))
    return backend


def _warehouse(tmp_path, name="cube.sqlite"):
    plugin = Plugin()
    plugin.set_params(sqlite_path=str(tmp_path / name))
    return plugin


def _rig(tmp_path, mode="external_payload", tag=""):
    lake = _store(tmp_path, "cas" + tag)
    stores = {STORE_ID: GrantedByteStore(lake, HOST_BEARER_GRANT)}
    pub = ArchivePublisher(tmp_path / ("archive" + tag), store_id=STORE_ID, domain_id="i6n-dom",
                           producer_peer_id="peer-1", mode=mode, clock=lambda: TS)
    archive = ExternalPayloadArchive(stores, write_store_id=STORE_ID)
    wh = _warehouse(tmp_path, f"cube{tag}.sqlite")
    etl = ArchiveETL(pub.outbox, ExternalPayloadArchive(stores), DirectSink(wh), confirmation_depth=6)
    return pub, archive, etl, wh, lake


def _canonical(wh):
    with wh.engine().connect() as c:
        rows = c.exec_driver_sql(
            "SELECT experiment, candidate_identity_sha256, fold, horizon_hours, head_offset_hours, role,"
            " population, split, receipt_metric, receipt_unit, schema_version, derived_metric,"
            " derived_unit, value, report_sha256 FROM gov_doin_metric_canonical ORDER BY 1,2,3,4,5,6,7,8,12"
        ).fetchall()
    return [tuple(r) for r in rows]


def _count(wh, relation):
    with wh.engine().connect() as c:
        return c.exec_driver_sql(f"SELECT count(*) FROM {relation}").scalar()


def _record_two(pub):
    good = candidate(0)
    failed = candidate(1, disposition="FAILED", fitness=None)
    a = pub.record_candidate(good, [receipt(good)])
    b = pub.record_candidate(failed, [])
    return a, b


# --------------------------------------------------------------------------- tests
def test_full_flow_states_are_separate_and_green_only_at_the_end(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    chain = LocalChain()
    a, _b = _record_two(pub)
    assert a["states"]["LOCAL_COMPLETE"] and not a["states"]["OBJECT_DURABLE"]
    assert pub.status()["green"] == 0
    pub.pump(archive, chain.submit)
    item = pub.outbox.item(a["item_id"])
    assert item["states"]["OBJECT_DURABLE"] and not item["states"]["CHAIN_FINALIZED"]
    chain.mint(empty_after=2)
    shallow = etl.run(chain.blocks)
    assert shallow["projected_reports"] == 0
    assert pub.status()["green"] == 0                    # CONFIRMING is not finalized
    chain.mint(empty_after=6, include=False)
    summary = etl.run(chain.blocks)
    assert summary["errors"] == []
    status = pub.status()
    assert status["green"] == 2 and status["missing"] == []
    # All candidates are kept: the failed one has its disposition and no invented score.
    with wh.engine().connect() as c:
        tags = [json.loads(r[0]) for r in c.exec_driver_sql(
            "SELECT tags_json FROM gov_report WHERE phase = 'I6N_SWARM_DOIN_CANDIDATE'").fetchall()]
        fitness = c.exec_driver_sql(
            "SELECT count(*) FROM gov_metric m JOIN gov_report r USING (report_sha256)"
            " WHERE r.phase = 'I6N_SWARM_DOIN_CANDIDATE' AND m.metric = 'fitness'").scalar()
    assert sorted(t["disposition"] for t in tags) == ["COMPLETE", "FAILED"]
    assert fitness == 1
    assert all("genotype_json" in t and "effective_config_json" in t for t in tags)


def test_full_and_external_payload_project_identical_analytic_rows(tmp_path):
    rows = {}
    for mode in ("external_payload", "full_payload"):
        pub, archive, etl, wh, _ = _rig(tmp_path, mode=mode, tag=mode)
        chain = LocalChain()
        _record_two(pub)
        pub.pump(archive, chain.submit)
        chain.mint()
        assert etl.run(chain.blocks)["errors"] == []
        rows[mode] = _canonical(wh)
        assert pub.status()["green"] == 2
    assert rows["external_payload"] and rows["external_payload"] == rows["full_payload"]
    # mae is derived from sums/counts: 0.05*n/n
    maes = [r[13] for r in rows["full_payload"] if r[11] == "mae" and r[4] == 12 and r[7] == "TRAIN"]
    assert maes == [pytest.approx(0.05)]


def test_failed_put_never_commits(tmp_path):
    pub, _archive, etl, wh, lake = _rig(tmp_path)

    class Broken:
        def describe(self):
            return {}

        def put_bytes(self, content):
            raise OSError("lake unavailable")

        def get_bytes(self, digest):
            raise OSError("lake unavailable")

    chain = LocalChain()
    _record_two(pub)
    result = pub.pump(ExternalPayloadArchive({STORE_ID: Broken()}, write_store_id=STORE_ID), chain.submit)
    assert result["submitted"] == 0 and result["errors"]
    assert chain.submitted == [] and chain.pending == []
    chain.mint()
    etl.run(chain.blocks)
    assert _count(wh, "gov_metric") == 0
    assert all(not i["states"]["OBJECT_DURABLE"] for i in pub.outbox.items())


def test_crash_between_every_step_recovers_without_duplicates(tmp_path):
    pub, archive, etl, wh, lake = _rig(tmp_path)
    chain = LocalChain()
    a, _ = _record_two(pub)

    # 1. crash after the PUT, before OBJECT_DURABLE is recorded
    def crash_after_put(*args, **kwargs):
        raise SystemExit("power loss")

    with pytest.raises(SystemExit):
        pub.pump(archive, chain.submit, _after_store=crash_after_put)
    assert not pub.outbox.item(a["item_id"])["states"]["OBJECT_DURABLE"]
    # 2. crash after the reference reached the node, before the outbox recorded it
    def crash_after_submit(tx):
        chain.submit(tx)
        raise SystemExit("power loss")

    with pytest.raises(SystemExit):
        pub.pump(archive, crash_after_submit)
    reopened = ArchivePublisher(tmp_path / "archive", store_id=STORE_ID, domain_id="i6n-dom",
                                producer_peer_id="peer-1", mode="external_payload", clock=lambda: TS)
    reopened.pump(archive, chain.submit)          # same deterministic tx id: the node deduplicates
    assert len(chain.pending) == 2
    chain.mint()
    # 3. crash after projection, before readback/ACK
    etl.crash_before_ack = True
    with pytest.raises(SystemExit):
        etl.run(chain.blocks)
    etl.crash_before_ack = False
    assert reopened.status()["per_state"]["OLAP_READBACK_VERIFIED"] == 0   # projected is not ACKed
    fresh = ArchiveETL(reopened.outbox, ExternalPayloadArchive({STORE_ID: GrantedByteStore(lake, HOST_BEARER_GRANT)}),
                       DirectSink(wh), confirmation_depth=6)
    assert fresh.run(chain.blocks)["errors"] == []
    assert reopened.status()["green"] == 2

    def declared_metrics():
        with wh.engine().connect() as c:
            return c.exec_driver_sql("SELECT sum(n_metrics) FROM gov_report").scalar()

    recovered = _count(wh, "gov_metric")
    assert recovered == declared_metrics()           # each report's metrics stored exactly once
    # 4. a second ETL pass is a no-op
    fresh.run(chain.blocks)
    assert _count(wh, "gov_metric") == recovered


def test_wrong_bytes_digest_and_locator_are_rejected_before_projection(tmp_path):
    pub, archive, etl, wh, lake = _rig(tmp_path)
    chain = LocalChain()
    a, _ = _record_two(pub)
    pub.pump(archive, chain.submit)
    chain.mint()
    # wrong bytes: the stored body no longer hashes to its name
    path = lake.path_of(a["body_digest"])
    path.chmod(0o644)
    original = path.read_bytes()
    path.write_bytes(original.replace(b"i6n-swarm-test", b"i6n-swarm-TEST"))
    result = etl.run(chain.blocks)
    assert any("HASH_MISMATCH" in e["error"] or "DIGEST" in e["error"] for e in result["errors"])
    assert pub.outbox.item(a["item_id"])["states"]["OLAP_READBACK_VERIFIED"] is None
    path.write_bytes(original)
    # wrong locator: a store the reader is not configured for
    stranger = ArchiveETL(pub.outbox, ExternalPayloadArchive({"other": GrantedByteStore(lake, HOST_BEARER_GRANT)}),
                          DirectSink(_warehouse(tmp_path, "other.sqlite")), confirmation_depth=6)
    assert any("LOCATOR_STORE_UNKNOWN" in e["error"] for e in stranger.run(chain.blocks)["errors"])
    # wrong digest: a forged reference transaction breaks the chain check; nothing projects
    forged_chain = list(chain.blocks)
    tx = forged_chain[1].transactions[0]
    forged = tx.model_copy(update={"payload": dict(tx.payload, manifest_digest="e" * 64)})
    forged_chain[1] = Block(header=forged_chain[1].header, transactions=[forged] + forged_chain[1].transactions[1:],
                            hash=forged_chain[1].hash)
    fresh_wh = _warehouse(tmp_path, "forged.sqlite")
    forged_etl = ArchiveETL(pub.outbox, ExternalPayloadArchive({STORE_ID: GrantedByteStore(lake, HOST_BEARER_GRANT)}),
                            DirectSink(fresh_wh), confirmation_depth=6)
    out = forged_etl.run(forged_chain)
    assert out["chain_verified"] is False and out["errors"]
    assert _count(fresh_wh, "gov_metric") == 0


def test_duplicate_publication_does_not_add_metrics(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    chain = LocalChain()
    a, _ = _record_two(pub)
    again = pub.record_candidate(candidate(0), [receipt(candidate(0))])
    assert again["item_id"] == a["item_id"] and len(pub.outbox.items()) == 2
    pub.pump(archive, chain.submit)
    chain.mint()
    etl.run(chain.blocks)
    before = _count(wh, "gov_metric"), len(_canonical(wh))
    # the same reference transaction replayed into a later block (a peer re-broadcast)
    tx = chain.blocks[1].transactions[0]
    chain.blocks.append(_block(chain.blocks[-1], [tx], "peer-2"))
    for _ in range(6):
        chain.blocks.append(_block(chain.blocks[-1], [], "peer-2"))
    etl.run(chain.blocks)
    assert (_count(wh, "gov_metric"), len(_canonical(wh))) == before


def test_reorg_revokes_canonical_projection_and_preserves_history(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    chain = LocalChain()
    a, _ = _record_two(pub)
    pub.pump(archive, chain.submit)
    chain.mint()
    etl.run(chain.blocks)
    assert pub.status()["green"] == 2
    projected = _canonical(wh)
    old_block = chain.blocks[1].hash
    # A heavier competing branch from genesis without the reference.
    fork = [chain.blocks[0]]
    for i in range(12):
        fork.append(_block(fork[-1], [], "peer-9", salt=1000))
    summary = etl.run(fork)
    assert summary["revoked_blocks"] == [old_block]
    assert _canonical(wh) == []
    assert _count(wh, "gov_metric") > 0                       # history kept
    assert pub.status()["green"] == 0
    item = pub.outbox.item(a["item_id"])
    assert item["states"]["CHAIN_FINALIZED"] is None and item["states"]["OBJECT_DURABLE"]
    assert any(h["event"] == "REVOKED_REORG" for h in pub.outbox.history(a["item_id"]))
    # The publisher resubmits the same reference; once finalized on the new branch it projects again.
    chain.blocks = fork
    pub.pump(archive, chain.submit)
    chain.mint()
    assert etl.run(chain.blocks)["errors"] == []
    again = _canonical(wh)
    assert [r[:14] for r in again] == [r[:14] for r in projected]
    assert pub.status()["green"] == 2


def test_snapshot_restore_reproduces_the_canonical_projection(tmp_path):
    pub, archive, etl, wh, lake = _rig(tmp_path)
    chain = LocalChain()
    _record_two(pub)
    pub.pump(archive, chain.submit)
    chain.mint()
    etl.run(chain.blocks)
    rows = _canonical(wh)
    wh._engine.dispose()
    manifest = take_snapshot({"cas": tmp_path / "cas", "outbox": pub.outbox.path,
                              "warehouse": tmp_path / "cube.sqlite"}, tmp_path / "snap")
    assert manifest["files"] and all(len(f["sha256"]) == 64 for f in manifest["files"])
    restored = tmp_path / "restored"
    verify_snapshot(tmp_path / "snap", restore_to=restored)
    lake2 = FileByteStore()
    lake2.set_params(root=str(restored / "cas"))
    wh2 = Plugin()
    wh2.set_params(sqlite_path=str(restored / "warehouse" / "cube.sqlite"))
    assert _canonical(wh2) == rows
    pub2 = ArchivePublisher.from_outbox(restored / "outbox" / pub.outbox.path.name)
    assert pub2.status()["green"] == 2
    etl2 = ArchiveETL(pub2.outbox, ExternalPayloadArchive({STORE_ID: GrantedByteStore(lake2, HOST_BEARER_GRANT)}),
                      DirectSink(wh2), confirmation_depth=6)
    assert etl2.run(chain.blocks)["errors"] == []
    assert _canonical(wh2) == rows
    # a corrupted snapshot file is refused
    victim = next(p for p in (tmp_path / "snap").rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    victim.chmod(0o644)
    victim.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SNAPSHOT_DIGEST_MISMATCH"):
        verify_snapshot(tmp_path / "snap", restore_to=tmp_path / "restored2")


def test_new_peer_resolves_the_body_through_the_lake_http_routes(tmp_path):
    from data_lake_service.web import create_app as lake_app
    from data_warehouse_service.config import load as wh_load
    from data_warehouse_service.web import create_app as wh_app

    lake = _store(tmp_path)
    app = lake_app({"service_token": "t", "store_id": STORE_ID}, lake,
                   {"capabilities": list(lake.capabilities())})
    client = app.test_client()

    def transport(method, url, data=None, headers=None):
        path = url.split("://", 1)[1].split("/", 1)[1]
        response = client.open("/" + path, method=method, data=data, headers=headers)
        return response.status_code, response.get_data()

    http_store = HttpByteStore("http://lake.invalid", token_provider=lambda: "t", transport=transport)
    pub = ArchivePublisher(tmp_path / "archive", store_id=STORE_ID, domain_id="i6n-dom",
                           producer_peer_id="peer-1", mode="external_payload", clock=lambda: TS)
    chain = LocalChain()
    _record_two(pub)
    pub.pump(ExternalPayloadArchive({STORE_ID: http_store}, write_store_id=STORE_ID), chain.submit)
    chain.mint()
    # The new peer has only the chain and read access to the lake; no outbox, no local bytes.
    peer = ExternalPayloadArchive({STORE_ID: HttpByteStore("http://lake.invalid", token_provider=lambda: "t",
                                                           transport=transport)})
    from doin_node.archive.external import canonical_references

    refs = canonical_references(chain.blocks, confirmation_depth=6, finalized_only=True)
    bodies = [peer.resolve(r.reference) for r in refs]
    assert len(bodies) == 2 and all(b.payload["schema"] == records.ROUND_SCHEMA for b in bodies)
    # ... and projects them through the warehouse host routes with exact readback.
    wh = _warehouse(tmp_path, "peer.sqlite")
    wh_client = wh_app(wh_load(None, {"service_token": "w", "store_id": "cube"}), wh,
                       {"capabilities": ["describe", "write_doin_projection", "read_doin_projection",
                                         "revoke_doin_projection"]}).test_client()

    def wh_transport(method, url, data=None, headers=None):
        path = url.split("://", 1)[1].split("/", 1)[1]
        response = wh_client.open("/" + path, method=method, data=data, headers=headers)
        return response.status_code, response.get_data()

    from tools.i6n_archive.outbox import Outbox

    peer_etl = ArchiveETL(Outbox(tmp_path / "peer-inbox.sqlite"), peer,
                          HttpSink("http://wh.invalid", token_provider=lambda: "w", transport=wh_transport),
                          confirmation_depth=6)
    assert peer_etl.run(chain.blocks)["errors"] == []
    assert len(_canonical(wh)) > 0


def test_invented_scores_and_test_rows_are_refused(tmp_path):
    pub, *_ = _rig(tmp_path)
    with pytest.raises(records.ArchiveRecordError, match="fitness"):
        pub.record_candidate(candidate(2, disposition="FAILED", fitness=0.3), [])
    good = candidate(3)
    leaked = receipt(good)
    leaked["rows"][0]["split"] = "TEST"
    with pytest.raises(records.ArchiveRecordError, match="TEST"):
        pub.record_candidate(good, [leaked])
    wrong_cfg = candidate(4)
    wrong_cfg["effective_config_sha256"] = "0" * 64
    with pytest.raises(records.ArchiveRecordError, match="effective_config_sha256"):
        pub.record_candidate(wrong_cfg, [])
    assert pub.outbox.items() == []


def test_duckdb_disposable_instance_projects_the_same_rows(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("duckdb_engine")
    from predictor_duckdb_store.provider import PredictorDuckdbStore

    pub, archive, etl, wh, _ = _rig(tmp_path)
    chain = LocalChain()
    _record_two(pub)
    pub.pump(archive, chain.submit)
    chain.mint()
    etl.run(chain.blocks)
    duck = PredictorDuckdbStore()
    duck.set_params(duckdb_path=str(tmp_path / "cube.duckdb"), schema="main", memory_limit="512MB",
                    threads=1, min_free_bytes=1)
    from tools.i6n_archive.outbox import Outbox

    duck_etl = ArchiveETL(Outbox(tmp_path / "duck-inbox.sqlite"), etl.archive, DirectSink(duck), confirmation_depth=6)
    assert duck_etl.run(chain.blocks)["errors"] == []
    with duck.engine().connect() as c:
        duck_rows = [tuple(r) for r in c.exec_driver_sql(
            "SELECT experiment, candidate_identity_sha256, fold, horizon_hours, head_offset_hours, role,"
            " population, split, receipt_metric, receipt_unit, schema_version, derived_metric,"
            " derived_unit, value, report_sha256 FROM main.gov_doin_metric_canonical ORDER BY 1,2,3,4,5,6,7,8,12"
        ).fetchall()]
    duck._engine.dispose()
    assert duck_rows == _canonical(wh)


def test_cli_status_is_read_only_and_spool_is_idempotent(tmp_path, capsys):
    from tools.i6n_archive.__main__ import _spool_submitter, main

    config = tmp_path / "private.json"
    config.write_text(json.dumps({"root": str(tmp_path / "archive")}))
    assert main(["status", "--config", str(config)]) == 0
    assert json.loads(capsys.readouterr().out) == {"state": "NO_OUTBOX"}
    pub, archive, _etl, _wh, _ = _rig(tmp_path)
    _record_two(pub)
    spool = tmp_path / "spool"
    submit = _spool_submitter(spool)
    pub.pump(archive, submit)
    files = sorted(spool.iterdir())
    assert len(files) == 2
    for path in files:
        from doin_core.models import Transaction

        tx = Transaction.model_validate_json(path.read_text())
        assert tx.id == path.stem == tx.compute_id()
        submit(tx)                                     # replay: same bytes, no error, no new file
    assert sorted(spool.iterdir()) == files
    assert main(["status", "--config", str(config)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["items"] == 2 and status["green"] == 0 and status["submitted_unfinalized"] == 2


def test_snapshot_cli_take_and_verify(tmp_path, capsys):
    from tools.i6n_archive.snapshot import main

    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.bin").write_bytes(b"abc")
    assert main(["take", "--source", f"cas={tmp_path / 'src'}", "--dest", str(tmp_path / "snap")]) == 0
    took = json.loads(capsys.readouterr().out)
    assert main(["verify", "--dest", str(tmp_path / "snap"), "--restore-to", str(tmp_path / "back")]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert took["files_sha256"] == checked["files_sha256"] and (tmp_path / "back/cas/a.bin").read_bytes() == b"abc"
    manifest = (tmp_path / "snap" / "MANIFEST.json").read_text()
    assert str(tmp_path) not in manifest               # no absolute source paths in the manifest


def test_interface_a2_roles_weekly_validation_rows_and_support_digest(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    cand = candidate(5)
    weekly = receipt(cand, fold="W2024-01-07")
    pub.record_candidate(cand, [receipt(cand), weekly])
    chain = LocalChain()
    pub.pump(archive, chain.submit)
    chain.mint()
    assert etl.run(chain.blocks)["errors"] == []
    splits = {(r[2], r[7]) for r in _canonical(wh)}
    assert ("W2024-01-07", "VALIDATION") in splits and ("2023Q1", "VALIDATION") not in splits
    with wh.engine().connect() as c:
        tags = [json.loads(r[0]) for r in c.exec_driver_sql(
            "SELECT tags_json FROM gov_report WHERE phase = 'I6N_SWARM_DOIN'").fetchall()]
    assert {t["role"] for t in tags} == {"MANAGEMENT"} and {t["c2_role"] for t in tags} == {"management_origin"}
    assert all(len(t["support_sha256"]) == 64 for t in tags if t["status"] == "SCORED")
    old_role = receipt(candidate(6))
    old_role["rows"][0]["role"] = "management_origin"
    with pytest.raises(records.ArchiveRecordError, match="MANAGEMENT or ENTRY"):
        pub.record_candidate(candidate(6), [old_role])


# --------------------------------------------------------------------------- lane C review (F1, F2, R-1, R-2)
def _flow(pub, archive, etl, chain):
    pub.pump(archive, chain.submit)
    chain.mint(empty_after=6)
    return etl.run(chain.blocks)


def _decision(seq, outcomes, h=12, supersedes=None):
    return {"schema": records.DECISION_SCHEMA, "experiment": "i6n-swarm-test", "plan_sha256": PLAN,
            "horizon_hours": h, "sequence": seq, "supersedes_decision_sha256": supersedes,
            "basis": {"rule": "best complete fitness + minimal control", "folds": ["2023Q1"]},
            "outcomes": [{"search_candidate_sha256": k, "outcome": v} for k, v in sorted(outcomes.items())]}


def test_f1_search_outcome_is_an_append_only_decision_event(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    chain = LocalChain()
    c0, c1 = candidate(0), candidate(1)
    pub.record_candidate(c0, [receipt(c0)])
    pub.record_candidate(c1, [receipt(c1)])
    assert _flow(pub, archive, etl, chain)["errors"] == []
    first = pub.record_decision(_decision(1, {c0["search_candidate_sha256"]: "WINNER",
                                              c1["search_candidate_sha256"]: "CONTROL"}))
    assert _flow(pub, archive, etl, chain)["errors"] == []
    second = pub.record_decision(_decision(2, {c0["search_candidate_sha256"]: "NOT_SELECTED",
                                               c1["search_candidate_sha256"]: "WINNER"},
                                           supersedes=first["body_digest"]))
    assert second["item_id"] != first["item_id"]
    assert _flow(pub, archive, etl, chain)["errors"] == []
    status = pub.status()
    assert status["items"] == 4 and status["green"] == 4
    with wh.engine().connect() as c:
        current = dict(c.exec_driver_sql(
            "SELECT search_candidate_sha256, outcome FROM gov_doin_candidate_decision_current").fetchall())
        won = dict(c.exec_driver_sql(
            "SELECT search_candidate_sha256, won FROM gov_doin_candidate_decision_current").fetchall())
        history = c.exec_driver_sql(
            "SELECT count(*) FROM gov_doin_metric_canonical WHERE receipt_metric = 'SEARCH_OUTCOME'"
            " AND derived_metric = 'won'").scalar()
    assert current == {c0["search_candidate_sha256"]: "NOT_SELECTED", c1["search_candidate_sha256"]: "WINNER"}
    assert won == {c0["search_candidate_sha256"]: 0.0, c1["search_candidate_sha256"]: 1.0}
    assert history == 4                                   # both decisions kept, append-only
    with pytest.raises(records.ArchiveRecordError, match="WON_IS_A_DECISION_EVENT"):
        pub.record_candidate(dict(candidate(2), won=True), [])
    with pytest.raises(records.ArchiveRecordError, match="DECISION_TWO_WINNERS"):
        pub.record_decision(_decision(3, {c0["search_candidate_sha256"]: "WINNER",
                                          c1["search_candidate_sha256"]: "WINNER"}))


def test_f2_attempts_of_one_fold_project_one_canonical_measurement(tmp_path):
    rows = {}
    for tag, n_attempts in (("one", 1), ("two", 2)):
        pub, archive, etl, wh, _ = _rig(tmp_path, tag=tag)
        chain = LocalChain()
        c0 = candidate(0)
        r1 = receipt(c0)
        receipts = [r1]
        if n_attempts == 2:
            r2 = json.loads(json.dumps(r1))
            r2["attempt"] = {"index": 2, "parent_status_sha256": "a" * 64, "fencing_token": "tok-2"}
            receipts.append(r2)
        pub.record_candidate(c0, receipts)
        assert _flow(pub, archive, etl, chain)["errors"] == []
        assert pub.status()["green"] == 1
        rows[tag] = [r for r in _canonical(wh) if r[5] != "CANDIDATE"]
        with wh.engine().connect() as c:
            tags = json.loads(c.exec_driver_sql(
                "SELECT tags_json FROM gov_report WHERE phase = 'I6N_SWARM_DOIN_CANDIDATE'").scalar())
        history = json.loads(tags["attempt_history_json"])
        assert [(h["fold"], h["index"], h["canonical"]) for h in history] == \
            [("2023Q1", i, i == 1) for i in range(1, n_attempts + 1)]
    assert rows["one"] == rows["two"]                    # never double-counted
    c0 = candidate(0)
    bad = json.loads(json.dumps(receipt(c0)))
    bad["attempt"] = {"index": 2, "parent_status_sha256": None, "fencing_token": "t"}
    with pytest.raises(records.ArchiveRecordError, match="ATTEMPT_LINEAGE"):
        records.round_payload(c0, [receipt(c0), bad])
    many = [json.loads(json.dumps(receipt(c0))) for _ in range(4)]
    for i, r in enumerate(many, 1):
        r["attempt"] = {"index": i, "parent_status_sha256": None if i == 1 else "b" * 64, "fencing_token": "t"}
    with pytest.raises(records.ArchiveRecordError, match="ATTEMPTS_EXCEEDED"):
        records.round_payload(c0, many)


def test_f1_row_key_conflicts_are_refused_at_record_time(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    c0 = candidate(0)
    pub.record_candidate(c0, [receipt(c0)])
    with pytest.raises(records.ArchiveRecordError, match="ROW_KEY_CONFLICT"):
        pub.record_candidate(c0, [receipt(c0, scale=2.0)])
    assert len(pub.outbox.items()) == 1


def test_r1_validation_rows_are_refused_on_inner_folds():
    c0 = candidate(0)
    r = receipt(c0, fold="2023Q1")
    r["rows"].append(dict(r["rows"][3], split="VALIDATION", n=7))
    with pytest.raises(records.ArchiveRecordError, match="VALIDATION_ON_INNER_FOLD"):
        records.round_payload(c0, [r])
    for fold in ("2023Q1_DCA", "W2024-12-29"):
        records.round_payload(c0, [receipt(c0, fold=fold)])


class _CountingSink:
    def __init__(self, inner):
        self.inner, self.writes, self.reads = inner, 0, 0

    def write(self, document):
        self.writes += 1
        return self.inner.write(document)

    def read(self, document):
        self.reads += 1
        return self.inner.read(document)

    def revoke(self, document):
        return self.inner.revoke(document)


class _CountingArchive:
    def __init__(self, inner):
        self.inner, self.resolves = inner, 0

    def resolve(self, reference):
        self.resolves += 1
        return self.inner.resolve(reference)


def test_r2_etl_work_is_proportional_to_new_items(tmp_path):
    pub, archive, etl, wh, lake = _rig(tmp_path)
    sink = _CountingSink(DirectSink(wh))
    reader = _CountingArchive(ExternalPayloadArchive({STORE_ID: GrantedByteStore(lake, HOST_BEARER_GRANT)}))
    counted = ArchiveETL(pub.outbox, reader, sink, confirmation_depth=6)
    chain = LocalChain()
    for i in range(3):
        c = candidate(i)
        pub.record_candidate(c, [receipt(c)])
    _flow(pub, archive, counted, chain)
    per_candidate = sink.writes // 3
    assert reader.resolves == 3
    sink.writes = sink.reads = reader.resolves = 0
    assert counted.run(chain.blocks)["errors"] == []      # nothing new: no work
    assert (sink.writes, sink.reads, reader.resolves) == (0, 0, 0)
    for i in range(3, 5):
        c = candidate(i)
        pub.record_candidate(c, [receipt(c)])
    assert _flow(pub, archive, counted, chain)["errors"] == []
    assert reader.resolves == 2 and sink.writes == 2 * per_candidate
    assert pub.status()["green"] == 5


def test_f3_probe_takes_the_token_only_from_the_environment(tmp_path, monkeypatch):
    from data_lake_service.web import create_app as lake_app
    from tools.i6n_archive import probe

    lake = _store(tmp_path)
    client = lake_app({"service_token": "env-token", "store_id": STORE_ID}, lake,
                      {"capabilities": list(lake.capabilities())}).test_client()

    def transport(method, url, data=None, headers=None):
        response = client.open("/" + url.split("://", 1)[1].split("/", 1)[1], method=method, data=data,
                               headers=headers)
        return response.status_code, response.get_data()

    monkeypatch.delenv("I6N_TEST_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="TOKEN_ENV_MISSING"):
        probe.probe_lake("http://lake.invalid", "I6N_TEST_TOKEN", transport)
    monkeypatch.setenv("I6N_TEST_TOKEN", "env-token")
    result = probe.probe_lake("http://lake.invalid", "I6N_TEST_TOKEN", transport)
    assert result["ok"] is True and "env-token" not in json.dumps(result)
    import inspect

    source = inspect.getsource(probe)
    assert "read_text" not in source and "open(" not in source     # no token file is ever read


# --------------------------------------------------------------------------- off-chain readback (deployment order 2026-10-12)
def test_offchain_readback_keeps_chain_finalized_false_until_a_chain_finalizes(tmp_path):
    pub, archive, etl, wh, _ = _rig(tmp_path)
    chain = LocalChain()
    a, _ = _record_two(pub)
    pub.pump(archive, chain.submit)
    summary = etl.run_offchain()
    assert summary["errors"] == [] and summary["readback_verified"] == 2
    item = pub.outbox.item(a["item_id"])
    assert item["states"]["OLAP_READBACK_VERIFIED"]["chain_finalized"] is False
    assert item["states"]["OLAP_READBACK_VERIFIED"]["finality_state"] == "OFFCHAIN_VERIFIED"
    assert item["states"]["CHAIN_FINALIZED"] is None and pub.status()["green"] == 0
    assert pub.status()["readback_verified_offchain"] == 2
    rows_offchain = _canonical(wh)
    assert rows_offchain
    again = etl.run_offchain()                                  # idempotent, no work
    assert again["readback_verified"] == 0 and again["skipped_done"] == 2
    chain.mint()                                                # a chain finalizes later
    assert etl.run(chain.blocks)["errors"] == []
    assert pub.status()["green"] == 2
    assert [r[:14] for r in _canonical(wh)] == [r[:14] for r in rows_offchain]   # no double count
    with wh.engine().connect() as c:
        states = {r[0] for r in c.exec_driver_sql("SELECT DISTINCT finality_state FROM gov_doin_metric_canonical")}
    assert states == {"FINALIZED"}
    # a later reorg must never revoke the no-block projections' history, only real blocks
    fork = [chain.blocks[0]] + [_block(chain.blocks[0], [], "peer-9", salt=500)]
    for _ in range(10):
        fork.append(_block(fork[-1], [], "peer-9", salt=500))
    assert dps_revoked_blocks(etl.run(fork)) == [chain.blocks[1].hash]


def dps_revoked_blocks(summary):
    return summary["revoked_blocks"]


def test_offchain_failure_is_visible_and_retryable_without_refit(tmp_path):
    pub, archive, etl, wh, lake = _rig(tmp_path)
    chain = LocalChain()
    a, _ = _record_two(pub)
    pub.pump(archive, chain.submit)
    path = lake.path_of(a["body_digest"])
    path.chmod(0o644)
    original = path.read_bytes()
    path.write_bytes(b"x" + original)
    failed = etl.run_offchain()
    assert any(e["item_id"] == a["item_id"] for e in failed["errors"])
    assert pub.outbox.item(a["item_id"])["states"]["OLAP_READBACK_VERIFIED"] is None
    path.write_bytes(original)
    retried = etl.run_offchain()
    assert retried["errors"] == [] and retried["readback_verified"] == 1


def test_lifecycle_separates_queued_claimed_updating_completed_archived_readback(tmp_path):
    from tools.i6n_archive.lifecycle import compute

    pub, archive, etl, wh, _ = _rig(tmp_path)
    c0, c1, c2 = candidate(0), candidate(1), candidate(2)
    pub.record_candidate(c0, [receipt(c0)])
    pub.record_candidate(c1, [receipt(c1)])
    chain = LocalChain()
    pub.pump(archive, chain.submit)
    etl.run_offchain()                       # c0, c1 readback-verified off-chain
    pub.record_candidate(c2, [receipt(c2)])  # c2 only LOCAL_COMPLETE
    swarm = {"campaign_id": "cid", "levels": {
        "swarm": {"state": "RUNNING", "eta": {"kind": "projected", "seconds": 10.0}},   # no basis_n -> unknown
        "horizon": [{"node_id": "search:h12", "state": "RUNNING",
                     "eta": {"kind": "projected", "seconds": 100.0, "low": 80.0, "high": 120.0, "basis_n": 3}}],
        "fold": [{"fit_id": "f1", "state": "QUEUED"},
                 {"fit_id": "f2", "state": "RUNNING", "lease": {"token": "t"}, "eta": None},
                 {"fit_id": "f3", "state": "RUNNING", "lease": {"token": "t"},
                  "eta": {"kind": "measured", "seconds": 30.0, "source": "evaluator_epochs"}},
                 {"fit_id": "f4", "state": "COMPLETE"}],
        "candidate": [{"search_candidate_sha256": c["search_candidate_sha256"], "state": "COMPLETE"}
                      for c in (c0, c1, c2)] + [{"search_candidate_sha256": "f" * 64, "state": "RUNNING"}],
        "publication": {"state": "QUEUED"}}}
    out = compute(swarm, pub.outbox, now="t0")
    assert out["fold_lifecycle_counts"] == {"QUEUED": 1, "CLAIMED": 1, "UPDATING": 1, "COMPLETED": 1,
                                            "FAILED": 0, "INTERRUPTED": 0}
    assert out["archive_lifecycle_counts"] == {"NOT_RECORDED": 1, "LOCAL_COMPLETE": 1, "ARCHIVED": 0,
                                               "READBACK_VERIFIED": 2}
    assert out["chain_finalized_candidates"] == 0
    assert out["swarm"]["eta"]["kind"] == "unknown"
    assert out["nodes"][0]["eta"] == {"kind": "projected", "seconds": 100.0, "low": 80.0, "high": 120.0,
                                      "basis_n": 3, "source": None}
    assert {c["readback_finality"] for c in out["candidates"] if c["archive"] == "READBACK_VERIFIED"} == \
        {"OFFCHAIN_VERIFIED"}


def test_snapshot_refuses_a_live_duckdb_wal_and_copies_it_when_the_owner_is_stopped(tmp_path):
    src = tmp_path / "w"
    src.mkdir()
    (src / "cube.duckdb").write_bytes(b"db")
    (src / "cube.duckdb.wal").write_bytes(b"wal")
    with pytest.raises(ValueError, match="DUCKDB_WAL_PRESENT"):
        take_snapshot({"warehouse": src / "cube.duckdb"}, tmp_path / "s1")
    manifest = take_snapshot({"warehouse": src / "cube.duckdb"}, tmp_path / "s2", owner_stopped=True)
    assert sorted(f["path"] for f in manifest["files"]) == ["warehouse/cube.duckdb", "warehouse/cube.duckdb.wal"]
    verify_snapshot(tmp_path / "s2", restore_to=tmp_path / "r")
    assert (tmp_path / "r/warehouse/cube.duckdb.wal").read_bytes() == b"wal"
