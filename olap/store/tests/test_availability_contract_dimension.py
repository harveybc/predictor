"""Resolving a delivery to the contract it references, after the producer is gone.

S2 of `docs/handoffs/MUSASHI_TO_SATOSHI_COUNTERS_ARCHIVE_AND_TERMS_2026_09_15.md`:

    "a fresh reader must retrieve the exact retained contract after the temporary stack/config
     is removed, verify its digest and report use_class and UNKNOWN without inference. […]
     Missing or mismatched references yield an explicit unresolved outcome, never zero lag or
     a guessed use class."

The gap this closes is one I reported myself: `gov_terminal_dataset` stores the contract
DIGEST, so the cube could show that a delivery referenced *some* contract and could not say
what that contract said. Asking the producer is not an answer, because the producer is exactly
what may no longer exist.

These run against SQLite by default and against PostgreSQL when `S2_PG_URL` names a disposable
database. No production database is touched by any test here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))

from predictor_olap_store.query import CANONICALIZATION, Plugin  # noqa: E402
from sqlalchemy import text  # noqa: E402


def canonical(contract: dict) -> str:
    """The producer's canonicalization, reproduced exactly: this is what the digest is over."""
    return json.dumps(contract, sort_keys=True, separators=(",", ":"))


def archive_contract(resource="ethusdt_4h.parquet") -> tuple[str, str]:
    contract = {
        "resource_id": resource,
        "available_time_column": "open_time",
        "time_unit": "ms",
        "timezone": "utc",
        "availability": {
            "label": "WINDOW_START",
            # the whole point: a retrospective archive does not know its completion lag
            "completion_lag_max": "UNKNOWN",
            "timezone_evidence": "UNKNOWN",
            "use_class": "ARCHIVE_RETROSPECTIVE",
        },
    }
    body = canonical(contract)
    return hashlib.sha256(body.encode("ascii")).hexdigest(), body


#: A DISPOSABLE PostgreSQL database name. U2 asks for both engines, because a view, a
#: LEFT JOIN and TEXT columns are where dialects differ and production runs PostgreSQL. The
#: database must already exist and be throwaway; nothing here creates or drops one, and the
#: production name is refused outright.
PG_DATABASE = os.environ.get("U2_PG_DATABASE")
PROTECTED = {"predictor_olap"}

#: A DuckDB file for the same rules. The governed logic is shared, so the same rules must hold
#: on every engine the cube can run on; a rule that passes only on one is not a property.
DUCKDB_PATH = os.environ.get("U2_DUCKDB_PATH")

ENGINES = (["sqlite"] + (["postgres"] if PG_DATABASE else [])
           + (["duckdb"] if DUCKDB_PATH else []))


def _reopen_duckdb(tmp_path):
    from predictor_duckdb_store.provider import PredictorDuckdbStore

    other = PredictorDuckdbStore()
    other.set_params(duckdb_path=str(tmp_path / "cube.duckdb"), schema="main",
                     memory_limit="1GB", threads=2, min_free_bytes=1)
    other.engine()
    return other


@pytest.fixture(params=ENGINES)
def store(request, tmp_path):
    plugin = Plugin()
    if request.param == "postgres":
        if PG_DATABASE in PROTECTED:
            pytest.fail(f"refusing to run against the production database {PG_DATABASE!r}")
        os.environ["PGDATABASE"] = PG_DATABASE
        plugin.set_params(sqlite_path=None, schema="public")
        plugin.engine()
        # each test starts from an empty dimension; the database is disposable by contract
        with plugin.write_engine().begin() as conn:
            conn.execute(text("DELETE FROM gov_availability_contract"))
            conn.execute(text("DELETE FROM gov_terminal_dataset"))

        def reopen():
            other = Plugin()
            other.set_params(sqlite_path=None, schema="public")
            other.engine()
            return other

        plugin.reopen_for_test = reopen
        return plugin
    if request.param == "duckdb":
        from predictor_duckdb_store.provider import PredictorDuckdbStore

        plugin = PredictorDuckdbStore()
        plugin.set_params(duckdb_path=str(tmp_path / "cube.duckdb"), schema="main",
                          memory_limit="1GB", threads=2, min_free_bytes=1)
        plugin.engine()
        plugin.reopen_for_test = lambda: _reopen_duckdb(tmp_path)
        return plugin
    plugin.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    plugin.engine()

    def reopen():
        other = Plugin()
        other.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
        other.engine()
        return other

    plugin.reopen_for_test = reopen
    return plugin


def delivery_row(store, *, delivery_id, contract_sha256, terminal="a" * 64):
    """A terminal dataset row written directly: the join is what is under test here."""
    from sqlalchemy import text

    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
            " resource_id, role, sha256, bytes, availability_contract_sha256,"
            " verification_state) VALUES (:t, :d, 'financial_files', 'ethusdt_4h.parquet',"
            " 'input_data_file', :s, 1024, :c, 'VERIFIED_TRANSFER')"),
            {"t": terminal, "d": delivery_id, "s": "b" * 64, "c": contract_sha256})


def test_an_unresolved_reference_says_so_and_offers_no_lag(store):
    """The state before the contract is retained, and the state that must stay honest."""
    digest, _body = archive_contract()
    delivery_row(store, delivery_id="d-unresolved", contract_sha256=digest)

    resolved = store.resolve_delivery_availability("d-unresolved")
    assert resolved["contract_resolution"] == "UNRESOLVED_REFERENCE_ABSENT"
    assert resolved["availability_contract_sha256"] == digest, "the reference is still shown"
    assert resolved["use_class"] is None, "no guessed class"
    assert resolved["completion_lag_max"] is None, (
        "an unresolved lag must be absent; a zero here is the defect this table exists for")


def test_a_retained_contract_resolves_and_UNKNOWN_survives_as_itself(store):
    digest, body = archive_contract()
    delivery_row(store, delivery_id="d-archive", contract_sha256=digest)
    outcome = store.write_availability_contracts([
        {"contract_sha256": digest, "canonical_bytes": body}])
    assert outcome == {"stored": 1, "already_stored": 0, "contracts": [digest]}

    resolved = store.resolve_delivery_availability("d-archive")
    assert resolved["contract_resolution"] == "VERIFIED"
    assert resolved["use_class"] == "ARCHIVE_RETROSPECTIVE"
    assert resolved["completion_lag_max"] == "UNKNOWN", (
        "the string the producer declared, not a number and not a null")
    assert resolved["timezone_evidence"] == "UNKNOWN"
    assert resolved["stored_canonicalization"] == CANONICALIZATION
    assert resolved["stored_digest_algorithm"] == "sha256"
    assert resolved["verified_sha256"] == digest, (
        "VERIFIED means the READER hashed the bytes, not that a key matched")


def test_the_retained_bytes_are_the_bytes_the_digest_was_taken_over(store):
    """A fresh reader can re-verify rather than trust the row."""
    digest, body = archive_contract()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    delivery_row(store, delivery_id="d-verify", contract_sha256=digest)

    retained = store.resolve_delivery_availability("d-verify")["canonical_bytes"]
    assert hashlib.sha256(retained.encode("ascii")).hexdigest() == digest
    assert json.loads(retained)["availability"]["completion_lag_max"] == "UNKNOWN"


def test_bytes_that_do_not_hash_to_the_declared_identity_are_refused(store):
    """Otherwise anything could be filed under a digest a delivery already trusts."""
    digest, body = archive_contract()
    forged = body.replace('"UNKNOWN"', '"0s"', 1)
    assert forged != body
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([
            {"contract_sha256": digest, "canonical_bytes": forged}])
    assert "mismatch" in str(refusal.value)
    delivery_row(store, delivery_id="d-forged", contract_sha256=digest)
    assert store.resolve_delivery_availability(
        "d-forged")["contract_resolution"] == "UNRESOLVED_REFERENCE_ABSENT"


def test_semantics_are_read_out_of_the_bytes_and_cannot_be_supplied_beside_them(store):
    """A caller cannot label a contract; the label comes from the contract."""
    digest, body = archive_contract()
    store.write_availability_contracts([{
        "contract_sha256": digest, "canonical_bytes": body,
        "use_class": "LIVE_EQUIVALENT", "completion_lag_max": "0s"}])
    delivery_row(store, delivery_id="d-claimed", contract_sha256=digest)

    resolved = store.resolve_delivery_availability("d-claimed")
    assert resolved["use_class"] == "ARCHIVE_RETROSPECTIVE"
    assert resolved["completion_lag_max"] == "UNKNOWN"


def test_a_contract_with_no_availability_block_is_refused(store):
    body = canonical({"resource_id": "x", "time_unit": "ms"})
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([
            {"contract_sha256": digest, "canonical_bytes": body}])
    assert "availability" in str(refusal.value)


def test_retention_is_idempotent_and_the_first_bytes_win(store):
    digest, body = archive_contract()
    first = store.write_availability_contracts([
        {"contract_sha256": digest, "canonical_bytes": body}])
    second = store.write_availability_contracts([
        {"contract_sha256": digest, "canonical_bytes": body}])
    assert first["stored"] == 1 and second["stored"] == 0
    assert second["already_stored"] == 1


def test_two_contracts_for_two_resources_do_not_collide(store):
    a_digest, a_body = archive_contract("ethusdt_4h.parquet")
    b_digest, b_body = archive_contract("btcusdt_4h.parquet")
    assert a_digest != b_digest
    store.write_availability_contracts([
        {"contract_sha256": a_digest, "canonical_bytes": a_body},
        {"contract_sha256": b_digest, "canonical_bytes": b_body}])
    delivery_row(store, delivery_id="d-a", contract_sha256=a_digest)
    delivery_row(store, delivery_id="d-b", contract_sha256=b_digest)
    assert json.loads(store.resolve_delivery_availability("d-a")["canonical_bytes"])[
        "resource_id"] == "ethusdt_4h.parquet"
    assert json.loads(store.resolve_delivery_availability("d-b")["canonical_bytes"])[
        "resource_id"] == "btcusdt_4h.parquet"


def test_a_delivery_nobody_ever_recorded_is_not_an_unresolved_contract(store):
    """Three outcomes, kept apart: verified, unresolved, and never seen."""
    answer = store.resolve_delivery_availability("d-nothing")
    assert answer["contract_resolution"] == "NO_SUCH_DELIVERY"
    assert answer["use_class"] is None and answer["completion_lag_max"] is None
    assert "availability_contract_sha256" not in answer, (
        "there is no reference to report: nothing recorded this delivery at all")


def test_the_migration_is_additive_and_repeatable(store, tmp_path):
    """Running the schema step again must not disturb a row that is already there."""
    digest, body = archive_contract()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    # the same database opened by a NEW provider instance: the schema step must run again
    # over a populated store and leave what is there alone
    again = store.reopen_for_test()
    delivery_row(again, delivery_id="d-after-migration", contract_sha256=digest)
    assert again.resolve_delivery_availability(
        "d-after-migration")["use_class"] == "ARCHIVE_RETROSPECTIVE"


# --- U2: what the reader must refuse, each for its own distinct reason -------------------

def tamper(store, digest, replacement):
    """Change the retained bytes while leaving the key and the cached columns alone."""
    with store.write_engine().begin() as conn:
        conn.execute(text("UPDATE gov_availability_contract SET canonical_bytes = :b "
                          "WHERE contract_sha256 = :c"), {"b": replacement, "c": digest})


def stored_archive(store, delivery_id="d"):
    digest, body = archive_contract()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    delivery_row(store, delivery_id=delivery_id, contract_sha256=digest)
    return digest, body


def test_drifted_bytes_are_caught_by_the_reader_and_not_only_by_a_test(store):
    """Musashi's counterexample, as a rule.

    The key and the cached columns are untouched; only the retained bytes change. Before this,
    the reader returned the view's row and reported RESOLVED with `UNKNOWN` from bytes that
    said `0s`. The reader must hash what it is about to show.
    """
    digest, body = stored_archive(store, "d-drift")
    tamper(store, digest, body.replace('"UNKNOWN"', '"0s"', 1))

    answer = store.resolve_delivery_availability("d-drift")
    assert answer["contract_resolution"] == "UNRESOLVED_DIGEST_MISMATCH"
    assert answer["use_class"] is None
    assert answer["completion_lag_max"] is None, "no availability claim survives a mismatch"
    assert answer["recomputed_sha256"] != digest
    assert answer["stored_completion_lag_max"] == "UNKNOWN", (
        "what the database holds stays visible - it is just not an answer")


def test_the_cached_columns_alone_never_become_the_answer(store):
    """The mirror of the drift: bytes verify, cached columns were edited. Still no claim."""
    digest, _body = stored_archive(store, "d-cached")
    with store.write_engine().begin() as conn:
        conn.execute(text("UPDATE gov_availability_contract SET use_class = 'LIVE_EQUIVALENT',"
                          " completion_lag_max = '0s' WHERE contract_sha256 = :c"),
                     {"c": digest})

    answer = store.resolve_delivery_availability("d-cached")
    assert answer["contract_resolution"] == "UNRESOLVED_STORED_SEMANTICS_DISAGREE"
    assert answer["use_class"] is None and answer["completion_lag_max"] is None
    assert answer["disagreements"]["use_class"] == {
        "stored": "LIVE_EQUIVALENT", "bytes_say": "ARCHIVE_RETROSPECTIVE"}
    assert answer["disagreements"]["completion_lag_max"]["bytes_say"] == "UNKNOWN"


def test_a_canonicalization_this_store_cannot_reproduce_is_refused_on_write(store):
    digest, body = archive_contract()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([{
            "contract_sha256": digest, "canonical_bytes": body,
            "canonicalization": "cbor.deterministic.v9"}])
    assert "unsupported canonicalization" in str(refusal.value)


def test_a_contract_stored_under_an_unverifiable_format_is_not_displayed(store):
    """Defence in depth: a row that got in another way still yields no claim."""
    digest, body = stored_archive(store, "d-format")
    with store.write_engine().begin() as conn:
        conn.execute(text("UPDATE gov_availability_contract SET canonicalization = 'cbor.v9'"
                          " WHERE contract_sha256 = :c"), {"c": digest})

    answer = store.resolve_delivery_availability("d-format")
    assert answer["contract_resolution"] == "UNRESOLVED_UNSUPPORTED_FORMAT"
    assert answer["use_class"] is None
    assert "cbor.v9" in answer["reason"]


def test_a_lag_that_is_not_a_duration_is_refused_rather_than_stringified(store):
    """`str({'hours': 4})` is a plausible-looking string that can never be compared again."""
    contract = {"resource_id": "x", "availability": {
        "label": "WINDOW_START", "completion_lag_max": {"hours": 4},
        "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "OFFLINE_DAY_GRANULAR"}}
    body = canonical(contract)
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])
    assert "completion_lag_max" in str(refusal.value) and "dict" in str(refusal.value)


def test_a_use_class_this_store_cannot_interpret_is_refused_on_write(store):
    contract = {"resource_id": "x", "availability": {
        "label": "WINDOW_START", "completion_lag_max": "4h",
        "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "SOMETHING_NEW"}}
    body = canonical(contract)
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])
    assert "SOMETHING_NEW" in str(refusal.value)


def test_bytes_that_are_not_a_contract_are_malformed_and_not_a_mismatch(store):
    """A distinct outcome: the bytes verify against their digest but say nothing usable."""
    body = "not json at all"
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_availability_contract (contract_sha256, canonical_bytes,"
            " digest_algorithm, canonicalization, use_class, completion_lag_max, first_seen)"
            " VALUES (:c, :b, 'sha256', :z, 'ARCHIVE_RETROSPECTIVE', 'UNKNOWN', 'now')"),
            {"c": digest, "b": body, "z": CANONICALIZATION})
    delivery_row(store, delivery_id="d-malformed", contract_sha256=digest)

    answer = store.resolve_delivery_availability("d-malformed")
    assert answer["contract_resolution"] == "UNRESOLVED_MALFORMED_CONTRACT"
    assert answer["use_class"] is None


def test_the_same_delivery_recorded_against_two_contracts_is_ambiguous(store):
    """Repeated delivery references: which contract is true is not the reader's guess."""
    a_digest, a_body = archive_contract("a.parquet")
    b_digest, b_body = archive_contract("b.parquet")
    store.write_availability_contracts([
        {"contract_sha256": a_digest, "canonical_bytes": a_body},
        {"contract_sha256": b_digest, "canonical_bytes": b_body}])
    delivery_row(store, delivery_id="d-twice", contract_sha256=a_digest, terminal="a" * 64)
    delivery_row(store, delivery_id="d-twice", contract_sha256=b_digest, terminal="c" * 64)

    answer = store.resolve_delivery_availability("d-twice")
    assert answer["contract_resolution"] == "UNRESOLVED_AMBIGUOUS_DELIVERY"
    assert answer["use_class"] is None
    assert sorted(answer["references"]) == sorted([a_digest, b_digest])


def test_the_same_delivery_recorded_twice_for_the_same_contract_still_verifies(store):
    """A repeated reference that AGREES is not an ambiguity, and must not be refused."""
    digest, body = archive_contract()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    delivery_row(store, delivery_id="d-repeat", contract_sha256=digest, terminal="a" * 64)
    delivery_row(store, delivery_id="d-repeat", contract_sha256=digest, terminal="c" * 64)

    answer = store.resolve_delivery_availability("d-repeat")
    assert answer["contract_resolution"] == "VERIFIED"
    assert answer["completion_lag_max"] == "UNKNOWN"


def test_the_view_says_stored_and_never_says_verified(store):
    """A raw SQL join may expose what is stored; it may not imply it checked anything."""
    stored_archive(store, "d-view")
    with store.engine().connect() as conn:
        row = dict(conn.execute(text(
            "SELECT * FROM gov_delivery_availability WHERE delivery_id = 'd-view'")).first()
            ._mapping)
    assert row["contract_reference"] == "STORED"
    assert "contract_resolution" not in row, "the view must not offer a verdict it cannot make"
    for name in ("use_class", "completion_lag_max"):
        assert name not in row, f"{name} would read as verified; it is stored_{name}"
    assert row["stored_use_class"] == "ARCHIVE_RETROSPECTIVE"
