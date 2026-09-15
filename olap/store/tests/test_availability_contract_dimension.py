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
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))

from predictor_olap_store.query import CANONICALIZATION, Plugin  # noqa: E402


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


@pytest.fixture
def store(tmp_path):
    plugin = Plugin()
    plugin.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    plugin.engine()
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
    assert resolved["contract_resolution"] == "UNRESOLVED"
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
    assert resolved["contract_resolution"] == "RESOLVED"
    assert resolved["use_class"] == "ARCHIVE_RETROSPECTIVE"
    assert resolved["completion_lag_max"] == "UNKNOWN", (
        "the string the producer declared, not a number and not a null")
    assert resolved["timezone_evidence"] == "UNKNOWN"
    assert resolved["canonicalization"] == CANONICALIZATION
    assert resolved["digest_algorithm"] == "sha256"


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
    assert store.resolve_delivery_availability("d-forged")["contract_resolution"] == "UNRESOLVED"


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
    """Three outcomes, kept apart: resolved, unresolved, and never seen."""
    assert store.resolve_delivery_availability("d-nothing") == {
        "delivery_id": "d-nothing", "contract_resolution": "NO_SUCH_DELIVERY"}


def test_the_migration_is_additive_and_repeatable(store, tmp_path):
    """Running the schema step again must not disturb a row that is already there."""
    digest, body = archive_contract()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    again = Plugin()
    again.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    again.engine()
    delivery_row(again, delivery_id="d-after-migration", contract_sha256=digest)
    assert again.resolve_delivery_availability(
        "d-after-migration")["use_class"] == "ARCHIVE_RETROSPECTIVE"
