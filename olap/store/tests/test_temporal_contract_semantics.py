"""Intact bytes are not a valid contract, and the reader may not inherit the writer's checks.

V1 of `docs/handoffs/MUSASHI_U1_U4_REVIEW_AND_V1_V4_2026_09_15.md`. The finding is mine: I made
the reader verify that retained bytes hash to the digest they are filed under and treated that
as sufficient. An `ARCHIVE_RETROSPECTIVE` declaring `0s`, `-1` or `not-a-duration` hashes
perfectly well, and every one of them came back VERIFIED with the lag handed over as if it
meant something.

The rules enforced here are **the producer's own** — the same ones
`financial_data_store.inventory.availability_scope` applies — not a duration convention
invented for the warehouse. Zero lag stays legitimate where the contract permits it
(`LIVE_EQUIVALENT`); this is not a blanket ban on zero.

Every rule runs on SQLite and, when `U2_PG_DATABASE` names a disposable database, on
PostgreSQL. Test cleanup never points at production: the production name is refused outright.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))

from predictor_olap_store.query import (  # noqa: E402
    CANONICALIZATION,
    Plugin,
    TemporalContractError,
    validate_availability_block,
)

PG_DATABASE = os.environ.get("U2_PG_DATABASE")
PROTECTED = {"predictor_olap"}
ENGINES = ["sqlite"] + (["postgres"] if PG_DATABASE else [])


@pytest.fixture(params=ENGINES)
def store(request, tmp_path):
    plugin = Plugin()
    if request.param == "postgres":
        if PG_DATABASE in PROTECTED:
            pytest.fail(f"refusing to run against the production database {PG_DATABASE!r}")
        os.environ["PGDATABASE"] = PG_DATABASE
        plugin.set_params(sqlite_path=None, schema="public")
        plugin.engine()
        with plugin.write_engine().begin() as conn:
            conn.execute(text("DELETE FROM gov_availability_contract"))
            conn.execute(text("DELETE FROM gov_terminal_dataset"))
        return plugin
    plugin.set_params(sqlite_path=str(tmp_path / "cube.sqlite"))
    plugin.engine()
    return plugin


def canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def contract(lag="UNKNOWN", use_class="ARCHIVE_RETROSPECTIVE", label="WINDOW_START",
             evidence="UNKNOWN"):
    return {"resource_id": "r", "available_time_column": "t", "time_unit": "ms",
            "timezone": "utc",
            "availability": {"label": label, "completion_lag_max": lag,
                             "timezone_evidence": evidence, "use_class": use_class}}


def plant(store, body_text: str, delivery_id: str, *, canonicalization=CANONICALIZATION,
          use_class="ARCHIVE_RETROSPECTIVE", lag="UNKNOWN"):
    """Insert a contract row DIRECTLY, bypassing the writer entirely.

    V2 requires exactly this: an invalid stored row constructed independently of the writer, so
    that a missing reader check cannot be hidden by a writer check. Rows like these are also
    what a migration, an older version or a manual repair can leave behind.
    """
    digest = hashlib.sha256(body_text.encode("ascii")).hexdigest()
    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_availability_contract (contract_sha256, canonical_bytes,"
            " digest_algorithm, canonicalization, use_class, completion_lag_max,"
            " availability_label, timezone_evidence, first_seen)"
            " VALUES (:c, :b, 'sha256', :z, :u, :l, 'WINDOW_START', 'UNKNOWN', 'planted')"),
            {"c": digest, "b": body_text, "z": canonicalization, "u": use_class, "l": str(lag)})
        conn.execute(text(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
            " resource_id, role, sha256, bytes, availability_contract_sha256,"
            " verification_state) VALUES ('t', :d, 'l', 'r', 'archive', 'b', 1, :c,"
            " 'VERIFIED_TRANSFER')"), {"d": delivery_id, "c": digest})
    return digest


# --- the positive control, which must keep working ---------------------------------------

def test_an_archive_that_says_unknown_is_still_verified(store):
    body = canonical(contract())
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
            " resource_id, role, sha256, bytes, availability_contract_sha256,"
            " verification_state) VALUES ('t', 'd-ok', 'l', 'r', 'archive', 'b', 1, :c,"
            " 'VERIFIED_TRANSFER')"), {"c": digest})

    answer = store.resolve_delivery_availability("d-ok")
    assert answer["contract_resolution"] == "VERIFIED"
    assert answer["completion_lag_max"] == "UNKNOWN"
    assert answer["use_class"] == "ARCHIVE_RETROSPECTIVE"


def test_zero_lag_remains_legitimate_where_the_contract_permits_it(store):
    """LIVE_EQUIVALENT requires zero. The rule is not a blanket prohibition on zero."""
    body = canonical(contract(lag="0s", use_class="LIVE_EQUIVALENT",
                              evidence="PRODUCER_STATEMENT"))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
            " resource_id, role, sha256, bytes, availability_contract_sha256,"
            " verification_state) VALUES ('t', 'd-live', 'l', 'r', 'input', 'b', 1, :c,"
            " 'VERIFIED_TRANSFER')"), {"c": digest})

    answer = store.resolve_delivery_availability("d-live")
    assert answer["contract_resolution"] == "VERIFIED"
    assert answer["completion_lag_max"] == "0s"


def test_an_ordinary_class_keeps_its_real_duration(store):
    body = canonical(contract(lag="4h", use_class="OFFLINE_DAY_GRANULAR",
                              evidence="PRODUCER_STATEMENT"))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    store.write_availability_contracts([{"contract_sha256": digest, "canonical_bytes": body}])
    with store.write_engine().begin() as conn:
        conn.execute(text(
            "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
            " resource_id, role, sha256, bytes, availability_contract_sha256,"
            " verification_state) VALUES ('t', 'd-4h', 'l', 'r', 'input', 'b', 1, :c,"
            " 'VERIFIED_TRANSFER')"), {"c": digest})
    assert store.resolve_delivery_availability("d-4h")["completion_lag_max"] == "4h"


# --- the three counterexamples, refused at the WRITER --------------------------------------

@pytest.mark.parametrize("lag,fragment", [
    ("0s", "UNKNOWN"),
    ("not-a-duration", "UNKNOWN"),
    (-1, "UNKNOWN"),
    (0, "UNKNOWN"),
])
def test_an_archive_with_a_known_lag_is_refused_on_write(store, lag, fragment):
    body = canonical(contract(lag=lag))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])
    assert fragment in str(refusal.value)


@pytest.mark.parametrize("lag", ["not-a-duration", -1, float("nan"), True, {"hours": 4}])
def test_an_ordinary_class_refuses_a_lag_that_is_not_a_duration(store, lag):
    try:
        body = canonical(contract(lag=lag, use_class="OFFLINE_DAY_GRANULAR",
                                  evidence="PRODUCER_STATEMENT"))
    except (TypeError, ValueError):
        pytest.skip("value cannot be serialised at all")
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError):
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])


def test_a_live_equivalent_with_a_nonzero_lag_is_refused(store):
    body = canonical(contract(lag="4h", use_class="LIVE_EQUIVALENT",
                              evidence="PRODUCER_STATEMENT"))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])
    assert "LIVE_EQUIVALENT" in str(refusal.value)


def test_an_ordinary_class_may_not_borrow_unknown(store):
    body = canonical(contract(lag="UNKNOWN", use_class="OFFLINE_DAY_GRANULAR",
                              evidence="PRODUCER_STATEMENT"))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()
    with pytest.raises(ValueError) as refusal:
        store.write_availability_contracts([{"contract_sha256": digest,
                                             "canonical_bytes": body}])
    assert "ARCHIVE_RETROSPECTIVE" in str(refusal.value)


# --- and refused AGAIN at the reader, on rows the writer never saw -------------------------

@pytest.mark.parametrize("lag", ["0s", "not-a-duration", -1])
def test_a_planted_archive_with_a_known_lag_is_refused_by_the_reader(store, lag):
    """The row is inserted directly. If only the writer checked, this would come back VERIFIED."""
    body = canonical(contract(lag=lag))
    plant(store, body, f"d-planted-{abs(hash(str(lag)))}", lag=lag)

    answer = store.resolve_delivery_availability(f"d-planted-{abs(hash(str(lag)))}")
    assert answer["contract_resolution"] == "UNRESOLVED_INVALID_TEMPORAL_SEMANTICS"
    assert answer["use_class"] is None, "no authoritative semantics from an invalid contract"
    assert answer["completion_lag_max"] is None
    assert "UNKNOWN" in answer["reason"]


def test_a_planted_row_with_duplicate_keys_is_ambiguous_and_refused(store):
    """Most JSON parsers keep one of the two silently; two readers may keep different ones."""
    body = ('{"availability":{"completion_lag_max":"UNKNOWN","label":"WINDOW_START",'
            '"timezone_evidence":"UNKNOWN","use_class":"ARCHIVE_RETROSPECTIVE",'
            '"use_class":"LIVE_EQUIVALENT"}}')
    plant(store, body, "d-dupe")

    answer = store.resolve_delivery_availability("d-dupe")
    assert answer["contract_resolution"] == "UNRESOLVED_MALFORMED_CONTRACT"
    assert "duplicate" in answer["reason"]
    assert answer["use_class"] is None


def test_bytes_that_are_not_their_declared_canonical_form_are_refused_not_normalised(store):
    """A pretty-printed contract hashes fine; its digest is a digest of a different spelling."""
    body = json.dumps(contract(), sort_keys=True, indent=1)      # NOT the declared form
    digest = plant(store, body, "d-noncanonical")

    answer = store.resolve_delivery_availability("d-noncanonical")
    assert answer["contract_resolution"] == "UNRESOLVED_MALFORMED_CONTRACT"
    assert "canonical" in answer["reason"]
    with store.engine().connect() as conn:
        stored = conn.execute(text(
            "SELECT canonical_bytes FROM gov_availability_contract "
            "WHERE contract_sha256 = :c"), {"c": digest}).scalar()
    assert stored == body, "the evidence is preserved exactly, not tidied under its own digest"


def test_a_planted_row_never_crashes_the_reader(store):
    """A typed unresolved outcome, not an exception, whatever nonsense is stored."""
    for index, body in enumerate(["", "null", "[]", '{"availability":42}',
                                  '{"availability":{}}']):
        plant(store, body if body else "x", f"d-junk-{index}")
        answer = store.resolve_delivery_availability(f"d-junk-{index}")
        assert answer["contract_resolution"].startswith("UNRESOLVED")
        assert answer["use_class"] is None


# --- the validator itself, exercised directly ---------------------------------------------

def test_the_validator_is_the_producers_rules_and_not_a_new_convention():
    """Spot-checks against the producer's accepted set, by value rather than by reference."""
    assert validate_availability_block({
        "label": "WINDOW_START", "completion_lag_max": "UNKNOWN",
        "timezone_evidence": "UNKNOWN", "use_class": "ARCHIVE_RETROSPECTIVE",
    })["completion_lag"] is None, "an archive's lag is unobserved, not zero"

    for block, fragment in [
        ({"label": "NOT_A_LABEL", "completion_lag_max": "4h",
          "timezone_evidence": "PRODUCER_STATEMENT", "use_class": "OFFLINE_DAY_GRANULAR"},
         "label"),
        ({"label": "WINDOW_START", "completion_lag_max": "4h",
          "timezone_evidence": "GUESSED", "use_class": "OFFLINE_DAY_GRANULAR"},
         "timezone_evidence"),
        ({"label": "WINDOW_START", "completion_lag_max": "4h",
          "timezone_evidence": "PRODUCER_STATEMENT"}, "exactly"),
    ]:
        with pytest.raises(TemporalContractError) as refusal:
            validate_availability_block(block)
        assert fragment in str(refusal.value)
