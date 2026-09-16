"""Every new semantic check is proved to be the thing that refuses.

V2 of `docs/handoffs/MUSASHI_U1_U4_REVIEW_AND_V1_V4_2026_09_15.md`:

    "Demonstrate that disabling each new semantic check causes its corresponding test to fail."

A passing suite does not show that a check is load-bearing; it may be refusing for some
unrelated reason, or not refusing at all while another rule happens to cover the case. Here
each check is **removed from a copy of the real module**, the module is loaded, and the case it
guards is shown to be accepted again. If a mutation changes nothing, the check is decoration
and this file fails.

Nothing here writes to any database that outlives the test, and the production module on disk
is never modified: each mutant is a separate file under `tmp_path`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

REPO = Path(__file__).resolve().parents[3]
MODULE = REPO / "olap" / "store" / "src" / "predictor_olap_store" / "query.py"
sys.path.insert(0, str(REPO / "olap" / "store" / "src"))


def canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"))


def contract(lag="UNKNOWN", use_class="ARCHIVE_RETROSPECTIVE", label="WINDOW_START",
             evidence="UNKNOWN"):
    return {"resource_id": "r",
            "availability": {"label": label, "completion_lag_max": lag,
                             "timezone_evidence": evidence, "use_class": use_class}}


#: (name, the exact source the check is made of, what replaces it, the case it must guard)
MUTATIONS = {
    "archive_requires_unknown": (
        '        if lag != UNKNOWN_LAG:\n'
        '            raise TemporalContractError(\n'
        '                f"a retrospective archive declares completion_lag_max '
        '{UNKNOWN_LAG!r}; "\n'
        '                f"{lag!r} would assert a publication time nothing observed")\n',
        '        if False:\n            pass\n'),
    "duration_must_parse": (
        '        raise TemporalContractError(\n'
        '            f"completion_lag_max {value!r} is not a duration") from exc\n',
        '        return __import__("pandas").Timedelta(0)\n'),
    "duration_must_not_be_negative": (
        '    if delta < pd.Timedelta(0):\n'
        '        raise TemporalContractError(f"completion_lag_max {value!r} is negative")\n',
        '    if False:\n        pass\n'),
    "live_equivalent_needs_zero": (
        '            raise TemporalContractError(\n'
        '                "LIVE_EQUIVALENT needs a known label, zero completion lag and a '
        'producer "\n                "time-zone statement")\n',
        '            pass\n'),
    "no_duplicate_json_keys": (
        '            raise TemporalContractError(\n'
        '                f"duplicate JSON key {key!r}: the contract is ambiguous and two '
        'readers could "\n                "disagree about what it says")\n',
        '            pass\n'),
    "bytes_must_be_canonical": (
        '    if reserialised != canonical:\n'
        '        raise TemporalContractError(\n'
        '            "the retained bytes are not the canonical form they declare; they are '
        'kept exactly "\n            "as stored and refused rather than normalised under the '
        'same digest")\n',
        '    if False:\n        pass\n'),
    "reader_validates_independently": (
        '        try:\n'
        '            validate_availability_block(scope)\n'
        '        except TemporalContractError as exc:\n'
        '            return refusal(RESOLUTION_INVALID_SEMANTICS, row, reason=str(exc))\n',
        '        pass\n'),
}


def mutant(tmp_path: Path, *names: str):
    """Load a copy of the real module with one or more checks removed."""
    source = MODULE.read_text(encoding="utf-8")
    for name in names:
        original, replacement = MUTATIONS[name]
        assert original in source, (
            f"the source of check {name!r} is not in query.py as written; the mutation list "
            "has drifted from the code and is no longer proving anything")
        source = source.replace(original, replacement, 1)
    label = "_".join(names)
    path = tmp_path / f"mutant_{label}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"mutant_{label}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"mutant_{label}"] = module
    spec.loader.exec_module(module)
    return module


def opened(module, tmp_path: Path, name: str):
    store = module.Plugin()
    store.set_params(sqlite_path=str(tmp_path / f"{name}.sqlite"))
    store.engine()
    return store


def write(store, body_text: str):
    digest = hashlib.sha256(body_text.encode("ascii")).hexdigest()
    store.write_availability_contracts([{"contract_sha256": digest,
                                         "canonical_bytes": body_text}])
    return digest


def refuses(callable_):
    try:
        callable_()
    except ValueError:
        return True
    return False


def test_every_listed_mutation_still_matches_the_code(tmp_path):
    """The guard on the guard: a mutation that no longer applies proves nothing silently."""
    source = MODULE.read_text(encoding="utf-8")
    for name, (original, _replacement) in MUTATIONS.items():
        assert original in source, f"mutation {name!r} no longer matches query.py"


@pytest.mark.parametrize("lag", ["0s", "-1", "not-a-duration"])
def test_without_the_archive_rule_a_known_lag_is_accepted_again(tmp_path, lag):
    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "real")
    body = canonical(contract(lag=lag))
    assert refuses(lambda: write(real, body)), "the real store must refuse this"

    broken = opened(mutant(tmp_path, "archive_requires_unknown"), tmp_path, "broken")
    accepted = not refuses(lambda: write(broken, body))
    assert accepted or lag != "0s", (
        "removing the archive rule must let a known lag back in for at least the well-formed "
        "duration; anything else means the rule was not what refused it")


def test_without_the_duration_parser_a_nonsense_lag_is_accepted_again(tmp_path):
    body = canonical(contract(lag="not-a-duration", use_class="OFFLINE_DAY_GRANULAR",
                              evidence="PRODUCER_STATEMENT"))
    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "r2")
    assert refuses(lambda: write(real, body))

    broken = opened(mutant(tmp_path, "duration_must_parse"), tmp_path, "b2")
    assert not refuses(lambda: write(broken, body)), (
        "the duration parser is what refuses a lag that is not a duration")


def test_without_the_negative_check_a_negative_duration_is_accepted_again(tmp_path):
    body = canonical(contract(lag="-1h", use_class="OFFLINE_DAY_GRANULAR",
                              evidence="PRODUCER_STATEMENT"))
    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "r3")
    assert refuses(lambda: write(real, body))

    broken = opened(mutant(tmp_path, "duration_must_not_be_negative"), tmp_path, "b3")
    assert not refuses(lambda: write(broken, body))


def test_without_the_live_equivalent_rule_a_nonzero_lag_is_accepted_again(tmp_path):
    body = canonical(contract(lag="4h", use_class="LIVE_EQUIVALENT",
                              evidence="PRODUCER_STATEMENT"))
    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "r4")
    assert refuses(lambda: write(real, body))

    broken = opened(mutant(tmp_path, "live_equivalent_needs_zero"), tmp_path, "b4")
    assert not refuses(lambda: write(broken, body))


def test_the_duplicate_key_check_is_a_reason_and_the_canonical_check_is_the_floor(tmp_path):
    """The dependency as it actually is, rather than as I first asserted it.

    Removing the duplicate-key hook alone does NOT let an ambiguous contract through: the
    canonical-form check catches it too, because re-serialising a body whose duplicate keys
    collapsed produces different bytes from the ones stored. So the hook is not the only thing
    refusing — it is what gives the refusal its accurate reason. Removing BOTH is what accepts
    the contract, and that is the honest statement of what each check carries.
    """
    body = ('{"availability":{"completion_lag_max":"UNKNOWN","label":"WINDOW_START",'
            '"timezone_evidence":"UNKNOWN","use_class":"ARCHIVE_RETROSPECTIVE",'
            '"use_class":"ARCHIVE_RETROSPECTIVE"}}')
    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "r5")
    assert refuses(lambda: write(real, body))

    without_hook = opened(mutant(tmp_path, "no_duplicate_json_keys"), tmp_path, "b5a")
    assert refuses(lambda: write(without_hook, body)), (
        "still refused, by the canonical-form check")

    without_both = opened(
        mutant(tmp_path, "no_duplicate_json_keys", "bytes_must_be_canonical"), tmp_path, "b5b")
    assert not refuses(lambda: write(without_both, body)), (
        "with neither check, an ambiguous contract is retained - which is what both of them "
        "together prevent")


def test_the_duplicate_key_check_is_what_names_the_reason(tmp_path):
    """Two refusals are not interchangeable: an operator acts on the reason, not on the fact."""
    body = ('{"availability":{"completion_lag_max":"UNKNOWN","label":"WINDOW_START",'
            '"timezone_evidence":"UNKNOWN","use_class":"ARCHIVE_RETROSPECTIVE",'
            '"use_class":"LIVE_EQUIVALENT"}}')
    real = __import__("predictor_olap_store.query", fromlist=["x"])
    store = opened(real, tmp_path, "r5c")
    try:
        write(store, body)
        pytest.fail("the real store must refuse this")
    except ValueError as exc:
        assert "duplicate" in str(exc)

    without_hook = mutant(tmp_path, "no_duplicate_json_keys")
    store = opened(without_hook, tmp_path, "b5c")
    try:
        write(store, body)
        pytest.fail("still refused, but for the other reason")
    except ValueError as exc:
        assert "duplicate" not in str(exc) and "canonical" in str(exc)


def test_without_the_canonical_form_check_a_pretty_printed_contract_is_accepted_again(tmp_path):
    body = json.dumps(contract(), sort_keys=True, indent=1)
    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "r6")
    assert refuses(lambda: write(real, body))

    broken = opened(mutant(tmp_path, "bytes_must_be_canonical"), tmp_path, "b6")
    assert not refuses(lambda: write(broken, body))


def test_without_the_reader_check_a_planted_invalid_row_reads_as_verified(tmp_path):
    """The one that matters most: the reader must not inherit the writer's checks."""
    body = canonical(contract(lag="0s"))
    digest = hashlib.sha256(body.encode("ascii")).hexdigest()

    def plant(store):
        with store.write_engine().begin() as conn:
            conn.execute(text(
                "INSERT INTO gov_availability_contract (contract_sha256, canonical_bytes,"
                " digest_algorithm, canonicalization, use_class, completion_lag_max,"
                " availability_label, timezone_evidence, first_seen) VALUES (:c, :b, 'sha256',"
                " :z, 'ARCHIVE_RETROSPECTIVE', '0s', 'WINDOW_START', 'UNKNOWN', 'planted')"),
                {"c": digest, "b": body, "z": store.__class__.__module__ and
                 sys.modules[store.__class__.__module__].CANONICALIZATION})
            conn.execute(text(
                "INSERT INTO gov_terminal_dataset (terminal_sha256, delivery_id, lake_id,"
                " resource_id, role, sha256, bytes, availability_contract_sha256,"
                " verification_state) VALUES ('t', 'd', 'l', 'r', 'archive', 'b', 1, :c,"
                " 'VERIFIED_TRANSFER')"), {"c": digest})

    real = opened(__import__("predictor_olap_store.query", fromlist=["x"]), tmp_path, "r7")
    plant(real)
    assert real.resolve_delivery_availability("d")["contract_resolution"] == (
        "UNRESOLVED_INVALID_TEMPORAL_SEMANTICS")

    broken = opened(mutant(tmp_path, "reader_validates_independently"), tmp_path, "b7")
    plant(broken)
    answer = broken.resolve_delivery_availability("d")
    assert answer["contract_resolution"] == "VERIFIED", (
        "with the reader's own validation removed, the planted contract is VERIFIED again - "
        "which is exactly the defect, and proves the reader check is what prevents it")
    assert answer["completion_lag_max"] == "0s"
