"""FRED point-in-time provenance record after the reviewer's ALFRED recovery (@3508e149).

The reviewer's artifacts are cited by path and digest; nothing is re-downloaded
here and no network call is made. These are record/contract properties only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.i6dmo import provenance
from tools.i6n_contracts import feature_availability as fa

FRED = ("fred.rates.dgs3mo.diff_5d", "fred.rates.dgs7.logret_5d")
PROV = Path("docs/i6n/direct_multi_output_static/market_time/provenance")


def test_false_no_unauthenticated_vintage_path_claim_is_gone():
    text = json.dumps(json.loads(provenance.PRODUCERS_PATH.read_text()))
    for claim in ("no ALFRED/vintage adapter exists in any sibling repository",
                  "a governed ALFRED fetch is NOT possible",
                  "a governed FRED_API_KEY EnvironmentFile for the adapter"):
        assert claim not in text, claim
    assert "public ALFRED Download Data form" in text
    assert "no credential" in text or "unauthenticated" in text


def test_clock_state_is_typed_partial_and_carries_the_measurement():
    rows = provenance.producer_status()
    for name in FRED:
        clock = rows[name]["clock"]
        assert clock["state"] == provenance.CLOCK_PARTIAL
        assert clock["proven"] is False
        assert clock["declared_rule"] == "(D+2) 00:00 UTC, as-of last value, NaN if older than 168 h"
        m = clock["measurement"]
        assert m["observations_compared"] == 1587
        assert m["observations_not_covered_by_declared_clock"] == 339
        assert m["fraction_not_covered"] == pytest.approx(0.2136, abs=5e-5)
        assert "Friday" in m["pattern"] and "holiday" in m["pattern"].lower()
        assert m["interpretation"] == provenance.CLOCK_REFUTATION_NOTE
        assert "look-ahead" in m["not_a_claim_of"]
        assert m["ruling_owner"] in ("Musashi", "owner", "Musashi/owner")
        assert clock["covered_if"].startswith("H.15")
        assert clock["still_missing"].startswith("per-release H.15 timestamps")
        assert "404" in clock["still_missing"]
        ev = clock["evidence"]
        assert ev["authentication"] == "NONE"
        for row in ev["artifacts"]:
            assert row["path"].startswith("docs/i6n/direct_multi_output_static/market_time/")
            assert len(row["sha256"]) == 64
        paths = {row["path"] for row in ev["artifacts"]}
        assert any("PROV01_RESULT.json" in p for p in paths)
        assert any("ALFRED_PROVENANCE.json" in p for p in paths)
        assert any("H15_RELEASE_CLOCK.json" in p for p in paths)
        assert any(name.split(".")[2].split("_")[0].upper() in p for p in paths)


def test_cited_digests_match_the_reviewer_artifacts_on_disk():
    rows = provenance.producer_status()
    seen = 0
    for name in FRED:
        for row in rows[name]["clock"]["evidence"]["artifacts"]:
            path = Path(row["path"])
            assert path.is_file(), path
            assert provenance.file_sha256(path) == row["sha256"], path
            seen += 1
    assert seen >= 8


def test_value_level_backfill_exception_is_recorded_in_the_replay_rule():
    rows = provenance.producer_status()
    for name in FRED:
        exc = rows[name]["clock"]["replay_rule"]["value_level_exceptions"]
        assert len(exc) == 1
        e = exc[0]
        assert e["observation"] == "2020-08-19"
        assert e["published_initially"] == "EMPTY"
        assert e["backfilled_at_vintage"] == "2020-08-25"
        assert e["under_declared_asof_rule"] == "STALE_OR_NAN_NEVER_THE_BACKFILLED_VALUE"
        assert e["never_reads"] == {"fred.rates.dgs3mo.diff_5d": "0.11",
                                    "fred.rates.dgs7.logret_5d": "0.48"}[name]
        assert "TEST replay" in e["binding_on"]
        assert any("PROV02_BACKFILL.json" in a["path"]
                   for a in rows[name]["clock"]["evidence"]["artifacts"])
        assert rows[name]["clock"]["replay_rule"]["source"].endswith(
            "market_time/provenance/ALFRED_PROVENANCE.json")


def test_successor_contract_carries_the_alfred_evidence_and_the_exception():
    contract = provenance.load_successor_contract()
    by = {d["feature"]: d for d in contract["declarations"]}
    for name in FRED:
        ev = by[name]["availability_evidence"]
        assert "ALFRED" in ev and "realtime_start" in ev
        assert "339/1587" in ev or "339 of 1587" in ev
        assert "2020-08-19" in ev
        assert by[name]["availability_source"] == "UNVERIFIED"
        assert by[name]["consumption"]["source"] == "UNVERIFIED"
    assert contract["contract_sha256"] == fa.contract_sha256(contract)
    assert contract["status"] == "PREPARED_NOT_ADOPTED"
    # the five recovered hourly features stay promoted
    assert by["px.rv5"]["availability_source"] == "VERIFIED_EVIDENCE"
    assert fa.gate(list(provenance.MANIFEST), contract)["unverified_features"] == list(FRED)


def test_confirmatory_test_gate_still_refuses_and_blocks_only_test():
    with pytest.raises(provenance.ProvenanceError) as err:
        provenance.confirmatory_gate()
    assert err.value.code == "PROVENANCE_UNRESOLVED"
    report = provenance.availability_gate(list(provenance.MANIFEST))
    assert report["status"] == "PASS"
    assert report["confirmatory_test_blocked"] is True
    assert sorted(report["producer_unresolved"]) == list(FRED)
    assert report["mode"] == "DEVELOPMENT_VALIDATION_ONLY"
    assert report["blocks"] == provenance.BLOCKS_TEST_ONLY
    assert "development" in report["blocks"].lower() and "TEST" in report["blocks"]
    for name in FRED:
        assert provenance.producer_status()[name]["blocks"] == provenance.BLOCKS_TEST_ONLY
