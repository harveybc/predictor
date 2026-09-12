"""The three banks are joined, never merged.

These tests exist because the cheapest way to fabricate a
scientific claim in this program would be to put synthetic or
financial evidence in the same table as public evidence and then
read the table as if the rows were interchangeable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import bank_index as bi  # noqa: E402

INDEX = REPO / "examples/research/crispdm_bank_index.v1.json"


@pytest.fixture(scope="module")
def index() -> dict:
    if not INDEX.is_file():
        pytest.skip("bank index not built in this checkout")
    return json.loads(INDEX.read_text())


# --------------------------------------------------------------
# the rule itself
# --------------------------------------------------------------

def test_synthetic_can_calibrate_and_nothing_more():
    bi.assert_no_promotion(bi.SYNTHETIC, "LAB_CALIBRATED")
    for forbidden in ("PUBLICLY_EVALUATED", "PUBLICLY_ELIGIBLE",
                      "DOMAIN_REVALIDATED", "LIVE_ELIGIBLE"):
        with pytest.raises(SystemExit, match="never promotes"):
            bi.assert_no_promotion(bi.SYNTHETIC, forbidden)


def test_financial_can_revalidate_but_never_publicly_qualify():
    bi.assert_no_promotion(bi.FINANCIAL, "DOMAIN_REVALIDATED")
    for forbidden in ("PUBLICLY_EVALUATED", "PUBLICLY_ELIGIBLE",
                      "LAB_CALIBRATED", "LIVE_ELIGIBLE"):
        with pytest.raises(SystemExit, match="never promotes"):
            bi.assert_no_promotion(bi.FINANCIAL, forbidden)


def test_public_bank_still_cannot_grant_live_eligibility():
    bi.assert_no_promotion(bi.PUBLIC, "PUBLICLY_ELIGIBLE")
    with pytest.raises(SystemExit, match="never promotes"):
        bi.assert_no_promotion(bi.PUBLIC, "LIVE_ELIGIBLE")
    with pytest.raises(SystemExit, match="never promotes"):
        bi.assert_no_promotion(bi.PUBLIC, "DOMAIN_REVALIDATED")


def test_unknown_authority_refuses_rather_than_defaulting():
    with pytest.raises(SystemExit, match="unknown bank"):
        bi.assert_no_promotion("SOME_NEW_BANK", "LAB_CALIBRATED")


# --------------------------------------------------------------
# the index carries authority into every row
# --------------------------------------------------------------

def test_every_row_carries_its_bank_authority(index):
    assert index["row_count"] == len(index["common_rows"])
    authorities = {bi.PUBLIC, bi.SYNTHETIC, bi.FINANCIAL}
    for row in index["common_rows"]:
        assert row["authority"] in authorities
        assert row["bank"] in index["banks"]
        assert index["banks"][row["bank"]][
            "authority_class"] == row["authority"]


def test_no_row_can_claim_a_state_its_bank_lacks(index):
    """The executable version of the join rule, applied to every
    row actually present in the index."""
    for row in index["common_rows"]:
        bi.assert_no_promotion(row["authority"],
                               "MECHANICALLY_ADMISSIBLE",
                               subject=row["id"])
        if row["authority"] != bi.PUBLIC:
            with pytest.raises(SystemExit):
                bi.assert_no_promotion(row["authority"],
                                       "PUBLICLY_ELIGIBLE",
                                       subject=row["id"])


def test_three_banks_keep_distinct_authorities(index):
    got = {k: v["authority_class"]
           for k, v in index["banks"].items()}
    assert got == {
        "public_forecasting": bi.PUBLIC,
        "synthetic_known_mechanism": bi.SYNTHETIC,
        "financial_domain": bi.FINANCIAL,
    }
    assert len(set(got.values())) == 3


# --------------------------------------------------------------
# each bank reports what its producer published, unaltered
# --------------------------------------------------------------

def test_public_bank_is_reconciled_not_reacquired(index):
    b = index["banks"]["public_forecasting"]
    assert "never downloads or scores" in b["reacquisition"]
    assert b["series_admissible_total"] == 4650
    assert sum(b["admissible_series_by_family"].values()) == 4650
    for d in b["datasets"]:
        assert d["license_id"] != "UNKNOWN" or \
            d["admission"] != "ADMITTED"
        assert d["bytes_sha256"] != "UNAVAILABLE"


def test_public_screen_verdict_is_carried_with_its_scope(index):
    b = index["banks"]["public_forecasting"]
    if "screen_adjudication" not in b:
        pytest.skip("no adjudication bound in this build")
    s = b["screen_adjudication"]
    assert s["verdict"] in ("ADVANCE_TO_DOMAIN_VALIDATION",
                            "DOES_NOT_ADVANCE", "INCONCLUSIVE")
    assert "no family-level or public-eligibility claim" in \
        s["scope"]
    assert "nothing else" in s["authority_note"]


def test_synthetic_bank_declares_calibration_only(index):
    b = index["banks"]["synthetic_known_mechanism"]
    assert b["authority_class"] == bi.SYNTHETIC
    assert "confirms nothing" in b["confirmatory_status"]
    assert b["generator_count"] == len(b["generators"])
    assert all(g["role"] == "CALIBRATION_ONLY"
               for g in b["generators"])


def test_synthetic_generators_carry_their_mechanism(index):
    b = index["banks"]["synthetic_known_mechanism"]
    t1 = [g for g in b["generators"]
          if g["generator_id"].startswith("t1::")]
    assert t1, "T1 generators missing"
    g = t1[0]
    assert set(g["mechanism"]) >= {"family", "noise_type",
                                   "snr_level", "seed"}
    r = g["reconstruction"]
    assert r["clean_signal_sha256"] != "UNAVAILABLE"
    assert r["observed_signal_sha256"] != "UNAVAILABLE"


def test_financial_bank_is_development_only(index):
    b = index["banks"]["financial_domain"]
    assert b["authority_class"] == bi.FINANCIAL
    assert "never a substitute for the public bank" in \
        b["exposed_views_note"]
    assert b["coverage"]["conceptual_variables"] == 1965
    assert b["availability_contract"]["instances_found"] == 0


def test_financial_gaps_survive_the_join(index):
    """A join must not make the financial bank look complete."""
    g = index["banks"]["financial_domain"]["gap_counts"]
    assert g["availability_gap"]["variables"] == 1965
    assert g["license_gap"]["variables"] == 1965
    assert g["unit_gap"]["variables"] == 1965


# --------------------------------------------------------------
# identity
# --------------------------------------------------------------

def test_index_self_digest_re_derives(index):
    assert bi._sha(index, "index_sha256") == index["index_sha256"]


def test_mutating_an_authority_breaks_the_digest(index):
    mutated = json.loads(json.dumps(index))
    mutated["banks"]["synthetic_known_mechanism"][
        "authority_class"] = bi.PUBLIC
    assert bi._sha(mutated, "index_sha256") != \
        index["index_sha256"]


def test_relabelled_bank_is_still_refused_by_the_rule():
    """Even if someone rewrites the label, the rule is keyed on
    the authority CONSTANT, so an invented label refuses."""
    with pytest.raises(SystemExit, match="unknown bank"):
        bi.assert_no_promotion("PUBLIC_FORECASTING_EVIDENCE_v2",
                               "PUBLICLY_ELIGIBLE")
