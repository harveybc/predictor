"""Counterexamples for the 366-candidate readiness ledger."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))

import generate_m2_readiness as ledger  # noqa: E402


def _ids(count, prefix="f"):
    return [f"{prefix}{index:03d}" for index in range(count)]


def _row(feature_id):
    return {"feature_id": feature_id}


def test_missing_candidate_is_not_dropped():
    expected = _ids(ledger.DENOMINATOR)
    rows = [_row(feature) for feature in expected[:-1]]
    with pytest.raises(ledger.ReadinessError) as caught:
        ledger.require_population(rows, expected)
    assert caught.value.code == "MISSING_CANDIDATE"
    assert caught.value.detail == expected[-1]


def test_duplicate_candidate_is_not_collapsed():
    expected = _ids(ledger.DENOMINATOR)
    rows = [_row(feature) for feature in expected]
    rows[-1] = _row(expected[0])
    with pytest.raises(ledger.ReadinessError) as caught:
        ledger.require_population(rows, expected)
    assert caught.value.code == "DUPLICATE_CANDIDATE"
    assert caught.value.detail == expected[0]


def test_not_identified_is_not_causal_rejection_or_zero():
    feature = {
        "feature_id": "neutral.example",
        "coverage_status": "IN_EXTRACTIBILITY_QUEUE",
        "rung2_state_counts": {"NOT_IDENTIFIED": 14},
        "rung3_state_counts": {"NOT_IDENTIFIED": 14},
    }
    assert ledger.ps3c_status(feature) == "NOT_IDENTIFIED"
    with pytest.raises(ledger.ReadinessError) as rejected:
        ledger.refuse_false_causal_rejection("REJECTED", feature["rung2_state_counts"])
    assert rejected.value.code == "FALSE_CAUSAL_REJECTION"
    with pytest.raises(ledger.ReadinessError) as zeroed:
        ledger.refuse_missing_cast("0")
    assert zeroed.value.code == "MISSING_CAST_TO_ZERO_OR_REJECTED"
    with pytest.raises(ledger.ReadinessError) as numeric:
        ledger.refuse_false_causal_rejection(0, feature["rung2_state_counts"])
    assert numeric.value.code == "FALSE_CAUSAL_REJECTION"


def test_stale_ps4_digest_is_rejected(tmp_path):
    path = tmp_path / "profile_rows.jsonl"
    path.write_bytes(b'{"feature_id":"tv.wav_d1"}\n')
    actual = ledger.digest_path(path)
    assert ledger.require_fresh_digest(path, actual) == actual
    with pytest.raises(ledger.ReadinessError) as caught:
        ledger.require_fresh_digest(path, "0" * 64)
    assert caught.value.code == "STALE_PS4_DIGEST"


def test_reconstruction_is_not_selection_utility():
    mixed = {
        "feature_id": ledger.VIX_FEATURE,
        "selection_decision": "NOT_ISSUED",
        "reconstruction_selects": False,
        "utility": "mixed",
    }
    assert ledger.require_reconstruction_not_selection(mixed) == "NOT_SELECTION"
    selected = dict(mixed, selection_decision="SELECTED", reconstruction_selects=True)
    with pytest.raises(ledger.ReadinessError) as caught:
        ledger.require_reconstruction_not_selection(selected)
    assert caught.value.code == "RECONSTRUCTION_PRESENTED_AS_SELECTION"
    overwritten = {
        "feature_id": ledger.SIAMESE_FEATURE,
        "family": ledger.SIAMESE_FAMILY,
        "results_overwritten": True,
        "promote": True,
        "selection_decision": "SELECTED",
    }
    with pytest.raises(ledger.ReadinessError) as promoted:
        ledger.require_siamese_not_promoted(overwritten)
    assert promoted.value.code == "SIAMESE_RESULTS_OVERWRITTEN"


def test_retained_ledger_keeps_366_and_ten_measured_transforms():
    built = ledger.build(ROOT)
    rows = built["rows"]
    assert len(rows) == 366
    assert len({row["feature_id"] for row in rows}) == 366
    assert [int(row["row_index"]) for row in rows] == list(range(366))
    assert all(row["denominator"] == "366" for row in rows)
    measured = [row["feature_id"] for row in rows if row["ps4_status"] == "MEASURED_SUBPOPULATION"]
    assert set(measured) == set(ledger.MEASURED_FEATURES)
    assert built["report"]["ps4_complete_for_366"] is False
    assert built["report"]["selection_manifest"] == "NOT_ISSUED"
    historical = ledger.read_csv(ROOT / ledger.BASE / "source_transform_coverage" / "transform_coverage.csv")
    assert len(historical) == 9
    assert all(row["ps1_ps4_profile_state"] == "PENDING_PROFILE" for row in historical)
    integration = built["integration"]
    assert [row["feature_id"] for row in integration] == list(ledger.MEASURED_FEATURES)
    assert all(row["prior_status"] == "PENDING_PROFILE" and row["profile_status"] == "MEASURED" for row in integration)
    profile_rows = ledger.digest_path(ROOT / ledger.BASE / "ps4_transform_profile" / "profile_rows.jsonl")
    assert all(row["profile_rows_sha256"] == profile_rows for row in integration)
    by_id = {row["feature_id"]: row for row in rows}
    vix = by_id[ledger.VIX_FEATURE]
    assert vix["ps3r_status"] == "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    assert vix["evidence_digest"] == ledger.VIX_RESULTS_SHA256
    assert vix["ps4_status"] == "SCHEDULED_NOT_MEASURED"
    assert "selecc" not in vix["missing_next_action"].lower() or "no selecciona" in vix["missing_next_action"]
    dgs30 = by_id[ledger.DGS30_FEATURE]
    assert dgs30["ps3r_status"] == "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    assert dgs30["evidence_digest"] == ledger.DGS30_RESULTS_SHA256
    assert dgs30["ps4_status"] == "SCHEDULED_NOT_MEASURED"
    siamese = by_id[ledger.SIAMESE_FEATURE]
    assert siamese["ps3r_status"] == "LANE_E_MEASURED_NOT_SELECTION"
    assert "past_to_current_siamese" in siamese["missing_next_action"]
    scheduled = {row["feature_id"] for row in built["schedule"]}
    assert ledger.DGS30_FEATURE in scheduled
    dprime = by_id[ledger.DPRIME_FEATURE]
    assert dprime["ps3r_status"] == "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    assert dprime["evidence_digest"] == ledger.DPRIME_RESULTS_SHA256
    assert dprime["ps4_status"] == "SCHEDULED_NOT_MEASURED"
    assert ledger.DPRIME_FEATURE in scheduled
    aud = by_id[ledger.AUD_EWMA_FEATURE]
    assert aud["ps3r_status"] == "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    assert aud["evidence_digest"] == ledger.AUD_EWMA_RESULTS_SHA256
    assert aud["ps4_status"] == "SCHEDULED_NOT_MEASURED"
    assert ledger.AUD_EWMA_FEATURE in scheduled
    logret = by_id[ledger.AUD_LOGRET1H_FEATURE]
    assert logret["ps3r_status"] == "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    assert logret["evidence_digest"] == ledger.AUD_LOGRET1H_RESULTS_SHA256
    assert logret["ps4_status"] == "SCHEDULED_NOT_MEASURED"
    assert ledger.AUD_LOGRET1H_FEATURE in scheduled
    logret24 = by_id[ledger.AUD_LOGRET24H_FEATURE]
    assert logret24["ps3r_status"] == "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    assert logret24["evidence_digest"] == ledger.AUD_LOGRET24H_RESULTS_SHA256
    assert logret24["ps4_status"] == "SCHEDULED_NOT_MEASURED"
    assert ledger.AUD_LOGRET24H_FEATURE in scheduled
    assert ledger.VIX_FEATURE in scheduled
    assert ledger.SIAMESE_FEATURE in scheduled
    assert len(built["schedule"]) == len(scheduled) * len(ledger.PS4_METRICS) * len(ledger.INNER_FOLDS)
    assert all(set(row) == set(ledger.SCHEDULE_FIELDS) for row in built["schedule"])
    neutral = [row for row in rows if row["ps3c_status"] == "NOT_IDENTIFIED"]
    assert neutral
    assert all(row["ps3c_status"] not in {"0", "REJECTED"} for row in rows)
    comparison = built["comparison"]
    assert comparison["permission_to_train"] is False
    assert comparison["primary_k"] == 24
    assert comparison["sealed_sensitivity_k"] == [8, 16, 24, 32, 48]
    assert [item["arm_id"] for item in comparison["arms"]] == list(ledger.ARMS)
    assert comparison["manifesto_final"] == "NO_EMITIDO"
    assert "train" in comparison["forbidden"]
    assert comparison["preparation_identity"] == built["report"]["ps5_preparation_identity"]
    assert all(row["ps5_status"] == "PREPARED_INPUT_NOT_TRAINED" for row in rows)
    again = ledger.build(ROOT)
    assert ledger.render(built)["readiness_rows.csv"] == ledger.render(again)["readiness_rows.csv"]
    forbidden_path = "/" + "home" + "/"
    assert forbidden_path not in json.dumps(built["report"])
    assert forbidden_path not in json.dumps(comparison)
