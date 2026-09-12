"""C39-C41 (order 2026-09-11): a descriptor bound to the bytes it measured.

The C30 pilot produced honest numbers nobody could check. Its identity
was built from the variable name, the partition, the descriptor, the
value and the contract — and NOT from the data. Two different files
giving the same descriptive number collided as one observation.

C40's temporal defects were subtler and worse, because they changed
the numbers rather than their bookkeeping: the dependence descriptors
ran on the array with non-finite entries REMOVED, so "lag 1" meant
"the next value that happens to exist".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import characterization as ch                        # noqa: E402

BINDING = {
    "source_id": "d/x.csv", "source_sha256": "a" * 64,
    "window_sha256": "b" * 64, "window_contract": "rows 0..999",
    "code_identity": "c" * 64,
    "protocol_version": ch.PROTOCOL_VERSION,
    "side": "x", "contract_role": "input", "units": "usd",
    "terminal_attempt": "run-1",
}


def measure(values, **kw):
    return ch.characterize_series(
        values, variable_id="v", partition_key="p",
        bank_authority=ch.BANK_FINANCIAL,
        measured_at="2026-09-12T00:00:00Z",
        binding=kw.pop("binding", BINDING), **kw)


def by_name(rows):
    return {r["descriptor"]: r for r in rows}


# ================================================================ C39
def test_a_measurement_without_a_binding_refuses():
    with pytest.raises(SystemExit, match="missing"):
        measure([1.0, 2.0, 3.0, 4.0], binding={})


@pytest.mark.parametrize("field", sorted(ch.BINDING_KEYS))
def test_every_binding_field_is_required(field):
    partial = {k: v for k, v in BINDING.items() if k != field}
    with pytest.raises(SystemExit, match=field):
        measure([1.0, 2.0, 3.0, 4.0], binding=partial)


def test_an_undeclared_binding_field_refuses():
    with pytest.raises(SystemExit, match="undeclared binding"):
        measure([1.0, 2.0, 3.0], binding=dict(BINDING, extra="x"))


def test_different_source_bytes_are_different_observations():
    vals = [float(i) for i in range(50)]
    a = measure(vals)[0]
    b = measure(vals, binding=dict(BINDING, source_sha256="f" * 64))[0]
    assert a["value_text"] == b["value_text"], "the same number"
    assert a["observation_sha256"] != b["observation_sha256"], (
        "and yet two different observations")


def test_a_different_window_is_a_different_observation():
    vals = [float(i) for i in range(50)]
    a = measure(vals)[0]
    b = measure(vals, binding=dict(BINDING, window_sha256="9" * 64))[0]
    assert a["observation_sha256"] != b["observation_sha256"]


def test_cost_has_its_own_identity_and_does_not_move_the_science():
    vals = [float(i) for i in range(50)]
    a, b = measure(vals)[0], measure(vals)[0]
    assert a["observation_sha256"] == b["observation_sha256"], (
        "two measurements of the same bytes are ONE observation")
    assert "measurement_sha256" in a, (
        "cost and instant carry a separate identity so they can be "
        "repeated without collapsing the observation")


def test_every_row_carries_the_whole_binding():
    rows = measure([float(i) for i in range(50)])
    for r in rows:
        for k in ch.BINDING_KEYS:
            assert r[k] == BINDING[k], (r["descriptor"], k)
        assert r["binding_state"] == "BOUND_TO_SOURCE_BYTES"


def test_the_early_return_path_is_bound_too():
    """A series with too few finite values still produces rows."""
    rows = measure([float("nan"), 1.0])
    assert any(r["descriptor"] == "insufficient_finite_observations"
               for r in rows)
    for r in rows:
        assert r["source_sha256"] == BINDING["source_sha256"]


# ================================================================ C40
def test_a_gap_is_skipped_not_closed():
    t = np.arange(600.0)
    clean = np.sin(2 * np.pi * t / 12.0)
    holed = clean.copy()
    holed[2::3] = np.nan
    a, b = holed[:-1], holed[1:]
    keep = np.isfinite(a) & np.isfinite(b)
    truth = float(np.corrcoef(a[keep], b[keep])[0, 1])
    finite = holed[np.isfinite(holed)]
    compacted = float(np.corrcoef(finite[:-1], finite[1:])[0, 1])
    got = by_name(measure(holed))["autocorrelation_lag1"]["value"]
    assert got == pytest.approx(truth, abs=1e-9)
    assert abs(compacted - truth) > 0.1, (
        "the PRE behaviour must be materially different, or this test "
        "proves nothing")


def test_the_spectrum_refuses_on_a_gapped_series():
    t = np.arange(256.0)
    holed = np.sin(2 * np.pi * t / 16.0)
    holed[100] = np.nan
    row = by_name(measure(holed))["spectral_centroid"]
    assert row["identifiable"] is False
    assert row["value"] is None
    assert "NOT IDENTIFIABLE" in row["descriptor_contract"]


def test_a_predeclared_imputation_makes_the_spectrum_identifiable():
    t = np.arange(256.0)
    holed = np.sin(2 * np.pi * t / 16.0)
    holed[100] = np.nan
    row = by_name(measure(holed, imputation="linear_interpolation"))[
        "spectral_centroid"]
    assert row["identifiable"] is True
    assert "linear_interpolation" in row["descriptor_contract"], (
        "the imputation must be REPORTED, not merely applied")


def test_an_unknown_imputation_refuses():
    t = np.arange(64.0)
    holed = np.sin(t)
    holed[10] = np.nan
    with pytest.raises(SystemExit, match="unknown imputation"):
        measure(holed, imputation="whatever_looks_smooth")


def test_differences_are_taken_between_adjacent_original_positions():
    v = np.array([1.0, 2.0, np.nan, 10.0, 11.0, 12.0])
    row = by_name(measure(v))["difference_to_level_dispersion"]
    a, b = v[:-1], v[1:]
    k = np.isfinite(a) & np.isfinite(b)
    d = b[k] - a[k]
    finite = v[np.isfinite(v)]
    expected = float(np.std(d, ddof=1) / np.std(finite, ddof=1))
    assert row["value"] == pytest.approx(expected)


def test_snr_keeps_exact_alignment():
    t = np.arange(200.0)
    clean = np.sin(2 * np.pi * t / 10.0)
    observed = clean + 0.01
    observed[50] = np.nan
    row = by_name(measure(observed, noise_reference=clean))[
        "signal_to_noise_db"]
    assert row["identifiable"] is True
    assert row["value"] > 20, (
        "a near-perfect reconstruction must score high; a misaligned "
        "comparison would not")


def test_a_non_finite_value_never_leaves_as_an_identifiable_number():
    """A range that overflows float64 must not silence the batch."""
    rows = measure([1e308, 1e308, -1e308, 1e308, 1e308])
    assert rows, "one pathological variable must still produce rows"
    for r in rows:
        if r["value_text"] == "NOT_FINITE" or \
                r["value_text"].startswith("NOT_COMPUTABLE:"):
            assert r["identifiable"] is False
            assert r["value"] is None
        if r["identifiable"]:
            assert r["value"] is not None
            assert np.isfinite(r["value"])
    assert any(r["value_text"].startswith("NOT_COMPUTABLE:")
               or r["value_text"] == "NOT_FINITE" for r in rows), (
        "the pathological descriptors must be reported as typed "
        "absences, not as numbers and not as a crash")


# ================================================================ C41
def test_the_driver_excludes_temporal_identifiers(tmp_path):
    sys.path.insert(0, str(REPO / "tools"))
    import importlib
    mod = importlib.import_module("tools.run_characterization_v2")
    for c in ("DATE_TIME", "date", "timestamp", "Time"):
        assert mod.is_temporal(c), c
    for c in ("CLOSE", "rsi_14", "volume"):
        assert not mod.is_temporal(c), c
    csv = tmp_path / "x.csv"
    csv.write_text("DATE_TIME,a,b\n2020-01-01,1,2\n2020-01-02,3,4\n")
    _h, axis, numeric, axis_values = mod.read_columns(csv)
    assert axis == ["DATE_TIME"]
    assert set(numeric) == {"a", "b"}
    contract = mod.axis_contract(csv, axis, axis_values)
    assert contract["state"] == "DECLARED_AS_AXIS_NOT_MEASURED"
    assert contract["monotonic_non_decreasing"] is True


def test_the_ledger_separates_the_two_populations():
    ledger = (REPO / "docs/audits/evidence"
              / "characterization_v2_ledger_2026_09_12.json")
    if not ledger.is_file():
        pytest.skip("the v2 ledger has not been produced on this host")
    doc = json.loads(ledger.read_text())
    cov = doc["coverage"]
    assert cov["conceptual_variables_attempted"] > 0
    assert cov["physical_appearances_attempted"] > 0
    assert cov["conceptual_variables_attempted"] != \
        cov["physical_appearances_attempted"], (
        "reporting one as the other is how 7,860 appearances once "
        "looked like 7,860 variables")
    assert set(doc["outcomes"]) == {"MEASURED", "NOT_IDENTIFIABLE",
                                    "UNAVAILABLE", "FAILED"}
    assert doc["rows_by_bank"].get("PUBLIC_FORECASTING_EVIDENCE", 0) > 0, (
        "the audit found no public rows at all")
    assert doc["rows_bound_to_source"] == doc["rows_total"]


def test_the_public_pilot_touches_no_confirmatory_material():
    ledger = (REPO / "docs/audits/evidence"
              / "characterization_v2_ledger_2026_09_12.json")
    if not ledger.is_file():
        pytest.skip("the v2 ledger has not been produced on this host")
    doc = json.loads(ledger.read_text())
    public = doc["public_bank"]
    for s in public.get("skipped", []):
        assert s["admission"] != "EXCLUDED_FROM_T2_CONFIRMATORY"
    for att in doc["attempts"]:
        if att["bank"] == "PUBLIC_FORECASTING_EVIDENCE":
            assert att.get("confirmatory_material") is False


def test_a_measurement_still_never_ranks():
    rows = measure([float(i) for i in range(50)])
    ch.assert_no_selection(rows)
    with pytest.raises(SystemExit):
        ch.assert_no_selection(rows + [{"descriptor": "importance_rank"}])
