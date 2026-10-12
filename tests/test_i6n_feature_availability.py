"""Point-in-time availability gate for I6N input features (owner order 2026-10-12).

Every external feature declares event_time, available_time, timezone, an
availability source and a revision policy; a window may consume a value only
when available_time <= t.  STLFSI4 is the motivating case: a Friday-dated weekly
value consumed from about +23 h while it is published the following Thursday.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone

import pytest

from tools.i6n_contracts import feature_availability as fa

UTC = timezone.utc
STLFSI4 = "fred.stress.stlfsi4.level"


def _contract():
    return fa.load_contract()


def test_contract_retains_exactly_the_nineteen_features_and_excludes_stlfsi4():
    contract = _contract()
    retained = fa.retained_features(contract)
    assert len(retained) == 19 and len(set(retained)) == 19
    assert STLFSI4 not in retained
    assert fa.declaration(contract, STLFSI4)["status"] == "EXCLUDED"
    assert retained == list(fa.MANIFEST_NO_STLFSI4)
    assert list(fa.LEGACY_MANIFEST_20[:5] + fa.LEGACY_MANIFEST_20[6:]) == list(fa.MANIFEST_NO_STLFSI4)
    assert fa.LEGACY_MANIFEST_20[5] == STLFSI4


def test_gate_passes_the_retained_manifest_and_types_every_source():
    report = fa.gate(list(fa.MANIFEST_NO_STLFSI4))
    assert report["status"] == "PASS"
    assert report["manifest_sha256"] == fa.manifest_sha256(fa.MANIFEST_NO_STLFSI4)
    sources = {f: row["availability_source"] for f, row in report["features"].items()}
    assert set(sources) == set(fa.MANIFEST_NO_STLFSI4)
    assert set(sources.values()) <= set(fa.SOURCES)
    # Nothing silently promoted: unverifiable features stay explicitly UNVERIFIED.
    for feature in ("fred.rates.dgs3mo.diff_5d", "fred.rates.dgs7.logret_5d", "fx.gbpusd.logret_1h",
                    "px.close_loc", "px.log_co", "px.rv5", "rg.atr_ratio"):
        assert sources[feature] == "UNVERIFIED"
    for row in report["features"].values():
        assert row["violations"] == 0 and row["min_margin_hours"] >= 0


def test_test_promotion_gate_refuses_the_seven_unverified_features():
    report = fa.gate(list(fa.MANIFEST_NO_STLFSI4))
    assert report["unverified_features"] == [
        "fred.rates.dgs3mo.diff_5d",
        "fred.rates.dgs7.logret_5d",
        "fx.gbpusd.logret_1h",
        "px.close_loc",
        "px.log_co",
        "px.rv5",
        "rg.atr_ratio",
    ]
    with pytest.raises(fa.AvailabilityGateError, match="UNVERIFIED_AVAILABILITY"):
        fa.gate(list(fa.MANIFEST_NO_STLFSI4), require_verified=True)


def test_gate_refuses_an_undeclared_external_feature():
    features = list(fa.MANIFEST_NO_STLFSI4[:-1]) + ["fred.new.series.level"]
    with pytest.raises(fa.AvailabilityGateError, match="UNDECLARED_FEATURE"):
        fa.gate(features)


def test_gate_refuses_the_excluded_stlfsi4_and_the_legacy_manifest():
    with pytest.raises(fa.AvailabilityGateError, match="EXCLUDED_FEATURE"):
        fa.gate(list(fa.LEGACY_MANIFEST_20))


def test_stlfsi4_declaration_fails_availability_on_its_own_terms():
    decl = deepcopy(fa.declaration(_contract(), STLFSI4))
    decl["status"] = "RETAINED"  # even if someone re-admitted it, the clock check refuses it
    check = fa.verify_declaration(decl, date(2019, 1, 1), date(2024, 12, 31))
    assert check["violations"] > 250  # every Friday-dated weekly value
    # Friday 2023-03-03 value: consumed from Friday + 23 h, published the next Thursday.
    friday = date(2023, 3, 3)
    consumed = fa.consumed_from(decl, friday)
    available = fa.available_at(decl, friday)
    assert consumed == datetime(2023, 3, 3, 23, tzinfo=UTC)
    assert available.date() == date(2023, 3, 9) and available.weekday() == 3
    assert (available - consumed) > timedelta(days=5)
    contract = _contract()
    contract = deepcopy(contract)
    for row in contract["declarations"]:
        if row["feature"] == STLFSI4:
            row["status"] = "RETAINED"
    with pytest.raises(fa.AvailabilityGateError, match="AVAILABILITY_VIOLATION"):
        fa.gate([STLFSI4], contract=fa.validate_contract(contract, seal=False))


@pytest.mark.parametrize("field", fa.REQUIRED_FIELDS)
def test_every_required_declaration_field_is_enforced(field):
    contract = deepcopy(_contract())
    row = next(r for r in contract["declarations"] if r["feature"] == "fred.rates.dgs5.level")
    row.pop(field)
    with pytest.raises(fa.AvailabilityGateError, match="INVALID_DECLARATION"):
        fa.validate_contract(contract, seal=False)


@pytest.mark.parametrize(("path", "value"), [
    (("availability_source",), "TRUSTED"),
    (("consumption", "source"), "PROBABLY_FINE"),
    (("timezone",), "Not/AZone"),
    (("available_time", "kind"), "SOMETIME"),
    (("revision_policy", "kind"), ""),
])
def test_declaration_values_are_typed(path, value):
    contract = deepcopy(_contract())
    row = next(r for r in contract["declarations"] if r["feature"] == "fred.rates.dgs5.level")
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(fa.AvailabilityGateError, match="INVALID_DECLARATION"):
        fa.validate_contract(contract, seal=False)


def test_consumption_earlier_than_publication_is_a_violation():
    contract = deepcopy(_contract())
    row = next(r for r in contract["declarations"] if r["feature"] == "fred.rates.dgs5.level")
    row["consumption"]["lag_hours"] = row["available_time"]["lag_hours"] - 1
    with pytest.raises(fa.AvailabilityGateError, match="AVAILABILITY_VIOLATION"):
        fa.gate(list(fa.MANIFEST_NO_STLFSI4), contract=fa.validate_contract(contract, seal=False))


def test_window_may_consume_a_value_only_when_available_time_is_not_after_t():
    decl = fa.declaration(_contract(), "fred.rates.dgs5.level")
    day = date(2023, 3, 3)
    available = fa.available_at(decl, day)
    assert fa.admissible(decl, day, available)
    assert not fa.admissible(decl, day, available - timedelta(seconds=1))
    ts = [available - timedelta(hours=1), available, available + timedelta(hours=1)]
    assert fa.window_mask(decl, [day, day, day], ts) == [False, True, True]
    bar = fa.declaration(_contract(), "px.log_co")
    t = datetime(2023, 3, 3, 14, tzinfo=UTC)
    assert fa.admissible(bar, t, t) and not fa.admissible(bar, t, t - timedelta(hours=1))


def test_contract_seal_detects_any_edit():
    contract = deepcopy(_contract())
    contract["declarations"][0]["availability_evidence"] += " edited"
    with pytest.raises(fa.AvailabilityGateError, match="CONTRACT_SEAL_MISMATCH"):
        fa.validate_contract(contract)
    assert fa.contract_sha256(_contract()) == _contract()["contract_sha256"]


def test_gate_binds_contract_digest_in_its_report():
    report = fa.gate(list(fa.MANIFEST_NO_STLFSI4))
    assert report["contract_sha256"] == _contract()["contract_sha256"]
    assert report["checked_span"] == {"start": "2019-01-01", "end": "2024-12-31"}
