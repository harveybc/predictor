"""Versioned EURUSD instrument/cost configuration (order 85ea134e section 3).

The execution-policy receipt must derive from the sealed versioned file whose
every field cites retained sources and carries an explicit status; a field
without source/status is refused, conflicts record both values and the chosen
conservative default, and nothing claims broker verification.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.i6dmo import strategy_adapter as sa

FILE = Path(sa.__file__).parent / "instrument_eurusd.v1.json"


@pytest.fixture
def doc():
    return json.loads(FILE.read_text())


def test_versioned_file_exists_and_seals(doc):
    loaded = sa.load_instrument_config(FILE)
    body = {k: v for k, v in doc.items() if k != "config_sha256"}
    assert doc["config_sha256"] == sa._canonical_sha(body)
    assert loaded["config_sha256"] == doc["config_sha256"]
    assert loaded["schema"] == "i6dmo.instrument_config.v1"
    assert loaded["instrument_id"] == "EURUSD"
    assert loaded["config_version"].startswith("i6dmo-eurusd-")


def test_every_field_cites_sources_and_status(doc):
    statuses = {"EVIDENCED", "ASSUMED_NOT_BROKER_VERIFIED"}
    assert doc["fields"], "fields must not be empty"
    for name, field in doc["fields"].items():
        assert "value" in field, name
        assert field.get("status") in statuses, name
        sources = field.get("sources")
        assert isinstance(sources, list) and sources and all(
            isinstance(s, str) and s.strip() for s in sources), name
        if "conflict" in field:
            conflict = field["conflict"]
            assert len(conflict["values"]) >= 2, name
            assert any(v["value"] == field["value"] for v in conflict["values"]), name
            assert isinstance(conflict.get("reason"), str) and conflict["reason"], name


def test_cost_values_are_the_conservative_resolution_of_the_retained_conflict(doc):
    f = doc["fields"]
    # The pinned plugin's point arithmetic gives 0.00002/0.00001; the lts broker's
    # pip_value 0.0001 gives 0.0002/0.0001 for the same nominal 2/1 "pips".
    spread = f["costs.spread_price"]
    assert spread["value"] == 0.0002 and spread["status"] == "ASSUMED_NOT_BROKER_VERIFIED"
    assert {v["value"] for v in spread["conflict"]["values"]} == {0.00002, 0.0002}
    slip = f["costs.slippage_price"]
    assert slip["value"] == 0.0001 and {v["value"] for v in slip["conflict"]["values"]} == {0.00001, 0.0001}
    assert f["costs.commission_per_lot_round_trip"]["value"] == 7.0
    assert f["costs.swap_per_lot_per_day"]["value"] == 10.0
    for key in ("costs.commission_per_lot_round_trip", "costs.swap_per_lot_per_day"):
        assert f[key]["status"] == "ASSUMED_NOT_BROKER_VERIFIED"
    assert doc["cost_verification"] == "NOT_BROKER_VERIFIED"
    assert f["pip_cost_parameter"]["value"] == 0.00001  # the pinned policy's POINT unit


def test_receipt_derives_from_the_versioned_file():
    receipt = sa.derive_execution_policy_from_file(FILE)
    assert receipt["schema"] == "i6dmo.execution_policy.v1"
    assert receipt["instrument"]["costs"]["spread_price"] == 0.0002
    assert receipt["instrument"]["cost_verification"] == "NOT_BROKER_VERIFIED"
    assert receipt["instrument_config_sha256"] == json.loads(FILE.read_text())["config_sha256"]
    params = sa.execution_from_policy(receipt)
    assert params["execution"].spread_price == 0.0002
    assert params["execution"].swap_per_lot_per_day == 10.0
    assert params["min_long"] == 2


def test_tampered_file_is_refused(tmp_path, doc):
    doc["fields"]["costs.spread_price"]["value"] = 0.0
    path = tmp_path / "instrument_eurusd.v1.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.load_instrument_config(path)
    assert e.value.code == "INSTRUMENT_FILE_TAMPERED"


def _reseal(doc):
    body = {k: v for k, v in doc.items() if k != "config_sha256"}
    return dict(body, config_sha256=sa._canonical_sha(body))


@pytest.mark.parametrize("change,code", [
    (lambda d: d["fields"]["costs.spread_price"].pop("sources"), "INSTRUMENT_FIELD_UNSOURCED"),
    (lambda d: d["fields"]["lot_units"].update(sources=[]), "INSTRUMENT_FIELD_UNSOURCED"),
    (lambda d: d["fields"]["pip_size"].pop("status"), "INSTRUMENT_FIELD_UNSOURCED"),
    (lambda d: d["fields"]["pip_size"].update(status="GUESSED"), "INSTRUMENT_FIELD_UNSOURCED"),
    (lambda d: d["fields"]["costs.spread_price"]["conflict"].pop("reason"),
     "INSTRUMENT_FIELD_UNSOURCED"),
    (lambda d: d.update(cost_verification="BROKER_VERIFIED"), "COST_VERIFICATION_INVALID"),
])
def test_unsourced_or_overclaimed_fields_are_refused(tmp_path, doc, change, code):
    change(doc)
    path = tmp_path / "instrument.json"
    path.write_text(json.dumps(_reseal(doc)))
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.derive_execution_policy_from_file(path)
    assert e.value.code == code


def test_missing_file_is_the_typed_wait(tmp_path):
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.load_instrument_config(tmp_path / "absent.json")
    assert e.value.code == "INSTRUMENT_CONFIG_MISSING"


# ---------------------------------------------------------------------------
# Reviewer refutation of the retained-MT5-rows claim
# (retsu/i6dmo-market-time-review-20261015 @3508e149 market_time/provenance/
#  MT5_COST_EVIDENCE.json: no mt5-bridge.sqlite exists; the retained artifacts
#  are 2026-08 ETHUSD demo snapshots with zero symbol_snapshots rows).

REFUTED_LOCATOR = "<home>/.local/state/lts/mt5-bridge.sqlite"


def test_no_text_in_the_file_claims_retained_mt5_rows_or_spread_snapshots(doc):
    blob = json.dumps(doc).lower()
    for false_claim in ("retains live spread snapshots", "retained lab rows exist",
                        "transcribing them is a later evidence task"):
        assert false_claim not in blob, false_claim


def test_refuted_evidence_is_recorded_typed_so_nobody_recites_it(doc):
    entries = doc["refuted_evidence"]
    assert isinstance(entries, list) and entries
    entry = next(e for e in entries if e["locator"] == REFUTED_LOCATOR)
    assert entry["status"] == "REFUTED_NOT_FOUND"
    assert entry["reviewer"].startswith("retsu/i6dmo-market-time-review-20261015")
    assert "MT5_COST_EVIDENCE.json" in entry["finding_artifact"]
    # The five retained snapshots are described by content, with their digests.
    assert entry["retained_snapshots"]["distinct_snapshots"] == 5
    assert entry["retained_snapshots"]["symbol_snapshot_rows"] == 0
    assert entry["retained_snapshots"]["contains_spread_field"] is False
    assert entry["retained_snapshots"]["instrument"] == "ETHUSD"
    assert entry["retained_snapshots"]["environment"] == "demo"
    assert len(entry["retained_snapshots"]["sha256"]) == 5
    assert all(len(h) == 64 for h in entry["retained_snapshots"]["sha256"])
    assert "does not exist locally" in entry["consequence"]
    assert entry["promotable_cost_fields"] == []


def test_a_field_may_not_cite_refuted_evidence(tmp_path, doc):
    doc["fields"]["costs.spread_price"]["sources"].append(
        f"MT5 lab rows at {REFUTED_LOCATOR}")
    path = tmp_path / "instrument.json"
    path.write_text(json.dumps(_reseal(doc)))
    with pytest.raises(sa.StrategyAdapterError) as e:
        sa.derive_execution_policy_from_file(path)
    assert e.value.code == "INSTRUMENT_CITES_REFUTED_EVIDENCE"


def test_schema_citation_keeps_the_five_non_cost_fields_evidenced(doc):
    evidenced = {n for n, f in doc["fields"].items() if f["status"] == "EVIDENCED"}
    assert evidenced == {"price_decimals", "point_size", "pip_size", "pip_cost_parameter",
                         "lot_units"}
    assert all(doc["fields"][n]["status"] == "ASSUMED_NOT_BROKER_VERIFIED"
               for n in doc["fields"] if n.startswith("costs."))
    # The bridge-lab SCHEMA does exist, so citing it is still evidence; the two
    # fields that cite it name the source file and claim nothing about rows.
    citing = [n for n, f in doc["fields"].items()
              if any("mt5_bridge_lab.py" in s for s in f["sources"])]
    assert citing, "the schema citation is retained"
    for name in citing:
        assert doc["fields"][name]["status"] == "EVIDENCED"
        for source in doc["fields"][name]["sources"]:
            if "mt5_bridge_lab.py" in source:
                assert "schema" in source.lower() and REFUTED_LOCATOR not in source


def test_seal_changed_and_the_receipt_rederives():
    doc = json.loads(FILE.read_text())
    stale = "f244314197ae9eca3cbf92e1d8e5eb063ef02e907508af42f34419ba7a8abd94"
    assert doc["config_sha256"] != stale
    receipt = sa.derive_execution_policy_from_file(FILE)
    assert receipt["instrument_config_sha256"] == doc["config_sha256"]
    assert receipt["instrument"]["costs"]["spread_price"] == 0.0002
    assert receipt["instrument"]["cost_verification"] == "NOT_BROKER_VERIFIED"
