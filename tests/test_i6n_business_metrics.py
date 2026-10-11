"""SYNTHETIC-only C1 -> C2 integration acceptance and bypass tests.

Traceability: exact M/boundaries -> population tests; identity/no hard-row
dropping -> refusal tests; paired arithmetic/coverage -> scoring tests;
stored-metric distrust -> fresh-rescore test; historical S seal and C1 E/U
receipt timing -> historical/usage tests. No real calendar or data is read.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from tools.i6n_contracts import business_metrics as bm
from tools.i6n_contracts import eligibility as c1
from tools.i6n_contracts import paired_metrics as c2


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
SEALED_AT = "2026-10-11T12:00:00Z"  # SYNTHETIC artifact creation time, not outcome availability.
ASSET = "SYNTHETIC_ASSET"
SPLIT = "SYNTHETIC_VALIDATION"
ENTRY = "entry_origin"
MANAGEMENT = "management_origin"


def stamp(hour):
    return c2.iso_utc(BASE + timedelta(hours=hour))


def config():
    return {
        "schema_version": c1.CONFIG_SCHEMA_VERSION,
        "description": "SYNTHETIC, invented bounded fixture, not a market calendar",
        "market_calendar": {
            "calendar_id": "SYNTHETIC_CALENDAR", "version": "SYNTHETIC_V1",
            "weeks": [{"week_id": "SYNTHETIC_W1", "market_open": stamp(0),
                       "market_close": stamp(12), "entry_cutoff": stamp(6)}],
        },
        "entry_delay_after_open_hours": 1, "force_flat_before_close_hours": 2,
        "roles": ["ENTRY", "MANAGEMENT"], "horizons_hours": [2, 3],
        "origin_grid": {"kind": "EXPLICIT", "timestamps": [stamp(h) for h in (1, 5, 6, 7, 8, 10)]},
        "input_unavailable_origins": [],
        "splits": [{"split_id": SPLIT, "start": stamp(0), "end": stamp(12),
                    "label_available_by": stamp(12)}],
        "label_evidence": {"observed_targets": [
            {"timestamp": stamp(h), "available_at": stamp(h)} for h in range(13)]},
    }


def plan(raw=None):
    return c1.prepare_support_plan(config() if raw is None else raw)


def population(p, **over):
    return bm.build_opportunity_population(
        p, **dict({"asset": ASSET, "split_id": SPLIT, "sealed_at": SEALED_AT}, **over))


def document(p):
    pop = population(p)
    records = []
    for key in pop["keys"]:
        origin = c2.parse_utc("origin", key["origin"])
        hour = (origin - BASE).total_seconds() / 3600
        h = key["horizon_hours"]
        records.append({**key, "target_timestamp": stamp(hour + h),
                        "price_origin": 100 + hour, "price_target": 100 + hour + h,
                        "prediction": 100 + hour + h + 0.5,
                        "checkpoint": "SYNTHETIC_CHECKPOINT", "unit": "SYNTHETIC_PRICE",
                        "scale": "ORIGINAL_PRICE"})
    return {
        "schema_version": c2.INPUT_SCHEMA_VERSION,
        "contract": {"roles": [ENTRY, MANAGEMENT], "horizons_hours": [2, 3],
                     "calendar_id": bm.calendar_identity(p), "protocol_id": "SYNTHETIC_PROTOCOL",
                     "config_id": "SYNTHETIC_CONFIG", "unit": "SYNTHETIC_PRICE",
                     "scale": "ORIGINAL_PRICE"},
        "opportunity_population": pop, "forecasts": records,
        "forecast_failures": [], "usage_receipts": [],
    }


def score(p, doc, **over):
    return bm.score_business_metrics(
        p, doc, **dict({"asset": ASSET, "split_id": SPLIT, "sealed_at": SEALED_AT}, **over))


def group(result, h=2, role=MANAGEMENT, kind="S"):
    return next(g for g in result["groups"] if
                (g["population_kind"], g["horizon_hours"], g["role"]) == (kind, h, role))


def refuse(code, function, *args, **kwargs):
    with pytest.raises(bm.BusinessMetricsError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code


def test_exact_per_head_m_matches_c1_and_all_exclusions_reconcile():
    p = plan()
    pop = population(p)
    expected = {(stamp(t), h, r) for r, ts in
                ((ENTRY, {2: (1, 5), 3: (1, 5)}),
                 (MANAGEMENT, {2: (1, 5, 6, 7), 3: (1, 5, 6)}))
                for h, origins in ts.items() for t in origins}
    assert {(k["origin"], k["horizon_hours"], k["role"]) for k in pop["keys"]} == expected
    assert {k["asset"] for k in pop["keys"]} == {ASSET}
    for row in c1.iter_support_rows(p):
        role = {"ENTRY": ENTRY, "MANAGEMENT": MANAGEMENT}[row["role"]]
        keys = [k for k in pop["keys"] if (k["role"], k["horizon_hours"]) == (role, row["horizon_hours"])]
        counts = {e["reason"]: e["count"] for e in pop["exclusions"]
                  if (e["role"], e["horizon_hours"]) == (role, row["horizon_hours"])}
        assert len(keys) == row["label"]["label_eligible"]
        assert counts == row["operational_exclusions"]
        assert len(keys) + sum(counts.values()) == row["origins_enumerated"]
    assert c2.validate(document(p))


def test_current_origin_role_and_exact_force_flat_are_per_head():
    keys = population(plan())["keys"]

    def at(t):
        return {(k["horizon_hours"], k["role"]) for k in keys if k["origin"] == stamp(t)}

    assert at(6) == {(2, MANAGEMENT), (3, MANAGEMENT)}  # ENTRY cutoff is current origin.
    assert at(7) == {(2, MANAGEMENT)}  # h=3 hits B exactly; h=2 survives.
    assert at(8) == set()  # h=2 hits B exactly.
    assert at(10) == set()  # Current origin is B.


@pytest.mark.parametrize("change,code", [
    ("observed", "LABEL_EVIDENCE_REQUIRED"), ("splits", "SINGLE_SPLIT_REQUIRED"),
    ("outside", "OUTSIDE_CALENDAR_ORIGIN"),
])
def test_plan_preconditions(change, code):
    raw = config()
    if change == "observed":
        raw["label_evidence"] = None
    elif change == "splits":
        raw["splits"].append({"split_id": "SYNTHETIC_OTHER", "start": stamp(12),
                              "end": stamp(24), "label_available_by": None})
    else:
        raw["origin_grid"]["timestamps"].append(stamp(12))
    refuse(code, population, plan(raw))


@pytest.mark.parametrize("over,code", [
    ({"asset": ""}, "IDENTITY_INVALID"), ({"asset": " X "}, "IDENTITY_INVALID"),
    ({"split_id": "SYNTHETIC_OTHER"}, "SPLIT_MISMATCH"),
    ({"asset": None}, "IDENTITY_INVALID"), ({"split_id": ""}, "IDENTITY_INVALID"),
])
def test_explicit_identity(over, code):
    refuse(code, population, plan(), **over)


@pytest.mark.parametrize("field,value", [
    ("calendar_id", "SYNTHETIC_OTHER"), ("roles", [MANAGEMENT, "ENTRY"]),
    ("roles", [ENTRY]), ("horizons_hours", [2]), ("horizons_hours", [2, 3, 4]),
])
def test_contract_mismatch(field, value):
    p = plan()
    doc = document(p)
    doc["contract"][field] = value
    refuse("CONTRACT_MISMATCH", score, p, doc)


def test_calendar_version_is_part_of_c2_identity():
    p = plan()
    raw = config()
    raw["market_calendar"]["version"] = "SYNTHETIC_V2"
    other = plan(raw)
    assert bm.calendar_identity(other) != bm.calendar_identity(p)
    refuse("CONTRACT_MISMATCH", score, other, document(p))


@pytest.mark.parametrize("change", ["drop", "add", "exclusions", "asset", "origin", "seal", "id"])
def test_resealed_population_tampering_is_rejected(change):
    p = plan()
    doc = document(p)
    pop = doc["opportunity_population"]
    if change == "drop":
        pop["keys"].pop()
    elif change == "add":
        pop["keys"].append({"asset": ASSET, "origin": stamp(8), "horizon_hours": 2, "role": MANAGEMENT})
    elif change == "exclusions":
        pop["exclusions"][0]["count"] += 1
    elif change == "asset":
        pop["keys"][0]["asset"] = "SYNTHETIC_OTHER"
    elif change == "origin":
        pop["keys"][0]["origin"] = stamp(2)
    elif change == "seal":
        pop["sealed_at"] = stamp(-2)
    else:
        pop["population_id"] = "SYNTHETIC_FORGED"
    doc["opportunity_population"] = c2.seal_population(pop)
    refuse("SUPPORT_MISMATCH", score, p, doc)


def test_split_identity_and_bounds_bind_population_even_when_keys_match():
    p = plan()
    doc = document(p)
    raw = config()
    raw["splits"][0]["split_id"] = "SYNTHETIC_OTHER"
    refuse("SUPPORT_MISMATCH", score, plan(raw), doc, split_id="SYNTHETIC_OTHER")
    raw = config()
    raw["splits"][0]["end"] = stamp(13)
    refuse("SUPPORT_MISMATCH", score, plan(raw), doc)


def test_paired_arithmetic_uses_distinct_horizon_naive():
    p = plan()
    result = score(p, document(p))
    for h in (2, 3):
        g = group(result, h)
        assert g["mae"] == 0.5
        assert g["mse"] == 0.25
        assert g["naive_mae"] == h
        assert g["naive_mse"] == h * h
        assert g["skill_mae"] == (0.5 - h) / h
        assert g["coverage"] == 1
    c2.verify_result(result)


@pytest.mark.parametrize("typed", [False, True])
def test_missing_prediction_stays_in_denominator_and_valid_head_survives(typed):
    p = plan()
    doc = document(p)
    missing = next(r for r in doc["forecasts"] if r["origin"] == stamp(7))
    doc["forecasts"].remove(missing)
    if typed:
        doc["forecast_failures"] = [{k: missing[k] for k in ("asset", "origin", "horizon_hours")} |
                                    {"reason": "SYNTHETIC_PREDICTION_FAILURE"}]
    g = group(score(p, doc))
    assert g["state"] == "INCOMPLETE"
    assert (g["n_declared"], g["n"], g["coverage_denominator"], g["coverage"]) == (4, 3, 4, 0.75)
    assert g["failures"] == {"SYNTHETIC_PREDICTION_FAILURE" if typed else "MISSING_FORECAST": 1}


@pytest.mark.parametrize("change,reason", [
    ("missing", "TARGET_NOT_OBSERVED"), ("late", "TARGET_AVAILABLE_AFTER_LABEL_CUTOFF"),
    ("input", "INPUT_UNAVAILABLE"), ("split", "TARGET_OUTSIDE_DECLARED_SPLITS"),
])
def test_missing_label_and_other_c1_exclusions_are_typed(change, reason):
    raw = config()
    if change == "missing":
        raw["label_evidence"]["observed_targets"] = [r for r in raw["label_evidence"]["observed_targets"]
                                                       if r["timestamp"] != stamp(9)]
    elif change == "late":
        raw["label_evidence"]["observed_targets"][9]["available_at"] = stamp(13)
    elif change == "input":
        raw["input_unavailable_origins"] = [stamp(7)]
    else:
        raw["splits"][0]["end"] = stamp(9)
    p = plan(raw)
    g = group(score(p, document(p)))
    assert g["n_declared"] == 3
    assert g["upstream_exclusions"][reason] == 1
    assert g["failures"] == {}
    assert g["coverage"] == 1


def test_c2_null_label_cannot_shrink_declared_c1_support():
    p = plan()
    doc = document(p)
    next(r for r in doc["forecasts"] if r["origin"] == stamp(7))["price_target"] = None
    # Both records observing target 9 must agree about absence.
    for r in doc["forecasts"]:
        if r["target_timestamp"] == stamp(9):
            r["price_target"] = None
    with pytest.raises(c2.PairedMetricsError) as caught:
        score(p, doc)
    assert caught.value.code == "DECLARED_LABEL_ABSENT"


def test_all_missing_labels_produce_empty_population_not_zero_error():
    raw = config()
    raw["label_evidence"]["observed_targets"] = []
    p = plan(raw)
    result = score(p, document(p))
    assert population(p)["keys"] == []
    assert group(result)["state"] == "NO_ELIGIBLE_ROWS"
    assert group(result)["mae"] is None
    assert group(result)["upstream_exclusions"]["TARGET_NOT_OBSERVED"] == 4


def test_rescores_from_forecasts_and_refuses_plausible_forged_stored_metrics():
    p = plan()
    doc = document(p)
    original = deepcopy(doc)
    stored = c2.score(doc)
    assert score(p, doc, stored_result=stored) == stored
    forged_input = deepcopy(doc)
    for row in forged_input["forecasts"]:
        row["prediction"] += 1
    forged = c2.score(forged_input)
    c2.verify_result(forged)  # Self-consistent and sealed, but not our input forecasts.
    refuse("RESULT_MISMATCH", score, p, doc, stored_result=forged)
    assert group(score(p, forged_input))["mae"] == 1.5
    assert doc == original


def test_unsealed_edit_and_dropped_hard_row_forecast_are_rejected():
    p = plan()
    doc = document(p)
    dropped = doc["opportunity_population"]["keys"].pop()
    doc["forecasts"] = [r for r in doc["forecasts"] if
                        any(r[k] != dropped[k] for k in dropped)]
    refuse("SUPPORT_MISMATCH", score, p, doc)
    doc["opportunity_population"] = c2.seal_population(doc["opportunity_population"])
    refuse("SUPPORT_MISMATCH", score, p, doc)


@pytest.mark.parametrize("collection", ["forecasts", "forecast_failures", "usage_receipts"])
@pytest.mark.parametrize("change", ["asset", "origin"])
def test_extra_inputs_cannot_cross_asset_or_origin_scope(collection, change):
    p = plan()
    doc = document(p)
    row = deepcopy(doc["forecasts"][0])
    if change == "asset":
        row["asset"] = "SYNTHETIC_OTHER_ASSET"
    else:
        row["origin"] = stamp(24)
        row["target_timestamp"] = stamp(24 + row["horizon_hours"])
    if collection == "forecasts":
        doc[collection].append(row)
    elif collection == "forecast_failures":
        doc[collection].append({k: row[k] for k in ("asset", "origin", "horizon_hours")} |
                               {"reason": "SYNTHETIC_FAILURE"})
    else:
        doc[collection].append({k: row[k] for k in
                               ("asset", "origin", "horizon_hours", "role", "target_timestamp", "checkpoint")} |
                              {"receipt_id": "SYNTHETIC_RECEIPT", "recorded_at": row["origin"],
                               "calendar_id": doc["contract"]["calendar_id"],
                               "protocol_id": doc["contract"]["protocol_id"], "action": "NO_ORDER",
                               "forecast_sha256": c2.forecast_digest(row)})
    refuse("INPUT_SCOPE_MISMATCH", score, p, doc)


def test_ineligible_sibling_forecast_does_not_remove_valid_head():
    p = plan()
    doc = document(p)
    valid = next(r for r in doc["forecasts"] if r["origin"] == stamp(7))
    invalid = {**valid, "horizon_hours": 3, "target_timestamp": stamp(10),
               "price_target": 110, "prediction": 110.5}
    doc["forecasts"].append(invalid)
    result = score(p, doc)
    assert group(result, 2)["n"] == 4
    assert group(result, 3)["n"] == 3
    assert result["counts"]["forecasts_outside_populations"] == 1


def test_multiple_supplied_weeks_pool_c1_exclusions_once_per_head_role():
    raw = config()
    raw["market_calendar"]["weeks"].append({
        "week_id": "SYNTHETIC_W2", "market_open": stamp(25),
        "market_close": stamp(37), "entry_cutoff": stamp(31),
    })
    raw["origin_grid"]["timestamps"] += [stamp(h + 25) for h in (1, 5, 6, 7, 8, 10)]
    raw["splits"][0]["end"] = stamp(38)
    raw["splits"][0]["label_available_by"] = stamp(38)
    raw["label_evidence"]["observed_targets"] += [
        {"timestamp": stamp(h + 25), "available_at": stamp(h + 25)} for h in range(13)]
    p = plan(raw)
    result = score(p, document(p))
    assert group(result)["n"] == 8
    assert group(result)["upstream_exclusions"] == {
        "TARGET_AT_OR_AFTER_FORCE_FLAT": 2, "ORIGIN_AT_OR_AFTER_FORCE_FLAT": 2}
    assert len(population(p)["keys"]) == 2 * len(population(plan())["keys"])


def test_empty_evidence_accepts_actual_post_outcome_artifact_seal():
    raw = config()
    raw["label_evidence"]["observed_targets"] = []
    pop = population(plan(raw))
    assert pop["keys"] == []
    assert pop["sealed_at"] == SEALED_AT


def test_calendar_encoding_is_not_ambiguous_concatenation():
    left, right = config(), config()
    left["market_calendar"].update(calendar_id="SYNTHETIC:a", version="b")
    right["market_calendar"].update(calendar_id="SYNTHETIC", version="a:b")
    assert bm.calendar_identity(plan(left)) != bm.calendar_identity(plan(right))


def add_usage(doc, hour, h=2, role=MANAGEMENT, *, missing_label=False, forecast=True):
    row = {
        "asset": ASSET, "origin": stamp(hour), "horizon_hours": h, "role": role,
        "target_timestamp": stamp(hour + h), "price_origin": 100 + hour,
        "price_target": None if missing_label else 100 + hour + h,
        "prediction": 100 + hour + h + 0.5, "checkpoint": "SYNTHETIC_CHECKPOINT",
        "unit": "SYNTHETIC_PRICE", "scale": "ORIGINAL_PRICE",
    }
    if forecast and row not in doc["forecasts"]:
        doc["forecasts"].append(row)
    receipt = {k: row[k] for k in
               ("asset", "origin", "horizon_hours", "role", "target_timestamp", "checkpoint")}
    receipt.update(receipt_id="SYNTHETIC_RECEIPT", recorded_at=row["origin"],
                   calendar_id=doc["contract"]["calendar_id"],
                   protocol_id=doc["contract"]["protocol_id"], action="NO_ORDER",
                   forecast_sha256=c2.forecast_digest(row))
    doc["usage_receipts"].append(receipt)
    return receipt


def test_2026_actual_seal_accepts_2024_historical_m_without_backdating():
    p = plan()
    doc = document(p)
    result = score(p, doc)
    assert {c2.parse_utc("origin", k["origin"]).year for k in
            doc["opportunity_population"]["keys"]} == {2024}
    assert doc["opportunity_population"]["sealed_at"] == SEALED_AT
    assert result["opportunity_sealed_at"] == SEALED_AT
    assert result == c2.score(doc)
    assert result["state"] == "COMPLETE"


def test_eligible_missing_future_label_u_is_allowed_and_c2_excludes_it():
    raw = config()
    raw["label_evidence"]["observed_targets"] = [
        r for r in raw["label_evidence"]["observed_targets"] if r["timestamp"] != stamp(9)]
    p = plan(raw)
    doc = document(p)
    add_usage(doc, 7, missing_label=True)
    result = score(p, doc)
    g = group(result, kind="U")
    assert (g["n_declared"], g["n"], g["n_excluded"]) == (1, 0, 1)
    assert g["state"] == "NO_OBSERVABLE_LABELS"
    assert g["exclusions"] == {"LABEL_NOT_OBSERVABLE": 1}
    assert g["failures"] == {}
    assert group(result)["upstream_exclusions"]["TARGET_NOT_OBSERVED"] == 1


@pytest.mark.parametrize("recorded_hour,code", [
    (6, "RECEIPT_BEFORE_ORIGIN"), (9, "RECEIPT_AFTER_OUTCOME"),
    (10, "RECEIPT_AFTER_OUTCOME"),
])
def test_u_receipt_time_constraints_remain_unchanged(recorded_hour, code):
    p = plan()
    doc = document(p)
    receipt = add_usage(doc, 7)
    receipt["recorded_at"] = stamp(recorded_hour)
    with pytest.raises(c2.PairedMetricsError) as caught:
        score(p, doc)
    assert caught.value.code == code


@pytest.mark.parametrize("hour,h,role", [
    (6, 2, ENTRY), (7, 3, MANAGEMENT), (8, 2, MANAGEMENT), (10, 2, MANAGEMENT),
])
@pytest.mark.parametrize("forecast", [False, True])
def test_c1_ineligible_u_receipt_is_rejected_with_or_without_forecast(hour, h, role, forecast):
    p = plan()
    doc = document(p)
    add_usage(doc, hour, h, role, forecast=forecast)
    assert c2.validate(doc)  # Native C2 has no C1 calendar eligibility check.
    refuse("USAGE_OPERATIONALLY_INELIGIBLE", score, p, doc)


def test_u_valid_head_at_same_origin_as_invalid_sibling_is_allowed():
    p = plan()
    doc = document(p)
    add_usage(doc, 7, 2)
    g = group(score(p, doc), kind="U")
    assert (g["state"], g["n"], g["coverage"]) == ("COMPLETE", 1, 1)
