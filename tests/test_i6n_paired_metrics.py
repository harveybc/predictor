"""Contract tests for I6-N component C2: strictly paired model/naive metrics.

Every fixture is synthetic. Prices are binary fractions so that sums of
partial sums equal full sums exactly and equality assertions are exact.
``rec(offset, ...)`` places origins 100 hours apart so that no two fixture
records observe the same (asset, instant) unless a test says so: the scorer
requires ONE observed price per (asset, instant).
"""

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.i6n_contracts import paired_metrics as pm
from tools.i6n_contracts import paired_metrics_cli as cli


UNIT = "USD_per_EUR"
SCALE = "ORIGINAL_PRICE"
CALENDAR = "SYNTHETIC_CALENDAR_V1"
PROTOCOL = "SYNTHETIC_BUSINESS_WEEKLY_PROTOCOL"
ENTRY = "entry_origin"
MANAGEMENT = "management_origin"
BASE = datetime(2024, 1, 8, 12, tzinfo=timezone.utc)
REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "tools" / "i6n_contracts" / "examples" / "paired_metrics_synthetic.json"


def iso(value):
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def contract(config_id="synthetic-config-a", **over):
    value = {
        "unit": UNIT,
        "scale": SCALE,
        "roles": [ENTRY, MANAGEMENT],
        "horizons_hours": [2, 72],
        "protocol_id": PROTOCOL,
        "calendar_id": CALENDAR,
        "config_id": config_id,
    }
    value.update(over)
    return value


def rec(offset, h, p0=1.0, p1=1.5, pred=1.25, role=ENTRY, hour=None, **over):
    origin = BASE + timedelta(hours=offset * 100 if hour is None else hour)
    value = {
        "asset": "SYNTHETIC_PAIR",
        "origin": iso(origin),
        "horizon_hours": h,
        "target_timestamp": iso(origin + timedelta(hours=h)),
        "role": role,
        "price_origin": p0,
        "price_target": p1,
        "prediction": pred,
        "checkpoint": "synthetic-week-1",
        "unit": UNIT,
        "scale": SCALE,
    }
    value.update(over)
    return value


def key(record):
    return {k: record[k] for k in ("asset", "origin", "horizon_hours", "role")}


def population(keys, exclusions=(), population_id="synthetic-S"):
    return pm.seal_population({
        "population_id": population_id,
        "sealed_at": iso(BASE - timedelta(days=30)),
        "keys": list(keys),
        "exclusions": list(exclusions),
    })


def document(records, keys=None, receipts=(), failures=(), exclusions=(), config_id="synthetic-config-a",
             contract_over=None):
    return {
        "schema_version": pm.INPUT_SCHEMA_VERSION,
        "contract": dict(contract(config_id), **(contract_over or {})),
        "forecasts": list(records),
        "forecast_failures": list(failures),
        "opportunity_population": population(
            [key(r) for r in records] if keys is None else keys, exclusions),
        "usage_receipts": list(receipts),
    }


def receipt(record, receipt_id, minutes_after_origin=0, action="NO_ORDER", **over):
    origin = datetime.fromisoformat(record["origin"])
    value = {
        "receipt_id": receipt_id,
        "asset": record["asset"],
        "origin": record["origin"],
        "horizon_hours": record["horizon_hours"],
        "role": record["role"],
        "target_timestamp": record["target_timestamp"],
        "checkpoint": record["checkpoint"],
        "calendar_id": CALENDAR,
        "protocol_id": PROTOCOL,
        "recorded_at": iso(origin + timedelta(minutes=minutes_after_origin)),
        "action": action,
        "forecast_sha256": pm.forecast_digest(record),
    }
    value.update(over)
    return value


def group(result, kind, h, role=ENTRY):
    found = [g for g in result["groups"]
             if g["population_kind"] == kind and g["horizon_hours"] == h and g["role"] == role]
    assert len(found) == 1
    return found[0]


def refused(code, doc):
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.score(doc)
    assert caught.value.code == code, caught.value
    return caught.value


# --- per-horizon naive and skill -------------------------------------------------

def test_naive_is_computed_per_horizon_and_differs_between_h2_and_h72():
    records = [
        rec(0, 2, p0=1.0, p1=1.25, pred=1.125),
        rec(1, 2, p0=1.0, p1=1.5, pred=1.25),
        rec(0, 72, p0=1.0, p1=2.0, pred=1.5),
    ]
    result = pm.score(document(records))
    h2, h72 = group(result, "S", 2), group(result, "S", 72)
    assert (h2["n"], h2["naive_mae"], h2["naive_mse"]) == (2, 0.375, 0.15625)
    assert (h2["mae"], h2["mse"]) == (0.1875, 0.0390625)
    assert (h72["n"], h72["naive_mae"], h72["mae"]) == (1, 1.0, 0.5)
    assert h2["naive_mae"] != h72["naive_mae"]
    assert h2["skill_mae"] == (0.1875 - 0.375) / 0.375 == -0.5
    assert h72["skill_mae"] == -0.5
    assert h2["state"] == h72["state"] == "COMPLETE"


def test_model_equal_to_naive_gives_zero_skill():
    records = [rec(0, 2, p0=1.0, p1=1.5, pred=1.0), rec(1, 2, p0=1.25, p1=1.0, pred=1.25)]
    g = group(pm.score(document(records)), "S", 2)
    assert g["mae"] == g["naive_mae"] == 0.375
    assert g["skill_mae"] == 0.0 and g["skill_mse"] == 0.0


def test_smaller_error_gives_negative_skill_and_larger_error_positive_skill():
    better = group(pm.score(document([rec(0, 2, p0=1.0, p1=1.5, pred=1.25)])), "S", 2)
    worse = group(pm.score(document([rec(0, 2, p0=1.0, p1=1.5, pred=2.5)])), "S", 2)
    assert better["skill_mae"] == -0.5 and better["skill_mse"] == -0.75
    assert worse["skill_mae"] == 1.0


def test_zero_naive_leaves_skill_undefined_with_reason_and_json_has_no_nan_or_infinity():
    result = pm.score(document([rec(0, 2, p0=1.0, p1=1.0, pred=1.5)]))
    g = group(result, "S", 2)
    assert g["naive_mae"] == 0.0 and g["mae"] == 0.5
    assert g["skill_mae"] is None and g["skill_mae_undefined_reason"] == "NAIVE_MAE_ZERO"
    assert g["skill_mse"] is None and g["skill_mse_undefined_reason"] == "NAIVE_MSE_ZERO"
    text = json.dumps(result, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


# --- validation before computation ------------------------------------------------

@pytest.mark.parametrize("over", [{"unit": "pips"}, {"scale": "NORMALIZED_RETURN"}])
def test_scale_or_unit_difference_is_rejected(over):
    records = [rec(0, 2), rec(1, 2, **over)]
    refused("SCALE_MISMATCH", document(records))


def test_target_timestamp_not_equal_origin_plus_h_is_rejected():
    bad = rec(0, 2)
    bad["target_timestamp"] = iso(BASE + timedelta(hours=3))
    refused("TARGET_TIMESTAMP_MISMATCH", document([bad]))


def test_timestamp_without_zone_is_rejected():
    bad = rec(0, 2, origin="2024-01-08T12:00:00", target_timestamp="2024-01-08T14:00:00")
    refused("TIMESTAMP_NOT_AWARE", document([bad], keys=[]))


def test_duplicate_forecast_is_rejected_even_when_written_in_another_offset():
    first = rec(0, 2)
    same_instant = rec(0, 2, origin="2024-01-08T07:00:00-05:00",
                       target_timestamp="2024-01-08T09:00:00-05:00")
    refused("DUPLICATE_RECORD", document([first, same_instant], keys=[key(first)]))


def test_conflicting_values_for_one_forecast_across_roles_are_rejected():
    entry = rec(0, 2, pred=1.25, role=ENTRY)
    management = rec(0, 2, pred=1.5, role=MANAGEMENT)
    refused("FORECAST_CONFLICT", document([entry, management]))


def test_duplicate_population_key_is_rejected():
    record = rec(0, 2)
    refused("DUPLICATE_POPULATION_KEY", document([record], keys=[key(record), key(record)]))


@pytest.mark.parametrize("field,value,code", [
    ("prediction", float("nan"), "NONFINITE_VALUE"),
    ("price_origin", float("inf"), "NONFINITE_VALUE"),
    ("prediction", True, "NONFINITE_VALUE"),
    ("horizon_hours", True, "INVALID_HORIZON"),
    ("horizon_hours", 0, "INVALID_HORIZON"),
    ("checkpoint", "", "IDENTITY_INVALID"),
])
def test_nonfinite_values_and_bad_identities_are_rejected(field, value, code):
    bad = rec(0, 2)
    bad[field] = value
    refused(code, document([bad], keys=[]))


def test_tampered_sealed_population_is_rejected():
    record = rec(0, 2)
    doc = document([record])
    doc["opportunity_population"]["keys"].append(key(rec(5, 2)))
    refused("POPULATION_SEAL_INVALID", doc)


# --- coverage, failures and comparison against a sealed support ------------------

def test_one_missing_forecast_yields_incomplete_and_coverage_not_favourable_selection():
    records = [rec(i, 2, p0=1.0, p1=1.5, pred=1.25) for i in range(4)]
    support = [key(r) for r in records]
    complete = pm.score(document(records, keys=support))
    worst_missing = [records[0], records[1], records[2]]
    incomplete = pm.score(document(worst_missing, keys=support, config_id="synthetic-config-b"))
    g = group(incomplete, "S", 2)
    assert incomplete["state"] == "INCOMPLETE" and g["state"] == "INCOMPLETE"
    assert (g["n_declared"], g["n"]) == (4, 3)
    assert (g["coverage_numerator"], g["coverage_denominator"]) == (3, 4)
    assert g["coverage"] == 0.75
    assert g["failures"] == {"MISSING_FORECAST": 1}
    comparison = pm.compare_configs(complete, incomplete)
    assert comparison["state"] == "NOT_COMPARABLE_INCOMPLETE"
    assert comparison["comparisons"] is None
    coverage = [c for c in comparison["coverage"] if c["horizon_hours"] == 2 and c["role"] == ENTRY]
    assert coverage == [{"horizon_hours": 2, "role": ENTRY,
                         "a": {"state": "COMPLETE", "n": 4, "n_declared": 4},
                         "b": {"state": "INCOMPLETE", "n": 3, "n_declared": 4}}]
    assert group(complete, "S", 2)["n"] == 4


def test_declared_forecast_failure_is_counted_with_its_reason():
    records = [rec(0, 2), rec(1, 2)]
    failure = {"asset": "SYNTHETIC_PAIR", "origin": records[1]["origin"], "horizon_hours": 2,
               "reason": "INPUT_UNAVAILABLE"}
    g = group(pm.score(document([records[0]], keys=[key(r) for r in records],
                                failures=[failure])), "S", 2)
    assert g["state"] == "INCOMPLETE" and g["failures"] == {"INPUT_UNAVAILABLE": 1}


def test_comparison_requires_the_identical_previously_sealed_support():
    a_records = [rec(i, 2) for i in range(3)]
    a = pm.score(document(a_records))
    b = pm.score(document(a_records[:2], config_id="synthetic-config-b"))
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.compare_configs(a, b)
    assert caught.value.code == "SUPPORT_MISMATCH"


def test_complete_comparison_on_sealed_support_reports_paired_deltas():
    support_records = [rec(i, 2, p0=1.0, p1=1.5) for i in range(2)]
    a = pm.score(document([dict(r, prediction=1.25) for r in support_records]))
    b = pm.score(document([dict(r, prediction=1.5) for r in support_records],
                          config_id="synthetic-config-b"))
    comparison = pm.compare_configs(a, b)
    assert comparison["state"] == "COMPARABLE"
    row = [c for c in comparison["comparisons"] if c["horizon_hours"] == 2 and c["role"] == ENTRY][0]
    assert (row["mae_a"], row["mae_b"], row["delta_mae"], row["naive_mae"]) == (0.25, 0.0, -0.25, 0.5)


# --- aggregation -----------------------------------------------------------------

def test_aggregating_sums_and_counts_equals_full_scoring_not_unweighted_week_mean():
    week1 = [rec(0, 2, p0=1.0, p1=1.0, pred=5.0)]
    week2 = [rec(168 + i, 2, p0=1.0, p1=1.5, pred=1.0, checkpoint="synthetic-week-2")
             for i in range(3)]
    g1 = group(pm.score(document(week1)), "S", 2)
    g2 = group(pm.score(document(week2)), "S", 2)
    full = group(pm.score(document(week1 + week2)), "S", 2)
    pooled = pm.aggregate_groups([g1, g2])
    for field in ("n", "sum_abs_error", "sum_squared_error", "naive_sum_abs_error",
                  "naive_sum_squared_error", "mae", "mse", "naive_mae", "naive_mse", "skill_mae"):
        assert pooled[field] == full[field], field
    assert pooled["mae"] == (4.0 + 3 * 0.5) / 4
    assert pooled["mae"] != (g1["mae"] + g2["mae"]) / 2
    assert pooled["row_keys_sha256"] == full["row_keys_sha256"]


def test_aggregation_refuses_to_count_a_row_twice():
    g = group(pm.score(document([rec(0, 2)])), "S", 2)
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.aggregate_groups([g, g])
    assert caught.value.code == "DOUBLE_COUNT"


def test_aggregating_roles_counts_a_shared_forecast_once():
    entry = rec(0, 2, p0=1.0, p1=1.5, pred=1.25, role=ENTRY)
    management = dict(entry, role=MANAGEMENT)
    other = rec(1, 2, p0=1.0, p1=1.25, pred=1.0, role=MANAGEMENT)
    result = pm.score(document([entry, management, other]))
    assert group(result, "S", 2, ENTRY)["n"] == 1
    assert group(result, "S", 2, MANAGEMENT)["n"] == 2
    every = group(result, "S", 2, pm.ALL_ROLES)
    assert every["n"] == 2
    assert every["sum_abs_error"] == 0.25 + 0.25


# --- opportunity S_h versus real use U_h -----------------------------------------

def test_usage_and_opportunity_populations_differ_but_each_is_exactly_paired():
    a, b, c = (rec(i, 2, p0=1.0, p1=1.5, pred=1.25) for i in range(3))
    d = rec(3, 2, p0=1.0, p1=2.0, pred=1.0)
    receipts = [receipt(b, "r-b", action="NO_ORDER"), receipt(d, "r-d", action="OPEN")]
    result = pm.score(document([a, b, c, d], keys=[key(a), key(b), key(c)], receipts=receipts))
    s, u = group(result, "S", 2), group(result, "U", 2)
    assert (s["n"], u["n"]) == (3, 2)
    assert s["row_keys_sha256"] != u["row_keys_sha256"]
    by_key = {(r["asset"], r["origin"], r["horizon_hours"], r["role"]): r for r in (a, b, c, d)}
    for g, members in ((s, (a, b, c)), (u, (b, d))):
        expected = sorted([r["asset"], r["origin"], r["horizon_hours"], r["role"]] for r in members)
        assert g["row_keys"] == expected and g["row_keys_sha256"] == pm.canonical_sha256(expected)
        # Paired within each population, recomputed independently from the input rows.
        rows = [by_key[tuple(k)] for k in g["row_keys"]]
        assert g["mae"] == sum(abs(r["prediction"] - r["price_target"]) for r in rows) / len(rows)
        assert g["naive_mae"] == sum(abs(r["price_origin"] - r["price_target"]) for r in rows) / len(rows)
        # One row list feeds both sums: no tautological model/naive row-key digest pair is claimed.
        assert "model_row_keys_sha256" not in g and "naive_row_keys_sha256" not in g
    assert (u["mae"], u["naive_mae"]) == ((0.25 + 1.0) / 2, (0.5 + 1.0) / 2)
    assert (s["mae"], s["naive_mae"]) == (0.25, 0.5)
    assert result["counts"]["forecasts_outside_populations"] == 0


def test_usage_is_built_only_from_receipts_never_from_predictions():
    result = pm.score(document([rec(0, 2), rec(1, 2)]))
    u = group(result, "U", 2)
    assert u["state"] == "NO_RECEIPTS" and u["n"] == 0 and u["mae"] is None
    assert u["score_unavailable_reason"] == "EMPTY_POPULATION"


def test_receipt_recorded_at_or_after_the_outcome_is_rejected():
    record = rec(0, 2)
    late = receipt(record, "r-late", minutes_after_origin=120)
    refused("RECEIPT_AFTER_OUTCOME", document([record], receipts=[late]))
    early = receipt(record, "r-early", minutes_after_origin=-1)
    refused("RECEIPT_BEFORE_ORIGIN", document([record], receipts=[early]))


def test_receipt_for_a_different_prediction_is_rejected():
    record = rec(0, 2, pred=1.25)
    forged = receipt(record, "r-1", forecast_sha256=pm.forecast_digest(dict(record, prediction=1.5)))
    refused("RECEIPT_CONFLICT", document([record], receipts=[forged]))


def test_duplicate_receipt_is_rejected():
    record = rec(0, 2)
    refused("DUPLICATE_RECEIPT", document([record], receipts=[
        receipt(record, "r-1"), receipt(record, "r-2", minutes_after_origin=1)]))


def test_receipt_without_a_scored_forecast_is_incomplete_not_dropped():
    kept = rec(0, 2)
    lost = rec(1, 2)
    result = pm.score(document([kept], keys=[key(kept)],
                               receipts=[receipt(kept, "r-1"), receipt(lost, "r-2")]))
    u = group(result, "U", 2)
    assert u["state"] == "INCOMPLETE" and u["failures"] == {"MISSING_FORECAST": 1}
    assert (u["coverage_numerator"], u["coverage_denominator"]) == (1, 2)


def test_unobservable_label_is_excluded_from_usage_and_refused_in_opportunity():
    open_label = rec(0, 2, p1=None)
    result = pm.score(document([open_label], keys=[], receipts=[receipt(open_label, "r-1")]))
    u = group(result, "U", 2)
    assert u["n"] == 0 and u["n_excluded"] == 1
    assert u["exclusions"] == {"LABEL_NOT_OBSERVABLE": 1}
    refused("DECLARED_LABEL_ABSENT", document([open_label]))


def test_empty_population_has_no_score_but_keeps_exclusion_counts():
    exclusions = [{"horizon_hours": 72, "role": ENTRY, "reason": "TARGET_AT_OR_AFTER_B", "count": 5}]
    result = pm.score(document([rec(0, 2)], exclusions=exclusions))
    g = group(result, "S", 72)
    assert g["state"] == "NO_ELIGIBLE_ROWS" and g["n"] == 0
    assert g["mae"] is None and g["naive_mae"] is None and g["skill_mae"] is None
    assert g["score_unavailable_reason"] == "EMPTY_POPULATION"
    assert g["exclusions"] == {"TARGET_AT_OR_AFTER_B": 5}


# --- output records ---------------------------------------------------------------

def test_olap_records_are_flat_scalars_without_nan():
    result = pm.score(document([rec(0, 2), rec(0, 72, p1=1.0, pred=1.0)]))
    assert result["olap_records"]
    for row in result["olap_records"]:
        assert row["record_type"] in ("I6N_PAIRED_METRIC", "I6N_POPULATION_COUNT")
        assert all(v is None or isinstance(v, (str, int, float, bool)) for v in row.values())
    json.dumps(result["olap_records"], allow_nan=False)
    assert result["claims"]["warehouse_written"] is False
    assert result["design_basis"]["contracts"][
        "docs/tres_temas_entrevista/program_v3/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md"
    ] == "604f94809c41613e7346ed630d698591fadd7ecc9961b9e3cb04fcad00c01c76"
    pm.verify_result(result)
    tampered = dict(result, state="COMPLETE" if result["state"] != "COMPLETE" else "INCOMPLETE")
    with pytest.raises(pm.PairedMetricsError):
        pm.verify_result(tampered)


def test_canonical_digest_is_byte_identical_to_the_i6m_record_digest():
    from tools.i6m_queue import digest
    value = {"b": [1, 2.5, None], "a": "x", "c": {"z": True}}
    assert pm.canonical_sha256(value) == digest(value)


# --- CLI: atomic status, reuse, refusal, read-only status -------------------------

def _write_config(path, doc):
    path.write_text(json.dumps(doc, indent=1))
    return path


def test_cli_run_writes_result_and_complete_status(tmp_path, capsys):
    config = _write_config(tmp_path / "config.json", document([rec(0, 2), rec(0, 72)]))
    out = tmp_path / "out"
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_COMPLETE
    status = json.loads((out / cli.STATUS_NAME).read_text())
    for field in ("schema_version", "state", "units_done", "units_total", "started_at",
                  "last_progress_at", "duration_seconds", "eta_seconds", "eta_unavailable_reason",
                  "identity", "result", "error"):
        assert field in status, field
    assert status["state"] == "COMPLETED" and status["error"] is None
    assert status["units_done"] == status["units_total"] == 2 * 2 * 3
    assert set(status["identity"]) == {"config_file_sha256", "input_sha256", "config_sha256", "code_sha256"}
    assert status["result"]["result_state"] == "COMPLETE"
    assert str(tmp_path) not in (out / cli.STATUS_NAME).read_text()
    wrapped = json.loads((out / cli.RESULT_NAME).read_text())
    pm.verify_result(wrapped["result"])
    assert not list(out.glob("*.tmp"))


def test_cli_identical_input_reuses_intact_terminal_output(tmp_path, capsys):
    config = _write_config(tmp_path / "config.json", document([rec(0, 2)]))
    out = tmp_path / "out"
    cli.main(["--config", str(config), "--output", str(out)])
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    capsys.readouterr()
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_COMPLETE
    assert json.loads(capsys.readouterr().out)["reused"] is True
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_cli_different_input_never_overwrites_previous_output(tmp_path, capsys):
    out = tmp_path / "out"
    first = _write_config(tmp_path / "a.json", document([rec(0, 2)]))
    second = _write_config(tmp_path / "b.json", document([rec(0, 2, pred=1.5)]))
    cli.main(["--config", str(first), "--output", str(out)])
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    capsys.readouterr()
    assert cli.main(["--config", str(second), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_IDENTITY_MISMATCH"
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_cli_validation_failure_writes_typed_failed_status_and_no_result(tmp_path, capsys):
    bad = rec(0, 2)
    bad["target_timestamp"] = iso(BASE)
    config = _write_config(tmp_path / "config.json", document([bad], keys=[]))
    out = tmp_path / "out"
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    status = json.loads((out / cli.STATUS_NAME).read_text())
    assert status["state"] == "FAILED"
    assert status["error"]["code"] == "TARGET_TIMESTAMP_MISMATCH"
    assert not (out / cli.RESULT_NAME).exists()


def test_cli_incomplete_result_has_its_own_exit_code(tmp_path, capsys):
    records = [rec(0, 2), rec(1, 2)]
    config = _write_config(tmp_path / "config.json", document(records[:1], keys=[key(r) for r in records]))
    assert cli.main(["--config", str(config), "--output", str(tmp_path / "o")]) == cli.EXIT_INCOMPLETE


def test_cli_status_is_read_only_and_returns_once(tmp_path, capsys):
    absent = tmp_path / "never"
    assert cli.main(["--status", "--output", str(absent)]) == cli.EXIT_STATUS_ABSENT
    assert json.loads(capsys.readouterr().out)["state"] == "ABSENT"
    assert not absent.exists()
    config = _write_config(tmp_path / "config.json", document([rec(0, 2)]))
    out = tmp_path / "out"
    cli.main(["--config", str(config), "--output", str(out)])
    capsys.readouterr()
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()}
    assert cli.main(["--status", "--output", str(out)]) == cli.EXIT_COMPLETE
    assert json.loads(capsys.readouterr().out) == json.loads((out / cli.STATUS_NAME).read_text())
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in out.iterdir()} == before


def test_synthetic_example_json_is_valid_and_complete(tmp_path, capsys):
    doc = json.loads(EXAMPLE.read_text())
    result = pm.score(doc)
    assert result["state"] == "COMPLETE"
    assert group(result, "U", 2)["n"] >= 1
    assert cli.main(["--config", str(EXAMPLE), "--output", str(tmp_path / "o")]) == cli.EXIT_COMPLETE


# --- fix round 1: one observed price per (asset, instant) ------------------------

def test_same_origin_price_must_agree_across_horizons():
    refused("PRICE_CONFLICT", document([rec(0, 2, p0=1.0), rec(0, 72, p0=1.75)]))


def test_same_target_instant_price_must_agree_across_rows():
    long_row = rec(0, 72, p1=1.5)
    short_row = rec(None, 2, hour=70, p1=1.25)
    assert long_row["target_timestamp"] == short_row["target_timestamp"]
    refused("PRICE_CONFLICT", document([long_row, short_row]))


def test_target_price_must_agree_with_the_origin_price_observed_at_the_same_instant():
    first = rec(None, 2, hour=0, p1=1.5)
    second = rec(None, 2, hour=2, p0=1.25)
    refused("PRICE_CONFLICT", document([first, second]))


def test_unobservable_label_conflicts_with_a_price_observed_at_that_instant():
    open_label = rec(None, 2, hour=0, p1=None)
    observed = rec(None, 2, hour=2, p0=1.0)
    refused("PRICE_CONFLICT", document([open_label, observed], keys=[key(observed)],
                                       receipts=[receipt(open_label, "r-1")]))


def test_consistent_shared_prices_are_accepted():
    records = [rec(None, 2, hour=0, p0=1.0, p1=1.5), rec(None, 2, hour=2, p0=1.5, p1=1.25),
               rec(None, 72, hour=0, p0=1.0, p1=2.0)]
    assert pm.score(document(records))["state"] == "COMPLETE"


# --- fix round 1: aggregation identity and sealed groups -------------------------

def _two_weeks(over_b=None, h_b=2):
    g1 = group(pm.score(document([rec(0, 2)])), "S", 2)
    g2 = group(pm.score(document([rec(5, h_b)], contract_over=over_b)), "S", h_b)
    return g1, g2


@pytest.mark.parametrize("field,value", [
    ("config_id", "synthetic-config-b"), ("protocol_id", "OTHER_PROTOCOL"), ("calendar_id", "OTHER_CALENDAR")])
def test_aggregation_refuses_groups_from_different_config_protocol_or_calendar(field, value):
    g1, g2 = _two_weeks({field: value})
    assert g1[field] != g2[field]
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.aggregate_groups([g1, g2])
    assert caught.value.code == "AGGREGATE_MISMATCH"


def test_aggregation_refuses_groups_of_different_horizons():
    g1, g2 = _two_weeks(h_b=72)
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.aggregate_groups([g1, g2])
    assert caught.value.code == "AGGREGATE_MISMATCH"


def test_aggregation_refuses_a_tampered_unsealed_group():
    g1, g2 = _two_weeks()
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.aggregate_groups([g1, dict(g2, n=100, sum_abs_error=0.0)])
    assert caught.value.code == "GROUP_SEAL_INVALID"


def _resealed(g, **over):
    body = {k: v for k, v in g.items() if k != "sha256"}
    body.update(over)
    return dict(body, sha256=pm.canonical_sha256(body))


def test_aggregation_refuses_a_resealed_group_whose_counts_or_sums_disagree():
    g1, g2 = _two_weeks()
    for tampered in (_resealed(g2, n=100), _resealed(g2, sum_abs_error=0.0)):
        with pytest.raises(pm.PairedMetricsError) as caught:
            pm.aggregate_groups([g1, tampered])
        assert caught.value.code == "GROUP_INCONSISTENT"


def test_aggregation_refuses_a_resealed_n_that_disagrees_with_its_rows_even_with_consistent_means():
    g1, g2 = _two_weeks()
    n = 2
    assert g2["n"] == 1 and len(g2["row_keys"]) == 1
    means = {"mae": g2["sum_abs_error"] / n, "mse": g2["sum_squared_error"] / n,
             "naive_mae": g2["naive_sum_abs_error"] / n, "naive_mse": g2["naive_sum_squared_error"] / n}
    means["skill_mae"] = (means["mae"] - means["naive_mae"]) / means["naive_mae"]
    means["skill_mse"] = (means["mse"] - means["naive_mse"]) / means["naive_mse"]
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.aggregate_groups([g1, _resealed(g2, n=n, n_declared=n, coverage_numerator=n, **means)])
    assert caught.value.code == "GROUP_INCONSISTENT"


def test_aggregation_refuses_row_keys_that_do_not_match_their_digest():
    g1, g2 = _two_weeks()
    other = [["SYNTHETIC_PAIR", "2024-01-30T12:00:00Z", 2, ENTRY]]
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.aggregate_groups([g1, _resealed(g2, row_keys=other)])
    assert caught.value.code == "ROW_DIGEST_INVALID"


def test_aggregate_is_sealed_and_carries_its_identity():
    g1, g2 = _two_weeks()
    pooled = pm.aggregate_groups([g1, g2])
    assert pm.canonical_sha256({k: v for k, v in pooled.items() if k != "sha256"}) == pooled["sha256"]
    for field in ("config_id", "protocol_id", "calendar_id", "contract_sha256"):
        assert pooled[field] == g1[field] == g2[field]


# --- fix round 1: ALL_ROLES keeps the sealed exclusion counts per role -------------

def test_all_roles_group_reports_the_sealed_exclusions_per_role():
    exclusions = [{"horizon_hours": 2, "role": ENTRY, "reason": "ORIGIN_OUTSIDE_ENTRY_WINDOW", "count": 5},
                  {"horizon_hours": 2, "role": MANAGEMENT, "reason": "ORIGIN_OUTSIDE_ENTRY_WINDOW", "count": 3}]
    result = pm.score(document([rec(0, 2), rec(0, 2, role=MANAGEMENT)], exclusions=exclusions))
    every = group(result, "S", 2, pm.ALL_ROLES)
    assert every["upstream_exclusions_by_role"] == {
        ENTRY: {"ORIGIN_OUTSIDE_ENTRY_WINDOW": 5}, MANAGEMENT: {"ORIGIN_OUTSIDE_ENTRY_WINDOW": 3}}
    assert every["exclusions_scope"] == "C2_ONLY_UPSTREAM_REPORTED_PER_ROLE"
    entry = group(result, "S", 2, ENTRY)
    assert entry["exclusions_scope"] == "UPSTREAM_AND_C2"
    assert entry["exclusions"] == {"ORIGIN_OUTSIDE_ENTRY_WINDOW": 5}
    assert group(result, "U", 2, pm.ALL_ROLES)["upstream_exclusions_by_role"] == {ENTRY: {}, MANAGEMENT: {}}


# --- fix round 1: overflow is a typed refusal, never an untyped crash ---------------

def test_integer_price_too_large_for_a_float_is_a_typed_refusal():
    refused("NONFINITE_VALUE", document([rec(0, 2, p0=10 ** 400)], keys=[]))


def test_horizon_whose_target_overflows_the_calendar_is_a_typed_refusal():
    big = 10 ** 12
    record = rec(0, 2)
    record["horizon_hours"] = big
    record["target_timestamp"] = record["origin"]
    refused("TIMESTAMP_OUT_OF_RANGE", document([record], keys=[], contract_over={"horizons_hours": [2, big]}))


def test_origin_at_the_end_of_the_calendar_is_a_typed_refusal():
    record = rec(0, 2, origin="9999-12-31T23:00:00Z", target_timestamp="9999-12-31T23:00:00Z")
    refused("TIMESTAMP_OUT_OF_RANGE", document([record], keys=[]))
    early = rec(0, 2, origin="0001-01-01T00:00:00+01:00", target_timestamp="0001-01-01T01:00:00Z")
    refused("TIMESTAMP_OUT_OF_RANGE", document([early], keys=[]))


def test_error_sums_that_overflow_are_a_typed_refusal():
    records = [rec(i, 2, p0=0.0, p1=0.0, pred=1.0e308) for i in range(2)]
    refused("NONFINITE_RESULT", document(records))


def test_skill_that_overflows_is_undefined_with_a_reason():
    result = pm.score(document([rec(0, 2, p0=0.0, p1=1e-320, pred=1.0)]))
    g = group(result, "S", 2)
    assert g["skill_mae"] is None and g["skill_mae_undefined_reason"] == "SKILL_MAE_NOT_FINITE"
    json.dumps(result, allow_nan=False)


# --- fix round 1: declared identities, failures, receipts --------------------------

def test_undeclared_horizon_or_role_is_rejected_wherever_it_appears():
    refused("HORIZON_UNDECLARED", document([rec(0, 24)], keys=[]))
    refused("ROLE_UNDECLARED", document([rec(0, 2, role="exit_origin")], keys=[]))
    record = rec(0, 2)
    refused("ROLE_UNDECLARED", document([record], keys=[dict(key(record), role="exit_origin")]))
    refused("HORIZON_UNDECLARED", document([record], exclusions=[
        {"horizon_hours": 24, "role": ENTRY, "reason": "X", "count": 1}]))
    refused("ROLE_UNDECLARED", document([record], exclusions=[
        {"horizon_hours": 2, "role": "exit_origin", "reason": "X", "count": 1}]))


def _failure(record, reason="INPUT_UNAVAILABLE"):
    return {"asset": record["asset"], "origin": record["origin"],
            "horizon_hours": record["horizon_hours"], "reason": reason}


def test_duplicate_failure_is_rejected():
    lost = rec(1, 2)
    refused("DUPLICATE_FAILURE", document([], keys=[key(lost)],
                                          failures=[_failure(lost), _failure(lost, "OTHER")]))


def test_identity_both_failed_and_forecast_is_rejected():
    record = rec(0, 2)
    refused("FORECAST_CONFLICT", document([record], failures=[_failure(record)]))


@pytest.mark.parametrize("field", ["calendar_id", "protocol_id"])
def test_receipt_under_another_calendar_or_protocol_is_rejected(field):
    record = rec(0, 2)
    refused("RECEIPT_PROTOCOL_MISMATCH", document([record], receipts=[receipt(record, "r-1", **{field: "OTHER"})]))


def test_coverage_denominator_excludes_unobservable_labels_but_keeps_failures():
    kept, open_label, lost = rec(0, 2), rec(1, 2, p1=None), rec(2, 2)
    result = pm.score(document([kept, open_label], keys=[key(kept)], receipts=[
        receipt(kept, "r-1"), receipt(open_label, "r-2"), receipt(lost, "r-3")]))
    u = group(result, "U", 2)
    assert (u["n"], u["n_declared"], u["n_excluded"]) == (1, 3, 1)
    assert (u["coverage_numerator"], u["coverage_denominator"], u["coverage"]) == (1, 2, 0.5)
    assert u["state"] == "INCOMPLETE"


# --- fix round 1: comparison guards ------------------------------------------------

def test_comparison_refuses_a_different_naive_on_the_same_support():
    records = [rec(i, 2, p0=1.0, p1=1.5) for i in range(2)]
    a = pm.score(document(records))
    b = pm.score(document([dict(r, price_origin=1.25) for r in records], config_id="synthetic-config-b"))
    assert a["opportunity_support_sha256"] == b["opportunity_support_sha256"]
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.compare_configs(a, b)
    assert caught.value.code == "NAIVE_MISMATCH"


def test_comparison_refuses_a_different_calendar():
    records = [rec(0, 2)]
    a = pm.score(document(records))
    b = pm.score(document(records, config_id="synthetic-config-b", contract_over={"calendar_id": "OTHER"}))
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.compare_configs(a, b)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_comparison_refuses_different_group_sets():
    records = [rec(0, 2)]
    a = pm.score(document(records))
    b = pm.score(document(records, config_id="synthetic-config-b"))
    body = {k: v for k, v in b.items() if k != "sha256"}
    body["groups"] = [g for g in body["groups"] if not (g["population_kind"] == "S" and g["horizon_hours"] == 72)]
    with pytest.raises(pm.PairedMetricsError) as caught:
        pm.compare_configs(a, dict(body, sha256=pm.canonical_sha256(body)))
    assert caught.value.code == "CONTRACT_MISMATCH"


# --- fix round 1: CLI typed failures, status exit codes, identity guards ----------

def test_cli_unexpected_exception_writes_typed_failed_status(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path / "config.json", document([rec(0, 2)]))
    out = tmp_path / "out"

    def explode(document, progress=None):
        progress(1, 12)
        raise OverflowError("synthetic")

    monkeypatch.setattr(cli.pm, "score", explode)
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    status = json.loads((out / cli.STATUS_NAME).read_text())
    assert status["state"] == "FAILED"
    assert status["error"]["code"] == "UNEXPECTED_ERROR" and "OverflowError" in status["error"]["detail"]
    capsys.readouterr()
    assert cli.main(["--status", "--output", str(out)]) == cli.EXIT_REFUSED


def test_cli_interrupt_is_recorded_as_failed_and_reraised(tmp_path, capsys, monkeypatch):
    config = _write_config(tmp_path / "config.json", document([rec(0, 2)]))
    out = tmp_path / "out"

    def interrupt(document, progress=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.pm, "score", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--config", str(config), "--output", str(out)])
    status = json.loads((out / cli.STATUS_NAME).read_text())
    assert status["state"] == "FAILED" and status["error"]["code"] == "INTERRUPTED"


def test_cli_status_of_a_running_or_crashed_run_is_nonzero_and_rerun_completes(tmp_path, capsys):
    config = _write_config(tmp_path / "config.json", document([rec(0, 2)]))
    out = tmp_path / "out"
    identity = cli.identity_for(config.read_bytes())
    crashed = cli._Status(out / cli.STATUS_NAME, identity, cli._now(), 0.0)
    crashed.progress(3, 12)
    assert cli.main(["--status", "--output", str(out)]) == cli.EXIT_RUNNING
    assert json.loads(capsys.readouterr().out)["state"] == "RUNNING"
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_COMPLETE
    capsys.readouterr()
    assert cli.main(["--status", "--output", str(out)]) == cli.EXIT_COMPLETE


@pytest.mark.parametrize("text", ["[]", "{\"state\": \"SOMETHING_ELSE\"}", "null"])
def test_cli_status_on_a_non_object_or_unknown_state_is_typed(tmp_path, capsys, text):
    out = tmp_path / "out"
    out.mkdir()
    (out / cli.STATUS_NAME).write_text(text)
    assert cli.main(["--status", "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "STATUS_UNREADABLE"


def test_cli_missing_config_is_a_typed_refusal_and_creates_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(["--config", str(tmp_path / "absent.json"), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "CONFIG_UNREADABLE"
    assert not out.exists()


def test_cli_non_object_status_in_the_output_is_refused_as_corrupt(tmp_path, capsys):
    config = _write_config(tmp_path / "config.json", document([rec(0, 2)]))
    out = tmp_path / "out"
    out.mkdir()
    (out / cli.STATUS_NAME).write_text("[]")
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_CORRUPT"
    assert (out / cli.STATUS_NAME).read_text() == "[]"


def _completed(tmp_path, doc, name="a.json"):
    config = _write_config(tmp_path / name, doc)
    out = tmp_path / "out"
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_COMPLETE
    return config, out


def test_cli_reused_result_with_a_tampered_inner_result_is_corrupt(tmp_path, capsys):
    config, out = _completed(tmp_path, document([rec(0, 2)]))
    wrapped = json.loads((out / cli.RESULT_NAME).read_text())
    wrapped["result"]["state"] = "INCOMPLETE"
    body = {k: v for k, v in wrapped.items() if k != "sha256"}
    wrapped["sha256"] = pm.canonical_sha256(body)
    (out / cli.RESULT_NAME).write_text(json.dumps(wrapped))
    capsys.readouterr()
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_CORRUPT"


def test_cli_reused_result_file_with_a_broken_wrapper_digest_is_corrupt(tmp_path, capsys):
    config, out = _completed(tmp_path, document([rec(0, 2)]))
    wrapped = json.loads((out / cli.RESULT_NAME).read_text())
    wrapped["schema_version"] = "tampered"
    (out / cli.RESULT_NAME).write_text(json.dumps(wrapped))
    capsys.readouterr()
    assert cli.main(["--config", str(config), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_CORRUPT"


def test_cli_status_only_identity_guard_refuses_a_different_input(tmp_path, capsys):
    bad = rec(0, 2)
    bad["target_timestamp"] = iso(BASE)
    out = tmp_path / "out"
    first = _write_config(tmp_path / "a.json", document([bad], keys=[]))
    assert cli.main(["--config", str(first), "--output", str(out)]) == cli.EXIT_REFUSED
    assert not (out / cli.RESULT_NAME).exists()
    before = (out / cli.STATUS_NAME).read_bytes()
    second = _write_config(tmp_path / "b.json", document([rec(0, 2)]))
    capsys.readouterr()
    assert cli.main(["--config", str(second), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_IDENTITY_MISMATCH"
    assert (out / cli.STATUS_NAME).read_bytes() == before and not (out / cli.RESULT_NAME).exists()


def test_cli_result_only_identity_guard_refuses_a_different_input(tmp_path, capsys):
    _, out = _completed(tmp_path, document([rec(0, 2)]))
    (out / cli.STATUS_NAME).unlink()
    before = (out / cli.RESULT_NAME).read_bytes()
    second = _write_config(tmp_path / "b.json", document([rec(0, 2, pred=1.5)]))
    capsys.readouterr()
    assert cli.main(["--config", str(second), "--output", str(out)]) == cli.EXIT_REFUSED
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "OUTPUT_IDENTITY_MISMATCH"
    assert (out / cli.RESULT_NAME).read_bytes() == before and not (out / cli.STATUS_NAME).exists()
