"""Contract tests for I6-N component C2: strictly paired model/naive metrics.

Every fixture is synthetic. Prices are binary fractions so that sums of
partial sums equal full sums exactly and equality assertions are exact.
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


def contract(config_id="synthetic-config-a"):
    return {
        "unit": UNIT,
        "scale": SCALE,
        "roles": [ENTRY, MANAGEMENT],
        "horizons_hours": [2, 72],
        "protocol_id": PROTOCOL,
        "calendar_id": CALENDAR,
        "config_id": config_id,
    }


def rec(offset, h, p0=1.0, p1=1.5, pred=1.25, role=ENTRY, **over):
    origin = BASE + timedelta(hours=offset)
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


def document(records, keys=None, receipts=(), failures=(), exclusions=(), config_id="synthetic-config-a"):
    return {
        "schema_version": pm.INPUT_SCHEMA_VERSION,
        "contract": contract(config_id),
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
    for g in (s, u):
        assert g["model_row_keys_sha256"] == g["naive_row_keys_sha256"] == g["row_keys_sha256"]
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
