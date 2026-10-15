"""RAW_PRIORITY: a distinct, finite long control; no short relabelling or R2."""
import json
import sys
from types import SimpleNamespace

import pytest

from tools.i6dmo import matched_runner as MR


def test_profiles_are_two_closed_arm_sets_not_arbitrary_horizons():
    assert MR.contrast_profile(("mg1", "mg2m"))["horizons"] == (2, 3, 4, 5, 6, 12)
    p = MR.contrast_profile(("mg12m", "mg1l"))
    assert p["arms"] == ("mg1l", "mg12m")
    assert p["horizons"] == (18, 24, 36, 48, 60, 72)
    assert p["tie_keeps"] == "mg12m"
    assert p["family"] == "long"
    for arms in (("mg1", "mg12m"), ("mg1l",), ("mg1l", "mg12m", "mg1")):
        with pytest.raises(ValueError, match="MATCHED_ARMS_REQUIRED"):
            MR.contrast_profile(arms)


def _metrics(arm, value):
    from tools.i6dmo import reporting as R
    heads = [18, 24, 36, 48, 60, 72]
    rows = {str(h): dict(n=10, n_declared=10, n_failed=0, state="COMPLETE",
        mae=1. + value, naive_mae=1., relative_error=value, row_ids_sha256="d" * 64,
        c2_row_keys_sha256="4" * 64,
        sum_abs_error=(1. + value) * 10, naive_sum_abs_error=10.,
        relative_error_undefined_reason=None, eligible_mae_below_naive=value < 0) for h in heads}
    return dict(schema="i6dmo.paired_metrics.v1", split="DEVELOPMENT", unit=R.UNIT,
        scale="original_price", agreement=dict(status="AGREE"),
        c2=dict(state="COMPLETE", sha256="1" * 64, contract_sha256="2" * 64, input_sha256="3" * 64),
        per_horizon=rows, matched=dict(schema=R.MATCHED_REPORT_SCHEMA, population="COMMON_DEVELOPMENT",
        arm=arm, scientific_fit=True, fit_device="gpu", common_population_sha256="a" * 64,
        development_row_ids_sha256="b" * 64, development_head_mask_sha256="c" * 64,
        development_counts={str(h): 10 for h in heads}, capacity=dict(arm=arm),
        manual_pooled_replay={h: {k: r[k] for k in ("n", "mae", "naive_mae", "relative_error")}
                              for h, r in rows.items()}, selection=R.selection_score(rows, heads)))


@pytest.fixture
def capacity_stub(monkeypatch):
    class CapacityError(ValueError):
        pass
    def delta(a, b):
        assert (a["arm"], b["arm"]) == ("mg1l", "mg12m")
        return dict(total=25344, heads=25344)
    # This is a decision-unit test, not a model capacity or TensorFlow check.
    stub = SimpleNamespace(CapacityError=CapacityError, matched_capacity_delta=delta)
    import tools.i6dmo
    monkeypatch.setitem(sys.modules, "tools.i6dmo.model", stub)
    monkeypatch.setattr(tools.i6dmo, "model", stub, raising=False)


def test_long_decision_uses_long_arm_names_and_all_six_heads(capacity_stub):
    from tools.i6dmo import reporting
    a, b = _metrics("mg1l", -.02), _metrics("mg12m", -.01)
    result = reporting.decide_matched(a, b)
    assert result["selected"] == "mg1l"
    assert set(result["per_head"]) == {"18", "24", "36", "48", "60", "72"}
    assert MR.disposition_of(result) == "MG1_LONG_SELECTED"
    a = _metrics("mg1l", -.01)
    result = reporting.decide_matched(a, b)
    assert result["selected"] == "mg12m"
    assert MR.disposition_of(result) == "MG12_RETAINED_WITHIN_TOLERANCE"


def test_long_decision_refuses_short_or_inconsistent_horizon_metadata(capacity_stub):
    from tools.i6dmo import reporting
    a, b = _metrics("mg1l", -.02), _metrics("mg12m", -.01)
    b["matched"]["selection"]["heads"] = [2, 3, 4, 5, 6, 12]
    with pytest.raises(reporting.ReportFailure, match="MATCHED_HORIZONS"):
        reporting.decide_matched(a, b)


@pytest.mark.parametrize("change", [
    lambda m: m["per_horizon"].pop("72"),
    lambda m: m["matched"]["selection"].update(score=-.9),
    lambda m: m["matched"].update(common_population_sha256=""),
    lambda m: m["matched"].update(development_row_ids_sha256=None),
    lambda m: m["matched"].update(development_head_mask_sha256=""),
    lambda m: m["per_horizon"]["72"].update(n=9),
    lambda m: m["per_horizon"]["72"].update(n_failed=1, state="INCOMPLETE"),
    lambda m: m["per_horizon"]["72"].update(relative_error=float("nan")),
    lambda m: m["per_horizon"]["72"].update(naive_mae=10.),
    lambda m: m["matched"]["development_counts"].update({"72": 9}),
    lambda m: m["matched"].update(selection=None),
    lambda m: m["per_horizon"]["72"].update(mae=1e308, naive_mae=1e-308),
])
def test_long_decision_rejects_actual_rows_or_cache_contradictions(capacity_stub, change):
    from tools.i6dmo import reporting
    a, b = _metrics("mg1l", -.02), _metrics("mg12m", -.01)
    change(a)
    # Empty matching identities are not a common population either.
    for key in ("common_population_sha256", "development_row_ids_sha256", "development_head_mask_sha256"):
        if not a["matched"].get(key):
            b["matched"][key] = a["matched"].get(key)
    with pytest.raises(reporting.ReportFailure):
        reporting.decide_matched(a, b)


def test_long_comparison_does_not_mutate_valid_reports_and_all_heads_may_lose(capacity_stub):
    import copy
    from tools.i6dmo import reporting
    a, b = _metrics("mg1l", .2), _metrics("mg12m", .1)
    before = copy.deepcopy((a, b))
    result = reporting.decide_matched(a, b)
    assert result["selected"] == "mg12m"
    assert result["scores"] == pytest.approx({"mg1l": .2, "mg12m": .1})
    assert (a, b) == before


def test_long_validation_cannot_be_disabled_with_engineering_opt_out(capacity_stub):
    from tools.i6dmo import reporting
    a, b = _metrics("mg1l", -.02), _metrics("mg12m", -.01)
    del a["per_horizon"]["72"]
    with pytest.raises(reporting.ReportFailure, match="MATCHED_HORIZONS"):
        reporting.decide_matched(a, b, require_scientific=False)


def test_missing_long_report_never_selects_from_the_remaining_arm(tmp_path):
    out = tmp_path / "out"
    terminal = MR.run_unit(dict(kind="comparison", unit_id="comparison", campaign_id="fixture",
        identity_sha256="d" * 64, test_open=False, upstream={}, effective_config=dict(
            mode="MATCHED_LONG_CONTRAST", arms={"mg1l": {}, "mg12m": {}}, data_pins={})),
        out, tmp_path / "progress.json")
    assert terminal["disposition"] == "INCOMPLETE_COMPARISON"
    comparison = json.loads((out / "COMPARISON.json").read_text())
    assert set(comparison["missing"]) == {"ARM_MISSING:mg1l", "ARM_MISSING:mg12m"}
    assert comparison["verdict"]["selected"] is None
    assert comparison["test_read"] is False


def test_long_arm_view_does_not_read_an_unrelated_long_family():
    from tools.i6dmo import reporting, unit_runner
    design = {"experiment_id": "LONG_MATCHED_FIXTURE"}
    spec = dict(arm="mg1l", family="long", kind="fit", effective_config=dict(
        design=design, long=None))
    assert unit_runner._family_design(spec) == design
    assert reporting._design(spec) == design


def test_long_cli_is_not_the_short_command():
    from tools.i6dmo.cli import build_parser
    args = build_parser().parse_args(["matched-long", "status", "--root", "/fixture"])
    assert args.matched_family == "long"


def test_campaign_configured_cadence_has_explicit_bounds():
    from tools.i6dmo import deploy
    v = deploy.timer_cadence(dict(timer_cadence=dict(worker_seconds=5, cpu_worker_seconds=10,
        tick_seconds=10, publish_seconds=60, boot_seconds=30, accuracy_seconds=1)))
    assert v["WORKER_INTERVAL"] == 5 and v["PUBLISH_INTERVAL"] == 60
    with pytest.raises(ValueError, match="TIMER_CADENCE"):
        deploy.timer_cadence(dict(timer_cadence=dict(worker_seconds=0)))
