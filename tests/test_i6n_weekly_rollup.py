"""Lane C: annual roll-up of weekly receipts by sums/counts with identical naive."""

import json
import math

import pytest

from tools import i6n_weekly_rollup as wr

EXPECTED = {"W2024-01-07": {"n": 3, "origins_sha256": "a" * 64},
            "W2024-01-14": {"n": 1, "origins_sha256": "b" * 64},
            "W2024-12-29": {"n": 0, "origins_sha256": "c" * 64}}


def row(fold, n, err, naive, arm="cand-a", h=12, **kw):
    r = {"schema_version": "i6n.candidate_receipt.v1", "experiment": "I6N_WEEKLY",
         "candidate_identity_sha256": arm, "fold": fold, "horizon_hours": h,
         "head_offset_hours": h, "head_role": "ENDPOINT", "comparison": True,
         "role": "management_origin", "population": "S", "split": "VALIDATION",
         "metric": "ABS_AND_SQUARED_ERROR_SUMS", "unit": "PRICE", "scale": "ORIGINAL_PRICE",
         "n": n, "sum_abs_error": err, "sum_sq_error": err * err,
         "naive_sum_abs_error": naive, "naive_sum_sq_error": naive * naive,
         "c2_receipt_sha256": "d" * 64, "status": "SCORED",
         "support_sha256": EXPECTED[fold]["origins_sha256"]}
    r.update(kw)
    return r


def acc(arm="cand-a", h=12):
    return wr.AnnualAccumulator(experiment="I6N_WEEKLY", candidate_identity_sha256=arm,
                                horizon_hours=h, role="management_origin", expected=EXPECTED)


def test_annual_mae_is_sum_over_sum_not_mean_of_weeks():
    a = acc()
    a.add(row("W2024-01-07", 3, 0.003, 0.006))
    a.add(row("W2024-01-14", 1, 0.005, 0.004))
    s = a.summary()
    assert s["state"] == "ANNUAL_COMPLETE"
    assert s["n"] == 4
    assert math.isclose(s["mae"], 0.008 / 4)
    assert math.isclose(s["naive_mae"], 0.010 / 4)
    assert math.isclose(s["skill_mae"], (0.002 - 0.0025) / 0.0025)
    weekly_mean = (0.001 + 0.005) / 2
    assert math.isclose(s["weekly_mean_mae_diagnostic"], weekly_mean)
    assert not math.isclose(s["mae"], weekly_mean)
    assert s["weeks_no_support"] == ["W2024-12-29"]
    assert s["scope"] == wr.ANNUAL_SCOPE


def test_missing_week_is_incomplete_not_annual():
    a = acc()
    a.add(row("W2024-01-07", 3, 0.003, 0.006))
    s = a.summary()
    assert s["state"] == "INCOMPLETE" and s["mae"] is None and s["skill_mae"] is None
    assert s["missing_folds"] == ["W2024-01-14"]


def test_receipt_must_match_frozen_support():
    a = acc()
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-01-07", 2, 0.003, 0.006))
    assert e.value.code == "SUPPORT_COUNT_MISMATCH"
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-01-07", 3, 0.003, 0.006, support_sha256="f" * 64))
    assert e.value.code == "SUPPORT_DIGEST_MISMATCH"
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-12-29", 1, 0.003, 0.006))
    assert e.value.code in ("SUPPORT_COUNT_MISMATCH", "SUPPORT_DIGEST_MISMATCH")
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-01-07", 3, 0.003, 0.006, split="INNER"))
    assert e.value.code == "IDENTITY_MISMATCH"
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-01-07", 3, 0.003, 0.006, h=72, head_offset_hours=72))
    assert e.value.code == "IDENTITY_MISMATCH"


def test_accumulated_state_survives_restart_and_duplicates():
    a = acc()
    assert a.add(row("W2024-01-07", 3, 0.003, 0.006)) == "ADDED"
    saved = json.loads(json.dumps(a.to_json()))
    b = wr.AnnualAccumulator.from_json(saved)
    assert b.add(row("W2024-01-07", 3, 0.003, 0.006)) == "DUPLICATE"
    assert b.add(row("W2024-01-14", 1, 0.005, 0.004)) == "ADDED"
    one = acc()
    one.add(row("W2024-01-14", 1, 0.005, 0.004))
    one.add(row("W2024-01-07", 3, 0.003, 0.006))
    assert b.summary() == one.summary()
    with pytest.raises(wr.RollupError) as e:
        b.add(row("W2024-01-14", 1, 0.006, 0.004))
    assert e.value.code == "CONFLICTING_DUPLICATE"
    tampered = dict(saved, state_sha256="0" * 64)
    with pytest.raises(wr.RollupError):
        wr.AnnualAccumulator.from_json(tampered)


def test_identical_naive_required_between_arms():
    a, b = acc("cand-a"), acc("control")
    for x, err in ((a, 0.003), (b, 0.004)):
        x.add(row("W2024-01-07", 3, err, 0.006, arm=x.candidate_identity_sha256))
        x.add(row("W2024-01-14", 1, err, 0.004, arm=x.candidate_identity_sha256))
    cmp = wr.compare_annual(a, b)
    assert cmp["state"] == "COMPARABLE" and cmp["n"] == 4
    assert math.isclose(cmp["delta_mae"], (0.008 - 0.006) / 4)
    c = acc("other")
    c.add(row("W2024-01-07", 3, 0.003, 0.0061, arm="other"))
    c.add(row("W2024-01-14", 1, 0.003, 0.004, arm="other"))
    with pytest.raises(wr.RollupError) as e:
        wr.compare_annual(a, c)
    assert e.value.code == "NAIVE_MISMATCH"
    d = acc("partial")
    d.add(row("W2024-01-07", 3, 0.003, 0.006, arm="partial"))
    assert wr.compare_annual(a, d)["state"] == "NOT_COMPARABLE_INCOMPLETE"


def test_inner_fold_score_never_compared_with_annual():
    inner = wr.fold_summary(dict(row("W2024-01-07", 3, 0.003, 0.006, h=72, head_offset_hours=72),
                                 fold="2023Q4", split="INNER", n=70))
    a = acc()
    a.add(row("W2024-01-07", 3, 0.003, 0.006))
    a.add(row("W2024-01-14", 1, 0.005, 0.004))
    with pytest.raises(wr.RollupError) as e:
        wr.assert_comparable(inner, a.summary())
    assert e.value.code in ("NOT_COMPARABLE_SCOPE", "NOT_COMPARABLE_HORIZON")
    same_scope_other_h = dict(inner, horizon_hours=12)
    with pytest.raises(wr.RollupError) as e:
        wr.assert_comparable(inner, dict(same_scope_other_h, fold="2023Q4"))
    assert e.value.code == "NOT_COMPARABLE_HORIZON"


def test_no_support_week_accepts_only_typed_empty_row():
    a = acc()
    assert a.add(row("W2024-12-29", 0, 0.0, 0.0, status="NO_SPLIT_SUPPORT",
                     sum_abs_error=None, sum_sq_error=None, naive_sum_abs_error=None,
                     naive_sum_sq_error=None)) == "ADDED"
    a.add(row("W2024-01-07", 3, 0.003, 0.006))
    a.add(row("W2024-01-14", 1, 0.005, 0.004))
    assert a.summary()["n"] == 4
