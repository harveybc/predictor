"""Lane C: annual roll-up of weekly receipts by sums/counts with identical naive.

The roll-up is keyed by the ARM (finalist genotype or effective-config hash and
arm role). Each weekly refit has its own A ``candidate_identity_sha256`` (the
cell config differs every week), which is kept per row (review E-C1).
"""

import hashlib
import json
import math

import pytest

from tools import i6n_weekly_rollup as wr

EXPECTED = {"W2024-01-07": {"n": 3, "origins_sha256": "a" * 64},
            "W2024-01-14": {"n": 1, "origins_sha256": "b" * 64},
            "W2024-12-29": {"n": 0, "origins_sha256": "c" * 64}}
ARM = "1" * 64
OTHER_ARM = "2" * 64


def week_identity(fold, arm=ARM):
    return hashlib.sha256(f"{arm}:{fold}".encode()).hexdigest()


def row(fold, n, err, naive, arm=ARM, h=12, **kw):
    r = {"schema_version": "i6n.candidate_receipt.v1", "experiment": "I6N_WEEKLY",
         "candidate_identity_sha256": week_identity(fold, arm), "fold": fold, "horizon_hours": h,
         "head_offset_hours": h, "head_role": "ENDPOINT", "comparison": True,
         "role": "MANAGEMENT", "population": "S", "split": "VALIDATION",
         "metric": "ABS_AND_SQUARED_ERROR_SUMS", "unit": "PRICE", "scale": "ORIGINAL_PRICE",
         "n": n, "sum_abs_error": err, "sum_sq_error": err * err,
         "naive_sum_abs_error": naive, "naive_sum_sq_error": naive * naive,
         "c2_receipt_sha256": "d" * 64, "status": "SCORED",
         "support_sha256": EXPECTED[fold]["origins_sha256"]}
    r.update(kw)
    return r


def acc(arm=ARM, h=12, arm_role="BEST"):
    return wr.AnnualAccumulator(experiment="I6N_WEEKLY", arm_sha256=arm, arm_role=arm_role,
                                horizon_hours=h, role="MANAGEMENT", expected=EXPECTED)


def full(a, arm=ARM, err=(0.003, 0.005), naive=(0.006, 0.004)):
    a.add(row("W2024-01-07", 3, err[0], naive[0], arm=arm), arm_sha256=arm)
    a.add(row("W2024-01-14", 1, err[1], naive[1], arm=arm), arm_sha256=arm)
    return a


def test_regression_e_c1_weekly_identities_differ_and_year_completes():
    a = acc()
    assert a.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM) == "ADDED"
    assert a.add(row("W2024-01-14", 1, 0.005, 0.004), arm_sha256=ARM) == "ADDED"
    s = a.summary()
    assert s["state"] == "ANNUAL_COMPLETE"
    ids = {f: m["candidate_identity_sha256"] for f, m in s["canonical_measurements"].items()}
    assert ids == {"W2024-01-07": week_identity("W2024-01-07"), "W2024-01-14": week_identity("W2024-01-14")}
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-01-07", 3, 0.003, 0.006, arm=OTHER_ARM), arm_sha256=OTHER_ARM)
    assert e.value.code == "ARM_MISMATCH"


def test_add_receipt_takes_arm_and_attempt_from_the_header():
    a = acc()
    for fold, n, err, naive in (("W2024-01-07", 3, 0.003, 0.006), ("W2024-01-14", 1, 0.005, 0.004)):
        r = row(fold, n, err, naive)
        other_head = dict(r, head_offset_hours=6, head_role="INTERMEDIATE", comparison=False)
        receipt = {"schema_version": "i6n.candidate_receipt.v1", "fold": fold,
                   "genotype_canonical_sha256": ARM, "effective_config_sha256": "e" * 64,
                   "attempt": {"index": 1}, "rows": [other_head, r]}
        assert a.add_receipt(receipt) == ["ADDED"]
    assert a.summary()["state"] == "ANNUAL_COMPLETE"


def test_annual_mae_is_sum_over_sum_not_mean_of_weeks():
    s = full(acc()).summary()
    assert s["n"] == 4
    assert math.isclose(s["mae"], 0.008 / 4)
    assert math.isclose(s["naive_mae"], 0.010 / 4)
    assert math.isclose(s["skill_mae"], (0.002 - 0.0025) / 0.0025)
    weekly_mean = (0.001 + 0.005) / 2
    assert math.isclose(s["weekly_mean_mae_diagnostic"], weekly_mean)
    assert not math.isclose(s["mae"], weekly_mean)
    assert s["weeks_no_support"] == ["W2024-12-29"]
    assert s["scope"] == wr.ANNUAL_SCOPE and s["arm_role"] == "BEST" and s["arm_sha256"] == ARM


def test_missing_week_is_incomplete_not_annual():
    a = acc()
    a.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM)
    s = a.summary()
    assert s["state"] == "INCOMPLETE" and s["mae"] is None and s["skill_mae"] is None
    assert s["missing_folds"] == ["W2024-01-14"]


def test_receipt_must_match_frozen_support():
    a = acc()
    cases = ((dict(n=2), "SUPPORT_COUNT_MISMATCH"),
             (dict(support_sha256="f" * 64), "SUPPORT_DIGEST_MISMATCH"),
             (dict(split="INNER"), "IDENTITY_MISMATCH"),
             (dict(h=72, head_offset_hours=72), "IDENTITY_MISMATCH"),
             (dict(role="management_origin"), "IDENTITY_MISMATCH"),
             (dict(candidate_identity_sha256="xyz"), "RECEIPT_INVALID"))
    for kw, code in cases:
        n, h = kw.get("n", 3), kw.get("h", 12)
        extra = {k: v for k, v in kw.items() if k not in ("n", "h")}
        with pytest.raises(wr.RollupError) as e:
            a.add(row("W2024-01-07", n, 0.003, 0.006, h=h, **extra), arm_sha256=ARM)
        assert e.value.code == code, kw
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-12-29", 1, 0.003, 0.006), arm_sha256=ARM)
    assert e.value.code in ("SUPPORT_COUNT_MISMATCH", "SUPPORT_DIGEST_MISMATCH")


def test_lowest_attempt_is_the_canonical_measurement():
    a = acc()
    assert a.add(row("W2024-01-07", 3, 0.009, 0.006), arm_sha256=ARM, attempt_index=2) == "ADDED"
    assert a.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM, attempt_index=1) == "ADDED"
    a.add(row("W2024-01-14", 1, 0.005, 0.004), arm_sha256=ARM)
    s = a.summary()
    assert s["canonical_measurements"]["W2024-01-07"]["attempt_index"] == 1
    assert math.isclose(s["sum_abs_error"], 0.008)
    assert s["non_canonical_attempts"] == {"W2024-01-07": [2]}
    with pytest.raises(wr.RollupError) as e:
        a.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM, attempt_index=4)
    assert e.value.code == "ATTEMPTS_EXCEEDED"


def test_accumulated_state_survives_restart_and_duplicates():
    a = acc()
    assert a.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM) == "ADDED"
    saved = json.loads(json.dumps(a.to_json()))
    b = wr.AnnualAccumulator.from_json(saved)
    assert b.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM) == "DUPLICATE"
    assert b.add(row("W2024-01-14", 1, 0.005, 0.004), arm_sha256=ARM) == "ADDED"
    one = acc()
    one.add(row("W2024-01-14", 1, 0.005, 0.004), arm_sha256=ARM)
    one.add(row("W2024-01-07", 3, 0.003, 0.006), arm_sha256=ARM)
    assert b.summary() == one.summary()
    with pytest.raises(wr.RollupError) as e:
        b.add(row("W2024-01-14", 1, 0.006, 0.004), arm_sha256=ARM)
    assert e.value.code == "CONFLICTING_DUPLICATE"
    with pytest.raises(wr.RollupError):
        wr.AnnualAccumulator.from_json(dict(saved, state_sha256="0" * 64))


def test_identical_naive_required_between_arms():
    a = full(acc())
    b = full(acc(OTHER_ARM, arm_role="MINIMAL_CONTROL"), arm=OTHER_ARM, err=(0.004, 0.004))
    cmp = wr.compare_annual(a, b)
    assert cmp["state"] == "COMPARABLE" and cmp["n"] == 4
    assert math.isclose(cmp["delta_mae"], 0.0, abs_tol=1e-12)
    c = full(acc("3" * 64), arm="3" * 64, naive=(0.0061, 0.004))
    with pytest.raises(wr.RollupError) as e:
        wr.compare_annual(a, c)
    assert e.value.code == "NAIVE_MISMATCH"
    d = acc("4" * 64)
    d.add(row("W2024-01-07", 3, 0.003, 0.006, arm="4" * 64), arm_sha256="4" * 64)
    assert wr.compare_annual(a, d)["state"] == "NOT_COMPARABLE_INCOMPLETE"


def test_inner_fold_score_never_compared_with_annual():
    inner = wr.fold_summary(dict(row("W2024-01-07", 3, 0.003, 0.006, h=72, head_offset_hours=72),
                                 fold="2023Q4", split="INNER", n=70))
    with pytest.raises(wr.RollupError) as e:
        wr.assert_comparable(inner, full(acc()).summary())
    assert e.value.code in ("NOT_COMPARABLE_SCOPE", "NOT_COMPARABLE_HORIZON")
    with pytest.raises(wr.RollupError) as e:
        wr.assert_comparable(inner, dict(inner, horizon_hours=12))
    assert e.value.code == "NOT_COMPARABLE_HORIZON"


def test_no_support_week_accepts_only_typed_empty_row():
    a = acc()
    assert a.add(row("W2024-12-29", 0, 0.0, 0.0, status="NO_SPLIT_SUPPORT", sum_abs_error=None,
                     sum_sq_error=None, naive_sum_abs_error=None, naive_sum_sq_error=None),
                 arm_sha256=ARM) == "ADDED"
    assert full(a).summary()["n"] == 4


def test_legacy_single_identity_api_is_a_typed_refusal():
    with pytest.raises(wr.RollupError) as e:
        wr.AnnualAccumulator(experiment="I6N_WEEKLY", candidate_identity_sha256=week_identity("W2024-01-07"),
                             horizon_hours=12, role="MANAGEMENT", expected=EXPECTED)
    assert e.value.code == "LEGACY_IDENTITY_KEY"
    with pytest.raises(wr.RollupError) as e:
        acc().add(row("W2024-01-07", 3, 0.003, 0.006))
    assert e.value.code == "ARM_MISMATCH"
