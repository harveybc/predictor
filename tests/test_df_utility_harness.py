"""M2–M4: the utility harness is causal by construction and its budgets are observed.

Requirements this module binds (each rule names the one it proves):
  R1 protocol validated and sealed; branches, target/model pairs, family, blocks policy
  R2 representations built by the REAL operator, fit on each block's train only
  R3 features consumed only when emitted at or before the row's decision instant
  R4 identity by observation id; discordant ids/times refused; gaps kept
  R5 eligibility from the verified matrix's cell record, never a flag
  R6 a future-reading control is refused (out of the record AND caught by the prefix check)
  R7 scaler and fit inside the training block only
  R8 scipy t over blocks; all-or-insufficient; ADVANCES only under a calibration record
  R9 budgets enforced in an isolated child; RESOURCE_EXCEEDED with cost, no partial score
  R10 the reserved holdout is adjudicated once per (reserve identity, protocol)
Nothing here scores project data or any reserve.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


H = _load("df_utility_harness")
ops = _load("df_d3_operators")
contract = _load("df_d3_contract")

FAMILY = ("fab/v0/mad_extremes_trailing/transformed", "fab/v0/mad_extremes_trailing/augmented",
          "fab/v0/delta_run_length/transformed")


def proto(**over):
    base = dict(target="return", horizon=1, model="ridge", window=4, n_blocks=4, margin=0.0,
                seed=7, family=FAMILY, min_rows_per_block=30, calibration_plan=dict(FIXTURE_PLAN))
    base.update(over)
    return H.Protocol(**base)


#: fixture plan: the exchangeable null, a LOW bound confidence so 14 simulations can support a
#: decision in a test; a real freeze predeclares 0.95 and the simulations it implies
FIXTURE_PLAN = {"generator": "white_null", "n": 2400, "bound_confidence": 0.5,
                "n_sims": H.sims_required_for_zero(0.05 / 3, 0.5)}       # 42 for this family
_RECORDS = {}


def calibrated(p=None, operator=None):
    """A protocol carrying a real calibration record for `operator` (default MAD), computed
    once per operator and reused; the record is bound to the fixture protocol's base."""
    p = p or proto()
    operator = operator or MAD
    key = (operator.KIND, p.base_sha256())
    if key not in _RECORDS:
        _RECORDS[key] = H.calibrate(p, operator, plan=FIXTURE_PLAN, seed=11)
    return p.with_calibration(_RECORDS[key])


def record_for(*operators, unit="fab", variable="v0", verdict="MECHANICALLY_ACCEPTED"):
    cells = {}
    for op in operators:
        cells[(unit, variable, op.KIND)] = {"verdict": verdict,
                                            "spec_sha256": contract.spec_sha256(op.describe())}
    return {"freeze_sha256": "f" * 64, "design_sha256": "d" * 64, "cells": cells}


def fabricated(n=2400, seed=3, truth="extreme"):
    """A series whose next increment depends on how extreme the last value is relative to a
    trailing median (truth='extreme': the mad-extremes score carries it) or on nothing."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        drive = 0.0
        if truth == "extreme" and t > 20:
            window = x[t - 16:t]
            med = np.median(window)
            mad = np.median(np.abs(window - med)) or 1.0
            drive = -0.6 * np.sign(x[t - 1] - med) * min(3.0, abs(x[t - 1] - med) / mad)
        x[t] = x[t - 1] + drive + rng.normal(0, 1.0)
    return x


MAD = ops.build("mad_extremes_trailing")
DELTA = ops.build("delta_run_length")


# --- R1 protocol ------------------------------------------------------------------------------------------

def test_R1_the_protocol_validates_domains_pairs_family_and_seals():
    doc = proto().sealed()
    assert doc["comparisons"] == 3 and doc["alpha_adjusted"] == pytest.approx(0.05 / 3)
    with pytest.raises(H.ProtocolRefusal, match="pairs with model"):
        proto(target="direction", model="ridge")
    with pytest.raises(H.ProtocolRefusal, match="n_blocks"):
        proto(n_blocks=2)
    with pytest.raises(H.ProtocolRefusal, match="branches"):
        proto(branches=("raw", "nope"))
    with pytest.raises(H.ProtocolRefusal, match="family"):
        proto(family=("a", "a"))
    with pytest.raises(H.ProtocolRefusal, match="calibration"):
        proto(calibration={"n_sims": 1})


def test_R1_an_undeclared_branch_or_contrast_is_refused_before_scoring():
    s = H.series(fabricated())
    out = H.contrast(s, MAD, calibrated(), contrast_id=FAMILY[1], eligibility=record_for(MAD),
                     unit="fab", variable="v0", branch_b="augmented")
    assert out["outcome"] == H.REFUSED and "not declared" in out["why"]
    out = H.contrast(s, MAD, calibrated(), contrast_id="not/in/family", eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.REFUSED and "sealed family" in out["why"]


# --- R2/R3/R4 identity, emission alignment -------------------------------------------------------------

def test_R4_discordant_ids_or_times_are_refused():
    with pytest.raises(ValueError, match="strictly increasing"):
        H.series([1.0, 2.0, 3.0], ids=[1, 1, 2])
    with pytest.raises(ValueError, match="backwards"):
        H.series([1.0, 2.0, 3.0], timestamps=[2, 1, 3])
    with pytest.raises(ValueError, match="available before"):
        H.series([1.0, 2.0], timestamps=[1, 2], available_at=[0, 2])


def test_R3_an_output_emitted_after_the_decision_is_used_only_at_a_later_row():
    values = np.arange(10, dtype=float)
    available = np.ones(10, dtype=bool)
    emitted = np.arange(10, dtype=float) + 2          # every output is two samples late
    decision = np.arange(10, dtype=float)
    X, ok = H._lags_by_emission(values, available, emitted, decision, window=2)
    assert ok[4] and list(X[4]) == [2.0, 1.0]          # at t=4 the newest usable output is i=2
    assert not ok[1]


def test_R4_repeated_values_with_distinct_ids_are_distinct_rows():
    x = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 600)
    s = H.series(x, ids=np.arange(2400) * 10)
    p = calibrated(operator=DELTA)
    out = H.contrast(s, DELTA, p, contrast_id=FAMILY[2], eligibility=record_for(DELTA),
                     unit="fab", variable="v0")
    assert out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE)
    ids = [b["validation_ids"] for b in out["blocks"]]
    assert all(a[1] < b[0] for a, b in zip(ids, ids[1:]))     # blocks by id, non-overlapping


# --- R5/R6 eligibility and the future-reading control ---------------------------------------------------

class FutureLabelOperator(ops.DeltaRunLength):
    """The reviewer's control: emits x[t+1] - x[t] as if at t, declaring itself causal."""
    KIND = "future_label_control"

    def transform(self, x, state):
        v = np.asarray(x["values"], dtype=float)
        out = np.zeros(v.size)
        out[:-1] = v[1:] - v[:-1]
        avail = np.ones(v.size, dtype=bool)
        avail[-1] = False
        avail[0] = False                              # honours its declared warm-up of 1
        return self._pack(x, out.tolist(), avail.tolist())


def test_R5_R6_the_future_label_control_is_refused_out_of_the_record_and_by_the_prefix_check():
    s = H.series(fabricated(truth="noise"))
    p = calibrated()
    fut = FutureLabelOperator()
    out = H.contrast(s, fut, p, contrast_id=FAMILY[2], eligibility=record_for(DELTA),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.REFUSED and "not eligible" in out["why"]
    # even with a forged eligibility entry, the prefix check catches the leak before scoring
    forged = record_for(fut)
    out = H.contrast(s, fut, p, contrast_id=FAMILY[2], eligibility=forged, unit="fab", variable="v0")
    assert out["outcome"] == H.REFUSED and "not causal" in out["why"]


def test_R5_eligibility_is_bound_to_the_exact_cell_and_declaration():
    s = H.series(fabricated())
    p = calibrated()
    other_cell = record_for(MAD, unit="other")
    out = H.contrast(s, MAD, p, contrast_id=FAMILY[0], eligibility=other_cell, unit="fab", variable="v0")
    assert out["outcome"] == H.REFUSED and "not in the verified record" in out["why"]
    refused = record_for(MAD, verdict="INCONCLUSIVE")
    out = H.contrast(s, MAD, p, contrast_id=FAMILY[0], eligibility=refused, unit="fab", variable="v0")
    assert out["outcome"] == H.REFUSED and "INCONCLUSIVE" in out["why"]
    stale = record_for(MAD)
    next(iter(stale["cells"].values()))["spec_sha256"] = "0" * 64
    out = H.contrast(s, MAD, p, contrast_id=FAMILY[0], eligibility=stale, unit="fab", variable="v0")
    assert out["outcome"] == H.REFUSED and "declaration" in out["why"]


def test_R5_eligibility_record_must_come_from_a_verified_cells_file(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"schema": "d3_mechanics_cells.v1", "verified": False,
                                                 "freeze_sha256": "f", "design_sha256": "d", "cells": []}))
    with pytest.raises(ValueError, match="VERIFIED"):
        H.eligibility_record(tmp_path / "c.json")


# --- R2/R7 real operator, train-only fit, tail change ---------------------------------------------------

def test_R7_the_scaler_and_the_fit_see_only_the_training_block():
    Xtr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    a, b = H._standardise(Xtr, np.array([[100.0, 100.0]]))
    a2, b2 = H._standardise(Xtr, np.array([[-100.0, 0.0]]))
    assert np.allclose(a, a2)                                   # train rows unchanged by validation


def test_R2_a_tail_change_does_not_change_the_prefix_features_of_a_causal_operator():
    x = fabricated()
    s1 = H.series(x)
    x2 = x.copy()
    x2[1800:] = x2[1800:] + 50.0
    s2 = H.series(x2)
    r1 = H.represent(MAD, s1, 900)
    r2 = H.represent(MAD, s2, 900)
    assert np.array_equal(r1["values"][:1800], r2["values"][:1800])
    assert np.array_equal(r1["emitted_at"][:1800], r2["emitted_at"][:1800])
    ok, where = H.prefix_consistent(MAD, s1, r1["state"], r1, [500, 1200], s1["available_at"])
    assert ok and where is None


def test_R2_restart_by_checkpoint_gives_the_same_features_for_a_stateful_operator():
    cusum = ops.build("cusum_causal")
    x = fabricated(n=600)
    s = H.series(x)
    whole = H.represent(cusum, s, 300)
    state = cusum.fit(H._as_operator_input(s, upto=300))
    head = cusum.transform(H._as_operator_input(s, upto=400), state)
    resumed = cusum.resume(cusum.checkpoint())
    tail_x = {"values": [float(v) for v in x[400:]], "timestamps": list(range(400, 600)),
              "available_at": list(range(400, 600)), "period_seconds": 1.0}
    tail = cusum.transform(tail_x, resumed)
    assert np.allclose(np.r_[head["values"], tail["values"]], whole["values"])


def test_R2_a_legitimate_historical_lag_representation_is_accepted_and_may_advance():
    s = H.series(fabricated(truth="extreme"))
    p = calibrated()
    out = H.contrast(s, MAD, p, contrast_id=FAMILY[0], eligibility=record_for(MAD), unit="fab", variable="v0")
    assert out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE), out
    assert out["representation"]["operator"] == "mad_extremes_trailing"
    assert out["blocks_used"] == 4 and out["cost"]["cpu_seconds"] > 0


def test_R2_a_noise_truth_does_not_advance_negative_control():
    s = H.series(fabricated(truth="noise"))
    out = H.contrast(s, MAD, calibrated(), contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.DOES_NOT_ADVANCE and abs(out["delta_mean"]) < 0.3


# --- R8 inference -------------------------------------------------------------------------------------------

def test_R8_the_t_quantile_is_scipys_and_low_df_tails_are_wide():
    from scipy.stats import t as student_t
    mean, se, t_crit, lower = H.t_interval_lower(np.array([0.1, 0.3]), 0.05)
    assert t_crit == pytest.approx(float(student_t.ppf(0.975, 1))) == pytest.approx(12.706204736174694)
    _, _, t3, _ = H.t_interval_lower(np.array([0.1, 0.3, 0.2, 0.4]), 0.05 / 3)
    assert t3 == pytest.approx(float(student_t.ppf(1 - 0.05 / 6, 3)))


def test_R8_without_a_supporting_record_the_result_is_descriptive_never_a_decision():
    s = H.series(fabricated(truth="extreme"))
    out = H.contrast(s, MAD, proto(), contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED and "delta_mean" in out
    assert "no calibration record" in out["why"]


def test_N2_a_record_with_zero_simulations_nan_rate_or_unknown_generator_is_refused():
    rec = calibrated().calibration
    for bad in ({"n_sims": 0, "scored": 0, "failed": 0, "per_sim": []},
                {"false_advance_rate": float("nan")},
                {"false_advance_rate": float("inf")},
                {"false_advance_rate": -0.1},
                {"upper_bound": float("nan")},
                {"generator": "not-a-measured-null"},
                {"generator": "ar1_null", "null": False},
                {"scored": 0, "failed": rec["n_sims"]},
                {"failed": 3},
                {"per_sim_sha256": "0" * 64}):
        with pytest.raises(H.ProtocolRefusal):
            proto().with_calibration({**rec, **bad})


def test_N2_a_record_transferred_to_another_operator_protocol_length_or_family_does_not_decide():
    s = H.series(fabricated(truth="extreme"))
    p = calibrated()                                    # MAD's record
    out = H.contrast(s, DELTA, p, contrast_id=FAMILY[2], eligibility=record_for(DELTA),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED and "another operator" in out["why"]
    other = H.Protocol(**{**p.__dict__, "margin": 0.01})          # another protocol, same record
    out = H.contrast(s, MAD, other, contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED and "another protocol" in out["why"]
    short = H.series(fabricated(n=1800, truth="extreme"))
    out = H.contrast(short, MAD, p, contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED and "length" in out["why"]
    moved = H.Protocol(**{**p.__dict__, "family": FAMILY[:2]})
    out = H.contrast(s, MAD, moved, contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED


def test_N2_the_decision_is_gated_on_the_upper_bound_not_the_point_estimate():
    rec = dict(calibrated().calibration)
    assert rec["advances"] == 0 and rec["false_advance_rate"] == 0.0
    assert rec["upper_bound"] == pytest.approx(H.clopper_pearson_upper(0, rec["scored"], 0.5))
    assert rec["upper_bound"] <= proto().alpha_adjusted            # the fixture plan supports it
    # the same simulations under a plan sealed at 95%: the point estimate is still 0, the
    # derived bound is not — no decision (O1: the bound is derived, the plan is the protocol's)
    p95 = proto(calibration_plan={**FIXTURE_PLAN, "bound_confidence": 0.95})
    strict = {**rec, "bound_confidence": 0.95, "plan": {**rec["plan"], "bound_confidence": 0.95},
              "upper_bound": H.clopper_pearson_upper(0, rec["scored"], 0.95),
              "protocol_base_sha256": p95.base_sha256()}
    p = p95.with_calibration(strict)
    assert strict["upper_bound"] > p.alpha_adjusted
    s = H.series(fabricated(truth="extreme"))
    out = H.contrast(s, MAD, p, contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INCONCLUSIVE_UNCALIBRATED and "upper bound" in out["why"]
    assert H.sims_required_for_zero(0.05 / 4, 0.95) == 239


def test_N2_every_simulation_is_kept_and_the_rate_can_be_recounted():
    rec = calibrated().calibration
    assert len(rec["per_sim"]) == rec["n_sims"] == rec["scored"] + rec["failed"]
    recount = sum(1 for x in rec["per_sim"] if x["outcome"] == "ADVANCES")
    assert recount == rec["advances"]
    assert rec["operator"]["kind"] == "mad_extremes_trailing" and rec["n"] == 2400
    assert rec["protocol_base_sha256"] == proto().base_sha256()
    assert rec["harness_sha256"] and rec["cost"]["cpu_seconds"] > 0


def test_N2_the_dependent_null_with_an_independent_target_is_a_null_and_ar1_is_not():
    assert H.GENERATORS["ar1_features_independent_target"]["null"] is True
    assert H.GENERATORS["ar1_null"]["null"] is False
    rec = H.calibrate(proto(), DELTA, generator="ar1_features_independent_target", n_sims=4,
                      seed=2, n=900, bound_confidence=0.5)
    assert rec["null"] is True and rec["scored"] + rec["failed"] == 4
    with pytest.raises(H.ProtocolRefusal, match="null of no effect"):
        proto(calibration_plan={"generator": "ar1_null", "n_sims": 5, "n": 900,
                                "bound_confidence": 0.95})


def test_N2_protocol_numeric_domains_are_validated():
    for bad in (dict(margin=float("nan")), dict(margin=float("inf")), dict(margin=-1.0),
                dict(alpha=float("nan")), dict(ridge_lambda=-1.0), dict(seed=1.5),
                dict(calibration_plan={"generator": "white_null", "n_sims": 0, "n": 100,
                                       "bound_confidence": 0.95})):
        with pytest.raises(H.ProtocolRefusal):
            proto(**bad)


def test_R8_a_short_block_makes_the_whole_contrast_insufficient():
    s = H.series(fabricated(n=150))
    out = H.contrast(s, MAD, calibrated(), contrast_id=FAMILY[0], eligibility=record_for(MAD),
                     unit="fab", variable="v0")
    assert out["outcome"] == H.INSUFFICIENT_ROWS and out["policy"] == "all_or_insufficient"


# --- R9 observed budgets in an isolated child -----------------------------------------------------------

def _job(s, p, **over):
    rec = record_for(MAD)
    job = {"contrast_id": FAMILY[0], "unit": "fab", "variable": "v0", "operator": MAD.KIND,
           "protocol": p.sealed(), "series": {"values": s["values"].tolist()},
           "eligibility_inline": {"freeze_sha256": rec["freeze_sha256"],
                                  "design_sha256": rec["design_sha256"],
                                  "cells": [{"key": list(k), "cell": v} for k, v in rec["cells"].items()]}}
    job.update(over)
    return job


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_R9_a_slow_model_exhausts_the_observed_budget_with_cost_and_no_partial_score(tmp_path):
    s = H.series(fabricated())
    out = H.run_isolated(_job(s, calibrated(), slow_seconds=30.0), attempt_dir=tmp_path / "slow",
                         assigned_bytes=1 << 30, wall_seconds=4.0, cpu_seconds=4)
    assert out["outcome"] == H.RESOURCE_EXCEEDED and out["score"] is None
    assert "LIMIT" in out["reason"] and out["cost"]["wall_seconds"] >= 3.5
    assert not (tmp_path / "slow" / "contrast.json").exists()


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_R9_a_contrast_within_budget_completes_with_measured_cost_and_a_verified_output(tmp_path):
    s = H.series(fabricated())
    out = H.run_isolated(_job(s, calibrated()), attempt_dir=tmp_path / "ok",
                         assigned_bytes=1 << 30, wall_seconds=300.0, cpu_seconds=300)
    assert out["outcome"] in (H.ADVANCES, H.DOES_NOT_ADVANCE), out
    assert out["cost"]["cpu_seconds"] > 0 and out["cost"]["peak_rss_bytes"] > 0
    assert out["output_sha256"] and out["score"]["outcome"] == out["outcome"]


# --- R10 the reserve --------------------------------------------------------------------------------------

def test_R10_the_reserve_is_adjudicated_once_per_identity_and_protocol(tmp_path):
    p = calibrated()
    calls = []
    out = H.adjudicate_holdout({"campaign_sha256": "a" * 64}, p, lambda: calls.append(1) or {"ok": 1},
                               state_dir=tmp_path)
    assert out == {"ok": 1}
    with pytest.raises(SystemExit, match="second look"):
        H.adjudicate_holdout({"campaign_sha256": "a" * 64}, p, lambda: calls.append(2), state_dir=tmp_path)
    # a different reserve or a different protocol is a different adjudication; a nameless
    # reserve is refused
    H.adjudicate_holdout({"campaign_sha256": "b" * 64}, p, lambda: calls.append(3), state_dir=tmp_path)
    with pytest.raises(SystemExit, match="identified by"):
        H.adjudicate_holdout({}, p, lambda: None, state_dir=tmp_path)
    assert calls == [1, 3]


# --- N1: the score is the verified output file, never the process summary ---------------------------

@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_N1_the_parents_score_is_the_verified_output_file_and_a_resume_never_reruns(tmp_path):
    s = H.series(fabricated())
    out = H.run_isolated(_job(s, calibrated()), attempt_dir=tmp_path / "ok",
                         assigned_bytes=1 << 30, wall_seconds=300.0, cpu_seconds=300)
    on_disk = json.loads((tmp_path / "ok" / "contrast.json").read_text())
    assert out["score"]["delta_mean"] == on_disk["delta_mean"] != 0.0
    assert out["score"]["schema"] == H.CONTRAST_SCHEMA and out["outcome"] == on_disk["outcome"]
    summary = json.loads((tmp_path / "ok" / "result.json").read_text())
    assert "delta_mean" not in summary                     # the summary never carried it
    again = H.run_isolated(_job(s, calibrated()), attempt_dir=tmp_path / "ok",
                           assigned_bytes=1 << 30, wall_seconds=300.0, cpu_seconds=300)
    assert again["resumed"] is True and again["score"]["delta_mean"] == on_disk["delta_mean"]
    assert again["cost"] == out["cost"]                    # the recorded outcome, not a rerun


def test_N1_a_missing_altered_or_discordant_output_is_a_typed_refusal_with_no_score(tmp_path):
    job = {"contrast_id": "c", "protocol": {"protocol_sha256": "p" * 64}}
    body = json.dumps({"schema": H.CONTRAST_SCHEMA, "contrast_id": "c", "protocol_sha256": "p" * 64,
                       "outcome": H.DOES_NOT_ADVANCE, "delta_mean": 0.1, "delta_se": 0.01,
                       "delta_lower": 0.05}).encode()
    digest = __import__("hashlib").sha256(body).hexdigest()
    result = {"output_file": "contrast.json", "output_sha256": digest, "outcome": H.DOES_NOT_ADVANCE}
    (tmp_path / "contrast.json").write_bytes(body)
    score, refusal = H.verified_score(tmp_path, result, {"output_sha256": digest}, job)
    assert refusal is None and score["delta_mean"] == 0.1
    (tmp_path / "contrast.json").write_bytes(body.replace(b"0.1", b"0.9"))
    score, refusal = H.verified_score(tmp_path, result, {"output_sha256": digest}, job)
    assert score is None and refusal["outcome"] == H.SCORE_UNVERIFIED and "bytes" in refusal["why"]
    (tmp_path / "contrast.json").unlink()
    score, refusal = H.verified_score(tmp_path, result, {"output_sha256": digest}, job)
    assert score is None and "absent" in refusal["why"]
    (tmp_path / "contrast.json").write_bytes(body)
    score, refusal = H.verified_score(tmp_path, result, {"output_sha256": digest},
                                      {"contrast_id": "other", "protocol": {"protocol_sha256": "p" * 64}})
    assert score is None and "identity" in refusal["why"]
    nan_body = body.replace(b"0.1", b"NaN")
    (tmp_path / "contrast.json").write_bytes(nan_body)
    d2 = __import__("hashlib").sha256(nan_body).hexdigest()
    score, refusal = H.verified_score(tmp_path, dict(result, output_sha256=d2), {"output_sha256": d2}, job)
    assert score is None and "not finite" in refusal["why"]


@pytest.mark.skipif(not Path("/usr/bin/systemd-run").exists(), reason="no systemd user scope")
def test_N3_a_preparatory_calibration_child_completes_with_its_record_as_the_verified_output(tmp_path):
    """The rehearsal v5 fell on this: a calibration record carries no 'outcome'."""
    p = proto()
    job = {"kind": "calibrate", "contrast_id": FAMILY[0], "operator": DELTA.KIND,
           "protocol": p.sealed(), "plan": {"generator": "white_null", "n_sims": 2, "n": 300,
                                            "bound_confidence": 0.5}, "seed": 5}
    out = H.run_isolated(job, attempt_dir=tmp_path / "cal", assigned_bytes=1 << 30,
                         wall_seconds=300.0, cpu_seconds=300)
    assert out["outcome"] == "COMPLETED" and out["score"]["schema"] == "df_utility_calibration.v1"
    assert out["score"]["n_sims"] == 2 and out["output_sha256"]
    assert (tmp_path / "cal" / "outcome.json").is_file()


# --- O1: the bound is re-derived from the simulations, never read ------------------------------------

def _resealed(rec, **over):
    """A record with overrides and its per_sim digest recomputed (the tamper an insider could do)."""
    out = {**rec, **over}
    out["per_sim_sha256"] = H.sha_obj(out["per_sim"])
    return out


def test_O1_a_provided_bound_is_never_used_the_bound_is_rederived_from_the_simulations():
    p = calibrated()
    rec = dict(p.calibration)
    d = H.derive_calibration(rec)
    assert d["problems"] == [] and d["advances"] == rec["advances"] and d["scored"] == rec["scored"]
    assert d["upper_bound"] == pytest.approx(rec["upper_bound"])
    for bad in ({"upper_bound": 0.0}, {"upper_bound": rec["upper_bound"] / 2},
                {"advances": 0, "false_advance_rate": 0.0} if rec["advances"] else
                {"advances": 1, "false_advance_rate": 1 / rec["scored"]}):
        tampered = {**rec, **bad}
        assert H.derive_calibration(tampered)["problems"]
        with pytest.raises(H.ProtocolRefusal):
            proto().with_calibration(tampered)
        assert H.calibration_supports(H.Protocol(**{**p.__dict__, "calibration": None}), MAD, 2400,
                                      record=tampered)[0] is False


def test_O1_counts_changed_with_per_sim_intact_are_refused_even_with_a_valid_digest():
    rec = dict(calibrated().calibration)
    one_advance = [dict(x, outcome="ADVANCES", delta_lower=0.5, delta_mean=0.6) if i == 0 else x
                   for i, x in enumerate(rec["per_sim"])]
    honest = _resealed(rec, per_sim=one_advance, advances=1, false_advance_rate=1 / rec["scored"],
                       upper_bound=H.clopper_pearson_upper(1, rec["scored"], rec["bound_confidence"]))
    assert H.calibration_record_problems(honest) == []
    hidden = _resealed(honest, advances=0, false_advance_rate=0.0,
                       upper_bound=H.clopper_pearson_upper(0, rec["scored"], rec["bound_confidence"]))
    problems = H.calibration_record_problems(hidden)
    assert problems and any("advances" in x for x in problems)


def test_O1_every_simulation_is_validated_label_delta_indices_seeds_and_finiteness():
    rec = dict(calibrated().calibration)
    sims = rec["per_sim"]

    def with_sim(i, **over):
        return _resealed(rec, per_sim=[dict(x, **over) if k == i else x for k, x in enumerate(sims)])

    label_vs_delta = with_sim(0, outcome="ADVANCES")           # delta_lower <= margin, says ADVANCES
    assert any("consistent" in x for x in H.calibration_record_problems(label_vs_delta))
    label_vs_delta2 = with_sim(0, delta_lower=0.7)              # delta_lower > margin, says DOES_NOT
    assert any("consistent" in x for x in H.calibration_record_problems(label_vs_delta2))
    dup_index = with_sim(1, index=0)
    assert any("index" in x for x in H.calibration_record_problems(dup_index))
    dup_seed = with_sim(1, seed=sims[0]["seed"])
    assert any("seed" in x for x in H.calibration_record_problems(dup_seed))
    nan_delta = with_sim(2, delta_lower=float("nan"))
    assert any("finite" in x for x in H.calibration_record_problems(nan_delta))
    unknown_label = with_sim(2, outcome="MAYBE")
    assert any("outcome" in x for x in H.calibration_record_problems(unknown_label))
    missing_field = _resealed(rec, per_sim=[{k: v for k, v in x.items() if k != "seed"} if i == 3 else x
                                           for i, x in enumerate(sims)])
    assert H.calibration_record_problems(missing_field)
    short = _resealed(rec, per_sim=sims[:-1])                   # partial denominator
    assert H.calibration_record_problems(short)


def test_O1_a_plan_or_confidence_discordant_with_the_protocol_does_not_decide():
    p = calibrated()
    rec = dict(p.calibration)
    other_conf = {**rec, "bound_confidence": 0.3, "plan": {**rec["plan"], "bound_confidence": 0.3},
                  "upper_bound": H.clopper_pearson_upper(rec["advances"], rec["scored"], 0.3)}
    bare = H.Protocol(**{**p.__dict__, "calibration": None})
    ok, why = H.calibration_supports(bare, MAD, 2400, record=other_conf)
    assert ok is False and "plan" in why
    inner = {**rec, "bound_confidence": 0.3}                    # record disagrees with its own plan
    assert any("plan" in x for x in H.calibration_record_problems(inner))
    other_margin = {**rec, "margin": 0.01}
    ok, why = H.calibration_supports(bare, MAD, 2400, record=other_margin)
    assert ok is False
    assert H.calibration_supports(bare, MAD, 2400, record=rec)[0] is True


def test_O1_the_failure_policy_is_declared_and_failed_simulations_count_against_the_operator():
    rec = dict(calibrated().calibration)
    assert H.FAILURE_POLICY == "WORST_CASE_FAILED_COUNTED_AS_ADVANCES"
    failed_one = _resealed(rec, per_sim=[{"index": x["index"], "seed": x["seed"],
                                          "outcome": H.INSUFFICIENT_ROWS, "why": "short block"}
                                         if i == 0 else x for i, x in enumerate(rec["per_sim"])],
                           scored=rec["scored"] - 1, failed=1,
                           false_advance_rate=rec["advances"] / (rec["scored"] - 1),
                           upper_bound=H.clopper_pearson_upper(rec["advances"], rec["scored"] - 1,
                                                               rec["bound_confidence"]))
    assert H.calibration_record_problems(failed_one) == []
    d = H.derive_calibration(failed_one)
    assert d["failed"] == 1 and d["failure_policy"] == H.FAILURE_POLICY
    assert d["decision_bound"] == pytest.approx(
        H.clopper_pearson_upper(rec["advances"] + 1, rec["n_sims"], rec["bound_confidence"]))
    assert d["decision_bound"] > d["upper_bound"]
    p = proto()
    ok, why = H.calibration_supports(p, MAD, 2400, record=failed_one)
    assert ok is False and "worst case" in why


def test_O1_a_record_from_another_harness_code_does_not_decide():
    p = calibrated()
    rec = {**p.calibration, "harness_sha256": "e" * 64}
    bare = H.Protocol(**{**p.__dict__, "calibration": None})
    ok, why = H.calibration_supports(bare, MAD, 2400, record=rec)
    assert ok is False and "harness" in why
    ok, why = H.calibration_supports(bare, MAD, 2400, record=rec, harness_sha256="e" * 64)
    assert ok is True                                          # the code that applied, declared


# --- O2: a resumed attempt has a single truthful outcome -------------------------------------------

def _attempt(path, body, *, summary_outcome, job, digest=None):
    import hashlib
    path.mkdir(parents=True, exist_ok=True)
    digest = digest or hashlib.sha256(body).hexdigest()
    (path / "contrast.json").write_bytes(body)
    (path / "job.json").write_text(json.dumps({**job, "attempt_dir": str(path)}))
    (path / "result.json").write_text(json.dumps({"status": "COMPLETED", "reason": "",
                                                  "output_file": "contrast.json",
                                                  "output_sha256": digest, "outcome": summary_outcome}))
    (path / "outcome.json").write_text(json.dumps({"status": "COMPLETED",
                                                   "verified": {"output_sha256": digest},
                                                   "summary": {"outcome": summary_outcome, "reason": "",
                                                               "cost": {"cpu_seconds": 1.0},
                                                               "output_sha256": digest}}))


def test_O2_a_resumed_attempt_whose_evidence_fails_verification_reports_only_SCORE_UNVERIFIED(tmp_path):
    job = {"contrast_id": "c", "protocol": {"protocol_sha256": "p" * 64}, "kind": "contrast"}
    body = json.dumps({"schema": H.CONTRAST_SCHEMA, "contrast_id": "c", "protocol_sha256": "p" * 64,
                       "outcome": H.ADVANCES, "delta_mean": 0.3, "delta_se": 0.01,
                       "delta_lower": 0.2}).encode()
    run = lambda d, j=job: H.run_isolated(j, attempt_dir=d, assigned_bytes=1 << 20,
                                          wall_seconds=1, cpu_seconds=1)
    _attempt(tmp_path / "good", body, summary_outcome=H.ADVANCES, job=job)
    out = run(tmp_path / "good")
    assert out["outcome"] == H.ADVANCES and out["score"]["delta_mean"] == 0.3 and out["resumed"]
    cases = {}
    _attempt(tmp_path / "absent", body, summary_outcome=H.ADVANCES, job=job)
    (tmp_path / "absent" / "contrast.json").unlink()
    cases["absent"] = run(tmp_path / "absent")
    _attempt(tmp_path / "bytes", body, summary_outcome=H.ADVANCES, job=job)
    (tmp_path / "bytes" / "contrast.json").write_bytes(body.replace(b"0.3", b"0.9"))
    cases["bytes"] = run(tmp_path / "bytes")
    _attempt(tmp_path / "id", body, summary_outcome=H.ADVANCES, job=job)
    cases["id"] = run(tmp_path / "id", dict(job, contrast_id="other"))
    _attempt(tmp_path / "proto", body, summary_outcome=H.ADVANCES, job=job)
    cases["proto"] = run(tmp_path / "proto", dict(job, protocol={"protocol_sha256": "q" * 64}))
    garbage = b"{not json"
    _attempt(tmp_path / "parse", garbage, summary_outcome=H.ADVANCES, job=job)
    cases["parse"] = run(tmp_path / "parse")
    for name, out in cases.items():
        assert out["outcome"] == H.SCORE_UNVERIFIED, name
        assert out["score"] is None and out["resumed"] is True, name
        assert out["refusal"]["outcome"] == H.SCORE_UNVERIFIED and out["refusal"]["why"], name
        assert out["history"]["outcome"] == H.ADVANCES, name      # kept as history only
        assert "delta_mean" not in json.dumps({k: v for k, v in out.items() if k != "history"})


def test_O2_a_resumed_attempt_is_bound_to_the_recorded_job(tmp_path):
    job = {"contrast_id": "c", "protocol": {"protocol_sha256": "p" * 64}, "kind": "contrast",
           "operator": "mad_extremes_trailing", "unit": "u", "variable": "v0"}
    body = json.dumps({"schema": H.CONTRAST_SCHEMA, "contrast_id": "c", "protocol_sha256": "p" * 64,
                       "outcome": H.DOES_NOT_ADVANCE, "delta_mean": -0.1, "delta_se": 0.01,
                       "delta_lower": -0.2}).encode()
    _attempt(tmp_path / "a", body, summary_outcome=H.DOES_NOT_ADVANCE, job=job)
    same = H.run_isolated(dict(job), attempt_dir=tmp_path / "a", assigned_bytes=1, wall_seconds=1, cpu_seconds=1)
    assert same["outcome"] == H.DOES_NOT_ADVANCE
    changed = H.run_isolated(dict(job, operator="cusum_causal"), attempt_dir=tmp_path / "a",
                             assigned_bytes=1, wall_seconds=1, cpu_seconds=1)
    assert changed["outcome"] == H.SCORE_UNVERIFIED and "job" in changed["refusal"]["why"]
    (tmp_path / "a" / "job.json").unlink()
    unbound = H.run_isolated(dict(job), attempt_dir=tmp_path / "a", assigned_bytes=1, wall_seconds=1, cpu_seconds=1)
    assert unbound["outcome"] == H.SCORE_UNVERIFIED and "job" in unbound["refusal"]["why"]


def test_O2_a_resumed_preparatory_record_is_revalidated_against_the_job(tmp_path):
    p = proto()
    rec = calibrated(p).calibration
    body = json.dumps(rec, sort_keys=True).encode()
    job = {"kind": "calibrate", "contrast_id": FAMILY[0], "operator": MAD.KIND, "protocol": p.sealed(),
           "plan": rec["plan"], "seed": rec["seed"]}
    d = tmp_path / "cal"
    _attempt(d, body, summary_outcome="COMPLETED", job=job)
    (d / "calibration.json").write_bytes(body)
    (d / "result.json").write_text(json.dumps({"status": "COMPLETED", "reason": "", "output_file": "calibration.json",
                                               "output_sha256": __import__("hashlib").sha256(body).hexdigest()}))
    ok = H.run_isolated(dict(job), attempt_dir=d, assigned_bytes=1, wall_seconds=1, cpu_seconds=1)
    assert ok["outcome"] == "COMPLETED" and ok["score"]["upper_bound"] == rec["upper_bound"]
    other = H.run_isolated(dict(job, operator=DELTA.KIND), attempt_dir=d, assigned_bytes=1, wall_seconds=1, cpu_seconds=1)
    assert other["outcome"] == H.SCORE_UNVERIFIED and other["score"] is None
    tampered = {**rec, "upper_bound": 0.0}
    tb = json.dumps(tampered, sort_keys=True).encode()
    d2 = tmp_path / "cal2"
    _attempt(d2, tb, summary_outcome="COMPLETED", job=job)
    (d2 / "calibration.json").write_bytes(tb)
    (d2 / "result.json").write_text(json.dumps({"status": "COMPLETED", "reason": "", "output_file": "calibration.json",
                                                "output_sha256": __import__("hashlib").sha256(tb).hexdigest()}))
    bad = H.run_isolated(dict(job), attempt_dir=d2, assigned_bytes=1, wall_seconds=1, cpu_seconds=1)
    assert bad["outcome"] == H.SCORE_UNVERIFIED and bad["score"] is None and "bound" in bad["refusal"]["why"]
