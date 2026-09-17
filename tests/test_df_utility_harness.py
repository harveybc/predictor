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
    strict = {**rec, "bound_confidence": 0.95,
              "upper_bound": H.clopper_pearson_upper(0, rec["scored"], 0.95)}
    p = proto().with_calibration(strict)
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
