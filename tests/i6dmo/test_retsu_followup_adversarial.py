"""Retsu follow-up review (9f8bd67a): adversarial regressions for R4 (launcher-admission accounting),
R2 (matched dispatch fairness) and R3 (strict scientific defaults).

Real queue, real worker (claim -> governed child -> _finish -> stored terminal), real reporting;
the arm children are ``retsu_adversarial_child.py`` (orchestration only: no fit, no data read).
The three ``xfail(strict=True)`` refutations of the review (C1 x2, C3) are fixed and now pass as
ordinary tests; the observations that recorded the old behaviour assert the corrected one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.i6dmo.test_matched_fit_path import world  # noqa: F401  (fixture)
from tests.i6dmo.test_matched_scientific_defaults import report_case, upstreams  # noqa: F401  (fixtures)
from tests.i6dmo.test_ops_matched_runner import _metrics, make_matched, states
from tests.i6dmo.ops_fakes import facts
from tools.i6dmo import campaign, design as D, reporting, status, worker as W
from tools.i6dmo.queue import MAX_PHYSICAL_ATTEMPTS, Queue

CHILD = Path(__file__).with_name("retsu_adversarial_child.py")
REFUSAL = "LAUNCHER_ADMISSION_REFUSED"


def entries():
    fake = [sys.executable, str(CHILD.with_name("ops_matched_fake.py")), "--kind", "report"]
    return dict(campaign.MATCHED_ENTRYPOINTS, fit=[sys.executable, str(CHILD)], report=fake)


def probe(role):
    doc = facts()
    doc["runtime"] = dict(D.MATCHED_RUNTIME)
    return doc


def run_worker(root, role, env):
    return W.worker_once(root, role, facts_provider=probe, executor=W.LocalExecutor(), poll_interval=0.02,
                         extra_env=env)


@pytest.fixture
def campaign_root(tmp_path):
    root = make_matched(tmp_path, entries=entries())
    campaign.tick(root, systemctl=lambda *a: dict(rc=0, stdout=""))
    return root


def env_for(tmp_path, **modes):
    env = {"RETSU_COUNTER": str(tmp_path / "counter.txt")}
    for unit, mode in modes.items():
        env["RETSU_MODE_" + unit] = mode
    return env


def invocations(tmp_path, unit):
    path = tmp_path / "counter.txt"
    return [ln for ln in (path.read_text().splitlines() if path.exists() else []) if ln.startswith(unit + ":")]


def attempts(root, unit):
    q = campaign.open_queue(root)
    try:
        return [(a["idx"], a["outcome"], a["cause"]) for a in q.attempts(unit)]
    finally:
        q.close()


def unit_state(root, unit):
    q = campaign.open_queue(root)
    try:
        row = q.unit(unit)
        return row["state"], row["cause"]
    finally:
        q.close()


def drive_preferred(root, env, rounds):
    return [run_worker(root, "preferred", env).get("result") for _ in range(rounds)]


# =========================================================================== R4 (worker level)

def test_r4_genuine_crash_after_start_is_terminal_and_never_retried(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="crash", mg2m_fit="crash")
    results = drive_preferred(campaign_root, env, MAX_PHYSICAL_ATTEMPTS + 2)
    assert results[0] == "FAILED"
    assert len(invocations(tmp_path, "mg1.fit")) == 1
    assert unit_state(campaign_root, "mg1.fit")[0] == "FAILED"
    assert attempts(campaign_root, "mg1.fit") == [(1, "FAILED", "EXIT_1")]


def test_r4_timeout_is_terminal_and_never_retried(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="timeout", mg2m_fit="timeout")
    drive_preferred(campaign_root, env, MAX_PHYSICAL_ATTEMPTS + 2)
    assert len(invocations(tmp_path, "mg1.fit")) == 1
    assert attempts(campaign_root, "mg1.fit") == [(1, "FAILED_TIMEOUT", "rc=124")]
    assert unit_state(campaign_root, "mg1.fit")[0] == "FAILED"


def test_r4_genuine_refusals_keep_history_and_do_not_spend_budget(campaign_root, tmp_path):
    n = MAX_PHYSICAL_ATTEMPTS + 2
    env = env_for(tmp_path, mg1_fit=",".join(["refuse75"] * n + ["crash"]), mg2m_fit="refuse75")
    for _ in range(2 * (n + 2)):  # C3: proven refusals rotate the arms, so mg1 is re-picked every other round
        run_worker(campaign_root, "preferred", env)
        if unit_state(campaign_root, "mg1.fit")[0] == "FAILED":
            break
    rows = [r for r in attempts(campaign_root, "mg1.fit")]
    assert [r[0] for r in rows] == list(range(1, len(rows) + 1))          # indices contiguous, preserved
    assert all(r[1:] == ("INTERRUPTED", REFUSAL) for r in rows[:-1])
    assert rows[-1][1:] == ("FAILED", "EXIT_1")                              # the real crash still ends it


def test_r4_started_then_75_must_spend_budget_and_terminate(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="started_then_75", mg2m_fit="refuse75")
    for _ in range(MAX_PHYSICAL_ATTEMPTS + 3):
        run_worker(campaign_root, "preferred", env)
    assert len(invocations(tmp_path, "mg1.fit")) <= MAX_PHYSICAL_ATTEMPTS
    assert unit_state(campaign_root, "mg1.fit")[0] in ("FAILED", "COMPLETE")


def test_r4_claims_not_started_with_outputs_must_spend_budget(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="claims_not_started", mg2m_fit="refuse75")
    for _ in range(MAX_PHYSICAL_ATTEMPTS + 3):
        run_worker(campaign_root, "preferred", env)
    assert len(invocations(tmp_path, "mg1.fit")) <= MAX_PHYSICAL_ATTEMPTS
    assert unit_state(campaign_root, "mg1.fit")[0] == "FAILED"


def test_r4_started_then_75_is_a_spent_failure_not_a_refusal(campaign_root, tmp_path):
    """C1 (was the 'unbounded today' observation): exit 75 after the child left output is a failure."""
    env = env_for(tmp_path, mg1_fit="started_then_75", mg2m_fit="refuse75")
    for _ in range(2 * MAX_PHYSICAL_ATTEMPTS):
        run_worker(campaign_root, "preferred", env)
    rows = attempts(campaign_root, "mg1.fit")
    assert len(invocations(tmp_path, "mg1.fit")) == len(rows) == 1
    assert rows[0][1:] == ("FAILED", "EXIT_75") and REFUSAL not in {r[2] for r in rows}
    assert unit_state(campaign_root, "mg1.fit")[0] == "FAILED"


# =========================================================================== R4 (queue, mixed sequences)

@pytest.fixture
def queue(tmp_path):
    clock = {"t": 100.0}
    q = Queue(tmp_path / "q.sqlite", clock=lambda: clock["t"])
    q.ensure_unit("u", dict(kind="fit", device="gpu", identity_sha256="a" * 64), state="QUEUED")
    yield q, clock
    q.close()


def _claim(q):
    return q.claim("u", owner="o", role="r", host_key="h", boot_id="b", device_class="c", heavy=False, ttl=60)


@pytest.mark.parametrize("real", [["CHILD_VANISHED", "SUPERVISOR_LOST", "LEASE_EXPIRED"],
                                  ["LEASE_EXPIRED", "NOT_LAUNCHED", "LAUNCH_FAILED"]])
def test_r4_mixed_sequences_still_exhaust_on_real_interruptions(queue, real):
    q, clock = queue
    history = []
    for cause in [REFUSAL, real[0], REFUSAL, REFUSAL, real[1], "INFRA_PREFIT:NO_GPU_VISIBLE@p", real[2]]:
        got = _claim(q)
        assert got["result"] == "CLAIMED", (cause, got)
        assert got["attempt"]["index"] == len(history) + 1
        if cause == "LEASE_EXPIRED":
            clock["t"] += 120                     # the lease lapses; the next claim interrupts it itself
        else:
            q.interrupt(got["token"], cause)
        if cause != "LEASE_EXPIRED":
            history = [(a["idx"], a["outcome"], a["cause"]) for a in q.attempts("u")]
            assert [h[0] for h in history] == list(range(1, len(history) + 1))
        else:
            history.append((len(history) + 1, "RUNNING", None))
    final = _claim(q)
    assert final == dict(result="REFUSED", cause="ATTEMPTS_EXHAUSTED"), final
    rows = q.attempts("u")
    assert [a["idx"] for a in rows] == list(range(1, 8))
    assert [a["cause"] for a in rows].count(REFUSAL) == 3
    assert q.unit("u")["state"] == "FAILED"


def test_r4_refusal_after_exhaustion_cannot_reopen(queue):
    q, _ = queue
    for _ in range(MAX_PHYSICAL_ATTEMPTS):
        q.interrupt(_claim(q)["token"], "SUPERVISOR_LOST")
    assert _claim(q)["cause"] == "ATTEMPTS_EXHAUSTED"
    assert _claim(q) == dict(result="REFUSED", cause="ATTEMPTS_EXHAUSTED")
    assert len(q.attempts("u")) == MAX_PHYSICAL_ATTEMPTS


def test_r4_persistent_refusal_is_typed_admission_blocked_after_the_threshold(campaign_root, tmp_path):
    """C3 (was the 'no typed status' observation): a persistent launcher refusal surfaces typed."""
    env = env_for(tmp_path, mg1_fit="refuse75", mg2m_fit="refuse75")
    drive_preferred(campaign_root, env, 3)
    doc = status.compute(campaign_root, level="campaign", now=10 ** 10)
    blocked = [f for f in doc.get("failures") or [] if f["kind"] == "ADMISSION_BLOCKED"]
    assert blocked and {f["cause"] for f in blocked} == {REFUSAL}
    assert "INFRA_PREFIT_BLOCKED" not in {f["kind"] for f in doc["failures"]}
    early = status.compute(campaign_root, level="campaign")
    assert not [f for f in early["failures"] if f["kind"] == "ADMISSION_BLOCKED"]  # bounded threshold


# =========================================================================== R2

def test_r2_refused_arm_does_not_starve_the_healthy_arm_on_the_real_worker(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="infra_prefit", mg2m_fit="fake")
    results = drive_preferred(campaign_root, env, 2)
    assert results[0] == "WAIT_RESOURCE"
    assert unit_state(campaign_root, "mg2m.fit")[0] == "COMPLETE", (results, states(campaign_root))
    assert len(invocations(tmp_path, "mg1.fit")) == 1


def test_r2_alternates_under_persistent_refusal_of_both_arms(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="infra_prefit", mg2m_fit="infra_prefit")
    drive_preferred(campaign_root, env, 6)
    a, b = len(invocations(tmp_path, "mg1.fit")), len(invocations(tmp_path, "mg2m.fit"))
    assert abs(a - b) <= 1 and a + b == 6


def test_r2_post_training_failure_is_terminal_and_not_refusal_rotation(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="infra_after_fit", mg2m_fit="fake")
    drive_preferred(campaign_root, env, 3)
    assert unit_state(campaign_root, "mg1.fit") == ("FAILED", "FAILED:TYPED:NO_GPU_VISIBLE")
    assert attempts(campaign_root, "mg1.fit") == [(1, "FAILED", "TYPED:NO_GPU_VISIBLE")]
    assert len(invocations(tmp_path, "mg1.fit")) == 1
    assert unit_state(campaign_root, "mg2m.fit")[0] == "COMPLETE"


def test_r2_admission_refusals_rotate_and_are_harmless(campaign_root, tmp_path):
    """C3: a proven LAUNCHER_ADMISSION_REFUSED now counts in the fairness rank like INFRA_PREFIT (it
    was 'do not rotate' at 9f8bd67a): the other arm runs next, then the refused arm completes."""
    env = env_for(tmp_path, mg1_fit="refuse75,fake", mg2m_fit="fake")
    drive_preferred(campaign_root, env, 1)
    assert attempts(campaign_root, "mg1.fit")[0][2] == REFUSAL
    drive_preferred(campaign_root, env, 1)
    assert unit_state(campaign_root, "mg2m.fit")[0] == "COMPLETE"
    drive_preferred(campaign_root, env, 1)
    assert unit_state(campaign_root, "mg1.fit")[0] == "COMPLETE"


# =========================================================================== R3

STRICT_EXTRA = ["false", "False", "0", 0.0, np.False_, [], {}]


def _pair():
    return [dict(m, matched=dict(m["matched"], scientific_fit=True, fit_device="gpu"))
            for m in (_metrics("mg1", -0.02), _metrics("mg2m", -0.01))]


@pytest.mark.parametrize("flag", STRICT_EXTRA)
def test_r3_only_literal_false_opts_out_of_the_decision(flag):
    pair = _pair()
    pair[0]["matched"].update(scientific_fit=False, fit_device="cpu")
    with pytest.raises(reporting.ReportFailure, match="NON_SCIENTIFIC_FIT"):
        reporting.decide_matched(*pair, require_scientific=flag)


@pytest.mark.parametrize("value", ["true", 1, "gpu", None])
def test_r3_truthy_lookalikes_are_not_scientific(value):
    pair = _pair()
    pair[1]["matched"]["scientific_fit"] = value
    with pytest.raises(reporting.ReportFailure, match="NON_SCIENTIFIC_FIT"):
        reporting.decide_matched(*pair)


def test_r3_literal_false_still_opts_out():
    pair = _pair()
    pair[0]["matched"].update(scientific_fit=False, fit_device="cpu")
    assert reporting.decide_matched(*pair, require_scientific=False)["decision"] in ("LOWER_SCORE",
                                                                                  "TIE_RETAIN_CHEAPER")


@pytest.mark.parametrize("flag", ["false", "False", 0.0, np.False_, None, 0])
def test_r3_real_report_only_literal_false_opts_out(report_case, flag):
    result, metrics, _ = report_case(False, "cpu", flag)
    assert result["status"] == "FAILED" and result["disposition"] == "NON_SCIENTIFIC_FIT", (flag, result)
    assert metrics is None


def test_r3_real_report_json_false_opts_out(report_case):
    result, metrics, _ = report_case(False, "cpu", json.loads("false"))
    assert result["status"] == "COMPLETE" and metrics["matched"]["require_scientific_fit"] is False


def test_r2_persistent_launcher_refusal_of_one_arm_must_not_starve_the_other(campaign_root, tmp_path):
    env = env_for(tmp_path, mg1_fit="refuse75", mg2m_fit="fake")
    drive_preferred(campaign_root, env, 4)
    assert unit_state(campaign_root, "mg2m.fit")[0] == "COMPLETE"


def test_r2_launcher_refusal_rotates_arms_and_never_lowers_the_cap(campaign_root, tmp_path):
    """C3 (was the 'starves today' observation): the refused arm yields; its cap is never lowered."""
    env = env_for(tmp_path, mg1_fit="refuse75", mg2m_fit="fake")
    drive_preferred(campaign_root, env, 4)
    assert invocations(tmp_path, "mg2m.fit") and len(invocations(tmp_path, "mg1.fit")) < 4
    q = campaign.open_queue(campaign_root)
    try:
        caps = {a["cap_bytes"] for a in q.attempts("mg1.fit")}
    finally:
        q.close()
    assert len(caps) == 1  # every refused attempt asked for the same cap
