"""R4: pure scratch-SQLite queue checks; run with --noconftest (no TensorFlow)."""
import sys

import pytest

from tools.i6dmo.queue import MAX_PHYSICAL_ATTEMPTS, Queue

UNIT = "mg1.report"
REFUSAL = "LAUNCHER_ADMISSION_REFUSED"


@pytest.fixture
def queue(tmp_path):
    q = Queue(tmp_path / "queue.sqlite", clock=lambda: 100.0)
    q.ensure_unit(UNIT, dict(kind="report", device="cpu", identity_sha256="a" * 64), state="QUEUED")
    yield q
    q.close()


def claim(q):
    return q.claim(UNIT, owner="fixture", role="coordinator", host_key="fixture",
                   boot_id="fixture", device_class="cpu", heavy=False, ttl=60)


@pytest.mark.parametrize("cause", [REFUSAL, "INFRA_PREFIT:INPUT_PIN_MISMATCH"])
def test_preexecution_refusals_preserve_history_and_keep_claiming(queue, cause):
    previous = []
    for index in range(1, MAX_PHYSICAL_ATTEMPTS + 4):
        got = claim(queue)
        assert got["result"] == "CLAIMED", got
        assert got["attempt"]["index"] == index
        assert queue.attempts(UNIT)[:-1] == previous
        assert queue.interrupt(got["token"], cause)
        previous = queue.attempts(UNIT)
        assert [a["idx"] for a in previous] == list(range(1, index + 1))
        assert all(a["outcome"] == "INTERRUPTED" and a["cause"] == cause for a in previous)
    assert queue.terminal(UNIT) is None
    assert queue.unit(UNIT)["state"] == "INTERRUPTED"


@pytest.mark.parametrize("cause", ["SUPERVISOR_CRASH", "POST_TRAINING_FAILURE",
                                   REFUSAL + ":UNPROVEN", "prefix:" + REFUSAL])
def test_real_or_unproven_interruptions_still_exhaust(queue, cause):
    for index in range(1, MAX_PHYSICAL_ATTEMPTS + 1):
        got = claim(queue)
        assert got["result"] == "CLAIMED"
        assert got["attempt"]["index"] == index
        assert queue.interrupt(got["token"], cause)
    before = queue.attempts(UNIT)
    assert claim(queue) == dict(result="REFUSED", cause="ATTEMPTS_EXHAUSTED")
    assert queue.attempts(UNIT) == before
    assert queue.unit(UNIT)["state"] == "FAILED"


def test_admission_refusal_does_not_consume_remaining_crash_budget(queue):
    first = claim(queue)
    assert queue.interrupt(first["token"], REFUSAL)
    for index in range(1, MAX_PHYSICAL_ATTEMPTS + 1):
        got = claim(queue)
        assert got["result"] == "CLAIMED", got
        assert got["attempt"]["index"] == index + 1
        assert queue.interrupt(got["token"], "SUPERVISOR_CRASH")
    assert claim(queue) == dict(result="REFUSED", cause="ATTEMPTS_EXHAUSTED")


@pytest.mark.parametrize("outcome", ["FAILED", "FAILED_OOM", "FAILED_TIMEOUT"])
def test_terminal_failures_are_not_retried_even_with_refusal_cause(queue, outcome):
    got = claim(queue)
    queue.fail(UNIT, got["token"], outcome, REFUSAL)
    before = queue.attempts(UNIT)
    assert claim(queue) == dict(result="REFUSED", cause=f"{outcome}:{REFUSAL}")
    assert queue.attempts(UNIT) == before
    assert queue.unit(UNIT)["state"] == "FAILED"


def test_completed_terminal_is_adopted_without_another_attempt(queue):
    got = claim(queue)
    queue.complete(UNIT, got["token"], dict(status="COMPLETE"), "b" * 64,
                   identity_sha256="a" * 64)
    before = queue.attempts(UNIT)
    terminal = queue.terminal(UNIT)
    assert claim(queue) == dict(result="ADOPT", terminal_sha256="b" * 64)
    assert queue.attempts(UNIT) == before
    assert queue.terminal(UNIT) == terminal


def test_no_tensorflow_loaded():
    assert "tensorflow" not in sys.modules
