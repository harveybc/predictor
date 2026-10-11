"""Lane B ledger: global identity, leases, fencing, attempts and recovery.

Tiny fixtures, no evaluator, no TensorFlow. Time is injected.
"""
import json

import pytest

from tools.i6n_swarm import common, ledger as L


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def spec(fold="F1", genotype="a" * 64, config="c" * 64, **extra):
    base = dict(stage="search", dag_node="search:h12", horizon=12, fold=fold, experiment="exp1",
                genotype_canonical_sha256=genotype, effective_config_sha256=config,
                search_candidate_sha256="d" * 64, device_class="rtx5090", fixture=True)
    base.update(extra)
    return base


def receipt_for(fit_id, s, value=0.5):
    return dict(fit_id=fit_id, fold=s["fold"], experiment=s["experiment"],
                effective_config_sha256=s["effective_config_sha256"],
                genotype_canonical_sha256=s["genotype_canonical_sha256"],
                shared_receipt=dict(status="ok", fitness=value), candidate_receipt=dict(seed=6102026))


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    led = L.Ledger(tmp_path / "ledger.sqlite3", clock=clock)
    s = spec()
    fit = common.fit_id(experiment="exp1", fold="F1", effective_config={"x": 1}, context={"c": 1}, support={"s": 1})
    led.ensure_unit(fit, s)
    return led, clock, fit, s, tmp_path


def test_claim_then_held_by_other_owner(env):
    led, clock, fit, _, _ = env
    first = led.claim(fit, owner="n1", host="hA", boot_id="b1", device_class="rtx5090", ttl=60)
    assert first["result"] == "CLAIMED" and first["attempt"]["index"] == 1
    assert first["attempt"]["kind"] == "INITIAL"
    second = led.claim(fit, owner="n2", host="hB", boot_id="b2", device_class="rtx4090", ttl=60)
    assert second["result"] == "HELD" and second["holder"]["owner"] == "n1"
    assert led.unit(fit)["state"] == "RUNNING"


def test_completed_fit_is_adopted_never_repeated(env):
    led, _, fit, s, _ = env
    c = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    assert led.complete(fit, c["token"], receipt_for(fit, s)) == "ACCEPTED"
    again = led.claim(fit, "n2", "hB", "b2", "rtx4090", 60)
    assert again["result"] == "ADOPT"
    assert again["terminal"]["receipt"]["shared_receipt"]["fitness"] == 0.5
    assert len(led.attempts(fit)) == 1
    assert led.unit(fit)["state"] == "COMPLETE"


def test_disconnect_during_claim_fences_old_token(env):
    led, clock, fit, _, _ = env
    old = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    clock.now += 61  # holder vanished before any renewal
    new = led.claim(fit, "n2", "hB", "b2", "rtx4090", 60)
    assert new["result"] == "CLAIMED"
    assert new["token"] > old["token"]
    assert new["attempt"]["kind"] == "RECOVERY" and new["attempt"]["index"] == 2
    assert new["attempt"]["parent_attempt"] == old["attempt"]["attempt_id"]
    assert led.attempts(fit)[0]["outcome"] == "INTERRUPTED"
    assert led.attempts(fit)[0]["cause"] == "LEASE_EXPIRED"
    assert led.renew(fit, old["token"], ttl=60) is False
    assert led.renew(fit, new["token"], ttl=60) is True


def test_renew_after_expiry_refused_so_holder_self_fences(env):
    led, clock, fit, _, _ = env
    c = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    clock.now += 30
    assert led.renew(fit, c["token"], ttl=60, progress={"epoch": 3}) is True
    clock.now += 61
    assert led.renew(fit, c["token"], ttl=60) is False


def test_after_fit_before_ack_late_receipt_adopted_and_supersedes(env):
    led, clock, fit, s, _ = env
    old = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    clock.now += 61
    new = led.claim(fit, "n2", "hB", "b2", "rtx4090", 60)
    # n1 had finished its fit inside its lease but the ACK was lost; it comes back.
    assert led.complete(fit, old["token"], receipt_for(fit, s, 0.25)) == "ACCEPTED_LATE"
    assert led.renew(fit, new["token"], ttl=60) is False  # recovery must self-fence
    assert led.complete(fit, new["token"], receipt_for(fit, s, 0.75)) == "DUPLICATE_DISCARDED"
    assert led.terminal(fit)["receipt"]["shared_receipt"]["fitness"] == 0.25
    outcomes = [a["outcome"] for a in led.attempts(fit)]
    assert outcomes == ["COMPLETE", "SUPERSEDED"]
    assert len(led.outbox_pending()) == 1


def test_ack_lost_after_ledger_commit_is_adopted(env):
    led, _, fit, s, _ = env
    c = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.complete(fit, c["token"], receipt_for(fit, s))
    # the DOIN broadcast was lost; another node claims the same candidate index
    assert led.claim(fit, "n2", "hB", "b2", "rtx4090", 60)["result"] == "ADOPT"
    assert led.complete(fit, c["token"], receipt_for(fit, s)) == "ALREADY"
    assert len(led.outbox_pending()) == 1


def test_receipt_of_another_candidate_is_rejected(env):
    led, _, fit, s, _ = env
    c = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    wrong = receipt_for(fit, s)
    wrong["effective_config_sha256"] = "e" * 64
    with pytest.raises(L.ReceiptMismatch):
        led.complete(fit, c["token"], wrong)
    other = receipt_for("f" * 64, s)
    with pytest.raises(L.ReceiptMismatch):
        led.complete(fit, c["token"], other)
    assert led.terminal(fit) is None


def test_one_recovery_without_checkpoint_then_failed(env):
    led, _, fit, _, _ = env
    a = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.report(fit, a["token"], "INTERRUPTED", "NODE_SHUTDOWN", parent_status_sha256="1" * 64)
    b = led.claim(fit, "n1", "hA", "b2", "rtx5090", 60)
    assert b["attempt"]["kind"] == "RECOVERY" and b["attempt"]["parent_status_sha256"] == "1" * 64
    led.report(fit, b["token"], "INTERRUPTED", "NODE_SHUTDOWN")
    c = led.claim(fit, "n1", "hA", "b3", "rtx5090", 60)
    assert c["result"] == "REFUSED" and c["cause"] == "RECOVERY_EXHAUSTED"
    assert led.unit(fit)["state"] == "FAILED"


def test_complete_checkpoint_resumes_and_three_physical_attempts_max(env):
    led, _, fit, _, _ = env
    ck = dict(optimizer=True, rng=True, epoch=4, stopping=True, path="/x/ck")
    a = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.report(fit, a["token"], "INTERRUPTED", "HOST_REBOOT", checkpoint=ck)
    b = led.claim(fit, "n1", "hA", "b2", "rtx5090", 60)
    assert b["attempt"]["kind"] == "RESUME" and b["attempt"]["checkpoint"]["epoch"] == 4
    led.report(fit, b["token"], "INTERRUPTED", "HOST_REBOOT", checkpoint=ck)
    c = led.claim(fit, "n1", "hA", "b3", "rtx5090", 60)
    assert c["attempt"]["index"] == 3
    led.report(fit, c["token"], "INTERRUPTED", "HOST_REBOOT", checkpoint=ck)
    d = led.claim(fit, "n1", "hA", "b4", "rtx5090", 60)
    assert d["result"] == "REFUSED" and d["cause"] == "ATTEMPTS_EXHAUSTED"


def test_incomplete_checkpoint_is_not_a_resume(env):
    led, _, fit, _, _ = env
    a = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.report(fit, a["token"], "INTERRUPTED", "HOST_REBOOT", checkpoint=dict(optimizer=True, epoch=4))
    assert led.claim(fit, "n1", "hA", "b2", "rtx5090", 60)["attempt"]["kind"] == "RECOVERY"


@pytest.mark.parametrize("outcome", ["FAILED_OOM", "FAILED_TIMEOUT", "FAILED"])
def test_terminal_failures_never_retry(env, outcome):
    led, _, fit, _, _ = env
    a = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.report(fit, a["token"], outcome, "cause-x")
    again = led.claim(fit, "n2", "hB", "b2", "rtx4090", 60)
    assert again["result"] == "REFUSED"
    assert led.unit(fit)["state"] == "FAILED"
    assert len(led.attempts(fit)) == 1


def test_boot_recovery_marks_previous_boot_interrupted(env):
    led, _, fit, s, _ = env
    other = common.sha256_json({"other": 1})
    led.ensure_unit(other, spec(fold="F2"))
    a = led.claim(fit, "n1", "hA", "boot-old", "rtx5090", 3600)
    led.claim(other, "n2", "hB", "boot-b", "rtx4090", 3600)
    changed = led.recover(host="hA", boot_id="boot-new")
    assert [c["fit_id"] for c in changed] == [fit]
    assert led.attempts(fit)[0]["cause"] == "HOST_REBOOT"
    assert led.unit(fit)["state"] == "INTERRUPTED"
    assert led.unit(other)["state"] == "RUNNING"
    assert led.renew(fit, a["token"], ttl=60) is False


def test_one_gpu_job_per_host_unless_admitted(env):
    led, _, fit, _, _ = env
    other = common.sha256_json({"other": 2})
    led.ensure_unit(other, spec(fold="F2"))
    led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    busy = led.claim(other, "n1b", "hA", "b1", "rtx5070ti", 60)
    assert busy["result"] == "WAIT_RESOURCE" and busy["cause"] == "HOST_GPU_BUSY"
    assert led.unit(other)["state"] == "WAIT_RESOURCE"
    ok = led.claim(other, "n1b", "hA", "b1", "rtx5070ti", 60, exclusive_host=False)
    assert ok["result"] == "CLAIMED"


def test_state_and_fencing_survive_restart(env):
    led, clock, fit, s, path = env
    a = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.report(fit, a["token"], "INTERRUPTED", "NODE_SHUTDOWN")
    reopened = L.Ledger(path / "ledger.sqlite3", clock=clock)
    b = reopened.claim(fit, "n1", "hA", "b2", "rtx5090", 60)
    assert b["token"] > a["token"] and b["attempt"]["index"] == 2
    reopened.complete(fit, b["token"], receipt_for(fit, s))
    third = L.Ledger(path / "ledger.sqlite3", clock=clock)
    assert third.unit(fit)["state"] == "COMPLETE"
    assert len(third.outbox_pending()) == 1


def test_outbox_delivery_is_durable_and_idempotent(env):
    led, _, fit, s, path = env
    c = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    led.complete(fit, c["token"], receipt_for(fit, s))
    item = led.outbox_pending()[0]
    assert item["payload"]["seed"] == 6102026 and item["payload"]["direction"] == "minimize"
    led.outbox_mark(item["id"], {"item_id": "x" * 64})
    led.outbox_mark(item["id"], {"item_id": "x" * 64})
    assert led.outbox_pending() == []
    assert L.Ledger(path / "ledger.sqlite3").outbox_pending() == []


def test_prior_receipt_adopted_by_identity_without_attempt(env):
    led, _, fit, s, _ = env
    assert led.adopt_prior(fit, receipt_for(fit, s, 0.9), source={"path": "/prior", "sha256": "9" * 64}) == "ADOPTED_PRIOR"
    res = led.claim(fit, "n1", "hA", "b1", "rtx5090", 60)
    assert res["result"] == "ADOPT" and res["terminal"]["source"] == "PRIOR"
    assert led.attempts(fit) == []


def test_readonly_open_cannot_write(env):
    led, _, fit, _, path = env
    ro = L.Ledger.open_readonly(path / "ledger.sqlite3")
    assert ro.unit(fit)["state"] == "QUEUED"
    with pytest.raises(Exception):
        ro.claim(fit, "n1", "hA", "b1", "rtx5090", 60)


def test_unit_spec_is_immutable(env):
    led, _, fit, s, _ = env
    led.ensure_unit(fit, s)  # idempotent
    changed = dict(s, horizon=72)
    with pytest.raises(ValueError):
        led.ensure_unit(fit, changed)


def test_fit_id_is_plan_and_directory_independent():
    a = common.fit_id(experiment="e", fold="F1", effective_config={"a": 1, "b": 2}, context={"k": 1}, support={"s": 1})
    b = common.fit_id(experiment="e", fold="F1", effective_config={"b": 2, "a": 1}, context={"k": 1}, support={"s": 1})
    c = common.fit_id(experiment="e", fold="F2", effective_config={"a": 1, "b": 2}, context={"k": 1}, support={"s": 1})
    assert a == b != c
    assert json.loads(common.canonical({"b": 1, "a": 2})) == {"a": 2, "b": 1}
