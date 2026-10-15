"""P6 (SC08/SC09/SC10/SC12 + prior-root adoption): orchestration on SCRATCH roots with the REAL sealed
designs and the REAL population/input pins (read only). Arm children are the lane's
orchestration-only fake (tests/i6dmo/ops_matched_fake.py, wrapped to sleep) because no fit may run;
the comparison is the REAL hook. systemctl is an injected recorder; the warehouse is the in-memory
fake. Nothing is deployed, no timer or service is touched.

  A. prior-root adoption: root A runs with FAKE fit/report entrypoints; root B is a real-entrypoint
     init with --prior-root A. Does B adopt A's fake terminals as its matched fits?
  B. SC09: worker SIGKILLed while its fit child runs; duplicate claim while held; recovery
     re-attaches/collects; outbox outage retried; post-CLOSE ticks/publishes/workers are inert.
     A COMPLETED fit is never repeated.
  C. SC12: a killed worker+child is STALE after its lease (never RUNNING indefinitely); a silent
     worker service is SERVICE_SILENT; a failed arm child is FAILED and the comparison closes
     INCOMPLETE_COMPARISON automatically.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 4G -t 1800s -n i6dmo-review -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p6_orchestration.py
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

from tests.i6dmo.ops_fakes import FakeWarehouse, facts, private_spec  # noqa: E402
from tools.i6dmo import campaign, matched_runner as MR, queue as Q, status, worker as W  # noqa: E402

WORK = C.SCRATCH / "p6"
FAKE = C.REPO / "tests/i6dmo/ops_matched_fake.py"
SLOW = C.REVIEW / "p6_slow_fake.py"
RUNTIME = {"tensorflow": "2.18.0", "keras": "3.9.0"}
CODE = "f" * 40
CALLS = []


def systemctl(*args):
    CALLS.append(list(args))
    return dict(rc=0, stdout="")


def provider(role):
    doc = facts()
    doc["runtime"] = dict(RUNTIME)
    return doc


def spec_path(base):
    base.mkdir(parents=True, exist_ok=True)
    spec = private_spec(base)
    for role in spec["roles"].values():
        role.update(data_root=str(C.INPUT.parent.parent), experiment_storage_root=str(C.storage_root()))
    path = base / "private_spec.json"
    path.write_text(json.dumps(spec))
    return path


def real_pins():
    rel = C.SEAL["private_location"].split("$EXPERIMENT_STORAGE_ROOT/", 1)[1]
    return dict(common_population=dict(base="experiment_storage", relpath=rel,
                                       sha256=C.sha(C.POPULATION / "POPULATION.json"), population_sha256=C.DIGEST),
                matched_input=dict(relpath=str(C.INPUT.relative_to(C.INPUT.parent.parent)),
                                   sha256=C.sha(C.INPUT / "INPUT.npz")))


def fake_entries(slow=False):
    base = [sys.executable, str(SLOW if slow else FAKE)]
    return dict(campaign.MATCHED_ENTRYPOINTS, fit=base + ["--kind", "fit"], report=base + ["--kind", "report"])


def init(base, name, *, entries, prior=()):
    root = base / name
    out = campaign.init_matched_contrast(C.DESIGNS, spec_path(base), root, code_commit=CODE, data_pins=real_pins(),
                                         entrypoints=entries, prior_roots=prior)
    return root, out


def worker(root, role, env=None):
    return W.worker_once(root, role, facts_provider=provider, executor=W.LocalExecutor(), poll_interval=0.05,
                         extra_env=dict(env or {}))


def drive(root, env=None, rounds=12):
    log = []
    for _ in range(rounds):
        campaign.tick(root, systemctl=systemctl)
        progressed = False
        for role in ("preferred", "secondary", "coordinator"):
            r = worker(root, role, env)
            log.append((role, r.get("result"), r.get("unit_id"), r.get("cause")))
            if r["result"] not in ("IDLE", "CLOSED", "WAIT_ARTIFACT", "WAIT_RESOURCE"):
                progressed = True
        if not progressed:
            break
    campaign.tick(root, systemctl=systemctl)
    return log


def states(root):
    q = campaign.open_queue(root)
    try:
        return {u["unit_id"]: [u["state"], u["cause"]] for u in q.units()}
    finally:
        q.close()


def counter_lines(path):
    return Path(path).read_text().splitlines() if Path(path).exists() else []


def deliver(root, attempts=4, fail_post=0):
    wh = FakeWarehouse(fail_post=fail_post)
    outs = []
    for _ in range(attempts):
        o = campaign.publish(root, warehouse=wh, systemctl=systemctl)
        outs.append((o.get("result"), o.get("close"), (o.get("publication") or {}).get("state")))
        if (root / "CLOSE.json").exists():
            break
    return outs, wh


def part_a(base):
    counter = base / "a_counter.txt"
    env = {"I6DMO_FAKE_COUNTER": str(counter)}
    a_root, a_out = init(base, "A_fake_entrypoints", entries=fake_entries())
    drive(a_root, env)
    deliver(a_root)
    res = {"A_states": states(a_root), "A_closed": (a_root / "CLOSE.json").exists(),
           "A_fake_children_invoked": counter_lines(counter)}
    sealed = {arm: {"seal_sha256": json.loads(C.DESIGNS[arm].read_text())["seal_sha256"]} for arm in ("mg1", "mg2m")}
    try:
        MR._check_donor(a_root, sealed, real_pins())
        res["prepare_check_donor_on_fake_root"] = "ACCEPTED"
    except MR.PrepareRefused as exc:
        res["prepare_check_donor_on_fake_root"] = f"REFUSED:{exc}"
    b_root, b_out = init(base, "B_real_entrypoints_prior_A", entries=None, prior=[a_root])
    eff_a = json.loads((a_root / "EFFECTIVE_CONFIG.json").read_text())
    eff_b = json.loads((b_root / "EFFECTIVE_CONFIG.json").read_text())
    res["identities_equal_A_vs_B"] = {u: x == y for u, x, y in zip(a_out["units"], *(
        [r["identity_sha256"] for r in json.loads((root / "PLAN.json").read_text())["units"]] for root in (a_root, b_root)))}
    res["entrypoints_differ_A_vs_B"] = eff_a["entrypoints"]["fit"] != eff_b["entrypoints"]["fit"]
    before = len(counter_lines(counter))
    log = drive(b_root, env, rounds=6)
    res["B_worker_log"] = log
    res["B_states"] = states(b_root)
    res["B_new_child_invocations"] = len(counter_lines(counter)) - before
    res["B_attempt_dirs"] = sorted(p.name for p in (b_root / "attempts").glob("*")) if (b_root / "attempts").exists() else []
    cmp = None
    q = campaign.open_queue(b_root)
    try:
        t = q.terminal("comparison")
        cmp = (t or {}).get("terminal", {}).get("disposition") if t else None
        res["B_comparison_source"] = (t or {}).get("source") if t else None
    finally:
        q.close()
    res["B_comparison_disposition"] = cmp
    adopted = [u for u, (s, c) in res["B_states"].items() if c == "ADOPTED"]
    res["B_adopted_units"] = adopted
    res["DEFECT_fake_fits_adopted_as_real"] = "mg1.fit" in adopted and res["B_new_child_invocations"] == 0
    return res


def _spawn_worker(root, role, env):
    code = ("import sys,json; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "import p6_orchestration as P; print(json.dumps(P.worker(%r, %r, %r)))") % (
        str(C.REPO), str(C.REVIEW), str(root), role, env)
    return subprocess.Popen([sys.executable, "-c", code], cwd=C.REPO, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, start_new_session=True)


def _running(root, unit_id):
    q = campaign.open_queue(root)
    try:
        return [a for a in q.running() if a["unit_id"] == unit_id]
    finally:
        q.close()


def _wait_for(pred, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.2)
    return False


def part_b(base):
    counter = base / "b_counter.txt"
    env = {"I6DMO_FAKE_COUNTER": str(counter), "RETSU_SLOW_SECONDS": "6"}
    root, _ = init(base, "C_sc09", entries=fake_entries(slow=True))
    campaign.tick(root, systemctl=systemctl)
    res = {}
    proc = _spawn_worker(root, "preferred", env)
    claimed = _wait_for(lambda: bool(_running(root, "mg1.fit") or _running(root, "mg2m.fit")))
    unit = "mg1.fit" if _running(root, "mg1.fit") else "mg2m.fit"
    time.sleep(1.0)
    os.killpg(proc.pid, signal.SIGKILL)  # the worker dies; the child has its own session and survives
    proc.wait()
    res["claimed_then_worker_killed"] = {"claimed": claimed, "unit": unit}
    q = Q.Queue(root / "state" / "queue.sqlite3")
    try:
        dup = q.claim(unit, owner="dup", role="secondary", host_key="host-secondary", boot_id="b",
                      device_class="rtx4090", heavy=True, ttl=600)
    finally:
        q.close()
    res["duplicate_claim_while_held"] = dup.get("result")
    st = status.compute(root, level="unit", unit=unit)
    res["status_unit_right_after_kill"] = st.get("state")
    rec = worker(root, "preferred", env)  # restart: recover_role re-attaches or collects the child
    res["restart_worker_result"] = rec.get("result")
    log = drive(root, env)
    res["drive_log_tail"] = log[-6:]
    res["delivery_with_outage"], wh = deliver(root, attempts=6, fail_post=3)
    res["closed"] = (root / "CLOSE.json").exists()
    fits_before = [line for line in counter_lines(counter) if line.startswith("fit:")]
    for _ in range(2):  # post-CLOSE ticks/publishes/workers are inert
        campaign.tick(root, systemctl=systemctl)
        campaign.publish(root, warehouse=wh, systemctl=systemctl)
        for role in ("preferred", "secondary", "coordinator"):
            worker(root, role, env)
    fits_after = [line for line in counter_lines(counter) if line.startswith("fit:")]
    q = campaign.open_queue(root)
    try:
        attempts = {u: [dict(idx=a["idx"], outcome=a["outcome"], cause=a["cause"]) for a in q.attempts(u)]
                    for u in ("mg1.fit", "mg2m.fit")}
    finally:
        q.close()
    res["fit_child_invocations"] = fits_after
    res["fit_attempts"] = attempts
    res["post_close_new_fits"] = len(fits_after) - len(fits_before)
    res["completed_fit_never_repeated"] = sorted(fits_after) == ["fit:mg1.fit", "fit:mg2m.fit"]
    res["timers_receipt"] = json.loads((root / "TIMERS_STOPPED.json").read_text()).get("state") \
        if (root / "TIMERS_STOPPED.json").exists() else None
    return res


def part_c(base):
    res = {}
    counter = base / "c_counter.txt"
    env = {"I6DMO_FAKE_COUNTER": str(counter), "RETSU_SLOW_SECONDS": "30"}
    root, _ = init(base, "D_sc12_stale", entries=fake_entries(slow=True))
    campaign.tick(root, systemctl=systemctl)
    proc = _spawn_worker(root, "preferred", env)
    _wait_for(lambda: bool(_running(root, "mg1.fit") or _running(root, "mg2m.fit")))
    unit = "mg1.fit" if _running(root, "mg1.fit") else "mg2m.fit"
    time.sleep(1.0)
    attempt = _running(root, unit)[0]
    child_pid = (attempt.get("launch") or {}).get("pid")
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    if child_pid:
        try:
            os.killpg(child_pid, signal.SIGKILL)  # host crash: worker and child both gone
        except ProcessLookupError:
            pass
    now = time.time()
    arm = unit.split(".")[0]
    s_now = status.compute(root, level="arm", arm=arm, now=now)
    s_late = status.compute(root, level="arm", arm=arm, now=now + W.TTL + 5)
    c_late = status.compute(root, level="campaign", now=now + 1801)
    res["killed_unit"] = unit
    res["arm_state_now"] = s_now["arms"][arm]["state"]
    res["arm_state_after_lease"] = s_late["arms"][arm]["state"]
    res["campaign_state_after_1801s"] = c_late.get("state")
    res["campaign_failures_after_1801s"] = sorted({f["kind"] for f in (c_late.get("failures") or [])})
    rec = W.recover_role(root, "preferred", executor=W.LocalExecutor())
    res["recover_after_crash"] = rec
    # a failed arm child: FAILED, never RUNNING; the comparison closes INCOMPLETE_COMPARISON
    counter2 = base / "c2_counter.txt"
    root2, _ = init(base, "E_failed_arm", entries=fake_entries())
    env2 = {"I6DMO_FAKE_COUNTER": str(counter2), "I6DMO_FAKE_FAIL_UNIT": "mg1.fit"}
    drive(root2, env2)
    deliver(root2)
    res["failed_arm_states"] = states(root2)
    a2 = status.compute(root2, level="arm")
    res["failed_arm_status"] = {a: v["state"] for a, v in a2["arms"].items()}
    close = json.loads((root2 / "CLOSE.json").read_text()) if (root2 / "CLOSE.json").exists() else None
    res["failed_arm_close"] = None if close is None else {"state": close.get("state"),
                                                         "disposition": close.get("disposition"),
                                                         "trigger": close.get("trigger")}
    return res


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    out = {"A_prior_root_adoption": part_a(WORK / "a"), "B_sc09": part_b(WORK / "b"), "C_sc12": part_c(WORK / "c"),
           "systemctl_calls_recorded": CALLS,
           "note": "fake arm children (no fit); REAL designs, REAL population/input pins, REAL comparison hook; "
                   "systemctl injected; no deploy"}
    C.write("p6_orchestration.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
