"""Matched short contrast runner (NEXT_CONTRASTS_PLAN_20261015 section 4; SC08/SC09/SC10/SC12).

Real: the sealed matched designs (``matched_short/EXPERIMENT.mg1.json`` / ``EXPERIMENT.mg2m.json``)
through the real model-lane validator, campaign init/identity, tick, governed worker, stored-terminal
verification, the REAL comparison hook through ``unit_runner --kind comparison``, outbox with exact
readback, mirrors, automatic CLOSE, status, the ``python -m tools.i6dmo matched-short`` entrypoint and,
for preparation, a REAL common population built by the data lane from the synthetic fixture.

ORCHESTRATION ONLY: the arm fit/report children are :mod:`tests.i6dmo.ops_matched_fake` until the
model lane's matched adapters land; the reviewer runs those. The real-adapter path is proven here
only up to its typed handshake (MATCHED_*_ADAPTER_UNAVAILABLE: nothing is claimed, nothing is fitted).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.i6dmo.ops_fakes import FakeWarehouse, facts, private_spec

REPO = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).with_name("ops_matched_fake.py")
MATCHED = REPO / "docs/i6n/direct_multi_output_static/matched_short"
DESIGNS = {"mg1": MATCHED / "EXPERIMENT.mg1.json", "mg2m": MATCHED / "EXPERIMENT.mg2m.json"}
RUNTIME = {"tensorflow": "2.18.0", "keras": "3.9.0"}
POPULATION = "b" * 64


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def host(tmp_path):
    """Private spec whose roles resolve a data root and an experiment storage root (fixture files)."""
    tmp_path = Path(tmp_path)
    data, storage = tmp_path / "data", tmp_path / "storage"
    (data / "input").mkdir(parents=True, exist_ok=True)
    (storage / "population").mkdir(parents=True, exist_ok=True)
    if not (data / "input" / "INPUT.npz").exists():
        (data / "input" / "INPUT.npz").write_bytes(b"fixture-input")
        (storage / "population" / "POPULATION.json").write_text(json.dumps(dict(schema="fixture")))
    spec = private_spec(tmp_path)
    for role in spec["roles"].values():
        role.update(data_root=str(data), experiment_storage_root=str(storage))
    path = tmp_path / "private_spec.json"
    path.write_text(json.dumps(spec))
    return path, data, storage


def pins(data, storage, **overrides):
    out = dict(common_population=dict(base="experiment_storage", relpath="population",
                                      sha256=_sha(storage / "population" / "POPULATION.json"),
                                      population_sha256=POPULATION),
               matched_input=dict(relpath="input", sha256=_sha(data / "input" / "INPUT.npz")))
    out.update(overrides)
    return out


def entrypoints():
    from tools.i6dmo import campaign
    base = [sys.executable, str(FAKE)]
    return dict(campaign.MATCHED_ENTRYPOINTS, fit=base + ["--kind", "fit"], report=base + ["--kind", "report"])


def make_matched(tmp_path, *, name="matched", designs=None, data_pins=None, prior_roots=(), code_commit="c" * 40,
                 entries="fakes"):
    from tools.i6dmo import campaign
    spec_path, data, storage = host(tmp_path)
    root = Path(tmp_path) / name
    campaign.init_matched_contrast(designs or DESIGNS, spec_path, root, adapter=None, code_commit=code_commit,
                                   data_pins=data_pins or pins(data, storage),
                                   entrypoints=entrypoints() if entries == "fakes" else entries,
                                   prior_roots=prior_roots)
    return root


def provider(runtime=None, per_role=None):
    def probe(role):
        doc = facts()
        doc["runtime"] = dict((per_role or {}).get(role) or runtime or RUNTIME)
        return doc
    return probe


def worker(root, role, env=None, **kw):
    from tools.i6dmo import worker as W
    return W.worker_once(root, role, facts_provider=kw.pop("facts_provider", provider()),
                         executor=kw.pop("executor", W.LocalExecutor()), poll_interval=0.02,
                         extra_env=dict(env or {}), **kw)


def drive(root, env=None, rounds=10, **kw):
    from tools.i6dmo import campaign
    for _ in range(rounds):
        campaign.tick(root)
        progressed = False
        for role in ("preferred", "secondary", "coordinator"):
            if worker(root, role, env, **kw)["result"] not in ("IDLE", "CLOSED", "WAIT_ARTIFACT", "WAIT_RESOURCE"):
                progressed = True
        if not progressed:
            break
    return campaign.tick(root)


def states(root):
    from tools.i6dmo import campaign
    q = campaign.open_queue(root)
    try:
        return {u["unit_id"]: u["state"] for u in q.units()}
    finally:
        q.close()


def comparison(root):
    from tools.i6dmo import campaign
    q = campaign.open_queue(root)
    try:
        term = q.terminal("comparison")
    finally:
        q.close()
    if term is None:
        return None
    return json.loads((root / "terminals" / term["identity_sha256"] / "out" / "COMPARISON.json").read_text())


def fits(counter):
    counter = Path(counter)
    return [line for line in counter.read_text().splitlines() if line.startswith("fit:")] if counter.exists() else []


# ------------------------------------------------------------------------------- the graph
def test_graph_is_two_sealed_matched_arms_and_one_comparison_without_test(tmp_path):
    from tools.i6dmo import campaign
    root = make_matched(tmp_path)
    plan = campaign.load_plan(root)
    units = {u["unit_id"]: u for u in plan["units"]}
    assert list(units) == ["mg1.fit", "mg1.report", "mg2m.fit", "mg2m.report", "comparison"]
    assert units["comparison"]["depends_on"] == ["mg1.report", "mg2m.report"]
    assert units["mg1.fit"]["resource_class"] == "short_mg1_matched"
    assert units["mg2m.fit"]["resource_class"] == "short_mg2_matched"
    assert units["mg1.fit"]["identity_sha256"] != units["mg2m.fit"]["identity_sha256"]
    assert plan["mode"] == "MATCHED_SHORT_CONTRAST" and plan["test_units_enumerated"] is False
    assert all(u["test_open"] is False and "test" not in u["windows"] for u in plan["units"])
    with pytest.raises(campaign.Refused, match="TEST_NOT_ENUMERABLE_IN_MATCHED_CONTRAST"):
        campaign.enumerate_test_units(root)
    effective = campaign.load_effective(root)
    assert effective["comparison"]["tolerance"] == 1e-8 and effective["comparison"]["tie_keeps"] == "mg2m"
    # the sealed identities are carried verbatim (seals 4786951a / 6e8689d4 unchanged)
    assert effective["arms"]["mg1"]["design"]["seal_sha256"].startswith("4786951a")
    assert effective["arms"]["mg2m"]["design"]["seal_sha256"].startswith("6e8689d4")


def test_init_refuses_closed_relabelled_test_open_or_unfrozen_inputs(tmp_path):
    from tools.i6dmo import campaign
    closed_mg2 = REPO / "docs/i6n/direct_multi_output_static/EXPERIMENT.mg2.json"
    with pytest.raises(campaign.Refused, match="MATCHED_IDENTITY_REQUIRED: mg2m"):
        make_matched(tmp_path, name="a", designs=dict(DESIGNS, mg2m=closed_mg2))
    with pytest.raises(campaign.Refused, match="MATCHED_IDENTITY_REQUIRED: mg1 design declares arm mg2m"):
        make_matched(tmp_path, name="b", designs=dict(mg1=DESIGNS["mg2m"], mg2m=DESIGNS["mg2m"]))
    doc = json.loads(DESIGNS["mg1"].read_text())
    doc["splits"]["test"]["open"] = True
    bad = tmp_path / "open.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(campaign.Refused, match="TEST_OPEN_IN_DESIGN: mg1"):
        make_matched(tmp_path, name="c", designs=dict(DESIGNS, mg1=bad))
    doc = json.loads(DESIGNS["mg1"].read_text())
    doc["training"]["patience"] = 11  # a changed sealed design is refused by the real validator
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(doc))
    with pytest.raises(campaign.Refused):
        make_matched(tmp_path, name="d", designs=dict(DESIGNS, mg1=changed))
    spec_path, data, storage = host(tmp_path)
    unfrozen = pins(data, storage)
    unfrozen.pop("common_population")
    with pytest.raises(campaign.Refused, match="UNFROZEN_COMPARISON"):
        make_matched(tmp_path, name="e", data_pins=unfrozen)
    assert not any((tmp_path / n / "PLAN.json").exists() for n in "abcde")


# ------------------------------------------------------------------- decision and SC08 close
def test_full_contrast_decides_publishes_mirrors_and_closes_automatically(tmp_path):
    from tools.i6dmo import campaign, status
    counter = tmp_path / "c.txt"
    root = make_matched(tmp_path)
    drive(root, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert set(states(root).values()) == {"COMPLETE"}
    doc = comparison(root)
    assert doc["decision"]["disposition"] == "MG1_SELECTED" and doc["decision"]["selected"] == "mg1"
    assert doc["decision"]["scores"]["mg1"] == pytest.approx(-0.02)
    assert doc["decision"]["scores"]["mg2m"] == pytest.approx(-0.01)
    assert doc["common_population_sha256"] == POPULATION
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2m.fit"]
    out = campaign.publish(root, warehouse=FakeWarehouse())
    assert out["publication"]["state"] == "DELIVERED" and out["close"] == "CLOSED"
    close = json.loads((root / "CLOSE.json").read_text())
    assert close["disposition"] == "MG1_SELECTED" and close["trigger"] == "AUTOMATIC"
    assert set(close["closure_inputs"]) >= {"PUBLICATION.json", "MIRRORS.json", "OUTBOX_READBACK.json"}
    assert status.compute(root, level="campaign")["state"] == "CLOSED"
    assert drive(root)["close"] == "ALREADY_CLOSED"
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2m.fit"]


def test_tie_within_tolerance_keeps_mg2m_and_a_scientific_negative_is_complete():
    from tools.i6dmo import matched_runner as MR
    heads = lambda v: {h: v for h in MR.HORIZONS}  # noqa: E731
    assert MR.decide(dict(mg1=heads(-0.01 - 5e-9), mg2m=heads(-0.01)))["disposition"] == "MG2_RETAINED_WITHIN_TOLERANCE"
    assert MR.decide(dict(mg1=heads(-0.01 - 2e-8), mg2m=heads(-0.01)))["disposition"] == "MG1_SELECTED"
    assert MR.decide(dict(mg1=heads(0.3), mg2m=heads(0.2)))["disposition"] == "MG2_SELECTED"  # both lose to naive
    incomplete = MR.decide(dict(mg2m=heads(-0.01)))
    assert incomplete["disposition"] == "INCOMPLETE_COMPARISON" and incomplete["missing"] == ["ARM_MISSING:mg1"]


def test_a_failed_arm_closes_incomplete_comparison_with_the_missing_evidence(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    root = make_matched(tmp_path)
    drive(root, env={"I6DMO_FAKE_COUNTER": str(counter), "I6DMO_FAKE_FAIL_UNIT": "mg1.fit"})
    st = states(root)
    assert st["mg1.fit"] == "FAILED" and st["mg1.report"] == "BLOCKED" and st["comparison"] == "COMPLETE"
    doc = comparison(root)
    assert doc["decision"]["disposition"] == "INCOMPLETE_COMPARISON"
    assert doc["decision"]["missing"] == ["ARM_MISSING:mg1"]
    assert campaign.publish(root, warehouse=FakeWarehouse())["close"] == "CLOSED"
    assert json.loads((root / "CLOSE.json").read_text())["disposition"] == "INCOMPLETE_COMPARISON"
    assert fits(counter).count("fit:mg1.fit") == 1  # a technical failure is never retried


def test_a_missing_head_is_incomplete_and_population_or_row_mismatches_are_refused(tmp_path):
    from tools.i6dmo import status
    root = make_matched(tmp_path)
    drive(root, env={"I6DMO_FAKE_DROP_HEAD": "mg2m:12"})
    doc = comparison(root)
    assert doc["decision"]["disposition"] == "INCOMPLETE_COMPARISON"
    assert doc["decision"]["missing"] == ["HEAD_MISSING:mg2m:h12"]
    other = make_matched(tmp_path, name="population")
    drive(other, env={"I6DMO_FAKE_POPULATION": "e" * 64})
    assert states(other)["comparison"] == "FAILED"
    cause = status.compute(other, level="unit", unit="comparison")["cause"]
    assert "COMMON_POPULATION_NOT_DECLARED_OR_DIFFERS:mg1.report" in cause
    third = make_matched(tmp_path, name="rows")
    drive(third, env={"I6DMO_FAKE_ROWS_MG1": "9" * 64})
    assert "COMMON_ROWS_DIFFER:h2" in status.compute(third, level="unit", unit="comparison")["cause"]


def test_real_adapters_are_never_claimed_before_the_model_lane_declares_them(tmp_path):
    """The default (real) entry points hold arm units typed: no claim, no attempt, no fit."""
    from tools.i6dmo import campaign, status
    root = make_matched(tmp_path, entries=None)
    campaign.tick(root)
    out = worker(root, "preferred")
    assert out["result"] == "WAIT_ARTIFACT"
    assert out["cause"].startswith("MATCHED_FIT_ADAPTER_UNAVAILABLE:MATCHED_POPULATION_CLI owner=model lane")
    assert status.compute(root, level="unit", unit="mg1.fit")["attempts"] == []


# ---------------------------------------------------------------------------------- runtime
def test_runtime_pin_mismatch_routes_away_and_never_runs_a_fit(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    root = make_matched(tmp_path)
    campaign.tick(root)
    probe = provider(per_role={"preferred": {"tensorflow": "2.21.0", "keras": "3.9.0"}})
    out = worker(root, "preferred", {"I6DMO_FAKE_COUNTER": str(counter)}, facts_provider=probe)
    assert out["result"] == "WAIT_RESOURCE"
    assert out["cause"].startswith("RUNTIME_PIN_MISMATCH:tensorflow=2.21.0!=2.18.0")
    assert fits(counter) == []
    out = worker(root, "secondary", {"I6DMO_FAKE_COUNTER": str(counter)}, facts_provider=probe)
    assert out["result"] == "COMPLETE" and fits(counter) == ["fit:mg1.fit"]  # routed to the pinned runtime
    unverified = provider(per_role={"preferred": {}})
    out = worker(root, "preferred", facts_provider=lambda r: {k: v for k, v in unverified(r).items()
                                                              if k != "runtime"})
    assert out["result"] == "WAIT_RESOURCE" and out["cause"].startswith("RUNTIME_UNVERIFIED")


# ---------------------------------------------------------------------------------- SC09
def test_restart_duplicate_claims_unavailable_host_and_outbox_retry_never_refit(tmp_path):
    from tools.i6dmo import campaign, queue, worker as W
    counter = tmp_path / "c.txt"
    env = {"I6DMO_FAKE_COUNTER": str(counter)}
    root = make_matched(tmp_path)
    campaign.tick(root)

    class Down(W.LocalExecutor):
        def start(self, **kw):
            raise RuntimeError("host unreachable")
    out = worker(root, "preferred", env, executor=Down())
    assert out["result"] == "INTERRUPTED" and out["cause"] == "LAUNCH_FAILED" and fits(counter) == []
    q = queue.Queue(root / "state" / "queue.sqlite3")
    try:
        held = q.claim("mg1.fit", owner="other", role="secondary", host_key="host-secondary", boot_id="b",
                       device_class="rtx4090", heavy=True, ttl=600)
        assert held["result"] == "CLAIMED"
        again = q.claim("mg1.fit", owner="dup", role="preferred", host_key="host-preferred", boot_id="b",
                        device_class="rtx5090", heavy=True, ttl=600)
        assert again["result"] == "HELD"  # a duplicate claim never starts a second child
    finally:
        q.close()
    assert W.recover_role(root, "secondary")["interrupted"] == ["mg1.fit"]  # coordinator restart
    drive(root, env=env)
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2m.fit"]
    flaky = FakeWarehouse(fail_post=3)
    for _ in range(3):  # delivery outage: retries never re-run a completed fit
        campaign.publish(root, warehouse=flaky)
        for role in ("preferred", "secondary", "coordinator"):
            worker(root, role, env)
    assert (root / "CLOSE.json").exists()
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2m.fit"]


def test_a_measured_peak_of_one_arm_never_caps_the_other(tmp_path):
    from tools.i6dmo import campaign
    root = make_matched(tmp_path)
    campaign.tick(root)
    worker(root, "preferred", {"I6DMO_FAKE_PEAK": str(6 << 30)})  # mg1.fit measured at 6 GiB
    q = campaign.open_queue(root)
    try:
        assert q.measured_peak(kind="fit", family="short", resource_class="short_mg1_matched") == 6 << 30
        assert q.measured_peak(kind="fit", family="short", resource_class="short_mg2_matched") is None
    finally:
        q.close()
    worker(root, "preferred")
    q = campaign.open_queue(root)
    try:
        cap = [a["cap_bytes"] for a in q.admissions() if a["unit_id"] == "mg2m.fit"][-1]
    finally:
        q.close()
    assert cap == campaign.load_deployment(root)["launcher"]["initial_cap_bytes"]  # not 1.25 x 6 GiB


# ---------------------------------------------------------------------------------- SC10
def test_identical_identity_is_adopted_and_a_changed_source_or_code_never_reuses(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    donor = make_matched(tmp_path, name="donor")
    drive(donor, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert len(fits(counter)) == 2
    same = make_matched(tmp_path, name="same", prior_roots=[donor])
    drive(same, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert len(fits(counter)) == 2  # identical identities: verified terminals adopted, nothing refit
    spec_path, data, storage = host(tmp_path)
    source = pins(data, storage, matched_input=dict(relpath="input", sha256="3" * 64))
    resourced = make_matched(tmp_path, name="resourced", data_pins=source, prior_roots=[donor])
    recoded = make_matched(tmp_path, name="recoded", code_commit="d" * 40, prior_roots=[donor])
    ids = lambda r: {u["unit_id"]: u["identity_sha256"] for u in campaign.load_plan(r)["units"]}  # noqa: E731
    old = ids(donor)
    assert ids(same) == old
    for root in (resourced, recoded):
        assert all(ids(root)[u] != old[u] for u in old)
    drive(recoded, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert len(fits(counter)) == 4  # a new code pin is a new identity: never relabelled, refit


# ---------------------------------------------------------------------------------- SC12
def test_status_at_campaign_arm_and_unit_levels(tmp_path):
    from tools.i6dmo import campaign, queue, status, worker as W
    root = make_matched(tmp_path)
    campaign.tick(root)
    now = time.time()
    q = queue.Queue(root / "state" / "queue.sqlite3", clock=lambda: now - 100)
    claim = q.claim("mg1.fit", owner="w", role="preferred", host_key="host-preferred", boot_id="b",
                    device_class="rtx5090", heavy=True, ttl=600)
    q.close()
    folder = W.attempt_dir(root, "mg1.fit", claim["attempt"]["index"])
    folder.mkdir(parents=True)
    (folder / "progress.json").write_text(json.dumps(dict(done=4, total=200, unit="epoch", updated_at=now - 5)))
    arms = status.compute(root, level="arm", now=now)["arms"]
    assert arms["mg1"]["state"] == "RUNNING" and arms["mg1"]["complete"] == 0 and arms["mg1"]["denominator"] == 2
    assert arms["mg1"]["live_children"][0]["role"] == "preferred"
    assert arms["mg1"]["last_progress_at"] == pytest.approx(now - 5)
    assert arms["mg1"]["eta"]["kind"] == "UPPER_BOUND_AT_DECLARED_CEILING"
    assert arms["mg2m"]["state"] == "WAITING" and arms["mg2m"]["eta"]["kind"] == "UNKNOWN"
    camp = status.compute(root, level="campaign", now=now)
    assert camp["arms"]["mg1"]["denominator"] == 2
    assert camp["eta"]["fit"]["families"]["mg2m"]["kind"] == "UNMEASURED"
    stale = status.compute(root, level="arm", arm="mg1", now=now + 3600)["arms"]["mg1"]
    assert stale["state"] == "STALE" and stale["failure_causes"][0]["state"] == "STALE"
    unit = status.compute(root, level="unit", unit="mg1.fit", now=now)
    assert unit["state"] == "RUNNING" and unit["progress"]["done"] == 4


# ------------------------------------------------------------------ B: preparation + CLI
@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """A REAL common population built and sealed by the data lane from the synthetic fixture."""
    from tests.i6dmo import _synth
    from tools.i6dmo import common_population as CP, design as D
    base = tmp_path_factory.mktemp("frozen")
    data, storage = base / "data", base / "storage"
    src = _synth.make_input(data / "input")
    short = _synth.load_design()
    dds = {a: CP.matched_data_design(short, a) for a in CP.ARMS}
    exps = {a: D.matched_market_time_design(short, a, dds, calendar={"calendar_id": "SYNTH_FX_NY17_FIXTURE",
                                                                    "version": "synth-v1"}, runtime=RUNTIME)
            for a in CP.ARMS}
    pop = CP.build_common_population(src, exps, dds, code_commit="TEST")
    pop_dir = storage / "i6dmo" / "matched-short-20261015" / "population"
    seal = CP.write_common_population(pop, pop_dir)
    seal["private_location"] = "$EXPERIMENT_STORAGE_ROOT/i6dmo/matched-short-20261015/population"
    seal_path = base / "POPULATION_SEAL.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True))
    designs = {}
    for arm, doc in exps.items():
        designs[arm] = base / f"EXPERIMENT.{arm}.json"
        designs[arm].write_text(json.dumps(doc))
    return dict(base=base, data=data, storage=storage, input=src, seal=seal_path, designs=designs,
                digest=seal["population_digest"], pop_dir=pop_dir)


def _host_for(frozen, tmp_path, *, storage=None):
    run_root = tmp_path / "run"
    run_root.mkdir(exist_ok=True)
    env = tmp_path / "warehouse.env"
    if not env.exists():
        env.write_text("SECRET_SHOULD_NEVER_BE_READ=1\n")
        os.chmod(env, 0)  # unreadable: preparation must work by path only
    spec = private_spec(tmp_path)
    for role in spec["roles"].values():
        role.update(data_root=str(frozen["data"]), run_root=str(run_root),
                    experiment_storage_root=str(storage or frozen["storage"]))
    spec["warehouse"]["environment_file"] = str(env)
    private = tmp_path / "private"
    private.mkdir(exist_ok=True)
    spec_path = private / "deployment.json"
    spec_path.write_text(json.dumps(spec))
    return spec_path, private


def _prepare(frozen, spec_path, private, **kw):
    from tools.i6dmo import matched_runner as MR
    args = dict(deployment=spec_path, input_dir=frozen["input"], out=private / "prepared.json",
                designs=frozen["designs"], population_seal=frozen["seal"],
                probe=lambda r, d: dict(mem_available_bytes=64 << 30, disk_free_bytes=500 << 30))
    args.update(kw)
    return MR.prepare(**args)


def test_prepare_resolves_the_frozen_population_by_path_and_refuses_explicitly(frozen, tmp_path):
    from tools.i6dmo import matched_runner as MR
    spec_path, private = _host_for(frozen, tmp_path)
    doc = _prepare(frozen, spec_path, private)
    assert doc["warehouse_environment_file"] == "PRESENT_BY_PATH"
    assert doc["data_pins"]["common_population"] == dict(
        base="experiment_storage", relpath="i6dmo/matched-short-20261015/population",
        sha256=_sha(frozen["pop_dir"] / "POPULATION.json"), population_sha256=frozen["digest"])
    assert doc["designs"]["mg1"]["runtime"] == RUNTIME
    text = (private / "prepared.json").read_text()
    assert "SECRET_SHOULD_NEVER_BE_READ" not in text
    assert oct(os.stat(private / "prepared.json").st_mode)[-3:] == "600"
    assert MR.validate(private / "prepared.json")["state"] == "VALID"
    wrong_seal = tmp_path / "wrong_seal.json"
    seal = json.loads(frozen["seal"].read_text())
    wrong_seal.write_text(json.dumps(dict(seal, population_digest="f" * 64)))
    other_storage = tmp_path / "empty_storage"
    other_storage.mkdir()
    cases = [
        (dict(test=True), "TEST_REQUEST"),
        (dict(population_seal=wrong_seal), "UNFROZEN_COMPARISON:POPULATION_DIGEST_MISMATCH"),
        (dict(designs=None), "UNFROZEN_COMPARISON:mg1 design is not the one the population was frozen for"),
        (dict(input_dir=tmp_path / "nowhere"), "MISSING_ARTIFACT:input"),
        (dict(probe=lambda r, d: dict(mem_available_bytes=1 << 30, disk_free_bytes=500 << 30)), "INADEQUATE_MEMORY"),
        (dict(probe=lambda r, d: dict(mem_available_bytes=64 << 30, disk_free_bytes=1 << 30)), "INADEQUATE_DISK"),
        (dict(out=REPO / "docs" / "prepared.json"), "PRIVATE_OUTPUT_INSIDE_GIT"),
        (dict(prior_roots=[frozen["data"]]), "INCOMPATIBLE_DONOR"),
        (dict(population_dir=other_storage), "UNFROZEN_COMPARISON:POPULATION_FILE_DIGEST_MISMATCH"),
    ]
    for kw, cause in cases:
        with pytest.raises(MR.PrepareRefused, match=cause):
            _prepare(frozen, spec_path, private, **kw)
    (tmp_path / "u").mkdir()
    unmounted, private2 = _host_for(frozen, tmp_path / "u", storage=tmp_path / "not-mounted")
    with pytest.raises(MR.PrepareRefused, match="UNAVAILABLE_MOUNT:experiment_storage_root"):
        _prepare(frozen, unmounted, private2)


def test_preflight_checks_each_gpu_worker_runtime_against_the_sealed_pin(frozen, tmp_path):
    from tools.i6dmo import matched_runner as MR
    spec_path, private = _host_for(frozen, tmp_path)
    _prepare(frozen, spec_path, private)
    prepared = json.loads((private / "prepared.json").read_text())
    pop_file = str(frozen["storage"] / prepared["data_pins"]["common_population"]["relpath"] / "POPULATION.json")
    inp_file = f"{prepared['data_pins']['matched_input']['relpath']}/INPUT.npz"

    def probe(runtime):
        def run(role, files):
            doc = facts()
            doc.update(runtime=dict(runtime.get(role, RUNTIME)),
                       files={inp_file: prepared["data_pins"]["matched_input"]["sha256"],
                              pop_file: prepared["data_pins"]["common_population"]["sha256"]})
            return doc
        return run
    ok = MR.preflight(private / "prepared.json", private / "pre", probe=probe({}))
    assert ok["state"] == "PASS" and ok["runtime_pin"] == RUNTIME
    bad = MR.preflight(private / "prepared.json", private / "pre2",
                       probe=probe({"preferred": {"tensorflow": "2.21.0", "keras": "3.9.0"}}))
    assert bad["state"] == "FAIL" and "preferred:runtime_pin" in bad["launch_blockers"]
    cause = next(c for c in bad["roles"]["preferred"] if c["check"] == "runtime_pin")["detail"]["cause"]
    assert cause.startswith("RUNTIME_PIN_MISMATCH:tensorflow=2.21.0!=2.18.0") and "never reseal" in cause


def test_cli_matched_short_end_to_end_on_a_local_fixture(frozen, tmp_path, monkeypatch, capsys):
    """prepare -> validate -> preflight -> init -> finite drive -> publish/CLOSE -> status, through the CLI.

    The arm children are the orchestration fakes (``--entrypoints``); the comparison is the real hook."""
    from tools.i6dmo import campaign, cli, matched_runner as MR, preflight as P
    spec_path, private = _host_for(frozen, tmp_path)
    monkeypatch.setattr(MR, "_local_probe", lambda r, d: dict(mem_available_bytes=64 << 30, disk_free_bytes=500 << 30))
    prepared = private / "prepared.json"

    def run(*argv):
        rc = cli.main(list(argv))
        return rc, json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    common = ["--deployment", str(spec_path), "--input-dir", str(frozen["input"]),
              "--mg1-design", str(frozen["designs"]["mg1"]), "--mg2m-design", str(frozen["designs"]["mg2m"]),
              "--population-seal", str(frozen["seal"])]
    rc, out = run("matched-short", "prepare", *common, "--out", str(prepared))
    assert rc == 0 and out["state"] == "PREPARED" and out["population_sha256"] == frozen["digest"]
    rc, out = run("matched-short", "prepare", *common, "--out", str(private / "t.json"), "--test")
    assert rc == 2 and out["cause"].startswith("TEST_REQUEST")
    rc, out = run("matched-short", "validate", "--prepared", str(prepared))
    assert rc == 0 and out["state"] == "VALID"
    doc = json.loads(prepared.read_text())

    def fake_probe(spec, role, files=(), check_strategy=False, run=None):
        base = facts()
        base.update(runtime=dict(RUNTIME), files={f: (_sha(Path(spec["roles"][role]["data_root"]) / f)
                                                      if not os.path.isabs(f) else _sha(f)) for f in files})
        return base
    monkeypatch.setattr(P, "probe_role", fake_probe)
    rc, out = run("matched-short", "preflight", "--prepared", str(prepared), "--out-dir", str(private / "pre"))
    assert rc == 0 and out["state"] == "PASS" and out["runtime_pin"] == RUNTIME
    root = tmp_path / "campaign"
    fakes = private / "entrypoints.json"
    fakes.write_text(json.dumps(entrypoints()))
    rc, out = run("matched-short", "init", "--prepared", str(prepared), "--root", str(root), "--code-commit", "c" * 40,
                  "--entrypoints", str(fakes))
    assert rc == 0 and out["mode"] == "MATCHED_SHORT_CONTRAST"
    effective = campaign.load_effective(root)
    assert effective["data_pins"] == doc["data_pins"]
    rc, out = run("deploy", "--root", str(root), "--deployment", str(spec_path), "--render-to", str(private / "units"))
    assert rc == 0 and out["render_only"] is True and any(u.endswith("tick.timer") for u in out["units"])
    rc, out = run("tick", "--root", str(root))
    assert rc == 0 and out["campaign_state"] == "WAITING_FOR_WORKER"
    drive(root, env={"I6DMO_FAKE_POPULATION": frozen["digest"]})
    assert comparison(root)["decision"]["disposition"] == "MG1_SELECTED"
    assert campaign.publish(root, warehouse=FakeWarehouse())["close"] == "CLOSED"
    for level, extra in (("campaign", []), ("arm", ["--arm", "mg2m"]), ("unit", ["--unit", "mg1.fit"])):
        rc, out = run("matched-short", "status", "--root", str(root), "--level", level, *extra)
        assert rc == 0 and out["schema"] == "i6dmo.status.v1"
    rc, out = run("status", "--root", str(root), "--level", "arm", "--arm", "mg2m", "--json")
    assert rc == 0 and set(out["arms"]) == {"mg2m"} and out["campaign_state"] == "CLOSED"


@pytest.mark.parametrize("command", [
    ["matched-short", "--help"], ["matched-short", "prepare", "--help"], ["matched-short", "validate", "--help"],
    ["matched-short", "preflight", "--help"], ["matched-short", "init", "--help"],
    ["matched-short", "status", "--help"], ["deploy", "--help"], ["status", "--help"], ["tick", "--help"],
    ["worker", "--help"], ["publish", "--help"]])
def test_every_documented_command_has_help(command):
    proc = subprocess.run([sys.executable, "-m", "tools.i6dmo", *command], cwd=REPO, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0 and "usage:" in proc.stdout
