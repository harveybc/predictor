"""Matched short contrast runner (NEXT_CONTRASTS_PLAN_20261015 section 4; SC08/SC09/SC10/SC12).

ORCHESTRATION ONLY for the arms: the fit/report children are :mod:`tests.i6dmo.ops_matched_fake`
until the model lane's matched adapters land (the reviewer runs those). Everything else is real:
campaign init/identity, tick, governed worker, stored-terminal verification, the REAL comparison
hook through ``unit_runner --kind comparison``, outbox with exact readback, mirrors, automatic
CLOSE, status and the ``python -m tools.i6dmo matched-short`` entrypoint.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.i6dmo.ops_fakes import FakeWarehouse, facts_provider, private_spec

REPO = Path(__file__).resolve().parents[2]
FAKE = Path(__file__).with_name("ops_matched_fake.py")
MG2 = REPO / "docs/i6n/direct_multi_output_static/EXPERIMENT.mg2.json"
POPULATION = "b" * 64


def market_adapter():
    """Stand-in for the model-lane validator: the fixtures are not the data lane's sealed designs."""
    def validate_design(design):
        ok = design.get("schema") == "i6dmo.market_time_design.v1"
        return dict(ok=ok, errors=[] if ok else ["schema"], design_sha256=hashlib.sha256(
            json.dumps(design, sort_keys=True).encode()).hexdigest())
    return SimpleNamespace(validate_design=validate_design,
                           effective_config=lambda d: dict(schema="fake.effective", design=d))


def write_designs(tmp_path, **mg1_overrides):
    base = json.loads(MG2.read_text())
    base.pop("seal_sha256", None)
    mg2 = dict(base, experiment_id="I6-DMO-SHORT-MG2-MATCHED-R0-FIXTURE")
    mg1 = json.loads(json.dumps(mg2))
    mg1["experiment_id"] = "I6-DMO-SHORT-MG1-MATCHED-R0-FIXTURE"
    mg1["market_time"].update(observations=25, spacing_open_market_hours=1, rule="MG1_MARKET_EXACT")
    for key, value in mg1_overrides.items():
        mg1[key] = value
    paths = {}
    for arm, doc in (("mg1", mg1), ("mg2", mg2)):
        paths[arm] = Path(tmp_path) / f"design.{arm}.json"
        paths[arm].write_text(json.dumps(doc))
    return paths


def pins(rules=("MG1_MARKET_EXACT", "MG2_MARKET_EXACT")):
    out = {}
    for arm, rule in zip(("mg1", "mg2"), rules):
        out[f"{arm}_bundle"] = dict(relpath=f"bundles/{arm}", sha256=hashlib.sha256(arm.encode()).hexdigest(),
                                    bundle_sha256=hashlib.sha256(f"b{arm}".encode()).hexdigest(), input_sampling=rule)
        out[f"{arm}_input"] = dict(relpath="input", sha256="1" * 64)
    out["common_population"] = dict(relpath="population", sha256="2" * 64, population_sha256=POPULATION)
    return out


def entrypoints():
    from tools.i6dmo import campaign
    base = [sys.executable, str(FAKE)]
    return dict(campaign.MATCHED_ENTRYPOINTS, fit=base + ["--kind", "fit"], report=base + ["--kind", "report"])


def make_matched(tmp_path, *, name="matched", designs=None, data_pins=None, prior_roots=(), code_commit="c" * 40):
    from tools.i6dmo import campaign
    tmp_path = Path(tmp_path)
    spec_path = tmp_path / "private_spec.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(private_spec(tmp_path)))
    root = tmp_path / name
    campaign.init_matched_contrast(designs or write_designs(tmp_path), spec_path, root, adapter=market_adapter(),
                                   code_commit=code_commit, data_pins=data_pins or pins(),
                                   entrypoints=entrypoints(), prior_roots=prior_roots)
    return root


def worker(root, role, env=None, **kw):
    from tools.i6dmo import worker as W
    return W.worker_once(root, role, facts_provider=kw.pop("provider", facts_provider()),
                         executor=kw.pop("executor", W.LocalExecutor()), poll_interval=0.02,
                         extra_env=dict(env or {}), **kw)


def drive(root, env=None, rounds=10):
    from tools.i6dmo import campaign
    for _ in range(rounds):
        campaign.tick(root)
        progressed = False
        for role in ("preferred", "secondary", "coordinator"):
            if worker(root, role, env)["result"] not in ("IDLE", "CLOSED", "WAIT_ARTIFACT", "WAIT_RESOURCE"):
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
def test_graph_is_two_matched_arms_and_one_comparison_without_test(tmp_path):
    from tools.i6dmo import campaign
    root = make_matched(tmp_path)
    plan = campaign.load_plan(root)
    units = {u["unit_id"]: u for u in plan["units"]}
    assert list(units) == ["mg1.fit", "mg1.report", "mg2.fit", "mg2.report", "comparison"]
    assert units["comparison"]["depends_on"] == ["mg1.report", "mg2.report"]
    assert units["mg1.fit"]["resource_class"] == "short_mg1_matched"
    assert units["mg2.fit"]["resource_class"] == "short_mg2_matched"
    assert units["mg1.fit"]["identity_sha256"] != units["mg2.fit"]["identity_sha256"]
    assert plan["mode"] == "MATCHED_SHORT_CONTRAST" and plan["test_units_enumerated"] is False
    assert all(u["test_open"] is False and "test" not in u["windows"] for u in plan["units"])
    with pytest.raises(campaign.Refused, match="TEST_NOT_ENUMERABLE_IN_MATCHED_CONTRAST"):
        campaign.enumerate_test_units(root)
    effective = campaign.load_effective(root)
    assert effective["comparison"]["tolerance"] == 1e-8 and effective["comparison"]["tie_keeps"] == "mg2"


@pytest.mark.parametrize("change,cause", [
    (dict(seed=1), "MATCHED_ARMS_DIFFER: seed"),
    (dict(horizons_hours=[2, 3, 4, 5, 6, 24]), "MATCHED_ARMS_DIFFER: horizons_hours"),
])
def test_init_refuses_arms_that_differ_beyond_the_sampling_recipe(tmp_path, change, cause):
    from tools.i6dmo import campaign
    with pytest.raises(campaign.Refused, match=cause):
        make_matched(tmp_path, designs=write_designs(tmp_path, **change))


def test_init_refuses_test_unfrozen_or_mismatched_inputs(tmp_path):
    from tools.i6dmo import campaign
    designs = write_designs(tmp_path)
    doc = json.loads(designs["mg1"].read_text())
    doc["splits"]["test"]["open"] = True
    bad = tmp_path / "open.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(campaign.Refused, match="TEST_OPEN_IN_DESIGN: mg1"):
        make_matched(tmp_path, name="a", designs=dict(designs, mg1=bad))
    unfrozen = pins()
    unfrozen.pop("common_population")
    with pytest.raises(campaign.Refused, match="UNFROZEN_COMPARISON"):
        make_matched(tmp_path, name="b", data_pins=unfrozen)
    with pytest.raises(campaign.Refused, match="BUNDLE_SAMPLING: mg1_bundle"):
        make_matched(tmp_path, name="c", data_pins=pins(rules=("MG2_MARKET_EXACT", "MG2_MARKET_EXACT")))
    swapped = write_designs(tmp_path)
    doc = json.loads(swapped["mg1"].read_text())
    doc["market_time"]["observations"] = 13
    swapped["mg1"].write_text(json.dumps(doc))
    with pytest.raises(campaign.Refused, match="MATCHED_ARM_RECIPE: mg1"):
        make_matched(tmp_path, name="d", designs=swapped)


# ------------------------------------------------------------------- decision and SC08 close
def test_full_contrast_decides_publishes_mirrors_and_closes_automatically(tmp_path):
    from tools.i6dmo import campaign, status
    counter = tmp_path / "c.txt"
    root = make_matched(tmp_path)
    drive(root, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert states(root) == {u: "COMPLETE" for u in ("mg1.fit", "mg1.report", "mg2.fit", "mg2.report", "comparison")}
    doc = comparison(root)
    assert doc["decision"]["disposition"] == "MG1_SELECTED" and doc["decision"]["selected"] == "mg1"
    assert doc["decision"]["scores"]["mg1"] == pytest.approx(-0.02) and doc["decision"]["scores"]["mg2"] == pytest.approx(-0.01)
    assert doc["common_population_sha256"] == POPULATION
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2.fit"]
    out = campaign.publish(root, warehouse=FakeWarehouse())
    assert out["publication"]["state"] == "DELIVERED" and out["close"] == "CLOSED"
    close = json.loads((root / "CLOSE.json").read_text())
    assert close["disposition"] == "MG1_SELECTED" and close["trigger"] == "AUTOMATIC"
    assert set(close["closure_inputs"]) >= {"PUBLICATION.json", "MIRRORS.json", "OUTBOX_READBACK.json"}
    assert status.compute(root, level="campaign")["state"] == "CLOSED"
    assert drive(root)["close"] == "ALREADY_CLOSED"
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2.fit"]


def test_tie_within_tolerance_keeps_mg2_and_a_scientific_negative_is_complete():
    from tools.i6dmo import matched_runner as MR
    heads = lambda v: {h: v for h in MR.HORIZONS}  # noqa: E731
    assert MR.decide(dict(mg1=heads(-0.01 - 5e-9), mg2=heads(-0.01)))["disposition"] == "MG2_RETAINED_WITHIN_TOLERANCE"
    assert MR.decide(dict(mg1=heads(-0.01 - 2e-8), mg2=heads(-0.01)))["disposition"] == "MG1_SELECTED"
    assert MR.decide(dict(mg1=heads(0.3), mg2=heads(0.2)))["disposition"] == "MG2_SELECTED"  # both lose to naive
    incomplete = MR.decide(dict(mg2=heads(-0.01)))
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


def test_a_missing_head_is_incomplete_and_a_population_mismatch_is_refused(tmp_path):
    root = make_matched(tmp_path)
    drive(root, env={"I6DMO_FAKE_DROP_HEAD": "mg2:12"})
    doc = comparison(root)
    assert doc["decision"]["disposition"] == "INCOMPLETE_COMPARISON"
    assert doc["decision"]["missing"] == ["HEAD_MISSING:mg2:h12"]
    other = make_matched(tmp_path, name="population")
    drive(other, env={"I6DMO_FAKE_POPULATION": "e" * 64})
    assert states(other)["comparison"] == "FAILED"
    from tools.i6dmo import status
    cause = status.compute(other, level="unit", unit="comparison")["cause"]
    assert "COMMON_POPULATION_NOT_DECLARED_OR_DIFFERS:mg1.report" in cause
    third = make_matched(tmp_path, name="rows")
    drive(third, env={"I6DMO_FAKE_ROWS_MG1": "9" * 64})
    assert "COMMON_ROWS_DIFFER:h2" in status.compute(third, level="unit", unit="comparison")["cause"]


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
    # a duplicate claim while a lease is held is refused, never a second child
    q = queue.Queue(root / "state" / "queue.sqlite3")
    try:
        held = q.claim("mg1.fit", owner="other", role="secondary", host_key="host-secondary", boot_id="b",
                       device_class="rtx4090", heavy=True, ttl=600)
        assert held["result"] == "CLAIMED"
        again = q.claim("mg1.fit", owner="dup", role="preferred", host_key="host-preferred", boot_id="b",
                        device_class="rtx5090", heavy=True, ttl=600)
        assert again["result"] == "HELD"
    finally:
        q.close()
    # coordinator restart: the orphaned attempt (never launched) is interrupted, then claimed once
    assert W.recover_role(root, "secondary")["interrupted"] == ["mg1.fit"]
    drive(root, env=env)
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2.fit"]
    # delivery outage: retries never re-run a completed fit
    flaky = FakeWarehouse(fail_post=3)
    for _ in range(3):
        campaign.publish(root, warehouse=flaky)
        for role in ("preferred", "secondary", "coordinator"):
            worker(root, role, env)
    assert (root / "CLOSE.json").exists()
    assert sorted(fits(counter)) == ["fit:mg1.fit", "fit:mg2.fit"]


def test_a_measured_peak_of_one_arm_never_caps_the_other(tmp_path):
    from tools.i6dmo import campaign
    root = make_matched(tmp_path)
    campaign.tick(root)
    worker(root, "preferred", {"I6DMO_FAKE_PEAK": str(6 << 30)})  # the first arm's fit, measured at 6 GiB
    q = campaign.open_queue(root)
    try:
        first = next(a for a in q.attempts() if a["outcome"] == "COMPLETE")
        assert q.measured_peak(kind="fit", family="short", resource_class=q.unit(first["unit_id"])["spec"]
                               ["resource_class"]) == 6 << 30
        other = "mg2.fit" if first["unit_id"] == "mg1.fit" else "mg1.fit"
        assert q.measured_peak(kind="fit", family="short",
                               resource_class=q.unit(other)["spec"]["resource_class"]) is None
    finally:
        q.close()
    worker(root, "preferred")
    q = campaign.open_queue(root)
    try:
        cap = [a["cap_bytes"] for a in q.admissions() if a["unit_id"] == other][-1]
    finally:
        q.close()
    spec = campaign.load_deployment(root)
    assert cap == spec["launcher"]["initial_cap_bytes"]  # not 1.25 x the other arm's 6 GiB peak


# ---------------------------------------------------------------------------------- SC10
def test_changed_design_or_source_never_reuses_an_old_identity(tmp_path):
    from tools.i6dmo import campaign
    counter = tmp_path / "c.txt"
    donor = make_matched(tmp_path, name="donor")
    drive(donor, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert len(fits(counter)) == 2
    same = make_matched(tmp_path, name="same", prior_roots=[donor])
    drive(same, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert len(fits(counter)) == 2  # identical identities: verified terminals adopted, nothing refit
    (tmp_path / "changed_design").mkdir()
    designs = write_designs(tmp_path / "changed_design")
    doc = json.loads(designs["mg1"].read_text())
    doc["training"] = dict(doc["training"], patience=11)
    designs["mg1"].write_text(json.dumps(doc))
    changed = make_matched(tmp_path, name="changed", designs=designs, prior_roots=[donor])
    source = pins()
    source["mg2_input"] = dict(source["mg2_input"], sha256="3" * 64)
    resourced = make_matched(tmp_path, name="resourced", data_pins=source, prior_roots=[donor])
    ids = lambda r: {u["unit_id"]: u["identity_sha256"] for u in campaign.load_plan(r)["units"]}  # noqa: E731
    old, new_design, new_source = ids(donor), ids(changed), ids(resourced)
    # a changed mg1 design changes every mg1 identity and the comparison; mg2 is untouched and reusable
    assert all(new_design[u] != old[u] for u in ("mg1.fit", "mg1.report", "comparison"))
    assert all(new_design[u] == old[u] for u in ("mg2.fit", "mg2.report"))
    # a changed source pin changes every identity
    assert all(new_source[u] != old[u] for u in old)
    drive(changed, env={"I6DMO_FAKE_COUNTER": str(counter)})
    assert fits(counter)[2:] == ["fit:mg1.fit"]  # only the changed arm refits; mg2 adopted verified


# ---------------------------------------------------------------------------------- SC12
def test_status_at_campaign_arm_and_unit_levels(tmp_path):
    from tools.i6dmo import campaign, queue, status
    root = make_matched(tmp_path)
    campaign.tick(root)
    now = time.time()
    q = queue.Queue(root / "state" / "queue.sqlite3", clock=lambda: now - 100)
    claim = q.claim("mg1.fit", owner="w", role="preferred", host_key="host-preferred", boot_id="b",
                    device_class="rtx5090", heavy=True, ttl=600)
    q.close()
    from tools.i6dmo import worker as W
    folder = W.attempt_dir(root, "mg1.fit", claim["attempt"]["index"])
    folder.mkdir(parents=True)
    (folder / "progress.json").write_text(json.dumps(dict(done=4, total=200, unit="epoch", updated_at=now - 5)))
    arms = status.compute(root, level="arm", now=now)["arms"]
    assert arms["mg1"]["state"] == "RUNNING" and arms["mg1"]["complete"] == 0 and arms["mg1"]["denominator"] == 2
    assert arms["mg1"]["live_children"][0]["role"] == "preferred"
    assert arms["mg1"]["last_progress_at"] == pytest.approx(now - 5)
    assert arms["mg1"]["eta"]["kind"] == "UPPER_BOUND_AT_DECLARED_CEILING"
    assert arms["mg2"]["state"] == "WAITING" and arms["mg2"]["eta"]["kind"] == "UNKNOWN"
    camp = status.compute(root, level="campaign", now=now)
    assert camp["arms"]["mg1"]["denominator"] == 2 and camp["eta"]["fit"]["families"]["mg2"]["kind"] == "UNMEASURED"
    stale = status.compute(root, level="arm", arm="mg1", now=now + 3600)["arms"]["mg1"]
    assert stale["state"] == "STALE" and stale["failure_causes"][0]["state"] == "STALE"
    unit = status.compute(root, level="unit", unit="mg1.fit", now=now)
    assert unit["state"] == "RUNNING" and unit["progress"]["done"] == 4


# ------------------------------------------------------------------ B: preparation + CLI
def _fixture_host(tmp_path):
    data = tmp_path / "data"
    for arm in ("mg1", "mg2"):
        (data / "bundles" / arm).mkdir(parents=True)
        rule = "MG1_MARKET_EXACT" if arm == "mg1" else "MG2_MARKET_EXACT"
        (data / "bundles" / arm / "BUNDLE.json").write_text(json.dumps(dict(
            bundle_sha256=hashlib.sha256(arm.encode()).hexdigest(), input_sampling=dict(rule=rule))))
    (data / "input").mkdir()
    (data / "input" / "INPUT.npz").write_bytes(b"fixture")
    (data / "population").mkdir()
    (data / "population" / "POPULATION.json").write_text(json.dumps(dict(schema="i6dmo.common_population.v1")))
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = tmp_path / "warehouse.env"
    env.write_text("SECRET_SHOULD_NEVER_BE_READ=1\n")
    os.chmod(env, 0)  # unreadable: preparation must work by path only
    spec = private_spec(tmp_path)
    for role in spec["roles"].values():
        role.update(data_root=str(data), run_root=str(run_root))
    spec["warehouse"]["environment_file"] = str(env)
    spec_path = tmp_path / "private" / "deployment.json"
    spec_path.parent.mkdir()
    spec_path.write_text(json.dumps(spec))
    designs = write_designs(tmp_path)
    return dict(data=data, spec=spec_path, designs=designs, env=env, private=tmp_path / "private")


def _loader(path, *, expected_digest):
    if expected_digest != POPULATION:
        raise ValueError("POPULATION_DIGEST_MISMATCH")
    return dict(path=str(path), digest=expected_digest)


def _prepare(host, **kw):
    from tools.i6dmo import matched_runner as MR
    args = dict(deployment=host["spec"], designs=host["designs"],
                bundles={a: host["data"] / "bundles" / a for a in MR.ARMS},
                inputs={a: host["data"] / "input" for a in MR.ARMS},
                common_population=host["data"] / "population", population_digest=POPULATION,
                out=host["private"] / "prepared.json", probe=lambda r, d: dict(mem_available_bytes=64 << 30,
                                                                               disk_free_bytes=500 << 30),
                population_loader=_loader)
    args.update(kw)
    return MR.prepare(**args)


def _fix_seed(host):
    for arm in ("mg1", "mg2"):
        doc = json.loads(host["designs"][arm].read_text())
        assert doc["seed"] == 6102026


def test_prepare_resolves_pins_by_path_and_refuses_explicitly(tmp_path):
    from tools.i6dmo import matched_runner as MR
    host = _fixture_host(tmp_path)
    _fix_seed(host)
    doc = _prepare(host)
    assert doc["warehouse_environment_file"] == "PRESENT_BY_PATH"
    assert set(doc["data_pins"]) == {"mg1_bundle", "mg1_input", "mg2_bundle", "mg2_input", "common_population"}
    text = (host["private"] / "prepared.json").read_text()
    assert "SECRET_SHOULD_NEVER_BE_READ" not in text and oct(os.stat(host["private"] / "prepared.json").st_mode)[-3:] == "600"
    assert MR.validate(host["private"] / "prepared.json", population_loader=_loader)["state"] == "VALID"
    cases = [
        (dict(test=True), "TEST_REQUEST"),
        (dict(population_digest="f" * 64), "UNFROZEN_COMPARISON"),
        (dict(population_loader=None), "UNFROZEN_COMPARISON"),
        (dict(common_population=host["data"] / "absent"), "UNFROZEN_COMPARISON"),
        (dict(bundles=dict(mg1=host["data"] / "bundles" / "mg2", mg2=host["data"] / "bundles" / "mg2")),
         "INCOMPATIBLE_BUNDLE:mg1"),
        (dict(inputs=dict(mg1=tmp_path / "nowhere", mg2=host["data"] / "input")), "MISSING_ARTIFACT:mg1 input"),
        (dict(probe=lambda r, d: dict(mem_available_bytes=1 << 30, disk_free_bytes=500 << 30)), "INADEQUATE_MEMORY"),
        (dict(probe=lambda r, d: dict(mem_available_bytes=64 << 30, disk_free_bytes=1 << 30)), "INADEQUATE_DISK"),
        (dict(out=REPO / "docs" / "prepared.json"), "PRIVATE_OUTPUT_INSIDE_GIT"),
        (dict(prior_roots=[tmp_path / "data"]), "INCOMPATIBLE_DONOR"),
    ]
    import sys as _sys
    _sys.modules.pop("tools.i6dmo.common_population", None)
    for kw, cause in cases:
        with pytest.raises(MR.PrepareRefused, match=cause):
            _prepare(host, **kw)
    spec = json.loads(host["spec"].read_text())
    spec["roles"]["coordinator"]["data_root"] = str(tmp_path / "unmounted")
    host["spec"].write_text(json.dumps(spec))
    with pytest.raises(MR.PrepareRefused, match="UNAVAILABLE_MOUNT:data_root"):
        _prepare(host)


def test_cli_matched_short_end_to_end_on_a_local_fixture(tmp_path, monkeypatch, capsys):
    """prepare -> validate -> init -> finite drive -> publish/CLOSE -> status at every level, through the CLI.

    The arm children are the orchestration fakes (``--entrypoints``); the comparison is the real hook."""
    from tools.i6dmo import cli, matched_runner as MR
    host = _fixture_host(tmp_path)
    module = ModuleType("tools.i6dmo.common_population")
    module.load_common_population = _loader
    monkeypatch.setitem(sys.modules, "tools.i6dmo.common_population", module)
    monkeypatch.setattr(MR, "_local_probe", lambda r, d: dict(mem_available_bytes=64 << 30, disk_free_bytes=500 << 30))
    prepared = host["private"] / "prepared.json"

    def run(*argv):
        rc = cli.main(list(argv))
        return rc, json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rc, out = run("matched-short", "prepare", "--deployment", str(host["spec"]),
                  "--mg1-design", str(host["designs"]["mg1"]), "--mg2-design", str(host["designs"]["mg2"]),
                  "--mg1-bundle", str(host["data"] / "bundles" / "mg1"),
                  "--mg2-bundle", str(host["data"] / "bundles" / "mg2"),
                  "--mg1-input", str(host["data"] / "input"),
                  "--common-population", str(host["data"] / "population"), "--population-digest", POPULATION,
                  "--out", str(prepared))
    assert rc == 0 and out["state"] == "PREPARED"
    rc, out = run("matched-short", "prepare", "--deployment", str(host["spec"]),
                  "--mg1-design", str(host["designs"]["mg1"]), "--mg2-design", str(host["designs"]["mg2"]),
                  "--mg1-bundle", str(host["data"] / "bundles" / "mg1"),
                  "--mg2-bundle", str(host["data"] / "bundles" / "mg2"),
                  "--mg1-input", str(host["data"] / "input"),
                  "--common-population", str(host["data"] / "population"), "--population-digest", POPULATION,
                  "--out", str(host["private"] / "t.json"), "--test")
    assert rc == 2 and out["cause"].startswith("TEST_REQUEST")
    rc, out = run("matched-short", "validate", "--prepared", str(prepared))
    assert rc == 0 and out["state"] == "VALID"
    from tools.i6dmo import campaign
    monkeypatch.setattr(campaign, "_adapter", lambda adapter=None: market_adapter())
    root = tmp_path / "campaign"
    fakes = host["private"] / "entrypoints.json"
    fakes.write_text(json.dumps(entrypoints()))
    rc, out = run("matched-short", "init", "--prepared", str(prepared), "--root", str(root), "--code-commit", "c" * 40,
                  "--entrypoints", str(fakes))
    assert rc == 0 and out["mode"] == "MATCHED_SHORT_CONTRAST"
    effective = json.loads((root / "EFFECTIVE_CONFIG.json").read_text())
    assert effective["entrypoints"]["comparison"][-2:] == ["--kind", "comparison"]
    drive(root)
    assert comparison(root)["decision"]["disposition"] == "MG1_SELECTED"
    assert campaign.publish(root, warehouse=FakeWarehouse())["close"] == "CLOSED"
    for level in ("campaign", "arm", "unit"):
        args = ["matched-short", "status", "--root", str(root), "--level", level]
        if level == "unit":
            args += ["--unit", "mg1.fit"]
        rc, out = run(*args)
        assert rc == 0 and out["schema"] == "i6dmo.status.v1"
    rc, out = run("status", "--root", str(root), "--level", "arm", "--arm", "mg2", "--json")
    assert rc == 0 and set(out["arms"]) == {"mg2"}


@pytest.mark.parametrize("command", [
    ["matched-short", "--help"], ["matched-short", "prepare", "--help"], ["matched-short", "validate", "--help"],
    ["matched-short", "preflight", "--help"], ["matched-short", "init", "--help"],
    ["matched-short", "status", "--help"], ["deploy", "--help"], ["status", "--help"], ["tick", "--help"],
    ["worker", "--help"], ["publish", "--help"]])
def test_every_documented_command_has_help(command):
    proc = subprocess.run([sys.executable, "-m", "tools.i6dmo", *command], cwd=REPO, capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 0 and "usage:" in proc.stdout
