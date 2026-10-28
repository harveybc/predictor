"""Matched short contrast, REAL chain end to end on CPU (no fake arm child, no injected arrays_loader).

A tiny synthetic common population is built and sealed with the data lane's REAL
``build_common_population`` from a sparse synthetic input (the model lane's fixture shape), with
both arm designs sealed by ``design.matched_market_time_design`` under the synthetic fixture
calendar. Then, through the ordinary automation only:

    init_matched_contrast (engineering CPU fixture; refused for any non-synthetic calendar)
    -> deploy --no-start (injected systemctl recorder)
    -> tick / worker_once: real claim -> real ``python -m tools.i6dmo.unit_runner``
       -> fit: real ``python -m tools.i6dmo.train --common-population ... --input-dir ...`` subprocess
          for BOTH arms (2 epochs, ``I6DMO_TEST_TRAINING_OVERRIDES``: receipt scientific=false)
       -> report: real ``reporting.run_unit`` -> ``run_matched_unit`` (expected weeks from the
          sealed calendar through market_grid)
       -> comparison: real hook -> ``reporting.decide_matched`` (the one decision rule)
    -> publish to the local test sink with exact readback -> two scratch mirrors
    -> automatic CLOSE -> TIMERS_STOPPED (this campaign's units only) -> inert re-tick.

No GPU, no TEST bar, no real systemd, no production OLAP.
"""
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from tests.i6dmo import _synth  # noqa: E402
from tests.i6dmo.ops_fakes import FakeWarehouse, facts, private_spec  # noqa: E402
from tests.i6dmo.test_matched_fit_path import FIXTURE_CAL, make_sparse_input  # noqa: E402

OVERRIDES = json.dumps({"max_epochs": 2})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return dict(rc=0, stdout="")


@pytest.fixture(autouse=True)
def _real_systemctl_forbidden(monkeypatch):
    from tools.i6dmo import deploy
    monkeypatch.setattr(deploy, "default_systemctl", lambda *a: pytest.fail(f"real systemctl called: {a}"))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from tools.i6dmo import common_population as CP, design as D
    base = tmp_path_factory.mktemp("matched_real")
    data, storage = base / "data", base / "storage"
    data.mkdir()
    storage.mkdir()
    src = make_sparse_input(data / "input")
    short = _synth.load_design()
    dds = {a: CP.matched_data_design(short, a) for a in CP.ARMS}
    designs = {}
    for arm in CP.ARMS:
        designs[arm] = base / f"EXPERIMENT.{arm}.json"
        designs[arm].write_text(json.dumps(D.matched_market_time_design(short, arm, dds, calendar=FIXTURE_CAL),
                                           indent=1, sort_keys=True))
    exps = {a: json.loads(designs[a].read_text()) for a in CP.ARMS}
    seal = CP.write_common_population(CP.build_common_population(src, exps, dds, code_commit="TEST"),
                                      storage / "population")
    return dict(base=base, data=data, storage=storage, input=src, designs=designs, exps=exps,
                digest=seal["population_digest"])


def _campaign(world, name):
    from tools.i6dmo import campaign
    base = world["base"] / name
    base.mkdir()
    spec = private_spec(base)
    for role in spec["roles"].values():
        role.update(data_root=str(world["data"]), experiment_storage_root=str(world["storage"]))
    spec_path = base / "private_spec.json"
    spec_path.write_text(json.dumps(spec))
    pins = dict(common_population=dict(base="experiment_storage", relpath="population",
                                       sha256=_sha(world["storage"] / "population" / "POPULATION.json"),
                                       population_sha256=world["digest"]),
                matched_input=dict(relpath="input", sha256=_sha(world["input"] / "INPUT.npz")))
    root = base / "campaign"
    campaign.init_matched_contrast(world["designs"], spec_path, root, adapter=None, code_commit="c" * 40,
                                   data_pins=pins, engineering_cpu_fixture=True)
    return root, spec_path


def _provider(world):
    from tools.i6dmo import matched_runner as MR
    runtime = MR.runtime_pin(world["exps"]["mg1"])

    def probe(role):
        doc = facts()
        doc["runtime"] = dict(runtime)
        return doc
    return probe


def test_matched_contrast_real_chain_fit_report_decide_publish_close_stop(world):
    from tools.i6dmo import campaign, deploy, status, worker as W
    root, spec_path = _campaign(world, "chain")
    effective = campaign.load_effective(root)
    assert effective["entrypoints"]["fit"][-2:] == ["--kind", "fit"]  # the real unit_runner, not a fake
    assert "unit_runner" in " ".join(effective["entrypoints"]["report"])
    deploy.deploy(root, unit_dir=world["base"] / "chain" / "units", systemctl=Recorder(), start=False)
    env = {"I6DMO_TEST_TRAINING_OVERRIDES": OVERRIDES}
    results = []
    for _ in range(8):
        campaign.tick(root, systemctl=Recorder())
        progressed = False
        for role in ("preferred", "secondary", "coordinator"):
            out = W.worker_once(root, role, facts_provider=_provider(world), executor=W.LocalExecutor(),
                                poll_interval=0.2, extra_env=env)
            results.append((role, out["result"], out.get("unit_id"), out.get("cause")))
            if out["result"] not in ("IDLE", "WAIT_ARTIFACT", "WAIT_RESOURCE", "CLOSED"):
                progressed = True
        if not progressed:
            break
    q = campaign.open_queue(root)
    try:
        st = {u["unit_id"]: (u["state"], u["cause"]) for u in q.units()}
        terms = {u: q.terminal(u) for u in st}
    finally:
        q.close()
    assert all(s == "COMPLETE" for s, _ in st.values()), (st, results)
    # real trainer receipts: CPU, two epochs, scientific=false (software test), matched identity
    for arm in ("mg1", "mg2m"):
        stored = root / "terminals" / terms[f"{arm}.fit"]["identity_sha256"] / "out" / "train" / "RECEIPT.json"
        receipt = json.loads(stored.read_text())
        assert receipt["status"] == "COMPLETED" and receipt["device"] == "cpu"
        assert receipt["identity"]["matched_arm"] == arm
        assert receipt["identity"]["common_population_sha256"] == world["digest"]
        metrics_art = terms[f"{arm}.report"]["terminal"]["artifacts"]["metrics"]
        metrics = json.loads((root / "terminals" / terms[f"{arm}.report"]["identity_sha256"] / "out" /
                              metrics_art["path"]).read_text())
        assert metrics["matched"]["arm"] == arm
        assert metrics["matched"]["expected_weeks_source"] == "sealed_calendar_market_grid_validation_weeks"
        assert terms[f"{arm}.report"]["terminal"]["disposition"] == "MATCHED_PAIRED_METRICS_AGREE"
    comp = terms["comparison"]["terminal"]
    doc = json.loads((root / "terminals" / terms["comparison"]["identity_sha256"] / "out" /
                      "COMPARISON.json").read_text())
    assert doc["rule"] == "reporting.decide_matched"
    assert comp["disposition"] in ("MG1_SELECTED", "MG2_SELECTED", "MG2_RETAINED_WITHIN_TOLERANCE",
                                   "INCOMPLETE_COMPARISON")
    assert doc["verdict"]["capacity_delta"]["delta"]["heads"] == 2304
    # delivery, mirrors, automatic CLOSE and the campaign-scoped stop
    stop = Recorder()
    cid = campaign.load_plan(root)["campaign_id"]
    out = campaign.publish(root, warehouse=FakeWarehouse(), systemctl=stop,
                           self_unit=f"{deploy.prefix(cid)}publish.service")
    assert out["publication"]["state"] == "DELIVERED" and out["publication"]["mismatch"] == 0
    assert out["close"] == "CLOSED" and out["timers"] == "STOPPED"
    close = json.loads((root / "CLOSE.json").read_text())
    assert close["disposition"] == comp["disposition"] and close["trigger"] == "AUTOMATIC"
    timers = json.loads((root / "TIMERS_STOPPED.json").read_text())
    assert timers["state"] == "STOPPED" and timers["close_sha256"] == _sha(root / "CLOSE.json")
    assert stop.calls and all(c[-1].startswith(deploy.prefix(cid)) for c in stop.calls)
    assert status.compute(root, level="campaign")["state"] == "CLOSED"
    calls = len(stop.calls)
    assert campaign.tick(root, systemctl=stop)["close"] == "ALREADY_CLOSED" and len(stop.calls) == calls
    fits = [r for r in results if r[2] and r[2].endswith(".fit") and r[1] == "COMPLETE"]
    assert sorted(r[2] for r in fits) == ["mg1.fit", "mg2m.fit"]  # each arm fitted exactly once


def test_engineering_cpu_fixture_is_refused_for_the_real_sealed_designs(world, tmp_path):
    from tests.i6dmo.test_ops_matched_runner import DESIGNS, host, pins
    from tools.i6dmo import campaign
    spec_path, data, storage = host(tmp_path)
    with pytest.raises(campaign.Refused, match="ENGINEERING_CPU_FIXTURE_REQUIRES_SYNTHETIC_CALENDAR"):
        campaign.init_matched_contrast(DESIGNS, spec_path, tmp_path / "c", adapter=None, code_commit="c" * 40,
                                       data_pins=pins(data, storage), engineering_cpu_fixture=True)


def test_review_d2_cpu_fit_is_never_scientific_and_a_scientific_report_refuses_it(world):
    """Reviewer p8 g: ``train --device cpu`` WITHOUT any override stamped scientific=true (seen on a
    receipt that stopped at a wrong digest). Now a CPU fit is never scientific, and a scientific
    campaign's real report refuses a non-scientific fit typed."""
    import os
    import subprocess
    import sys
    from tests.i6dmo.test_matched_fit_path import _spec, _stored_terminal
    from tools.i6dmo import reporting
    repo = Path(__file__).resolve().parents[2]

    def train(out, digest, overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith("I6DMO_TEST_")}
        env["CUDA_VISIBLE_DEVICES"] = ""
        if overrides:
            env["I6DMO_TEST_TRAINING_OVERRIDES"] = OVERRIDES
        return subprocess.run([sys.executable, "-m", "tools.i6dmo.train", "--design", str(world["designs"]["mg2m"]),
                               "--out", str(out), "--device", "cpu", "--json",
                               "--common-population", str(world["storage"] / "population"),
                               "--common-population-digest", digest, "--input-dir", str(world["input"])],
                              capture_output=True, text=True, env=env, cwd=repo, timeout=500)
    stopped = world["base"] / "cpu_no_override" / "out" / "train"
    train(stopped, "0" * 64, overrides=False)  # the reviewer's probe: no override, wrong digest
    receipt = json.loads((stopped / "RECEIPT.json").read_text())
    assert not receipt["overrides"] and receipt["scientific"] is False
    out = world["base"] / "cpu_fit" / "out" / "train"
    run = train(out, world["digest"], overrides=True)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    model_world = dict(population=world["storage"] / "population", input=world["input"], digest=world["digest"],
                       exps=world["exps"])
    spec = _spec(model_world, "mg2m", _stored_terminal(out))
    spec["matched"]["require_scientific_fit"] = True
    refused = reporting.run_unit(spec, world["base"] / "cpu_report", world["base"] / "cpu_report.progress")
    assert refused["status"] == "FAILED" and refused["disposition"] == "NON_SCIENTIFIC_FIT"
    spec["matched"]["require_scientific_fit"] = False  # an engineering campaign may still score it
    allowed = reporting.run_unit(spec, world["base"] / "cpu_report2", world["base"] / "cpu_report2.progress")
    assert allowed["status"] == "COMPLETE"
    metrics = json.loads((world["base"] / "cpu_report2" / "METRICS.json").read_text())
    assert metrics["matched"]["scientific_fit"] is False and metrics["matched"]["fit_device"] == "cpu"
