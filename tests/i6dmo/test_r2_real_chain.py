"""R0-versus-R2 contrast, REAL chain end to end on CPU (no fake child, no injected loader).

A tiny synthetic common population is built and sealed with the data lane's REAL
``build_common_population`` (the matched fixture shape, synthetic calendar), and the R2 graph is
sealed from the COMMITTED ruled contract with ``init_r2_contrast`` (engineering CPU fixture: the
fixture design/population replace the contract's real bindings and only the ruled objective's
``max_epochs``/``patience`` are shortened, both recorded non-scientific). Then, through the
ordinary automation only:

    19 x r2.pretrain.NN : real ``unit_runner --kind pretrain`` -> ``r2_runner.run_pretrain``
        (``model.branch_pretraining_corpus`` / ``pretrain_branch`` / ``write_branch_donor_terminal``)
    r0.fit / r2.fit     : real ``python -m tools.i6dmo.train`` (R2 with ``--r2-donors`` ->
        ``train.r2_import_into`` -> ``model.import_branch_donors``)
    r0/r2.report        : real ``reporting.run_matched_unit``
    r2.probe            : real frozen predictive probe (REPORT ONLY)
    comparison          : real ``r2_design.decide_r2`` with the literal ``require_scientific=False``
    -> publish to the local test sink with exact readback -> two scratch mirrors
    -> automatic CLOSE -> TIMERS_STOPPED (this campaign's units only).

No GPU, no TEST bar, no real systemd, no production OLAP.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from tests.i6dmo import _synth  # noqa: E402
from tests.i6dmo.ops_fakes import FakeWarehouse, facts, private_spec  # noqa: E402
from tests.i6dmo.test_matched_fit_path import FIXTURE_CAL, make_sparse_input  # noqa: E402

FIT_OVERRIDES = json.dumps({"max_epochs": 2})
OBJECTIVE_OVERRIDES = {"max_epochs": 2, "patience": 1}
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/i6n/direct_multi_output_static/matched_short/R2_CONTRAST_DESIGN.ruled.json"
REAL_MG2M = ROOT / "docs/i6n/direct_multi_output_static/matched_short/EXPERIMENT.mg2m.json"


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
    base = tmp_path_factory.mktemp("r2_real")
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


def _pins(world):
    return dict(common_population=dict(base="experiment_storage", relpath="population",
                                       sha256=_sha(world["storage"] / "population" / "POPULATION.json"),
                                       population_sha256=world["digest"]),
                matched_input=dict(relpath="input", sha256=_sha(world["input"] / "INPUT.npz")))


def _spec(world, base):
    spec = private_spec(base)
    for role in spec["roles"].values():
        role.update(data_root=str(world["data"]), experiment_storage_root=str(world["storage"]))
    path = base / "private_spec.json"
    path.write_text(json.dumps(spec))
    return path


def _campaign(world, name, **kw):
    from tools.i6dmo import campaign
    base = world["base"] / name
    base.mkdir()
    spec_path = _spec(world, base)
    root = base / "campaign"
    campaign.init_r2_contrast(CONTRACT, world["designs"]["mg2m"], spec_path, root, code_commit="c" * 40,
                              data_pins=_pins(world), engineering_cpu_fixture=True,
                              engineering_objective_overrides=OBJECTIVE_OVERRIDES, **kw)
    return root, spec_path


def _provider(world):
    from tools.i6dmo import matched_runner as MR
    runtime = MR.runtime_pin(world["exps"]["mg2m"])

    def probe(role):
        doc = facts()
        doc["runtime"] = dict(runtime)
        return doc
    return probe


def _drive(root, world, rounds=14):
    from tools.i6dmo import campaign, worker as W
    env = {"I6DMO_TEST_TRAINING_OVERRIDES": FIT_OVERRIDES}
    results = []
    for _ in range(rounds):
        campaign.tick(root, systemctl=Recorder())
        progressed = False
        for role in ("preferred", "secondary", "coordinator"):
            for _ in range(4):
                out = W.worker_once(root, role, facts_provider=_provider(world), executor=W.LocalExecutor(),
                                    poll_interval=0.2, extra_env=env)
                results.append((role, out["result"], out.get("unit_id"), out.get("cause")))
                if out["result"] in ("IDLE", "WAIT_ARTIFACT", "WAIT_RESOURCE", "CLOSED"):
                    break
                progressed = True
        if not progressed:
            break
    return results


def _stored(root, term, art):
    return root / "terminals" / term["identity_sha256"] / "out" / term["terminal"]["artifacts"][art]["path"]


@pytest.fixture(scope="module")
def chain(world):
    """The whole graph run once through the real automation (shared by the assertions below)."""
    from tools.i6dmo import campaign
    root, spec_path = _campaign(world, "chain")
    from tools.i6dmo import deploy
    deploy.deploy(root, unit_dir=world["base"] / "chain" / "units", systemctl=Recorder(), start=False)
    results = _drive(root, world)
    q = campaign.open_queue(root)
    try:
        states = {u["unit_id"]: (u["state"], u["cause"]) for u in q.units()}
        terms = {u: q.terminal(u) for u in states}
        attempts = q.attempts()
    finally:
        q.close()
    return dict(root=root, spec=spec_path, results=results, states=states, terms=terms, attempts=attempts)


def test_r2_contrast_real_chain_pretrain_import_fit_report_decide_publish_close_stop(world, chain):
    from tools.i6dmo import campaign, deploy, model as M, status
    root, terms = chain["root"], chain["terms"]
    effective = campaign.load_effective(root)
    plan = campaign.load_plan(root)
    assert plan["mode"] == "R2_CONTRAST" and effective["engineering_cpu_fixture"] is True
    assert effective["r2"]["engineering_objective_overrides"] == OBJECTIVE_OVERRIDES
    assert effective["r2"]["contract_seal_sha256"] == json.loads(CONTRACT.read_text())["seal_sha256"]
    for kind in ("fit", "report", "pretrain", "evaluation", "comparison"):
        assert "unit_runner" in " ".join(effective["entrypoints"][kind])  # the real adapters, no fake
    assert all(s == "COMPLETE" for s, _ in chain["states"].values()), (chain["states"], chain["results"])
    features = campaign.r2_features(effective)
    assert len(features) == 19
    # 19 pretraining units: the real model-lane body, seed rule, donor terminal retained by the store
    for index, feature in enumerate(features):
        term = terms[campaign.r2_pretrain_unit_id(index)]
        donor = json.loads(_stored(root, term, "donor_terminal").read_text())
        assert term["terminal"]["artifacts"]["donor_terminal"]["sha256"] == _sha(_stored(root, term, "donor_terminal"))
        assert donor["schema"] == M.R2_PRETRAIN_TERMINAL_SCHEMA and donor["feature"] == feature
        assert donor["applied_seed"] == M.r2_seed(index) == 6102026 + index
        assert donor["campaign_id"] == plan["campaign_id"] and donor["code_pin"] == "c" * 40
        assert donor["objective"]["fit_settings"]["max_epochs"] == 2
        assert donor["learned_corpus"]["sha256"] == term["terminal"]["learned_corpus_sha256"]
    # R2 fit: imports exactly these 19 donors and binds the import record; R0 carries no import
    r2_receipt = json.loads(_stored(root, terms["r2.fit"], "RECEIPT").read_text())
    r0_receipt = json.loads(_stored(root, terms["r0.fit"], "RECEIPT").read_text())
    record = json.loads(_stored(root, terms["r2.fit"], "R2_IMPORT.json").read_text())
    assert r2_receipt["identity"]["r2_import_sha256"] == record["import_sha256"]
    assert "r2_import_sha256" not in r0_receipt["identity"]
    assert r0_receipt["device"] == r2_receipt["device"] == "cpu" and r2_receipt["scientific"] is False
    assert record["donor_terminals"] == {
        f: terms[campaign.r2_pretrain_unit_id(i)]["terminal"]["artifacts"]["donor_terminal"]["sha256"]
        for i, f in enumerate(features)}
    # the decision: the ONE rule, literal non-scientific relaxation, both reports authenticated
    comp = json.loads(_stored(root, terms["comparison"], "comparison").read_text())
    assert comp["rule"] == "tools.i6dmo.r2_design.decide_r2" and comp["require_scientific"] is False
    assert set(comp["inputs"]) == {"r0.report", "r2.report"}  # the probe never reaches the decision
    assert comp["verdict"]["r2_import_sha256"] == record["import_sha256"]
    assert terms["comparison"]["terminal"]["disposition"] in ("R2_SELECTED", "R0_RETAINED", "INCOMPLETE_COMPARISON")
    probe = json.loads(_stored(root, terms["r2.probe"], "probe").read_text())
    assert probe["use"] == "REPORT_ONLY_NEVER_A_GATE_NEVER_SELECTS"
    # every unit ran exactly once (no repeat of a completed pretraining or fit)
    done = [r[2] for r in chain["results"] if r[1] == "COMPLETE"]
    assert sorted(done) == sorted(chain["states"]) and len(done) == len(set(done)) == 25
    # separate OLAP facts
    facts_seen = {r["tags"].get("fact") for t in terms.values() for r in t["terminal"].get("reports", [])}
    assert {"r2.pretraining.loss_curve", "r2.pretraining.reconstruction", "r2.pretraining.compute_cost",
            "r2.pretraining.predictive_probe", "r0.fit.fit_inner_diagnostics", "r2.fit.fit_inner_diagnostics",
            "r0_r2.fit.compute_cost", "r0_r2.development.per_head_metrics", "r0_r2.decision"} <= facts_seen
    # status at every level before CLOSE
    st = status.compute(root, level="campaign")
    assert st["r2"]["pretraining"] == dict(complete=19, denominator=19, failed=[])
    assert set(st["regimes"]) == {"R2_PRETRAIN", "R0", "R2", "R2_PROBE", "DECISION"}
    stages = status.compute(root, level="stage")["stages"]
    assert [s["stage"] for s in stages] == list(campaign.R2_STAGES)
    assert status.compute(root, level="arm", arm="R2_PRETRAIN")["arms"]["R2_PRETRAIN"]["denominator"] == 19
    assert status.compute(root, level="unit", unit="r2.pretrain.07")["state"] == "COMPLETE"
    # delivery with exact readback, mirrors, automatic CLOSE and the campaign-scoped stop
    stop = Recorder()
    cid = plan["campaign_id"]
    out = campaign.publish(root, warehouse=FakeWarehouse(), systemctl=stop,
                           self_unit=f"{deploy.prefix(cid)}publish.service")
    assert out["publication"]["state"] == "DELIVERED" and out["publication"]["mismatch"] == 0
    assert out["close"] == "CLOSED" and out["timers"] == "STOPPED"
    close = json.loads((root / "CLOSE.json").read_text())
    assert close["disposition"] == terms["comparison"]["terminal"]["disposition"] and close["trigger"] == "AUTOMATIC"
    timers = json.loads((root / "TIMERS_STOPPED.json").read_text())
    assert timers["state"] == "STOPPED" and timers["close_sha256"] == _sha(root / "CLOSE.json")
    assert stop.calls and all(c[-1].startswith(deploy.prefix(cid)) for c in stop.calls)
    assert status.compute(root, level="campaign")["state"] == "CLOSED"
    calls = len(stop.calls)
    assert campaign.tick(root, systemctl=stop)["close"] == "ALREADY_CLOSED" and len(stop.calls) == calls


def _unit_spec(chain, unit_id):
    root = chain["root"]
    attempt = next(a for a in chain["attempts"] if a["unit_id"] == unit_id and a["outcome"] == "COMPLETE")
    return json.loads((root / attempt["attempt_dir"] / "UNIT_SPEC.json").read_text())


def test_pretraining_and_probe_never_open_development_or_test(chain, tmp_path, monkeypatch):
    from tools.i6dmo import r2_runner, train as T
    opened = []
    real = T.arm_split

    def spy(population, design, split):
        opened.append(split)
        return real(population, design, split)
    monkeypatch.setattr(T, "arm_split", spy)
    for unit_id, body in (("r2.pretrain.03", r2_runner.run_pretrain), ("r2.probe", r2_runner.run_probe)):
        spec = _unit_spec(chain, unit_id)
        out = tmp_path / unit_id / "out"
        doc = body(spec, out, tmp_path / unit_id / "progress.json")
        assert doc["status"] == "COMPLETE", doc
    assert opened and set(opened) <= {"FIT", "INNER"}
    with pytest.raises(ValueError, match="R2_SPLIT_FORBIDDEN"):
        r2_runner.split_tensors(None, {}, "DEVELOPMENT")
    with pytest.raises(ValueError, match="R2_SPLIT_FORBIDDEN"):
        r2_runner.split_tensors(None, {}, "TEST")


def test_pretraining_replay_is_bitwise_and_budget_stop_is_typed(chain, tmp_path, monkeypatch):
    from tools.i6dmo import model as M, r2_runner
    spec = _unit_spec(chain, "r2.pretrain.05")
    doc = r2_runner.run_pretrain(spec, tmp_path / "a" / "out", tmp_path / "a" / "p.json")
    stored = chain["terms"]["r2.pretrain.05"]
    # CPU replay: the donor weights are bitwise identical (the terminal also records wall-clock timings)
    assert doc["artifacts"]["donor.npz"]["sha256"] == stored["terminal"]["artifacts"]["donor.npz"]["sha256"]
    assert doc["learned_corpus_sha256"] == stored["terminal"]["learned_corpus_sha256"]

    def budget(*a, **k):
        raise M.DonorImportError("R2_PRETRAINING_BUDGET_STOP", "max_updates reached")
    monkeypatch.setattr(M, "pretrain_branch", budget)
    doc = r2_runner.run_pretrain(spec, tmp_path / "b" / "out", tmp_path / "b" / "p.json")
    assert doc["status"] == "FAILED" and doc["disposition"] == "R2_PRETRAINING_BUDGET_STOP"
    assert doc["costs"]["fit_seconds"] > 0 and not doc["artifacts"]  # started: never an INFRA wait, no donor


def test_failed_pretraining_blocks_r2_and_closes_incomplete_never_r0_only(world, chain, tmp_path):
    from tools.i6dmo import campaign, r2_runner
    root, _ = _campaign(world, "budget")
    q = campaign.open_queue(root)
    try:
        claim = q.claim("r2.pretrain.03", owner="test", role="coordinator", host_key="h", boot_id="b",
                        device_class="cpu", heavy=False, ttl=60, cap_bytes=1)
        q.fail("r2.pretrain.03", claim["token"], "FAILED", "TYPED:R2_PRETRAINING_BUDGET_STOP")
    finally:
        q.close()
    for _ in range(3):  # one dependency level per tick
        campaign.tick(root, systemctl=Recorder())
    q = campaign.open_queue(root)
    try:
        st = {u["unit_id"]: u["state"] for u in q.units()}
    finally:
        q.close()
    assert st["r2.fit"] == st["r2.report"] == st["r2.probe"] == "BLOCKED"
    assert st["comparison"] == "WAIT_DEPENDENCY"  # still waits for the R0 report to be terminal
    # the comparison with ONLY an authenticated R0 report: INCOMPLETE_COMPARISON, never R0 selected
    spec = _unit_spec(chain, "comparison")
    spec["upstream"] = {"r0.report": spec["upstream"]["r0.report"], "r2.report": None, "r2.fit": None}
    doc = r2_runner.run_comparison(spec, tmp_path / "out", tmp_path / "p.json")
    assert doc["status"] == "COMPLETE" and doc["disposition"] == "INCOMPLETE_COMPARISON"
    assert doc["gates"]["selected"] is None


def test_probe_never_gates_or_selects(chain, tmp_path):
    from tools.i6dmo import campaign, r2_runner
    plan = campaign.load_plan(chain["root"])
    assert not [u["unit_id"] for u in plan["units"] if "r2.probe" in u["depends_on"]]
    spec = _unit_spec(chain, "comparison")
    assert "r2.probe" not in spec["upstream"]
    first = r2_runner.run_comparison(spec, tmp_path / "a", tmp_path / "pa.json")
    # corrupt the stored probe: the decision is byte-identical (it never reads it)
    probe = _stored(chain["root"], chain["terms"]["r2.probe"], "probe")
    saved = probe.read_bytes()
    try:
        probe.write_text(json.dumps({"probe_gain_equal_head_mean": 1e9}))
        second = r2_runner.run_comparison(spec, tmp_path / "b", tmp_path / "pb.json")
    finally:
        probe.write_bytes(saved)
    assert first["disposition"] == second["disposition"]
    assert (tmp_path / "a" / "COMPARISON.json").read_bytes() == (tmp_path / "b" / "COMPARISON.json").read_bytes()


def test_scientific_path_refuses_cpu_fits(world, chain, tmp_path):
    from tools.i6dmo import campaign, r2_runner, worker as W
    # a real (non-synthetic) design can never be an engineering CPU fixture
    spec_path = _spec(world, tmp_path)
    with pytest.raises(campaign.Refused, match="ENGINEERING_CPU_FIXTURE_REQUIRES_SYNTHETIC_CALENDAR"):
        campaign.init_r2_contrast(CONTRACT, REAL_MG2M, spec_path, tmp_path / "x", code_commit="c" * 40,
                                  data_pins=_pins(world), engineering_cpu_fixture=True)
    # a scientific campaign runs the ruled objective exactly
    with pytest.raises(campaign.Refused, match="R2_OBJECTIVE_OVERRIDE"):
        campaign.init_r2_contrast(CONTRACT, REAL_MG2M, spec_path, tmp_path / "y", code_commit="c" * 40,
                                  data_pins=_pins(world), engineering_objective_overrides={"max_epochs": 2})
    # and binds the contract's population
    with pytest.raises(campaign.Refused, match="R2_CONTRACT_BINDING"):
        campaign.init_r2_contrast(CONTRACT, REAL_MG2M, spec_path, tmp_path / "z", code_commit="c" * 40,
                                  data_pins=_pins(world))
    # the decision refuses the CPU fits when the campaign is scientific (literal relaxation only)
    spec = _unit_spec(chain, "comparison")
    spec["effective_config"] = dict(spec["effective_config"], engineering_cpu_fixture=False)
    doc = r2_runner.run_comparison(spec, tmp_path / "out", tmp_path / "p.json")
    assert doc["status"] == "FAILED" and doc["disposition"].startswith("R2_NON_SCIENTIFIC_FIT")
    # and the worker never admits a CPU fit/pretraining unit in a scientific R2 campaign
    sci = dict(campaign.load_effective(chain["root"]), engineering_cpu_fixture=False)
    unit = dict(campaign.load_plan(chain["root"])["units"][0])
    assert W.scientific_device_problem(sci, unit).startswith("R2_SCIENTIFIC_CPU_FIT_REFUSED")
    assert W.scientific_device_problem(sci, dict(unit, device="gpu")) is None


def test_scientific_init_binds_contract_and_seals_separate_admission_classes(world, tmp_path):
    from tools.i6dmo import campaign, worker as W
    contract = json.loads(CONTRACT.read_text())
    pins = _pins(world)
    pins["common_population"]["population_sha256"] = contract["shared"]["population_sha256"]
    out = campaign.init_r2_contrast(CONTRACT, REAL_MG2M, _spec(world, tmp_path), tmp_path / "sci",
                                    code_commit="c" * 40, data_pins=pins)
    assert out["fit_device"] == "gpu" and out["contract_binding"].startswith("SCIENTIFIC")
    effective = campaign.load_effective(tmp_path / "sci")
    plan = campaign.load_plan(tmp_path / "sci")
    units = {u["unit_id"]: u for u in plan["units"]}
    assert all(u["device"] == "gpu" for u in units.values() if u["kind"] in ("fit", "pretrain"))
    classes = {f"{u['resource_class']}.{u['kind']}" for u in units.values() if u["kind"] in ("fit", "pretrain")}
    assert classes == {"r2_branch_pretrain.pretrain", "r2_contrast_r0_fit.fit", "r2_contrast_r2_fit.fit"}
    assert set(effective["class_caps"]) == classes
    assert all(c["cap_bytes"] > 0 and c["basis"] for c in effective["class_caps"].values())
    assert effective["r2"]["objective"] == contract["pretraining"]["objective"]
    assert units["r2.fit"]["depends_on"] == [f"r2.pretrain.{i:02d}" for i in range(19)]
    assert units["r0.fit"]["depends_on"] == [] and "r2.probe" not in units["comparison"]["depends_on"]
    assert not any(u["test_open"] for u in units.values())
    with pytest.raises(campaign.Refused, match="TEST_NOT_ENUMERABLE"):
        campaign.enumerate_test_units(tmp_path / "sci")
    # an R2 fit waits (typed) until all 19 donor references are stored; never launched as an R0
    assert W.r2_donor_problem(tmp_path / "sci", effective, units["r2.fit"]).startswith(
        "R2_DONOR_REFERENCES_INCOMPLETE")
    assert W.r2_donor_problem(tmp_path / "sci", effective, units["r0.fit"]) is None


def test_restart_duplicate_claim_and_outbox_retry_never_repeat(world, chain, tmp_path):
    """On a copy of the completed chain: a queue restored without its terminal rows ADOPTS the stored
    terminals (no child), a row left inconsistent with its own terminal is repaired without a launch,
    a duplicate pass claims nothing, and an outbox retry re-enqueues nothing."""
    import sqlite3
    from tools.i6dmo import campaign, worker as W
    from tools.i6dmo.outbox import Outbox
    copy = tmp_path / "copy"
    shutil.copytree(chain["root"], copy, ignore=shutil.ignore_patterns("CLOSE.json", "TIMERS_STOPPED.json"))
    for name in ("CLOSE.json", "TIMERS_STOPPED.json"):
        assert not (copy / name).exists()
    shutil.rmtree(copy / "backup", ignore_errors=True)
    q = campaign.open_queue(copy)
    try:
        before = len(q.attempts())
        q.set_state("r2.pretrain.01", "INTERRUPTED", "SIMULATED_INCONSISTENT_ROW")
    finally:
        q.close()
    db = sqlite3.connect(copy / "state" / "queue.sqlite3")
    try:  # a queue restored from an earlier backup: the stored terminals exist, the rows do not
        for unit_id in ("r2.pretrain.00", "r0.fit", "r2.fit"):
            db.execute("DELETE FROM terminals WHERE unit_id=?", (unit_id,))
            db.execute("UPDATE units SET state='QUEUED', cause=NULL WHERE unit_id=?", (unit_id,))
        db.commit()
    finally:
        db.close()

    class NoLaunch(W.LocalExecutor):
        def start(self, **kw):
            raise AssertionError("a completed pretraining or fit was launched again")
    seen = []
    for _ in range(8):
        out = W.worker_once(copy, "coordinator", facts_provider=_provider(world), executor=NoLaunch(),
                            poll_interval=0.1)
        seen.append((out["result"], out.get("unit_id")))
        if out["result"] == "IDLE":
            break
    assert sorted(u for r, u in seen if r == "ADOPTED") == ["r0.fit", "r2.fit", "r2.pretrain.00"]
    assert [u for r, u in seen if r == "ALREADY_TERMINAL"] == ["r2.pretrain.01"]
    assert W.worker_once(copy, "preferred", facts_provider=_provider(world), executor=NoLaunch())["result"] == "IDLE"
    q = campaign.open_queue(copy)
    try:
        assert len(q.attempts()) == before
    finally:
        q.close()
    box = Outbox(copy / "state" / "outbox.sqlite3")
    try:
        items = len(box.items())
        for unit_id in ("r2.pretrain.00", "r0.fit"):
            for report in chain["terms"][unit_id]["terminal"]["reports"]:
                box.enqueue(report, source_unit=unit_id)  # an outbox retry: same digest, no new item
        assert len(box.items()) == items
    finally:
        box.close()


def test_r2_fit_imports_only_this_campaigns_authenticated_donors(chain, tmp_path):
    from tools.i6dmo import model as M, train as T, unit_runner
    spec = _unit_spec(chain, "r2.fit")
    donors = spec["r2_donors"]
    feature = next(iter(donors["donors"]))
    cp = dict(path=spec["pins"]["common_population"]["path"],
              expected_digest=spec["pins"]["common_population"]["population_sha256"],
              input_dir=spec["pins"]["matched_input"]["path"])
    design = tmp_path / "design.json"
    design.write_text(json.dumps(spec["effective_config"]["design"]))
    effective = T.effective_from_design(str(design))

    def attempt(mutated):
        bundle = M.build_dmo(effective["model"])
        with pytest.raises(T.TrainDisposition) as err:
            T.r2_import_into(bundle, cp, effective, mutated)
        return err.value.code
    forged = json.loads(json.dumps(donors))
    forged["donors"][feature]["terminal"]["sha256"] = "0" * 64
    assert attempt(forged) == "R2_DONOR_TERMINAL_DIGEST"
    other = json.loads(json.dumps(donors))
    other["provenance"]["campaign_id"] = "i6dmo-000000000000"
    assert attempt(other) == "R2_DONOR_PROVENANCE_MISMATCH"
    missing = json.loads(json.dumps(donors))
    missing["donors"].pop(feature)
    assert attempt(missing) == "R2_DONOR_SET"
    # the unit adapter: an R2 fit without donors never runs (it would be an R0); R0 never imports
    for regime, payload, code in (("R2", None, "R2_DONOR_REFERENCES_MISSING"), ("R0", donors, "R2_R0_HAS_IMPORT")):
        out = tmp_path / regime / "out"
        out.parent.mkdir()
        assert unit_runner.run_fit(dict(spec, regime=regime, r2_donors=payload), out, tmp_path / "p.json") == 0
        assert json.loads((out / "TERMINAL.json").read_text())["disposition"] == code
