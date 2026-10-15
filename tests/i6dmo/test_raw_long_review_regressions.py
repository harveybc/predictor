"""Independent review F1-F6: required behavior, not expected-failure snapshots.

Run with --noconftest, hidden CUDA and crispdm-run -m 2G -t 180s.
All artifacts are synthetic and temporary. No fit, service or network is used.
"""

import copy
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from tests.i6dmo import test_raw_pair as pair_fixtures
from tests.i6dmo import test_ops_matched_runner as ops_fixtures
from tests.i6dmo.ops_fakes import FakeWarehouse
from tools.i6dmo import campaign as C, common_population as CP, design as D
from tools.i6dmo import deploy, unit_runner, worker as W
from tools.i6dmo import matched_runner as MR, raw_long_campaign as L
from tools.i6dmo import raw_pair as RP, reporting as R, status as S


@pytest.fixture
def pair_env(tmp_path):
    return pair_fixtures.env.__wrapped__(tmp_path)


def _replace_child(pin, change):
    path = Path(pin["path"])
    stored = json.loads(path.read_text())
    change(stored["terminal"])
    stored["child_terminal_sha256"] = pair_fixtures.write(
        path.parent / "CHILD_TERMINAL.json", stored["terminal"])["sha256"]
    pin.update(pair_fixtures.write(path, stored))


def _partial_short(metrics):
    row = metrics["per_horizon"]["2"]
    row.update(n_declared=row["n"] + 1, n_failed=1, state="INCOMPLETE")
    metrics["matched"]["development_counts"]["2"] = row["n_declared"]
    metrics["matched"]["selection"].update(
        status="INCOMPLETE", score=None,
        failed_heads=[dict(horizon=2, n_failed=1)])
    metrics.setdefault("c2", {})["state"] = "INCOMPLETE"


@pytest.mark.parametrize("corruption", [
    "incomplete_c2", "missing_c2", "disagreement", "lineage_c2", "partial_support",
])
def test_f1_short_metadata_refused_without_reading_prediction_bytes(
        pair_env, corruption, monkeypatch):
    pin = pair_env[0]["effective_config"]["raw_pair"]["short_report"]
    original_open = Path.open

    def no_arrays(path, *args, **kwargs):
        assert path.suffix != ".npz", "metadata validation must not open arrays"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_arrays)
    accepted = RP._report("mg2m.report", pin, "mg2m")
    assert accepted["record"]["prediction_bytes_verified"] is False

    if corruption == "lineage_c2":
        def change(child):
            child["lineage"].setdefault("c2", {})["sha256"] = "0" * 64
        _replace_child(pin, change)
    else:
        def change(metrics):
            if corruption == "incomplete_c2":
                metrics.setdefault("c2", {})["state"] = "INCOMPLETE"
            elif corruption == "missing_c2":
                metrics.pop("c2", None)
            elif corruption == "disagreement":
                metrics.setdefault("agreement", {})["status"] = "DISAGREE"
            else:
                _partial_short(metrics)
        pair_fixtures.replace_artifact(pin, "metrics", change)

    with pytest.raises((RP.Refusal, MR._Refusal, R.ReportFailure,
                        RP.SA._Refusal, RP.SA.StrategyAdapterError)):
        RP._report("mg2m.report", pin, "mg2m")


def test_f1_bad_short_c2_cannot_reach_positive_execution(
        pair_env, tmp_path, monkeypatch):
    pair_fixtures._execution_fixture(pair_env, tmp_path)
    control = RP.run_unit(pair_env[0], tmp_path / "control-out", tmp_path / "control-progress.json")
    assert (control["status"], control["gates"]["pair_status"]) == ("COMPLETE", "PASS"), control

    pin = pair_env[0]["effective_config"]["raw_pair"]["short_report"]

    def contradict(metrics):
        metrics.setdefault("c2", {})["state"] = "INCOMPLETE"
        metrics.setdefault("agreement", {})["status"] = "DISAGREE"

    pair_fixtures.replace_artifact(pin, "metrics", contradict)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid C2 must be refused before predictions, bars or strategy")

    monkeypatch.setattr(RP, "_predictions", forbidden)
    for name in ("_bars", "load_pinned_policy", "run_backtest"):
        monkeypatch.setattr(RP.SA, name, forbidden)
    result = RP.run_unit(*pair_env)
    assert result["status"] == "FAILED", result
    assert result["disposition"] != "RAW_PAIR_MISSING_ARTIFACT", result
    assert result["gates"].get("pair_status") not in ("ELIGIBLE", "PASS"), result
    assert "backtest" not in result["artifacts"]


@pytest.mark.parametrize("state", ["RUNNING", "CLOSED"])
def test_f2_existing_status_merges_a_real_campaign_state_once(state, tmp_path, monkeypatch):
    counts = dict(COMPLETE=6 if state == "CLOSED" else 0)
    monkeypatch.setattr(S, "compute", lambda *a, **k: dict(state=state, counts=counts))
    monkeypatch.setattr(C, "closed_doc", lambda *a: dict(state="CLOSED") if state == "CLOSED" else None)
    result = L._existing_status(tmp_path)
    assert result["counts"] == counts
    allowed = ("CLOSED",) if state == "CLOSED" else ("RUNNING", "DISPATCHED")
    assert result["state"] in allowed


@pytest.mark.parametrize("outcome", [
    "MG1_LONG_SELECTED", "MG12_SELECTED", "MG12_RETAINED_WITHIN_TOLERANCE",
    "PASS", "PAIR_NOT_ELIGIBLE", "RAW_PAIR_MISSING_ARTIFACT",
])
def test_f3_long_readiness_is_closable_only_for_scientific_outcomes(
        outcome, tmp_path, monkeypatch):
    is_pair = outcome in ("PASS", "PAIR_NOT_ELIGIBLE", "RAW_PAIR_MISSING_ARTIFACT")
    failed = outcome == "RAW_PAIR_MISSING_ARTIFACT"
    effective = dict(mode="MATCHED_LONG_CONTRAST", raw_pair={} if is_pair else None)
    gates = {"pair_status" if is_pair else "comparison_status": outcome}
    child = dict(status="FAILED" if failed else "COMPLETE", disposition=outcome, gates=gates)
    queue = SimpleNamespace(
        unit=lambda *a: dict(state="FAILED" if failed else "COMPLETE", cause=outcome),
        terminal=lambda *a: dict(terminal=child), events=lambda *a: [], close=lambda: None)
    monkeypatch.setattr(C, "load_effective", lambda *a: effective)
    monkeypatch.setattr(C, "open_queue", lambda *a: queue)
    monkeypatch.setattr(C, "readiness", C._matched_readiness)
    monkeypatch.setattr(C, "_outbox_readback", lambda *a: (None, []))
    monkeypatch.setattr(C, "_read_json_file", lambda *a: None)
    problems, ready, _ = C.closure_problems(tmp_path)
    assert ready["state"] == "TERMINAL"
    disposition_blocks = [p for p in problems if p.startswith("DISPOSITION_NOT_AUTO_CLOSABLE:")]
    assert bool(disposition_blocks) is failed, (ready, problems)
    # Delivery and mirrors are deliberately absent: this is only the disposition gate.
    assert "PUBLICATION_NOT_DELIVERED" in problems


def test_f4_same_head_modified_runtime_is_rejected_without_remote_execution(tmp_path, monkeypatch):
    code = "c" * 40
    data = tmp_path / "data"
    role = dict(workdir=str(tmp_path / "old-runtime"), data_root=str(data),
                experiment_storage_root=str(tmp_path / "storage"), python=sys.executable)
    spec = dict(roles=dict(coordinator=role), coordinator=dict(workdir=role["workdir"]))
    prepared = dict(launch_config=dict(input_dir=str(data / "input")),
                    population_dir=str(tmp_path / "population"), population_relpath="population")
    calls = []

    def remote(setting, argv, timeout=180):
        calls.append(list(argv))
        if argv[-2:] == ["rev-parse", "HEAD"]:
            return code
        if argv[0] == "python3" and "is_dir()" in argv[-1]:
            return "1"
        if "status" in argv:
            return " M tools/i6dmo/model.py\n"
        # Any requested byte verification sees an unreviewed synthetic runtime.
        raise ValueError("REVIEW_SOURCE_CHANGED: synthetic existing runtime")

    monkeypatch.setattr(L, "_remote", remote)
    monkeypatch.setattr(L, "_run", lambda *a, **k: pytest.fail("no real provisioning command allowed"))
    monkeypatch.setattr(L, "verify_release", lambda: dict(code_files={"tools/i6dmo/model.py": "a" * 64}))
    with pytest.raises(ValueError, match="REVIEW|SOURCE|RUNTIME|DIRTY"):
        L._provision(spec, prepared, code)
    assert calls


def _launch_config(tmp_path):
    return dict(schema="i6dmo.raw_long_launch.v1", work_root=str(tmp_path / "work"),
                source_deployment=str(tmp_path / "source-deployment.json"),
                input_dir=str(tmp_path / "input"), short_campaign=str(tmp_path / "short"),
                strategy_inputs=str(tmp_path / "strategy"))


def test_f5_orphan_population_requires_the_durable_expected_digest(tmp_path, monkeypatch):
    config = _launch_config(tmp_path)
    work, storage = Path(config["work_root"]), tmp_path / "storage"
    work.mkdir()
    population = storage / "i6dmo" / work.name / "population"
    population.mkdir(parents=True)
    (population / "POPULATION.json").write_text("{}")
    digest = "a" * 64
    (work / "POPULATION_SEAL.json").write_text(json.dumps(dict(population_digest=digest)))
    source = Path(config["input_dir"]) / "INPUT.npz"
    source.parent.mkdir()
    source.write_bytes(b"synthetic source pin; never loaded as training data")
    docs = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static"
    D.seal_matched_long(docs / "EXPERIMENT.short.json", work / "designs")
    seals = {arm: json.loads((work / "designs" / f"EXPERIMENT.{arm}.json").read_text())["seal_sha256"]
             for arm in MR.LONG_ARMS}
    (work / "POPULATION_BUILD.json").write_text(json.dumps(dict(
        population_digest=digest, input_sha256=MR._sha(source),
        design_seals=seals, generation_code="c" * 40)))
    Path(config["source_deployment"]).write_text(json.dumps(dict(
        roles=dict(coordinator=dict(experiment_storage_root=str(storage))))))
    monkeypatch.setattr(L, "_run", lambda *a, **k: "")
    monkeypatch.setattr(MR, "_inside_git", lambda *a: False)

    class ReachedVerifiedRecovery(Exception):
        pass

    def verify(path, *, expected_digest=None, input_dir=None):
        assert Path(path) == population
        assert expected_digest == digest, "orphan recovery must use the durable expected pin"
        assert input_dir == config["input_dir"]
        raise ReachedVerifiedRecovery

    monkeypatch.setattr(CP, "load_common_population", verify)
    with pytest.raises(ReachedVerifiedRecovery):
        L.prepare(config)


def _long_capacity(arm):
    hourly = arm == "mg1l"
    counts = dict(slots=145 if hourly else 13, features=19, core_input_channels=78,
                  branches=304, per_branch=16, fusion=0, time_encoder=0,
                  core=1980, core_layers=[1880, 100], per_head=4657 if hourly else 433,
                  per_head_hidden=4648 if hourly else 424, per_head_output=9,
                  heads=27942 if hourly else 2598, total=30226 if hourly else 4882)
    return dict(schema="i6dmo.capacity.v1", arm=arm, counts=counts)


@pytest.fixture
def long_metrics(tmp_path):
    metrics = []
    for arm, better in (("mg1l", {18}), ("mg12m", {18, 24})):
        pin = pair_fixtures.report(tmp_path, arm, better)
        doc, _ = MR._authenticated_report(arm + ".report", pin, None)
        doc["matched"]["capacity"] = _long_capacity(arm)
        metrics.append(doc)
    control = R.decide_matched(*metrics)
    assert control["decision"] == "LOWER_SCORE" and control["selected"] == "mg12m", control
    return metrics


@pytest.mark.parametrize("corruption", ["missing_head", "forged_score", "missing_support"])
def test_f6_comparison_rejects_forged_declarations(long_metrics, corruption):
    a, b = copy.deepcopy(long_metrics)
    if corruption == "missing_head":
        del a["per_horizon"]["72"]
    elif corruption == "forged_score":
        a["matched"]["selection"]["score"] = -0.9
    else:
        for doc in (a, b):
            doc["matched"].pop("development_row_ids_sha256", None)
    with pytest.raises(R.ReportFailure):
        R.decide_matched(a, b)


@pytest.fixture
def resume_fixture(tmp_path, monkeypatch):
    config = _launch_config(tmp_path)
    work = Path(config["work_root"])
    campaign = work / "campaign"
    campaign.mkdir(parents=True)
    Path(config["source_deployment"]).write_text("{}")
    spec = dict(roles=dict(coordinator=dict(workdir=str(tmp_path / "runtime"))))
    prepared = dict(launch_config=config, population_digest="a" * 64, raw_pair=dict(policy="fixture"),
                    source_deployment_sha256=MR._sha(config["source_deployment"]))
    (work / "PREPARED.json").write_text(json.dumps(prepared))
    effective = dict(mode=C.MODE_MATCHED_LONG, code_commit="c" * 40, raw_pair=prepared["raw_pair"],
                     data_pins=dict(common_population=dict(population_sha256=prepared["population_digest"])),
                     deployment_sha256=C.sha256_json(spec))
    monkeypatch.setattr(L, "verify_release", lambda: dict(code_files={}))
    monkeypatch.setattr(C, "load_effective", lambda *a: effective)
    monkeypatch.setattr(C, "load_deployment", lambda *a: spec)
    monkeypatch.setattr(C, "closed_doc", lambda *a: None)
    monkeypatch.setattr(C, "_finalized", lambda *a: False)
    monkeypatch.setattr(S, "compute", lambda *a, **k: dict(state="IN_PROGRESS"))
    monkeypatch.setattr(L, "prepare", lambda *a: pytest.fail("resume must not regenerate or refit"))
    calls = []
    monkeypatch.setattr(L, "_verify_runtime", lambda *a: calls.append("attest"))
    monkeypatch.setattr(deploy, "deploy", lambda *a: calls.append("deploy"))
    return config, campaign, effective, spec, calls


@pytest.mark.parametrize("state", ["already_deployed", "technical_finalized", "init_before_deploy"])
def test_f4_resume_only_enables_an_attested_unfinished_deployment(resume_fixture, state, monkeypatch):
    config, campaign, _, _, calls = resume_fixture
    if state == "already_deployed":
        (campaign / "DEPLOY.json").write_text("{}")
    elif state == "technical_finalized":
        monkeypatch.setattr(C, "_finalized", lambda *a: True)
    result = L._resume(config, campaign)
    assert calls == (["attest", "deploy"] if state == "init_before_deploy" else [])
    if state == "technical_finalized":
        assert result["state"] == "TECHNICAL_TERMINAL"


@pytest.mark.parametrize("changed", ["config", "population", "raw_pair", "deployment", "source_spec"])
def test_f4_resume_identity_mismatch_never_attests_or_deploys(resume_fixture, changed):
    config, campaign, effective, spec, calls = resume_fixture
    if changed == "config":
        config = dict(config, input_dir=config["input_dir"] + "-other")
    elif changed == "population":
        effective["data_pins"]["common_population"]["population_sha256"] = "b" * 64
    elif changed == "raw_pair":
        effective["raw_pair"] = dict(policy="different")
    elif changed == "deployment":
        spec["roles"]["coordinator"]["workdir"] += "-other"
    else:
        Path(config["source_deployment"]).write_text('{"changed": true}')
    with pytest.raises(ValueError, match="RESUME_IDENTITY_MISMATCH"):
        L._resume(config, campaign)
    assert calls == []


def test_prepare_dirty_source_refuses_before_materialization(tmp_path, monkeypatch):
    config = _launch_config(tmp_path)
    monkeypatch.setattr(L, "_run", lambda *a, **k: " M tools/i6dmo/model.py\n")
    monkeypatch.setattr(CP, "build_common_population", lambda *a, **k: pytest.fail("dirty source reached data"))
    with pytest.raises(ValueError, match="SOURCE_NOT_FROZEN"):
        L.prepare(config)
    assert not Path(config["work_root"]).exists()


def _finish_fixture_child(root, unit, child, source_out=None):
    folder = root / "attempts" / unit["unit_id"] / "a01"
    out = folder / "out"
    if source_out is not None:
        shutil.copytree(source_out, out)
    else:
        out.mkdir(parents=True)
    child = copy.deepcopy(child)
    child.update(schema="i6dmo.terminal.v1", unit_id=unit["unit_id"], identity_sha256=unit["identity_sha256"])
    if unit["kind"] == "report":
        metrics_path = out / child["artifacts"]["metrics"]["path"]
        metrics = json.loads(metrics_path.read_text())
        metrics["matched"]["capacity"] = _long_capacity(unit["arm"])
        metrics_path.write_text(json.dumps(metrics))
        child["artifacts"]["metrics"]["sha256"] = MR._sha(metrics_path)
    (out / "TERMINAL.json").write_text(json.dumps(child))
    q = C.open_queue(root)
    try:
        assert q.unit(unit["unit_id"])["state"] == "QUEUED"
        claim = q.claim(unit["unit_id"], owner="review-fixture", role="coordinator",
                        host_key="synthetic-host", boot_id="synthetic-boot",
                        device_class="cpu", heavy=False, ttl=60, cap_bytes=2 << 30)
        assert claim["result"] == "CLAIMED"
        info = dict(claim["attempt"], role="coordinator", cap_bytes=2 << 30)
        finished = W._finish(root, q, unit, claim["token"], 0, folder, info)
        assert finished["result"] == "COMPLETE", finished
        rejected = [event for event in q.events(1000) if event["event"] == "REPORT_REJECTED"]
        assert finished["reports_rejected"] == 0, rejected
    finally:
        q.close()
    W._verify_stored(root / "terminals" / unit["identity_sha256"], unit)


def _run_fixture_hook(root, plan, effective, unit, role):
    folder = root / "hook-results" / unit["unit_id"]
    folder.mkdir(parents=True)
    q = C.open_queue(root)
    try:
        spec = W.build_unit_spec(root, q, plan, effective, unit, role,
                                 device_class="cpu", gpu=None, cap=2 << 30, bundle=None,
                                 attempt=dict(index=1, kind="INITIAL"))
    finally:
        q.close()
    assert unit_runner.run_hook(unit["kind"], spec, folder, folder / "progress.json") == 0
    child = json.loads((folder / "TERMINAL.json").read_text())
    _finish_fixture_child(root, unit, child, folder)
    return child


def test_raw_pair_decision_report_is_accepted_by_the_real_olap_normalizer(pair_env):
    from tools.i6dmo.outbox import seal_report

    child = RP.run_unit(*pair_env)
    assert (child["status"], child["disposition"]) == ("COMPLETE", "PAIR_NOT_ELIGIBLE"), child
    assert child["reports"]
    for report in child["reports"]:
        assert seal_report(report)["report_sha256"]


@pytest.mark.parametrize("scenario", ["positive", "negative", "tie"])
def test_f3_finite_real_factory_hooks_outbox_mirrors_and_close(tmp_path, scenario, monkeypatch):
    """Fit terminals are synthetic; factory, comparison/pair hooks and CLOSE are real."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    env = pair_fixtures.env.__wrapped__(evidence)
    if scenario != "negative":
        pair_fixtures._execution_fixture(env, evidence)
    if scenario == "tie":
        import numpy as np
        reference = Path(env[0]["upstream"]["mg12m.report"]["path"]).parent
        with np.load(reference / "out" / "validation_predictions.npz", allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        env[0]["upstream"]["mg1l.report"] = pair_fixtures.report(evidence / "tie", "mg1l", {18, 24}, arrays)
    report_pin = env[0]["upstream"]["mg12m.report"]
    metrics, _ = MR._authenticated_report("mg12m.report", report_pin, None)
    spec_path, data, storage = ops_fixtures.host(tmp_path / "host")
    pins = ops_fixtures.pins(data, storage)
    pins["common_population"]["population_sha256"] = metrics["matched"]["common_population_sha256"]
    docs = Path(__file__).resolve().parents[2] / "docs/i6n/direct_multi_output_static/matched_long"
    designs = {arm: docs / f"EXPERIMENT.{arm}.json" for arm in MR.LONG_ARMS}
    root = tmp_path / "campaign"
    C.init_matched_contrast(designs, spec_path, root, code_commit="c" * 40, data_pins=pins,
                            raw_pair=env[0]["effective_config"]["raw_pair"])
    plan, effective = C.load_plan(root), C.load_effective(root)
    units = {u["unit_id"]: u for u in plan["units"]}
    assert len(units) == 6 and plan["mode"] == C.MODE_MATCHED_LONG
    assert not plan["test_units_enumerated"] and all(not u["test_open"] for u in units.values())
    assert all(u["stage"] in plan["stages"] for u in units.values())
    controls = []

    def systemctl(*args):
        controls.append(args)
        return dict(rc=0, stdout="inactive")

    deploy.deploy(root, unit_dir=tmp_path / "units", systemctl=systemctl, start=False)
    for arm in sorted(MR.LONG_ARMS):
        _finish_fixture_child(root, units[arm + ".fit"], dict(
            status="COMPLETE", disposition="SYNTHETIC_FIT_TERMINAL_NO_TRAINING_EXECUTED",
            artifacts={}, costs=dict(fit_seconds=0), reports=[]))
        C._tick(root)
        pin = env[0]["upstream"][arm + ".report"]
        stored = json.loads(Path(pin["path"]).read_text())
        _finish_fixture_child(root, units[arm + ".report"], stored["terminal"], Path(pin["path"]).parent / "out")
    C._tick(root)
    role = C.load_deployment(root)["roles"]["coordinator"]
    comparison = _run_fixture_hook(root, plan, effective, units["comparison"], role)
    assert comparison["disposition"] == ("MG12_RETAINED_WITHIN_TOLERANCE" if scenario == "tie" else "MG12_SELECTED")
    C._tick(root)
    if scenario == "negative":
        def forbidden(*args, **kwargs):
            pytest.fail("negative pair read predictions, bars, calendar or strategy")
        monkeypatch.setattr(RP, "_predictions", forbidden)
        for name in ("_bars", "load_pinned_policy", "run_backtest"):
            monkeypatch.setattr(RP.SA, name, forbidden)
        monkeypatch.setattr(RP.SA.c1, "load_calendar", forbidden)
    pair = _run_fixture_hook(root, plan, effective, units["raw.pair"], role)
    assert pair["disposition"] == ("PAIR_NOT_ELIGIBLE" if scenario == "negative" else "PASS")
    expected = "PAIR_NOT_ELIGIBLE" if scenario == "negative" else "RAW_PAIR_VALIDATION_COMPLETE"
    assert C.readiness(root)["disposition"] == expected
    warehouse = FakeWarehouse()
    published = C.publish(root, warehouse=warehouse, systemctl=systemctl, self_unit="review-fixture.service")
    assert published["close"] == "CLOSED", published
    close = C.closed_doc(root)
    assert close["disposition"] == expected and close["trigger"] == "AUTOMATIC"
    assert close["closure_inputs"]["OUTBOX_READBACK.json"]
    readback, problems = C._outbox_readback(root)
    assert not problems and readback["summary"]["delivered"] > 0
    spec = C.load_deployment(root)
    for mirror in (spec["mirrors"]["local"], spec["mirrors"]["second_host_path"]):
        assert MR._sha(Path(mirror) / "CLOSE.json") == MR._sha(root / "CLOSE.json")
    assert C._finalized(root)
    assert any(call[0] == "disable" for call in controls)
    assert all(call[-1].startswith(plan["campaign_id"]) for call in controls if call[0] in ("stop", "disable"))
    first_close = (root / "CLOSE.json").read_bytes()
    assert C.publish(root, warehouse=warehouse, systemctl=systemctl)["close"] == "ALREADY_CLOSED"
    assert (root / "CLOSE.json").read_bytes() == first_close
