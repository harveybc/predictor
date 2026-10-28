"""Finite short-to-long continuation; no model, services or network."""
import json
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.i6dmo import r2_downsampled_campaign as S
from tools.i6dmo import deploy, r2_recovery, status as campaign_status_module
from tools.i6dmo.queue import Queue

REAL_INIT = S.C.init_r2_contrast
REAL_DEPLOY = deploy.deploy
REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("disposition", ["R2_SELECTED", "R0_RETAINED"])
def test_scientific_short_closure_activates_long(disposition):
    assert S.next_action(dict(state="CLOSED", disposition=disposition, failed=0), None) == "DEPLOY_LONG"


@pytest.mark.parametrize("short", [
    dict(state="CLOSED", disposition="INCOMPLETE_COMPARISON", failed=1),
    dict(state="FAILED", failed=1), dict(state="TECHNICAL_TERMINAL", failed=0)])
def test_short_technical_failure_never_activates_long(short):
    assert S.next_action(short, None) == "TECHNICAL_TERMINAL"


def test_running_short_does_not_duplicate_deployment():
    assert S.next_action(dict(state="RUNNING", failed=0), None) == "WAIT_SHORT"
    assert S.next_action(None, None) == "DEPLOY_SHORT"


def test_long_close_finishes_outer_sequence():
    short = dict(state="CLOSED", disposition="R0_RETAINED", failed=0)
    assert S.next_action(short, dict(state="CLOSED", disposition="R2_SELECTED", failed=0)) == "CLOSED"
    assert S.next_action(short, dict(state="RUNNING", failed=0)) == "WAIT_LONG"
    assert S.next_action(short, dict(state="FAILED", failed=1)) == "TECHNICAL_TERMINAL"


def test_missing_spec_or_changed_inputs_cannot_silently_create_new_campaign(tmp_path):
    with pytest.raises(ValueError, match="SEQUENCE_CONFIG"):
        S.validate_config(dict(schema="i6dmo.r2_downsampled_sequence.v1"))


def test_status_preserves_failed_short_and_does_not_infer_a_long_measurement(tmp_path, monkeypatch):
    root = tmp_path / "sequence"
    (root / "short/campaign").mkdir(parents=True)
    (root / "short/campaign/PLAN.json").write_text("{}")
    monkeypatch.setattr(S, "campaign_status", lambda p: dict(state="FAILED", failed=1, eta=dict(kind="UNMEASURED")))
    result = S.status(root)
    assert result["action"] == "TECHNICAL_TERMINAL"
    assert result["families"]["long"] is None
    assert result["families"]["short"]["state"] == "FAILED"


def test_closed_result_does_not_claim_future_quarterly_campaign(tmp_path, monkeypatch):
    root = tmp_path / "sequence"
    for family in ("short", "long"):
        folder = root / family / "campaign"
        folder.mkdir(parents=True)
        (folder / "PLAN.json").write_text("{}")
    monkeypatch.setattr(S, "campaign_status", lambda p: dict(state="CLOSED", failed=0, disposition="R0_RETAINED"))
    result = S.status(root)
    assert result["action"] == "CLOSED"
    assert result["test_open"] is False
    assert set(result["families"]) == {"short", "long"}


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    S.atomic_json(path, doc)
    return path


@pytest.fixture
def sequence(tmp_path, monkeypatch):
    """Real sequence/SQLite/files; controlled preparation, factory, runtime and deploy adapters.

    RD09-RD12: no worker, network, model, production path or actual systemctl call.
    """
    docs = tmp_path / "docs"
    monkeypatch.setattr(S, "DOCS", docs)
    monkeypatch.setattr(S, "REPO", tmp_path)
    release_path = write(docs / "r2_downsampled/RELEASE_ACCEPTANCE.json",
                         dict(code_files={"fixture.py": "fixture"}, verdict="ACCEPT_FOR_EXECUTION"))
    monkeypatch.setattr(S, "RELEASE", release_path)
    monkeypatch.setattr(S, "release", lambda: json.loads(release_path.read_text()))
    env = SimpleNamespace(head="c" * 40, dirty=False, calls=[], crash=False, deploy_crash=False)
    monkeypatch.setattr(S.C, "_git_head", lambda: env.head)
    monkeypatch.setattr(S.L, "_run", lambda *a, **kw: "dirty" if env.dirty else "")
    spec = dict(roles={"coordinator": dict(workdir=str(tmp_path / "runtime"),
                experiment_storage_root=str(tmp_path / "experiment_storage"))})
    source_spec = write(tmp_path / "source_deployment.json", spec)
    source = tmp_path / "source"
    long_work = tmp_path / "raw_long"
    for root in (source, long_work / "campaign"):
        effective = S.C._seal(dict(code_commit="o" * 40, test_open=False,
            design=dict(splits=dict(test=dict(open=False))), data_pins={}))
        write(root / "EFFECTIVE_CONFIG.json", effective)
        write(root / "PLAN.json", dict(campaign_id="original",
            effective_config_sha256=S.C.sha256_json(effective)))
    write(long_work / "campaign/CLOSE.json", dict(state="CLOSED", disposition="RAW_PAIR_VALIDATION_COMPLETE"))
    config = dict(schema="i6dmo.r2_downsampled_sequence.v1", work_root=str(tmp_path / "sequence"),
        source_deployment=str(source_spec), short_source_campaign=str(source),
        short_source_pins=dict(plan_sha256=S.file_sha256(source / "PLAN.json"),
            effective_sha256=S.file_sha256(source / "EFFECTIVE_CONFIG.json"), code_commit="o" * 40),
        long_source_work_root=str(long_work), input_dir=str(tmp_path / "input"))
    write(docs / "r2_downsampled/SOURCE_PINS.json", dict(pins=config["short_source_pins"]))
    for family, arm in (("short", "mg2m"), ("long", "mg12m")):
        folder = docs / ("matched_short" if family == "short" else "r2_downsampled/long_source")
        design = dict(seal_sha256=family, splits=dict(test=dict(open=False)))
        write(folder / f"EXPERIMENT.{arm}.json", design)
        contract_path = docs / ("matched_short/R2_CONTRAST_DESIGN.ruled.json" if family == "short"
                               else "r2_downsampled/R2_CONTRAST_DESIGN.long.ruled.json")
        write(contract_path, dict(seal_sha256=family, shared=dict(arm=arm),
                                 pretraining=dict(objective={"monitor": family})))
        write((docs / "matched_short" if family == "short" else long_work) / "POPULATION_SEAL.json", {})
    def prepare(config, family, code, files):
        env.calls.append(("prepare", family))
        work = Path(config["work_root"]) / family
        deployment = write(work / "deployment.json", spec)
        prepared = write(work / "prepared.json", dict(data_pins={"population": {"sha256": family}}))
        recovery = None
        if family == "short":
            manifest = dict(manifest_sha256="m" * 64, source=dict(campaign_id="original"),
                pretraining_refs={}, unit_specs={}, donor_provenance={"original": True},
                r0_reuse=dict(decision="FRESH_R0_REQUIRED", failed=["changed_code"]))
            path = write(work / "RECOVERY.json", manifest)
            proof = write(Path(config["work_root"]) / "recovery/REMOTE_CUSTODY.json",
                          dict(schema="i6dmo.remote_custody_view.v1", fixture=True))
            recovery = dict(path=str(path), sha256=S.file_sha256(path), r0_target=dict(code=code),
                            physical_custody=dict(path=str(proof), sha256=S.file_sha256(proof)))
        return dict(state="PREPARED", deployment=str(deployment), prepared=str(prepared), recovery=recovery,
            contract=str(docs / ("matched_short/R2_CONTRAST_DESIGN.ruled.json" if family == "short"
                else "r2_downsampled/R2_CONTRAST_DESIGN.long.ruled.json")),
            design=str(docs / ("matched_short/EXPERIMENT.mg2m.json" if family == "short"
                else "r2_downsampled/long_source/EXPERIMENT.mg12m.json")))
    monkeypatch.setattr(S, "_prepare_family", prepare)
    monkeypatch.setattr(S.MR, "validate", lambda path: dict(state="VALID"))
    def load_manifest(path, *, expected_sha256, r0_target):
        assert S.file_sha256(path) == expected_sha256
        assert r0_target["code"] == env.head
        return json.loads(Path(path).read_text())
    monkeypatch.setattr(r2_recovery, "load_manifest", load_manifest)
    def init(contract, design, deployment, root, **kw):
        root = Path(root)
        env.calls.append(("init", root.name, copy.deepcopy(kw)))
        r2 = dict(contract_seal_sha256=json.loads(Path(contract).read_text())["seal_sha256"],
            contract_file_sha256=S.file_sha256(contract),
            objective=json.loads(Path(contract).read_text())["pretraining"]["objective"],
            r0_reuse=kw["r0_reuse"]["decision"], strict_reports=True)
        if kw["donor_recovery"]:
            recovery = kw["donor_recovery"]
            manifest = load_manifest(recovery["path"], expected_sha256=recovery["sha256"],
                                     r0_target=recovery["r0_target"])
            r2.update(recovered_unit_specs=manifest["unit_specs"],
                recovered_donor_provenance=manifest["donor_provenance"], donor_recovery=dict(
                    manifest_sha256=manifest["manifest_sha256"], file_sha256=recovery["sha256"],
                    physical_custody_sha256=recovery["physical_custody"]["sha256"],
                    source_campaign=manifest["source"]["campaign_id"], original_donors={}))
        effective = S.C._seal(dict(code_commit=kw["code_commit"], mode=S.C.MODE_R2, r2=r2,
            data_pins=kw["data_pins"], data_sha256=S.C.sha256_json(kw["data_pins"]),
            deployment_sha256=S.C.sha256_json(json.loads(Path(deployment).read_text())),
            class_caps=kw["class_caps"], entrypoints=S.C.R2_ENTRYPOINTS,
            design=json.loads(Path(design).read_text()), test_open=False, prior_roots=[],
            fit_device="gpu", engineering_cpu_fixture=False))
        write(root / "EFFECTIVE_CONFIG.json", effective)
        write(root / "private/deployment.json", json.loads(Path(deployment).read_text()))
        units = [dict(unit_id="r0.fit", identity_sha256="u" * 64, kind="fit", device="gpu", test_open=False)]
        q = S.C.open_queue(root)
        try:
            for unit in units:
                q.ensure_unit(unit["unit_id"], unit, state="QUEUED")
        finally:
            q.close()
        write(root / "PLAN.json", dict(campaign_id="fixture", mode=S.C.MODE_R2, units=units,
            effective_config_sha256=S.C.sha256_json(effective),
            effective_config_file_sha256=S.file_sha256(root / "EFFECTIVE_CONFIG.json"), test_units_enumerated=False))
        if env.crash:
            write(root / "ADOPTION_FACT.json", {"retained": True})
            raise RuntimeError("after PLAN and adoption fact")
    monkeypatch.setattr(S.C, "init_r2_contrast", init)
    def runtime(*args, **kw):
        env.calls.append(("runtime",))
    monkeypatch.setattr(S.L, "_verify_runtime", runtime)
    def deploy_adapter(root):
        env.calls.append(("deploy", Path(root).name))
        if env.deploy_crash:
            raise RuntimeError("deploy interrupted")
        write(Path(root) / "DEPLOY.json", {"state": "DEPLOYED"})
    monkeypatch.setattr(deploy, "deploy", deploy_adapter)
    def compute(root, level, now=None):
        assert level in ("campaign", "all")
        q = Queue.open_readonly(Path(root) / "state/queue.sqlite3")
        try:
            rows = q.units()
        finally:
            q.close()
        close = S.C.closed_doc(root)
        failed = sum(r["state"] == "FAILED" for r in rows)
        return dict(state="CLOSED" if close else "FAILED" if failed else "RUNNING",
                    counts=dict(FAILED=failed), close=close, eta=dict(kind="UNMEASURED"))
    monkeypatch.setattr(campaign_status_module, "compute", compute)
    monkeypatch.setattr(S.C, "closure_problems", lambda root: ([], {}, {}))
    env.config, env.root, env.prepare = config, Path(config["work_root"]), prepare
    return env


def finish(env, family, disposition="R0_RETAINED"):
    root = env.root / family / "campaign"
    q = S.C.open_queue(root)
    try:
        got = q.claim("r0.fit", owner="fixture", role="coordinator", host_key="fixture",
                      boot_id="fixture", device_class="cpu", heavy=False, ttl=60)
        q.complete("r0.fit", got["token"], dict(status="COMPLETE"), "f" * 64)
    finally:
        q.close()
    write(root / "CLOSE.json", dict(state="CLOSED", disposition=disposition))


def test_real_sequence_pipeline_resumes_without_reinit_and_negative_short_advances(sequence):
    env = sequence
    assert S.run(env.config)["action"] == "WAIT_SHORT"
    assert [c[1] for c in env.calls if c[0] == "init"] == ["campaign.building"]
    assert (env.root / "short/campaign/INITIALIZED.json").is_file()
    assert not (env.root / "short/campaign.building").exists()
    setup = json.loads((env.root / "SETUP.json").read_text())
    assert setup["release_sha256"] == S.file_sha256(S.RELEASE)
    before = copy.deepcopy(env.calls)
    S.run(env.config)
    assert env.calls == before
    finish(env, "short")
    assert S.run(env.config)["action"] == "WAIT_LONG"
    finish(env, "long", "R2_SELECTED")
    assert S.run(env.config)["action"] == "CLOSED"
    assert len([c for c in env.calls if c[0] == "init"]) == 2


def test_crash_after_plan_preserves_staging_and_resume_quarantines_without_blind_promotion(sequence):
    env = sequence
    env.crash = True
    with pytest.raises(RuntimeError, match="after PLAN"):
        S.run(env.config)
    assert not (env.root / "short/campaign").exists()
    assert (env.root / "short/campaign.building/ADOPTION_FACT.json").exists()
    assert not any(c[0] == "deploy" for c in env.calls)
    assert S.status(env.root)["action"] == "ERROR"
    env.crash = False
    S.run(env.config)
    retained = list((env.root / "short").glob("campaign.incomplete-*"))
    assert len(retained) == 1 and (retained[0] / "ADOPTION_FACT.json").exists()
    assert len([c for c in env.calls if c[0] == "init"]) == 2


def test_deploy_crash_resumes_verified_final_campaign_without_repeating_init(sequence):
    env = sequence
    env.deploy_crash = True
    with pytest.raises(RuntimeError, match="deploy interrupted"):
        S.run(env.config)
    assert (env.root / "short/campaign/INITIALIZED.json").exists()
    env.deploy_crash = False
    assert S.run(env.config)["action"] == "WAIT_SHORT"
    assert len([c for c in env.calls if c[0] == "init"]) == 1
    assert len([c for c in env.calls if c[0] == "deploy"]) == 2


@pytest.mark.parametrize("changed", ["release", "code", "dirty", "contract", "source", "deployment"])
def test_resume_refuses_changed_setup_identities_before_deploy(sequence, changed):
    env = sequence
    S.run(env.config)
    if changed == "code":
        env.head = "d" * 40
    elif changed == "dirty":
        env.dirty = True
    else:
        path = dict(release=S.RELEASE, contract=S.DOCS / "matched_short/R2_CONTRAST_DESIGN.ruled.json",
            source=Path(env.config["short_source_campaign"]) / "PLAN.json",
            deployment=Path(env.config["source_deployment"]))[changed]
        doc = json.loads(path.read_text())
        doc["changed"] = True
        write(path, doc)
    before = copy.deepcopy(env.calls)
    with pytest.raises(ValueError):
        S.run(env.config)
    assert env.calls == before
    assert S.status(env.root)["action"] == "ERROR"


@pytest.mark.parametrize("changed", ["code_commit", "data_pins", "deployment_sha256", "r2", "recovery", "plan", "physical_custody"])
def test_resume_rejects_changed_effective_or_recovery_even_with_resealed_effective(sequence, changed):
    env = sequence
    S.run(env.config)
    root = env.root / "short/campaign"
    (root / "DEPLOY.json").unlink()
    if changed == "recovery":
        write(env.root / "short/RECOVERY.json", {"forged": True})
    elif changed == "physical_custody":
        write(env.root / "recovery/REMOTE_CUSTODY.json", {"forged": True})
    elif changed == "plan":
        plan = S.C.load_plan(root)
        plan["effective_config_sha256"] = "wrong"
        write(root / "PLAN.json", plan)
    else:
        doc = S.C.load_effective(root)
        doc[changed] = {} if changed in ("data_pins", "r2") else "wrong"
        write(root / "EFFECTIVE_CONFIG.json", S.C._seal(doc))
    before = copy.deepcopy(env.calls)
    with pytest.raises((ValueError, S.C.Refused)):
        S.run(env.config)
    assert env.calls == before


def test_actual_failed_queue_stops_long_and_status_error_is_not_a_scientific_negative(sequence, monkeypatch):
    env = sequence
    S.run(env.config)
    q = S.C.open_queue(env.root / "short/campaign")
    try:
        q.set_state("r0.fit", "FAILED", cause="TECHNICAL_FIXTURE")
    finally:
        q.close()
    assert S.run(env.config)["action"] == "TECHNICAL_TERMINAL"
    assert not (env.root / "long/campaign").exists()
    def broken(*a, **kw):
        raise OSError("status fixture unreadable")
    monkeypatch.setattr(campaign_status_module, "compute", broken)
    result = S.status(env.root)
    assert result["action"] == "ERROR"
    assert result["families"]["short"]["state"] == "ERROR"
    assert "status fixture unreadable" in result["families"]["short"]["error"]


def test_real_factory_initializes_finite_long_graph_before_atomic_publication(sequence, monkeypatch):
    from tests.i6dmo.ops_fakes import private_spec
    from tools.i6dmo import design as D, r2_design as RD
    env = sequence
    original = REPO / "docs/i6n/direct_multi_output_static"
    design = D.load_design(original / "r2_downsampled/long_source/EXPERIMENT.mg12m.json")
    effective = D.effective_config(design)
    contract = json.loads((original / "matched_short/R2_CONTRAST_DESIGN.ruled.json").read_text())
    contract["shared"].update(arm="mg12m", experiment_seal_sha256=design["seal_sha256"],
        experiment_design_sha256=effective["design_sha256"], horizons_hours=design["horizons_hours"],
        population_sha256="a" * 64,
        fit_pinning=dict(runtime=design["sealed_identities"]["runtime"]))
    write(S.DOCS / "r2_downsampled/long_source/EXPERIMENT.mg12m.json", design)
    write(S.DOCS / "r2_downsampled/R2_CONTRAST_DESIGN.long.ruled.json", contract)
    # Contract validation is a controlled boundary here; the real factory/queue are exercised.
    monkeypatch.setattr(RD, "validate_r2_design", lambda *a, **kw: None)
    monkeypatch.setattr(S.C, "_storage_gate", lambda *a, **kw: None)
    monkeypatch.setattr(S.C, "init_r2_contrast", REAL_INIT)
    def prepare(*args):
        result = env.prepare(*args)
        write(Path(result["deployment"]), private_spec(env.root))
        write(Path(result["prepared"]), dict(data_pins=dict(
            common_population=dict(base="experiment_storage", relpath="population", sha256="b" * 64,
                                   population_sha256="a" * 64),
            matched_input=dict(relpath="input", sha256="c" * 64))))
        return result
    monkeypatch.setattr(S, "_prepare_family", prepare)
    S._deploy_family(env.config, "long", env.head, {})
    root = env.root / "long/campaign"
    assert (root / "INITIALIZED.json").exists()
    assert len(S.C.load_plan(root)["units"]) == 25
    q = Queue.open_readonly(root / "state/queue.sqlite3")
    try:
        assert len(q.units()) == 25 and q.attempts() == []
    finally:
        q.close()


def test_real_deploy_adapter_uses_final_path_and_runtime_checks_without_services(sequence, monkeypatch):
    env = sequence
    recorded = []
    monkeypatch.setattr(S.C, "_storage_gate", lambda *a, **kw: None)
    monkeypatch.setattr(S.C, "ensure_execution_policy", lambda *a, **kw: None)
    monkeypatch.setattr(deploy, "spec_has_envfile", lambda root: False)
    monkeypatch.setattr(deploy, "render_units", lambda *a, **kw: {"i6dmo-fixture-tick.timer": "[Timer]\n"})
    def systemctl(*args):
        recorded.append(args)
        return dict(rc=0, stdout="active")
    def adapter(root):
        assert Path(root).name == "campaign"
        assert (Path(root) / "INITIALIZED.json").exists()
        assert env.calls[-1][0] == "runtime"
        return REAL_DEPLOY(root, unit_dir=env.root / "fixture_units", systemctl=systemctl)
    monkeypatch.setattr(deploy, "deploy", adapter)
    assert S.run(env.config)["action"] == "WAIT_SHORT"
    assert (env.root / "short/campaign/DEPLOY.json").exists()
    assert ("enable", "--now", "i6dmo-fixture-tick.timer") in recorded
    assert not any(c[0] == "init" and c[1] != "campaign.building" for c in env.calls)


@pytest.mark.parametrize("previous", [None, "/fixture/unrelated-storage"])
@pytest.mark.parametrize("crash", [False, True])
def test_run_scopes_coordinator_storage_before_factory_and_restores_environment(sequence, monkeypatch, previous, crash):
    env = sequence
    if previous is None:
        monkeypatch.delenv("EXPERIMENT_STORAGE_ROOT", raising=False)
    else:
        monkeypatch.setenv("EXPERIMENT_STORAGE_ROOT", previous)
    spec = json.loads(Path(env.config["source_deployment"]).read_text())
    expected = spec["roles"]["coordinator"]["experiment_storage_root"]
    init = S.C.init_r2_contrast
    observed = []
    def checked(*args, **kwargs):
        observed.append(os.environ.get("EXPERIMENT_STORAGE_ROOT"))
        assert observed[-1] == expected
        return init(*args, **kwargs)
    monkeypatch.setattr(S.C, "init_r2_contrast", checked)
    env.crash = crash
    if crash:
        with pytest.raises(RuntimeError, match="after PLAN"):
            S.run(env.config)
    else:
        assert S.run(env.config)["action"] == "WAIT_SHORT"
    assert observed == [expected]
    assert os.environ.get("EXPERIMENT_STORAGE_ROOT") == previous


@pytest.mark.parametrize("value", [None, "relative/storage"])
def test_run_refuses_missing_or_relative_coordinator_storage_before_factory(sequence, value):
    env = sequence
    path = Path(env.config["source_deployment"])
    spec = json.loads(path.read_text())
    spec["roles"]["coordinator"]["experiment_storage_root"] = value
    write(path, spec)
    with pytest.raises(ValueError, match="SEQUENCE_STORAGE_ROOT_REQUIRED"):
        S.run(env.config)
    assert not any(c[0] == "init" for c in env.calls)


@pytest.mark.parametrize("directory", ["experiment_storage", "storage with spaces %f"])
def test_install_service_explicitly_binds_storage_without_real_systemctl(sequence, monkeypatch, directory):
    env = sequence
    source = Path(env.config["source_deployment"])
    spec = json.loads(source.read_text())
    storage = str(env.root.parent / directory)
    spec["roles"]["coordinator"]["experiment_storage_root"] = storage
    spec.update(launcher=dict(crispdm_run="/fixture/crispdm-run"),
                coordinator=dict(python="/fixture/python"))
    write(source, spec)
    path = write(env.root.parent / "sequence_config.json", env.config)
    home = env.root.parent / "fixture_home"
    monkeypatch.setattr(S.Path, "home", lambda: home)
    calls = []
    def systemctl(*args):
        calls.append(args)
        return dict(rc=0, stdout="")
    monkeypatch.setattr(deploy, "default_systemctl", systemctl)
    result = S.install(path)
    assert result["state"] == "ARMED"
    service = home / ".config/systemd/user" / result["unit"].replace(".timer", ".service")
    expected = json.dumps("EXPERIMENT_STORAGE_ROOT=" + storage).replace("%", "%%")
    assert "Environment=" + expected in service.read_text().splitlines()
    assert calls == [("daemon-reload",), ("enable", "--now", result["unit"])]
    assert not any(c[0] in ("init", "deploy") for c in env.calls)


@pytest.mark.parametrize("value", [None, "relative/storage"])
def test_install_refuses_missing_or_relative_storage_before_writing_units(sequence, monkeypatch, value):
    env = sequence
    source = Path(env.config["source_deployment"])
    spec = json.loads(source.read_text())
    spec["roles"]["coordinator"]["experiment_storage_root"] = value
    write(source, spec)
    config = write(env.root.parent / "sequence_config.json", env.config)
    home = env.root.parent / "fixture_home"
    monkeypatch.setattr(S.Path, "home", lambda: home)
    monkeypatch.setattr(deploy, "default_systemctl", lambda *a: pytest.fail("unexpected systemctl"))
    with pytest.raises(ValueError, match="SEQUENCE_STORAGE_ROOT_REQUIRED"):
        S.install(config)
    assert not home.exists()
