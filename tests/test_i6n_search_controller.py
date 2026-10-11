"""Controller contracts; generated fixtures only, never launch historical fits."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tests.test_i6n_historical_evaluator import prepared
from tests.test_i6n_historical_cell import fixture as historical_fixture, write_json
from tools import i6n_search_controller as ctl


def pin(path):
    return dict(path=str(path), sha256=ctl.cell.file_hash(path))


@pytest.fixture
def config(prepared, tmp_path):
    base, _, binding, request = prepared
    pilot = tmp_path / "pilot.json"
    write_json(pilot, dict(state="COMPLETED", identity=dict(config=base)))
    plan = tmp_path / "plan.md"
    plan.write_text("Bounded endpoint engineering allocation, not a campaign.\n")
    cfg = dict(schema=ctl.SCHEMA, scope="ENDPOINT_ONLY_ENGINEERING_H2",
        binding=pin(Path(binding["i6n_evaluation_context"]["output_root"]) / "BINDING.json"),
        plan=dict(**pin(plan), section="Initial endpoint allocation"),
        external_pilots=[dict(receipt=pin(pilot), genome={k: request["config"][k] for k in ctl.search._GENOME_KEYS})],
        output_root=str(tmp_path / "controller"), seed=base["training"]["seed"],
        budget=dict(max_candidates=4, population_size=2, generations=2, patience=2, max_proposals=128),
        admission=dict(min_spacing_hours=2, max_samples=25, max_core_width=32),
        resources=dict(child_memory_bytes=2*1024**3, child_wall_seconds=240, total_wall_seconds=900,
            min_mem_available_bytes=3*1024**3, poll_seconds=1,
            crispdm_run="/home/harveybc/.local/bin/crispdm-run",
            admission_root=str(tmp_path / "admission")),
        device=dict(cuda_visible_devices="", required_gpu_name=None))
    path = tmp_path / "private.json"
    write_json(path, cfg)
    return path, cfg


def fake_launch(cfg, opt, wire, directory, heartbeat):
    receipt = dict(deepcopy(opt["i6n_support"]), status="ok",
                   error_sums=deepcopy(opt["i6n_support"]["naive_sums"]),
                   metadata=dict(search_space_scope="ENDPOINT_ONLY"))
    result = dict(fitness=1., metrics=receipt, hyper_dict=ctl.search.effective_config(wire["genome"]))
    heartbeat(dict(state="RUNNING", progress=dict(event="epoch", epoch=1), eta_seconds=2))
    write_json(directory / "WORKER.json", dict(state="COMPLETED", result=result,
               cuda_visible_devices=cfg["device"]["cuda_visible_devices"]))
    return dict(returncode=0, resource=dict(observed_peak_bytes=1234, reclaim_cause="LEASE_RELEASED"))


@pytest.fixture
def simulated(monkeypatch):
    calls = []
    def launch(*args):
        calls.append(args[2]["identity"])
        return fake_launch(*args)
    monkeypatch.setattr(ctl, "launch", launch)
    monkeypatch.setattr(ctl, "host_gate", lambda cfg: None)
    monkeypatch.setattr(ctl, "verify_bridge_result", lambda *a: None)
    return calls


def test_private_config_pins_and_budget(config):
    path, cfg = config
    loaded = ctl.load_config(path)
    assert len(loaded[2]) == 1
    for change in (dict(max_candidates=25), dict(population_size=9), dict(max_proposals=True)):
        bad = deepcopy(cfg)
        bad["budget"].update(change)
        write_json(path, bad)
        with pytest.raises(ValueError):
            ctl.load_config(path)
    write_json(path, cfg)
    Path(cfg["plan"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        ctl.load_config(path)


def test_filters_before_fit_and_pilot_alias_exclusion(config):
    path, cfg = config
    _, opt, excluded, _ = ctl.load_config(path)
    plugin = ctl.EndpointPlugin(cfg["admission"], excluded)
    pilot = cfg["external_pilots"][0]["genome"]
    alias = deepcopy(pilot)
    alias["multiplier"] = 6
    assert plugin.reason(alias) == "EXTERNAL_PILOT_EXCLUDED"
    trajectory = deepcopy(pilot)
    trajectory.update(output_mode="trajectory", delta_y=1)
    assert plugin.reason(trajectory) == "TRAJECTORY_UNSUPPORTED"
    hourly = deepcopy(pilot)
    hourly["delta_x"] = 1
    assert plugin.reason(hourly) == "OUTSIDE_COST_CLASS"
    large = deepcopy(pilot)
    large["core"]["width"] = 64
    assert plugin.reason(large) == "OUTSIDE_COST_CLASS"
    state = plugin.create_shared_population(2, opt, seed=cfg["seed"])
    assert all(plugin.reason(w["genome"]) is None for w in state["population"])
    assert len({w["identity"] for w in state["population"]}) == 2


def test_full_bounded_adapter_loop_and_terminal_resume(config, simulated):
    path, cfg = config
    status = ctl.run(path)
    assert status["state"] == "COMPLETED" and status["scope"] == cfg["scope"]
    assert status["completed"] == 4 and len(simulated) == len(set(simulated)) == 4
    state = ctl.read_state(Path(cfg["output_root"]))
    assert state["shared"]["generation"] == 2
    assert state["best"]["fitness"] == 1.
    assert all(r["resource"]["observed_peak_bytes"] == 1234 for r in state["records"].values())
    assert (Path(cfg["output_root"]) / "TERMINAL.json").exists()
    assert ctl.run(path) == status and len(simulated) == 4


def test_crash_reservation_never_relaunched(config, simulated, monkeypatch):
    path, cfg = config
    def crash(*args):
        raise KeyboardInterrupt("interrupted")
    monkeypatch.setattr(ctl, "launch", crash)
    with pytest.raises(KeyboardInterrupt):
        ctl.run(path)
    state = ctl.read_state(Path(cfg["output_root"]))
    assert any(r["state"] == "RUNNING" for r in state["records"].values())
    assert ctl.run(path)["state"] == "INTERRUPTED"
    assert not simulated


def test_failure_consumes_budget_never_fitness(config, simulated, monkeypatch):
    path, cfg = config
    def failure(cfg, opt, wire, directory, heartbeat):
        simulated.append(wire["identity"])
        write_json(directory / "WORKER.json", dict(state="FAILED", error="typed test failure"))
        return dict(returncode=1, resource=dict(observed_peak_bytes=1234, reclaim_cause="LEASE_RELEASED"))
    monkeypatch.setattr(ctl, "launch", failure)
    status = ctl.run(path)
    assert status["stop_reason"] == "no_eligible_parents"
    assert status["failed"] == 2 and status["best"] is None
    ctl.run(path)
    assert len(simulated) == 2


def test_wall_budget_and_memory_gate_do_not_launch(config, simulated, monkeypatch):
    path, cfg = config
    cfg["resources"]["total_wall_seconds"] = 1
    write_json(path, cfg)
    assert ctl.run(path)["stop_reason"] == "wall_budget"
    assert not simulated


def test_memory_refusal_terminal(config, simulated, monkeypatch):
    path, cfg = config
    def refuse(*args):
        raise ValueError("MemAvailable below gate")
    monkeypatch.setattr(ctl, "host_gate", refuse)
    assert ctl.run(path)["state"] == "REFUSED"
    assert not simulated
    assert ctl.run(path)["state"] == "REFUSED"


def test_missing_resource_peak_stops_continuation(config, simulated, monkeypatch):
    path, cfg = config
    def unknown(*args):
        simulated.append(args[2]["identity"])
        result = fake_launch(*args)
        result["resource"] = None
        return result
    monkeypatch.setattr(ctl, "launch", unknown)
    assert ctl.run(path)["stop_reason"] == "resource_evidence_missing"
    assert len(simulated) == 1


def test_changed_identity_and_tampered_state_refused(config, simulated):
    path, cfg = config
    ctl.run(path)
    root = Path(cfg["output_root"])
    wrapper = json.loads((root / "STATE.json").read_text())
    wrapper["payload"]["best"]["fitness"] = 0
    write_json(root / "STATE.json", wrapper)
    with pytest.raises(ValueError, match="digest"):
        ctl.run(path)


def test_launcher_command_is_capped_subprocess(config):
    path, cfg = config
    command, child, label = ctl.command(cfg, path, "a"*64)
    assert command[:5] == [cfg["resources"]["crispdm_run"], "-m", str(2*1024**3), "-t", "240s"]
    assert "tools.i6n_search_controller" in child and "worker" in child
    assert str(path) in child and command[-len(child):] == child
    assert label.startswith("i6n-")


def test_gpu_policy_no_fallback(config):
    _, cfg = config
    cfg["device"] = dict(cuda_visible_devices="GPU-test", required_gpu_name="NVIDIA GeForce RTX 5090")
    assert ctl.device_environment(cfg)["CUDA_VISIBLE_DEVICES"] == "GPU-test"
    cfg["device"]["cuda_visible_devices"] = ""
    with pytest.raises(ValueError):
        ctl.device_environment(cfg)


@pytest.mark.parametrize("horizon", [12, 72])
def test_horizon_parameterized_config_and_contexts(config, prepared, horizon):
    import numpy as np
    from tests.test_i6n_historical_cell import repin
    path, cfg = config
    base, base_path, _, _ = prepared
    with np.load(base["input"]["path"]) as old:
        ts = np.unique(np.concatenate((old["ts"], old["ts"]+horizon*3600)))
    np.savez(base["input"]["path"], ts=ts, x=np.arange(len(ts)*20).reshape(-1, 20),
             price=1.2+.001*np.arange(len(ts))**2)
    repin(base)
    base.update(horizon_hours=horizon, samples=13, spacing_hours=horizon//6)
    write_json(base_path, base)
    binding = ctl.bridge.freeze(base_path, path.parent / f"bridge-{horizon}")
    cfg.update(scope=f"ENDPOINT_ONLY_ENGINEERING_H{horizon}", external_pilots=[],
               binding=pin(Path(binding["i6n_evaluation_context"]["output_root"]) / "BINDING.json"))
    write_json(path, cfg)
    _, opt, excluded, _ = ctl.load_config(path)
    assert opt["i6n_horizon"] == horizon and not excluded
    plugin = ctl.EndpointPlugin(cfg["admission"], excluded)
    for multiplier in (2, 4, 6):
        g = ctl.search.default_genome(horizon)
        g.update(multiplier=multiplier, delta_x=horizon//2)
        assert plugin.reason(g) is None
    assert "i6n_evaluator_command" in opt and "i6n_evaluator" not in opt


def test_real_isolated_worker_adapter_bridge_cpu(config, monkeypatch):
    already_loaded = "tensorflow" in ctl.sys.modules
    path, cfg = config
    cfg["budget"].update(max_candidates=1, population_size=1, generations=1, patience=1, max_proposals=32)
    write_json(path, cfg)
    genome = deepcopy(cfg["external_pilots"][0]["genome"])
    genome["input_transform"] = "difference"
    monkeypatch.setattr(ctl.search, "sample_genome", lambda h, rng: deepcopy(genome))
    monkeypatch.setattr(ctl, "host_gate", lambda cfg: None)
    def isolated(cfg, opt, wire, directory, heartbeat):
        _, child, _ = ctl.command(cfg, path, wire["identity"])
        process = ctl.subprocess.run(child, env=ctl.device_environment(cfg), cwd=ctl.cell.BASE,
                                     capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, process.stderr
        # Launcher accounting is stubbed here; the worker, adapter, fit and C2 are real.
        return dict(returncode=0, resource=dict(observed_peak_bytes=1234, reclaim_cause="LEASE_RELEASED"))
    monkeypatch.setattr(ctl, "launch", isolated)
    status = ctl.run(path)
    assert status["completed"] == 1 and status["best"] is not None
    monkeypatch.setattr(ctl, "launch", lambda *a: pytest.fail("repeated worker"))
    assert ctl.run(path) == status
    assert ("tensorflow" in ctl.sys.modules) == already_loaded


def test_bad_resource_crash_boundary_cannot_continue(config, simulated, monkeypatch):
    path, cfg = config
    def no_peak(*args):
        simulated.append(args[2]["identity"])
        return dict(fake_launch(*args), resource=None)
    monkeypatch.setattr(ctl, "launch", no_peak)
    ctl.run(path)
    root = Path(cfg["output_root"])
    state = ctl.read_state(root)
    record = next(iter(state["records"].values()))
    assert record["state"] == "FAILED"
    # Model the old crash window: completion saved before its resource check.
    record.update(state="COMPLETED", fitness=1.)
    state["state"] = "RUNNING"
    ctl.save(root, state)
    assert ctl.run(path)["stop_reason"] == "resource_evidence_missing"
    assert len(simulated) == 1


def test_explicit_secondary_device_only(config):
    _, cfg = config
    cfg["device"] = dict(cuda_visible_devices="GPU-secondary", required_gpu_name="NVIDIA GeForce RTX 4090 Laptop GPU")
    assert ctl.device_environment(cfg)["CUDA_VISIBLE_DEVICES"] == "GPU-secondary"
    cfg["device"]["required_gpu_name"] = "NVIDIA GeForce RTX 5070 Ti"
    with pytest.raises(ValueError, match="admitted"):
        ctl.device_environment(cfg)
