"""RP33/RP38 acceptance: an E1 run takes its bytes through governance or it does not run.

The whole route is exercised against a DISPOSABLE data-gov serving a DISPOSABLE panel (the production
services are never touched): the campaign is registered before anything is prepared, the panel is
delivered over HTTP and verified, the run consumes exactly those bytes, a unit's terminal reaches the
accounting through the outbox and the campaign reconciles. The refusals are exercised too: no lake, no
delivery, altered bytes, and a range over the archive.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
GOV_APP = Path.home() / "Documents/GitHub/.worktrees/musashi-n3-data-gov-20260914T063541Z"
PYTHON = Path.home() / "anaconda3/envs/trading-stack/bin/python3.12"
KEY = Path.home() / ".local/state/crispdm-data-foundation/satoshi-store-hosts-20260914T1215Z/predictor.key"
RUNTIME = Path.home() / ".local/state/crispdm-data-foundation/musashi-store-adoption-20260914T181826Z/5055.runtime.json"

pytestmark = pytest.mark.skipif(not (GOV_APP / "app" / "main.py").is_file() or not KEY.is_file() or not RUNTIME.is_file(),
                                reason="the data-gov application, its runtime configuration or the service key is not present here")


def _load(name, where=REPO / "tools"):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load("df_e1_governed")
A = _load("df_public_lake_adopt")
P = _load("df_e1_pilot")
E = _load("df_mod_e0")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _panel(path: Path) -> str:
    rng = np.random.default_rng(3)
    n = 3 * P.DAY
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    base = np.sin(2 * np.pi * np.arange(n) / P.DAY) + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({c: base * (i + 1) + i for i, c in enumerate(cols)})
    ts = pd.date_range("2009-01-01 00:00", periods=n, freq="min")
    df.insert(0, "timestamp_label", ts.strftime("%d/%m/%Y %H:%M:%S"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return A.sha_file(path)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """A disposable data-gov serving a disposable copy of the panel, with the SAME bounded contract the
    adoption proposes (untimed, no availability block, every range refused by the holdout)."""
    work = tmp_path_factory.mktemp("govroute")
    resource = "uci_235_individual_household_power/panel.parquet"
    digest = _panel(work / "panels" / resource)
    cfg = json.loads(RUNTIME.read_text())
    port = _free_port()
    entry = {"plugin": "files_lake", "lake_id": "public_panels", "title": "disposable panels", "description": "test",
             "kind": "lake", "engine": "files_inventory", "root_path": str(work / "panels"),
             "include_globs": [resource], "untimed": [resource], "time_column": None, "time_columns": {},
             "time_unit": None, "holdout_start": A.HOLDOUT_START,
             "resource_contracts": {resource: A.resource_contract("uci_235")}}
    # a DISPOSABLE DuckDB cube, served by the deployed warehouse host, so the terminal really lands in
    # an accounting and a warehouse; the production cube is never opened
    cube_port = _free_port()
    cube_cfg = json.loads(A.WAREHOUSE_CONFIG.read_text())
    cube_cfg["web_port"] = cube_port
    cube_cfg["backend"]["settings"] = {**cube_cfg["backend"]["settings"], "duckdb_path": str(work / "cube.duckdb"),
                                       "min_free_bytes": 1 << 20}
    cube_cfg["operator_config_path"] = str(work / "cube.pending.json")
    (work / "cube.host.json").write_text(json.dumps(cube_cfg, indent=1))
    # both disposable hosts share a disposable token, the way the deployed pair shares one through its
    # service environment; no production token is read or written here
    token = hashlib.sha256(str(work).encode()).hexdigest()
    cube_log = open(work / "cube.log", "w")
    cube_proc = subprocess.Popen([str(A.WAREHOUSE_PYTHON), "-m", "data_warehouse_service.main",
                                  "--load_config", str(work / "cube.host.json")],
                                 cwd=str(work), stdout=cube_log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "DATA_GOV_LAKE_TOKEN": token})
    cube = [l for l in cfg["lakes"] if l.get("lake_id") == "olap_cube"]
    for lake in cube:
        lake["base_url"] = f"http://127.0.0.1:{cube_port}"
        lake["lake_service_token"] = token
    cfg["lakes"] = [l for l in cfg["lakes"] if l.get("plugin") != "http_lake"] + cube + [entry]
    cfg["policies"] = [p for p in cfg["policies"] if p.get("lake") in ("predictor_examples", "olap_cube")] + A.policy_entries(["predictor"])
    cfg.update(web_port=port, accounting_db=str(work / "accounting.db"), spool_dir=str(work / "spool"),
               cuts_dir=str(work / "cuts"), save_config=str(work / "effective.json"),
               operator_config_path=str(work / "pending.json"))
    cfg_path = work / "stack.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    log = open(work / "service.log", "w")
    proc = subprocess.Popen([str(PYTHON), "-m", "app.main", "--load_config", str(cfg_path)], cwd=str(GOV_APP),
                            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(GOV_APP)})
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        if A.http_json(f"{url}/healthz")[0] == 200:
            break
        time.sleep(0.5)
    else:
        proc.kill()
        cube_proc.kill()
        pytest.skip(f"the disposable data-gov did not come up: {(work / 'service.log').read_text()[-500:]}")
    yield {"url": url, "work": work, "resource": resource, "digest": digest, "port": port, "cube_port": cube_port}
    for p_ in (proc, cube_proc):
        p_.terminate()
        try:
            p_.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p_.kill()
    log.close()
    cube_log.close()


def _design(stack, root: Path) -> dict:
    d = P.seal(window=30, horizon=10, dev_train_days=2, dev_val_days=1, seeds=(1,), max_updates=20, ae_updates=20,
               batch=32, patience_epochs=2, pilot_updates=5,
               declared_task={"context_physical_seconds": 1800, "horizon_physical_seconds": 600, "purge": 40,
                              "usable_windows_all_targets_valid": {}})
    d.pop("design_sha256")
    d["governed_bytes"] = {"path": str(stack["work"] / "panels" / stack["resource"]), "sha256": stack["digest"]}
    d["dev_subpartition"]["rows"] = [0, 3 * P.DAY]
    d["data_access"] = "GOVERNED_DELIVERY"
    d["design_sha256"] = E.sha_obj(d)
    root.mkdir(parents=True, exist_ok=True)
    (root / "DESIGN.json").write_text(json.dumps(d, indent=1))
    return d


def test_RP38_without_a_delivery_nothing_is_prepared_and_nothing_is_fitted(stack, tmp_path):
    root = tmp_path / "nodelivery"
    design = _design(stack, root)
    with pytest.raises(G.GovernanceUnavailable, match="no governed delivery"):
        P.prepare(design, root)
    assert not (root / "DATA.npz").exists()


def test_RP38_an_absent_or_silent_governance_host_blocks_new_work(stack, tmp_path):
    root = tmp_path / "nohost"
    design = _design(stack, root)
    with pytest.raises(G.GovernanceUnavailable, match="unreachable|not registered"):
        G.acquire(run_id="t", root=root, lake="public_panels", resource=stack["resource"],
                  gov_url=f"http://127.0.0.1:{_free_port()}", api_key_file=KEY,
                  design_sha256=design["design_sha256"], cache_dir=root / "cache")
    assert not (root / "DELIVERIES.json").exists()


def test_RP38_a_resource_the_lake_does_not_serve_blocks_new_work(stack, tmp_path):
    root = tmp_path / "noresource"
    design = _design(stack, root)
    with pytest.raises((G.GovernanceUnavailable, SystemExit)):
        G.acquire(run_id="t", root=root, lake="public_panels", resource="uci_501_beijing/panel.parquet",
                  gov_url=stack["url"], api_key_file=KEY,
                  design_sha256=design["design_sha256"], cache_dir=root / "cache")


def test_RP38_the_campaign_precedes_the_data_and_the_run_consumes_the_delivered_bytes(stack, tmp_path):
    root = tmp_path / "route"
    design = _design(stack, root)
    doc = G.acquire(run_id="rp38-route", root=root, lake="public_panels", resource=stack["resource"],
                    unit_id="prepare", gov_url=stack["url"], api_key_file=KEY,
                    design_sha256=design["design_sha256"], cache_dir=root / "cache",
                    expect_sha256=stack["digest"])
    first = doc["units"]["prepare"]
    assert first["sha256"] == stack["digest"] and first["bytes_on_disk_sha256"] == stack["digest"]
    assert first["availability_use"] == "UNDECLARED" and first["campaign_sha256"] and not first["cached"]
    # a second unit gets its own campaign and delivery, served from the verified cache: reuse is measured
    doc = G.acquire(run_id="rp38-route", root=root, lake="public_panels", resource=stack["resource"],
                    unit_id="ae_s1", gov_url=stack["url"], api_key_file=KEY,
                    design_sha256=design["design_sha256"], cache_dir=root / "cache", expect_sha256=stack["digest"])
    assert doc["units"]["ae_s1"]["cached"] and doc["transfer"]["transferred_units"] == 1 and doc["transfer"]["cache_reused_units"] == 1
    assert doc["units"]["ae_s1"]["campaign_sha256"] != first["campaign_sha256"]
    data = P.prepare(design, root)
    assert data["data_access"] == "GOVERNED_DELIVERY" and data["panel_sha256"] == stack["digest"]
    assert data["campaign_sha256"] == first["campaign_sha256"] and data["delivery"]["delivery_id"] == first["delivery_id"]
    # the delivered bytes are the ones on disk; if they change afterwards the run refuses to continue
    shutil.copy2(root / "DELIVERIES.json", root / "DELIVERIES.backup.json")
    delivered = Path(first["path"])
    tampered = tmp_path / "tampered.parquet"
    shutil.copy2(delivered, tampered)
    with open(delivered, "ab") as fh:
        fh.write(b"0")
    with pytest.raises(SystemExit, match="changed after the delivery"):
        G.require_delivery(root, design)
    shutil.copy2(tampered, delivered)
    assert G.require_delivery(root, design)["delivery"]["sha256"] == stack["digest"]
    with pytest.raises(G.GovernanceUnavailable, match="has no governed delivery of its own"):
        G.require_delivery(root, design, "R0_s1")


def _run_governed(stack, root, design, **kw):
    """The REAL runner entry point, against the disposable governance stack."""
    return P.run(design, root=root, run_id=kw.pop("run_id", "rp42-run"), cap_seconds=kw.pop("cap", 600.0),
                 already_spent=0.0, gov_url=stack["url"], api_key_file=KEY, lake="public_panels",
                 resource=stack["resource"], outbox_dir=str(root / "outbox"), trace=lambda *a, **k: None, **kw)


def test_RP42_the_real_runner_acquires_registers_and_reports_every_unit(stack, tmp_path, monkeypatch):
    """The dictum's probe on the real run measured 1 dispatch, 0 delivery checks, 0 registrations, 0
    reports and 1 local terminal. Here the same entry point is driven with the expensive child
    replaced by a deterministic one, and every unit must be delivered, registered and reported."""
    root = tmp_path / "governed-run"
    design = _design(stack, root)
    calls = {"children": []}

    def fake_isolated(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        calls["children"].append(job["cell_id"])
        Path(attempt_dir).mkdir(parents=True, exist_ok=True)
        (Path(attempt_dir) / "job.json").write_text(json.dumps({**job, "attempt_dir": str(attempt_dir)}, default=str))
        return {"outcome": "COMPLETED", "reason": "", "cost": {"cpu_seconds": 1.0, "wall_seconds": 1.0, "host": "test"},
                "score": {"kind": job["kind"], "cell_id": job["cell_id"], "seed": job["seed"],
                          "regime": job.get("regime", "R0"), "detector_unchanged": False,
                          "gradient_proof": {"detector_receives_gradient": True},
                          "initial_checkpoint": {"full_digest": "0" * 64},
                          "regime_setup": {"detector_digest_after_setup": "0" * 64},
                          "scores": {"validation": {"model": {"mase_mean": 0.9, "mae_mean": 0.5, "status": "MEDIDO"}}},
                          "training": {"updates": 1, "stop_reason": "UPDATE_BUDGET",
                                       "censoring": {"verdict": "STOPPED_ON_VALIDATION"}},
                          "pretraining": {"updates": 1, "stop_reason": "UPDATE_BUDGET",
                                          "reconstruction_val_mse_masked": 0.1},
                          "detector_sha256": "0" * 64,
                          "parameters": {"trainable": 1, "total": 1, "frozen": 0}, "cost": {"fit_seconds": 1.0}}}
    monkeypatch.setattr(P, "run_isolated", fake_isolated)
    monkeypatch.setattr(P, "_closure_verdict", lambda attempt_dir, job, score: (score, None))
    report = _run_governed(stack, root, design, pilot_only=True)
    delivered = json.loads((root / "DELIVERIES.json").read_text())["units"]
    assert "prepare" in delivered and set(delivered) >= {c["cell_id"] for c in design["pilots"]}
    assert calls["children"], "no child ran"
    for record in report["terminals"]:
        assert record["governed"] and record["terminal_sent"] == 1 and record["terminal_pending"] == 0
        assert not record["reconciliation"]["missing_units"]
        assert record["terminal_sha256"], "the service's own terminal digest was not kept"
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())
    assert receipts["schema"] == "df_e1_terminal_receipts.v1"
    assert set(receipts["units"]) >= {"prepare"} | {c["cell_id"] for c in design["pilots"]}
    for unit, receipt in receipts["units"].items():
        assert receipt["terminal_sha256"] and receipt["accepted_at"] and receipt["campaign_sha256"]
        assert receipt["reconciliation"]["http"] == 200 and not receipt["reconciliation"]["missing_units"]
    assert not (root / "TERMINALS").exists() or not list((root / "TERMINALS").glob("*.json"))
    assert not (root / "CAMPAIGN_PROPOSAL.json").exists()


def test_RP42_a_unit_without_its_own_delivery_or_with_a_foreign_one_does_not_run(stack, tmp_path, monkeypatch):
    root = tmp_path / "missing-delivery"
    design = _design(stack, root)
    G.acquire(run_id="rp42-partial", root=root, lake="public_panels", resource=stack["resource"], unit_id="prepare",
              gov_url=stack["url"], api_key_file=KEY, design_sha256=design["design_sha256"],
              cache_dir=root / "cache", expect_sha256=stack["digest"])
    data = P.prepare(design, root)
    with pytest.raises(G.GovernanceUnavailable, match="has no governed delivery of its own"):
        G.require_delivery(root, design, "pilot_ae")
    # a receipt from ANOTHER run does not serve this one
    foreign = json.loads((root / "DELIVERIES.json").read_text())
    foreign["design_sha256"] = "0" * 64
    (root / "DELIVERIES.json").write_text(json.dumps(foreign))
    with pytest.raises(SystemExit, match="belongs to another design"):
        G.require_delivery(root, design, "prepare")


def test_RP42_cached_data_does_not_skip_the_delivery_check(stack, tmp_path):
    root = tmp_path / "cached"
    design = _design(stack, root)
    G.acquire(run_id="rp42-cached", root=root, lake="public_panels", resource=stack["resource"], unit_id="prepare",
              gov_url=stack["url"], api_key_file=KEY, design_sha256=design["design_sha256"],
              cache_dir=root / "cache", expect_sha256=stack["digest"])
    first = P.prepare(design, root)
    assert (root / "DATA.npz").is_file()
    again = P.prepare(design, root)
    assert again["delivery_recheck"]["delivery_id"] == json.loads((root / "DELIVERIES.json").read_text())["units"]["prepare"]["delivery_id"]
    (root / "DELIVERIES.json").unlink()
    with pytest.raises(G.GovernanceUnavailable, match="no governed delivery"):
        P.prepare(design, root)


def test_RP42_a_child_that_really_fails_still_closes_its_unit(stack, tmp_path, monkeypatch):
    root = tmp_path / "failing-child"
    design = _design(stack, root)

    def failing(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        Path(attempt_dir).mkdir(parents=True, exist_ok=True)
        return {"outcome": "RESOURCE_EXCEEDED", "reason": "CPU_TIME_LIMIT", "score": None,
                "cost": {"cpu_seconds": 2.0, "wall_seconds": 2.0, "host": "test"}}
    monkeypatch.setattr(P, "run_isolated", failing)
    report = _run_governed(stack, root, design, pilot_only=True, run_id="rp42-fail")
    assert report["stopped"] and "COST_PILOT_FAILED" in report["stopped"]
    assert report["terminals"] and all(r["governed"] for r in report["terminals"])
    assert {r["unit_id"] for r in report["terminals"]} >= {"prepare", "pilot_ae"}
    assert any(r["status"] == "FAILED" for r in report["terminals"])       # the child that failed closed too
    assert all(not r["reconciliation"]["missing_units"] for r in report["terminals"])
    # RP51: the population is named, and the units that never ran are not absorbed into a total
    result = report["governance_result"]
    assert result["all_units_governed"] is False and result["units_incomplete"]
    assert result["population"]["prepare"] == "CLOSED" and result["population"]["pilot_ae"] == "CLOSED"
    assert all(result["population"][c["cell_id"]] == "NOT_STARTED" for c in design["cells"])
    assert result["counts"]["PENDING"] == 0, result["reasons"]


def test_RP38_a_units_terminal_reaches_the_accounting_and_the_campaign_reconciles(stack, tmp_path):
    root = tmp_path / "terminal"
    design = _design(stack, root)
    G.acquire(run_id="rp38-terminal", root=root, lake="public_panels", resource=stack["resource"],
              unit_id="ae_s1", gov_url=stack["url"], api_key_file=KEY, design_sha256=design["design_sha256"],
              cache_dir=root / "cache", expect_sha256=stack["digest"])
    R = _load("df_utility_run")
    terminal = R._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": 1.0, "cpu_seconds": 1.0},
                           metrics=[R._metric("e1.mase_validation", 0.9, "mase", split="validation", horizon=10)],
                           started=R.now_iso(), finished=R.now_iso(),
                           tags={"purpose": "E1_DEV_PILOT", "classification": "NON_GOVERNING", "phase": "DEVELOPMENT"})
    out = G.report_terminal(root, "ae_s1", terminal, gov_url=stack["url"], api_key_file=KEY,
                            outbox_dir=str(tmp_path / "outbox"))
    assert out["flushed"]["sent"] == 1 and out["flushed"]["pending"] == 0, out["flushed"]["failures"]
    rec = out["reconciliation"]
    assert rec["http"] == 200 and not rec["accounting_only"] and not rec["lake_only"]
    assert "ae_s1" not in (rec["missing_units"] or [])


# --- RP51: receipts, population and recovery, over the real HTTP stack -------------------------------

def test_RP51_a_destination_that_goes_down_and_comes_back_loses_no_terminal(stack, tmp_path, monkeypatch):
    """The outbox keeps what the service refused, and a second flush closes the unit."""
    root = tmp_path / "recovering"
    design = _design(stack, root)
    G.acquire(run_id="rp51-down", root=root, lake="public_panels", resource=stack["resource"], unit_id="ae_s1",
              gov_url=stack["url"], api_key_file=KEY, design_sha256=design["design_sha256"],
              cache_dir=root / "cache", expect_sha256=stack["digest"])
    R = _load("df_utility_run")
    terminal = R._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": 1.0, "cpu_seconds": 1.0},
                           metrics=[R._metric("e1.mase_validation", 0.9, "mase", split="validation", horizon=10)],
                           started=R.now_iso(), finished=R.now_iso(),
                           tags={"purpose": "E1_DEV_PILOT", "classification": "NON_GOVERNING", "phase": "DEVELOPMENT"})
    outbox_dir = tmp_path / "outbox"
    # the destination is unreachable: the terminal stays pending and nothing is lost
    with pytest.raises(Exception):               # the destination is down: the send fails, nothing is lost
        G.report_terminal(root, "ae_s1", terminal, gov_url=f"http://127.0.0.1:{_free_port()}",
                          api_key_file=KEY, outbox_dir=str(outbox_dir))
    GR = _load("governed_run")
    # each unit reports through its own spool, so parallel children never flush each other's envelopes
    spool = outbox_dir / "units" / "ae_s1"
    assert not list((outbox_dir / "pending").glob("*.json")) if (outbox_dir / "pending").is_dir() else True
    pending = GR.TerminalOutbox(spool).status()["pending"]
    assert len(pending) == 1 and pending[0]["unit_id"] == "ae_s1"
    assert not (root / "TERMINAL_RECEIPTS.json").is_file(), "a receipt was written for a terminal nobody accepted"
    # the destination returns: the SAME envelope is accepted, once
    again = G.report_terminal(root, "ae_s1", terminal, gov_url=stack["url"], api_key_file=KEY,
                              outbox_dir=str(outbox_dir))
    assert again["flushed"]["sent"] == 1 and again["flushed"]["pending"] == 0
    assert again["persisted_receipt"]["terminal_sha256"] and not again["reconciliation"]["missing_units"]
    third = G.report_terminal(root, "ae_s1", terminal, gov_url=stack["url"], api_key_file=KEY,
                              outbox_dir=str(outbox_dir))
    assert third["flushed"]["sent"] in (0, 1) and third["flushed"]["pending"] == 0     # idempotent, never duplicated
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    assert list(receipts) == ["ae_s1"]


def test_RP51_an_empty_report_does_not_pass_and_a_registered_unit_that_did_not_close_is_named(stack, tmp_path):
    root = tmp_path / "empty"
    design = _design(stack, root)
    RC = _load("df_e1_receipts")
    empty = P._governed_summary.__wrapped__ if hasattr(P._governed_summary, "__wrapped__") else P._governed_summary
    (root / "DELIVERIES.json").write_text(json.dumps({"schema": "df_e1_governed_acquisition.v1",
                                                      "design_sha256": design["design_sha256"],
                                                      "lake": "public_panels", "resource": stack["resource"],
                                                      "role": "panel", "units": {}}))
    states = RC.population_states(design, root)
    assert states["complete"] is False and states["counts"]["NOT_STARTED"] == len(states["units"])
    result = empty(root, {"terminals": []}, design)
    assert result["all_units_governed"] is False and result["units_incomplete"]
    # a unit whose campaign exists but never closed is PENDING, with its reason
    G.acquire(run_id="rp51-pending", root=root, lake="public_panels", resource=stack["resource"], unit_id="pilot_ae",
              gov_url=stack["url"], api_key_file=KEY, design_sha256=design["design_sha256"],
              cache_dir=root / "cache", expect_sha256=stack["digest"])
    states = RC.population_states(design, root)
    assert states["units"]["pilot_ae"] == RC.PENDING and "registered" in states["reasons"]["pilot_ae"]
    assert states["counts"]["PENDING"] == 1


def test_RP51_the_last_unit_of_a_run_is_closed_like_any_other(stack, tmp_path, monkeypatch):
    root = tmp_path / "last-unit"
    design = _design(stack, root)

    def fake_isolated(job, *, attempt_dir, assigned_bytes, wall_seconds, cpu_seconds):
        Path(attempt_dir).mkdir(parents=True, exist_ok=True)
        (Path(attempt_dir) / "job.json").write_text(json.dumps({**job, "attempt_dir": str(attempt_dir)}, default=str))
        return {"outcome": "COMPLETED", "reason": "", "cost": {"cpu_seconds": 1.0, "wall_seconds": 1.0, "host": "test"},
                "score": {"kind": job["kind"], "cell_id": job["cell_id"], "seed": job["seed"],
                          "regime": job.get("regime", "R0"), "detector_unchanged": False,
                          "gradient_proof": {"detector_receives_gradient": True},
                          "initial_checkpoint": {"full_digest": "0" * 64},
                          "regime_setup": {"detector_digest_after_setup": "0" * 64},
                          "scores": {"validation": {"model": {"mase_mean": 0.9, "mae_mean": 0.5, "status": "MEDIDO"}}},
                          "training": {"updates": 1, "stop_reason": "UPDATE_BUDGET",
                                       "censoring": {"verdict": "STOPPED_ON_VALIDATION"}},
                          "pretraining": {"updates": 1, "stop_reason": "UPDATE_BUDGET",
                                          "reconstruction_val_mse_masked": 0.1},
                          "detector_sha256": "0" * 64,
                          "parameters": {"trainable": 1, "total": 1, "frozen": 0}, "cost": {"fit_seconds": 1.0}}}
    monkeypatch.setattr(P, "run_isolated", fake_isolated)
    monkeypatch.setattr(P, "_closure_verdict", lambda attempt_dir, job, score: (score, None))
    report = _run_governed(stack, root, design, pilot_only=True, run_id="rp51-last")
    result = report["governance_result"]
    closed = [u for u, s in result["population"].items() if s == "CLOSED"]
    assert set(closed) >= {"prepare", "pilot_ae", "pilot_fit"}
    receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())["units"]
    assert set(receipts) == set(closed)
    last = report["terminals"][-1]
    assert last["terminal_sent"] == 1 and last["terminal_pending"] == 0 and last["terminal_sha256"]
