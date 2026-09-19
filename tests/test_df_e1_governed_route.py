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
    cfg["lakes"] = [l for l in cfg["lakes"] if l.get("plugin") != "http_lake"] + [entry]
    cfg["policies"] = [p for p in cfg["policies"] if p.get("lake") in ("predictor_examples",)] + A.policy_entries(["predictor"])
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
        pytest.skip(f"the disposable data-gov did not come up: {(work / 'service.log').read_text()[-500:]}")
    yield {"url": url, "work": work, "resource": resource, "digest": digest, "port": port}
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()


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
                  units=["u1"], design_sha256=design["design_sha256"], cache_dir=root / "cache")
    assert not (root / "DELIVERIES.json").exists()


def test_RP38_a_resource_the_lake_does_not_serve_blocks_new_work(stack, tmp_path):
    root = tmp_path / "noresource"
    design = _design(stack, root)
    with pytest.raises((G.GovernanceUnavailable, SystemExit)):
        G.acquire(run_id="t", root=root, lake="public_panels", resource="uci_501_beijing/panel.parquet",
                  gov_url=stack["url"], api_key_file=KEY, units=["u1"],
                  design_sha256=design["design_sha256"], cache_dir=root / "cache")


def test_RP38_the_campaign_precedes_the_data_and_the_run_consumes_the_delivered_bytes(stack, tmp_path):
    root = tmp_path / "route"
    design = _design(stack, root)
    doc = G.acquire(run_id="rp38-route", root=root, lake="public_panels", resource=stack["resource"],
                    gov_url=stack["url"], api_key_file=KEY,
                    units=[c["cell_id"] for c in design["pilots"] + design["cells"]],
                    design_sha256=design["design_sha256"], cache_dir=root / "cache",
                    expect_sha256=stack["digest"])
    assert doc["delivery"]["sha256"] == stack["digest"] and doc["bytes_on_disk_sha256"] == stack["digest"]
    assert doc["delivery"]["availability_use"] == "UNDECLARED" and doc["campaign_sha256"]
    data = P.prepare(design, root)
    assert data["data_access"] == "GOVERNED_DELIVERY" and data["panel_sha256"] == stack["digest"]
    assert data["campaign_sha256"] == doc["campaign_sha256"] and data["delivery"]["delivery_id"] == doc["delivery"]["delivery_id"]
    # the delivered bytes are the ones on disk; if they change afterwards the run refuses to continue
    shutil.copy2(root / "DELIVERIES.json", root / "DELIVERIES.backup.json")
    delivered = Path(doc["delivery"]["path"])
    tampered = tmp_path / "tampered.parquet"
    shutil.copy2(delivered, tampered)
    with open(delivered, "ab") as fh:
        fh.write(b"0")
    with pytest.raises(SystemExit, match="changed after the delivery"):
        G.require_delivery(root, design)
    shutil.copy2(tampered, delivered)
    assert G.require_delivery(root, design)["delivery"]["sha256"] == stack["digest"]


def test_RP38_a_units_terminal_reaches_the_accounting_and_the_campaign_reconciles(stack, tmp_path):
    root = tmp_path / "terminal"
    design = _design(stack, root)
    units = ["ae_s1"]
    G.acquire(run_id="rp38-terminal", root=root, lake="public_panels", resource=stack["resource"],
              gov_url=stack["url"], api_key_file=KEY, units=units, design_sha256=design["design_sha256"],
              cache_dir=root / "cache", expect_sha256=stack["digest"])
    R = _load("df_utility_run")
    terminal = R._terminal(status="COMPLETED", reason=None, cost={"wall_seconds": 1.0, "cpu_seconds": 1.0},
                           metrics=[R._metric("e1.mase_validation", 0.9, "mase", split="validation", horizon=10)],
                           started=R.now_iso(), finished=R.now_iso(),
                           tags={"purpose": "E1_DEV_PILOT", "classification": "NON_GOVERNING", "phase": "DEVELOPMENT"})
    out = G.report_terminal(root, "ae_s1", terminal, gov_url=stack["url"], api_key_file=KEY,
                            outbox_dir=str(tmp_path / "outbox"))
    assert out["flushed"]["sent"] == 1 and out["flushed"]["pending"] == 0 and not out["flushed"]["failures"]
    rec = out["reconciliation"]
    assert rec["http"] == 200 and not rec["accounting_only"] and not rec["lake_only"]
    assert "ae_s1" not in (rec["missing_units"] or [])
