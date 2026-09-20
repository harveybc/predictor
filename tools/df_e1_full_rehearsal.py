#!/usr/bin/env python3
"""RP55: a WHOLE new run, rehearsed in isolation, before anything is adopted.

Not another mock: this starts the three services on disposable copies, seals a small design over a
SYNTHETIC panel (declared as such), and drives the REAL runner end to end — acquisition per unit,
preparation from the delivered bytes, an auto-encoder and a fit that really train in their own
subprocesses, terminals accepted by the service, receipts persisted by the client, campaigns
reconciled, and the warehouse read back BY CONTENT through the canonical reader.

It produces two outcomes on purpose:

    a small POSITIVE   the units that must close, do close, with their metrics in the cube;
    a real FAILURE     one unit is given an impossible budget so its child fails; its campaign is
                       closed too, its state is named, and the population is not called complete.

Then it restores: the disposable services are stopped and their health is checked, so the rehearsal
leaves nothing running and nothing changed outside its own directory.

    python tools/df_e1_full_rehearsal.py --out REHEARSAL.json [--keep]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
HOME = Path.home()
SCHEMA = "df_e1_full_rehearsal.v1"


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = _load("df_public_lake_adopt")
P = _load("df_e1_pilot")
C = _load("df_e1_close")
G = _load("df_e1_governed")
RC = _load("df_e1_receipts")
E = _load("df_mod_e0")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def synthetic_panel(path: Path, *, days: int = 5, seed: int = 21) -> str:
    """A SYNTHETIC panel with the household panel's shape. It is a software fixture, declared as one."""
    rng = np.random.default_rng(seed)
    n = days * P.DAY
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    base = np.sin(2 * np.pi * np.arange(n) / P.DAY) + 0.1 * rng.normal(size=n)
    frame = pd.DataFrame({c: base * (i + 1) + rng.normal(size=n) * 0.05 + i for i, c in enumerate(cols)})
    stamps = pd.date_range("2009-03-01 00:00", periods=n, freq="min")
    frame.insert(0, "timestamp_label", stamps.strftime("%d/%m/%Y %H:%M:%S"))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return A.sha_file(path)


def start_stack(work: Path, panel_root: Path, resource: str, report: dict):
    """The three services, on disposable copies, with the corrected provider serving the lake."""
    token = hashlib.sha256(str(work).encode()).hexdigest()
    ports = {"lake": free_port(), "cube": free_port(), "gov": free_port()}
    contract = A.resource_contract("uci_235")
    host_cfg = {"store_id": A.LAKE_ID, "title": "rehearsal panels", "description": "disposable", "kind": "lake",
                "engine": "files_inventory", "transport": "http", "web_host": "127.0.0.1", "web_port": ports["lake"],
                "operator_config_path": str(work / "lake.pending.json"),
                "backend": {"entry_point": "financial_files", "distribution": "financial-data-store",
                            "settings": {"root_path": str(panel_root), "include_globs": [resource],
                                         "untimed": [resource], "holdout_start": A.HOLDOUT_START,
                                         "resource_contracts": {resource: contract}}}}
    (work / "lake.host.json").write_text(json.dumps(host_cfg, indent=1))
    cube_cfg = json.loads(A.WAREHOUSE_CONFIG.read_text())
    cube_cfg["web_port"] = ports["cube"]
    cube_cfg["backend"]["settings"] = {**cube_cfg["backend"]["settings"], "duckdb_path": str(work / "cube.duckdb"),
                                       "min_free_bytes": 1 << 20}
    cube_cfg["operator_config_path"] = str(work / "cube.pending.json")
    (work / "cube.host.json").write_text(json.dumps(cube_cfg, indent=1))
    cfg = json.loads(A.RUNTIME_CONFIG.read_text())
    stack = json.loads(json.dumps(cfg))
    stack["lakes"] = [l for l in stack["lakes"] if l.get("lake_id") == "olap_cube"] + [A.lake_entry_http(ports["lake"], token)]
    for lake in stack["lakes"]:
        if lake.get("lake_id") == "olap_cube":
            lake["base_url"] = f"http://127.0.0.1:{ports['cube']}"
            lake["lake_service_token"] = token
    stack["policies"] = [p for p in stack["policies"] if p.get("lake") == "olap_cube"] + A.policy_entries(["predictor"])
    stack.update(web_port=ports["gov"], accounting_db=str(work / "accounting.db"), spool_dir=str(work / "spool"),
                 cuts_dir=str(work / "cuts"), save_config=str(work / "effective.json"),
                 operator_config_path=str(work / "gov.pending.json"))
    (work / "gov.json").write_text(json.dumps(stack, indent=1))
    procs, logs = {}, {}
    for name, argv, cwd, extra in (
            ("lake", [str(A.LAKE_HOST_PYTHON), "-m", "data_lake_service.main", "--load_config", str(work / "lake.host.json")],
             work, {"PYTHONPATH": str(A.CANDIDATE_PROVIDER)}),
            ("cube", [str(A.WAREHOUSE_PYTHON), "-m", "data_warehouse_service.main", "--load_config", str(work / "cube.host.json")],
             work, {}),
            ("gov", [str(A.PYTHON), "-m", "app.main", "--load_config", str(work / "gov.json")], A.GOV_APP,
             {"PYTHONPATH": str(A.GOV_APP)})):
        logs[name] = open(work / f"{name}.log", "w")
        procs[name] = subprocess.Popen(argv, cwd=str(cwd), stdout=logs[name], stderr=subprocess.STDOUT,
                                       env={**os.environ, "DATA_GOV_LAKE_TOKEN": token, **extra})
    for name, port in (("lake", ports["lake"]), ("cube", ports["cube"]), ("gov", ports["gov"])):
        if not A._service_healthy(f"http://127.0.0.1:{port}", tries=120):
            report["startup_failed"] = {name: (work / f"{name}.log").read_text()[-2000:]}
            raise SystemExit(f"REFUSED: the disposable {name} did not come up")
    return procs, logs, ports, token


def seal_small(panel: Path, digest: str, *, root: Path) -> dict:
    design = P.seal(window=30, horizon=10, dev_train_days=3, dev_val_days=1, seeds=(1,), max_updates=30,
                    ae_updates=30, batch=32, patience_epochs=2, pilot_updates=10, core_kind="conv3",
                    data_access="GOVERNED_DELIVERY",
                    declared_task={"context_physical_seconds": 1800, "horizon_physical_seconds": 600,
                                   "purge": 40, "usable_windows_all_targets_valid": {}})
    design.pop("design_sha256")
    design["governed_bytes"] = {"path": str(panel), "sha256": digest}
    design["dev_subpartition"]["rows"] = [0, 4 * P.DAY]
    design["what_this_is"] = ("A REHEARSAL of the whole governed route on a SYNTHETIC panel: it trains, but it "
                              "answers no scientific question and is never reported as an experiment.")
    design["design_sha256"] = E.sha_obj(design)
    root.mkdir(parents=True, exist_ok=True)
    (root / "DESIGN.json").write_text(json.dumps(design, indent=1))
    return design


def rehearse(out_path: Path, *, keep: bool = False) -> dict:
    work = Path(tempfile.mkdtemp(prefix="rp55-full-"))
    resource = "uci_235_individual_household_power/panel.parquet"
    panel_root = work / "panels"
    digest = synthetic_panel(panel_root / resource)
    report = {"schema": SCHEMA, "at": now_iso(), "work": str(work), "host": os.uname().nodename,
              "panel": {"synthetic": True, "sha256": digest, "declared": "a software fixture, not governed data"}}
    procs = logs = None
    try:
        procs, logs, ports, token = start_stack(work, panel_root, resource, report)
        report["ports"] = ports
        url = f"http://127.0.0.1:{ports['gov']}"
        key_file = work / "predictor.key"
        key_file.write_text(A.API_KEY_FILE.read_text().strip())
        # --- the POSITIVE: a small run that must close -------------------------------------------
        root = work / "run"
        design = seal_small(panel_root / resource, digest, root=root)
        started = time.process_time()
        result = P.run(design, root=root, run_id="rp55-positive", cap_seconds=3000.0, already_spent=0.0,
                       pilot_only=True, trace=lambda *a, **k: None, gov_url=url, api_key_file=key_file,
                       lake=A.LAKE_ID, resource=resource, outbox_dir=str(work / "outbox"))
        report["positive"] = {"stopped": result.get("stopped"), "governance": result.get("governance_result"),
                              "terminals": result.get("terminals"), "cpu_seconds": round(time.process_time() - started, 2)}
        receipts = json.loads((root / "TERMINAL_RECEIPTS.json").read_text())
        report["positive"]["receipts"] = receipts
        report["positive"]["warehouse"] = {}
        for unit, receipt in receipts["units"].items():
            content = A._warehouse_content(f"http://127.0.0.1:{ports['cube']}", token, receipt["campaign_sha256"],
                                           {unit: {"status": receipt["status"], "metrics": [], "costs": {},
                                                   "tags": {}}})
            report["positive"]["warehouse"][unit] = {"present": content.get("per_unit", {}).get(unit, {}).get("present"),
                                                     "terminal_sha256": content.get("per_unit", {}).get(unit, {}).get("terminal_sha256"),
                                                     "matches_receipt": content.get("per_unit", {}).get(unit, {}).get("terminal_sha256")
                                                                        == receipt["terminal_sha256"]}
        closure = C.close(root, do_replay=False)
        report["positive"]["closure"] = {"verdict": closure["verdict"], "counts": closure["counts"],
                                         "governed_units": closure["governance"]["units_governed"]}
        report["positive"]["ok"] = bool(
            result.get("governance_result", {}).get("all_units_governed") is not None
            and all(v["matches_receipt"] for v in report["positive"]["warehouse"].values())
            and closure["counts"]["metrics_verified"] >= 1)
        # --- the FAILURE: one unit cannot run, and that is closed too -----------------------------
        bad_root = work / "run-failure"
        bad = seal_small(panel_root / resource, digest, root=bad_root)
        bad_design = json.loads(json.dumps(bad))
        # a REAL failure of the real path: the child is given less memory than TensorFlow needs, so it
        # is killed by its own cgroup and the runner sees RESOURCE_EXCEEDED
        bad_design["budget"] = {**bad_design["budget"], "task_memory_bytes": 48 << 20, "wall_seconds": 120.0,
                                "cpu_seconds": 120}
        bad_design.pop("design_sha256")
        bad_design["design_sha256"] = E.sha_obj(bad_design)
        shutil.rmtree(bad_root)
        bad_root.mkdir(parents=True)
        (bad_root / "DESIGN.json").write_text(json.dumps(bad_design, indent=1))
        failed = P.run(bad_design, root=bad_root, run_id="rp55-failure", cap_seconds=1200.0, already_spent=0.0,
                       pilot_only=True, trace=lambda *a, **k: None, gov_url=url, api_key_file=key_file,
                       lake=A.LAKE_ID, resource=resource, outbox_dir=str(work / "outbox-failure"))
        states = failed.get("governance_result") or {}
        report["failure"] = {"stopped": failed.get("stopped"), "population": states.get("population"),
                             "reasons": states.get("reasons"), "counts": states.get("counts"),
                             "all_units_governed": states.get("all_units_governed"),
                             "terminals": failed.get("terminals")}
        report["failure"]["ok"] = bool(failed.get("stopped") and states.get("all_units_governed") is False
                                       and states.get("counts", {}).get("PENDING", 0) == 0
                                       and any(t.get("status") in ("FAILED", "INCONCLUSIVE")
                                               for t in failed.get("terminals", [])))
        report["ok"] = bool(report["positive"]["ok"] and report["failure"]["ok"])
    finally:
        report["restoration"] = _stop(procs, logs, work, report)
        out_path.write_text(json.dumps(report, indent=1, default=str))
        if not keep and report.get("ok"):
            shutil.rmtree(work, ignore_errors=True)
    return report


def _stop(procs, logs, work: Path, report: dict) -> dict:
    """Stop what this rehearsal started and prove nothing of it is still answering."""
    stopped = {}
    for name, proc in (procs or {}).items():
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        stopped[name] = {"returncode": proc.returncode}
    for name, handle in (logs or {}).items():
        handle.close()
        report[f"{name}_log_tail"] = (work / f"{name}.log").read_text()[-400:]
    for name, port in (report.get("ports") or {}).items():
        stopped.setdefault(name, {})["still_answering"] = A.http_json(f"http://127.0.0.1:{port}/healthz")[0] == 200
    stopped["production_untouched"] = {u: A.service_state(u)["ActiveState"] for u in
                                       ("crispdm-data-gov.service", "crispdm-data-warehouse-olap.service")}
    stopped["all_stopped"] = all(not v.get("still_answering") for v in stopped.values() if isinstance(v, dict))
    return stopped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    doc = rehearse(a.out, keep=a.keep)
    print(json.dumps({"ok": doc.get("ok"), "positive": (doc.get("positive") or {}).get("ok"),
                      "failure": (doc.get("failure") or {}).get("ok"),
                      "restoration": (doc.get("restoration") or {}).get("all_stopped"),
                      "closure": (doc.get("positive") or {}).get("closure")}, indent=1, default=str))
    return 0 if doc.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
