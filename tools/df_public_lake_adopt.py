#!/usr/bin/env python3
"""RP33: publish the already characterised public panels as a BOUNDED resource of the deployed
data-gov, by the existing operating procedure (inventory -> backup -> rehearsal on a disposable
stack -> change -> restart -> post-check -> write-once receipt), so that the E1 work reads its
bytes through governance instead of off the filesystem.

What is published and what is NOT:
  * two resources only — the household (UCI 235) and electricity (UCI 321) panels already
    characterised in E1_TASKS; the other panels under the same root are NOT in the globs.
  * `untimed`: the panels' timestamp labels are DATA, not evidence of when a row became
    available, so the lake may not derive availability from them. No `availability` block is
    declared, which is what the lake calls UNDECLARED and what every receipt will carry.
  * `holdout_start` in the distant past: every date-ranged request is refused. Only the whole
    resource is deliverable, AS_IS. UNKNOWN availability stays UNKNOWN: no point-in-time and no
    live-equivalent eligibility is created by having a digest.
  * use class ARCHIVE/DEV, stated in the lake's description and in the resource sheet.

No new backend, no new service, no copy of the bytes, no global census: one lake entry and one
policy inside the config the deployed data-gov already loads, and only that service restarts.

    python tools/df_public_lake_adopt.py inventory --out INVENTORY.json
    python tools/df_public_lake_adopt.py rehearse  --out REHEARSAL.json
    python tools/df_public_lake_adopt.py adopt     --state-dir DIR          # backup + change + restart + post-check
    python tools/df_public_lake_adopt.py route     --out ROUTE.json         # the full route against the live service
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HOME = Path.home()
SERVICE = "crispdm-data-gov.service"
GOV_URL = "http://127.0.0.1:5055"
RUNTIME_CONFIG = HOME / ".local/state/crispdm-data-foundation/musashi-store-adoption-20260914T181826Z/5055.runtime.json"
API_KEY_FILE = HOME / ".local/state/crispdm-data-foundation/satoshi-store-hosts-20260914T1215Z/predictor.key"
PANEL_ROOT = HOME / ".local/state/crispdm-data-foundation/public_panels_c126_v2"
TASKS_FILE = REPO / "docs/tres_temas_entrevista/program_v3/E1_TASKS.json"
GOV_APP = HOME / "Documents/GitHub/.worktrees/musashi-n3-data-gov-20260914T063541Z"
PYTHON = HOME / "anaconda3/envs/trading-stack/bin/python3.12"

LAKE_ID = "public_panels"
RESOURCES = {
    "uci_235_individual_household_power/panel.parquet": "uci_235",
    "uci_321_electricityloaddiagrams20112014/panel.parquet": "uci_321",
}
#: Every date range is refused against this holdout; only the whole resource is deliverable.
HOLDOUT_START = "1970-01-01"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(argv, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=kw.pop("timeout", 300), **kw)


def http_json(url: str, token: str | None = None, *, method="GET", body=None, timeout=60, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    head = {"Content-Type": "application/json", **(headers or {})}
    if token:
        head["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, method=method, headers=head)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw = answer.read().decode(errors="replace")
            try:
                payload = json.loads(raw or "{}")
            except ValueError:                                   # /healthz answers plain text
                payload = {"body": raw[:300]}
            return answer.status, payload, dict(answer.headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode(errors="replace")
        try:
            payload = json.loads(raw or "{}")
        except ValueError:
            payload = {"error": raw[:300]}
        return exc.code, payload, dict(exc.headers or {})
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc)}, {}


# --- the published contract ----------------------------------------------------------------------

def resource_contract(family: str) -> dict:
    """The five structural keys the lake requires, with NO availability block: the delivery's
    scope is UNDECLARED and every receipt says so."""
    ts = {"uci_235": ("%d/%m/%Y %H:%M:%S", "60s"), "uci_321": ("%Y-%m-%d %H:%M:%S", "900s")}[family]
    return {"event_time_column": "timestamp_label", "available_time_column": "timestamp_label",
            "timezone": "NAIVE_WALL_CLOCK", "time_unit": None, "frequency": ts[1]}


def lake_entry() -> dict:
    tasks = json.loads(TASKS_FILE.read_text())
    contracts, declared = {}, {}
    for resource, family in RESOURCES.items():
        fam = tasks["families"][family]
        path = PANEL_ROOT / resource
        if not path.is_file():
            raise SystemExit(f"REFUSED: {path} is not a file")
        digest = sha_file(path)
        if digest != fam["governed_bytes"]["sha256"]:
            raise SystemExit(f"REFUSED: {resource} bytes are not the characterised ones")
        contracts[resource] = resource_contract(family)
        receipt = fam["governed_bytes"]["receipt"]
        declared[resource] = {"dataset_id": fam["dataset_id"], "sha256": digest, "bytes": path.stat().st_size,
                              "rows": receipt["rows"], "columns": receipt["columns"], "producer": fam["source"],
                              "parser": receipt["parser"], "parse_rule": receipt["parse_rule"],
                              "timestamp_format": receipt["timestamp_format"], "units_declared": receipt["units_declared"],
                              "licence": "OPEN_ATTRIBUTION (CC-BY-4.0), as the catalogue of this family records it",
                              "roles": fam["roles"], "use_class": "ARCHIVE/DEV",
                              "availability": "UNDECLARED: the labels are data, not evidence of publication time; "
                                              "no completion lag, no point-in-time and no live-equivalent eligibility",
                              "time_zone": "UNKNOWN in the producer's contract (household); Portuguese wall clock stated by the "
                                           "producer (electricity) — neither is evidence of availability"}
    entry = {"plugin": "files_lake", "lake_id": LAKE_ID, "title": "Public panels (UCI 235 / 321) — retrospective archive",
             "description": "Read-only lake over the already characterised public panels. ARCHIVE/DEV use: whole-resource AS_IS "
                            "deliveries only; every date range is refused; availability UNDECLARED.",
             "kind": "lake", "engine": "files_inventory", "root_path": str(PANEL_ROOT),
             "include_globs": sorted(RESOURCES), "untimed": sorted(RESOURCES),
             "time_column": None, "time_columns": {}, "time_unit": None,
             "resource_contracts": contracts, "holdout_start": HOLDOUT_START}
    return {"entry": entry, "declared": declared}


def policy_entries(principals: list) -> list:
    # `deny_from` is required by the service whenever the lake declares a holdout: it denies every
    # request for data at or after that day, which for this archive is every day there is.
    return [{"principal": p, "lake": LAKE_ID, "verbs": ["discover", "coverage", "read", "download"],
             "deny_from": HOLDOUT_START} for p in principals]


# --- inventory ------------------------------------------------------------------------------------

def service_state(unit: str) -> dict:
    out = run(["systemctl", "--user", "show", unit, "--no-pager",
               "--property=ActiveState,SubState,NRestarts,ExecMainStartTimestamp,FragmentPath"])
    state = dict(line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line)
    return {**state, "returncode": out.returncode}


def inventory() -> dict:
    cfg = json.loads(RUNTIME_CONFIG.read_text())
    listening = []
    for port in (5055, 5056, 5057, 5058):
        with socket.socket() as s:
            s.settimeout(1.0)
            listening.append({"port": port, "open": s.connect_ex(("127.0.0.1", port)) == 0})
    units = ["crispdm-data-gov.service", "crispdm-data-lake-financial.service", "crispdm-data-warehouse-olap.service",
             "crispdm-data-lake-synthetic.service", "crispdm-olap-loader-duckdb.service"]
    return {"schema": "df_public_lake_inventory.v1", "at": now_iso(), "host": os.uname().nodename,
            "runtime_config": {"path": str(RUNTIME_CONFIG), "sha256": sha_file(RUNTIME_CONFIG)},
            "lakes": [{k: v for k, v in lake.items() if k in ("lake_id", "plugin", "kind", "engine", "base_url", "root_path")}
                      for lake in cfg["lakes"]],
            "policies": cfg["policies"], "principals": sorted(cfg["principals"]),
            "accounting_db": cfg["accounting_db"], "services": {u: service_state(u) for u in units},
            "ports": listening,
            "public_panels_present": {r: (PANEL_ROOT / r).is_file() for r in RESOURCES},
            "public_panel_lake_registered": any(l.get("lake_id") == LAKE_ID for l in cfg["lakes"]),
            "note": "read-only inventory; no service was started, stopped or configured by this command"}


# --- the successor configuration --------------------------------------------------------------------

def successor_config(cfg: dict, *, principals: list) -> dict:
    out = json.loads(json.dumps(cfg))
    built = lake_entry()
    out["lakes"] = [l for l in out["lakes"] if l.get("lake_id") != LAKE_ID] + [built["entry"]]
    out["policies"] = [p for p in out["policies"] if p.get("lake") != LAKE_ID] + policy_entries(principals)
    return out


def config_is_additive(before: dict, after: dict) -> dict:
    """The change adds one lake and its policies and alters nothing else."""
    b_l = {l["lake_id"]: l for l in before["lakes"]}
    a_l = {l["lake_id"]: l for l in after["lakes"]}
    unchanged = all(json.dumps(b_l[k], sort_keys=True) == json.dumps(a_l.get(k), sort_keys=True) for k in b_l)
    added = sorted(set(a_l) - set(b_l))
    b_p = [p for p in before["policies"] if p.get("lake") != LAKE_ID]
    a_p = [p for p in after["policies"] if p.get("lake") != LAKE_ID]
    other = {k: v for k, v in before.items() if k not in ("lakes", "policies")}
    other_after = {k: v for k, v in after.items() if k not in ("lakes", "policies")}
    return {"previous_lakes_unchanged": unchanged, "lakes_added": added,
            "other_policies_unchanged": json.dumps(b_p, sort_keys=True) == json.dumps(a_p, sort_keys=True),
            "everything_else_unchanged": json.dumps(other, sort_keys=True) == json.dumps(other_after, sort_keys=True),
            "additive": unchanged and added == [LAKE_ID]
                        and json.dumps(b_p, sort_keys=True) == json.dumps(a_p, sort_keys=True)
                        and json.dumps(other, sort_keys=True) == json.dumps(other_after, sort_keys=True)}


# --- the route (used by the rehearsal and against the live service) -------------------------------

def route_checks(gov_url: str, token: str, *, cache_dir: Path, run_id: str, resource: str,
                 expect_sha: str, lake: str = LAKE_ID) -> dict:
    """Register a campaign BEFORE any read, download the whole resource, verify the bytes, and
    prove the refusals: a date range, an undeclared resource, and a download without a campaign."""
    sys.path.insert(0, str(REPO / "tools"))
    import governed_run as GR

    key = f"{run_id}-lake-route"
    gov = GR.GovHttp(gov_url, token, key)
    unit = "route-1"
    code_identity = GR.strict_code_identity(REPO)          # the real checkout identity; a clean tree is required
    campaign = {"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
                "project": "predictor", "code_identity": code_identity, "config_sha256": hashlib.sha256(json.dumps(
                    {"route": "rp33", "lake": lake, "resource": resource}, sort_keys=True).encode()).hexdigest(),
                "input_mode": "DATASETS", "synthetic_spec_sha256": None, "units": [unit],
                "datasets": [{"lake": lake, "resource": resource, "role": "panel", "from": None, "to": None}],
                "terminal_lake": "olap_cube"}
    status, receipt = gov.submit_campaign(campaign)
    out = {"campaign_http": status, "campaign_sha256": receipt.get("campaign_sha256")}
    if status not in (200, 201):
        return {**out, "error": receipt}
    sha = receipt["campaign_sha256"]
    status, info = gov.governed_download(sha, unit, lake, resource, "panel", str(cache_dir))
    out["download"] = {"http": status, "state": info.get("state"), "bytes": info.get("bytes"),
                       "sha256": info.get("sha256"), "availability": info.get("availability"),
                       "delivery_id": info.get("delivery_id"), "cached": info.get("cached"), "path": info.get("path")}
    out["bytes_match_characterised"] = info.get("sha256") == expect_sha
    on_disk = sha_file(Path(info["path"]))
    out["bytes_on_disk_sha256"] = on_disk
    out["delivered_bytes_verified"] = on_disk == expect_sha
    # the refusals: a date range must not be deliverable for this archive
    rng = http_json(f"{gov_url}/api/v2/download?lake={lake}&resource={resource}&role=panel&from=2007-01-01&to=2007-01-02",
                    token, headers={"X-Experiment-Key": key, "X-Campaign-SHA256": sha, "X-Unit-ID": unit})
    out["ranged_request"] = {"http": rng[0], "body": rng[1]}
    absent = http_json(f"{gov_url}/api/v2/download?lake={lake}&resource=uci_501_beijing_multisite_air_quality/panel.parquet&role=panel",
                       token, headers={"X-Experiment-Key": key, "X-Campaign-SHA256": sha, "X-Unit-ID": unit})
    out["undeclared_resource"] = {"http": absent[0], "body": absent[1]}
    nocamp = http_json(f"{gov_url}/api/v2/download?lake={lake}&resource={resource}&role=panel", token,
                       headers={"X-Experiment-Key": key})
    out["download_without_campaign"] = {"http": nocamp[0], "body": nocamp[1]}
    out["refusals_hold"] = (out["ranged_request"]["http"] >= 400 and out["undeclared_resource"]["http"] >= 400
                            and out["download_without_campaign"]["http"] >= 400)
    return out


# --- rehearsal on a disposable stack ----------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rehearse(out_path: Path, *, keep: bool = False) -> dict:
    cfg = json.loads(RUNTIME_CONFIG.read_text())
    token = API_KEY_FILE.read_text().strip()
    work = Path(tempfile.mkdtemp(prefix="rp33-rehearsal-"))
    port = free_port()
    stack = successor_config(cfg, principals=["predictor", "satoshi-gamma", "satoshi-dragon"])
    stack.update(web_port=port, accounting_db=str(work / "accounting.db"), spool_dir=str(work / "spool"),
                 cuts_dir=str(work / "cuts"), save_config=str(work / "effective.json"),
                 operator_config_path=str(work / "pending.json"))
    stack["lakes"] = [l for l in stack["lakes"] if l.get("plugin") != "http_lake"]      # the disposable stack talks to no other service
    stack["policies"] = [p for p in stack["policies"] if p.get("lake") not in {"financial_files", "olap_cube", "governance_smoke"}]
    cfg_path = work / "stack.json"
    cfg_path.write_text(json.dumps(stack, indent=1))
    log = open(work / "service.log", "w")
    proc = subprocess.Popen([str(PYTHON), "-m", "app.main", "--load_config", str(cfg_path)], cwd=str(GOV_APP),
                            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(GOV_APP)})
    report = {"schema": "df_public_lake_rehearsal.v1", "at": now_iso(), "work": str(work), "port": port,
              "stack": "disposable data-gov with the successor configuration; no production service touched"}
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            if http_json(f"{url}/healthz")[0] == 200:
                break
            time.sleep(0.5)
        else:
            report["startup_failed"] = (work / "service.log").read_text()[-3000:]
            out_path.write_text(json.dumps(report, indent=1, default=str))
            raise SystemExit(f"REFUSED: the disposable stack did not come up; see {out_path}")
        status, lakes, _ = http_json(f"{url}/api/v1/lakes", token)
        report["lakes"] = {"http": status, "ids": sorted(l.get("lake_id") for l in (lakes.get("lakes") or lakes.get("items") or []))
                           if isinstance(lakes, dict) else None, "raw": str(lakes)[:400]}
        built = lake_entry()
        for resource in sorted(RESOURCES):
            report[resource] = route_checks(url, token, cache_dir=work / "cache", run_id=f"rp33-rehearsal-{int(time.time())}",
                                            resource=resource, expect_sha=built["declared"][resource]["sha256"])
        report["route_ok"] = all(report[r].get("delivered_bytes_verified") and report[r].get("refusals_hold")
                                 for r in RESOURCES)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        report["service_log_tail"] = (work / "service.log").read_text()[-1500:]
        if not keep and report.get("route_ok"):
            shutil.rmtree(work, ignore_errors=True)
    out_path.write_text(json.dumps(report, indent=1, default=str))
    return report


# --- adoption --------------------------------------------------------------------------------------

def adopt(state_dir: Path, *, principals: list) -> dict:
    state_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = state_dir / "RECEIPT.json"
    if receipt_path.exists():
        raise SystemExit(f"REFUSED: {receipt_path} exists; a receipt is never written over")
    before_inventory = inventory()
    cfg = json.loads(RUNTIME_CONFIG.read_text())
    backup = state_dir / "5055.runtime.backup.json"
    shutil.copy2(RUNTIME_CONFIG, backup)
    after_cfg = successor_config(cfg, principals=principals)
    additive = config_is_additive(cfg, after_cfg)
    if not additive["additive"]:
        raise SystemExit(f"REFUSED: the change is not additive: {additive}")
    token = API_KEY_FILE.read_text().strip()
    receipt = {"schema": "df_public_lake_adoption.v1", "at": now_iso(), "host": os.uname().nodename,
               "service": SERVICE, "config": str(RUNTIME_CONFIG), "backup": str(backup),
               "backup_sha256": sha_file(backup), "before": before_inventory, "change": additive,
               "restore": f"cp {backup} {RUNTIME_CONFIG} && systemctl --user restart {SERVICE}",
               "declared": lake_entry()["declared"]}
    tmp = RUNTIME_CONFIG.with_suffix(".json.rp33.tmp")
    tmp.write_text(json.dumps(after_cfg, indent=1))
    os.replace(tmp, RUNTIME_CONFIG)
    restarted = run(["systemctl", "--user", "restart", SERVICE], timeout=180)
    receipt["restart"] = {"returncode": restarted.returncode, "stderr": restarted.stderr[-300:]}
    ok = False
    try:
        for _ in range(120):
            if http_json(f"{GOV_URL}/healthz")[0] == 200:
                break
            time.sleep(0.5)
        built = lake_entry()
        resource = "uci_235_individual_household_power/panel.parquet"
        receipt["post_check"] = route_checks(GOV_URL, token, cache_dir=HOME / ".cache/data-gov",
                                             run_id=f"rp33-adopt-{int(time.time())}", resource=resource,
                                             expect_sha=built["declared"][resource]["sha256"])
        receipt["after"] = inventory()
        ok = bool(receipt["post_check"].get("delivered_bytes_verified") and receipt["post_check"].get("refusals_hold")
                  and receipt["after"]["public_panel_lake_registered"]
                  and all(receipt["after"]["services"][u]["ActiveState"] == "active" for u in receipt["after"]["services"]))
    finally:
        receipt["adopted"] = ok
        if not ok:
            shutil.copy2(backup, RUNTIME_CONFIG)
            back = run(["systemctl", "--user", "restart", SERVICE], timeout=180)
            receipt["rolled_back"] = {"returncode": back.returncode, "config_sha256": sha_file(RUNTIME_CONFIG)}
        receipt_path.write_text(json.dumps(receipt, indent=1, default=str))
    return receipt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inventory", "contract", "rehearse", "adopt", "route"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--state-dir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--resource", default="uci_235_individual_household_power/panel.parquet")
    ap.add_argument("--principals", nargs="*", default=["predictor", "satoshi-gamma", "satoshi-dragon"])
    a = ap.parse_args(argv)
    if a.command == "inventory":
        doc = inventory()
    elif a.command == "contract":
        doc = lake_entry()
    elif a.command == "rehearse":
        doc = rehearse(a.out or Path("REHEARSAL.json"), keep=a.keep)
    elif a.command == "adopt":
        if not a.state_dir:
            raise SystemExit("--state-dir is required for adopt")
        doc = adopt(a.state_dir, principals=a.principals)
    else:
        built = lake_entry()
        doc = route_checks(GOV_URL, API_KEY_FILE.read_text().strip(), cache_dir=HOME / ".cache/data-gov",
                           run_id=f"rp33-route-{int(time.time())}", resource=a.resource,
                           expect_sha=built["declared"][a.resource]["sha256"])
    if a.out and a.command != "rehearse":
        a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(doc, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
