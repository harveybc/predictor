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
#: the deployed warehouse host, used in the rehearsal against a DISPOSABLE DuckDB file
WAREHOUSE_PYTHON = HOME / ".venvs/store-hosts-duckdb-prod/bin/python"
WAREHOUSE_CONFIG = HOME / ".local/state/crispdm-duckdb/prod/5057.host.duckdb.json"

LAKE_ID = "public_panels"
RESOURCES = {
    "uci_235_individual_household_power/panel.parquet": "uci_235",
    "uci_321_electricityloaddiagrams20112014/panel.parquet": "uci_321",
}
#: RP41: the refusal of a date range must follow from the ARCHIVE's own semantics, not from a date
#: chosen afterwards. These panels are retrospective archives: nobody observed when any row became
#: available, so no point-in-time slice of them can be served. That is expressed by withholding the
#: archive from the first day it contains — 2006-12-16 for the household panel, the earlier of the
#: two — so every date-ranged request falls inside the withheld span and is denied, while the whole
#: resource remains deliverable AS_IS. The value is the data's own first day, not an arbitrary one.
ARCHIVE_FIRST_DAY = {"uci_235": "2006-12-16", "uci_321": "2011-01-01"}
HOLDOUT_START = min(ARCHIVE_FIRST_DAY.values())
HOLDOUT_REASON = ("retrospective archive: its rows' availability was never observed, so the archive is withheld from "
                  "its own first day and only whole-resource AS_IS delivery is possible")
#: the deployed external lake host (the same package the financial and synthetic lakes run)
LAKE_HOST_PYTHON = HOME / ".venvs/store-hosts/bin/python"
LAKE_HOST_PORT = 5059
LAKE_HOST_UNIT = "crispdm-data-lake-public-panels.service"


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


def lake_host_config(*, port: int, state_dir: Path, token_file: Path | None = None) -> dict:
    """The configuration of an EXTERNAL data-lake host serving the two panels — the same service and
    provider the financial and synthetic lakes run, not a files_lake embedded in data-gov."""
    built = lake_entry()
    return {"store_id": LAKE_ID, "title": "Public panels (UCI 235 / 321) — retrospective archive",
            "description": built["entry"]["description"], "kind": "lake", "engine": "files_inventory",
            "transport": "http", "web_host": "127.0.0.1", "web_port": int(port),
            "operator_config_path": str(Path(state_dir) / "public-panels.pending.json"),
            "backend": {"entry_point": "financial_files", "distribution": "financial-data-store",
                        # `untimed`: the lake may not parse these labels as availability evidence (they are
                        # local wall-clock strings whose zone the producer never stated). Every DATE-RANGED
                        # request is denied earlier, by the archive's holdout in the governance policy, so an
                        # untimed resource here is never a way around a range refusal.
                        "settings": {"root_path": str(PANEL_ROOT), "include_globs": sorted(RESOURCES),
                                     "untimed": sorted(RESOURCES), "holdout_start": HOLDOUT_START,
                                     "holdout_reason": HOLDOUT_REASON,
                                     "resource_contracts": built["entry"]["resource_contracts"]}}}


def lake_entry_http(port: int, token: str | None = None) -> dict:
    """How data-gov reaches that host: an http lake, exactly like the financial and synthetic ones."""
    entry = {"plugin": "http_lake", "lake_id": LAKE_ID, "title": "Public panels (UCI 235 / 321)",
             "description": lake_entry()["entry"]["description"], "kind": "lake", "engine": "files_inventory",
             "base_url": f"http://127.0.0.1:{int(port)}", "holdout_start": HOLDOUT_START}
    if token:
        entry["lake_service_token"] = token
    return entry


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
                 expect_sha: str, lake: str = LAKE_ID, outbox_dir: Path | None = None,
                 cube_url: str | None = None, cube_token: str | None = None) -> dict:
    """The WHOLE route for one resource, not a download probe (RP41).

    Register the campaign before any read, take the delivery, verify the bytes, CLOSE the unit with a
    terminal through the outbox, reconcile the campaign, and read the terminal back from the warehouse
    so the claim rests on content and not on a 200. A probe that then fails is closed too, with its
    own terminal, so no campaign is left open.
    """
    sys.path.insert(0, str(REPO / "tools"))
    import governed_run as GR

    key = f"{run_id}-{Path(resource).parent.name}-lake-route"
    gov = GR.GovHttp(gov_url, token, key)
    unit = "route-1"
    probe_unit = "route-probe-that-fails"
    code_identity = GR.strict_code_identity(REPO)
    config_sha = hashlib.sha256(json.dumps({"route": "rp41", "lake": lake, "resource": resource},
                                           sort_keys=True).encode()).hexdigest()
    campaign = {"schema": "governed_campaign.v1", "campaign_key": key, "classification": "NON_GOVERNING",
                "project": "predictor", "code_identity": code_identity, "config_sha256": config_sha,
                "input_mode": "DATASETS", "synthetic_spec_sha256": None, "units": [unit, probe_unit],
                "datasets": [{"lake": lake, "resource": resource, "role": "panel", "from": None, "to": None}],
                "terminal_lake": "olap_cube"}
    status, receipt = gov.submit_campaign(campaign)
    out = {"campaign_http": status, "campaign_sha256": receipt.get("campaign_sha256"), "campaign_key": key,
           "units_registered": [unit, probe_unit]}
    if status not in (200, 201):
        return {**out, "error": receipt, "route_complete": False}
    sha = receipt["campaign_sha256"]
    outbox = GR.TerminalOutbox(Path(os.path.expanduser(str(outbox_dir or GR.DEFAULT_OUTBOX))).resolve())
    deliveries = {}
    for unit_id in (unit, probe_unit):
        http, info = gov.governed_download(sha, unit_id, lake, resource, "panel", str(cache_dir))
        deliveries[unit_id] = info
    info = deliveries[unit]
    out["download"] = {"http": 200, "verification_state": info.get("verification_state"), "bytes": info.get("bytes"),
                       "sha256": info.get("sha256"), "cached": info.get("cached"),
                       "availability_use": info.get("availability_use"), "availability_label": info.get("availability_label"),
                       "availability_contract_sha256": info.get("availability_contract_sha256"),
                       "delivery_id": info.get("delivery_id"), "path": info.get("path")}
    out["cache_reuse"] = {"second_unit_cached": bool(deliveries[probe_unit].get("cached")),
                          "same_digest": deliveries[probe_unit].get("sha256") == info.get("sha256")}
    out["bytes_match_characterised"] = info.get("sha256") == expect_sha
    on_disk = sha_file(Path(info["path"]))
    out["bytes_on_disk_sha256"] = on_disk
    out["delivered_bytes_verified"] = on_disk == expect_sha
    out["availability_stays_undeclared"] = (info.get("availability_use") == "UNDECLARED"
                                            and info.get("availability_label") == "UNKNOWN")
    # --- close BOTH units: the one that worked and the probe that then fails -----------------------
    import df_utility_run as R
    terminals = {}
    for unit_id, state, reason in ((unit, "COMPLETED", None),
                                   (probe_unit, "FAILED", "route probe: deliberate failure after its delivery, "
                                                          "closed so the campaign has no open unit")):
        body = R._terminal(status=state, reason=reason, cost={"wall_seconds": 0.0, "cpu_seconds": 0.0},
                           metrics=[R._metric("lake.delivered_bytes", float(info.get("bytes") or 0), "bytes")]
                           if state == "COMPLETED" else [],
                           started=R.now_iso(), finished=R.now_iso(),
                           tags={"purpose": "PUBLIC_LAKE_ROUTE", "classification": "NON_GOVERNING", "phase": "DEVELOPMENT",
                                 "lake": lake, "resource": resource})
        body["deliveries"] = [deliveries[unit_id]["delivery_id"]]
        outbox.put({"campaign_sha256": sha, "unit_id": unit_id, "terminal": body})
        terminals[unit_id] = {"status": state, "terminal_sha256": None}
    flushed = GR._send_pending(gov, outbox)
    out["terminals"] = {"sent": flushed["sent"], "pending": flushed["pending"], "failures": flushed["failures"],
                        "units": terminals}
    rstatus, rbody = gov.reconcile_campaign(sha)
    out["reconciliation"] = {"http": rstatus, "missing_units": rbody.get("missing_units"),
                             "accounting_only": rbody.get("accounting_only"), "lake_only": rbody.get("lake_only")}
    out["campaign_closed"] = bool(rstatus == 200 and not rbody.get("missing_units")
                                  and not rbody.get("accounting_only") and not rbody.get("lake_only"))
    # --- the warehouse, by content -------------------------------------------------------------------
    if cube_url:
        query = {"sql": "SELECT unit_id, status FROM main.gov_terminal WHERE campaign_sha256 = ? ORDER BY unit_id LIMIT 10",
                 "params": [sha]}
        qstatus, qbody, _ = http_json(f"{cube_url}/api/v1/query", cube_token, method="POST", body=query)
        rows = (qbody or {}).get("rows") or []
        out["warehouse"] = {"http": qstatus, "rows": rows,
                            "both_units_present": {str(r[0]) for r in rows} >= {unit, probe_unit} if rows else False,
                            "probe_recorded_as_failed": any(str(r[0]) == probe_unit and str(r[1]) == "FAILED" for r in rows)}
    # --- the refusals ---------------------------------------------------------------------------------
    ranged_key = key + "-ranged"
    gov_r = GR.GovHttp(gov_url, token, ranged_key)
    ranged_campaign = {**campaign, "campaign_key": ranged_key, "units": ["ranged-1"],
                       "datasets": [{"lake": lake, "resource": resource, "role": "panel",
                                     "from": "2007-01-01", "to": "2007-01-02"}]}
    rc_status, rc = gov_r.submit_campaign(ranged_campaign)
    rng = http_json(f"{gov_url}/api/v2/download?lake={lake}&resource={resource}&role=panel&from=2007-01-01&to=2007-01-02",
                    token, headers={"X-Experiment-Key": ranged_key,
                                    "X-Campaign-SHA256": rc.get("campaign_sha256") or "", "X-Unit-ID": "ranged-1"})
    out["ranged_request"] = {"campaign_http": rc_status, "campaign_body": rc if rc_status >= 400 else None,
                             "download_http": rng[0], "body": rng[1],
                             "refused_at": "CAMPAIGN_REGISTRATION" if rc_status >= 400 else "DOWNLOAD",
                             "refused": rc_status >= 400 or rng[0] >= 400,
                             "semantics": HOLDOUT_REASON, "holdout_start": HOLDOUT_START}
    absent = http_json(f"{gov_url}/api/v2/download?lake={lake}&resource=uci_501_beijing_multisite_air_quality/panel.parquet&role=panel",
                       token, headers={"X-Experiment-Key": key, "X-Campaign-SHA256": sha, "X-Unit-ID": unit})
    out["undeclared_resource"] = {"http": absent[0], "body": absent[1]}
    nocamp = http_json(f"{gov_url}/api/v2/download?lake={lake}&resource={resource}&role=panel", token,
                       headers={"X-Experiment-Key": key})
    out["download_without_campaign"] = {"http": nocamp[0], "body": nocamp[1]}
    out["refusals_hold"] = (out["ranged_request"]["refused"] and out["undeclared_resource"]["http"] >= 400
                            and out["download_without_campaign"]["http"] >= 400)
    out["route_complete"] = bool(out["delivered_bytes_verified"] and out["refusals_hold"] and out["campaign_closed"]
                                 and out["availability_stays_undeclared"] and not out["terminals"]["pending"]
                                 and (out.get("warehouse", {}).get("both_units_present", True)))
    return out


# --- rehearsal on a disposable stack ----------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rehearse(out_path: Path, *, keep: bool = False, lake_port: int | None = None) -> dict:
    """The candidate change, rehearsed on DISPOSABLE copies of the three hosts it touches: the
    external lake host serving the panels, data-gov pointing at it, and a DuckDB warehouse. The route
    that runs here is the same function the adoption's post-check runs."""
    cfg = json.loads(RUNTIME_CONFIG.read_text())
    token = API_KEY_FILE.read_text().strip()
    work = Path(tempfile.mkdtemp(prefix="rp41-rehearsal-"))
    lake_token = hashlib.sha256((str(work) + "rehearsal").encode()).hexdigest()
    port, cube_port, lake_port = free_port(), free_port(), lake_port or free_port()
    host_cfg = lake_host_config(port=lake_port, state_dir=work)
    host_path = work / "public-panels.host.json"
    host_path.write_text(json.dumps(host_cfg, indent=1))
    cube_cfg = json.loads(WAREHOUSE_CONFIG.read_text())
    cube_cfg["web_port"] = cube_port
    cube_cfg["backend"]["settings"] = {**cube_cfg["backend"]["settings"], "duckdb_path": str(work / "cube.duckdb"),
                                       "min_free_bytes": 1 << 20}
    cube_cfg["operator_config_path"] = str(work / "cube.pending.json")
    cube_path = work / "cube.host.json"
    cube_path.write_text(json.dumps(cube_cfg, indent=1))
    stack = json.loads(json.dumps(cfg))
    stack["lakes"] = [l for l in stack["lakes"] if l.get("lake_id") in ("olap_cube",)] + [lake_entry_http(lake_port, lake_token)]
    for lake in stack["lakes"]:
        if lake.get("lake_id") == "olap_cube":
            lake["base_url"] = f"http://127.0.0.1:{cube_port}"
            lake["lake_service_token"] = lake_token
    stack["policies"] = [p for p in stack["policies"] if p.get("lake") in ("olap_cube",)] + policy_entries(
        ["predictor", "satoshi-gamma", "satoshi-dragon"])
    stack.update(web_port=port, accounting_db=str(work / "accounting.db"), spool_dir=str(work / "spool"),
                 cuts_dir=str(work / "cuts"), save_config=str(work / "effective.json"),
                 operator_config_path=str(work / "pending.json"))
    cfg_path = work / "stack.json"
    cfg_path.write_text(json.dumps(stack, indent=1))
    # the configuration that would actually be DEPLOYED (production ports), for the binding
    deployed = json.loads(json.dumps(cfg))
    deployed["lakes"] = [l for l in deployed["lakes"] if l.get("lake_id") != LAKE_ID] + [lake_entry_http(LAKE_HOST_PORT, None)]
    deployed["policies"] = [p for p in deployed["policies"] if p.get("lake") != LAKE_ID] + policy_entries(
        ["predictor", "satoshi-gamma", "satoshi-dragon"])
    report = {"schema": "df_public_lake_rehearsal.v2", "at": now_iso(), "work": str(work),
              "ports": {"data_gov": port, "lake_host": lake_port, "warehouse": cube_port},
              "stack": "disposable data-gov + disposable EXTERNAL lake host + disposable DuckDB warehouse",
              "external_host": "data_lake_service with the financial_files provider, the same package the deployed "
                               "financial and synthetic lakes run: no files_lake embedded in data-gov",
              "holdout": {"start": HOLDOUT_START, "reason": HOLDOUT_REASON}}
    logs = {}
    procs = {}
    try:
        for name, argv, cwd in (
                ("lake_host", [str(LAKE_HOST_PYTHON), "-m", "data_lake_service.main", "--load_config", str(host_path)], work),
                ("warehouse", [str(WAREHOUSE_PYTHON), "-m", "data_warehouse_service.main", "--load_config", str(cube_path)], work),
                ("data_gov", [str(PYTHON), "-m", "app.main", "--load_config", str(cfg_path)], GOV_APP)):
            logs[name] = open(work / f"{name}.log", "w")
            env = {**os.environ, "DATA_GOV_LAKE_TOKEN": lake_token}
            if name == "data_gov":
                env["PYTHONPATH"] = str(GOV_APP)
            procs[name] = subprocess.Popen(argv, cwd=str(cwd), stdout=logs[name], stderr=subprocess.STDOUT, env=env)
        for name, url in (("lake_host", f"http://127.0.0.1:{lake_port}"), ("warehouse", f"http://127.0.0.1:{cube_port}"),
                          ("data_gov", f"http://127.0.0.1:{port}")):
            if not _service_healthy(url, tries=120):
                report["startup_failed"] = {name: (work / f"{name}.log").read_text()[-2000:]}
                out_path.write_text(json.dumps(report, indent=1, default=str))
                raise SystemExit(f"REFUSED: the disposable {name} did not come up; see {out_path}")
        url = f"http://127.0.0.1:{port}"
        status, lakes, _ = http_json(f"{url}/api/v1/lakes", token)
        report["lakes"] = {"http": status, "raw": str(lakes)[:400]}
        built = lake_entry()
        for resource in sorted(RESOURCES):
            report[resource] = route_checks(url, token, cache_dir=work / "cache",
                                            run_id=f"rp41-rehearsal-{int(time.time())}", resource=resource,
                                            expect_sha=built["declared"][resource]["sha256"],
                                            outbox_dir=work / "outbox", cube_url=f"http://127.0.0.1:{cube_port}",
                                            cube_token=lake_token)
        report["route_ok"] = all(report[r].get("route_complete") for r in RESOURCES)
        report["binding"] = rehearsal_binding(host_path, deployed)
    finally:
        for name, proc in procs.items():
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        for name, handle in logs.items():
            handle.close()
            report[f"{name}_log_tail"] = (work / f"{name}.log").read_text()[-600:]
        out_path.write_text(json.dumps(report, indent=1, default=str))
        if not keep and report.get("route_ok"):
            shutil.rmtree(work, ignore_errors=True)
    return report


# --- adoption --------------------------------------------------------------------------------------

def _service_healthy(url: str, *, tries: int = 120) -> bool:
    for _ in range(tries):
        if http_json(f"{url}/healthz")[0] == 200:
            return True
        time.sleep(0.5)
    return False


def _restore(backup: Path, receipt: dict) -> None:
    """Put the previous configuration back and CHECK the service came back with it. A rollback that
    fails is recorded as a failed rollback, never as a silent success."""
    record = {"attempted_at": now_iso()}
    try:
        shutil.copy2(backup, RUNTIME_CONFIG)
        record["config_restored"] = sha_file(RUNTIME_CONFIG) == receipt.get("backup_sha256")
        back = run(["systemctl", "--user", "restart", SERVICE], timeout=180)
        record.update(restart_returncode=back.returncode, restart_stderr=back.stderr[-200:])
        record["service_healthy"] = _service_healthy(GOV_URL)
        state = service_state(SERVICE)
        record["service_state"] = {k: state.get(k) for k in ("ActiveState", "SubState", "NRestarts")}
        record["lake_absent_again"] = not any(l.get("lake_id") == LAKE_ID for l in
                                              json.loads(RUNTIME_CONFIG.read_text())["lakes"])
        record["rolled_back"] = bool(record["config_restored"] and record["service_healthy"]
                                     and record["lake_absent_again"] and back.returncode == 0)
    except BaseException as exc:                               # a failed rollback is the worst state: say so
        record.update(rolled_back=False, error=f"{type(exc).__name__}: {exc}"[:300])
    receipt["rollback"] = record


def adopt(state_dir: Path, *, principals: list, rehearsal: Path | None = None, lake_port: int = LAKE_HOST_PORT) -> dict:
    """Adopt the bounded resource. Every mutation is inside the recovery block, the receipt is written
    whatever happens, and the caller learns the outcome from `adopted` (the CLI exits non-zero)."""
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = state_dir / "RECEIPT.json"
    if receipt_path.exists():
        raise SystemExit(f"REFUSED: {receipt_path} exists; a receipt is never written over")
    before_inventory = inventory()
    cfg = json.loads(RUNTIME_CONFIG.read_text())
    backup = state_dir / "5055.runtime.backup.json"
    shutil.copy2(RUNTIME_CONFIG, backup)
    token = API_KEY_FILE.read_text().strip()
    lake_token = hashlib.sha256((str(state_dir) + "public-panels").encode()).hexdigest()
    host_cfg = lake_host_config(port=lake_port, state_dir=state_dir)
    host_path = state_dir / "public-panels.host.json"
    host_path.write_text(json.dumps(host_cfg, indent=1))
    after_cfg = json.loads(json.dumps(cfg))
    after_cfg["lakes"] = [l for l in after_cfg["lakes"] if l.get("lake_id") != LAKE_ID] + [lake_entry_http(lake_port, lake_token)]
    after_cfg["policies"] = [p for p in after_cfg["policies"] if p.get("lake") != LAKE_ID] + policy_entries(principals)
    additive = config_is_additive(cfg, after_cfg)
    receipt = {"schema": "df_public_lake_adoption.v2", "at": now_iso(), "host": os.uname().nodename,
               "service": SERVICE, "lake_host_unit": LAKE_HOST_UNIT, "config": str(RUNTIME_CONFIG),
               "backup": str(backup), "backup_sha256": sha_file(backup), "before": before_inventory,
               "change": additive, "lake_host_config": str(host_path), "lake_host_config_sha256": sha_file(host_path),
               "restore": f"cp {backup} {RUNTIME_CONFIG} && systemctl --user restart {SERVICE}",
               "declared": lake_entry()["declared"], "adopted": False,
               "binding": _rehearsal_binding(rehearsal, host_path, after_cfg) if rehearsal else
                          {"accepted": False, "why": "no rehearsal receipt was supplied"}}
    if not additive["additive"]:
        receipt["refused"] = f"the change is not additive: {additive}"
        receipt_path.write_text(json.dumps(receipt, indent=1, default=str))
        return receipt
    if not receipt["binding"].get("accepted"):
        receipt["refused"] = ("the rehearsal receipt does not bind to the bytes being adopted: "
                              f"{receipt['binding'].get('why')}")
        receipt_path.write_text(json.dumps(receipt, indent=1, default=str))
        return receipt
    ok = False
    try:
        # 1. the external lake host, started with its own environment and configuration
        unit_env = state_dir / "public-panels.service.env"
        unit_env.write_text(f"DATA_GOV_LAKE_TOKEN={lake_token}\n")
        unit_file = HOME / ".config/systemd/user" / LAKE_HOST_UNIT
        unit_file.write_text(f"""[Unit]
Description=Governed store service (public panels, {lake_port})
After=network.target

[Service]
Type=simple
WorkingDirectory={state_dir}
EnvironmentFile={unit_env}
ExecStart={LAKE_HOST_PYTHON} -m data_lake_service.main --load_config {host_path}
MemoryMax=2G
MemorySwapMax=0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
""")
        receipt["lake_host_unit_file"] = str(unit_file)
        run(["systemctl", "--user", "daemon-reload"], timeout=120)
        started = run(["systemctl", "--user", "start", LAKE_HOST_UNIT], timeout=180)
        receipt["lake_host_start"] = {"returncode": started.returncode, "stderr": started.stderr[-300:]}
        receipt["lake_host_healthy"] = _service_healthy(f"http://127.0.0.1:{lake_port}")
        if started.returncode != 0 or not receipt["lake_host_healthy"]:
            raise RuntimeError("the public-panels lake host did not come up")
        # 2. data-gov learns about it
        tmp = RUNTIME_CONFIG.with_suffix(".json.rp41.tmp")
        tmp.write_text(json.dumps(after_cfg, indent=1))
        os.replace(tmp, RUNTIME_CONFIG)
        receipt["config_written_sha256"] = sha_file(RUNTIME_CONFIG)
        restarted = run(["systemctl", "--user", "restart", SERVICE], timeout=180)
        receipt["restart"] = {"returncode": restarted.returncode, "stderr": restarted.stderr[-300:]}
        if restarted.returncode != 0:
            raise RuntimeError(f"the data-gov restart returned {restarted.returncode}")
        if not _service_healthy(GOV_URL):
            raise RuntimeError("data-gov did not answer /healthz after the restart")
        # 3. the acceptance: the whole route, on the service that is now running
        built = lake_entry()
        receipt["post_check"] = {}
        for resource in sorted(RESOURCES):
            receipt["post_check"][resource] = route_checks(
                GOV_URL, token, cache_dir=HOME / ".cache/data-gov", run_id=f"rp46-adopt-{int(time.time())}",
                resource=resource, expect_sha=built["declared"][resource]["sha256"],
                cube_url="http://127.0.0.1:5057", cube_token=_cube_token())
        receipt["after"] = inventory()
        ok = bool(all(v.get("route_complete") for v in receipt["post_check"].values())
                  and receipt["after"]["public_panel_lake_registered"]
                  and all(receipt["after"]["services"][u]["ActiveState"] == "active" for u in receipt["after"]["services"]))
        if not ok:
            raise RuntimeError("the post-check did not pass")
    except BaseException as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"[:400]
        ok = False
    finally:
        receipt["adopted"] = bool(ok)
        if not ok:
            try:
                run(["systemctl", "--user", "stop", LAKE_HOST_UNIT], timeout=120)
                unit = HOME / ".config/systemd/user" / LAKE_HOST_UNIT
                if unit.is_file():
                    unit.unlink()
                run(["systemctl", "--user", "daemon-reload"], timeout=120)
            except BaseException as exc:                       # noqa: BLE001
                receipt["lake_host_stop_error"] = f"{type(exc).__name__}: {exc}"[:200]
            _restore(backup, receipt)
        receipt_path.write_text(json.dumps(receipt, indent=1, default=str))
    return receipt


def _cube_token() -> str | None:
    env = HOME / ".local/state/crispdm-data-foundation/musashi-store-adoption-20260914T181826Z/5057.service.env"
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("DATA_GOV_LAKE_TOKEN="):
            return line.split("=", 1)[1].strip()
    return None


def _rehearsal_binding(rehearsal: Path, host_path: Path, after_cfg: dict) -> dict:
    """RP41: an adoption is allowed only by a rehearsal of THESE bytes — this lake-host configuration,
    this data-gov configuration, this provider package and this code identity."""
    try:
        report = json.loads(Path(rehearsal).read_text())
    except Exception as exc:                                    # noqa: BLE001
        return {"accepted": False, "why": f"the rehearsal receipt is unreadable: {exc}"}
    bound = report.get("binding") or {}
    now = rehearsal_binding(host_path, after_cfg)
    differences = [k for k, v in now.items() if bound.get(k) != v]
    return {"accepted": not differences and bool(report.get("route_ok")),
            "why": (f"the rehearsal differs in {differences}" if differences else
                    "the rehearsal did not pass" if not report.get("route_ok") else "bound"),
            "rehearsal": str(rehearsal), "expected": now, "rehearsed": bound,
            "route_ok": report.get("route_ok")}


def rehearsal_binding(host_path: Path, after_cfg: dict) -> dict:
    """The bytes an adoption and its rehearsal must share."""
    sys.path.insert(0, str(REPO / "tools"))
    import governed_run as GR
    provider = HOME / ".venvs/store-hosts/lib/python3.12/site-packages/financial_data_store/inventory.py"
    return {"lake_host_config_sha256": sha_file(Path(host_path)),
            "data_gov_config_sha256": hashlib.sha256(json.dumps(after_cfg, sort_keys=True).encode()).hexdigest(),
            "provider_sha256": sha_file(provider) if provider.is_file() else None,
            "code_identity": GR.strict_code_identity(REPO)["value"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["inventory", "contract", "rehearse", "adopt", "route"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--state-dir", type=Path, default=None)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--resource", default="uci_235_individual_household_power/panel.parquet")
    ap.add_argument("--principals", nargs="*", default=["predictor", "satoshi-gamma", "satoshi-dragon"])
    ap.add_argument("--rehearsal", type=Path, default=None, help="the rehearsal receipt that binds to these bytes")
    ap.add_argument("--lake-port", type=int, default=LAKE_HOST_PORT)
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
        doc = adopt(a.state_dir, principals=a.principals, rehearsal=a.rehearsal, lake_port=a.lake_port)
    else:
        built = lake_entry()
        doc = route_checks(GOV_URL, API_KEY_FILE.read_text().strip(), cache_dir=HOME / ".cache/data-gov",
                           run_id=f"rp33-route-{int(time.time())}", resource=a.resource,
                           expect_sha=built["declared"][a.resource]["sha256"])
    if a.out and a.command != "rehearse":
        a.out.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps(doc, indent=1, default=str)[:4000])
    if a.command == "adopt":
        return 0 if doc.get("adopted") else 2            # RP41: the CLI never reports success it did not have
    if a.command == "rehearse":
        return 0 if doc.get("route_ok") else 2
    if a.command == "route":
        return 0 if doc.get("route_complete") else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
