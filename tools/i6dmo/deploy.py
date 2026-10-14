"""Render and install the bounded systemd *user* units of ONE I6-DMO campaign (Satoshi runs it).

Units (all on the coordinator; children reach their host through the worker's executor):
  i6dmo-<cid>-tick.service/.timer            finite tick, CPU hidden, 1G
  i6dmo-<cid>-publish.service/.timer         outbox exact readback + mirrors; loads the existing
                                             warehouse EnvironmentFile (never read by this code)
  i6dmo-<cid>-<role>-worker.service/.timer   worker --once per role (KillMode=process: a supervisor
                                             restart never kills a child; it is re-attached)

Every file carries ``X-I6DMO-Campaign=<cid>``; installation refuses foreign files, non-campaign
names, a tampered effective config and a closed or retired campaign (no resurrection). Nothing in
these units opens TEST: TEST units exist only after ``tick`` records a READY readiness receipt.

PRIVATE_SPEC_DOC (never committed; JSON):
  {"schema": "i6dmo.private_deployment.v1",
   "roles": {"<role>": {"ssh_alias": "<alias>|null", "host_key": "<opaque>", "order": 0,
                        "python": "<venv python on that host>", "workdir": "<predictor checkout on that host>",
                        "run_root": "<scratch root on that host>", "data_root": "<bundle root on that host>",
                        "reserve_bytes": 2147483648, "cpu_units": false,
                        "gpus": [{"class": "rtx5090", "uuid": "<GPU uuid>", "heavy": true},
                                 {"class": "rtx5070ti", "uuid": "<GPU uuid>", "heavy": false,
                                  "parallel_slot": false}],
                        "launcher": {"crispdm_run": "<absolute path or null>"}}},
   "coordinator": {"python": "...", "workdir": "...", "unit_roles": ["preferred", "secondary", "coordinator"]},
   "warehouse": {"url": "<url>", "environment_file": "<existing EnvironmentFile>", "token_env": "WAREHOUSE_TOKEN"},
   "mirrors": {"local": "<path>", "second_host_role": "secondary", "second_host_path": "<path on that host>"},
   "launcher": {"crispdm_run": "<path or null>", "initial_cap_bytes": 8321499136, "wall_seconds": 86400,
                "cpu_cap_bytes": 4294967296},
   "tick_interval_seconds": 300}
"""
import os
import re
import subprocess
from pathlib import Path

from tools.i6n_swarm.common import atomic_json

from tools.i6dmo import campaign as C

MARK = "X-I6DMO-Campaign"
TEMPLATES = Path(__file__).with_name("systemd")
BOOTSTRAP_TEMPLATE = Path(__file__).with_name("bootstrap_private.sh.template")


def prefix(campaign_id):
    return f"{campaign_id}-" if campaign_id.startswith("i6dmo-") else f"i6dmo-{campaign_id}-"


def _fill(template, values):
    text = (TEMPLATES / template).read_text()
    for key, value in values.items():
        text = text.replace(f"@{key}@", str(value))
    if re.search(r"@[A-Z]+@", text):
        raise ValueError(f"unfilled template placeholder in {template}")
    return text


def render_units(root, roles=None):
    root = Path(root).resolve()
    C.load_effective(root)  # refuses a tampered seal or an open TEST
    plan = C.load_plan(root)
    spec = C.load_deployment(root)
    cid = plan["campaign_id"]
    p = prefix(cid)
    coord = spec.get("coordinator") or {}
    python = coord.get("python") or spec["roles"].get("coordinator", {}).get("python") or "python3"
    workdir = coord.get("workdir") or spec["roles"].get("coordinator", {}).get("workdir")
    roles = roles or coord.get("unit_roles") or sorted(spec["roles"])
    base = dict(CID=cid, MARK=f"{MARK}={cid}", PYTHON=python, WORKDIR=workdir, ROOT=root, PREFIX=p,
                INTERVAL=int(spec.get("tick_interval_seconds", 300)), ENVFILE=spec["warehouse"]["environment_file"])
    units = {f"{p}tick.service": _fill("i6dmo-tick.service.in", base),
             f"{p}tick.timer": _fill("i6dmo-tick.timer.in", base),
             f"{p}publish.service": _fill("i6dmo-publish.service.in", base),
             f"{p}publish.timer": _fill("i6dmo-publish.timer.in", base)}
    for role in roles:
        if role not in spec["roles"]:
            raise C.Refused(f"UNKNOWN_ROLE: {role}")
        values = dict(base, ROLE=role)
        units[f"{p}{role}-worker.service"] = _fill("i6dmo-worker.service.in", values)
        units[f"{p}{role}-worker.timer"] = _fill("i6dmo-worker.timer.in", values)
    return units


def default_systemctl(*args):
    out = subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, check=False)
    return dict(rc=out.returncode, stdout=out.stdout.strip())


def deploy(root, *, unit_dir=None, systemctl=None, start=True, roles=None, render_only=False):
    """start=False writes files and reloads only (nothing enabled); render_only calls no systemctl."""
    root = Path(root).resolve()
    if (root / "CLOSE.json").exists() or (root / "RETIRED").exists():
        raise C.Refused("CAMPAIGN_CLOSED_OR_RETIRED: refusing resurrection")
    rendered = render_units(root, roles)
    cid = C.load_plan(root)["campaign_id"]
    systemctl = systemctl or default_systemctl
    unit_dir = Path(unit_dir or Path.home() / ".config" / "systemd" / "user")
    unit_dir.mkdir(parents=True, exist_ok=True)
    for name in rendered:
        if not name.startswith(prefix(cid)):
            raise C.Refused(f"NON_CAMPAIGN_UNIT: {name}")
        path = unit_dir / name
        if path.exists() and f"{MARK}={cid}" not in path.read_text():
            raise C.Refused(f"FOREIGN_UNIT_FILE: {name}")
    written, unchanged = [], []
    for name, text in sorted(rendered.items()):
        path = unit_dir / name
        if path.exists() and path.read_text() == text:
            unchanged.append(name)
            continue
        tmp = path.with_name(f".{name}.tmp")
        tmp.write_text(text)
        os.replace(tmp, path)
        written.append(name)
    timers = sorted(n for n in rendered if n.endswith(".timer"))
    if render_only:  # pure rendering into unit_dir: no systemctl call of any kind
        return dict(schema="i6dmo.deploy.v1", campaign_id=cid, units=sorted(rendered), written=written,
                    unchanged=unchanged, timers=timers, render_only=True)
    if written:
        systemctl("daemon-reload")
    if start:
        systemctl("enable", "--now", *timers)
    observed = {n: dict(enabled=systemctl("is-enabled", n)["stdout"], active=systemctl("is-active", n)["stdout"])
                for n in sorted(rendered)}
    q = C.open_queue(root)
    try:
        children = [dict(unit_id=a["unit_id"], role=a["role"], device_class=a["device_class"],
                         launched=bool(a["launch"])) for a in q.running()]
    finally:
        q.close()
    doc = dict(schema="i6dmo.deploy.v1", campaign_id=cid, units=sorted(rendered), written=written,
               unchanged=unchanged, timers=timers, observed=observed, live_children=children,
               note="observed states are systemctl readings at deploy time; children start on the next worker tick")
    atomic_json(root / "DEPLOY.json", doc)
    return doc


def _inside_git(path):
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"], capture_output=True,
                         text=True, check=False)
    return out.returncode == 0 and out.stdout.strip() == "true"


def render_bootstrap(private_spec, root, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if _inside_git(out.parent):
        raise C.Refused("BOOTSTRAP_INSIDE_GIT: the rendered bootstrap must stay private")
    text = BOOTSTRAP_TEMPLATE.read_text().replace("@PRIVATE_SPEC@", str(Path(private_spec).resolve())).replace(
        "@ROOT@", str(Path(root).resolve()))
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(text)
    os.chmod(tmp, 0o700)
    os.replace(tmp, out)
    return dict(out=str(out), mode="0700")
