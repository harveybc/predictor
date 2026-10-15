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

from tools.i6n_swarm.common import atomic_json, file_sha256

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


def timer_cadence(spec):
    """Resolve bounded campaign-local cadence; omitted settings retain historical defaults."""
    raw = spec.get("timer_cadence") or {}
    bounds = dict(worker_seconds=(5, 600), cpu_worker_seconds=(5, 600),
                  tick_seconds=(5, 3600), publish_seconds=(30, 3600),
                  boot_seconds=(10, 600), accuracy_seconds=(1, 30))
    if not isinstance(raw, dict) or set(raw) - set(bounds):
        raise ValueError("TIMER_CADENCE: unknown settings")
    for key, value in raw.items():
        lo, hi = bounds[key]
        if type(value) is not int or not lo <= value <= hi:
            raise ValueError(f"TIMER_CADENCE: {key} requires integer {lo}..{hi}")
    boot = f"{raw['boot_seconds']}s" if "boot_seconds" in raw else None
    return dict(WORKER_INTERVAL=raw.get("worker_seconds", 120),
                CPU_WORKER_INTERVAL=raw.get("cpu_worker_seconds", 120),
                TICK_INTERVAL=raw.get("tick_seconds", int(spec.get("tick_interval_seconds", 300))),
                PUBLISH_INTERVAL=raw.get("publish_seconds", 600),
                WORKER_BOOT=boot or "3min", TICK_BOOT=boot or "2min", PUBLISH_BOOT=boot or "5min",
                TIMER_ACCURACY=(f"AccuracySec={raw['accuracy_seconds']}s" if "accuracy_seconds" in raw else ""))


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
    cadence = timer_cadence(spec)
    base = dict(CID=cid, MARK=f"{MARK}={cid}", PYTHON=python, WORKDIR=workdir, ROOT=root, PREFIX=p,
                **cadence, INTERVAL=cadence["TICK_INTERVAL"], ENVFILE=spec["warehouse"].get("environment_file") or "")
    units = {f"{p}tick.service": _fill("i6dmo-tick.service.in", base),
             f"{p}tick.timer": _fill("i6dmo-tick.timer.in", base)}
    if spec["warehouse"].get("environment_file"):  # else publication waits; the outbox keeps every fact
        units[f"{p}publish.service"] = _fill("i6dmo-publish.service.in", base)
        units[f"{p}publish.timer"] = _fill("i6dmo-publish.timer.in", base)
    for role in roles:
        if role not in spec["roles"]:
            raise C.Refused(f"UNKNOWN_ROLE: {role}")
        values = dict(base, ROLE=role)
        if not any(g.get("heavy") for g in spec["roles"][role].get("gpus", [])):
            values["WORKER_INTERVAL"] = cadence["CPU_WORKER_INTERVAL"]
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
    C._storage_gate(C.load_deployment(root), "deploy")
    C.ensure_execution_policy(root)  # idempotent; a typed wait is reported through readiness/worker
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
    waiting = [] if spec_has_envfile(root) else ["PUBLISH_WAITS_WAREHOUSE_ENVIRONMENT_FILE owner=Satoshi"]
    doc = dict(schema="i6dmo.deploy.v1", campaign_id=cid, units=sorted(rendered), written=written, waiting=waiting,
               unchanged=unchanged, timers=timers, observed=observed, live_children=children,
               note="observed states are systemctl readings at deploy time; children start on the next worker tick")
    atomic_json(root / "DEPLOY.json", doc)
    return doc


def spec_has_envfile(root):
    return bool(C.load_deployment(root)["warehouse"].get("environment_file"))


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


# ---------------------------------------------------------------- terminal: stop this campaign's units
TIMER_RECEIPT = "TIMERS_STOPPED.json"
TIMER_RECEIPT_SCHEMA = "i6dmo.timer_stop.v1"
STOP_DONE = ("STOPPED", "NOT_DEPLOYED")


def own_unit(cgroup_path="/proc/self/cgroup"):
    """The systemd unit this process runs in (its own oneshot service is never stopped from inside)."""
    try:
        text = Path(cgroup_path).read_text()
    except OSError:
        return None
    for part in reversed(text.replace("\n", "/").split("/")):
        if part.endswith(".service"):
            return part
    return None


def campaign_units(root):
    """Exactly the units THIS campaign's own deployment record lists (never a glob).

    A name must carry this campaign's prefix and the record must name this campaign; anything else
    is reported as REFUSED_NOT_THIS_CAMPAIGN and never touched.
    """
    import json
    root = Path(root)
    cid = C.load_plan(root)["campaign_id"]
    path = root / "DEPLOY.json"
    if not path.exists():
        return cid, None, []
    record = json.loads(path.read_text())
    if record.get("campaign_id") != cid:
        return cid, record, [dict(unit=None, action="REFUSED_NOT_THIS_CAMPAIGN",
                                  detail="DEPLOY.json names another campaign")]
    listed, refused = [], []
    for name in record.get("units") or []:
        if not isinstance(name, str) or not name.startswith(prefix(cid)) or "*" in name or "?" in name \
                or not name.endswith((".timer", ".service")):
            refused.append(dict(unit=name, action="REFUSED_NOT_THIS_CAMPAIGN"))
        else:
            listed.append(name)
    return cid, record, refused + [dict(unit=n) for n in listed]


def stop_campaign_units(root, *, trigger, systemctl=None, self_unit=None, clock=None):
    """Disable and stop this campaign's timers, stop its idle services; durable idempotent receipt.

    ``TIMERS_STOPPED.json`` (sibling of CLOSE.json) records every unit, the exact ``systemctl --user``
    call and its result, and the time. STOPPED / NOT_DEPLOYED receipts are final (a re-run is a no-op);
    a TIMER_STOP_FAILED receipt is retried by the next tick/publish and is visible in status. It
    never reopens or alters the scientific CLOSE.
    """
    import json
    import time
    root = Path(root)
    path = root / TIMER_RECEIPT
    if path.exists():
        previous = json.loads(path.read_text())
        if previous.get("state") in STOP_DONE:
            return previous
    import os
    clock = clock or time.time
    if systemctl is None and os.environ.get("PYTEST_CURRENT_TEST"):
        def systemctl(*args):  # tests must inject a runner: the real user systemd is never touched
            raise RuntimeError("SYSTEMCTL_NOT_INJECTED_UNDER_TEST")
    systemctl = systemctl or default_systemctl
    self_unit = own_unit() if self_unit is None else self_unit
    cid, record, entries = campaign_units(root)
    close = root / "CLOSE.json"
    results = []
    for entry in entries:
        name = entry["unit"]
        if entry.get("action"):
            results.append(entry)
            continue
        if name.endswith(".timer"):
            args = ("disable", "--now", name)
        elif name == self_unit:
            results.append(dict(unit=name, action="SELF_NOT_STOPPED",
                                detail="the invoking oneshot service exits by itself; its timer is disabled"))
            continue
        else:
            args = ("stop", name)
        try:
            out = systemctl(*args)
            rc, stdout = out.get("rc"), str(out.get("stdout", ""))[:200]
        except Exception as exc:  # an unavailable systemctl is a typed failure, never a crash
            rc, stdout = None, f"{type(exc).__name__}:{str(exc)[:160]}"
        results.append(dict(unit=name, action=" ".join(("systemctl", "--user") + args), rc=rc, stdout=stdout))
    failed = [r for r in results if r.get("action") == "REFUSED_NOT_THIS_CAMPAIGN"
              or (str(r.get("action", "")).startswith("systemctl") and r.get("rc") != 0)]
    state = "NOT_DEPLOYED" if record is None else ("TIMER_STOP_FAILED" if failed else "STOPPED")
    doc = dict(schema=TIMER_RECEIPT_SCHEMA, campaign_id=cid, state=state, trigger=trigger,
               close_sha256=file_sha256(close) if close.exists() else None,
               deploy_record_sha256=file_sha256(root / "DEPLOY.json") if record is not None else None,
               units=results, failed=[r.get("unit") for r in failed], at=clock(),
               note="only units listed in this campaign's own DEPLOY.json and carrying its id are touched")
    atomic_json(path, doc)
    return doc
