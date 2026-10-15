"""Service liveness receipts for one I6-DMO campaign (tick, publish, one worker per role).

Every timer-driven entry point (``tick``, ``publish``, ``worker --once``) records a small JSON
receipt under ``state/services/<service>.json``: when it started, when it finished, its result and,
on an exception, a typed cause. A supervising worker refreshes ``alive_at`` while its child runs.

``assess`` turns these receipts into typed failures for status, after a bounded threshold, so a
dead timer, a crashing service or a failed systemd unit is never shown as indefinite WAITING:

- ``SERVICE_ERROR``        the last run raised (immediate; the next successful run clears it);
- ``SERVICE_SILENT``       no run started/finished/refreshed for longer than the threshold;
- ``SERVICE_NEVER_RAN``    deployed (DEPLOY.json lists its timer) and no receipt after the threshold;
- ``SERVICE_UNIT_FAILED``  an optional bounded systemd reading reports the unit failed.

Nothing here polls, tails a log or starts anything. Receipts are mutable operational state (never
mirrored, never part of a closure proof).
"""
import json
import os
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from tools.i6n_swarm.common import atomic_json

SCHEMA = "i6dmo.service_receipt.v1"
#: Bounded threshold. Timers fire every 30-600 s (worker 120 s, tick <= 300 s, publish 600 s, or the
#: 30/60 s operational drop-ins), so 1800 s is at least three missed runs of the slowest timer.
SILENT_AFTER = 1800.0


def _dir(root):
    return Path(root) / "state" / "services"


def _path(root, service):
    return _dir(root) / f"{service}.json"


def read(root, service):
    try:
        return json.loads(_path(root, service).read_text())
    except (OSError, ValueError):
        return None


def _write(root, service, doc):
    _dir(root).mkdir(parents=True, exist_ok=True)
    atomic_json(_path(root, service), doc)


@contextmanager
def run(root, service, clock=time.time):
    """Record start/finish/result of one finite service run; an exception is a typed ERROR receipt."""
    previous = read(root, service) or {}
    doc = dict(schema=SCHEMA, service=service, pid=os.getpid(), started_at=clock(), finished_at=None,
               alive_at=clock(), result="RUNNING", cause=None,
               last_ok_at=previous.get("last_ok_at"), consecutive_errors=previous.get("consecutive_errors", 0))
    _write(root, service, doc)
    outcome = {}
    try:
        yield outcome
    except BaseException as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
        where = f"{Path(frame.filename).name}:{frame.lineno}" if frame else "?"
        doc.update(finished_at=clock(), alive_at=clock(), result="ERROR",
                   cause=f"{type(exc).__name__}:{str(exc)[:160]}@{where}",
                   consecutive_errors=doc["consecutive_errors"] + 1)
        _write(root, service, doc)
        raise
    doc.update(finished_at=clock(), alive_at=clock(), result=str(outcome.get("result", "OK"))[:80],
               cause=None, last_ok_at=clock(), consecutive_errors=0)
    _write(root, service, doc)


def alive(root, service, clock=time.time):
    """Refresh the liveness instant of a running service (a worker supervising its child)."""
    doc = read(root, service)
    if doc is None or doc.get("finished_at") is not None:
        return
    doc["alive_at"] = clock()
    _write(root, service, doc)


def expected_services(root):
    """Services the deployment installed (from DEPLOY.json timers) plus every service that ever ran."""
    expected = {}
    deploy = None
    try:
        deploy = json.loads((Path(root) / "DEPLOY.json").read_text())
        deployed_at = (Path(root) / "DEPLOY.json").stat().st_mtime
    except (OSError, ValueError):
        deployed_at = None
    if deploy:
        prefix = _prefix(deploy.get("campaign_id", ""))
        for timer in deploy.get("timers") or []:
            name = timer[len(prefix):] if timer.startswith(prefix) else timer
            name = name[:-len(".timer")] if name.endswith(".timer") else name
            if name.endswith("-worker"):
                name = "worker-" + name[:-len("-worker")]
            expected[name] = dict(unit=timer[:-len(".timer")] + ".service", deployed_at=deployed_at)
    if _dir(root).is_dir():
        for path in sorted(_dir(root).glob("*.json")):
            expected.setdefault(path.stem, dict(unit=None, deployed_at=deployed_at))
    return expected


def _prefix(campaign_id):
    return f"{campaign_id}-" if campaign_id.startswith("i6dmo-") else f"i6dmo-{campaign_id}-"


def systemd_probe(unit_names, run=subprocess.run):
    """One bounded reading of the campaign's user units (no polling)."""
    if not unit_names:
        return {}
    out = run(["systemctl", "--user", "show", *unit_names, "-p", "Id", "-p", "ActiveState", "-p", "Result"],
              capture_output=True, text=True, timeout=30, check=False)
    states, current = {}, {}
    for line in (out.stdout or "").splitlines() + [""]:
        if not line.strip():
            if current.get("Id"):
                states[current["Id"]] = dict(active_state=current.get("ActiveState"), result=current.get("Result"))
            current = {}
            continue
        key, _, value = line.partition("=")
        current[key] = value
    return states


def assess(root, *, now=None, silent_after=SILENT_AFTER, probe=None):
    """Per-service view and the typed failures (empty when every expected service is healthy)."""
    now = time.time() if now is None else now
    views, failures = {}, []
    expected = expected_services(root)
    units = {meta["unit"]: name for name, meta in expected.items() if meta.get("unit")}
    systemd = {}
    if probe is not None and units:
        try:
            systemd = probe(sorted(units)) or {}
        except Exception as exc:  # an unreadable systemd is reported, never mistaken for health
            failures.append(dict(kind="SERVICE_PROBE_UNAVAILABLE", service=None,
                                 cause=f"{type(exc).__name__}:{str(exc)[:120]}"))
    for name, meta in sorted(expected.items()):
        doc = read(root, name)
        last = None
        if doc is not None:
            last = max(v for v in (doc.get("alive_at"), doc.get("finished_at"), doc.get("started_at"))
                       if isinstance(v, (int, float)))
        view = dict(service=name, unit=meta.get("unit"), result=None if doc is None else doc.get("result"),
                    cause=None if doc is None else doc.get("cause"),
                    last_seen_age_seconds=None if last is None else max(0.0, now - last),
                    running=bool(doc and doc.get("finished_at") is None),
                    consecutive_errors=0 if doc is None else doc.get("consecutive_errors", 0),
                    systemd=systemd.get(meta.get("unit")) if meta.get("unit") else None)
        views[name] = view
        if doc is not None and doc.get("result") == "ERROR":
            failures.append(dict(kind="SERVICE_ERROR", service=name, cause=doc.get("cause"),
                                 consecutive_errors=view["consecutive_errors"]))
        if doc is None:
            deployed = meta.get("deployed_at")
            if deployed is not None and now - deployed > silent_after:
                failures.append(dict(kind="SERVICE_NEVER_RAN", service=name, threshold_seconds=silent_after,
                                     age_seconds=now - deployed))
        elif view["last_seen_age_seconds"] > silent_after:
            failures.append(dict(kind="SERVICE_SILENT", service=name, threshold_seconds=silent_after,
                                 age_seconds=view["last_seen_age_seconds"]))
        state = view["systemd"] or {}
        if state.get("active_state") == "failed" or (state.get("result") not in (None, "", "success")):
            failures.append(dict(kind="SERVICE_UNIT_FAILED", service=name, unit=meta.get("unit"),
                                 cause=f"ActiveState={state.get('active_state')} Result={state.get('result')}"))
    return views, failures
