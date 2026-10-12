"""Read-only campaign status: swarm / horizon(DAG node) / candidate / fold / publication.

Reads the sealed plan, the ledger (SQLite read-only mode) and heartbeat files.
It imports no TensorFlow, launches nothing and writes nothing.

ETA rule: only durations of COMPLETE production attempts (device class is not
``cpu_fixture``) of the same stage feed an ETA. With none, the ETA kind is
``unknown``. A RUNNING fold with its evaluator's own epoch ETA is ``measured``.
"""
import json
from pathlib import Path
import statistics

STATES = ("QUEUED", "RUNNING", "COMPLETE", "WAIT_RESOURCE", "INTERRUPTED", "FAILED", "WAIT_DEPENDENCY")


def _quantile(values, q):
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    low, high = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def _eta(remaining, durations, parallel):
    if remaining <= 0:
        return dict(kind="measured", seconds=0.0, low=0.0, high=0.0, basis_n=len(durations))
    if not durations:
        return dict(kind="unknown", seconds=None, low=None, high=None, basis_n=0,
                    cause="no measured production fit of this stage")
    mean = statistics.fmean(durations)
    lanes = max(1, parallel)
    return dict(kind="projected", seconds=remaining * mean / lanes,
                low=remaining * _quantile(durations, 0.25) / lanes,
                high=remaining * _quantile(durations, 0.75) / lanes, basis_n=len(durations))


def compute(root, now=None):
    root = Path(root)
    from tools.i6n_swarm import plan as P
    payload = P.load(root)["payload"]
    ledger_path = root / "state" / "ledger.sqlite3"
    if not ledger_path.exists():
        empty = dict(state="QUEUED", cause="NOT_STARTED: run start", counts={}, eta=dict(kind="unknown"))
        return dict(schema="i6n.swarm.status.v1", campaign_id=payload["campaign_id"],
                    levels=dict(swarm=empty, horizon=[], candidate=[], fold=[], publication=dict(
                        state="QUEUED", pending=0, delivered=0, cause="NOT_STARTED")))
    from tools.i6n_swarm.ledger import Ledger
    led = Ledger.open_readonly(ledger_path)
    try:
        return _compute(root, payload, led, now)
    finally:
        led.close()


def _compute(root, payload, led, now):
    units = led.units()
    attempts = led.attempts()
    by_fit = {}
    for a in attempts:
        by_fit.setdefault(a["fit_id"], []).append(a)
    unit_by_fit = {u["fit_id"]: u for u in units}
    durations = {}
    for a in attempts:
        if a["outcome"] == "COMPLETE" and a["device_class"] != "cpu_fixture" and a["ended_at"] is not None:
            stage = unit_by_fit[a["fit_id"]]["stage"]
            durations.setdefault(stage, []).append(a["ended_at"] - a["started_at"])
    running_hosts = {a["host"] for a in attempts if a["outcome"] == "RUNNING"}
    folds = []
    for u in units:
        history = by_fit.get(u["fit_id"], [])
        current = next((a for a in reversed(history) if a["outcome"] == "RUNNING"), None)
        item = dict(fit_id=u["fit_id"], dag_node=u["dag_node"], stage=u["stage"], horizon=u["horizon"], fold=u["fold"],
                    genotype_canonical_sha256=u["genotype"], state=u["state"], cause=u["cause"],
                    attempts=len([a for a in history if a["physical"]]), lease=None, eta=None)
        if current is not None:
            item["lease"] = dict(owner=current["owner"], host_key=current["host"], boot_id=current["boot_id"],
                                 token=current["token"], expires_at=current["lease_expires_at"],
                                 renewed_at=current["renewed_at"], kind=current["kind"], index=current["index"])
            progress = current.get("progress") or {}
            if progress.get("eta_seconds") is not None:
                item["eta"] = dict(kind="measured", seconds=progress["eta_seconds"], source="evaluator_epochs")
            else:
                d = durations.get(u["stage"], [])
                elapsed = (now or current["renewed_at"]) - current["started_at"]
                item["eta"] = (dict(kind="projected", seconds=max(0.0, statistics.fmean(d) - elapsed), basis_n=len(d))
                               if d else dict(kind="unknown", seconds=None, cause="no measured production fit"))
        folds.append(item)
    plan_nodes = {n["node_id"]: n for n in payload["dag"]}
    horizon = []
    for node in led.dag_nodes():
        node_units = [u for u in units if u["dag_node"] == node["node_id"]]
        counts = dict(expected=0, complete=0, failed=0, interrupted=0, running=0, queued=0, wait_resource=0)
        for u in node_units:
            key = {"COMPLETE": "complete", "FAILED": "failed", "INTERRUPTED": "interrupted", "RUNNING": "running",
                   "WAIT_RESOURCE": "wait_resource"}.get(u["state"], "queued")
            counts[key] += 1
        stage = node["stage"]
        if stage == "search":
            expected = payload["search"]["new_candidates_per_horizon"] * len(payload["folds"])
            counts["expected"] = max(expected, len(node_units))
        elif stage == "annual" and node["state"] in ("WAIT_DEPENDENCY",) and not node_units:
            ready = sum(1 for i in plan_nodes[f"freeze_weeks:h{node['spec']['horizon']}"]["spec"]["items"]
                        if i["state"] == "READY")
            counts["expected"] = 2 * ready
        else:
            counts["expected"] = len(node_units)
        remaining = counts["expected"] - counts["complete"] - counts["failed"]
        eta = _eta(remaining, durations.get(stage, []), len(running_hosts))
        if node["state"] == "COMPLETE":
            eta = dict(kind="measured", seconds=0.0, low=0.0, high=0.0, basis_n=len(durations.get(stage, [])))
        entry = dict(node_id=node["node_id"], stage=stage, state=node["state"], cause=node["cause"],
                     depends_on=node["depends_on"], counts=counts, eta=eta,
                     activated_at=node["activated_at"], completed_at=node["completed_at"])
        if node["summary"] and node["summary"].get("finalists") is not None:
            entry["finalists"] = [{k: f.get(k) for k in ("role", "genotype_canonical_sha256", "fitness")}
                                  for f in node["summary"]["finalists"]]
        horizon.append(entry)
    candidates = {}
    for u in units:
        if u["stage"] not in ("search", "annual", "multi_output"):
            continue
        key = (u["dag_node"], u["search_candidate"])
        c = candidates.setdefault(key, dict(dag_node=u["dag_node"], search_candidate_sha256=u["search_candidate"],
                                            genotype_canonical_sha256=u["genotype"], folds_total=0,
                                            folds_complete=0, states=[]))
        c["folds_total"] += 1
        c["folds_complete"] += u["state"] == "COMPLETE"
        c["states"].append(u["state"])
    candidate_level = []
    for c in candidates.values():
        states = set(c.pop("states"))
        c["state"] = ("COMPLETE" if states == {"COMPLETE"} else "FAILED" if "FAILED" in states and
                      not states - {"COMPLETE", "FAILED"} else "RUNNING" if "RUNNING" in states else
                      "INTERRUPTED" if "INTERRUPTED" in states else "WAIT_RESOURCE" if "WAIT_RESOURCE" in states
                      else "QUEUED")
        candidate_level.append(c)
    outbox = led.outbox_all()
    pending = sum(1 for o in outbox if o["state"] == "PENDING")
    collect = root / "collect" / "COLLECT.json"
    collect_state = json.loads(collect.read_text()) if collect.exists() else {}
    if pending and collect_state.get("publication_state") == "WAIT_DEPENDENCY":
        publication = dict(state="WAIT_DEPENDENCY", cause=collect_state.get("cause"))
    elif pending:
        publication = dict(state="QUEUED", cause="awaiting collect")
    else:
        publication = dict(state="COMPLETE" if outbox else "QUEUED", cause=None if outbox else "nothing terminal yet")
    publication.update(pending=pending, delivered=len(outbox) - pending,
                       note="LOCAL_COMPLETE only; chain/OLAP states belong to lane D's publisher status")
    node_states = [n["state"] for n in horizon]
    if any(s == "RUNNING" for s in node_states):
        swarm_state = "RUNNING"
    elif node_states and all(s == "COMPLETE" for s in node_states):
        swarm_state = "COMPLETE"
    elif any(s == "QUEUED" for s in node_states):
        swarm_state = "QUEUED"
    elif any(s == "WAIT_RESOURCE" for s in node_states):
        swarm_state = "WAIT_RESOURCE"
    elif any(s == "WAIT_DEPENDENCY" for s in node_states):
        swarm_state = "WAIT_DEPENDENCY"
    else:
        swarm_state = "FAILED"
    totals = dict(expected=0, complete=0, failed=0, interrupted=0, running=0, queued=0, wait_resource=0)
    for n in horizon:
        for k in totals:
            totals[k] += n["counts"][k]
    blocked = [f"{n['node_id']}: {n['state']} ({n['cause']})" for n in horizon
               if n["state"] in ("WAIT_DEPENDENCY", "WAIT_RESOURCE", "FAILED") and n["cause"]]
    seconds = [n["eta"]["seconds"] for n in horizon if n["eta"]["kind"] != "unknown"]
    unknown = any(n["eta"]["kind"] == "unknown" and n["state"] not in ("COMPLETE", "FAILED") for n in horizon)
    swarm = dict(state=swarm_state, cause="; ".join(blocked[:6]) or None, counts=totals,
                 eta=dict(kind="unknown" if unknown else "projected", seconds=None if unknown else sum(seconds),
                          note="sum over DAG nodes; unknown while any open node lacks a measured production fit"),
                 seed=payload["seed"], direction=payload["direction"], e_plan_sha256=payload["e_plan_sha256"])
    tick = root / "status" / "TICK.json"
    heartbeat = json.loads(tick.read_text()) if tick.exists() else None
    return dict(schema="i6n.swarm.status.v1", campaign_id=payload["campaign_id"], mode=payload["mode"], now=now,
                tick=heartbeat, levels=dict(swarm=swarm, horizon=horizon, candidate=candidate_level, fold=folds,
                                            publication=publication))
