#!/usr/bin/env python3
"""Declared versus observed state for M06 STATUS.json `agents` and `lanes` (corrective order
2026-10-03 §5).  Pure functions only: no I/O, no clock; the writer gathers the facts and passes them in.

  declared_state   copied from the registry row (what was ordered / adopted).
  observed_state   derived ONLY from facts observed by probes:
                     RUNNING                        a live crispdm lease whose job id matches the lane
                     QUEUED_NOT_RUNNING             a launcher waiting for admission (not GPU work)
                     QUEUE_CLAIMS_ACTIVE_NO_PROCESS the lane's queue has running/verifying rows but no
                                                    live job was observed (a contradiction, named)
                     ACTIVE_COMMIT_AFTER_ASSIGNMENT a branch tip committed after the assignment time and
                                                    within COMMIT_FRESH_S of now
                     IDLE_QUEUE_DRAINED             the lane's queue has no runnable rows and no live job
                     IDLE_RESULT_PRESENT            a declared result commit/artifact exists, nothing live
                     STALE_DECLARATION              no current observation of any kind
                   `running` is never emitted from a declaration.
  last_observed_at, evidence_path, next_transition accompany every row.

Queue ETA uses completed train+verify walls of the SAME family only (family = label without its
last `_` token, e.g. neat_g0000_i0005 -> neat_g0000).  When no cell of a pending family has
completed, the ETA names the first event that makes it estimable.
"""
from __future__ import annotations

import datetime as dt

COMMIT_FRESH_S = 6 * 3600
RUNNABLE = ("queued", "running", "trained", "verifying", "completed")
ACTIVE_ROWS = ("running", "verifying", "trained")


def iso(t):
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s):
    if not s:
        return None
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def family(label):
    parts = (label or "").split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else (label or "")


def queue_summary(name, raw, now, parallel_hosts=1, planned_total=None, excluded_statuses=()):
    """raw = {"status_counts": {status: n}, "candidates": [[cid, label, seed, status]],
              "attempts": [[cid, attempt, kind, status, elapsed_s, peak_bytes]]}"""
    st = dict(raw.get("status_counts") or {})
    live = {k: v for k, v in st.items() if k not in excluded_statuses}
    labels = {c[0]: c[1] for c in raw.get("candidates", [])}
    seeds = {c[0]: c[2] for c in raw.get("candidates", [])}
    pending = [c[1] for c in raw.get("candidates", []) if c[3] in RUNNABLE]
    walls = {}
    for cid, _att, kind, status, el, _peak in raw.get("attempts", []):
        if status == "completed" and el:
            walls.setdefault(family(labels.get(cid)), {"train": [], "verify": []}).setdefault(kind, []).append(el)

    def cost(fam):
        d = walls.get(fam)
        if not d or not d.get("train"):
            return None
        v = d.get("verify") or []
        return sum(d["train"]) / len(d["train"]) + (sum(v) / len(v) if v else 0.0)

    hosts = max(1, int(parallel_hosts or 1))
    costs = [cost(family(lbl)) for lbl in pending]
    enq = sum(live.values())
    unenqueued = max(0, int(planned_total) - enq) if planned_total else 0
    if pending and all(c is not None for c in costs):
        sec = sum(costs) / hosts
        eta = {"earliest": iso(now + 0.8 * sec), "latest": iso(now + 1.5 * sec), "basis": "same_family_completed_cells",
               "families": {f: round(cost(f)) for f in sorted({family(l) for l in pending})},
               "covers": f"{len(pending)} enqueued pending cells", "first_estimable_event": None}
    elif pending:
        missing = sorted({family(l) for l, c in zip(pending, costs) if c is None})
        eta = {"earliest": None, "latest": None, "basis": "not_estimable",
               "first_estimable_event": f"first train+verify completion of a cell of family {missing[0]}",
               "missing_families": missing}
    else:
        eta = {"earliest": None, "latest": None, "basis": "not_estimable",
               "first_estimable_event": ("first enqueued cell of the next generation completes train+verify"
                                         if unenqueued else "nothing pending in this queue")}
    if unenqueued:
        eta["not_covered"] = (f"{unenqueued} planned cells not yet enqueued; their ETA becomes estimable at the "
                              "first train+verify completion of their own family")
    retries = {}
    for cid, att, kind, status, _el, _pk in raw.get("attempts", []):
        retries.setdefault(cid, []).append(f"{kind}-{att}:{status}")
    lineage = [{"label": labels.get(c), "seed": seeds.get(c), "cid": c[:8], "attempts": a}
               for c, a in retries.items() if len(a) > 2 or any(":interrupted" in x or ":failed" in x for x in a)]
    return {"queue": name, "status_counts": st, "enqueued": enq, "planned_total": planned_total,
            "done": live.get("verified", 0) + live.get("failed", 0),
            "active_rows": sum(live.get(k, 0) for k in ACTIVE_ROWS),
            "runnable_rows": sum(live.get(k, 0) for k in RUNNABLE), "unenqueued": unenqueued,
            "eta": eta, "retry_lineage": lineage}


def observe(row, jobs, queues, branch_tips, results_present, now):
    """row: registry lane/agent row with optional `observe`: {"jobs": regex, "queues": [names],
    "branches": [{"repo", "branch", "assigned_at"}], "results": [ids]}.
    jobs: [{"id", "state", "heartbeat_path", "heartbeat_status", "host_alias"}] (already matched by regex
    is NOT assumed: matching happens here).  queues: {name: queue_summary}.  branch_tips:
    {"repo:branch": (commit_unix, sha)}.  results_present: {id: evidence_path or None}."""
    import re
    spec = row.get("observe") or {}
    out = {"declared_state": row.get("declared_state", row.get("state")), "observed_state": "STALE_DECLARATION",
           "last_observed_at": None, "evidence_path": None,
           "next_transition": row.get("next_transition") or "first live process, lease, heartbeat or commit after assignment"}
    rx = spec.get("jobs")
    mine = [j for j in jobs if rx and re.fullmatch(rx, j.get("id") or "")]
    run = [j for j in mine if j.get("state") == "running"]
    waiting = [j for j in mine if j.get("state") == "queued"]
    qs = [queues[q] for q in spec.get("queues", []) if q in queues and "error" not in queues[q]]
    if run:
        j = run[0]
        out.update(observed_state="RUNNING", last_observed_at=iso(now),
                   evidence_path=j.get("heartbeat_path") or f"lease:{j.get('lease_id') or j['id']}",
                   observed_jobs=[{"id": x["id"], "host_alias": x.get("host_alias"), "phase": x.get("phase"),
                                   "heartbeat_status": x.get("heartbeat_status"), "heartbeat_at": x.get("heartbeat_at"),
                                   "progress": (x.get("last_advance") or {}).get("progress")} for x in run])
    elif waiting:
        out.update(observed_state="QUEUED_NOT_RUNNING", last_observed_at=iso(now),
                   evidence_path=f"launcher:{waiting[0]['id']}")
    elif any(q["active_rows"] for q in qs):
        out.update(observed_state="QUEUE_CLAIMS_ACTIVE_NO_PROCESS", last_observed_at=iso(now),
                   evidence_path="queue:" + ",".join(q["queue"] for q in qs))
    else:
        fresh = []
        for b in spec.get("branches", []):
            tip = branch_tips.get(f"{b['repo']}:{b['branch']}")
            at = parse_iso(b.get("assigned_at"))
            if tip and at is not None and tip[0] > at and now - tip[0] <= COMMIT_FRESH_S:
                fresh.append((tip[0], f"{b['repo']}@{b['branch']}:{tip[1][:10]}"))
        if fresh:
            t, ev = max(fresh)
            out.update(observed_state="ACTIVE_COMMIT_AFTER_ASSIGNMENT", last_observed_at=iso(t), evidence_path=ev)
        elif qs and all(q["runnable_rows"] == 0 and not q.get("unenqueued") for q in qs):
            out.update(observed_state="IDLE_QUEUE_DRAINED", last_observed_at=iso(now),
                       evidence_path="queue:" + ",".join(q["queue"] for q in qs))
        else:
            res = [(r, results_present.get(r)) for r in spec.get("results", []) if results_present.get(r)]
            if res:
                out.update(observed_state="IDLE_RESULT_PRESENT", last_observed_at=iso(now), evidence_path=res[0][1])
    if qs:
        q = qs[0]
        out["queue"] = {"name": q["queue"], "counts": q["status_counts"], "eta": q["eta"]}
        if out["observed_state"] == "RUNNING":
            e = q["eta"]
            out["next_transition"] = (f"{out['next_transition']} | queue ETA {e['earliest']}..{e['latest']}"
                                      if e.get("earliest") else
                                      f"{out['next_transition']} | ETA estimable after: {e.get('first_estimable_event')}")
    if out["observed_state"] == "STALE_DECLARATION":
        brs = ", ".join(f"{b['repo']}@{b['branch']}" for b in spec.get("branches", []))
        out["stale_reason"] = ("no live lease/launcher, no queue activity, no commit after assignment"
                               + (f" on {brs}" if brs else " (no branch declared)"))
    return out


def observe_agents(agents, lanes_out):
    """Agents inherit the observation of their lane; their own declared_state stays theirs."""
    by_lane = {L.get("lane"): L for L in lanes_out}
    out = []
    for a in agents:
        L = by_lane.get(a.get("lane"))
        obs = {k: (L or {}).get(k) for k in ("observed_state", "last_observed_at", "evidence_path", "next_transition")}
        if not L:
            obs.update(observed_state="STALE_DECLARATION", next_transition="lane not in registry")
        row = {k: v for k, v in a.items() if k != "state"}      # a declared `state` is never emitted as fact
        out.append({**row, "declared_state": a.get("declared_state", a.get("state")), **obs})
    return out


def lane_rows(lanes, jobs, queues, branch_tips, results_present, now):
    out = []
    for L in lanes:
        row = {k: v for k, v in L.items() if k != "state"}
        out.append({**row, **observe(L, jobs, queues, branch_tips, results_present, now)})
    return out
