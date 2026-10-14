"""Bounded read of generated I6-DMO state: unit / stage / campaign / publication / fleet (DMO19).

Reads PLAN.json, the queue (SQLite read-only), progress files and the outbox. It launches nothing,
trains nothing and tails no log. Rules:
- unit progress is the child's own counter (epochs for fits) with its heartbeat age;
- a RUNNING attempt whose lease expired or whose heartbeat is older than STALE_AFTER is STALE;
- a fit's in-flight ETA is only an upper bound at the declared epoch ceiling (early stopping may end
  sooner); campaign ETAs are MEASURED only from completed production timings of the same phase
  (fit, score, publication separately), otherwise UNMEASURED;
- nothing is COMPLETE/CLOSED unless its terminals exist; stale or failed work is never complete.
"""
import json
import statistics
import time
from pathlib import Path

STALE_AFTER = 900.0
FIT_KINDS = ("fit", "final_refit")


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _open(root):
    from tools.i6dmo.queue import Queue
    path = Path(root) / "state" / "queue.sqlite3"
    return Queue.open_readonly(path) if path.exists() else None


def _unit_view(root, row, attempts, now):
    view = dict(unit_id=row["unit_id"], kind=row["kind"], family=row["spec"].get("family"),
                stage=row["spec"].get("stage"), state=row["state"], cause=row["cause"], disposition=row["disposition"],
                identity_sha256=row["identity_sha256"], test_open=bool(row["spec"].get("test_open")),
                attempts=[dict(index=a["idx"], kind=a["kind"], outcome=a["outcome"], cause=a["cause"], role=a["role"],
                               device_class=a["device_class"], started_at=a["started_at"], ended_at=a["ended_at"],
                               cap_bytes=a["cap_bytes"], peak_bytes=a["peak_bytes"]) for a in attempts],
                progress=None, heartbeat_age_seconds=None, eta=dict(kind="UNMEASURED"))
    running = [a for a in attempts if a["outcome"] == "RUNNING"]
    if not running:
        if row["state"] == "COMPLETE":
            view["eta"] = dict(kind="DONE", seconds=0.0)
        return view
    attempt = running[-1]
    folder = Path(root) / (attempt["attempt_dir"] or f"attempts/{row['unit_id']}/a{attempt['idx']:02d}")
    progress = _read_json(folder / "progress.json") or attempt["progress"]
    beat = attempt["renewed_at"]
    if isinstance(progress, dict) and isinstance(progress.get("updated_at"), (int, float)):
        beat = progress["updated_at"]
    view["progress"] = progress
    view["heartbeat_age_seconds"] = max(0.0, now - beat)
    if attempt["lease_expires_at"] <= now or view["heartbeat_age_seconds"] > STALE_AFTER:
        view["state"] = "STALE"
        view["cause"] = "LEASE_EXPIRED" if attempt["lease_expires_at"] <= now else "HEARTBEAT_OLD"
        return view
    done = (progress or {}).get("done")
    total = (progress or {}).get("total")
    elapsed = beat - attempt["started_at"]
    if isinstance(done, (int, float)) and done > 0 and isinstance(total, (int, float)) and elapsed > 0:
        rate = elapsed / done
        kind = "UPPER_BOUND_AT_DECLARED_CEILING" if row["kind"] in FIT_KINDS else "MEASURED_IN_FLIGHT_RATE"
        view["eta"] = dict(kind=kind, seconds=max(0.0, (total - done) * rate), seconds_per_unit=rate,
                           unit=(progress or {}).get("unit"), basis="this attempt's own progress counter")
    return view


def _eta(remaining, values):
    if remaining == 0:
        return dict(kind="DONE", seconds=0.0, basis_n=len(values))
    if not values:
        return dict(kind="UNMEASURED", seconds=None, basis_n=0, cause="no completed production timing of this phase")
    mean = statistics.fmean(values)
    return dict(kind="MEASURED", seconds=remaining * mean, per_item_seconds=mean, basis_n=len(values),
                remaining=remaining)


def _publication(root):
    path = Path(root) / "state" / "outbox.sqlite3"
    mirrors = _read_json(Path(root) / "MIRRORS.json")
    base = dict(mirrors=None if mirrors is None else dict(state=mirrors.get("state"),
                                                         targets={k: v.get("state") for k, v in
                                                                  mirrors.get("targets", {}).items()}))
    if not path.exists():
        return dict(base, state="NO_ITEMS", expected=0, delivered=0, pending=0, mismatch=0, delivery_seconds=[],
                    last_errors=[])
    from tools.i6dmo.outbox import Outbox
    box = Outbox.open_readonly(path)
    try:
        summary = box.summary()
    finally:
        box.close()
    if not summary["expected"]:
        state = "NO_ITEMS"
    elif summary["mismatch"]:
        state = "MISMATCH"
    elif summary["delivered"] == summary["expected"]:
        state = "DELIVERED"
    else:
        state = "PENDING"
    return dict(base, state=state, **summary)


def compute(root, level="campaign", unit=None, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    plan = _read_json(root / "PLAN.json")
    if plan is None:
        return dict(schema="i6dmo.status.v1", state="NOT_INITIALIZED")
    q = _open(root)
    try:
        rows = {u["unit_id"]: u for u in q.units()} if q else {}
        attempts = q.attempts() if q else []
        admissions = q.admissions(limit=20) if q else []
    finally:
        if q:
            q.close()
    by_unit = {}
    for a in attempts:
        by_unit.setdefault(a["unit_id"], []).append(a)
    views = {uid: _unit_view(root, row, by_unit.get(uid, []), now) for uid, row in rows.items()}
    if level == "unit":
        if unit not in views:
            return dict(schema="i6dmo.status.v1", state="UNKNOWN_UNIT", unit_id=unit)
        return dict(views[unit], schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], generated_at=now)
    publication = _publication(root)
    if level == "publication":
        return dict(publication, schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], generated_at=now)
    plan_ids = [u["unit_id"] for u in plan["units"]]
    states = [views[u]["state"] for u in plan_ids if u in views]
    counts = {s: states.count(s) for s in ("WAIT_ARTIFACT", "WAIT_OWNER_RULING", "WAIT_DESIGN_RULING",
                                           "WAIT_DEPENDENCY", "QUEUED",
                                           "WAIT_RESOURCE", "RUNNING", "STALE", "INTERRUPTED", "COMPLETE", "FAILED",
                                           "BLOCKED")}
    completed = [a for a in attempts if a["outcome"] == "COMPLETE" and a["costs"]]
    kind_of = {uid: row["kind"] for uid, row in rows.items()}
    family_of = {u["unit_id"]: u["family"] for u in plan["units"]}
    fit_times = {}
    for a in completed:
        if kind_of.get(a["unit_id"]) in FIT_KINDS:
            fit_times.setdefault(family_of.get(a["unit_id"]), []).append(
                a["costs"].get("fit_seconds") or (a["ended_at"] - a["started_at"]))
    score_times = [a["costs"]["score_seconds"] for a in completed if a["costs"].get("score_seconds") is not None]

    def _open_unit(uid):
        return views[uid]["state"] not in ("COMPLETE", "FAILED", "BLOCKED")

    remaining_fit_families, all_fit_families = {}, set()
    for u in plan["units"]:
        if u["kind"] in FIT_KINDS:
            all_fit_families.add(u["family"])
            if _open_unit(u["unit_id"]):
                remaining_fit_families[u["family"]] = remaining_fit_families.get(u["family"], 0) + 1
    families = {fam: _eta(remaining_fit_families.get(fam, 0), fit_times.get(fam, []))
                for fam in sorted(all_fit_families)}
    unmeasured = sorted(f for f in remaining_fit_families if families[f]["kind"] == "UNMEASURED")
    if not remaining_fit_families:
        fit_eta = dict(kind="DONE", seconds=0.0, families=families)
    elif unmeasured:
        fit_eta = dict(kind="UNMEASURED", seconds=None, families=families,
                       cause=("no completed fit of " + ", ".join(unmeasured)
                              + "; a measured fit of another family is never its ETA"))
    else:
        fit_eta = dict(kind="MEASURED", families=families,
                       seconds=sum(e["seconds"] for e in families.values() if e["seconds"]))
    remaining_scores = sum(1 for u in plan["units"] if _open_unit(u["unit_id"]))
    completed_costs = {a["unit_id"]: dict(fit_seconds=a["costs"].get("fit_seconds"),
                                          score_seconds=a["costs"].get("score_seconds"),
                                          wall_seconds=a["costs"].get("wall_seconds"),
                                          cgroup_peak_bytes=a["costs"].get("cgroup_peak_bytes"),
                                          device_class=a["device_class"]) for a in completed}
    eta = dict(fit=fit_eta, score=_eta(remaining_scores, score_times),
               measured_completed_costs=completed_costs,
               publication=(_eta(publication.get("pending", 0), publication.get("delivery_seconds", []))
                            if publication.get("expected") and not remaining_scores else
                            dict(kind="UNMEASURED", seconds=None, cause="publication population not yet complete",
                                 basis_n=len(publication.get("delivery_seconds", [])))),
               note="phases are separate; an in-flight first fit is not a completed-fit ETA, and a measured "
                    "short fit is never the long or end-to-end ETA")
    close = _read_json(root / "CLOSE.json")
    hold = _read_json(root / "AUDIT_HOLD.json")
    readiness = _read_json(root / "gates" / "TEST_READINESS.json")
    if close and close.get("state") == "CLOSED":
        state = "CLOSED"
    elif hold is not None:
        state = "HELD"
    elif counts["FAILED"]:
        state = "FAILED"
    elif counts["STALE"]:
        state = "STALE"
    elif counts["RUNNING"]:
        state = "RUNNING"
    elif counts["INTERRUPTED"] or counts["WAIT_RESOURCE"] or counts["QUEUED"]:
        state = "WAITING_FOR_WORKER"
    elif counts["WAIT_OWNER_RULING"]:
        state = "WAIT_OWNER_RULING"
    elif counts["WAIT_DESIGN_RULING"] and not (counts["WAIT_DEPENDENCY"] or counts["WAIT_ARTIFACT"]):
        state = "WAIT_DESIGN_RULING"
    elif counts["WAIT_ARTIFACT"]:
        state = "WAIT_ARTIFACT"
    elif counts["BLOCKED"]:
        state = "BLOCKED"
    else:
        state = "IN_PROGRESS"
    stages = []
    for stage in plan["stages"]:
        members = [u["unit_id"] for u in plan["units"] if u["stage"] == stage]
        terminal = [m for m in members if views[m]["state"] in ("COMPLETE", "FAILED", "BLOCKED")]
        stages.append(dict(stage=stage, expected=len(members), terminal=len(terminal),
                           complete=sum(1 for m in members if views[m]["state"] == "COMPLETE"),
                           failed=sum(1 for m in members if views[m]["state"] in ("FAILED", "BLOCKED")),
                           states={m: views[m]["state"] for m in members}))
    campaign = dict(schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], state=state, counts=counts,
                    hold=None if hold is None else dict(
                        verdict=hold.get("verdict"), reason=hold.get("reason"),
                        publication_continues=bool(hold.get("publication_of_retained_cost_facts_allowed"))),
                    expected=plan["expected"], test_units_enumerated=plan["test_units_enumerated"],
                    test_readiness=None if readiness is None else readiness.get("state"), eta=eta, generated_at=now,
                    progress_note="stage progress counts declared terminals; no invented percentages")
    if level == "campaign":
        return campaign
    if level == "stage":
        return dict(schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], stages=stages, generated_at=now)
    fleet = {}
    for view in views.values():
        for a in view["attempts"]:
            if a["outcome"] == "RUNNING":
                fleet.setdefault(a["role"], []).append(dict(unit_id=view["unit_id"], state=view["state"],
                                                            heartbeat_age_seconds=view["heartbeat_age_seconds"],
                                                            device_class=a["device_class"], cap_bytes=a["cap_bytes"]))
    fleet_doc = dict(running=fleet, recent_admissions=[dict(unit_id=a["unit_id"], role=a["role"], admit=a["admit"],
                                                            cause=a["cause"], cap_bytes=a["cap_bytes"], at=a["at"])
                                                       for a in admissions])
    if level == "fleet":
        return dict(fleet_doc, schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], generated_at=now)
    return dict(schema="i6dmo.status.v1", campaign=campaign, stages=stages, publication=publication, fleet=fleet_doc,
                units=[views[u] for u in plan_ids if u in views], generated_at=now)
