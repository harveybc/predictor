"""Bounded read of generated I6-DMO state: unit / stage / campaign / publication / fleet (DMO19).

Reads PLAN.json, the queue (SQLite read-only), progress files and the outbox. It launches nothing,
trains nothing and tails no log. Rules:
- unit progress is the child's own counter (epochs for fits) with its heartbeat age;
- a RUNNING attempt whose lease expired or whose heartbeat is older than STALE_AFTER is STALE;
- a fit's in-flight ETA is only an upper bound at the declared epoch ceiling (early stopping may end
  sooner); campaign ETAs are MEASURED only from completed production timings of the same phase
  (fit, score, publication separately), otherwise UNMEASURED;
- nothing is COMPLETE/CLOSED unless its terminals exist; stale or failed work is never complete;
- every unit shows its source-code identity (the pin its identity was computed with versus the
  campaign's successor pin), imported-fit reuse (ADOPT_ONLY with the original identity, pin and
  digests) and its decision (PAIR_NOT_ELIGIBLE, abstentions, eligible heads) as first-class fields;
- a dead timer, a crashing service, a failed systemd unit (optional bounded reading) or a stale
  attempt is a typed failure after a bounded threshold (``services.SILENT_AFTER``), never an
  indefinite WAITING.
"""
import json
import statistics
import time
from pathlib import Path

STALE_AFTER = 900.0
FIT_KINDS = ("fit", "final_refit")
#: Units whose completed fit time is a measured ETA basis for the remaining units of THEIR class
#: (a pretraining unit's time is never a fit's ETA, and the reverse).
ETA_FIT_KINDS = FIT_KINDS + ("pretrain",)


def _group_key(unit):
    """ETA/status class: the R2 regime (R2_PRETRAIN, R0, R2, ...), else the matched arm, else the family."""
    return unit.get("regime") or unit.get("arm") or unit["family"]


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _open(root):
    from tools.i6dmo.queue import Queue
    path = Path(root) / "state" / "queue.sqlite3"
    return Queue.open_readonly(path) if path.exists() else None


def _code_view(row, effective):
    identity = (row["spec"] or {}).get("identity") or {}
    unit_pin = identity.get("code_commit")
    campaign_pin = (effective or {}).get("code_commit")
    if unit_pin is None:
        relation = "IDENTITY_NOT_SEALED"
    elif unit_pin == campaign_pin:
        relation = "CAMPAIGN_PIN"
    elif (row["spec"] or {}).get("execution_policy") == "ADOPT_ONLY":
        relation = "ORIGINAL_PIN_IMPORTED"
    else:
        relation = "DIFFERS_UNEXPLAINED"
    return dict(unit_code_commit=unit_pin, campaign_code_commit=campaign_pin, relation=relation)


def _reuse_view(row, terminal, effective):
    adopted = terminal is not None and terminal.get("source") == "ADOPTED"
    if (row["spec"] or {}).get("execution_policy") != "ADOPT_ONLY" and not adopted:
        return None
    receipt = (effective or {}).get("reused_short_fit") or {}
    detail = (terminal or {}).get("source_detail") or {}
    return dict(policy=(row["spec"] or {}).get("execution_policy") or "ADOPTED_PRIOR_TERMINAL",
                terminal_source=None if terminal is None else terminal.get("source"),
                original_identity_sha256=row["identity_sha256"],
                original_code_commit=receipt.get("source_code_commit") or detail.get("source_code_commit"),
                reporting_code_commit=receipt.get("reporting_code_commit"),
                source_campaign=receipt.get("source_campaign") or detail.get("source_campaign")
                or detail.get("source_root_name"),
                source_terminal_sha256=receipt.get("source_terminal_sha256"),
                terminal_sha256=None if terminal is None else terminal.get("terminal_sha256"),
                training_repeated=receipt.get("training_repeated", False if receipt else None))


def _decision_view(row, terminal):
    doc = (terminal or {}).get("terminal") or {}
    gates = doc.get("gates") or {}
    if row["kind"] in ("pair", "strategy") and doc:
        return dict(decision=gates.get("pair_status") or doc.get("status"), status=doc.get("status"),
                    detail=doc.get("disposition"), strategy_status=gates.get("strategy_status"))
    if row["disposition"] == "PAIR_NOT_ELIGIBLE":
        return dict(decision="PAIR_NOT_ELIGIBLE", status="ABSTAINED", detail=row["cause"], strategy_status="ABSTAINED")
    if row["kind"] == "report" and doc.get("eligibility"):
        heads = doc["eligibility"]
        return dict(decision=doc.get("status"), eligible_heads=sorted(int(h) for h, r in heads.items()
                                                                      if r.get("eligible")),
                    heads=len(heads), detail=doc.get("disposition"))
    return None


def _unit_view(root, row, attempts, now, effective=None, terminal=None):
    view = dict(unit_id=row["unit_id"], kind=row["kind"], family=row["spec"].get("family"),
                stage=row["spec"].get("stage"), state=row["state"], cause=row["cause"], disposition=row["disposition"],
                identity_sha256=row["identity_sha256"], test_open=bool(row["spec"].get("test_open")),
                attempts=[dict(index=a["idx"], kind=a["kind"], outcome=a["outcome"], cause=a["cause"], role=a["role"],
                               device_class=a["device_class"], started_at=a["started_at"], ended_at=a["ended_at"],
                               cap_bytes=a["cap_bytes"], peak_bytes=a["peak_bytes"]) for a in attempts],
                progress=None, heartbeat_age_seconds=None, eta=dict(kind="UNMEASURED"),
                code=_code_view(row, effective), reuse=_reuse_view(row, terminal, effective),
                decision=_decision_view(row, terminal))
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


def _arm_views(plan, views, attempts_by_unit, now, key=None):
    """Per-arm status (matched contrast): state, count/denominator, live child, progress time,
    failure cause and a measured-or-UNKNOWN ETA from this arm's OWN completed fit only. With
    ``key`` (the R2 contrast: ``unit["regime"]``) the same view is computed per regime."""
    arms = {}
    for unit in plan["units"]:
        arm = (key or (lambda u: u.get("arm")))(unit)
        if not arm or unit["unit_id"] not in views:
            continue
        arms.setdefault(arm, []).append(views[unit["unit_id"]])
    out = {}
    for arm, members in sorted(arms.items()):
        states = [m["state"] for m in members]
        live = [dict(unit_id=m["unit_id"], role=a["role"], device_class=a["device_class"],
                     started_at=a["started_at"], heartbeat_age_seconds=m["heartbeat_age_seconds"],
                     progress=m["progress"], state=m["state"])
                for m in members for a in m["attempts"] if a["outcome"] == "RUNNING"]
        failures = [dict(unit_id=m["unit_id"], state=m["state"], cause=m["cause"]) for m in members
                    if m["state"] in ("FAILED", "BLOCKED", "STALE")]
        if all(s == "COMPLETE" for s in states):
            state, eta = "COMPLETE", dict(kind="DONE", seconds=0.0)
        else:
            state = ("FAILED" if any(s in ("FAILED", "BLOCKED") for s in states) else
                     "STALE" if "STALE" in states else "RUNNING" if "RUNNING" in states else "WAITING")
            running = [m for m in members if m["state"] == "RUNNING"]
            eta = running[0]["eta"] if running and running[0]["eta"]["kind"] != "UNMEASURED" else dict(
                kind="UNKNOWN", seconds=None,
                cause="no completed fit of this arm; another arm's or treatment's timing is never its ETA")
        progress_times = [m["progress"].get("updated_at") for m in members
                          if isinstance(m.get("progress"), dict) and m["progress"].get("updated_at")]
        out[arm] = dict(arm=arm, state=state, complete=states.count("COMPLETE"), denominator=len(members),
                        units={m["unit_id"]: m["state"] for m in members}, live_children=live,
                        last_progress_at=max(progress_times) if progress_times else None,
                        failure_causes=failures, eta=eta,
                        code_commits=sorted({m["code"]["unit_code_commit"] or "UNSEALED" for m in members}))
    return out


def compute(root, level="campaign", unit=None, now=None, systemd_probe=None, silent_after=None, arm=None):
    from tools.i6dmo import services
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
        terminals = {t["unit_id"]: t for t in q.terminals()} if q else {}
    finally:
        if q:
            q.close()
    effective = _read_json(root / "EFFECTIVE_CONFIG.json")
    by_unit = {}
    for a in attempts:
        by_unit.setdefault(a["unit_id"], []).append(a)
    views = {uid: _unit_view(root, row, by_unit.get(uid, []), now, effective, terminals.get(uid))
             for uid, row in rows.items()}
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
    # the ETA family key: a matched arm is its own class (its fit is never another arm's ETA)
    family_of = {u["unit_id"]: _group_key(u) for u in plan["units"]}
    fit_times = {}
    for a in completed:
        if kind_of.get(a["unit_id"]) in ETA_FIT_KINDS:
            fit_times.setdefault(family_of.get(a["unit_id"]), []).append(
                a["costs"].get("fit_seconds") or (a["ended_at"] - a["started_at"]))
    score_times = [a["costs"]["score_seconds"] for a in completed if a["costs"].get("score_seconds") is not None]

    def _open_unit(uid):
        return views[uid]["state"] not in ("COMPLETE", "FAILED", "BLOCKED")

    remaining_fit_families, all_fit_families = {}, set()
    for u in plan["units"]:
        if u["kind"] in ETA_FIT_KINDS:
            key = _group_key(u)
            all_fit_families.add(key)
            if _open_unit(u["unit_id"]):
                remaining_fit_families[key] = remaining_fit_families.get(key, 0) + 1
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
    closed = bool(close and close.get("state") == "CLOSED")
    service_views, failures = {}, []
    if not closed:
        service_views, service_failures = services.assess(
            root, now=now, silent_after=services.SILENT_AFTER if silent_after is None else silent_after,
            probe=systemd_probe)
        failures.extend(service_failures)
        failures.extend(dict(kind="UNIT_FAILED", unit=v["unit_id"], cause=v["cause"])
                        for u, v in views.items() if v["state"] == "FAILED")
        failures.extend(dict(kind="ATTEMPT_STALE", unit=v["unit_id"], cause=v["cause"],
                             heartbeat_age_seconds=v["heartbeat_age_seconds"])
                        for u, v in views.items() if v["state"] == "STALE")
        limit = services.SILENT_AFTER if silent_after is None else silent_after
        for v in views.values():
            # pre-fit infrastructure refusals wait (the science is not burned) but surface typed
            # once they persist beyond the bounded threshold
            for kind, match in (("INFRA_PREFIT_BLOCKED", lambda c: c.startswith("INFRA_PREFIT:")),
                                ("ADMISSION_BLOCKED", lambda c: c == "LAUNCHER_ADMISSION_REFUSED")):
                trailing = []
                for a in reversed(v["attempts"]):
                    if a["outcome"] == "INTERRUPTED" and match(str(a["cause"] or "")):
                        trailing.append(a)
                    else:
                        break
                if trailing and v["state"] in ("WAIT_RESOURCE", "INTERRUPTED", "QUEUED"):
                    first = min(a["ended_at"] or a["started_at"] for a in trailing)
                    if now - first > limit:
                        failures.append(dict(kind=kind, unit=v["unit_id"], cause=trailing[0]["cause"],
                                             refusals=len(trailing), age_seconds=now - first,
                                             threshold_seconds=limit))
    timers = _read_json(root / "TIMERS_STOPPED.json")
    if timers and timers.get("state") == "TIMER_STOP_FAILED":  # visible even after CLOSE; never reopens it
        failures.append(dict(kind="TIMER_STOP_FAILED", units=timers.get("failed"), trigger=timers.get("trigger"),
                             at=timers.get("at")))
    service_failed = any(f["kind"].startswith("SERVICE_") for f in failures)
    if closed:
        state = "CLOSED"
    elif hold is not None:
        state = "HELD"
    elif counts["FAILED"]:
        state = "FAILED"
    elif service_failed:
        state = "SERVICE_FAILED"
    elif any(f["kind"] == "INFRA_PREFIT_BLOCKED" for f in failures):
        state = "INFRASTRUCTURE_BLOCKED"
    elif any(f["kind"] == "ADMISSION_BLOCKED" for f in failures):
        state = "ADMISSION_BLOCKED"
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
    decisions = [dict(unit_id=v["unit_id"], **v["decision"]) for v in views.values()
                 if v["decision"] and v["decision"].get("decision") in ("PAIR_NOT_ELIGIBLE", "NOT_ELIGIBLE")]
    pins = {}
    for v in views.values():
        pins.setdefault(v["code"]["unit_code_commit"] or "UNSEALED", []).append(v["unit_id"])
    campaign = dict(schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], state=state, counts=counts,
                    failures=failures, services=service_views, decisions=decisions,
                    timers=None if not timers else dict(state=timers.get("state"), trigger=timers.get("trigger"),
                                                        units=[u.get("unit") for u in timers.get("units", [])],
                                                        failed=timers.get("failed"), at=timers.get("at")),
                    code_identity=dict(campaign_code_commit=(effective or {}).get("code_commit"),
                                       units_by_code_commit={k: sorted(v) for k, v in sorted(pins.items())}),
                    reuse=[dict(unit_id=v["unit_id"], **v["reuse"]) for v in views.values() if v["reuse"]],
                    close=None if not close else dict(
                        state=close.get("state"), disposition=close.get("disposition"),
                        trigger=close.get("trigger"), closure_inputs=close.get("closure_inputs")),
                    hold=None if hold is None else dict(
                        verdict=hold.get("verdict"), reason=hold.get("reason"),
                        publication_continues=bool(hold.get("publication_of_retained_cost_facts_allowed"))),
                    expected=plan["expected"], test_units_enumerated=plan["test_units_enumerated"],
                    test_readiness=None if readiness is None else readiness.get("state"), eta=eta, generated_at=now,
                    progress_note="stage progress counts declared terminals; no invented percentages")
    r2_mode = plan.get("mode") == "R2_CONTRAST"
    arms = _arm_views(plan, views, by_unit, now, key=(lambda u: u.get("regime")) if r2_mode else None)
    if arms:
        campaign["regimes" if r2_mode else "arms"] = {
            a: dict(state=v["state"], complete=v["complete"], denominator=v["denominator"], eta=v["eta"])
            for a, v in arms.items()}
    if r2_mode:
        comparison = views.get("comparison") or {}
        campaign["r2"] = dict(stage=next((s["stage"] for s in stages if s["terminal"] < s["expected"]), "CLOSED"),
                              decision=(comparison.get("disposition") if comparison.get("state") == "COMPLETE"
                                        else None),
                              pretraining=dict(complete=sum(1 for u in plan["units"] if u["kind"] == "pretrain"
                                                            and views[u["unit_id"]]["state"] == "COMPLETE"),
                                               denominator=sum(1 for u in plan["units"] if u["kind"] == "pretrain"),
                                               failed=[u["unit_id"] for u in plan["units"] if u["kind"] == "pretrain"
                                                       and views[u["unit_id"]]["state"] in ("FAILED", "BLOCKED")]),
                              probe_use="REPORT_ONLY_NEVER_A_GATE_NEVER_SELECTS",
                              fit_gpu_class=(effective or {}).get("fit_gpu_class"),
                              fit_gpu_class_holds=[dict(unit_id=v["unit_id"], cause=v["cause"])
                                                   for v in views.values()
                                                   if str(v["cause"] or "").startswith("FIT_GPU_CLASS_HOLD")])
    if level == "arm":
        if arm is not None and arm not in arms:
            return dict(schema="i6dmo.status.v1", state="UNKNOWN_ARM", arm=arm)
        chosen = {arm: arms[arm]} if arm is not None else arms
        return dict(schema="i6dmo.status.v1", campaign_id=plan["campaign_id"], campaign_state=state,
                    arms=chosen, generated_at=now)
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
