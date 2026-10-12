"""``collect``: publish terminal candidate dispositions and write result tables. Never trains.

Per candidate (dag node x search candidate) whose fold units are all terminal,
one ``i6n.archive.candidate.v1`` document is handed to lane D's
``ArchivePublisher.record_candidate`` with A's fold receipts unchanged. The
outbox rows are marked delivered only after the publisher returns. Without a
publisher the outbox stays pending (state WAIT_DEPENDENCY), nothing is lost.
"""
import csv
import io
import json
from pathlib import Path
import time

from tools.i6n_swarm import common, ledger as L, plan as P

COLUMNS = ("stage", "dag_node", "horizon", "search_candidate_sha256", "genotype_canonical_sha256", "fold", "disposition",
           "split", "population", "n", "error_sum", "naive_sum", "mae", "naive_mae", "skill_mae", "fitness")


def default_publisher(root, payload):
    try:
        from tools.i6n_archive.publisher import ArchivePublisher
    except ImportError:
        return None, "tools.i6n_archive.publisher (lane D) is not installed in this checkout"
    return ArchivePublisher(root=str(Path(root) / "archive"), store_id=f"i6n-swarm-{payload['campaign_id']}",
                            domain_id=f"i6n-swarm-{payload['campaign_id']}", producer_peer_id="coordinator",
                            mode="external_payload"), None


def _candidate_doc(payload, key_units, terminals, attempts_by_fit, fitness):
    first = key_units[0]
    spec = first["spec"]
    complete = all(u["state"] == "COMPLETE" for u in key_units)
    failures = [f"{u['fold']}: {u['cause']}" for u in key_units if u["state"] != "COMPLETE"]
    attempts, costs = [], dict(elapsed_seconds=0.0, fit_seconds=None, peak_rss_bytes=None)
    for u in key_units:
        for a in attempts_by_fit.get(u["fit_id"], []):
            if not a["physical"]:
                continue
            attempts.append(dict(fold=u["fold"], index=a["index"], state=a["outcome"],
                                 parent_status_sha256=a["parent_status_sha256"], fencing_token=str(a["token"]),
                                 reason=a["cause"]))
            if a["ended_at"] is not None:
                costs["elapsed_seconds"] += a["ended_at"] - a["started_at"]
        term = terminals.get(u["fit_id"])
        cost = (term or {}).get("receipt", {}).get("candidate_receipt", {}).get("cost") or {}
        if cost.get("process_peak_rss_bytes"):
            costs["peak_rss_bytes"] = max(costs["peak_rss_bytes"] or 0, cost["process_peak_rss_bytes"])
        if cost.get("fit_seconds") is not None:
            costs["fit_seconds"] = (costs["fit_seconds"] or 0.0) + cost["fit_seconds"]
    adopted = complete and all((terminals.get(u["fit_id"]) or {}).get("source") == "PRIOR" for u in key_units)
    genotype = spec.get("genotype", {})
    return dict(
        schema="i6n.archive.candidate.v1", experiment=payload["experiment"],
        search_candidate_sha256=first["search_candidate"], horizon_hours=first["horizon"],
        plan_sha256=payload["e_plan_sha256"],
        plan_metadata=dict(stage=first["stage"], dag_node=first["dag_node"], size_class=spec.get("size_class"),
                           campaign_id=payload["campaign_id"], generations=payload["search"]["generations"]),
        genotype=dict(active=genotype, inactive={}), genotype_canonical_sha256=first["genotype"],
        effective_config=spec.get("effective_config", {}), effective_config_sha256=first["effective_config_sha256"],
        disposition="ADOPTED_CACHE" if adopted else ("COMPLETE" if complete else "FAILED"),
        disposition_reason=None if complete else "; ".join(failures),
        fitness=fitness if complete else None, seed=common.SEED,
        lineage=dict(generation=spec.get("generation", 0), population_index=spec.get("population_index"),
                     operator=spec.get("operator", "UNRECORDED_AT_BRIDGE"), parents=spec.get("parents", [])),
        attempts=attempts, costs=costs)


OUTCOME = {"BEST": "WINNER", "MINIMAL_CONTROL": "CONTROL", "SMALL_ALTERNATIVE": "FINALIST"}


def _record_decisions(led, payload, publisher):
    """One append-only decision event per closed search (WINNER/FINALIST/CONTROL/NOT_SELECTED).

    The sequence and the previous body digest are kept as idempotent ledger DAG
    events, so a re-run never records the same decision twice and a changed
    basis supersedes the previous decision with sequence + 1.
    """
    recorded = 0
    for node in led.dag_nodes():
        summary = node["summary"] or {}
        if node["stage"] != "search" or node["state"] != "COMPLETE" or not summary.get("finalists"):
            continue
        roles = {f["search_candidate_sha256"]: OUTCOME.get(f["role"], "FINALIST") for f in summary["finalists"]}
        outcomes = [dict(search_candidate_sha256=c["search_candidate_sha256"],
                         outcome=roles.get(c["search_candidate_sha256"], "NOT_SELECTED"))
                    for c in sorted(summary.get("candidates", []), key=lambda c: c["search_candidate_sha256"])]
        basis = dict(rule=summary.get("finalist_rule"), folds=payload["folds"], stop_reason=summary.get("stop_reason"),
                     fitness={c["search_candidate_sha256"]: c.get("fitness") for c in summary.get("candidates", [])})
        basis_sha = common.sha256_json(dict(basis=basis, outcomes=outcomes))
        previous = [e for e in led.dag_events(node["node_id"]) if e["event"].startswith("DECISION_")]
        if previous and previous[-1]["detail"]["basis_sha256"] == basis_sha:
            continue
        sequence = len(previous) + 1
        decision = dict(schema="i6n.archive.decision.v1", experiment=payload["experiment"],
                        plan_sha256=payload["e_plan_sha256"], horizon_hours=node["spec"]["horizon"], sequence=sequence,
                        supersedes_decision_sha256=previous[-1]["detail"]["item_id"] if previous else None,
                        basis=basis, outcomes=outcomes)
        result = publisher.record_decision(decision)
        led.dag_event(node["node_id"], f"DECISION_{sequence:04d}",
                      dict(basis_sha256=basis_sha, item_id=result.get("item_id"), sequence=sequence))
        recorded += 1
    return recorded


def annual_rollups(led, payload, terminals):
    """Annual MAE per (horizon, finalist) through lane C's AnnualAccumulator; refusals are reported, not hidden.

    Expected weekly support comes from C's plan (VALIDATION n and origins digest per week);
    rows are A's candidate_receipt rows (split VALIDATION, endpoint head) passed unchanged.
    """
    pin = payload["inputs"].get("weekly_plan")
    try:
        from tools.i6n_weekly_rollup import AnnualAccumulator, RollupError
    except ImportError:
        return dict(state="WAIT_DEPENDENCY", cause="tools.i6n_weekly_rollup (lane C) is not installed")
    if not pin:
        return dict(state="WAIT_DEPENDENCY", cause="C weekly plan not pinned: expected weekly support unknown")
    c_plan = common.read_pinned(pin, "C weekly plan")
    out = {}
    for unit_group in _annual_groups(led):
        h, genotype, units = unit_group
        expected = {}
        for u in c_plan["units"]:
            if u.get("kind") != "WEEKLY_REFIT":
                continue
            split = (u.get("support") or {}).get(c_plan["fit_role"], {}).get(str(h), {}).get("VALIDATION")
            if split is not None:
                expected[u["unit_id"]] = dict(n=split["n"], origins_sha256=split["origins_sha256"])
        key = f"h{h}:{genotype}"
        rows = [r for u in units if u["fit_id"] in terminals
                for r in terminals[u["fit_id"]]["receipt"]["candidate_receipt"].get("rows", [])
                if r.get("split") == "VALIDATION" and r.get("head_offset_hours") == h]
        try:
            identity = rows[0]["candidate_identity_sha256"] if rows else genotype
            acc = AnnualAccumulator(experiment=payload["experiment"], candidate_identity_sha256=identity,
                                    horizon_hours=h, role=c_plan["fit_role"], expected=expected)
            for row in rows:
                acc.add(row)
            out[key] = acc.summary()
        except RollupError as exc:
            out[key] = dict(state="ROLLUP_REFUSED", code=getattr(exc, "code", None), detail=str(exc))
    return dict(state="COMPUTED", rollups=out)


def _annual_groups(led):
    groups = {}
    for u in led.units(stage="annual"):
        groups.setdefault((u["horizon"], u["genotype"]), []).append(u)
    return [(h, g, units) for (h, g), units in sorted(groups.items())]


def collect(root, publisher="auto", now=None, score_folds=None):
    root = Path(root)
    payload = P.load(root)["payload"]
    led = L.Ledger(root / "state" / "ledger.sqlite3", clock=(lambda: now) if now else time.time)
    cause = None
    if publisher == "auto":
        publisher, cause = default_publisher(root, payload)
    elif publisher is None:
        cause = "no publisher configured"
    from tools.i6n_swarm.bridge import resolve_score_folds
    score_folds = score_folds or resolve_score_folds()
    e_plan = P.e_plan(root)
    terminals = led.terminals()
    attempts_by_fit = {}
    for a in led.attempts():
        attempts_by_fit.setdefault(a["fit_id"], []).append(a)
    units = [u for u in led.units() if u["stage"] != "freeze"]
    groups = {}
    for u in units:
        groups.setdefault((u["dag_node"], u["search_candidate"]), []).append(u)
    pending = led.outbox_pending()
    pending_by_key = {}
    for item in pending:
        pending_by_key.setdefault((item["payload"]["dag_node"], item["payload"]["search_candidate_sha256"]), []).append(item)
    rows, published = [], 0
    for key, key_units in sorted(groups.items()):
        terminal = all(u["state"] in ("COMPLETE", "FAILED") for u in key_units)
        node_id = key[0]
        fitness = None
        if terminal and all(u["state"] == "COMPLETE" for u in key_units) and node_id.startswith("search:"):
            h = key_units[0]["horizon"]
            try:
                from tools.i6n_swarm.coordinator import swarm_context
                context = swarm_context(root, h, ledger=led)
                supports = {f["fold"]: f["support"] for f in context["folds"]}
                receipts = {u["fold"]: terminals[u["fit_id"]]["receipt"]["shared_receipt"] for u in key_units}
                from tools.i6n_swarm.bridge import score_candidate
                origins = {f["fold"]: f["origins_sha256"] for f in context["folds"] if f.get("origins_sha256")}
                fitness = score_candidate(score_folds, e_plan, receipts, supports, key_units[0]["genotype"], h,
                                          origins if len(origins) == len(supports) else None)["fitness"]
            except (LookupError, ValueError, KeyError):
                fitness = None
        for u in key_units:
            term = terminals.get(u["fit_id"])
            shared = (term or {}).get("receipt", {}).get("shared_receipt", {})
            disposition = "COMPLETE" if u["state"] == "COMPLETE" else u["state"]
            for split in ("train", "inner"):
                n = (shared.get("counts") or {}).get(split)
                err = (shared.get("error_sums") or {}).get(split)
                naive = (shared.get("naive_sums") or {}).get(split)
                mae = err / n if n and err is not None else None
                naive_mae = naive / n if n and naive is not None else None
                rows.append(dict(stage=u["stage"], dag_node=u["dag_node"], horizon=u["horizon"],
                                 search_candidate_sha256=u["search_candidate"], genotype_canonical_sha256=u["genotype"],
                                 fold=u["fold"], disposition=disposition, split=split.upper(), population="S", n=n,
                                 error_sum=err, naive_sum=naive, mae=mae, naive_mae=naive_mae,
                                 skill_mae=(mae - naive_mae) / naive_mae if mae is not None and naive_mae else None,
                                 fitness=fitness))
        items = pending_by_key.get(key, [])
        if not terminal or not items or publisher is None:
            continue
        doc = _candidate_doc(payload, key_units, terminals, attempts_by_fit, fitness)
        fold_receipts = [terminals[u["fit_id"]]["receipt"]["candidate_receipt"] for u in key_units
                         if u["fit_id"] in terminals]
        result = publisher.record_candidate(doc, fold_receipts)
        for item in items:
            led.outbox_mark(item["id"], dict(item_id=result.get("item_id"), states=result.get("states")))
        published += 1
    decisions = _record_decisions(led, payload, publisher) if publisher is not None else 0
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row[k] is None else row[k]) for k in COLUMNS})
    common.atomic_write(root / "collect" / "RESULTS.csv", buffer.getvalue())
    remaining = len(led.outbox_pending())
    if not led.outbox_all():
        state, cause = "QUEUED", cause or "no terminal fold yet"
    else:
        state = "COMPLETE" if remaining == 0 else ("WAIT_DEPENDENCY" if publisher is None else "QUEUED")
    report = dict(at=led.clock(), published=published, decisions=decisions, pending=remaining, publication_state=state, cause=cause,
                  results_sha256=common.file_sha256(root / "collect" / "RESULTS.csv"), rows=len(rows))
    common.atomic_json(root / "collect" / "ANNUAL.json", annual_rollups(led, payload, terminals))
    common.atomic_json(root / "collect" / "COLLECT.json", report)
    common.atomic_json(root / "collect" / "ARTIFACTS.json", dict(
        plan_sha256=P.load(root)["sha256"], e_plan_sha256=payload["e_plan_sha256"],
        results_csv=dict(path="collect/RESULTS.csv", sha256=report["results_sha256"]),
        ledger=dict(path="state/ledger.sqlite3", note="authoritative journal; not committed"),
        storage_modes_active=["LOCAL_LEDGER_OUTBOX"] + (["D_PUBLISHER_LOCAL_COMPLETE"] if published else [])))
    return report
