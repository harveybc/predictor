"""Campaign coordinator: preflight, idempotent start, and the DAG tick (successor activation).

The tick is run by the campaign's systemd timer on the coordinator host. It
never trains and never launches an evaluator: it recovers expired leases,
recomputes every DAG node from ledger facts, registers the fit units of a
stage whose dependencies are COMPLETE (search -> annual weekly -> multi-output),
selects finalists, and writes status. A failure blocks only its dependants.
"""
import getpass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from tools import i6n_search as search
from tools.i6n_swarm import common, ledger as L, plan as P

FREEZE_GENOTYPE = common.sha256_json("I6N_SWARM_FREEZE_NO_FIT")
TERMINAL_UNIT = ("COMPLETE", "FAILED")


def _ledger(root, now=None):
    clock = (lambda: now) if now is not None else time.time
    return L.Ledger(Path(root) / "state" / "ledger.sqlite3", clock=clock)


def default_systemctl(*args):
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=60).returncode


def default_linger():
    return Path("/var/lib/systemd/linger", getpass.getuser()).exists()


# ------------------------------------------------------------------ swarm context
def swarm_context(root, horizon, ledger=None):
    led = ledger or L.Ledger.open_readonly(Path(root) / "state" / "ledger.sqlite3")
    node = next((n for n in led.dag_nodes() if n["node_id"] == f"search:h{horizon}"), None)
    if node and node["summary"] and node["summary"].get("swarm_context"):
        return node["summary"]["swarm_context"]
    raise LookupError(f"search:h{horizon} has no frozen swarm context yet")


def freeze_fit_id(experiment, item_id, horizon, cell_pin):
    return common.sha256_json(dict(kind="FREEZE", experiment=experiment, fold=item_id, horizon=horizon,
                                   cell_config_sha256=cell_pin["sha256"]))


def freeze_spec(payload, node_id, horizon, item):
    return dict(stage="freeze", dag_node=node_id, horizon=horizon, fold=item["fold_id"] if "fold_id" in item else item["week_id"],
                experiment=payload["experiment"], genotype_canonical_sha256=FREEZE_GENOTYPE,
                effective_config_sha256=item["cell_config"]["sha256"],
                search_candidate_sha256=common.sha256_json([node_id, item["cell_config"]["sha256"]]),
                device_class="cpu", fixture=payload["mode"] == "fixture", cell_config=item["cell_config"],
                kind="FREEZE", score_outer=node_id.startswith("freeze_weeks:"))


def freeze_receipt_document(unit, binding_body, binding_pin):
    return dict(fit_id=unit["fit_id"], fold=unit["fold"], experiment=unit["experiment"],
                effective_config_sha256=unit["effective_config_sha256"],
                genotype_canonical_sha256=unit["genotype_canonical_sha256"], shared_receipt=binding_body,
                candidate_receipt=dict(kind="FREEZE", binding=binding_pin))


# ------------------------------------------------------------------ DAG evaluation
def _unit_counts(units):
    counts = dict(expected=len(units), complete=0, failed=0, interrupted=0, running=0, queued=0, wait_resource=0)
    for u in units:
        key = {"COMPLETE": "complete", "FAILED": "failed", "INTERRUPTED": "interrupted", "RUNNING": "running",
               "WAIT_RESOURCE": "wait_resource"}.get(u["state"], "queued")
        counts[key] += 1
    return counts


def _state_from_units(units):
    if not units:
        return "QUEUED", None
    states = {u["state"] for u in units}
    if states <= {"COMPLETE"}:
        return "COMPLETE", None
    pending = states - set(TERMINAL_UNIT)
    if not pending:
        failed = sum(u["state"] == "FAILED" for u in units)
        return "FAILED", f"GAPS: {failed} unit(s) FAILED; no complete result is claimed"
    if "RUNNING" in states:
        return "RUNNING", None
    if pending <= {"WAIT_RESOURCE"}:
        return "WAIT_RESOURCE", sorted({u["cause"] for u in units if u["state"] == "WAIT_RESOURCE"})[0]
    return "QUEUED", None


def _freeze_node(led, payload, node, terminals, data_root=None):
    h, items = node["spec"]["horizon"], node["spec"]["items"]
    waits = [f"{i.get('fold_id', i.get('week_id'))}: {i['cause']}" for i in items if i["state"] == "WAIT_DEPENDENCY"]
    created = False
    frozen = {}
    for item in items:
        if item["state"] != "READY":
            continue
        key = item.get("fold_id", item.get("week_id"))
        if item.get("binding"):
            frozen[key] = (common.read_pinned(item["binding"], f"binding {key}", root=data_root),
                           item["binding"]["sha256"])
            continue
        fit = freeze_fit_id(payload["experiment"], key, h, item["cell_config"])
        if led.unit(fit) is None:
            led.ensure_unit(fit, freeze_spec(payload, node["node_id"], h, item))
            created = True
        term = terminals.get(fit)
        if term:
            frozen[key] = (term["receipt"]["shared_receipt"], term["receipt"]["candidate_receipt"]["binding"]["sha256"])
    units = led.units(dag_node=node["node_id"])
    state, cause = _state_from_units(units)
    if waits:
        state, cause = "WAIT_DEPENDENCY", "; ".join(waits)
    elif state == "QUEUED" and not units:
        state = "COMPLETE"
    return state, cause, created, frozen


def refusal_code(exc):
    return f"SCORE_REFUSED:{getattr(exc, 'code', None) or type(exc).__name__}: {exc}"[:300]


def _candidate_fitness(units, terminals, context, e_plan, score_folds, production=False):
    """Fold-mean fitness per genotype using lane E's score_folds (no fitness unless all folds COMPLETE)."""
    by_genotype = {}
    for u in units:
        by_genotype.setdefault(u["genotype"], []).append(u)
    supports = {f["fold"]: f["support"] for f in context["folds"]}
    out = []
    for genotype, items in by_genotype.items():
        receipts = {u["fold"]: terminals[u["fit_id"]]["receipt"]["shared_receipt"] for u in items
                    if u["state"] == "COMPLETE" and u["fit_id"] in terminals}
        record = dict(genotype_canonical_sha256=genotype, effective_config=items[0]["spec"]["effective_config"],
                      size_class=items[0]["spec"].get("size_class"), fitness=None,
                      search_candidate_sha256=items[0]["search_candidate"])
        if set(receipts) == set(supports):
            from tools.i6n_swarm.bridge import origins_of, score_candidate
            try:
                scored = score_candidate(score_folds, e_plan, receipts, supports, genotype, context["horizon"],
                                         origins_of(context), production=production)
            except Exception as exc:  # one refused candidate never aborts the tick (E-B7)
                record["score_refused"] = refusal_code(exc)
            else:
                record.update(fitness=scored["fitness"], origins_binding=scored.get("origins_binding"),
                              aggregator=scored.get("aggregator"))
        out.append(record)
    return out


def select_finalists(records, seed_identity):
    complete = sorted((r for r in records if r["fitness"] is not None),
                      key=lambda r: (r["fitness"], r["genotype_canonical_sha256"]))
    if not complete:
        return []
    best = dict(complete[0], role="BEST")
    out = [best]
    rest = [r for r in complete[1:]]
    control = next((r for r in rest if r["genotype_canonical_sha256"] == seed_identity), None)
    if control is not None:
        out.append(dict(control, role="MINIMAL_CONTROL"))
    elif rest:
        small = sorted(rest, key=lambda r: (r["effective_config"]["samples"], r["fitness"]))
        out.append(dict(small[0], role="SMALL_ALTERNATIVE"))
    else:
        best["control_status"] = "CONTROL_UNAVAILABLE"
    return out


def _search_node(led, payload, node, terminals, frozen_folds, e_plan, score_folds):
    h = node["spec"]["horizon"]
    summary = dict(node["summary"] or {})
    if "swarm_context" not in summary:
        if frozen_folds is None:
            return node["state"], node["cause"], summary
        ordered = [(fold, *frozen_folds[fold]) for fold in payload["folds"]]
        summary["swarm_context"] = payload["swarm_contexts"].get(str(h)) or P.swarm_context(payload, h, ordered)
    units = led.units(dag_node=node["node_id"])
    events = {e["event"]: e for e in led.dag_events(node["node_id"])}
    if "SEARCH_TERMINAL" not in events:
        return ("RUNNING" if any(u["state"] == "RUNNING" for u in units) else "QUEUED"), None, summary
    if any(u["state"] == "RUNNING" for u in units):
        return "RUNNING", "SEARCH_TERMINAL_RECORDED_WAITING_FOR_RUNNING_FOLDS", summary
    persisted = led.search_state(node["node_id"])
    records = _candidate_fitness(units, terminals, summary["swarm_context"], e_plan, score_folds,
                                 production=payload["mode"] == "production")
    finalists = None
    if persisted is not None:
        try:
            from tools.i6n_search_plan import select_finalists as e_select
            finalists = e_select(e_plan, persisted["state"])
            summary["finalist_rule"] = "E.select_finalists"
        except (ImportError, KeyError, TypeError, ValueError):
            finalists = None
    if finalists is None:
        seed_id = payload["initial_genotypes"].get(str(h), {}).get("seed_genotype_canonical_sha256")
        finalists = select_finalists(records, seed_id)
        summary["finalist_rule"] = "B.best_plus_minimal_control"
    summary.update(candidates=records, stop_reason=events["SEARCH_TERMINAL"]["detail"],
                   finalists=[{k: f.get(k) for k in ("role", "genotype_canonical_sha256", "fitness", "size_class",
                                                      "search_candidate_sha256", "effective_config", "control_status")}
                              for f in finalists])
    if not finalists:
        return "FAILED", "NO_COMPLETE_CANDIDATES: no candidate has all folds COMPLETE", summary
    return "COMPLETE", None, summary


def _annual_node(led, payload, node, search_summary, frozen_weeks):
    h = node["spec"]["horizon"]
    if not node["spec"]["capable"]:
        return "WAIT_DEPENDENCY", node["spec"]["cause"], False
    created = False
    for finalist in search_summary["finalists"]:
        config = finalist["effective_config"]
        for week_id, (body, sha) in sorted(frozen_weeks.items()):
            fit = common.fit_id(experiment=payload["experiment"], fold=week_id, effective_config=config,
                                context=body["i6n_evaluation_context"], support=body["i6n_support"])
            if led.unit(fit) is not None:
                continue
            led.ensure_unit(fit, dict(
                stage="annual", dag_node=node["node_id"], horizon=h, fold=week_id, experiment=payload["experiment"],
                genotype_canonical_sha256=finalist["genotype_canonical_sha256"],
                effective_config_sha256=common.sha256_json(config),
                search_candidate_sha256=finalist["search_candidate_sha256"], device_class="gpu_admitted",
                fixture=payload["mode"] == "fixture", effective_config=config, finalist_role=finalist["role"],
                size_class=finalist.get("size_class"), binding_sha256=sha,
                request=dict(context=body["i6n_evaluation_context"], support=body["i6n_support"])))
            created = True
    state, cause = _state_from_units(led.units(dag_node=node["node_id"]))
    return state, cause, created


MULTI_OUTPUT_CAPABILITY = "DIRECT_MULTI_OUTPUT_ARM"


def family_finalists(nodes, horizons):
    """Search finalists in lane C's resolve format (selection on 2023 inner folds only)."""
    out = {}
    for h in horizons:
        summary = (nodes.get(f"search:h{h}") or {}).get("summary") or {}
        roles = {}
        for f in summary.get("finalists") or []:
            role = "MINIMAL_CONTROL" if f["role"] in ("MINIMAL_CONTROL", "SMALL_ALTERNATIVE") else f["role"]
            roles[role] = dict(selection_basis="INNER_FOLD_FITNESS", fitness=f["fitness"], source_role=f["role"],
                               search_candidate_sha256=f["search_candidate_sha256"],
                               genotype_canonical_sha256=f["genotype_canonical_sha256"])
        if roles:
            out[str(h)] = roles
    return out


def _multi_node(led, payload, node, nodes, data_root=None):
    """Lane C's arms template resolved from search finalists; only READY arms with an A/E config may run."""
    pin = payload["inputs"].get("multi_output")
    if not pin:
        return "WAIT_DEPENDENCY", "MULTI_OUTPUT_ARMS_UNDEFINED: no lane C arms manifest pinned", False, None
    from tools.i6n_weekly_multi_output import FAMILIES, MultiOutputError, resolve_arms
    manifest = common.read_pinned(pin, "multi-output arms")
    family = node["spec"]["family"]
    try:
        resolved = resolve_arms(manifest, family_finalists(nodes, FAMILIES[family]))
    except MultiOutputError as exc:
        return "FAILED", f"MULTI_OUTPUT_RESOLVE_REFUSED: {exc}", False, None
    ready = [a for a in resolved["arms"] if a["family"] == family and a["state"] == "READY"]
    summary = dict(resolved_manifest_sha256=resolved["manifest_sha256"], ready_arms=len(ready),
                   arms=len([a for a in resolved["arms"] if a["family"] == family]))
    if not ready:
        cause = next((a["cause"] for a in resolved["arms"] if a["family"] == family), "NO_ARMS_FOR_FAMILY")
        return "WAIT_DEPENDENCY", cause, False, summary
    if MULTI_OUTPUT_CAPABILITY not in payload.get("evaluator_capabilities", []):
        return ("WAIT_DEPENDENCY", "EVALUATOR_LACKS_DIRECT_MULTI_OUTPUT_ARM: lanes A/E have not provided an "
                "adapted multi-output config and evaluator path", False, summary)
    unbound = [a["arm_id"] for a in ready if not a.get("config") or not a.get("binding")]
    if unbound:
        return ("WAIT_DEPENDENCY", f"ARM_CONFIG_UNDEFINED: {len(unbound)} READY arms lack an A/E config or binding",
                False, summary)
    created = False
    for arm in ready:
        body = common.read_pinned(arm["binding"], f"arm {arm['arm_id']}", root=data_root)
        fit = common.fit_id(experiment=payload["experiment"], fold=arm["week_id"], effective_config=arm["config"],
                            context=body["i6n_evaluation_context"], support=body["i6n_support"])
        if led.unit(fit) is None:
            led.ensure_unit(fit, dict(
                stage="multi_output", dag_node=node["node_id"], horizon=max(arm["horizons"]), fold=arm["week_id"],
                experiment=payload["experiment"], genotype_canonical_sha256=arm["recipe"]["genotype_canonical_sha256"],
                effective_config_sha256=common.sha256_json(arm["config"]),
                search_candidate_sha256=arm["recipe"]["search_candidate_sha256"],
                device_class="gpu_admitted", fixture=payload["mode"] == "fixture", effective_config=arm["config"],
                arm_id=arm["arm_id"], binding_sha256=arm["binding"]["sha256"],
                request=dict(context=body["i6n_evaluation_context"], support=body["i6n_support"])))
            created = True
    state, cause = _state_from_units(led.units(dag_node=node["node_id"]))
    return state, cause, created, summary


def tick(root, now=None, score_folds=None):
    root = Path(root)
    envelope = P.load(root)
    payload = envelope["payload"]
    led = _ledger(root, now)
    recovered = led.recover()
    from tools.i6n_swarm.bridge import resolve_score_folds
    score_folds = score_folds or resolve_score_folds()
    e_plan = P.e_plan(root)
    terminals = led.terminals()
    nodes = {n["node_id"]: n for n in led.dag_nodes()}
    frozen = {}
    activated = []
    for node_id in [n["node_id"] for n in sorted(nodes.values(), key=lambda n: n["ord"])]:
        node = nodes[node_id]
        deps = [nodes[d] for d in node["depends_on"]]
        blocked = [d for d in deps if d["state"] != "COMPLETE"]
        summary = None
        if node["stage"] == "freeze":
            state, cause, created, items = _freeze_node(led, payload, node, terminals,
                                                        P.private(root).get("shared_data_root"))
            frozen[node_id] = items
        elif blocked and node["stage"] != "search":
            state, created = "WAIT_DEPENDENCY", False
            cause = "; ".join(f"{d['node_id']} is {d['state']}" + (f" ({d['cause']})" if d["cause"] else "")
                              for d in blocked)
        elif node["stage"] == "search":
            h = node["spec"]["horizon"]
            dep = nodes[f"freeze_folds:h{h}"]
            items = frozen.get(f"freeze_folds:h{h}", {})
            ready = items if dep["state"] == "COMPLETE" and set(items) == set(payload["folds"]) else None
            state, cause, summary = _search_node(led, payload, node, terminals, ready, e_plan, score_folds)
            created = False
            if "swarm_context" not in summary:
                state, cause = "WAIT_DEPENDENCY", f"freeze_folds:h{h} is {dep['state']}" + (f" ({dep['cause']})" if dep["cause"] else "")
        elif node["stage"] == "annual":
            h = node["spec"]["horizon"]
            state, cause, created = _annual_node(led, payload, node, nodes[f"search:h{h}"]["summary"],
                                                 frozen.get(f"freeze_weeks:h{h}", {}))
        else:
            state, cause, created, summary = _multi_node(led, payload, node, nodes,
                                                         P.private(root).get("shared_data_root"))
        if created:
            activated.append(node_id)
        led.dag_set(node_id, state, cause, summary)
        nodes[node_id] = dict(node, state=state, cause=cause, summary=summary if summary is not None else node["summary"])
    heartbeat = dict(at=led.clock(), recovered=len(recovered), activated=activated,
                     states={k: v["state"] for k, v in nodes.items()})
    common.atomic_json(root / "status" / "TICK.json", heartbeat)
    try:
        from tools.i6n_swarm import status as S
        common.atomic_json(root / "status" / "STATUS.json", S.compute(root, now=led.clock()))
    except Exception as exc:  # status is a view; it must never block activation
        common.atomic_json(root / "status" / "STATUS_ERROR.json", dict(error=f"{type(exc).__name__}: {exc}"))
    return dict(recovered=recovered, activated=activated, states=heartbeat["states"])


# ------------------------------------------------------------------ start / preflight
def units_for_roles(payload, roles):
    from tools.i6n_swarm import units as U
    return [name for role in roles for name in U.startable_units(payload["campaign_id"], role)]


def start(root, roles=(), unit_dir=None, systemctl=None, now=None):
    """Idempotent: DAG init, prior adoption, one tick, then start only inactive campaign units."""
    root = Path(root)
    payload = P.load(root)["payload"]
    led = _ledger(root, now)
    for node in payload["dag"]:
        led.dag_upsert(node["node_id"], node["stage"], node["ord"], node["depends_on"], node["spec"])
    adopted = adopt_prior(root, led, payload)
    result = tick(root, now=now)
    started = []
    names = units_for_roles(payload, roles)
    if names:
        systemctl = systemctl or default_systemctl
        unit_dir = Path(unit_dir or Path.home() / ".config" / "systemd" / "user")
        missing = [n for n in names if not (unit_dir / n).exists()]
        if missing:
            raise FileNotFoundError(f"install first; missing units {missing}")
        inactive = [n for n in names if systemctl("is-active", n) != 0]
        if inactive:
            if systemctl("start", *inactive) != 0:
                raise RuntimeError(f"systemctl start failed for {inactive}")
            started = inactive
    receipt = dict(at=led.clock(), started=started, adopted_prior=adopted, activated=result["activated"])
    common.atomic_json(root / "status" / "START.json", receipt)
    return receipt


def adopt_prior(root, led, payload):
    """Register prior receipts that lane E maps as reusable for exactly one fit_id.

    E's legacy mapping (i6n.legacy_mapping.v1) currently marks all 48 endpoint
    candidates ``fold_score_reuse: NONE``: they are recorded as not reusable
    and nothing is adopted. A reusable item must name its fit_id and a pinned
    receipt; it is adopted at the first claim of that fit (never refit).
    """
    pin = payload["inputs"].get("prior_candidates")
    if not pin:
        return dict(registered=0, not_reusable=0)
    mapping = common.read_pinned(pin, "prior mapping")
    registered = not_reusable = 0
    for item in mapping.get("candidates", []):
        reuse = item.get("fold_score_reuse", item.get("verdict"))
        if reuse not in ("FOLD_RECEIPT", "EQUIVALENT") or not item.get("fit_id") or not item.get("receipt"):
            not_reusable += 1
            continue
        led.register_prior(item["fit_id"], common.read_pinned(item["receipt"], "prior receipt"), source=item["receipt"])
        registered += 1
    return dict(registered=registered, not_reusable=not_reusable)


def preflight(root, which=shutil.which, linger=default_linger):
    root = Path(root)
    checks = []

    def check(name, ok, cause=None):
        checks.append(dict(name=name, ok=bool(ok), cause=None if ok else cause))

    try:
        payload = P.load(root)["payload"]
        check("plan_seal", True)
    except Exception as exc:
        check("plan_seal", False, str(exc))
        return dict(state="BLOCKED", checks=checks)
    bad = []
    for key, pin in payload["inputs"].items():
        if pin:
            try:
                common.read_pinned(pin, key)
            except Exception as exc:
                bad.append(f"{key}: {exc}")
    check("input_digests", not bad, "; ".join(bad))
    private = P.private(root)
    seed_ok, seed_cause = payload["seed"] == common.SEED and payload["direction"] == "minimize", []
    for role in private["hosts"]:
        path = root / "private" / "hosts" / role / "doin_node.json"
        if path.exists():
            for d in json.loads(path.read_text())["domains"]:
                oc = d["optimization_config"]
                if d["higher_is_better"] is not False or oc["shared_population_seed"] != common.SEED:
                    seed_ok = False
                    seed_cause.append(d["domain_id"])
    check("seed_direction", seed_ok, f"role/seed mismatch in {seed_cause}")
    admitted = [c for c, v in payload["class_caps"].items() if v["state"] == "ADMITTED"]
    check("class_admission", admitted, "no size class has an admitted pilot receipt")
    check("systemctl_user", which("systemctl"), "systemctl not found")
    check("user_linger", linger(), "user lingering disabled: user units would not start at boot")
    state_dir = root / "state"
    check("ledger_writable", os.access(state_dir, os.W_OK), "state directory not writable")
    token = Path(private["ledger"]["token_file"])
    check("ledger_token_private", token.exists() and P.is_private_ok(token), "token file missing or group/world readable")
    if payload["mode"] == "production":
        launcher = (private.get("launcher") or {}).get("crispdm_run")
        check("crispdm_run", launcher and Path(launcher).exists(), "admitted launcher missing")
        for role, host in private["hosts"].items():
            for device in host.get("devices", []):
                ok = device.get("cuda_visible_devices", "").startswith("GPU-") and "," not in device["cuda_visible_devices"]
                check(f"device_{role}_{device['cls']}", ok, "explicit single GPU UUID required")
    waiting = [n["node_id"] for n in payload["dag"] if n["stage"] == "freeze"
               and any(i["state"] == "WAIT_DEPENDENCY" for i in n["spec"]["items"])]
    checks.append(dict(name="data_dependencies", ok=True, cause=None, info=dict(wait_dependency_nodes=waiting)))
    return dict(state="READY" if all(c["ok"] for c in checks) else "BLOCKED", checks=checks,
                note="preflight launches no process and trains nothing")
