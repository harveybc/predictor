"""``plan``: build the sealed campaign DAG from pinned A/C/E inputs (no training, no data reads).

Inputs (all digest-pinned in the private config):
- E: sealed search plan JSON (``plan_sha256`` verified with E's formula);
- C: weekly views (folds per horizon, weekly refit units with per-horizon cells);
- optional multi-output arms manifest, optional prior mapping (E legacy mapping).

Outputs under the campaign root (which must not exist):
- PLAN.json      {payload, sha256}  sealed, no host names/addresses/UUIDs;
- private/       0700: per-role bridge node config and DOIN node config;
- state/ status/ collect/ archive/   empty destinations.
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from tools.i6n_swarm import common

SCHEMA = "i6n.swarm_campaign.plan.v1"
PRIVATE_SCHEMA = "i6n.swarm_campaign.private.v1"
CONTEXT_SCHEMA = "i6n.swarm.context.v1"
TEST_START = "2025-01-01"
RUNNABLE = ("READY", "READY_WITH_DECLARED_CLOCK_ASSUMPTION")
FAMILIES = {"short": (2, 3, 4, 5, 6, 12), "long": (18, 24, 36, 48, 60, 72)}
WORKER_ROLES = ("preferred", "secondary", "coordinator_gpu")


def _e_plan_sha(plan):
    return common.sha256_json({k: v for k, v in plan.items() if k not in ("plan_sha256", "seal")})


def _check_search(plan):
    if plan.get("schema") != "i6n.search_plan.v1" or plan.get("plan_sha256") != _e_plan_sha(plan):
        raise ValueError("E search plan schema or seal mismatch")
    if plan.get("seed") != common.SEED:
        raise ValueError("E plan seed must be 6102026")
    s = plan["search"]
    if not (1 <= s["population_size"] <= 8 and s["generations"] == 3 and s["new_candidates_per_horizon"] <= 24
            and s["mutation"] == 0.8 and s["immigration"] == 0.2 and s["tolerance"] == 1e-4
            and s["patience"] == 3 and s.get("direction", "minimize") == "minimize" and not s.get("crossover")):
        raise ValueError("E search settings differ from plan section 4")


def e_fold_ids(e_plan):
    folds = e_plan["folds"]
    return list(folds.get("quarters") or folds["fold_ids"])


def origins_from_c_plan(c_plan, unit_id, horizon):
    """C's per-split origins digests for (unit, h) on the fit role: E binds receipts to them."""
    unit = next((u for u in c_plan["units"] if u["unit_id"] == unit_id), None)
    if unit is None:
        raise ValueError(f"C plan lacks unit {unit_id}")
    if "support" in unit:
        split = unit["support"][c_plan["fit_role"]][str(horizon)]
    else:
        split = unit["support_fit_role_business"][str(horizon)]
    return dict(train=split["TRAIN"]["origins_sha256"], inner=split["INNER"]["origins_sha256"])


def _fold_map(e_plan, views, horizons):
    """E fold id -> C fold id. C may suffix a declared assumption (e.g. 2023Q1_DCA); never invent ids."""
    c_ids = set()
    for h in horizons:
        c_ids.update(e["fold_id"] for e in views["folds"].get(str(h), []))
    mapping = {}
    for e_id in e_fold_ids(e_plan):
        match = [c for c in c_ids if c == e_id or c.startswith(e_id + "_")]
        if len(match) != 1:
            raise ValueError(f"E fold {e_id} has {len(match)} matching C fold ids")
        mapping[e_id] = match[0]
    if set(mapping.values()) != c_ids:
        raise ValueError(f"C fold ids outside the E plan: {sorted(c_ids - set(mapping.values()))}")
    return mapping


def _no_test(label, *stamps):
    for stamp in stamps:
        if isinstance(stamp, str) and stamp[:10] >= TEST_START:
            raise ValueError(f"TEST period closed: {label} references {stamp}")


def _class_caps(cfg, classes):
    caps = {}
    for item in classes:
        admitted = cfg.get("class_admission", {}).get(item["class_id"])
        if admitted is None:
            caps[item["class_id"]] = dict(state="NOT_ADMITTED", cause="NO_ADMITTED_PILOT_RECEIPT")
            continue
        if "receipt" in admitted:
            receipt = common.read_pinned(admitted["receipt"], f"pilot receipt {item['class_id']}")
            peak = receipt.get("observed_peak_bytes")
            if receipt.get("state") != "COMPLETED" or type(peak) is not int or peak <= 0:
                raise ValueError(f"pilot receipt for {item['class_id']} lacks a completed measured peak")
            caps[item["class_id"]] = dict(state="ADMITTED", source="PILOT_RECEIPT", receipt_sha256=admitted["receipt"]["sha256"],
                                          child_memory_bytes=int(1.25 * peak), measured_peak_bytes=peak,
                                          child_wall_seconds=int(admitted["child_wall_seconds"]),
                                          vram_bytes=admitted.get("vram_bytes", 0),
                                          small_pilot=bool(admitted.get("small_pilot")))
        elif admitted.get("source") == "FIXTURE_DECLARED_NOT_MEASURED" and cfg["mode"] == "fixture":
            caps[item["class_id"]] = dict(state="ADMITTED", source=admitted["source"],
                                          child_memory_bytes=admitted["child_memory_bytes"],
                                          child_wall_seconds=admitted["child_wall_seconds"],
                                          vram_bytes=admitted.get("vram_bytes", 0),
                                          small_pilot=bool(admitted.get("small_pilot")))
        else:
            raise ValueError(f"class {item['class_id']}: caps come only from an admitted pilot receipt")
    return caps


def _binding(pin, horizon, what, root=None):
    body = common.read_pinned(pin, what, root=root)
    if body.get("i6n_horizon") != horizon:
        raise ValueError(f"{what}: binding horizon mismatch")
    return body


def swarm_context(payload, horizon, frozen):
    """frozen: ordered list of (fold_id, binding body, binding sha256)."""
    origins = payload.get("fold_origins", {}).get(str(horizon), {})
    folds = [dict(fold=fold_id, context=body["i6n_evaluation_context"], support=body["i6n_support"],
                  binding_sha256=sha, origins_sha256=origins.get(fold_id)) for fold_id, body, sha in frozen]
    # The search plan's aggregator keys folds by the campaign (C) fold ids, which carry any declared assumption.
    return dict(schema=CONTEXT_SCHEMA, experiment=payload["experiment"], plan_sha256=payload["e_plan_sha256"],
                horizon=horizon, dag_node=f"search:h{horizon}", fixture=payload["mode"] == "fixture",
                budget_new_candidates=payload["search"]["new_candidates_per_horizon"],
                classes=payload["classes"], folds=folds)


def doin_template(payload, private, role):
    """Everything a worker needs to render its DOIN node config later, without the campaign root."""
    host = private["hosts"][role]
    doin = private["doin"]
    return dict(campaign_id=payload["campaign_id"], node_label=f"i6n-swarm-{payload['campaign_id']}-{role}",
                mode=payload["mode"], search_plan_document=payload.get("search_plan_document"),
                admitted_classes=[k for k, v in payload["class_caps"].items() if v["state"] == "ADMITTED"],
                port=host["doin_port"], data_dir=str(Path(host["data_dir"]) / "doin"),
                bootstrap_peers=[a for other, a in sorted(doin.get("peers", {}).items()) if other != role],
                shared_min_peers=doin.get("shared_min_peers", 2), claim_timeout=doin.get("claim_timeout", 600),
                predictor_root=host["workdir"], search=payload["search"],
                node_config=str(Path(private["root"]) / "private" / "hosts" / role / "node.json"),
                e_plan=common.read_pinned(payload["inputs"]["search_plan"], "E search plan"),
                admitted_classes=sorted(k for k, v in payload["class_caps"].items() if v["state"] == "ADMITTED"))


def render_doin_config(template, contexts):
    s = template["search"]
    joint_plan = template.get("search_plan_document")
    if template.get("mode") == "production" and joint_plan is None:
        raise ValueError("production requires the sealed joint search plan")

    def domain(h, context):
        joint = {}
        if joint_plan is not None:
            joint = dict(i6n_plan=deepcopy(joint_plan),
                         i6n_admitted_classes=list(template.get("admitted_classes", [])),
                         i6n_folds={f["fold"]: dict(i6n_support=f["support"],
                             i6n_evaluation_context=f["context"],
                             **({"i6n_origins_sha256": f["origins_sha256"]} if f.get("origins_sha256") else {}))
                             for f in context["folds"]})
        return dict(domain_id=f"i6n-swarm-{template['campaign_id']}-h{h}", optimize=True, evaluate=False,
                    optimization_plugin="predictor", higher_is_better=False,
                    optimization_config=dict(
                        **joint,
                        predictor_root=template["predictor_root"], optimizer_plugin="i6n_optimizer",
                        shared_population=True, shared_population_size=s["population_size"],
                        shared_population_seed=common.SEED, n_generations=s["generations"],
                        optimization_patience=s["patience"], improvement_tolerance=s["tolerance"],
                        reproduction_failure_policy="fail_closed", higher_is_better=False,
                        # E's v2 path (E-B1): sealed plan, staged classes, minimal seed in generation 0.
                        i6n_plan=template["e_plan"], i6n_horizon=h,
                        i6n_admitted_classes=list(template["admitted_classes"]),
                        i6n_folds={f["fold"]: dict(i6n_support=f["support"], i6n_evaluation_context=f["context"],
                                                   **({"i6n_origins_sha256": f["origins_sha256"]}
                                                      if f.get("origins_sha256") else {}))
                                   for f in context["folds"]},
                        i6n_swarm=dict(node_config=template["node_config"], horizon=h, dag_node=f"search:h{h}",
                                       seed=common.SEED, context=context)))
    return dict(node_label=template["node_label"], port=template["port"], data_dir=template["data_dir"],
                bootstrap_peers=template["bootstrap_peers"], shared_min_peers=template["shared_min_peers"],
                shared_claim_timeout=template["claim_timeout"], shared_claim_result_patience=10**9,
                domains=[domain(int(h), ctx) for h, ctx in sorted(contexts.items(), key=lambda kv: int(kv[0]))])


def doin_node_config(payload, private, role, contexts):
    return render_doin_config(doin_template(payload, private, role), contexts)


def bridge_node_config(payload, private, role, device):
    host = private["hosts"][role]
    ledger = private["ledger"]
    caps = {k: v for k, v in payload["class_caps"].items() if v["state"] == "ADMITTED"}
    return dict(schema="i6n.swarm.node.v1", campaign_id=payload["campaign_id"], owner=f"{role}-{device['cls']}",
                host_key=role, device=device, fixture=payload["mode"] == "fixture",
                exclusive_host=device["cls"] != "rtx5070ti",
                evaluator_argv=host.get("evaluator_argv") or [host["python"], "-m", "tools.i6n_historical_evaluator", "evaluate"],
                freeze_argv=[host["python"], "-m", "tools.i6n_historical_evaluator", "freeze"],
                python=host["python"], workdir=host["workdir"], launcher=private.get("launcher"),
                class_caps=caps, lease=private["lease"],
                measured={k: dict(peak_bytes=v["measured_peak_bytes"]) for k, v in caps.items()
                          if v.get("measured_peak_bytes")},
                ledger=dict(url=ledger["url"], token_file=ledger["token_file"]),
                logs_dir=str(Path(host["data_dir"]) / "logs"), state_dir=str(Path(host["data_dir"]) / "state"),
                shared_data_root=private["shared_data_root"], experiment=payload["experiment"],
                doin_node_config=str(Path(private["root"]) / "private" / "hosts" / role / "doin_node.json"),
                doin_template=doin_template(payload, private, role),
                freeze_caps=private.get("freeze_caps") or dict(child_memory_bytes=4 * 1024**3, child_wall_seconds=3600),
                campaign_root=private["root"])


def _resolve_host(entry):
    """Optional reference into the private worker inventory (never copied into the sealed plan)."""
    entry = deepcopy(entry)
    ref = entry.pop("inventory", None)
    if ref:
        inventory = json.loads(Path(ref["path"]).read_text())
        item = next(w for w in inventory if w["role"] == ref["role"])
        entry.setdefault("python", item.get("fit_python"))
        entry.setdefault("workdir", item.get("worktree"))
        entry.setdefault("environment", item.get("environment", {}))
        if item.get("environment", {}).get("CUDA_VISIBLE_DEVICES"):
            for device in entry.get("devices", []):
                device.setdefault("cuda_visible_devices", item["environment"]["CUDA_VISIBLE_DEVICES"])
    return entry


def build(config_path, output_root):
    config_path, root = Path(config_path), Path(output_root)
    if root.exists():
        raise FileExistsError("campaign root must not exist")
    cfg = json.loads(config_path.read_text())
    if cfg.get("schema") != PRIVATE_SCHEMA:
        raise ValueError("private campaign config schema mismatch")
    if cfg.get("seed") != common.SEED:
        raise ValueError("campaign seed must be 6102026")
    if cfg.get("mode") not in ("fixture", "production"):
        raise ValueError("mode must be fixture or production")
    e_plan = common.read_pinned(cfg["search_plan"], "E search plan")
    _check_search(e_plan)
    views = common.read_pinned(cfg["weekly_views"], "C weekly views")
    multi = common.read_pinned(cfg["multi_output"], "multi-output arms") if cfg.get("multi_output") else None
    if multi is not None and multi.get("schema") != "i6n.multi_output_arms.v1":
        raise ValueError("multi-output manifest must be lane C's i6n.multi_output_arms.v1 template")
    horizons = list(e_plan["horizons_hours"])
    fold_map = _fold_map(e_plan, views, horizons)
    fold_ids = [fold_map[e_id] for e_id in e_fold_ids(e_plan)]
    c_plan = common.read_pinned(cfg["weekly_plan"], "C weekly plan") if cfg.get("weekly_plan") else None
    if cfg["mode"] == "production" and c_plan is None:
        raise ValueError("production ranking requires C's plan so E can verify origins bindings")
    origins = {}
    classes = [dict(class_id=c["class_id"], max_samples=c["max_samples"], max_width=c["max_width"],
                    min_delta_x_hours=c.get("min_delta_x_hours", 1)) for c in e_plan["classes"]]
    caps = _class_caps(cfg, e_plan["classes"])
    capable = "VALIDATION_OUTER_WEEK" in cfg.get("evaluator_capabilities", [])
    dag, frozen_ready = [], {}
    for h in horizons:
        entries = views["folds"].get(str(h), [])
        if sorted(e["fold_id"] for e in entries) != sorted(fold_ids):
            raise ValueError(f"C folds for h{h} differ from the campaign fold ids {fold_ids}")
        entries = sorted(entries, key=lambda e: fold_ids.index(e["fold_id"]))
        folds, frozen = [], []
        for e in entries:
            _no_test(f"fold {e['fold_id']}", e.get("cutoff_utc"))
            state = "READY" if e["state"] in RUNNABLE else e["state"]
            item = dict(fold_id=e["fold_id"], state=state, cause=e.get("cause"), unit_sha256=e.get("unit_sha256"),
                        binding=e.get("binding"), cell_config=e.get("cell_config"), c_state=e["state"],
                        declared_assumption=e.get("declared_assumption"))
            if c_plan is not None and state == "READY":
                origins[(str(h), e["fold_id"])] = origins_from_c_plan(c_plan, e.get("unit_id", e["fold_id"]), h)
            item["origins_sha256"] = origins.get((str(h), e["fold_id"]))
            if state == "READY" and e.get("binding"):
                frozen.append((e["fold_id"], _binding(e["binding"], h, f"fold {e['fold_id']} h{h}", cfg["shared_data_root"]),
                               e["binding"]["sha256"]))
            elif state == "READY" and not e.get("cell_config"):
                raise ValueError(f"READY fold {e['fold_id']} h{h} has neither binding nor cell config")
            folds.append(item)
        if len(frozen) == len(fold_ids):
            frozen_ready[str(h)] = frozen
        weeks = []
        for w in views["weeks"]:
            _no_test(f"week {w['week_id']}", w.get("cutoff_utc"), w.get("release_at_utc"), w["week_id"][1:])
            if h in w.get("horizons", []):
                binding = (w.get("bindings") or {}).get(str(h))
                cell = (w.get("cell_configs") or {}).get(str(h))
                if binding:
                    _binding(binding, h, f"week {w['week_id']} h{h}", cfg["shared_data_root"])
                weeks.append(dict(week_id=w["week_id"], state="READY", cause=None, binding=binding, cell_config=cell))
            else:
                cause = (w.get("cause") or {}).get(str(h)) or "NOT_LISTED_FOR_HORIZON"
                state = "WAIT_DEPENDENCY" if "WAIT" in str(cause) else "NO_ELIGIBLE_ROWS"
                weeks.append(dict(week_id=w["week_id"], state=state, cause=cause, binding=None, cell_config=None))
        dag.append(dict(node_id=f"freeze_folds:h{h}", stage="freeze", depends_on=[], spec=dict(horizon=h, items=folds)))
        dag.append(dict(node_id=f"freeze_weeks:h{h}", stage="freeze", depends_on=[], spec=dict(horizon=h, items=weeks)))
    for h in horizons:
        dag.append(dict(node_id=f"search:h{h}", stage="search", depends_on=[f"freeze_folds:h{h}"],
                        spec=dict(horizon=h, folds=fold_ids)))
    for h in horizons:
        dag.append(dict(node_id=f"annual:h{h}", stage="annual", depends_on=[f"search:h{h}", f"freeze_weeks:h{h}"],
                        spec=dict(horizon=h, capable=capable,
                                  cause=None if capable else "EVALUATOR_LACKS_VALIDATION_OUTER_WEEK_SCORING")))
    for family, members in FAMILIES.items():
        hs = [h for h in members if h in horizons]
        arms = [a for a in (multi or {}).get("arms", []) if a["family"] == family]
        for arm in arms:
            _no_test(f"arm {arm['arm_id']}", arm.get("cutoff_utc"), arm.get("release_at_utc"))
        dag.append(dict(node_id=f"multi_output:{family}", stage="multi_output",
                        depends_on=[f"annual:h{h}" for h in hs], spec=dict(family=family, horizons=hs)))
    for index, node in enumerate(dag):
        node["ord"] = index
    s = e_plan["search"]
    n_arms = len((multi or {}).get("arms", []))
    if n_arms > e_plan.get("multi_output", {}).get("max_dispositions", 212):
        raise ValueError("multi-output arms exceed the plan budget of 212")
    payload = dict(
        schema=SCHEMA, campaign_id=cfg["campaign_id"], mode=cfg["mode"], experiment=cfg["experiment"],
        seed=common.SEED, direction=common.DIRECTION, e_plan_sha256=e_plan["plan_sha256"],
        search=dict(population_size=s["population_size"], generations=s["generations"],
                    new_candidates_per_horizon=s["new_candidates_per_horizon"], mutation=s["mutation"],
                    immigration=s["immigration"], tolerance=s["tolerance"], patience=s["patience"]),
        horizons=horizons, folds=fold_ids, fold_map=fold_map, classes=classes, class_caps=caps,
        search_plan_document=e_plan if cfg["mode"] == "production" else None,
        fold_origins={h: {f: o for (hh, f), o in origins.items() if hh == h} for h in {k[0] for k in origins}},
        initial_genotypes=e_plan.get("initial_genotypes", {}),
        budget=dict(search_new_fits_max=s["new_candidates_per_horizon"] * len(fold_ids) * len(horizons),
                    max_physical_attempts_per_fit=3, annual_dispositions_max=2 * len(horizons) * 53,
                    multi_output_dispositions_max=212, multi_output_arms=n_arms),
        inputs=dict(search_plan=cfg["search_plan"], weekly_views=cfg["weekly_views"], weekly_plan=cfg.get("weekly_plan"),
                    multi_output=cfg.get("multi_output"), prior_candidates=cfg.get("prior_candidates")),
        evaluator_capabilities=sorted(cfg.get("evaluator_capabilities", [])),
        dag=dag, destinations=dict(ledger="state/ledger.sqlite3", status="status/STATUS.json",
                                   collect="collect", archive="archive"),
        swarm_contexts={})
    payload["swarm_contexts"] = {h: swarm_context(payload, int(h), frozen) for h, frozen in frozen_ready.items()}
    hosts = {}
    for entry in cfg["hosts"]:
        hosts[entry["role"]] = _resolve_host(entry)
    private = dict(root=str(root), hosts=hosts, ledger=cfg["ledger"], lease=cfg["lease"], doin=cfg["doin"],
                   launcher=cfg.get("launcher"), shared_data_root=cfg["shared_data_root"],
                   freeze_caps=cfg.get("freeze_caps"), tick_interval_seconds=cfg.get("tick_interval_seconds", 300),
                   local_roles=cfg.get("local_roles"))
    if cfg["mode"] == "production" and not cfg.get("launcher"):
        raise ValueError("production requires the admitted launcher (crispdm-run)")
    for sub in ("state", "status", "collect", "archive"):
        (root / sub).mkdir(parents=True)
    private_dir = root / "private"
    private_dir.mkdir(mode=0o700)
    private_digests = {}
    for role, host in hosts.items():
        if role == "coordinator" and not host.get("devices"):
            continue
        devices = host.get("devices", [])
        if not devices:
            continue
        directory = private_dir / "hosts" / role
        directory.mkdir(parents=True, mode=0o700)
        node = bridge_node_config(payload, private, role, devices[0])
        common.atomic_json(directory / "node.json", node, mode=0o600)
        common.atomic_json(directory / "doin_node.json",
                           doin_node_config(payload, private, role, payload["swarm_contexts"]), mode=0o600)
        private_digests[role] = common.file_sha256(directory / "node.json")
    common.atomic_json(private_dir / "campaign.json", dict(private, config_sha256=common.file_sha256(config_path)),
                       mode=0o600)
    payload["hosts"] = [dict(role=role, ledger_host=bool(h.get("ledger_host")),
                             device_classes=[d["cls"] for d in h.get("devices", [])],
                             node_config_sha256=private_digests.get(role)) for role, h in sorted(hosts.items())]
    common.atomic_json(root / "PLAN.json", dict(payload=payload, sha256=common.sha256_json(payload)))
    return payload


def load(root):
    envelope = json.loads((Path(root) / "PLAN.json").read_text())
    if common.sha256_json(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("campaign plan seal mismatch")
    return envelope


def private(root):
    return json.loads((Path(root) / "private" / "campaign.json").read_text())


def e_plan(root):
    payload = load(root)["payload"]
    plan = common.read_pinned(payload["inputs"]["search_plan"], "E search plan")
    if plan["plan_sha256"] != payload["e_plan_sha256"]:
        raise ValueError("E plan changed after sealing")
    # Scoring view: E's aggregator iterates plan["folds"]["fold_ids"]; receipts are keyed by the campaign
    # (C) fold ids, which may carry a declared clock assumption suffix. Same folds, same order.
    return dict(plan, folds=dict(plan["folds"], fold_ids=list(payload["folds"]), e_fold_map=payload["fold_map"]))


def is_private_ok(path):
    mode = os.stat(path).st_mode
    return not mode & 0o077

