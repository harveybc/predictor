"""Bounded isolated smoke: TWO real DOIN nodes, real ledger service, SYNTHETIC evaluators.

No production fit, no data, no TEST. Every evaluation is a synthetic child process
(``SYNTHETIC_EVALUATOR_NOT_A_FIT``) that emits a receipt shaped like lane A's, so
E's real ``score_folds`` ranks it with C-style origins VERIFIED. What is real:
two ``doin_node.UnifiedNode`` processes (separate roles, ports, data dirs,
identities) running the shared wrapper in ``i6n_swarm`` mode with the sealed
E plan; the HTTP lease service over the SQLite ledger; the bridge, runner,
orphan guard and fencing.

Checks: population exchange (both nodes fit candidates of the same generations),
claim exclusion (no fit_id is evaluated twice), terminal adoption (a verified
prior receipt is adopted by the swarm without calling the evaluator), failure
isolation (one candidate's fold fails; the search still terminates with
complete candidates).

    python -m tools.i6n_swarm.smoke --root <new dir> [--timeout 480]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from tools.i6n_swarm import common, ledger as L, plan as P

PREDICTOR = Path(__file__).resolve().parents[2]
EXPERIMENT = "SYNTHETIC-SMOKE"
LABEL = "SYNTHETIC_EVALUATOR_NOT_A_FIT"

SYNTHETIC = r'''
"""SYNTHETIC_EVALUATOR_NOT_A_FIT: smoke-test stand-in for lane A's evaluate; trains nothing."""
import hashlib, json, os, sys
req = json.load(sys.stdin)
ctx, cfg, sup = req["context"], req["config"], req["support"]
def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
gid = hashlib.sha256(json.dumps({"schema": "i6n-search-v1", "config": cfg}, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode("ascii")).hexdigest()
fit = sha(dict(experiment=ctx["experiment"], fold=ctx["fold"], effective_config_sha256=sha(cfg),
               context_sha256=sha(ctx), support_sha256=sha(sup)))
calls = os.environ["SMOKE_CALLS"]
try:
    fd = os.open(os.path.join(calls, fit), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(fd, json.dumps(dict(gid=gid, fold=ctx["fold"], attempt=req.get("attempt"), pid=os.getpid(),
                                 node=os.environ.get("SMOKE_NODE"))).encode())
    os.close(fd)
except FileExistsError:
    open(os.path.join(calls, "DUPLICATE-" + fit + "-" + str(os.getpid())), "w").write("x")
if gid == os.environ.get("SMOKE_FAIL_GENOTYPE") and ctx["fold"] == os.environ.get("SMOKE_FAIL_FOLD"):
    sys.stderr.write("SYNTHETIC failure for isolation check\n")
    sys.exit(3)
factor = 0.80 + int(gid[:6], 16) % 300 / 1000.0
h = cfg["horizon"]
rows = [dict(head_offset_hours=h, head_role="ENDPOINT", comparison=True, split=s.upper(),
             support_sha256=ctx["origins"][s]) for s in ("train", "inner")]
print(json.dumps(dict(status="ok", counts=sup["counts"], naive_sums=sup["naive_sums"],
                      support_digest=sup["support_digest"],
                      error_sums={k: v * factor for k, v in sup["naive_sums"].items()},
                      metadata=dict(synthetic=True, label="SYNTHETIC_EVALUATOR_NOT_A_FIT",
                                    candidate_receipt=dict(schema_version="i6n.candidate_receipt.v1",
                                        experiment=ctx["experiment"], fold=ctx["fold"], effective_config_sha256=sha(cfg),
                                        horizon_hours=h, genotype_canonical_sha256=gid, seed=6102026, rows=rows,
                                        synthetic=True)))))
'''


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def context(e_plan, horizon):
    folds = []
    for i, quarter in enumerate(e_plan["folds"]["quarters"]):
        fold = f"{quarter}_SYN"
        origins = dict(train=common.sha256_json(["origin", fold, "T"]), inner=common.sha256_json(["origin", fold, "I"]))
        support = dict(counts=dict(train=500 + i, inner=40 + i), naive_sums=dict(train=5.0 + i, inner=0.4 + i / 10),
                       support_digest=dict(train=common.sha256_json([fold, "T"]), inner=common.sha256_json([fold, "I"])))
        folds.append(dict(fold=fold, support=support, origins_sha256=origins, binding_sha256=None,
                          context=dict(schema="SYNTHETIC_CONTEXT", label=LABEL, experiment=EXPERIMENT, fold=fold,
                                       origins=origins)))
    return dict(schema=P.CONTEXT_SCHEMA, experiment=EXPERIMENT, plan_sha256=e_plan["plan_sha256"], horizon=horizon,
                dag_node=f"search:h{horizon}", fixture=True, budget_new_candidates=e_plan["search"]["new_candidates_per_horizon"],
                classes=[], folds=folds)


def first_population(e_plan, ctx):
    """Generation 0 exactly as the swarm will create it (E's v2 plugin, plan seed)."""
    from optimizer_plugins.i6n_optimizer import Plugin
    cfg = P.render_doin_config(dict(campaign_id="smoke", node_label="x", port=1, data_dir="/unused", bootstrap_peers=[],
                                    shared_min_peers=1, claim_timeout=600, predictor_root=str(PREDICTOR),
                                    node_config="/unused", search=e_plan["search"], e_plan=e_plan,
                                    admitted_classes=["S1"]), {str(ctx["horizon"]): ctx})["domains"][0]["optimization_config"]
    cfg = {k: v for k, v in cfg.items() if k not in ("predictor_root", "i6n_swarm")}
    return Plugin().create_shared_population(e_plan["search"]["population_size"], cfg, seed=6102026)["population"]


def run(root, timeout, late_join=0.0):
    from tools import i6n_search as search
    from tools.i6n_swarm import bridge as B
    root = Path(root)
    root.mkdir(parents=True)
    e_plan = json.loads((PREDICTOR / "docs/i6n/swarm/I6N_SEARCH_PLAN_v1_2.json").read_text())
    horizon = 12
    ctx = context(e_plan, horizon)
    (root / "synthetic_evaluator.py").write_text(SYNTHETIC)
    calls = root / "calls"
    calls.mkdir()
    token = root / "ledger.token"
    token.write_text(hashlib.sha256(os.urandom(32)).hexdigest())
    token.chmod(0o600)
    ledger_path = root / "ledger.sqlite3"
    led = L.Ledger(ledger_path)
    led.dag_upsert(f"search:h{horizon}", "search", 0, [], {"horizon": horizon})
    population = first_population(e_plan, ctx)
    # Terminal adoption: a verified prior receipt for every fold of the first generation-0 candidate.
    adopted = population[0]
    config = search.effective_config(adopted["genome"])
    (root / "prior-calls").mkdir()
    prior_env = dict(os.environ, SMOKE_CALLS=str(root / "prior-calls"))
    for fold in ctx["folds"]:
        fit = B.fold_fit_id(ctx, fold, config)
        led.ensure_unit(fit, B.unit_spec(ctx, fold, adopted, config))
        request = dict(config=config, generation=0, context=fold["context"], support=fold["support"])
        receipt = json.loads(subprocess.run([sys.executable, str(root / "synthetic_evaluator.py")],
                                            input=json.dumps(request), text=True, capture_output=True, check=True,
                                            env=prior_env).stdout)
        led.register_prior(fit, B.receipt_document(fit, ctx, fold, adopted, config, receipt),
                           source=dict(label="SMOKE_PRIOR_RECEIPT", synthetic=True))
    # Failure isolation: one fold of the second generation-0 candidate fails in its evaluator.
    failing = population[1]
    led.close()
    ledger_port, ports = free_port(), {"A": free_port(), "B": free_port()}
    procs, logs = {}, root / "logs"
    logs.mkdir()
    pythonpath = os.pathsep.join([str(PREDICTOR)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])
    base_env = dict(os.environ, PYTHONPATH=pythonpath, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                    SMOKE_CALLS=str(calls), SMOKE_FAIL_GENOTYPE=failing["identity"],
                    SMOKE_FAIL_FOLD=ctx["folds"][2]["fold"])
    procs["ledger"] = subprocess.Popen([sys.executable, "-m", "tools.i6n_swarm.lease_service", "--ledger", str(ledger_path),
                                        "--bind", "127.0.0.1", "--port", str(ledger_port), "--token-file", str(token)],
                                       env=base_env, stdout=open(logs / "ledger.log", "w"), stderr=subprocess.STDOUT,
                                       cwd=PREDICTOR)
    time.sleep(1.0)
    for role, other in (("A", "B"), ("B", "A")):
        data = root / f"node-{role}"
        data.mkdir()
        node = dict(schema="i6n.swarm.node.v1", campaign_id="smoke", owner=f"smoke-{role}", host_key=f"role-{role}",
                    device=dict(cls="cpu_fixture", cuda_visible_devices="", fixture=True), fixture=True,
                    exclusive_host=True, evaluator_argv=[sys.executable, str(root / "synthetic_evaluator.py")],
                    workdir=str(root), launcher=None, class_caps={}, logs_dir=str(data / "logs"),
                    state_dir=str(data / "state"), shared_data_root=str(root),
                    ledger=dict(url=f"http://127.0.0.1:{ledger_port}", token_file=str(token)),
                    lease=dict(ttl_seconds=30, renew_seconds=2, max_wait_seconds=180, poll_seconds=0.5))
        (data / "node.json").write_text(json.dumps(node))
        # late_join: A starts alone (shared_min_peers=0, as a lone secondary would); B joins later.
        min_peers = 0 if (late_join and role == "A") else 1
        template = dict(campaign_id="smoke", node_label=f"smoke-{role}", port=ports[role], data_dir=str(data / "doin"),
                        bootstrap_peers=[f"127.0.0.1:{ports[other]}"], shared_min_peers=min_peers, claim_timeout=120,
                        predictor_root=str(PREDICTOR), search=e_plan["search"], node_config=str(data / "node.json"),
                        e_plan=e_plan, admitted_classes=["S1"], mode="fixture")
        doin = P.render_doin_config(template, {str(horizon): ctx})
        doin.update(host="127.0.0.1", shared_claim_settle_seconds=0.3, shared_claim_confirmation_rounds=2,
                    shared_peer_wait_timeout=30, discovery_enabled=False, dashboard_enabled=False,
                    fee_market_enabled=False, network_protocol="flooding")
        (data / "doin_node.json").write_text(json.dumps(doin))
        if late_join and role == "B":
            time.sleep(late_join)
        procs[role] = subprocess.Popen([sys.executable, "-m", "tools.i6n_swarm.smoke_node", "--config",
                                        str(data / "doin_node.json")], env=dict(base_env, SMOKE_NODE=role,
                                        I6N_SWARM_NODE_CONFIG=str(data / "node.json")),
                                       stdout=open(logs / f"node-{role}.log", "w"), stderr=subprocess.STDOUT, cwd=PREDICTOR)
    deadline, terminal = time.monotonic() + timeout, None
    try:
        while time.monotonic() < deadline:
            time.sleep(3.0)
            if any(p.poll() is not None for p in procs.values()):
                break
            view = L.Ledger.open_readonly(ledger_path)
            events = view.dag_events(f"search:h{horizon}")
            view.close()
            if events:
                terminal = events[0]
                time.sleep(3.0)
                break
    finally:
        for name, proc in procs.items():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in procs.values():
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    return verify(root, ledger_path, horizon, terminal, adopted, failing, procs, late_join)


def verify(root, ledger_path, horizon, terminal, adopted, failing, procs, late_join=0.0):
    led = L.Ledger.open_readonly(ledger_path)
    units = led.units()
    attempts = led.attempts()
    calls = sorted(p.name for p in (root / "calls").iterdir())
    duplicates = [c for c in calls if c.startswith("DUPLICATE-")]
    physical = {}
    for a in attempts:
        if a["physical"]:
            physical.setdefault(a["fit_id"], []).append(a)
    owners_by_generation = {}
    for a in attempts:
        owners_by_generation.setdefault("all", set()).add(a["owner"])
    adopted_units = [u for u in units if u["genotype"] == adopted["identity"]]
    failing_units = [u for u in units if u["genotype"] == failing["identity"]]
    terminals = led.terminals()
    state = led.search_state(f"search:h{horizon}")
    checks = dict(
        search_terminated=terminal is not None,
        population_exchange=sorted(owners_by_generation.get("all", set())) == ["smoke-A", "smoke-B"],
        claim_exclusion=not duplicates and all(len(v) == 1 for v in physical.values())
        and len(calls) == len(physical),
        terminal_adoption=bool(adopted_units) and all(terminals.get(u["fit_id"], {}).get("source") == "PRIOR"
                                                      for u in adopted_units)
        and not any(u["fit_id"] in physical for u in adopted_units),
        failure_isolation=any(u["state"] == "FAILED" for u in failing_units)
        and sum(1 for u in units if u["state"] == "COMPLETE") > len(adopted_units),
    )
    summary = dict(schema="i6n.swarm.smoke.v1", label=LABEL, synthetic_evaluator=True, production_fit=False,
                   test_opened=False, late_join_seconds=late_join, checks=checks, passed=all(checks.values()),
                   stop=terminal["detail"] if terminal else None,
                   generations_persisted=state["generation"] if state else None,
                   units=len(units), complete=sum(u["state"] == "COMPLETE" for u in units),
                   failed=sum(u["state"] == "FAILED" for u in units), evaluator_calls=len(calls),
                   duplicates=len(duplicates), attempts_by_owner={o: sum(a["owner"] == o for a in attempts)
                                                                  for o in ("smoke-A", "smoke-B")},
                   exit_codes={k: p.returncode for k, p in procs.items()})
    led.close()
    common.atomic_json(root / "SMOKE.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--timeout", type=float, default=480)
    parser.add_argument("--late-join", type=float, default=0.0,
                        help="start node B this many seconds after node A (A runs alone with shared_min_peers=0)")
    args = parser.parse_args(argv)
    result = run(args.root, args.timeout, args.late_join)
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
