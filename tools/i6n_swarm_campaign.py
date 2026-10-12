"""I6-N swarm weekly campaign CLI (orders section 3).

    python -m tools.i6n_swarm_campaign plan      --config <private.json> --output <campaign-root>
    python -m tools.i6n_swarm_campaign preflight --root <campaign-root> --json
    python -m tools.i6n_swarm_campaign install   --root <campaign-root> [--role R ...]
    python -m tools.i6n_swarm_campaign start     --root <campaign-root> [--role R ...]
    python -m tools.i6n_swarm_campaign status    --root <campaign-root> --json
    python -m tools.i6n_swarm_campaign collect   --root <campaign-root>

``tick`` (internal) is what the coordinator timer runs. plan/preflight/status/
collect never train; status only reads; install writes only this campaign's
user units; start is idempotent. Modules are imported lazily so ``status``
never loads the runner, the evaluator or TensorFlow.
"""
import argparse
import json
import sys


def _roles(args, default):
    roles = args.role or default
    return [] if roles == ["none"] else roles  # "--role none": ledger/DAG only, no systemd units


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tools.i6n_swarm_campaign", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    for name in ("preflight", "status"):
        q = sub.add_parser(name)
        q.add_argument("--root", required=True)
        q.add_argument("--json", action="store_true")
    for name in ("install", "start"):
        q = sub.add_parser(name)
        q.add_argument("--root", required=True)
        q.add_argument("--role", action="append", help="host role(s) of THIS machine; default: from private config")
    q = sub.add_parser("collect")
    q.add_argument("--root", required=True)
    q = sub.add_parser("tick")
    q.add_argument("--root", required=True)
    args = parser.parse_args(argv)

    if args.command == "status":
        from tools.i6n_swarm import status
        result = status.compute(args.root)
    elif args.command == "plan":
        from tools.i6n_swarm import plan
        payload = plan.build(args.config, args.output)
        result = dict(state="PLANNED", campaign_id=payload["campaign_id"], dag_nodes=len(payload["dag"]),
                      e_plan_sha256=payload["e_plan_sha256"], budget=payload["budget"])
    elif args.command == "preflight":
        from tools.i6n_swarm import coordinator
        result = coordinator.preflight(args.root)
    elif args.command == "install":
        from tools.i6n_swarm import plan, units
        roles = _roles(args, plan.private(args.root).get("local_roles") or ["coordinator"])
        result = units.install(args.root, roles)
    elif args.command == "start":
        from tools.i6n_swarm import coordinator, plan
        roles = _roles(args, plan.private(args.root).get("local_roles") or ["coordinator"])
        result = coordinator.start(args.root, roles=roles)
    elif args.command == "collect":
        from tools.i6n_swarm import collect
        result = collect.collect(args.root)
    else:
        from tools.i6n_swarm import coordinator
        result = coordinator.tick(args.root)
    print(json.dumps(result, indent=None if getattr(args, "json", False) else 1, default=str, allow_nan=False))
    if args.command == "preflight" and result["state"] != "READY":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
