"""``python -m tools.i6dmo``: the I6-DMO public CLI (ACCEPTANCE.md, handoff section D).

validate-design (model lane, registered through ``tools.i6dmo.design.add_cli``), preflight, probe,
init, seal-long, deploy, render-bootstrap, tick, worker, status, publish, gate, close; the data lane's
commands are registered additively through ``tools.i6dmo.cli_data.register`` when present.
Structured output is JSON with ``--json``. No command prints a credential.
"""
import argparse
import json
import sys
from pathlib import Path


def _emit(value, as_json=True):
    print(json.dumps(value, sort_keys=True, default=str) if as_json else value)


def _fallback_validate(subparsers):
    p = subparsers.add_parser("validate-design", help="pure design validation (model lane unavailable)")
    p.add_argument("--design", required=True)
    p.add_argument("--json", action="store_true")

    def run(args):
        from tools.i6dmo import campaign as C
        design = json.loads(Path(args.design).read_text())
        errors = C._design_checks(design) + ["MODEL_LANE_VALIDATOR_UNAVAILABLE"]
        _emit(dict(schema="i6dmo.validate_design.v1", status="FAILED", code=errors[0], errors=errors, pure=True))
        return 2
    p.set_defaults(func=run)


def _load_json(path):
    return None if path is None else json.loads(Path(path).read_text())


def cmd_preflight(args):
    from tools.i6dmo import preflight
    doc = preflight.preflight(args.design, args.deployment, data_pins=_load_json(args.data_pins), out=args.out)
    _emit(doc)
    return 0 if doc["state"] == "PASS" else 3


def cmd_probe(args):
    from tools.i6dmo import preflight
    _emit(preflight.probe_local(run_root=args.run_root, data_root=args.data_root, files=args.file or (),
                                check_strategy=args.check_strategy))
    return 0


def cmd_init(args):
    from tools.i6dmo import campaign as C
    doc = C.init_campaign(args.design, args.deployment, args.root, code_commit=args.code_commit,
                          data_pins=_load_json(args.data_pins), entrypoints=_load_json(args.entrypoints),
                          prior_roots=args.prior_root or (), long_design=args.long_design)
    _emit(doc)
    return 0


def cmd_seal_long(args):
    from tools.i6dmo import campaign as C
    _emit(dict(sealed=C.seal_long(args.root, args.design)))
    return 0


def cmd_deploy(args):
    from tools.i6dmo import campaign as C
    from tools.i6dmo import deploy
    spec = json.loads(Path(args.deployment).read_text())
    if C.load_deployment(args.root) != spec:
        raise C.Refused("DEPLOYMENT_SPEC_DIFFERS: the root was initialised with another private spec")
    _emit(deploy.deploy(args.root, start=not args.no_start))
    return 0


def cmd_render_bootstrap(args):
    from tools.i6dmo import deploy
    _emit(deploy.render_bootstrap(args.deployment, args.root, args.out))
    return 0


def cmd_tick(args):
    from tools.i6dmo import campaign as C
    _emit(C.tick(args.root))
    return 0


def cmd_worker(args):
    from tools.i6dmo import worker
    if not args.once:
        raise SystemExit("worker requires --once (finite; the timer re-invokes it)")
    out = worker.worker_once(args.root, args.host_role)
    _emit(out)
    return 0 if out["result"] not in ("FAILED",) else 3


def cmd_status(args):
    from tools.i6dmo import status
    doc = status.compute(args.root, level=args.level, unit=args.unit)
    _emit(doc if args.json else json.dumps(doc, indent=1, sort_keys=True, default=str), as_json=args.json)
    return 0


def cmd_publish(args):
    from tools.i6dmo import campaign as C
    _emit(C.publish(args.root, mirror=not args.no_mirror))
    return 0


def cmd_gate(args):
    from tools.i6dmo import campaign as C
    receipt = json.loads(Path(args.receipt).read_text())
    _emit(C.record_gate(args.root, args.name, status=receipt["status"], evidence=receipt.get("evidence", {}),
                        splits_read=receipt["splits_read"], owner=receipt.get("owner")))
    return 0


def cmd_close(args):
    from tools.i6dmo import campaign as C
    doc = C.close_campaign(args.root)
    _emit(doc)
    return 0 if doc["state"] == "CLOSED" else 3


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    try:
        from tools.i6dmo import design
        design.add_cli(sub)
    except ImportError:
        _fallback_validate(sub)
    try:
        from tools.i6dmo import cli_data
        cli_data.register(sub)
    except ImportError:
        pass

    p = sub.add_parser("preflight")
    p.add_argument("--design", required=True)
    p.add_argument("--deployment", required=True)
    p.add_argument("--data-pins")
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("probe", help="read-only host facts (runs on the probed host)")
    p.add_argument("--run-root")
    p.add_argument("--data-root")
    p.add_argument("--file", action="append")
    p.add_argument("--check-strategy", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("init")
    p.add_argument("--design", required=True)
    p.add_argument("--deployment", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--data-pins", help="private JSON {short_bundle: {relpath, sha256}, ...}")
    p.add_argument("--entrypoints", help="optional JSON {kind: argv} overriding the unit_runner defaults")
    p.add_argument("--long-design")
    p.add_argument("--prior-root", action="append")
    p.add_argument("--code-commit")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("seal-long")
    p.add_argument("--root", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_seal_long)

    p = sub.add_parser("deploy")
    p.add_argument("--root", required=True)
    p.add_argument("--deployment", required=True)
    p.add_argument("--no-start", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_deploy)

    p = sub.add_parser("render-bootstrap")
    p.add_argument("--deployment", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_render_bootstrap)

    p = sub.add_parser("tick")
    p.add_argument("--root", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_tick)

    p = sub.add_parser("worker")
    p.add_argument("--root", required=True)
    p.add_argument("--host-role", required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    p.add_argument("--level", default="campaign", choices=("campaign", "unit", "stage", "publication", "fleet", "all"))
    p.add_argument("--unit")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("publish")
    p.add_argument("--root", required=True)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("gate", help="record an external gate receipt (DATA_PROVENANCE, SOFTWARE_TRACEABILITY, ...)")
    p.add_argument("--root", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--receipt", required=True, help="JSON {status, evidence, splits_read, owner}")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("close")
    p.add_argument("--root", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_close)
    return parser


def main(argv=None):
    from tools.i6dmo import campaign as C
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except C.Refused as exc:
        _emit(dict(state="REFUSED", cause=str(exc)))
        return 2


if __name__ == "__main__":
    sys.exit(main())
