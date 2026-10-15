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
                          prior_roots=args.prior_root or (), long_design=args.long_design,
                          strategy_execution=_load_json(getattr(args, "strategy_execution", None)),
                          mode=getattr(args, "mode", "ELAPSED_STATIC"),
                          completed_short_fit_source=getattr(args, "completed_short_fit_source", None))
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
    if args.render_to:
        _emit(deploy.deploy(args.root, unit_dir=args.render_to, render_only=True))
        return 0
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
    from tools.i6dmo import services, status
    probe = services.systemd_probe if getattr(args, "probe_systemd", False) else None
    doc = status.compute(args.root, level=args.level, unit=args.unit, systemd_probe=probe,
                         arm=getattr(args, "arm", None))
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


MATCHED_HELP = """Matched short contrast (NEXT_CONTRASTS_PLAN_20261015): MG1 (25 slots) vs MG2m (13 slots).

prepare   resolve the private deployment, the frozen common population (storage root + committed
          POPULATION_SEAL.json digest), the retained input and the warehouse EnvironmentFile BY PATH
          (never opened); write host-specific private JSON outside git
validate  re-verify a prepared JSON (designs, seal, input, population digest); launches nothing
preflight read-only per-role probe: disk, pinned files, GPU admission and the ACTUAL TF/Keras of
          every GPU worker against the sealed runtime pin (typed RUNTIME_PIN_MISMATCH)
init      seal both arms and enumerate the finite graph mg1/mg2m fit -> report -> comparison
status    campaign / arm / unit status (never launches or alters a fit)
Deploy with the existing `deploy` command (`--render-to` renders only)."""


def cmd_matched(args):
    from tools.i6dmo import campaign as C
    from tools.i6dmo import matched_runner as MR
    try:
        if args.matched_command == "prepare":
            designs = {a: d for a, d in (("mg1", args.mg1_design), ("mg2m", args.mg2m_design)) if d}
            doc = MR.prepare(deployment=args.deployment, input_dir=args.input_dir, out=args.out, role=args.role,
                             designs=designs or None, population_seal=args.population_seal,
                             population_dir=args.population_dir, prior_roots=args.prior_root or (),
                             test=args.test)
            _emit(dict(state="PREPARED", out=str(args.out),
                       designs={a: d["seal_sha256"] for a, d in doc["designs"].items()},
                       population_sha256=doc["data_pins"]["common_population"]["population_sha256"],
                       pins=sorted(doc["data_pins"]), warehouse_environment_file=doc["warehouse_environment_file"]))
            return 0
        if args.matched_command == "validate":
            _emit(MR.validate(args.prepared))
            return 0
        if args.matched_command == "preflight":
            result = MR.preflight(args.prepared, args.out_dir)
            _emit(dict(state=result["state"], launch_blockers=result["launch_blockers"],
                       runtime_pin=result["runtime_pin"], out=str(Path(args.out_dir) / "PRECHECK.matched.json")))
            return 0 if result["state"] == "PASS" else 3
        if args.matched_command == "init":
            MR.validate(args.prepared)
            doc = json.loads(Path(args.prepared).read_text())
            _emit(C.init_matched_contrast({a: d["path"] for a, d in doc["designs"].items()}, doc["deployment"],
                                          args.root, code_commit=args.code_commit, data_pins=doc["data_pins"],
                                          prior_roots=doc["prior_roots"], entrypoints=_load_json(args.entrypoints)))
            return 0
        if args.matched_command == "status":
            from tools.i6dmo import status
            _emit(status.compute(args.root, level=args.level, unit=args.unit, arm=args.arm))
            return 0
    except MR.PrepareRefused as exc:
        _emit(dict(state="REFUSED", cause=str(exc)))
        return 2
    raise SystemExit("unknown matched-short command")


def _add_matched(sub):
    p = sub.add_parser("matched-short", help="matched short MG1/MG2m contrast (prepare, validate, preflight, init, "
                                             "status)", description=MATCHED_HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    msub = p.add_subparsers(dest="matched_command", required=True)
    q = msub.add_parser("prepare", help="resolve pins by path and write the private prepared JSON")
    q.add_argument("--deployment", required=True, help="existing private deployment JSON (never committed)")
    q.add_argument("--input-dir", required=True, help="retained INPUT.npz directory under the role's data_root")
    q.add_argument("--out", required=True, help="private output JSON (refused inside a git checkout)")
    q.add_argument("--role", default="coordinator", help="role whose data_root/run_root/storage root are resolved")
    q.add_argument("--mg1-design", help="default: the committed matched_short/EXPERIMENT.mg1.json")
    q.add_argument("--mg2m-design", help="default: the committed matched_short/EXPERIMENT.mg2m.json")
    q.add_argument("--population-seal", help="default: the committed matched_short/POPULATION_SEAL.json")
    q.add_argument("--population-dir", help="default: the seal's private_location under the storage root")
    q.add_argument("--prior-root", action="append", help="a prior matched root whose verified terminals may be adopted")
    q.add_argument("--test", action="store_true", help="request TEST (always refused: this contrast has no TEST)")
    q.add_argument("--json", action="store_true")
    q = msub.add_parser("validate", help="re-verify a prepared JSON; launches nothing")
    q.add_argument("--prepared", required=True)
    q.add_argument("--json", action="store_true")
    q = msub.add_parser("preflight", help="read-only per-role probe incl. the TF/Keras runtime pin")
    q.add_argument("--prepared", required=True)
    q.add_argument("--out-dir", required=True, help="private directory for PRECHECK.matched.json")
    q.add_argument("--json", action="store_true")
    q = msub.add_parser("init", help="seal both arms and enumerate the finite contrast graph")
    q.add_argument("--prepared", required=True)
    q.add_argument("--root", required=True, help="NEW campaign root on ext4")
    q.add_argument("--code-commit", help="runtime commit (default: git HEAD of this checkout)")
    q.add_argument("--entrypoints", help="optional JSON {kind: argv} (fixtures only; default: unit_runner per kind)")
    q.add_argument("--json", action="store_true")
    q = msub.add_parser("status", help="campaign / arm / unit status; never launches or alters a fit")
    q.add_argument("--root", required=True)
    q.add_argument("--level", default="campaign", choices=("campaign", "arm", "unit", "all"))
    q.add_argument("--arm", help="mg1 | mg2m (with --level arm)")
    q.add_argument("--unit", help="e.g. mg1.fit (with --level unit)")
    q.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_matched)


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
    p.add_argument("--mode", default="ELAPSED_STATIC", choices=("ELAPSED_STATIC", "MARKET_TIME_PAIR"),
                   help="MARKET_TIME_PAIR seals MG2/MG12 up front and enumerates no TEST stage")
    p.add_argument("--strategy-execution", help="JSON with the versioned instrument configuration")
    p.add_argument("--prior-root", action="append")
    p.add_argument("--completed-short-fit-source", help="authenticate and reuse a completed short fit after a reporting-only repair")
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
    p.add_argument("--no-start", action="store_true", help="write unit files and daemon-reload; enable nothing")
    p.add_argument("--render-to", help="render unit files into this directory; no systemctl call at all")
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
    p.add_argument("--level", default="campaign",
                   choices=("campaign", "arm", "unit", "stage", "publication", "fleet", "all"))
    p.add_argument("--unit")
    p.add_argument("--arm", help="with --level arm: one matched arm (mg1|mg2); default every arm")
    p.add_argument("--probe-systemd", action="store_true",
                   help="one bounded `systemctl --user show` reading of the campaign units (failed unit -> typed)")
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

    _add_matched(sub)

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
