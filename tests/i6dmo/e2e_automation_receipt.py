"""One finite end-to-end automation receipt on CPU (recovery handoff 2026-10-15, SC08).

    CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 3G -t 600s -n i6dmo-repair-e2e -- \\
        python -m tests.i6dmo.e2e_automation_receipt --scratch <new dir under /var/tmp> --out <receipt.json>

In a throwaway scratch root (never a live campaign): the real worker/tick path over authenticated
report artifacts (sealed MG2/MG12 designs, real market bundles from the synthetic fixture, real
report writer; the trainer is a prediction fixture that trains nothing) -> pair.eligibility
NEGATIVE without OHLC pins -> strategy.validation typed abstention -> outbox delivered with exact
readback into the repository's local test sink (``FakeWarehouse``, not the production OLAP) ->
two scratch mirrors verified -> automatic CLOSE with closure-input snapshots -> ONLY this
campaign's own timers/services (from its DEPLOY.json, rendered into a scratch unit dir) stopped
through an INJECTED ``systemctl --user`` recorder (the real user systemd is never touched) with a
durable TIMERS_STOPPED.json receipt -> a re-tick, re-publish and worker run change nothing and
call systemctl no more. Paths in the receipt are relative to ``$SCRATCH``.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree(root):
    return {str(p.relative_to(root)): _sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m tests.i6dmo.e2e_automation_receipt")
    parser.add_argument("--scratch", required=True, help="NEW directory under /var/tmp (refused if non-empty)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    scratch = Path(args.scratch).resolve()
    if scratch.exists() and any(scratch.iterdir()):
        print("SCRATCH_NOT_EMPTY", file=sys.stderr)
        return 2
    if not str(scratch).startswith("/var/tmp/"):
        print("SCRATCH_MUST_BE_UNDER_VAR_TMP", file=sys.stderr)
        return 2
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    from tests.i6dmo import ops_real_chain as R
    from tests.i6dmo.ops_fakes import FakeWarehouse
    from tools.i6dmo import campaign, deploy, status

    class Recorder:
        def __init__(self):
            self.calls = []

        def __call__(self, *args):
            self.calls.append(list(args))
            return dict(rc=0, stdout="")

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "tools/i6dmo", "tests/i6dmo"], capture_output=True,
                           text=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip()
    data_root = R.build_market_data(scratch / "data")
    root = R.make_real_chain_campaign(scratch, data_root, name="campaign")
    deploy_rec = Recorder()
    deployed = deploy.deploy(root, unit_dir=scratch / "systemd-units", systemctl=deploy_rec, start=False)
    effective = campaign.load_effective(root)
    assert "strategy_bars" not in effective["data_pins"] and "strategy_calendar" not in effective["data_pins"]
    counter = scratch / "fit_invocations.txt"
    results = R.drive(root, env={"I6DMO_FAKE_COUNTER": str(counter)})
    states = R.states(root)
    q = campaign.open_queue(root)
    try:
        pair = q.terminal("pair.eligibility")
        attempts = {u: [dict(index=a["idx"], outcome=a["outcome"], role=a["role"]) for a in q.attempts(u)]
                    for u in states}
    finally:
        q.close()
    decision_path = root / "terminals" / pair["identity_sha256"] / "out" / "pair" / "ELIGIBILITY.json"
    decision = json.loads(decision_path.read_text())
    stop_rec = Recorder()
    cid = campaign.load_plan(root)["campaign_id"]
    me = f"{deploy.prefix(cid)}publish.service"
    tick_before_publish = campaign.tick(root, systemctl=stop_rec, self_unit=me)
    warehouse = FakeWarehouse()
    published = campaign.publish(root, warehouse=warehouse, systemctl=stop_rec, self_unit=me)
    timers = json.loads((root / "TIMERS_STOPPED.json").read_text())
    stop_calls = [list(c) for c in stop_rec.calls]
    close_path = root / "CLOSE.json"
    close = json.loads(close_path.read_text())
    spec = campaign.load_deployment(root)
    mirrors = {}
    for name, target in (("local", spec["mirrors"]["local"]), ("second_host", spec["mirrors"]["second_host_path"])):
        target = Path(target)
        mirrors[name] = dict(close_identical=(target / "CLOSE.json").read_bytes() == close_path.read_bytes(),
                             snapshots_present=all((target / d["snapshot"]).is_file()
                                                   and _sha(target / d["snapshot"]) == d["sha256"]
                                                   for d in close["closure_inputs"].values()),
                             files=sum(1 for p in target.rglob("*") if p.is_file()))
    mirrors_doc = json.loads((root / "MIRRORS.json").read_text())
    before = _tree(root)
    retick = campaign.tick(root, systemctl=stop_rec, self_unit=me)
    republish = campaign.publish(root, warehouse=warehouse, systemctl=stop_rec, self_unit=me)
    rework = R.worker(root, "coordinator")
    after = _tree(root)
    camp = status.compute(root, level="campaign")
    snapshots_ok = all(_sha(root / d["snapshot"]) == d["sha256"] for d in close["closure_inputs"].values())
    receipt = dict(
        schema="i6dmo.automation_repair_e2e_receipt.v1",
        code_commit=head, worktree_dirty_under_tools_or_tests=bool(dirty),
        scratch="$SCRATCH (throwaway, under /var/tmp; not a live campaign)",
        campaign_id=campaign.load_plan(root)["campaign_id"],
        design_seals=dict(mg2=effective["design"].get("seal_sha256"),
                          mg12=effective["long"]["design"].get("seal_sha256")),
        strategy_pins_present=False,
        unit_states={u: dict(state=s[0], disposition=s[1], cause=s[2]) for u, s in states.items()},
        attempts=attempts,
        fit_invocations=counter.read_text().splitlines(),
        worker_results=[r.get("result") for r in results],
        pair_decision=dict(status=decision["status"], eligible=decision["eligible"], missing=decision["missing"],
                           rule=decision["rule"], artifact_sha256=_sha(decision_path),
                           inputs=decision["inputs"]),
        tick_before_publish=dict(close=tick_before_publish["close"],
                                 close_problems=tick_before_publish["close_problems"]),
        publication={k: published["publication"].get(k) for k in ("state", "expected", "delivered", "pending",
                                                                   "mismatch")},
        sink=dict(kind="tests.i6dmo.ops_fakes.FakeWarehouse (local test sink, exact readback)",
                  reports=len(warehouse.reports), metric_rows=len(warehouse.rows)),
        close=dict(state=close["state"], disposition=close["disposition"], trigger=close["trigger"],
                   sha256=_sha(close_path), publication_sha256=close["publication_sha256"],
                   mirrors_sha256=close["mirrors_sha256"], closure_inputs=close["closure_inputs"],
                   snapshots_verify=snapshots_ok),
        mirrors=dict(state=mirrors_doc["state"], files=mirrors_doc["files"], close_sha256=mirrors_doc["close_sha256"],
                     publication_sha256=mirrors_doc["publication_sha256"], targets=mirrors),
        deployment=dict(rendered_units=deployed["units"], timers=deployed["timers"], enabled_nothing=True,
                        systemctl_calls_during_deploy=[list(c) for c in deploy_rec.calls],
                        unit_dir="$SCRATCH/systemd-units (scratch; never the user's systemd directory)"),
        timer_stop=dict(state=timers["state"], trigger=timers["trigger"], close_sha256=timers["close_sha256"],
                        receipt_sha256=_sha(root / "TIMERS_STOPPED.json"), at=timers["at"],
                        units=timers["units"], systemctl_calls=stop_calls,
                        only_this_campaign=all(c[-1].startswith(deploy.prefix(cid)) for c in stop_calls),
                        mirrored=mirrors_doc.get("timers_sha256") == _sha(root / "TIMERS_STOPPED.json"),
                        runner="INJECTED recorder (tests); production uses systemctl --user"),
        idempotence=dict(retick=retick.get("close"), systemctl_calls_after_close=len(stop_rec.calls) - len(stop_calls), republish=republish.get("result"), worker=rework["result"],
                         files=len(before), tree_unchanged=before == after,
                         tree_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()),
        status=dict(state=camp["state"], failures=camp["failures"], decisions=camp["decisions"],
                    close=camp["close"]["disposition"] if camp["close"] else None),
        test_read=False, wall_seconds=round(time.time() - started, 3))
    ok = (decision["status"] == "PAIR_NOT_ELIGIBLE" and states["strategy.validation"][1] == "PAIR_NOT_ELIGIBLE"
          and published["publication"]["state"] == "DELIVERED" and close["trigger"] == "AUTOMATIC"
          and snapshots_ok and before == after and all(m["close_identical"] and m["snapshots_present"]
                                                       for m in mirrors.values())
          and receipt["fit_invocations"] == ["fit:short.fit", "fit:long.fit"]
          and timers["state"] == "STOPPED" and receipt["timer_stop"]["only_this_campaign"]
          and receipt["timer_stop"]["mirrored"] and len(stop_rec.calls) == len(stop_calls)
          and sorted(c[-1] for c in stop_calls if c[:2] == ["disable", "--now"]) == sorted(deployed["timers"]))
    receipt["verdict"] = "PASS" if ok else "FAIL"
    text = json.dumps(receipt, indent=1, sort_keys=True).replace(str(scratch), "$SCRATCH")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    print(json.dumps(dict(verdict=receipt["verdict"], out=str(args.out), close_sha256=receipt["close"]["sha256"])))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
