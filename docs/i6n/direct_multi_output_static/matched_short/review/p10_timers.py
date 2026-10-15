"""P10 (recovery item: TIMERS_STOPPED touches only the campaign's own units). Scratch root, injected
systemctl recorder (the real user systemd is never called). DEPLOY.json is hand-written with the
campaign's own timer/service, its own invoking service, a FOREIGN campaign's timer, a glob and a
non-campaign unit.

Run:
  CUDA_VISIBLE_DEVICES="" crispdm-run -q -m 2G -t 600s -n i6dmo-review-p10 -- \
    python docs/i6n/direct_multi_output_static/matched_short/review/p10_timers.py
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import p6_orchestration as P  # noqa: E402

from tools.i6dmo import deploy  # noqa: E402

WORK = C.SCRATCH / "p10"


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    root, out = P.init(WORK, "root", entries=P.fake_entries())
    cid = out["campaign_id"]
    own = [f"{cid}-tick.timer", f"{cid}-tick.service", f"{cid}-publish.timer", f"{cid}-coordinator-worker.service"]
    foreign = ["i6dmo-9e1ebad89ff2-tick.timer", f"{cid}-*.timer", "crispdm-batch.slice", "dbus.service"]
    (root / "DEPLOY.json").write_text(json.dumps({"schema": "i6dmo.deploy.v1", "campaign_id": cid,
                                                  "units": own + foreign, "timers": [own[0], own[2]]}))
    calls = []

    def rec(*args):
        calls.append(list(args))
        return dict(rc=0, stdout="")
    first = deploy.stop_campaign_units(root, trigger="REVIEW_PROBE", systemctl=rec, self_unit=f"{cid}-tick.service")
    n1 = len(calls)
    second = deploy.stop_campaign_units(root, trigger="REVIEW_PROBE", systemctl=rec, self_unit=f"{cid}-tick.service")
    touched = sorted({c[-1] for c in calls})
    res = {"calls_first_pass": calls[:n1], "receipt_state_first": first["state"], "receipt_failed": first["failed"],
           "actions": [(u.get("unit") if u.get("unit") in own else ("<foreign>" if u.get("unit") else None),
                        u.get("action")) for u in first["units"]],
           "only_own_units_touched": all(t in own for t in touched),
           "foreign_untouched": not any(t in foreign for t in touched),
           "self_service_not_stopped": f"{cid}-tick.service" not in touched,
           "second_pass_retried": second["state"], "calls_second_pass": len(calls) - n1}
    # another campaign's DEPLOY.json in this root is refused wholesale
    (root / "DEPLOY.json").write_text(json.dumps({"schema": "i6dmo.deploy.v1", "campaign_id": "i6dmo-other",
                                                  "units": own}))
    (root / deploy.TIMER_RECEIPT).unlink()
    calls.clear()
    other = deploy.stop_campaign_units(root, trigger="REVIEW_PROBE", systemctl=rec)
    res["other_campaign_record"] = {"state": other["state"], "calls": len(calls)}
    C.write("p10_timers.json", res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
