"""Light supervised execution of one leased unit with measured admission.

TRAIN-only cost fitting is the first real weekly fit, not an extra repetition.
Its exact checkpoint is adopted by the scoring child. A killed/failed fit is
retained as a failed disposition; the supervisor never lowers caps or refits.
"""

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time

from tools.i6m_queue import atomic_json, seal, verify


def run(args):
    job = verify(json.loads(args.job.read_text()))
    root = args.job.parent.parent
    output = root / "cells" / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    terminal = output / "CELL.json"
    if terminal.exists():
        verify(json.loads(terminal.read_text()))
        return
    lock = root / ("GPU.lock" if job["model"] == "NEURAL" else "CPU.lock")
    with lock.open("a") as gate:
        fcntl.flock(gate, fcntl.LOCK_EX)
        cost = root/"cost"/f"{args.role}_n{job['arm']['samples']}.json"
        start = time.monotonic()
        cap = args.pilot_cap
        try:
            def execute(pilot, memory):
                name = "i6m-" + job["lease"][:16] + ("-pilot" if pilot else "-cell")
                marker = output/("STARTED_PILOT.json" if pilot else "STARTED_CELL.json")
                expected = output/("PILOT.json" if pilot else "CELL.json")
                if marker.exists():
                    # An interrupted supervisor may have a surviving admitted
                    # child. Wait for that owner instead of spawning a duplicate.
                    deadline = time.monotonic()+3960
                    while time.monotonic() < deadline:
                        active = subprocess.run(["systemctl", "--user", "list-units", "--type=scope", "--state=active", "--no-legend", "--plain"], capture_output=True, text=True, check=True).stdout
                        if "crispdm-"+name+"-" not in active:
                            break
                        time.sleep(15)
                    if expected.exists():
                        return 0
                    if pilot or not (output/"FIT.json").exists():
                        raise ValueError("interrupted fit retained; automatic refitting forbidden")
                    # The child authenticates FIT.json before score-only reuse.
                else:
                    atomic_json(marker, seal(dict(job_id=job["id"], lease=job["lease"], name=name)))
                command = [str(args.admission), "-m", str(memory), "-t", "3900s", "-n", name,
                           "--", str(args.python), "-m", "tools.i6m_cell", "--job", str(args.job),
                           "--design", str(root/"DESIGN.json"), "--input", str(root/("TRAIN.npz" if pilot else "INPUT.npz")),
                           "--output", str(output)] + (["--pilot"] if pilot else [])
                with (output/("PILOT.log" if pilot else "CELL.log")).open("a") as log:
                    return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

            if job["model"] == "NEURAL" and not cost.exists():
                if job["week"]["ordinal"] != 0:
                    raise ValueError("unmeasured shape requires an unclaimed first-week cost fit")
                if execute(True, cap) != 0:
                    raise ValueError("TRAIN_COST_FAILED; no lower retry")
                pilot = verify(json.loads((output/"PILOT.json").read_text()))
                peak = pilot.get("cgroup_peak_bytes")
                if type(peak) is not int or peak <= 0:
                    raise ValueError("missing own-cgroup pilot peak")
                production = math.ceil(1.25*peak)
                if production > args.maximum_cap:
                    raise ValueError("measured peak with 1.25 margin does not fit; no margin trimming")
                atomic_json(cost, seal(dict(samples=job["arm"]["samples"], role=args.role,
                            measured_peak_bytes=peak, production_cap_bytes=production,
                            pilot_job=job["id"], pilot_sha256=pilot["sha256"],
                            design_sha256=job["design_sha256"], fitting_repeated=False)))
            if job["model"] == "NEURAL":
                record = verify(json.loads(cost.read_text()))
                if record["design_sha256"] != job["design_sha256"]:
                    raise ValueError("cost evidence belongs to another design")
                cap = record["production_cap_bytes"]
            else:
                cap = 1024**3
            rc = execute(False, cap)
            if not terminal.exists():
                raise ValueError(f"child exited {rc} without a terminal (admission/hardware/budget)")
            cell = verify(json.loads(terminal.read_text()))
            peak = cell.get("cgroup_peak_bytes")
            if peak and 1.25*peak > cap:
                # Keep the actual result; increase the NEXT reservation, never refit it.
                required = math.ceil(1.25*peak)
                if job["model"] == "NEURAL" and required <= args.maximum_cap:
                    atomic_json(cost, seal(dict(record, measured_peak_bytes=peak, production_cap_bytes=required,
                                                evidence_cell=cell["sha256"])))
            atomic_json(output/"EXECUTION.json", seal(dict(lease=job["lease"], job_id=job["id"],
                        cap_bytes=cap, exit_code=rc, elapsed_seconds=time.monotonic()-start,
                        cell_sha256=cell["sha256"], pilot_cost_path=str(cost.name))))
        except Exception as exc:
            if not terminal.exists():
                atomic_json(terminal, seal(dict(job_id=job["id"], lease=job["lease"],
                            design_sha256=job["design_sha256"], status="FAILED", model=job["model"],
                            arm=job["arm"], week=job["week"], seed=job["seed"],
                            reason=f"SUPERVISOR: {exc}", metrics=None, elapsed_seconds=time.monotonic()-start,
                            cap_bytes=cap, test_read=False)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--pilot-cap", type=int, default=4*1024**3)
    p.add_argument("--maximum-cap", type=int, default=7936*1024**2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
