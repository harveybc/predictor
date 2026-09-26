"""Persist the small, independent audit probes; no fit, service mutation or broker call."""
import json
from pathlib import Path
import runpy
import subprocess
import hashlib

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    identity = runpy.run_path(str(HERE / "probe_day_20260926.py"))["collect"]()
    lag = runpy.run_path(str(HERE / "probe_lags_20260926.py"))["collect"]()
    gate = HERE.parents[3] / "predictor-q2deep-20260926/tools/df_memory_gated_run.py"
    memory = runpy.run_path(str(HERE / "probe_memory_admission_20260926.py"))["reproduce"](gate)
    output = HERE.parent / "evidence/DAY_REVIEW_2026_09_26/RESULTS.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    kills = []
    for args in (["-k"], ["-u", "systemd-oomd"]):
        lines = subprocess.check_output(["journalctl", "-b", *args, "--since", "2026-09-26 00:00:00",
                                         "--grep", "Killed", "-o", "json", "--no-pager"], text=True)
        for line in lines.splitlines():
            rec = json.loads(line)
            if "Killed process" in rec.get("MESSAGE", "") or "Killed /" in rec.get("MESSAGE", ""):
                kills.append({"journal_timestamp_us": rec["__REALTIME_TIMESTAMP"], "message": rec["MESSAGE"]})
    lts = Path.home() / "Documents/GitHub/lts"
    changed = subprocess.check_output(["git", "diff", "--name-only", "9090f49", "12bce5f"], cwd=lts, text=True)
    dependency = {rev: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{rev}:app/alpaca_paper_lab.py"], cwd=lts)).hexdigest() for rev in ("9090f49", "12bce5f")}
    output.write_text(json.dumps({"huber": identity, "lags": lag, "memory_gate": memory,
                                 "journal_kills": kills, "lts_changed_files": changed.splitlines(),
                                 "lts_timer_dependency_hashes": dependency}, indent=2) + "\n")
    m4 = subprocess.check_output(["prlimit", "--as=536870912", "--cpu=30", "--", "python3", "-B",
                                  str(output.parent / "m4_probe.py")], text=True)
    (output.parent / "M4_PROBE.txt").write_text(m4)
    print(output)
