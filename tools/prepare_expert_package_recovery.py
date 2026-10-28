#!/usr/bin/env python3
"""Create a separate resource-repair queue from retained pressure incidents."""
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import subprocess

from expert_package_worker import atomic, digest, identity, initialize, read, verify_pins


def prepare(parent, target):
    parent, target = Path(parent).resolve(), Path(target).resolve()
    cfg, close = read(parent / "ops/CONFIG.private.json"), read(parent / "ops/CLOSE.json")
    verify_pins(cfg)
    if close["state"] != "COMPLETE_WITH_FAILURES" or close["config_sha256"] != identity(cfg):
        raise ValueError("PARENT_NOT_A_CLOSED_FAILED_BATCH")
    states = read(parent / "ops/JOBS.json")
    selected = [n for n, r in states.items() if r["state"] == "FAILED"]
    if selected != ["cons_shortlist", "cons_shortlist_strict_enc"]:
        raise ValueError("UNEXPECTED_RECOVERY_SCOPE")
    admission = Path(os.environ.get("CRISPDM_ADMISSION_DIR", str(Path.home() / ".local/state/crispdm/admission")))
    evidence = {}
    evidence_files = []
    peak = 0
    for name in selected:
        text = Path(states[name]["log"]).read_text()
        match = re.search(r"crispdm-run:\s+record ([^\s:]+):", text)
        if not match:
            raise ValueError("PRESSURE_EVIDENCE_MISSING")
        incident = admission / "incidents" / (match[1] + ".json")
        retained = admission / "retained" / (match[1] + ".json")
        inc, record = read(incident), read(retained)
        if (inc["exit_cause"] != "PRESSURE_STOP_SUSTAINED_ABOVE_RESPOND"
                or inc["argv_sha256"] != record["lease"]["argv_sha256"]
                or inc["cap_bytes"] != 4 * 1024**3):
            raise ValueError("NOT_AN_ADMITTED_PRESSURE_REPAIR")
        value = record["observed_peak_bytes"]
        if value != inc["observed_tree_peak_bytes"] or not 0 < value <= inc["cap_bytes"]:
            raise ValueError("PEAK_EVIDENCE_MISMATCH")
        peak = max(peak, value)
        evidence[name] = {"incident_sha256": digest(incident), "retained_sha256": digest(retained),
                          "peak_bytes": value, "cap_bytes": inc["cap_bytes"],
                          "cause": inc["exit_cause"]}
        evidence_files.extend((incident, retained))
    memory_mib = math.ceil(peak * 1.25 / 1024**2)
    if not 4096 < memory_mib <= 6144:
        raise ValueError("RESOURCE_REPAIR_OUTSIDE_BOUNDED_RANGE")
    target.mkdir(parents=True, exist_ok=True)
    for path in evidence_files:
        destination = target / "ops/capacity_evidence" / path.parent.name / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for name in ("code", "data", "in", "hs_code_final.tar.gz", "gamma_queue_musashi.sh"):
        link = target / name
        if not link.exists():
            link.symlink_to(parent / name, target_is_directory=(parent / name).is_dir())
        elif link.resolve() != (parent / name).resolve():
            raise ValueError("TARGET_INPUT_DIFFERS")
    provenance = {"parent_config_sha256": identity(cfg), "parent_close_sha256": digest(parent / "ops/CLOSE.json"),
                  "incident_evidence": evidence, "memory_rule": "ceil(1.25*max(retained_tree_peak)/MiB)",
                  "memory_peak_is_truncated_lower_bound": True,
                  "science_flags_unchanged": True, "original_results_untouched": True}
    stem = "musashi-e3m-recovery-20261022"
    new = initialize(target, cfg["python"], cfg["launcher"], only_jobs=selected,
                     memory=str(memory_mib) + "M", timer_unit=stem + ".timer", recovery_source=provenance)
    units = Path.home() / ".config/systemd/user"
    old = "musashi-e3m-assigned-20261022"
    for suffix in ("service", "timer"):
        text = (units / (old + "." + suffix)).read_text()
        text = text.replace(str(parent), str(target)).replace(old, stem)
        (units / (stem + "." + suffix)).write_text(text)
    atomic(target / "ops/CAPACITY_REPAIR.json", provenance)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", stem + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", stem + ".service"], check=True)
    print({"jobs": selected, "memory": new["memory"], "config_sha256": identity(new)})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True)
    p.add_argument("--target", required=True)
    a = p.parse_args()
    prepare(a.parent, a.target)
