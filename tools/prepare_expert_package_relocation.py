#!/usr/bin/env python3
"""Admit only the unfinished arm on an explicitly authorized alternate GPU."""
import argparse
from pathlib import Path
import subprocess

from expert_package_worker import (atomic, digest, identity, initialize, process_identity,
                                   read, runtime_versions)


def validate_source(root, versions):
    root = Path(root)
    source = root / "source"
    cfg, close, states = [read(source / n) for n in ("CONFIG.private.json", "CLOSE.json", "JOBS.json")]
    if close["state"] != "COMPLETE_WITH_FAILURES" or close["config_sha256"] != identity(cfg):
        raise ValueError("SOURCE_IDENTITY_CHANGED")
    if states["cons_shortlist"]["state"] != "FAILED":
        raise ValueError("SOURCE_NOT_FAILED_ARM")
    if versions != cfg["runtime_versions"]:
        raise ValueError("RELOCATION_RUNTIME_CHANGED")
    verified = 0
    for name, expected in cfg["pins"].items():
        rel = Path(name).relative_to(cfg["root"])
        if rel.parts[0] not in ("code", "data", "in", "hs_code_final.tar.gz", "gamma_queue_musashi.sh"):
            continue
        if digest(root / rel) != expected:
            raise ValueError("RELOCATION_INPUT_CHANGED:" + str(rel))
        verified += 1
    if not verified:
        raise ValueError("RELOCATION_INPUT_PINS_EMPTY")
    return {"parent_config_sha256": identity(cfg), "parent_close_sha256": digest(source / "CLOSE.json"),
            "verified_input_files": verified, "only_job": "cons_shortlist",
            "science_flags_unchanged": True, "hardware_changed": True,
            "cross_hardware_bitwise_equivalence_claimed": False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--launcher", required=True)
    p.add_argument("--gpu-model", required=True)
    p.add_argument("--memory", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--desktop-pid", type=int)
    a = p.parse_args()
    root = Path(a.root).resolve()
    provenance = validate_source(root, runtime_versions(a.python))
    background = {}
    if a.desktop_pid:
        cmd = (Path("/proc") / str(a.desktop_pid) / "cmdline").read_bytes().split(b"\0")[0]
        if Path(cmd.decode()).name != "snapd-desktop-integration":
            raise ValueError("NOT_A_DESKTOP_PROCESS")
        background[str(a.desktop_pid)] = {**process_identity(a.desktop_pid), "max_memory_mib": 32}
    stem = "musashi-e3m-relocated-20261022"
    cfg = initialize(root, a.python, a.launcher, only_jobs=["cons_shortlist"], memory=a.memory,
                     timer_unit=stem + ".timer", recovery_source=provenance,
                     gpu_model=a.gpu_model, gpu_background_processes=background)
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    for suffix in ("service", "timer"):
        text = (Path(a.templates) / ("expert-package-worker." + suffix)).read_text()
        text = text.replace("/var/tmp/expert-guided-musashi-20261022", str(root))
        text = text.replace("musashi-e3m-assigned-20261022", stem)
        (units / (stem + "." + suffix)).write_text(text)
    atomic(root / "ops/RELOCATION.json", provenance)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", stem + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", stem + ".service"], check=True)
    print({"config_sha256": identity(cfg), "memory": a.memory, **provenance})


if __name__ == "__main__":
    main()
