#!/usr/bin/env python3
"""Verify the peer's T3 v2 package and install its finite sequential worker."""
import argparse
from pathlib import Path
import re
import subprocess

from expert_package_t3 import jobs
from expert_package_worker import atomic, digest, identity, initialize, process_identity, read, runtime_versions


SCIENCE = "1aaf367923fae467988b1ca87d5aa3c74321141f"
ARCHIVE = "e490aeb154a7d6acb80a8dc9d3a130f5cdb9b7eaef804690cfb466fa887bd80a"
PINS = {"T3_MANIFEST.json": "2ca6648c3c3868d17e5c52719fc0a1c605e7162534b5384c141cd5883df59b32",
        "T3_UNITS_musashi.json": "822816dc692ed724be3a5b63733e580f7018218d8396071b29df80406c668657",
        "PINS.sha256": "d6fd6415c84317d6bd6f2a9ab4f457241d96e74ae6bd8eefb2ea4d6c80546a1a",
        "data/full_pool_s/full_pool_eurusd_h1.parquet": "32edb4248d659661764ee5abf982a7ee9a0e7659e08a88c03b5a28ce2466ae67",
        "data/full_pool_s/MANIFEST.json": "10658d6004175379787a4f13d9d527274c495156324a1d33b6895bc4c48cf3f4",
        "in/FEATURE_LIST_shortlist_strict.json": "902622d7e47377034ac131d84b08fae9e497105f47aa29d1a1fec1b4099276b1"}


def verify_package(root):
    root = Path(root).resolve()
    for name, expected in PINS.items():
        if digest(root / name) != expected:
            raise ValueError("T3_INPUT_CHANGED:" + name)
    for line in (root / "PINS.sha256").read_text().splitlines():
        expected, name = line.split(None, 1)
        path = (root / name.lstrip("*")).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("T3_PEER_PIN_CHANGED:" + name)
    manifest, assignment = read(root / "T3_MANIFEST.json"), read(root / "T3_UNITS_musashi.json")
    planned = jobs(manifest, assignment)
    for job in planned:
        if digest(root / job["groups_file"]) != job["groups_sha256"]:
            raise ValueError("T3_GROUP_CHANGED:" + job["name"])
    if len(read(root / "in/FEATURE_LIST_shortlist_strict.json")["features"]) != 116:
        raise ValueError("T3_FEATURE_COUNT")
    return manifest, assignment


def validate_relocation(root, versions, selected=None):
    root = Path(root)
    source = root / "source/ops"
    cfg = read(source / "CONFIG.private.json")
    stop = read(source / "RELOCATION_STOP.private.json")
    status = read(source / "STATUS.json")
    if (cfg["campaign"] != "v4_t3" or identity(cfg) != stop["parent_config_sha256"]
            or stop["state"] != "STOPPED_FOR_OWNER_REQUESTED_RELOCATION"
            or status["state"] != "STOPPED_FOR_RELOCATION" or stop["remaining_scientific_children"]):
        raise ValueError("T3_SOURCE_NOT_AUTHENTICATED_AND_STOPPED")
    receipts = [read(p) for p in source.glob("*.RESULT.json")]
    completed = {r["job"]["name"] for r in receipts}
    if (len(completed) != stop["prior_status"]["done"] or len(completed) != len(receipts)
            or any(r["config_sha256"] != identity(cfg) for r in receipts)
            or (completed and (selected is None or set(selected) & completed))):
        raise ValueError("T3_RELOCATION_WOULD_REPEAT_COMPLETED_UNITS")
    if versions != cfg["runtime_versions"]:
        raise ValueError("T3_RELOCATION_RUNTIME_CHANGED")
    verified = 0
    for filename, expected in cfg["pins"].items():
        rel = Path(filename).relative_to(cfg["root"])
        if rel.parts[0] not in ("code", "data", "in", "t3", "hs_code_final.tar.gz",
                                "gamma_queue_musashi.sh", "PINS.sha256", "T3_MANIFEST.json",
                                "T3_UNITS_musashi.json"):
            continue
        if digest(root / rel) != expected:
            raise ValueError("T3_RELOCATION_INPUT_CHANGED:" + str(rel))
        verified += 1
    if not verified:
        raise ValueError("T3_RELOCATION_INPUT_PINS_EMPTY")
    return {"parent_config_sha256": identity(cfg), "source_stop_sha256": digest(source / "RELOCATION_STOP.private.json"),
            "source_config_file_sha256": digest(source / "CONFIG.private.json"), "verified_input_files": verified,
            "source_done": len(completed), "source_failed": stop["prior_status"]["failed"], "science_flags_unchanged": True,
            "hardware_changed": True, "cross_hardware_bitwise_equivalence_claimed": False}


def prepare(root, python, launcher, templates, *, relocate=False, gpu_model="5090", desktop_pid=None,
            selected=None, memory="4G", unit_stem=None):
    root = Path(root).resolve()
    manifest, assignment = verify_package(root)
    path = root / "gamma_queue_musashi.sh"
    if not path.exists():
        path.symlink_to("t3_queue_musashi.sh")
    elif path.resolve() != (root / "t3_queue_musashi.sh").resolve():
        raise ValueError("T3_QUEUE_ALIAS_CHANGED")
    provenance = {"science_commit": SCIENCE, "peer_package_commit": "8d6dbd5b",
                  "manifest": manifest["identity"], "new_SAC_fits": 0}
    if relocate:
        provenance.update(validate_relocation(root, runtime_versions(python), selected))
    elif gpu_model != "5090":
        raise ValueError("T3_ALTERNATE_GPU_REQUIRES_RELOCATION")
    background = {}
    if desktop_pid:
        cmd = (Path("/proc") / str(desktop_pid) / "cmdline").read_bytes().split(b"\0")[0]
        if Path(cmd.decode()).name != "snapd-desktop-integration":
            raise ValueError("NOT_A_DESKTOP_PROCESS")
        background[str(desktop_pid)] = {**process_identity(desktop_pid), "max_memory_mib": 32}
    stem = "musashi-e3m-t3-relocated-20261023" if relocate else "musashi-e3m-t3-20261023"
    if unit_stem:
        if not re.fullmatch(r"musashi-e3m-t3-[a-z-]+-20261023", unit_stem):
            raise ValueError("T3_INVALID_UNIT_STEM")
        stem = unit_stem
    if memory not in ("4G", "5G"):
        raise ValueError("T3_UNSUPPORTED_MEMORY_BUDGET")
    cfg = initialize(root, python, launcher, memory=memory, timer_unit=stem + ".timer",
                     archive_sha256=ARCHIVE, campaign="v4_t3", t3_manifest=manifest, t3_units=assignment,
                     recovery_source=provenance, gpu_model=gpu_model, gpu_background_processes=background,
                     only_jobs=selected)
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    for suffix in ("service", "timer"):
        text = (Path(templates) / ("expert-package-worker." + suffix)).read_text()
        text = text.replace("/var/tmp/expert-guided-musashi-20261022", str(root))
        text = text.replace("musashi-e3m-assigned-20261022", stem)
        (units / (stem + "." + suffix)).write_text(text)
    atomic(root / "ops/T3_DEPLOYMENT.json", {"science_commit": SCIENCE, "config_sha256": identity(cfg),
           "units": len(selected) if selected else 21, "manifest": manifest["identity"],
           "memory_GiB": int(memory[:-1]), "new_SAC_fits": 0})
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", stem + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", stem + ".service"], check=True)
    print({"config_sha256": identity(cfg), "units": len(selected) if selected else 21,
           "manifest": manifest["identity"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--launcher", required=True)
    p.add_argument("--templates", required=True)
    p.add_argument("--relocate", action="store_true")
    p.add_argument("--gpu-model", choices=("5090", "4070"), default="5090")
    p.add_argument("--desktop-pid", type=int)
    p.add_argument("--jobs", help="explicit comma-separated subset of the canonical odd units")
    p.add_argument("--memory", default="4G")
    p.add_argument("--unit-stem")
    a = p.parse_args()
    prepare(a.root, a.python, a.launcher, a.templates, relocate=a.relocate,
            gpu_model=a.gpu_model, desktop_pid=a.desktop_pid,
            selected=a.jobs.split(",") if a.jobs else None, memory=a.memory, unit_stem=a.unit_stem)
