#!/usr/bin/env python3
"""Admit the pinned v5 T3 package into the existing resumable worker protocol."""
import argparse
from pathlib import Path
import shutil
import subprocess

from expert_package_t3 import jobs
from expert_package_worker import atomic, digest, identity, initialize, read


SCIENCE = "dc8d4aac63ba973f67124c1e0968a123fb39bfa1"
ARCHIVE = "cb903872c8d2d12164b966575715c3396381d2c5fe1571508674ff479649a0a5"
PINS = {
    "PINS.sha256": "c20ddcd7e04da97a955861d9fec76ff5e66d44a4fb40d6d75ba2c25f426efaaa",
    "t3/T3_MANIFEST.json": "35636427e5f9fbfaf11010f8dd8bdfb8aee6e2866e49ad680b82f2cbc3a7716f",
    "t3/T3_UNITS_musashi.json": "27d239c498f2415078861e97890e9701f38e49c7e11039e2dfb44e3257e25fa5",
    "data/full_pool_s_a1/full_pool_eurusd_h1.parquet": "cd0eb8d20c0d0888f7814cda410557a79a927e5cc125e9de5c15dbe073d2f543",
    "data/full_pool_s_a1/MANIFEST.json": "7970e3c4feae0186ea5867063774d2855587e01ee068377e3100ba36b29ce535",
    "in/FEATURE_LIST_shortlist_strict.json": "80219f7cbe257cd33de781890b4737abfaa635787e7b74e081f831f871df8be4",
}


def verify(root):
    root = Path(root).resolve()
    for name, expected in PINS.items():
        if digest(root / name) != expected:
            raise ValueError("V5_PACKAGE_PIN_CHANGED:" + name)
    for line in (root / "PINS.sha256").read_text().splitlines():
        expected, name = line.split(None, 1)
        path = (root / name.lstrip("*")).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise ValueError("V5_PEER_PIN_CHANGED:" + name)
    manifest = read(root / "t3/T3_MANIFEST.json")
    assignment = read(root / "t3/T3_UNITS_musashi.json")
    for job in jobs(manifest, assignment):
        if digest(root / job["groups_file"]) != job["groups_sha256"]:
            raise ValueError("V5_GROUP_CHANGED:" + job["name"])
    if len(read(root / "in/FEATURE_LIST_shortlist_strict.json")["features"]) != 121:
        raise ValueError("V5_FEATURE_COUNT")
    return manifest, assignment


def shard(role):
    if role not in ("fast", "slow"):
        raise ValueError("V5_UNKNOWN_ROLE")
    ids = range(7, 45, 2) if role == "fast" else range(1, 7, 2)
    return [f"t3_unit_{i:02d}" for i in ids]


def prepare(root, python, launcher, templates, role):
    root = Path(root).resolve()
    manifest, assignment = verify(root)
    for name in ("T3_MANIFEST.json", "T3_UNITS_musashi.json"):
        target = root / name
        if target.exists() and digest(target) != digest(root / "t3" / name):
            raise ValueError("V5_ALIAS_CHANGED:" + name)
        shutil.copyfile(root / "t3" / name, target)
    queue = root / "gamma_queue_musashi.sh"
    source = root / "t3/t3_queue_musashi.sh"
    if queue.exists() and digest(queue) != digest(source):
        raise ValueError("V5_QUEUE_ALIAS_CHANGED")
    shutil.copyfile(source, queue)
    stem = f"musashi-e3m-t3-v5-{role}-20261024"
    # v4_t3 denotes the existing worker protocol; scientific identity is the new pinned manifest.
    cfg = initialize(root, python, launcher, memory="5G", timer_unit=stem + ".timer",
        archive_sha256=ARCHIVE, campaign="v4_t3", t3_manifest=manifest, t3_units=assignment,
        only_jobs=shard(role), gpu_model="5090" if role == "fast" else "4070",
        recovery_source={"science_commit": SCIENCE, "selection": "v5", "manifest": manifest["identity"],
                         "peer_package_commit": "6e58f4f0", "role": role, "science_flags_changed": False})
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    for suffix in ("service", "timer"):
        content = (Path(templates) / ("expert-package-worker." + suffix)).read_text()
        content = content.replace("/var/tmp/expert-guided-musashi-20261022", str(root)).replace("musashi-e3m-assigned-20261022", stem)
        (units / (stem + "." + suffix)).write_text(content)
    deployment = {"manifest": manifest["identity"], "config_sha256": identity(cfg), "jobs": shard(role), "memory_GiB": 5}
    atomic(root / "ops/T3_DEPLOYMENT.json", deployment)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", stem + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", stem + ".service"], check=True)
    print(deployment)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "python", "launcher", "templates"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--role", required=True, choices=("fast", "slow"))
    a = p.parse_args()
    prepare(a.root, a.python, a.launcher, a.templates, a.role)
