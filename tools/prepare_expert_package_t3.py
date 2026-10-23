#!/usr/bin/env python3
"""Verify the peer's T3 v2 package and install its finite sequential worker."""
import argparse
from pathlib import Path
import subprocess

from expert_package_t3 import jobs
from expert_package_worker import atomic, digest, identity, initialize, read


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


def prepare(root, python, launcher, templates):
    root = Path(root).resolve()
    manifest, assignment = verify_package(root)
    path = root / "gamma_queue_musashi.sh"
    if not path.exists():
        path.symlink_to("t3_queue_musashi.sh")
    elif path.resolve() != (root / "t3_queue_musashi.sh").resolve():
        raise ValueError("T3_QUEUE_ALIAS_CHANGED")
    stem = "musashi-e3m-t3-20261023"
    cfg = initialize(root, python, launcher, memory="4G", timer_unit=stem + ".timer",
                     archive_sha256=ARCHIVE, campaign="v4_t3", t3_manifest=manifest, t3_units=assignment,
                     recovery_source={"science_commit": SCIENCE, "peer_package_commit": "8d6dbd5b",
                                      "manifest": manifest["identity"], "new_SAC_fits": 0})
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    for suffix in ("service", "timer"):
        text = (Path(templates) / ("expert-package-worker." + suffix)).read_text()
        text = text.replace("/var/tmp/expert-guided-musashi-20261022", str(root))
        text = text.replace("musashi-e3m-assigned-20261022", stem)
        (units / (stem + "." + suffix)).write_text(text)
    atomic(root / "ops/T3_DEPLOYMENT.json", {"science_commit": SCIENCE, "config_sha256": identity(cfg),
           "units": 21, "manifest": manifest["identity"], "memory_GiB": 4, "new_SAC_fits": 0})
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", stem + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", stem + ".service"], check=True)
    print({"config_sha256": identity(cfg), "units": 21, "manifest": manifest["identity"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--launcher", required=True)
    p.add_argument("--templates", required=True)
    a = p.parse_args()
    prepare(a.root, a.python, a.launcher, a.templates)
