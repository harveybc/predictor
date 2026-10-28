#!/usr/bin/env python3
"""Install the finite, pinned T2 diagnostics with the existing operations worker."""
import argparse
from pathlib import Path
import subprocess

from expert_package_worker import atomic, digest, initialize, identity, read, t2_jobs


ARCHIVE_T2 = "2ee7094c6ec0b934d0715e4642bcdac4d67f65ec76c91d4b2fee9a52973f2083"
STORE = "32edb4248d659661764ee5abf982a7ee9a0e7659e08a88c03b5a28ce2466ae67"
MANIFEST = "10658d6004175379787a4f13d9d527274c495156324a1d33b6895bc4c48cf3f4"
LISTS = {"core15": (15, "86f8dddfb8057a683b1371d8d973271c19c29816cd85ecf4a0db3c9058398c0d"),
         "shortlist_strict": (116, "902622d7e47377034ac131d84b08fae9e497105f47aa29d1a1fec1b4099276b1")}


def prepare(root, python, launcher, templates):
    root = Path(root).resolve()
    peer = read(root / "in/T2_UNITS.json")
    t2_jobs(peer)
    if digest(root / "hs_code_t2.tar.gz") != ARCHIVE_T2:
        raise ValueError("T2_ARCHIVE_CHANGED")
    pool = root / "data/full_pool_s"
    if digest(pool / "MANIFEST.json") != MANIFEST or digest(pool / "full_pool_eurusd_h1.parquet") != STORE:
        raise ValueError("T2_STORE_CHANGED")
    for name, (count, expected) in LISTS.items():
        path = root / "in" / ("FEATURE_LIST_" + name + ".json")
        if digest(path) != expected or len(read(path)["features"]) != count:
            raise ValueError("T2_LIST_CHANGED:" + name)
    for alias, target in (("hs_code_final.tar.gz", "hs_code_t2.tar.gz"),
                          ("gamma_queue_musashi.sh", "gamma_queue_t2.sh")):
        path = root / alias
        if path.exists():
            if path.resolve() != (root / target).resolve():
                raise ValueError("T2_ALIAS_DIFFERS")
        else:
            path.symlink_to(target)
    stem = "musashi-e3m-t2-20261023"
    cfg = initialize(root, python, launcher, memory="4G", timer_unit=stem + ".timer",
                     archive_sha256=ARCHIVE_T2, campaign="v4_t2", t2_manifest=peer,
                     recovery_source={"science_commit": "186b131", "peer_units_commit": "58b2c5e1",
                                      "legacy_label_contract": "zigzag-only in pinned science; labels.mode is not emitted by this native schema",
                                      "new_opportunity_labels": False})
    units = Path.home() / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)
    for suffix in ("service", "timer"):
        text = (Path(templates) / ("expert-package-worker." + suffix)).read_text()
        text = text.replace("/var/tmp/expert-guided-musashi-20261022", str(root))
        text = text.replace("musashi-e3m-assigned-20261022", stem)
        (units / (stem + "." + suffix)).write_text(text)
    atomic(root / "ops/T2_DEPLOYMENT.json", {"config_sha256": identity(cfg),
           "units": len(t2_jobs(peer)), "science_commit": "186b131", "new_SAC_fits": 0})
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", stem + ".timer"], check=True)
    subprocess.run(["systemctl", "--user", "start", stem + ".service"], check=True)
    print({"config_sha256": identity(cfg), "units": 8, "memory": cfg["memory"], "gpu_uuid": cfg["gpu_uuid"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--python", required=True)
    p.add_argument("--launcher", required=True)
    p.add_argument("--templates", required=True)
    a = p.parse_args()
    prepare(a.root, a.python, a.launcher, a.templates)
