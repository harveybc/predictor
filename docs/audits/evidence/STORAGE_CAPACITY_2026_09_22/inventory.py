"""Read filesystem metadata and small run records, never prediction/data values.

Run locally or via ssh python3 - < inventory.py. Output is a point-in-time
inventory, not an authorization to delete a file or a content-verification proof.
"""

import collections
import datetime
import json
import os
from pathlib import Path


def measure():
    home = Path.home()
    fs = os.statvfs(home)
    root = home / ".local/state/crispdm-data-foundation"
    result = {
        "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "home_fs": {"capacity_bytes": fs.f_blocks * fs.f_frsize,
                    "available_bytes": fs.f_bavail * fs.f_frsize},
        "scope": "SOTA run artifacts only; metadata scan; no prediction values or content hashes read",
        "roots": [], "errors": [],
    }
    seen = set()
    for run in sorted(root.glob("sota_timefilter_ecl*")):
        if not run.is_dir() or run.is_symlink():
            continue
        tag = run.name.replace("omega", "coordinator").replace("dragon", "worker_a").replace("gamma", "worker_b")
        groups = collections.defaultdict(lambda: {"files": 0, "logical_bytes": 0, "allocated_unique_bytes": 0})
        records = []
        linked = 0
        for folder, dirs, files in os.walk(run, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(folder) / d).is_symlink()]
            for name in files:
                path = Path(folder) / name
                try:
                    st = path.lstat()
                    if not path.is_file() or path.is_symlink():
                        continue
                    if name == "arrays.npz":
                        group = "prediction_arrays_candidates_not_yet_deletable"
                    elif name == "METRICS_VAULT.json":
                        group = "metrics_vault"
                    elif name.endswith((".pth", ".pt", ".keras", ".h5")):
                        group = "checkpoints_including_working_duplicates"
                    elif name.startswith("BENCH_DATA"):
                        group = "shared_preparation"
                    else:
                        group = "other_artifacts"
                    g = groups[group]
                    g["files"] += 1
                    g["logical_bytes"] += st.st_size
                    inode = (st.st_dev, st.st_ino)
                    if inode not in seen:
                        g["allocated_unique_bytes"] += st.st_blocks * 512
                        seen.add(inode)
                    linked += st.st_nlink > 1
                    if name == "cell.json" and st.st_size < 10 * 1024**2:
                        c = json.loads(path.read_text())
                        records.append({"cell_id": c.get("cell", {}).get("cell_id"),
                                        "shape": c.get("shapes", {}).get("pred"),
                                        "dtype": c.get("dtype"),
                                        "parameters": c.get("n_parameters"),
                                        "checkpoint_bytes": c.get("checkpoint_bytes")})
                except (OSError, ValueError) as exc:
                    result["errors"].append({"root": tag, "file": name, "error": type(exc).__name__})
        result["roots"].append({"root": tag, "groups": dict(groups), "records": records,
                                "hardlinked_file_entries": linked})
    return result


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
