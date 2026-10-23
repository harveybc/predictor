"""Pinned T3 command and output contracts; no training or feature decisions."""
import hashlib
import json
from pathlib import Path
import shlex


def jobs(manifest, assignment, *, worker="musashi"):
    if (worker not in ("musashi", "satoshi") or manifest.get("schema") != "t3_manifest.v2"
            or manifest.get("identity") != "t3v2_8875abf84a63"
            or assignment.get("manifest") != manifest["identity"] or assignment.get("worker") != worker):
        raise ValueError("T3_MANIFEST_IDENTITY")
    all_units = manifest["units"]
    if len(all_units) != 43 or [u["id"] for u in all_units] != list(range(43)):
        raise ValueError("T3_MANIFEST_UNITS")
    assigned = assignment["units"]
    if [u["id"] for u in assigned] != list(range(1 if worker == "musashi" else 0, 43, 2)):
        raise ValueError("T3_ASSIGNED_UNITS")
    result = []
    for unit in assigned:
        i = unit["id"]
        declared = all_units[i]
        name = f"t3_unit_{i:02d}"
        if (unit["name"] != name or declared["name"] != name or declared["worker"] != worker
                or any(unit[k] != declared[k] for k in ("identity", "kind", "group"))
                or unit["kind"] not in (("conv",) if i == 0 else ("pointwise", "per_feature"))
                or declared["groups_file"] != f"unit_{i:02d}.json"):
            raise ValueError("T3_UNIT_IDENTITY")
        text = unit["command"]
        if hashlib.sha256(text.encode()).hexdigest()[:16] != unit["command_identity"]:
            raise ValueError("T3_COMMAND_IDENTITY")
        argv = shlex.split(text)
        if argv[:2] != ["tools/expert_guided_e3m.py", "run"] or len(argv[2:]) % 2:
            raise ValueError("T3_COMMAND_SHAPE")
        pairs = list(zip(argv[2::2], argv[3::2]))
        options = dict(pairs)
        expected = {"--admitted": "<root>/data/admitted", "--csv-dir": "<root>/data/sel15",
                    "--store": "<root>/data/full_pool_s/full_pool_eurusd_h1.parquet",
                    "--feature-list": "<root>/in/FEATURE_LIST_shortlist_strict.json",
                    "--groups": f"<root>/t3/groups/unit_{i:02d}.json", "--out": f"<root>/out/{name}",
                    "--variants": "FULL", "--units": "2", "--depth": "2", "--head": "16",
                    "--max-epochs": "1000", "--scaling": "fixed_train", "--nonstationary": "diff",
                    "--split-contract": "consolidated", "--agent-state": "none", "--window": "13",
                    "--stride": "2", "--exec-contract": "sac"}
        numeric = {"--dropout": 0.1, "--wd": 1e-4, "--min-orders-per-week": 2.0}
        if (len(options) != len(pairs) or set(options) != set(expected) | set(numeric)
                or any(options.get(k) != v for k, v in expected.items())
                or any(float(options[k]) != v for k, v in numeric.items())):
            raise ValueError("T3_COMMAND_CONTRACT")
        result.append({"name": name, "unit_id": i, "list": "FEATURE_LIST_shortlist_strict.json",
                       "split": "consolidated", "state": "none", "dropout": .1, "wd": 1e-4,
                       "groups_file": "t3/groups/" + declared["groups_file"],
                       "groups_sha256": declared["groups_file_sha256"],
                       "kind": unit["kind"], "group": unit["group"],
                       "peer_unit_identity": unit["identity"], "peer_command_identity": unit["command_identity"],
                       "peer_command": text})
    return result


def validate_extra(cfg, job, result, features):
    path = Path(cfg["root"]) / job["groups_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != job["groups_sha256"]:
        raise ValueError("T3_GROUP_FILE_CHANGED")
    spec = json.loads(path.read_text())
    tensor = [f.split("@")[0] for f in features]
    if len(tensor) != len(set(tensor)):
        raise ValueError("T3_AMBIGUOUS_TENSOR_BASE_NAMES")
    names, indices, widths, kinds = [], [], [], []
    for name, columns in spec["groups"].items():
        ii = [tensor.index(c.split("@")[0]) for c in columns if c.split("@")[0] in tensor]
        if ii:
            names.append(name)
            indices.append(ii)
            widths.append(int(spec.get("units", {}).get(name, 2)))
            kinds.append(spec.get("kinds", {}).get(name, "conv"))
    if sorted(i for group in indices for i in group) != list(range(len(features))):
        raise ValueError("T3_GROUPS_NOT_A_PARTITION")
    arch = result["arch"]
    if (arch.get("depth") != 2 or arch.get("group_names") != names or arch.get("groups") != indices
            or arch.get("group_units") != widths or arch.get("group_kinds") != kinds
            or any(k not in ("conv", "pointwise") for k in kinds)):
        raise ValueError("T3_GROUP_LAYOUT")
    if result.get("window") != {"observations": 13, "stride_rows": 2}:
        raise ValueError("T3_WINDOW_CONTRACT")
    if result.get("labels") != {"mode": "zigzag"}:
        raise ValueError("T3_LABEL_CONTRACT")
