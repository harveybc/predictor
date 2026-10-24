"""Pinned T3 command and output contracts; no training or feature decisions."""
import hashlib
import json
from pathlib import Path
import shlex

CONTRACTS = {
    "t3v2_8875abf84a63": (43, "full_pool_s", .1, 1e-4),
    "t3v2_7db1953f99ad": (45, "full_pool_s_a1", .3, .01),
}


def jobs(manifest, assignment, *, worker="musashi"):
    if (worker not in ("musashi", "satoshi") or manifest.get("schema") != "t3_manifest.v2"
            or manifest.get("identity") not in CONTRACTS
            or assignment.get("manifest") != manifest["identity"] or assignment.get("worker") != worker):
        raise ValueError("T3_MANIFEST_IDENTITY")
    count, store, dropout, wd = CONTRACTS[manifest["identity"]]
    all_units = manifest["units"]
    if len(all_units) != count or [u["id"] for u in all_units] != list(range(count)):
        raise ValueError("T3_MANIFEST_UNITS")
    assigned = assignment["units"]
    if [u["id"] for u in assigned] != list(range(1 if worker == "musashi" else 0, count, 2)):
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
                    "--store": f"<root>/data/{store}/full_pool_eurusd_h1.parquet",
                    "--feature-list": "<root>/in/FEATURE_LIST_shortlist_strict.json",
                    "--groups": f"<root>/t3/groups/unit_{i:02d}.json", "--out": f"<root>/out/{name}",
                    "--variants": "FULL", "--units": "2", "--depth": "2", "--head": "16",
                    "--max-epochs": "1000", "--scaling": "fixed_train", "--nonstationary": "diff",
                    "--split-contract": "consolidated", "--agent-state": "none", "--window": "13",
                    "--stride": "2", "--exec-contract": "sac"}
        numeric = {"--dropout": dropout, "--wd": wd, "--min-orders-per-week": 2.0}
        if (len(options) != len(pairs) or set(options) != set(expected) | set(numeric)
                or any(options.get(k) != v for k, v in expected.items())
                or any(float(options[k]) != v for k, v in numeric.items())):
            raise ValueError("T3_COMMAND_CONTRACT")
        result.append({"name": name, "unit_id": i, "list": "FEATURE_LIST_shortlist_strict.json",
                       "split": "consolidated", "state": "none", "dropout": dropout, "wd": wd,
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


def confirmation_job(cfg):
    root = Path(cfg["root"])
    ready_path = root / "in/T3_READY.json"
    if hashlib.sha256(ready_path.read_bytes()).hexdigest() != cfg["confirmation"]["readiness_sha256"]:
        raise ValueError("T3_COMBINATION_READINESS_CHANGED")
    ready = json.loads(ready_path.read_text())
    control_path = root / "in/T3_CONTROL.json"
    count = CONTRACTS[cfg["t3_manifest"]["identity"]][0]
    if (ready.get("state") != "READY_FOR_PER_GROUP_DECISION_AUDIT" or ready.get("count") != count
            or hashlib.sha256(control_path.read_bytes()).hexdigest() != ready["units"]["t3_unit_00"]["result_sha256"]):
        raise ValueError("T3_COMBINATION_CONTROL_OR_READINESS")
    path = root / "in/T3_DECISION.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["confirmation"]["decision_sha256"]:
        raise ValueError("T3_COMBINATION_DECISION_CHANGED")
    decision = json.loads(path.read_text())
    base = json.loads((root / "t3/groups/unit_00.json").read_text())
    combined_path = root / "t3/groups/unit_C.json"
    actual = json.loads(combined_path.read_text())
    rows = decision["groups"]
    if (decision["manifest"] != cfg["t3_manifest"]["identity"]
            or len(rows) != len(base["groups"]) or {r["group"] for r in rows} != set(base["groups"])):
        raise ValueError("T3_COMBINATION_DECISION_GROUPS")
    groups, widths, kinds = dict(base["groups"]), dict(base["units"]), dict(base["kinds"])
    for row in rows:
        name, choice = row["group"], row["chosen"]
        allowed = {u["kind"] for u in cfg["t3_manifest"]["units"] if u["group"] == name} | {"conv"}
        if choice not in allowed:
            raise ValueError("T3_COMBINATION_UNKNOWN_CHOICE")
        if choice == "pointwise":
            kinds[name] = "pointwise"
        elif choice == "per_feature":
            members = groups.pop(name)
            kinds.pop(name)
            for column in members:
                key = name + "/" + column
                groups[key], widths[key], kinds[key] = [column], base["units"].get(name, 2), "conv"
    if (actual.get("name") != "t3_unit_C" or actual.get("groups") != groups
            or actual.get("units") != widths or actual.get("kinds") != kinds):
        raise ValueError("T3_COMBINATION_GROUPS_CHANGED")
    prototype = jobs(cfg["t3_manifest"], cfg["t3_units"])[0]
    argv = shlex.split(prototype["peer_command"])
    argv[argv.index("--groups") + 1] = "<root>/t3/groups/unit_C.json"
    argv[argv.index("--out") + 1] = "<root>/out/t3_unit_C"
    command = shlex.join(argv)
    return {**prototype, "name": "t3_unit_C", "unit_id": "C", "kind": "combination", "group": None,
            "groups_file": "t3/groups/unit_C.json", "groups_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
            "peer_command": command, "peer_command_identity": hashlib.sha256(command.encode()).hexdigest()[:16],
            "peer_unit_identity": cfg["confirmation"]["decision_sha256"]}
