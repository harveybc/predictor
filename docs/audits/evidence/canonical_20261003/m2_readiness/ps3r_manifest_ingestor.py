"""Adopt PS3-R terminals from allowed roots. A feature name is not an allow-list."""

import hashlib
import json
from pathlib import Path


class IngestRefusal(ValueError):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kinds(path):
    counts = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        kind = row.get("row_kind") or row.get("kind")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _utility(path, trained):
    wins = {family: {"Ys": 0, "Yl": 0, "Yb": 0} for family in trained}
    seen = {family: {"Ys": 0, "Yl": 0, "Yb": 0} for family in trained}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if (row.get("row_kind") or row.get("kind")) != "probe_delta":
            continue
        family = row.get("trained")
        if family not in wins:
            continue
        target, loss = row.get("target"), row.get("loss")
        if target == "Y_s" and loss == "mae":
            slot = "Ys"
        elif target == "Y_l" and loss == "mae":
            slot = "Yl"
        elif target == "Y_b" and loss == "log_loss":
            slot = "Yb"
        else:
            continue
        seen[family][slot] += 1
        if row.get("delta_probe_random_minus_trained", 0) > 0:
            wins[family][slot] += 1
    mixed = False
    for family in trained:
        groups = seen[family]
        if groups["Ys"] != 30 or groups["Yl"] != 30 or groups["Yb"] != 10:
            return "UNSCORED"
        scored = wins[family]
        complete = scored["Ys"] == 30 and scored["Yl"] == 30 and scored["Yb"] == 10
        empty = scored["Ys"] == 0 and scored["Yl"] == 0 and scored["Yb"] == 0
        if not complete:
            mixed = True
        if empty:
            mixed = True
    return "mixed" if mixed else "uniform"


def _decision(feature_id, role, disposition, reason, digest=None, utility=None, locator=""):
    return {
        "feature_id": feature_id,
        "role": role,
        "disposition": disposition,
        "reason": reason,
        "results_sha256": digest,
        "utility": utility,
        "locator": locator,
        "ps3r_status": _status(disposition, utility, role),
    }


def _status(disposition, utility, role):
    if disposition != "ADOPTED" or role != "baseline":
        return ""
    if utility == "mixed":
        return "ACCEPTED_PS3R_CELL_MIXED_UTILITY"
    return "ACCEPTED_PS3R_TERMINAL_NOT_SELECTION"


def examine_directory(directory, role, rules):
    """A partial or live directory is NOT_TERMINAL. It is not zero and not complete."""

    directory = Path(directory)
    manifest_path = directory / "run_manifest.json"
    results_path = directory / "results.jsonl"
    feature_id = directory.name
    if not manifest_path.exists() or not results_path.exists():
        return _decision(feature_id, role, "NOT_TERMINAL", "PARTIAL_DIRECTORY")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        return _decision(feature_id, role, "NOT_TERMINAL", "STATUS_" + str(manifest.get("status")))
    file_sha = _sha256(results_path)
    if file_sha != manifest.get("results_sha256"):
        return _decision(feature_id, role, "REJECTED", "RESULTS_HASH", file_sha)
    features = manifest.get("features") or []
    if features != [feature_id]:
        return _decision(feature_id, role, "REJECTED", "FEATURE", file_sha)
    if int(manifest.get("seed", -1)) != int(rules["seed"]):
        return _decision(feature_id, role, "REJECTED", "SEED", file_sha)
    if list(manifest.get("families") or []) != list(rules["families"]):
        return _decision(feature_id, role, "REJECTED", "FAMILY", file_sha)
    if list(manifest.get("folds") or []) != list(rules["folds"]):
        return _decision(feature_id, role, "REJECTED", "FOLDS", file_sha)
    if manifest.get("code_commit") not in set(rules["allowed_revisions"]):
        return _decision(feature_id, role, "REJECTED", "REVISION", file_sha)
    if manifest.get("series_sha256") not in set(rules["allowed_input_digests"]):
        return _decision(feature_id, role, "REJECTED", "INPUT_DIGEST", file_sha)
    kinds = _kinds(results_path)
    if kinds != dict(rules["row_kinds"]):
        return _decision(feature_id, role, "REJECTED", "ROW_KINDS", file_sha)
    trained = [item for item in rules["families"] if item not in {"identity", "random"}]
    utility = _utility(results_path, trained)
    if utility == "UNSCORED":
        return _decision(feature_id, role, "REJECTED", "UNSCORED", file_sha)
    return _decision(feature_id, role, "ADOPTED", "VERIFIED", file_sha, utility, "")


def discover(root, config_path):
    root = Path(root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    found = []
    for role, rules in config.items():
        if rules.get("replaces_baseline") and role != "baseline":
            raise IngestRefusal("ROLE", role)
        for relative in rules["roots"]:
            base = root / relative
            if not base.exists():
                continue
            for child in sorted(path for path in base.iterdir() if path.is_dir()):
                item = examine_directory(child, role, rules)
                if item["disposition"] == "ADOPTED":
                    item["locator"] = str(Path(relative) / child.name / "results.jsonl")
                found.append(item)
    adopted = {}
    blocked = set()
    kept = []
    for item in found:
        if item["disposition"] != "ADOPTED":
            kept.append(item)
            continue
        key = (item["role"], item["feature_id"])
        if key in blocked:
            kept.append(_decision(item["feature_id"], item["role"], "REJECTED", "CONTRADICTORY_TERMINAL"))
            continue
        prior = adopted.get(key)
        if prior is None:
            adopted[key] = item
            continue
        if prior["results_sha256"] != item["results_sha256"]:
            adopted.pop(key)
            blocked.add(key)
            kept.append(_decision(prior["feature_id"], prior["role"], "REJECTED", "CONTRADICTORY_TERMINAL"))
            kept.append(_decision(item["feature_id"], item["role"], "REJECTED", "CONTRADICTORY_TERMINAL"))
            continue
        kept.append(_decision(item["feature_id"], item["role"], "REJECTED", "DUPLICATE_TERMINAL"))
    kept.extend(adopted.values())
    return kept
