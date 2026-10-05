"""Adopt PS3-R terminals from allowed roots. A feature name is not an allow-list."""

import hashlib
import json
import math
from pathlib import Path


class IngestRefusal(ValueError):
    def __init__(self, code, detail):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise IngestRefusal("ROW_TYPE", str(line_number))
        rows.append(row)
    return rows


def _kinds(rows):
    counts = {}
    for row in rows:
        kind = row.get("row_kind") or row.get("kind")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _utility(rows, trained):
    wins = {family: {"Ys": 0, "Yl": 0, "Yb": 0} for family in trained}
    seen = {family: {"Ys": 0, "Yl": 0, "Yb": 0} for family in trained}
    for row in rows:
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


def _probe_contract(rules):
    """Return trusted target domains, rejecting incomplete or ambiguous contracts."""

    raw = rules.get("probe_contract")
    if not isinstance(raw, dict) or not raw:
        return None
    contract = {}
    for target, specification in raw.items():
        if not isinstance(target, str) or not target or not isinstance(specification, dict):
            return None
        horizons = specification.get("horizon_indices")
        metrics = specification.get("metrics")
        delta_metric = specification.get("delta_metric")
        if (
            not isinstance(horizons, list)
            or not horizons
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in horizons)
            or len(horizons) != len(set(horizons))
            or not isinstance(metrics, list)
            or not metrics
            or any(not isinstance(value, str) or not value for value in metrics)
            or len(metrics) != len(set(metrics))
            or delta_metric not in metrics
        ):
            return None
        contract[target] = {
            "horizons": tuple(horizons),
            "metrics": tuple(metrics),
            "delta_metric": delta_metric,
        }

    families = tuple(rules.get("families") or ())
    folds = tuple(rules.get("folds") or ())
    trained = tuple(family for family in families if family not in {"identity", "random"})
    target_horizon_count = sum(len(item["horizons"]) for item in contract.values())
    expected_counts = {
        "probe": len(folds) * len(families) * target_horizon_count,
        "probe_delta": len(folds) * len(trained) * target_horizon_count,
    }
    declared_counts = rules.get("row_kinds") or {}
    if any(declared_counts.get(kind) != count for kind, count in expected_counts.items()):
        return None
    return contract


def _validate_rows(rows, feature_id, rules):
    """Bind retained rows to the authenticated manifest identity."""

    contract = _probe_contract(rules)
    if contract is None:
        return "PROBE_CONTRACT", "missing_or_invalid"
    families = tuple(rules["families"])
    family_set = set(families)
    folds = tuple(rules["folds"])
    fold_set = set(folds)
    seed = int(rules["seed"])
    trained = family_set - {"identity", "random"}
    fold_families = set()
    probe_metric_keys = set()
    delta_keys = set()
    summaries = 0
    summary_families = set()

    for row_number, row in enumerate(rows, 1):
        kind = row.get("row_kind") or row.get("kind")
        if row.get("feature_id") != feature_id:
            return "ROW_FEATURE", str(row_number)
        if "seed" in row:
            try:
                row_seed = int(row["seed"])
            except (TypeError, ValueError):
                return "ROW_SEED", str(row_number)
            if isinstance(row["seed"], bool) or row_seed != seed:
                return "ROW_SEED", str(row_number)

        if kind == "fold_family":
            if "seed" not in row:
                return "ROW_SEED", str(row_number)
            family = row.get("family")
            fold = row.get("fold_id")
            if family not in family_set:
                return "ROW_FAMILY", str(row_number)
            if fold not in fold_set:
                return "ROW_FOLD", str(row_number)
            key = (fold, family)
            if key in fold_families:
                return "ROW_DUPLICATE", str(row_number)
            fold_families.add(key)
        elif kind == "probe":
            family = row.get("representation")
            fold = row.get("fold_id")
            if family not in family_set:
                return "ROW_FAMILY", str(row_number)
            if fold not in fold_set:
                return "ROW_FOLD", str(row_number)
            target = row.get("target")
            horizon = row.get("horizon_index")
            specification = contract.get(target)
            if (
                specification is None
                or isinstance(horizon, bool)
                or horizon not in specification["horizons"]
            ):
                return "ROW_PROBE_COVERAGE", str(row_number)
            for metric in specification["metrics"]:
                value = row.get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    return "ROW_METRIC", f"{row_number}:{metric}"
                key = (fold, family, target, horizon, metric)
                if key in probe_metric_keys:
                    return "ROW_DUPLICATE", str(row_number)
                probe_metric_keys.add(key)
        elif kind == "probe_delta":
            family = row.get("trained")
            fold = row.get("fold_id")
            if family not in trained:
                return "ROW_FAMILY", str(row_number)
            if fold not in fold_set:
                return "ROW_FOLD", str(row_number)
            target = row.get("target")
            horizon = row.get("horizon_index")
            specification = contract.get(target)
            loss = row.get("loss")
            if (
                specification is None
                or isinstance(horizon, bool)
                or horizon not in specification["horizons"]
                or loss != specification["delta_metric"]
            ):
                return "ROW_DELTA_COVERAGE", str(row_number)
            key = (fold, family, target, horizon, loss)
            if key in delta_keys:
                return "ROW_DUPLICATE", str(row_number)
            delta_keys.add(key)
        elif kind == "feature_summary":
            summaries += 1
            for summary in row.get("probe_loss_across_folds") or []:
                representation = summary.get("representation")
                if representation not in family_set:
                    return "ROW_FAMILY", str(row_number)
                summary_families.add(representation)
                try:
                    n_folds = int(summary.get("n_folds", -1))
                except (TypeError, ValueError):
                    return "ROW_FOLD", str(row_number)
                if n_folds != len(folds):
                    return "ROW_FOLD", str(row_number)

    expected_fold_families = {(fold, family) for fold in folds for family in families}
    expected_probe_metric_keys = {
        (fold, family, target, horizon, metric)
        for fold in folds
        for family in families
        for target, specification in contract.items()
        for horizon in specification["horizons"]
        for metric in specification["metrics"]
    }
    expected_delta_keys = {
        (fold, family, target, horizon, specification["delta_metric"])
        for fold in folds
        for family in trained
        for target, specification in contract.items()
        for horizon in specification["horizons"]
    }
    if fold_families != expected_fold_families:
        return "ROW_IDENTITY_COVERAGE", "fold_family"
    if probe_metric_keys != expected_probe_metric_keys:
        return "ROW_PROBE_COVERAGE", "missing_or_extra"
    if delta_keys != expected_delta_keys:
        return "ROW_DELTA_COVERAGE", "missing_or_extra"
    if summaries != 1:
        return "ROW_IDENTITY_COVERAGE", "feature_summary"
    if summary_families != family_set:
        return "ROW_IDENTITY_COVERAGE", "summary_families"
    return None, None


def _decision(
    feature_id,
    role,
    disposition,
    reason,
    digest=None,
    utility=None,
    locator="",
    families=None,
    folds=None,
    seed=None,
    code_commit=None,
    input_digest=None,
):
    return {
        "feature_id": feature_id,
        "role": role,
        "disposition": disposition,
        "reason": reason,
        "results_sha256": digest,
        "utility": utility,
        "locator": locator,
        "families": list(families or []),
        "folds": list(folds or []),
        "seed": seed,
        "code_commit": code_commit,
        "input_digest": input_digest,
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
    expected = rules.get("expected_results_sha256") or {}
    if expected:
        if feature_id not in expected:
            return _decision(feature_id, role, "REJECTED", "UNDECLARED_TERMINAL", file_sha)
        if file_sha != expected[feature_id]:
            return _decision(feature_id, role, "REJECTED", "EXPECTED_RESULTS_HASH", file_sha)
    try:
        rows = _rows(results_path)
    except (json.JSONDecodeError, IngestRefusal) as error:
        return _decision(feature_id, role, "REJECTED", "ROW_PARSE", file_sha)
    kinds = _kinds(rows)
    if kinds != dict(rules["row_kinds"]):
        return _decision(feature_id, role, "REJECTED", "ROW_KINDS", file_sha)
    row_reason, row_detail = _validate_rows(rows, feature_id, rules)
    if row_reason:
        return _decision(feature_id, role, "REJECTED", row_reason, file_sha)
    trained = [item for item in rules["families"] if item not in {"identity", "random"}]
    utility = _utility(rows, trained)
    if utility == "UNSCORED":
        return _decision(feature_id, role, "REJECTED", "UNSCORED", file_sha)
    return _decision(
        feature_id,
        role,
        "ADOPTED",
        "VERIFIED",
        file_sha,
        utility,
        "",
        rules["families"],
        rules["folds"],
        int(rules["seed"]),
        manifest.get("code_commit"),
        manifest.get("series_sha256"),
    )


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
