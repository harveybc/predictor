"""FS-REP follower: representation dispositions for the heavy candidates.

Consumes every authenticated PS3-R terminal (baseline identity/random/ae/dae;
alternatives identity/random/masked_temporal_ae and identity/random/
past_to_current_siamese) as it lands in the coordinator mirror and recomputes,
unattended, the representation disposition table defined by
``docs/audits/evidence/canonical_20261003/fs_closure/fs_rep/DECISION_RULE.md``.

Pure-python aggregation: no TensorFlow, no target/validation/test reads.  Only
the probe rows the pilot already wrote are read.  Roles only: no host names,
IPs or GPU identifiers are written.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

RULE_VERSION = "fs_rep_rule.v2"
SCHEMA_PROGRESS = "fs_rep_progress.v1"
SCHEMA_CONFIG = "fs_rep_config.v1"

FOLDS = ("inner_2019", "inner_2020", "inner_2021", "inner_2022", "inner_2023")
FAMILY_ORDER = ("identity", "random", "ae", "dae", "masked_temporal_ae", "past_to_current_siamese")
TRAINED_ORDER = ("ae", "dae", "masked_temporal_ae", "past_to_current_siamese")
ROLE_FAMILIES = {
    "baseline": ("identity", "random", "ae", "dae"),
    "alt_mtae": ("identity", "random", "masked_temporal_ae"),
    "alt_p2c": ("identity", "random", "past_to_current_siamese"),
}
CELL_FAMILY_TO_ROLE = {
    "baseline": "baseline",
    "masked_temporal_ae": "alt_mtae",
    "past_to_current_siamese": "alt_p2c",
}
PROBE_CONTRACT = {
    "Y_s": {"horizons": (0, 1, 2, 3, 4, 5), "metrics": ("mae", "mse"), "loss": "mae"},
    "Y_l": {"horizons": (0, 1, 2, 3, 4, 5), "metrics": ("mae", "mse"), "loss": "mae"},
    "Y_b": {"horizons": (0, 1), "metrics": ("log_loss", "brier"), "loss": "log_loss"},
}
CELLS = tuple((t, h) for t in ("Y_s", "Y_l", "Y_b") for h in PROBE_CONTRACT[t]["horizons"])
RAW_PREFERENCE = ("baseline", "alt_mtae", "alt_p2c")
RECON_METRICS = (
    "mae_norm", "mse_norm", "mae_orig", "mse_orig", "mae_rel_train_constant",
    "mse_rel_train_constant", "acf_l1", "log_psd_l1", "mae_extremes", "dtw_mean",
    "train_constant_mae_norm", "train_constant_mse_norm",
)
MAJORITY_FOLDS = 4
MAJORITY_CELLS = 7
MAX_LOST_CELLS = 2
CKA_UNSTABLE = 0.5


class ConfigError(RuntimeError):
    pass


# --------------------------------------------------------------------------- utilities
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_json(path: Path, payload: Any) -> None:
    atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def fmean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def fstd(values: list[float]) -> float | None:
    return float(statistics.pstdev(values)) if len(values) > 1 else (0.0 if values else None)


def is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def expand(value: str, base: Path) -> Path:
    candidate = Path(os.path.expanduser(value))
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


# --------------------------------------------------------------------------- config
def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != SCHEMA_CONFIG:
        raise ConfigError(f"config schema must be {SCHEMA_CONFIG}")
    if cfg.get("rule_version") != RULE_VERSION:
        raise ConfigError(f"config rule_version {cfg.get('rule_version')!r} != tool {RULE_VERSION!r}")
    repo_root = expand(cfg["repo_root"], path.parent)
    rule_doc = repo_root / cfg["decision_rule_doc"]
    if not rule_doc.is_file():
        raise ConfigError(f"DECISION_RULE.md absent: {rule_doc}")
    if f"`RULE_VERSION = {RULE_VERSION}`" not in rule_doc.read_text():
        raise ConfigError("DECISION_RULE.md does not carry the tool's RULE_VERSION; refusing to compute")
    cfg["_repo_root"] = repo_root
    cfg["_config_path"] = path
    cfg["_output_dir"] = expand(cfg["output_dir"], repo_root)
    cfg["_state_dir"] = expand(cfg["state_dir"], repo_root)
    local_paths_file = expand(cfg["local_paths"], path.parent)
    if not local_paths_file.is_file():
        raise ConfigError(f"local_paths file absent (uncommitted, maps mirror role keys to directories): {local_paths_file}")
    local_paths = json.loads(local_paths_file.read_text())
    cfg["_local_paths"] = {key: expand(value, local_paths_file.parent) for key, value in local_paths.get("mirror_roots", {}).items()}
    for plan in cfg["plans"].values():
        plan["_tsv"] = expand(plan["tsv"], repo_root)
        missing = [key for key in plan["roots"] if key not in cfg["_local_paths"]]
        if missing:
            raise ConfigError(f"mirror role keys without a local path: {missing}")
        plan["_roots"] = [cfg["_local_paths"][key] for key in plan["roots"]]
    eta = cfg.get("eta_inputs", {})
    eta["_fs_gpu_eta"] = [expand(p, repo_root) for p in eta.get("fs_gpu_eta_candidates", [])]
    status_local = {k: expand(v, local_paths_file.parent) for k, v in local_paths.get("status_files", {}).items()}
    eta["_status_files"] = {k: status_local[k] for k in eta.get("status_files", []) if k in status_local}
    cfg["eta_inputs"] = eta
    cfg["_refit_input"] = expand(cfg["refit_input"], repo_root) if cfg.get("refit_input") else None
    cfg["_heavy"] = expand(cfg["heavy_candidates"], repo_root)
    return cfg


def load_heavy_candidates(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    out = {}
    for item in payload["features"]:
        if item.get("in_extractibility_queue"):
            out[item["feature_id"]] = item["batch"]
    return out


def load_plan(path: Path) -> list[dict[str, str]]:
    cells = []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            batch, feature, cell_family = row["cell_id"].split("::")
            role = CELL_FAMILY_TO_ROLE[cell_family]
            cells.append({
                "cell_id": row["cell_id"], "batch": batch, "feature_id": feature,
                "role": role, "result_dir": row["result_dir"],
            })
    return cells


# --------------------------------------------------------------------------- terminal validation
def validate_manifest(manifest: dict, feature: str, role: str, batch: str, cfg: dict) -> str | None:
    if manifest.get("schema") != "ut_pilot_run.v1":
        return "SCHEMA"
    if manifest.get("features") != [feature]:
        return "FEATURE"
    if list(manifest.get("families") or []) != list(ROLE_FAMILIES[role]):
        return "FAMILY"
    if list(manifest.get("folds") or []) != list(FOLDS):
        return "FOLDS"
    if manifest.get("seed") != cfg["seed"]:
        return "SEED"
    if manifest.get("code_commit") not in set(cfg["allowed_revisions"][role]):
        return "REVISION"
    if manifest.get("series_sha256") != cfg["batch_series_sha256"].get(batch):
        return "INPUT_DIGEST"
    args = manifest.get("args") or {}
    for key, expected in cfg["recipe"].items():
        actual = args.get(key)
        if isinstance(expected, float):
            if not is_num(actual) or abs(float(actual) - expected) > 1e-12:
                return f"RECIPE_{key}"
        elif actual != expected:
            return f"RECIPE_{key}"
    return None


def validate_rows(rows: list[dict], feature: str, role: str, seed: int) -> str | None:
    families = ROLE_FAMILIES[role]
    trained = [f for f in families if f not in ("identity", "random")]
    kinds = Counter(r.get("kind") for r in rows)
    expected_kinds = {
        "fold_family": len(FOLDS) * len(families),
        "probe": len(FOLDS) * len(families) * len(CELLS),
        "probe_delta": len(FOLDS) * len(trained) * len(CELLS),
        "feature_summary": 1,
    }
    if dict(kinds) != expected_kinds:
        return "ROW_KINDS"
    fold_families, probe_keys, delta_keys = set(), set(), set()
    for number, row in enumerate(rows, 1):
        if row.get("feature_id") != feature:
            return f"ROW_FEATURE:{number}"
        if "seed" in row and row["seed"] != seed:
            return f"ROW_SEED:{number}"
        kind = row["kind"]
        if kind == "fold_family":
            key = (row.get("fold_id"), row.get("family"))
            if key[0] not in FOLDS or key[1] not in families or key in fold_families:
                return f"ROW_FOLD_FAMILY:{number}"
            fold_families.add(key)
        elif kind == "probe":
            fam, fold, target, h = row.get("representation"), row.get("fold_id"), row.get("target"), row.get("horizon_index")
            spec = PROBE_CONTRACT.get(target)
            if fam not in families or fold not in FOLDS or spec is None or h not in spec["horizons"]:
                return f"ROW_PROBE_COVERAGE:{number}"
            if row.get("status") != "MEASURED":
                return f"ROW_PROBE_STATUS:{number}"
            for metric in spec["metrics"]:
                if not is_num(row.get(metric)):
                    return f"ROW_METRIC:{number}:{metric}"
            key = (fold, fam, target, h)
            if key in probe_keys:
                return f"ROW_DUPLICATE:{number}"
            probe_keys.add(key)
        elif kind == "probe_delta":
            fam, fold, target, h = row.get("trained"), row.get("fold_id"), row.get("target"), row.get("horizon_index")
            spec = PROBE_CONTRACT.get(target)
            if fam not in trained or fold not in FOLDS or spec is None or h not in spec["horizons"]:
                return f"ROW_DELTA_COVERAGE:{number}"
            if row.get("loss") != spec["loss"]:
                return f"ROW_DELTA_LOSS:{number}"
            if not is_num(row.get("delta_probe_random_minus_trained")) or not is_num(row.get("preservation_raw_minus_trained")):
                return f"ROW_DELTA_VALUE:{number}"
            key = (fold, fam, target, h)
            if key in delta_keys:
                return f"ROW_DUPLICATE:{number}"
            delta_keys.add(key)
    if len(fold_families) != expected_kinds["fold_family"]:
        return "ROW_FOLD_FAMILY_COVERAGE"
    if len(probe_keys) != expected_kinds["probe"]:
        return "ROW_PROBE_COVERAGE"
    if len(delta_keys) != expected_kinds["probe_delta"]:
        return "ROW_DELTA_COVERAGE"
    return None


# --------------------------------------------------------------------------- extraction (per terminal)
def extract_terminal(manifest: dict, rows: list[dict], role: str, results_sha: str) -> dict:
    """Reduce one admitted terminal to per-family, per-fold and per-cell scalars."""
    families = ROLE_FAMILIES[role]
    out: dict[str, Any] = {
        "role": role, "results_sha256": results_sha, "code_commit": manifest.get("code_commit"),
        "seed": manifest.get("seed"), "wall_seconds": manifest.get("wall_seconds"),
        "peak_rss_bytes": manifest.get("peak_rss_bytes"), "cgroup_peak_bytes": manifest.get("cgroup_peak_bytes"),
        "families": {},
        # one digest per fold; every family inside a terminal shares the fold's probe rows (lane D1)
        "train_row_ids_sha256": {fold: sorted({r.get("train_row_ids_sha256") for r in rows if r.get("kind") == "fold_family" and r.get("fold_id") == fold}) for fold in FOLDS},
    }
    summary = [r for r in rows if r["kind"] == "feature_summary"][-1]
    for fam in families:
        ff = {r["fold_id"]: r for r in rows if r["kind"] == "fold_family" and r["family"] == fam}
        rec_status = Counter(r["reconstruction"].get("status") for r in ff.values())
        recon: dict[str, list[float]] = defaultdict(list)
        for r in ff.values():
            if r["reconstruction"].get("status") == "MEASURED":
                for m in RECON_METRICS:
                    if is_num(r["reconstruction"].get(m)):
                        recon[m].append(float(r["reconstruction"][m]))
        rec_reason = next((r["reconstruction"].get("reason") for r in ff.values() if r["reconstruction"].get("status") != "MEASURED"), "")
        fam_out = {
            "architecture_id": next(iter(ff.values()))["architecture_id"],
            "latent_dim": next(iter(ff.values()))["latent_dim"],
            "reconstruction_status": "MEASURED" if rec_status.get("MEASURED") == len(FOLDS) else ("NOT_APPLICABLE" if rec_status.get("NOT_APPLICABLE") else "FAILED"),
            "reconstruction_reason": rec_reason,
            "reconstruction": {m: v for m, v in recon.items()},
            "participation_ratio": [float(r["effective_dimension"]["participation_ratio"]) for r in ff.values()],
            "n_components_95": [int(r["effective_dimension"]["n_components_95"]) for r in ff.values()],
            "fit_wall_seconds": [float(r["cost"]["fit_wall_seconds"]) for r in ff.values()],
            "updates": [int(r["cost"]["updates"]) for r in ff.values()],
            "epochs_run": [int(r["cost"]["epochs_run"]) for r in ff.values()],
            "params": max(int(r["cost"]["params"]) for r in ff.values()),
            "encode_latency_ms": [float(r["cost"]["encode_latency_ms_per_window"]) for r in ff.values()],
            "stop_reasons": sorted({str(r["fit_report"].get("stop_reason")) for r in ff.values()}),
            "stability": summary.get("stability", {}).get(fam, {}),
            "cells": {},
        }
        for (target, h) in CELLS:
            spec = PROBE_CONTRACT[target]
            probes = [r for r in rows if r["kind"] == "probe" and r["representation"] == fam and r["target"] == target and r["horizon_index"] == h]
            probes.sort(key=lambda r: FOLDS.index(r["fold_id"]))
            cell = {
                "loss": [float(r[spec["loss"]]) for r in probes],
                "loss_secondary": [float(r[spec["metrics"][1]]) for r in probes],
                "naives": {},
            }
            if target == "Y_b":
                cell["naives"]["prior_log_loss"] = [float(r["prior_log_loss"]) for r in probes]
            else:
                cell["naives"]["naive_zero_mae"] = [float(r["naive_zero_mae"]) for r in probes]
                cell["naives"]["naive_train_mean_mae"] = [float(r["naive_train_mean_mae"]) for r in probes]
            if fam not in ("identity", "random"):
                deltas = [r for r in rows if r["kind"] == "probe_delta" and r["trained"] == fam and r["target"] == target and r["horizon_index"] == h]
                deltas.sort(key=lambda r: FOLDS.index(r["fold_id"]))
                cell["delta"] = [float(r["delta_probe_random_minus_trained"]) for r in deltas]
                cell["preservation"] = [float(r["preservation_raw_minus_trained"]) for r in deltas]
            fam_out["cells"][f"{target}:{h}"] = cell
        out["families"][fam] = fam_out
    return out


def examine_cell(cell: dict, roots: list[Path], cfg: dict, cache_dir: Path) -> dict:
    """Terminal state of one planned cell across the roots that may hold it."""
    completed: list[dict] = []
    failed: list[dict] = []
    rejected: list[dict] = []
    for root in roots:
        directory = root / cell["result_dir"]
        manifest_path = directory / "run_manifest.json"
        results_path = directory / "results.jsonl"
        if manifest_path.is_file() and results_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                rejected.append({"reason": "MANIFEST_PARSE"})
                continue
            if manifest.get("status") != "COMPLETED":
                continue
            file_sha = sha256_file(results_path)
            if file_sha != manifest.get("results_sha256"):
                rejected.append({"reason": "RESULTS_HASH", "results_sha256": file_sha})
                continue
            reason = validate_manifest(manifest, cell["feature_id"], cell["role"], cell["batch"], cfg)
            if reason:
                rejected.append({"reason": reason, "results_sha256": file_sha})
                continue
            cached = cache_dir / f"{file_sha}.json"
            if cached.is_file():
                extracted = json.loads(cached.read_text())
            else:
                rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
                reason = validate_rows(rows, cell["feature_id"], cell["role"], cfg["seed"])
                if reason:
                    rejected.append({"reason": reason, "results_sha256": file_sha})
                    continue
                extracted = extract_terminal(manifest, rows, cell["role"], file_sha)
                atomic_json(cached, extracted)
            completed.append(extracted)
            continue
        if directory.is_dir():
            for name in sorted(directory.iterdir()):
                if name.name.startswith("FAILED") and name.suffix == ".json":
                    try:
                        payload = json.loads(name.read_text())
                    except (OSError, json.JSONDecodeError):
                        payload = {}
                    failed.append({"receipt_sha256": sha256_file(name), "rc": payload.get("rc"), "reason": "FAILED_RECEIPT"})
    distinct = {c["results_sha256"] for c in completed}
    if len(distinct) > 1:
        return {"state": "FAILED", "reason": "CONTRADICTORY_TERMINAL", "receipt_sha256": ",".join(sorted(distinct))}
    if completed:
        return {"state": "COMPLETED", "terminal": completed[0]}
    if rejected:
        r = rejected[0]
        return {"state": "FAILED", "reason": "REJECTED_" + r["reason"], "receipt_sha256": r.get("results_sha256", "")}
    if failed:
        return {"state": "FAILED", "reason": "FAILED_RECEIPT", "receipt_sha256": failed[0]["receipt_sha256"], "rc": failed[0].get("rc")}
    return {"state": "PENDING", "reason": "NOT_LANDED"}


# --------------------------------------------------------------------------- decision rule
def cell_verdicts(cell: dict) -> dict:
    """Sign-agreement verdicts for one trained family on one cell (DECISION_RULE §2)."""
    deltas, pres = cell.get("delta", []), cell.get("preservation", [])
    n = len(FOLDS)
    return {
        "beats_random": len(deltas) == n and sum(1 for d in deltas if d > 0) >= MAJORITY_FOLDS,
        "beats_raw": len(pres) == n and sum(1 for p in pres if p > 0) >= MAJORITY_FOLDS,
        "lost_to_raw": len(pres) == n and sum(1 for p in pres if p < 0) >= MAJORITY_FOLDS,
    }


def cell_skill(cell: dict) -> float | None:
    """Strict skill vs every same-row naive, on fold means (positive = beats all naives)."""
    loss = fmean(cell["loss"])
    if loss is None:
        return None
    skills = []
    for values in cell["naives"].values():
        naive = fmean(values)
        if naive is None or naive == 0:
            return None
        skills.append((naive - loss) / naive)
    return min(skills) if skills else None


def family_scores(fam_out: dict) -> dict:
    beats_random = beats_raw = lost = skill_cells = 0
    per_target_skill = Counter()
    for key, cell in fam_out["cells"].items():
        target = key.split(":")[0]
        if "delta" in cell:
            v = cell_verdicts(cell)
            beats_random += v["beats_random"]
            beats_raw += v["beats_raw"]
            lost += v["lost_to_raw"]
        s = cell_skill(cell)
        if s is not None and s > 0:
            skill_cells += 1
            per_target_skill[target] += 1
    return {
        "beats_random_cells": beats_random, "beats_raw_cells": beats_raw, "lost_to_raw_cells": lost,
        "indistinguishable_cells": len(CELLS) - beats_raw - lost,
        "skill_cells": skill_cells, "skill_cells_by_target": dict(per_target_skill),
    }


def load_refit(path: Path | None) -> dict[tuple[str, str], float] | None:
    """Paired-refit export (FS-CLOSE): feature_id, family, refit_gain (> 0 = better after refit).

    Rows without a finite refit_gain are skipped (the file fills as refits land).  A present
    file with no rows yet is an applied-but-empty table: every lookup is NOT_AVAILABLE.
    """
    if path is None or not path.is_file():
        return None
    table: dict[tuple[str, str], float] = {}
    if path.suffix == ".json":
        rows = json.loads(path.read_text())
    else:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
    for row in rows:
        try:
            gain = float(row.get("refit_gain"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(gain) and row.get("feature_id") and row.get("family"):
            table[(row["feature_id"], row["family"])] = gain
    return table


def decide(feature: str, terminals: dict[str, dict], refit: dict | None) -> dict:
    """Apply DECISION_RULE §3 to one candidate. ``terminals`` maps role -> examine_cell result."""
    states = {role: t["state"] for role, t in terminals.items()}
    flags: list[str] = []
    if any(s == "PENDING" for s in states.values()) or len(states) < 3:
        return {"decision": "PENDING", "flags": flags, "winner_scores": {}, "gates": {}}
    measured: dict[str, dict] = {}
    failed_families: list[str] = []
    for role, t in terminals.items():
        trained = [f for f in ROLE_FAMILIES[role] if f not in ("identity", "random")]
        if t["state"] == "COMPLETED":
            for fam in trained:
                measured[fam] = t["terminal"]["families"][fam]
        else:
            failed_families.extend(trained)
    for fold in FOLDS:
        digests = set()
        for t in terminals.values():
            if t["state"] == "COMPLETED":
                digests.update(t["terminal"]["train_row_ids_sha256"].get(fold, []))
        if len(digests) > 1:
            flags.append("PROBE_ROWS_DIFFER")
            break
    gates: dict[str, dict] = {}
    passing: list[tuple] = []
    any_skill = False
    for fam in TRAINED_ORDER:
        if fam not in measured:
            continue
        fo = measured[fam]
        sc = family_scores(fo)
        any_skill = any_skill or sc["skill_cells"] > 0
        g1 = sc["beats_random_cells"] >= MAJORITY_CELLS
        g2 = sc["beats_raw_cells"] >= MAJORITY_CELLS and sc["lost_to_raw_cells"] <= MAX_LOST_CELLS
        g3 = sc["skill_cells"] >= 1
        refit_gain = refit.get((feature, fam)) if refit is not None else None
        g4_applied = refit_gain is not None
        g4 = (refit_gain > 0) if g4_applied else None
        gates[fam] = {
            "G1_learned": g1, "G2_preserves": g2, "G3_utility": g3,
            "G4_refit": g4, "refit_gate_applied": g4_applied,
            "refit_gain": refit_gain if g4_applied else "NOT_AVAILABLE", **sc,
        }
        if g1 and g2 and g3 and (g4 or not g4_applied):
            passing.append((
                -(sc["beats_raw_cells"] - sc["lost_to_raw_cells"]),
                sum(fo["fit_wall_seconds"]), fo["params"], TRAINED_ORDER.index(fam), fam,
            ))
        stab = fo.get("stability", {})
        if stab.get("status") == "MEASURED" and is_num(stab.get("min")) and stab["min"] < CKA_UNSTABLE:
            flags.append(f"UNSTABLE_LATENT({fam})")
        if fo["latent_dim"] >= 8 and fo["n_components_95"] and all(n == 1 for n in fo["n_components_95"]):
            flags.append(f"COLLAPSED_LATENT({fam})")
        rel = fmean(fo["reconstruction"].get("mae_rel_train_constant", []))
        if rel is not None and rel >= 1:
            flags.append(f"RECONSTRUCTION_WORSE_THAN_TRAIN_CONSTANT({fam})")
    raw_fam = None
    for role in RAW_PREFERENCE:
        t = terminals.get(role)
        if t and t["state"] == "COMPLETED":
            raw_fam = t["terminal"]["families"]["identity"]
            break
    raw_skill = family_scores(raw_fam)["skill_cells"] > 0 if raw_fam else False
    flags.append("RAW_HAS_PROBE_SKILL" if raw_skill else "RAW_NO_PROBE_SKILL")
    raw_refit = refit.get((feature, "identity")) if refit is not None else None
    if raw_refit is not None:
        flags.append("RAW_REFIT_GAIN_POSITIVE" if raw_refit > 0 else "RAW_REFIT_GAIN_NONPOSITIVE")
    if not raw_skill and not any_skill:
        flags.append("NO_PROBE_SKILL_VS_NAIVE_ANY_FAMILY")
    if failed_families:
        flags.append("DECIDED_WITH_FAILED_FAMILIES")
        if not measured:
            flags.append("ALL_TRAINED_FAMILIES_FAILED")
    if passing:
        passing.sort()
        decision = passing[0][-1]
    elif measured and all(gates[f]["lost_to_raw_cells"] >= MAJORITY_CELLS for f in measured):
        decision = "RAW"
    elif not measured:
        decision = "RAW"
    else:
        decision = "NO_TRAINED_ADVANTAGE"
    return {"decision": decision, "flags": sorted(set(flags)), "gates": gates, "families_failed": failed_families,
            "raw_refit_gain": raw_refit, "refit_table_present": refit is not None}


# --------------------------------------------------------------------------- tables
DISPOSITION_COLUMNS = [
    "feature_id", "batch", "family", "family_kind", "terminal_role", "terminal_state", "terminal_reason",
    "receipt_sha256", "code_commit", "seed", "n_folds", "architecture_id", "latent_dim",
    "rec_status", "rec_reason", "rec_mae_norm", "rec_mse_norm", "rec_mae_orig", "rec_mse_orig",
    "rec_mae_rel_train_constant", "rec_mse_rel_train_constant", "rec_acf_l1", "rec_log_psd_l1",
    "rec_mae_extremes", "rec_dtw_mean",
    "stability_status", "stability_cka_mean", "stability_cka_min",
    "effdim_status", "effdim_participation_ratio_mean", "effdim_n_components_95_mean",
    "probe_status", "probe_loss_Ys_mean", "probe_loss_Yl_mean", "probe_loss_Yb_mean",
    "probe_skill_cells", "probe_skill_Ys_cells", "probe_skill_Yl_cells", "probe_skill_Yb_cells",
    "delta_status", "delta_beats_random_cells", "delta_mean_Ys", "delta_mean_Yl", "delta_mean_Yb",
    "preservation_status", "beats_raw_cells", "lost_to_raw_cells", "indistinguishable_cells",
    "preservation_mean_Ys", "preservation_mean_Yl", "preservation_mean_Yb",
    "refit_gate_applied", "refit_gain",
    "cost_status", "cost_fit_wall_seconds_sum", "cost_updates_sum", "cost_params", "cost_encode_latency_ms_mean",
    "terminal_wall_seconds", "terminal_peak_rss_bytes", "terminal_cgroup_peak_bytes",
    "G1_learned", "G2_preserves", "G3_utility", "G4_refit", "passes_all_gates",
    "candidate_decision", "decision_rule_version", "flags",
]


def _gate(value: Any) -> str:
    if value is None:
        return "NOT_APPLICABLE"
    return "PASS" if value else "FAIL"


def disposition_rows(feature: str, batch: str, terminals: dict[str, dict], decision: dict, catalog: list[list]) -> list[dict]:
    rows = []
    for fam in FAMILY_ORDER:
        if fam in ("identity", "random"):
            role = next((r for r in RAW_PREFERENCE if terminals.get(r, {}).get("state") == "COMPLETED"), None)
            if role is None:
                role = next((r for r in RAW_PREFERENCE if r in terminals), RAW_PREFERENCE[0])
        else:
            role = next(r for r, fams in ROLE_FAMILIES.items() if fam in fams)
        t = terminals.get(role, {"state": "PENDING", "reason": "NOT_PLANNED"})
        state = t["state"]
        kind = "raw" if fam == "identity" else ("random" if fam == "random" else "trained")
        row = {c: "" for c in DISPOSITION_COLUMNS}
        row.update({
            "feature_id": feature, "batch": batch, "family": fam, "family_kind": kind, "terminal_role": role,
            "terminal_state": state, "terminal_reason": t.get("reason", ""), "receipt_sha256": t.get("receipt_sha256", ""),
            "candidate_decision": decision["decision"], "decision_rule_version": RULE_VERSION,
            "flags": ";".join(decision.get("flags", [])),
            "refit_gate_applied": "false", "refit_gain": "NOT_AVAILABLE",
        })
        status_default = "FAILED" if state == "FAILED" else "PENDING"
        for col in ("rec_status", "stability_status", "effdim_status", "probe_status", "delta_status", "preservation_status", "cost_status"):
            row[col] = status_default
        for col in ("G1_learned", "G2_preserves", "G3_utility", "G4_refit", "passes_all_gates"):
            row[col] = status_default if kind == "trained" else "NOT_APPLICABLE"
        if kind != "trained":
            row["delta_status"] = row["preservation_status"] = "NOT_APPLICABLE"
        if state != "COMPLETED":
            rows.append(row)
            continue
        term = t["terminal"]
        fo = term["families"][fam]
        receipt = term["results_sha256"]
        row.update({
            "receipt_sha256": receipt, "code_commit": term["code_commit"], "seed": term["seed"], "n_folds": len(FOLDS),
            "architecture_id": fo["architecture_id"], "latent_dim": fo["latent_dim"],
            "rec_status": fo["reconstruction_status"], "rec_reason": fo["reconstruction_reason"],
            "terminal_wall_seconds": term["wall_seconds"], "terminal_peak_rss_bytes": term["peak_rss_bytes"],
            "terminal_cgroup_peak_bytes": term["cgroup_peak_bytes"],
        })
        base = [feature, batch, fam, role, "", "", "", receipt]
        if fo["reconstruction_status"] == "MEASURED":
            for m in RECON_METRICS:
                vals = fo["reconstruction"].get(m, [])
                if m in ("train_constant_mae_norm", "train_constant_mse_norm"):
                    pass
                else:
                    row[f"rec_{m}"] = fmean(vals)
                catalog.append(base[:4] + ["", "", f"reconstruction.{m}", len(vals), fmean(vals), fstd(vals), min(vals) if vals else None, max(vals) if vals else None, "MEASURED", receipt])
        else:
            for m in RECON_METRICS:
                catalog.append(base[:4] + ["", "", f"reconstruction.{m}", 0, None, None, None, None, fo["reconstruction_status"], receipt])
        stab = fo.get("stability", {})
        row["stability_status"] = stab.get("status", "FAILED")
        if stab.get("status") == "MEASURED":
            row["stability_cka_mean"], row["stability_cka_min"] = stab.get("mean"), stab.get("min")
        catalog.append(base[:4] + ["", "", "stability.linear_cka_mean", stab.get("pairs"), stab.get("mean"), None, stab.get("min"), None, stab.get("status", "FAILED"), receipt])
        row["effdim_status"] = "MEASURED"
        row["effdim_participation_ratio_mean"] = fmean(fo["participation_ratio"])
        row["effdim_n_components_95_mean"] = fmean([float(v) for v in fo["n_components_95"]])
        catalog.append(base[:4] + ["", "", "effective_dimension.participation_ratio", len(fo["participation_ratio"]), fmean(fo["participation_ratio"]), fstd(fo["participation_ratio"]), min(fo["participation_ratio"]), max(fo["participation_ratio"]), "MEASURED", receipt])
        catalog.append(base[:4] + ["", "", "effective_dimension.n_components_95", len(fo["n_components_95"]), fmean([float(v) for v in fo["n_components_95"]]), None, min(fo["n_components_95"]), max(fo["n_components_95"]), "MEASURED", receipt])
        row["cost_status"] = "MEASURED"
        row["cost_fit_wall_seconds_sum"] = sum(fo["fit_wall_seconds"])
        row["cost_updates_sum"] = sum(fo["updates"])
        row["cost_params"] = fo["params"]
        row["cost_encode_latency_ms_mean"] = fmean(fo["encode_latency_ms"])
        for m, vals in (("cost.fit_wall_seconds", fo["fit_wall_seconds"]), ("cost.updates", [float(v) for v in fo["updates"]]), ("cost.epochs_run", [float(v) for v in fo["epochs_run"]]), ("cost.encode_latency_ms_per_window", fo["encode_latency_ms"])):
            catalog.append(base[:4] + ["", "", m, len(vals), fmean(vals), fstd(vals), min(vals), max(vals), "MEASURED", receipt])
        catalog.append(base[:4] + ["", "", "cost.params", 1, fo["params"], None, fo["params"], fo["params"], "MEASURED", receipt])
        if fam == "identity" and decision.get("refit_table_present"):
            gain = decision.get("raw_refit_gain")
            row["refit_gate_applied"] = "true" if gain is not None else "false"
            row["refit_gain"] = gain if gain is not None else "NOT_AVAILABLE"
        sc = family_scores(fo)
        row["probe_status"] = "MEASURED"
        row["probe_skill_cells"] = sc["skill_cells"]
        for tgt, short in (("Y_s", "Ys"), ("Y_l", "Yl"), ("Y_b", "Yb")):
            losses = [fmean(fo["cells"][f"{tgt}:{h}"]["loss"]) for h in PROBE_CONTRACT[tgt]["horizons"]]
            row[f"probe_loss_{short}_mean"] = fmean([v for v in losses if v is not None])
            row[f"probe_skill_{short}_cells"] = sc["skill_cells_by_target"].get(tgt, 0)
        if kind == "trained":
            row["delta_status"] = row["preservation_status"] = "MEASURED"
            row["delta_beats_random_cells"] = sc["beats_random_cells"]
            row["beats_raw_cells"] = sc["beats_raw_cells"]
            row["lost_to_raw_cells"] = sc["lost_to_raw_cells"]
            row["indistinguishable_cells"] = sc["indistinguishable_cells"]
            for tgt, short in (("Y_s", "Ys"), ("Y_l", "Yl"), ("Y_b", "Yb")):
                ds = [fmean(fo["cells"][f"{tgt}:{h}"]["delta"]) for h in PROBE_CONTRACT[tgt]["horizons"]]
                ps = [fmean(fo["cells"][f"{tgt}:{h}"]["preservation"]) for h in PROBE_CONTRACT[tgt]["horizons"]]
                row[f"delta_mean_{short}"] = fmean(ds)
                row[f"preservation_mean_{short}"] = fmean(ps)
            g = decision.get("gates", {}).get(fam)
            if g:
                row["G1_learned"], row["G2_preserves"], row["G3_utility"] = _gate(g["G1_learned"]), _gate(g["G2_preserves"]), _gate(g["G3_utility"])
                row["G4_refit"] = _gate(g["G4_refit"]) if g["refit_gate_applied"] else "NOT_APPLICABLE"
                row["refit_gate_applied"] = "true" if g["refit_gate_applied"] else "false"
                row["refit_gain"] = g["refit_gain"]
                row["passes_all_gates"] = "PASS" if (g["G1_learned"] and g["G2_preserves"] and g["G3_utility"] and (g["G4_refit"] or not g["refit_gate_applied"])) else "FAIL"
            elif decision["decision"] == "PENDING":
                for col in ("G1_learned", "G2_preserves", "G3_utility", "G4_refit", "passes_all_gates"):
                    row[col] = "PENDING"
        for (tgt, h) in CELLS:
            cell = fo["cells"][f"{tgt}:{h}"]
            cb = [feature, batch, fam, role, tgt, h]
            spec = PROBE_CONTRACT[tgt]
            for m, vals in ((f"probe.{spec['loss']}", cell["loss"]), (f"probe.{spec['metrics'][1]}", cell["loss_secondary"])):
                catalog.append(cb + [m, len(vals), fmean(vals), fstd(vals), min(vals), max(vals), "MEASURED", receipt])
            for nm, vals in cell["naives"].items():
                catalog.append(cb + [f"probe.{nm}", len(vals), fmean(vals), fstd(vals), min(vals), max(vals), "MEASURED", receipt])
            s = cell_skill(cell)
            catalog.append(cb + ["probe.skill_strict_vs_all_naives", len(cell["loss"]), s, None, None, None, "MEASURED" if s is not None else "FAILED", receipt])
            if "delta" in cell:
                v = cell_verdicts(cell)
                for m, vals in (("probe_delta.random_minus_trained", cell["delta"]), ("probe_delta.preservation_raw_minus_trained", cell["preservation"])):
                    catalog.append(cb + [m, len(vals), fmean(vals), fstd(vals), min(vals), max(vals), "MEASURED", receipt])
                catalog.append(cb + ["probe_delta.folds_delta_positive", len(cell["delta"]), sum(1 for d in cell["delta"] if d > 0), None, None, None, "MEASURED", receipt])
                catalog.append(cb + ["probe_delta.folds_preservation_positive", len(cell["preservation"]), sum(1 for p in cell["preservation"] if p > 0), None, None, None, "MEASURED", receipt])
                catalog.append(cb + ["probe_delta.folds_preservation_negative", len(cell["preservation"]), sum(1 for p in cell["preservation"] if p < 0), None, None, None, "MEASURED", receipt])
                catalog.append(cb + ["probe_delta.verdict_beats_random", len(cell["delta"]), int(v["beats_random"]), None, None, None, "MEASURED", receipt])
                catalog.append(cb + ["probe_delta.verdict_beats_raw", len(cell["preservation"]), int(v["beats_raw"]), None, None, None, "MEASURED", receipt])
                catalog.append(cb + ["probe_delta.verdict_lost_to_raw", len(cell["preservation"]), int(v["lost_to_raw"]), None, None, None, "MEASURED", receipt])
        rows.append(row)
    return rows


CATALOG_COLUMNS = ["feature_id", "batch", "family", "terminal_role", "target", "horizon", "metric", "n_folds", "mean", "std", "min", "max", "status", "receipt_sha256"]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def csv_bytes(columns: list[str], rows: list) -> bytes:
    import io
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        values = [row.get(c) for c in columns] if isinstance(row, dict) else row
        writer.writerow([_fmt(v) for v in values])
    return buf.getvalue().encode()


# --------------------------------------------------------------------------- ETA
def eta_block(cfg: dict, now: datetime) -> dict:
    eta_cfg = cfg["eta_inputs"]
    for candidate in eta_cfg["_fs_gpu_eta"]:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text())
                return {"source": "FS-GPU ps3r_eta.json", "source_path": str(candidate.relative_to(cfg["_repo_root"])) if str(candidate).startswith(str(cfg["_repo_root"])) else candidate.name, "fs_gpu_eta": payload}
            except (OSError, json.JSONDecodeError, ValueError):
                pass
    queues = {}
    for name, path in eta_cfg["_status_files"].items():
        try:
            st = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            queues[name] = {"error": "STATUS_UNREADABLE"}
            continue
        counts, dur, workers = st.get("counts", {}), st.get("durations_seconds", {}), st.get("workers", 1) or 1
        remaining = int(counts.get("pending", 0)) + int(counts.get("running", 0))
        queues[name] = {
            "completed": counts.get("completed"), "failed": counts.get("failed"), "remaining": remaining,
            "total": counts.get("total"), "workers": workers,
            "median_seconds": dur.get("median"), "p90_seconds": dur.get("p90"), "sample_size": dur.get("sample_size"),
            "status_updated_from": path.name,
        }
    # The successor queue starts when the alternatives queue closes and has no own sample yet:
    # its durations are proxied by the baseline queue on the other worker (declared).
    def finish(start: datetime, remaining: int, per: float | None, workers: int) -> str | None:
        if per is None or remaining <= 0:
            return iso(start) if remaining <= 0 else None
        return iso(start + timedelta(seconds=math.ceil(remaining / max(workers, 1)) * per))
    alt, base, succ = queues.get("alternative", {}), queues.get("baseline_002", {}), queues.get("baseline_successor", {})
    for q in (alt, base):
        if "remaining" in q:
            q["eta_median_utc"] = finish(now, q["remaining"], q.get("median_seconds"), q["workers"])
            q["eta_p90_utc"] = finish(now, q["remaining"], q.get("p90_seconds"), q["workers"])
    if "remaining" in succ:
        proxy_median = succ.get("median_seconds") or base.get("median_seconds")
        proxy_p90 = succ.get("p90_seconds") or base.get("p90_seconds")
        succ["duration_proxy"] = "own sample" if succ.get("median_seconds") else "baseline_002 queue durations (other GPU) until the first successor cells close"
        start_m = datetime.strptime(alt["eta_median_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if alt.get("eta_median_utc") else now
        start_p = datetime.strptime(alt["eta_p90_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) if alt.get("eta_p90_utc") else now
        succ["eta_median_utc"] = finish(start_m, succ["remaining"], proxy_median, succ["workers"])
        succ["eta_p90_utc"] = finish(start_p, succ["remaining"], proxy_p90, succ["workers"])
    full = [q.get("eta_median_utc") for q in queues.values() if q.get("eta_median_utc")]
    full_p90 = [q.get("eta_p90_utc") for q in queues.values() if q.get("eta_p90_utc")]
    return {
        "source": "queue STATUS files (median / nearest-rank p90 of observed wall_seconds, one worker per queue)",
        "queues": queues,
        "full_coverage_eta_median_utc": max(full) if full and len(full) == len(queues) else None,
        "full_coverage_eta_p90_utc": max(full_p90) if full_p90 and len(full_p90) == len(queues) else None,
        "note": "ETA excludes work the alternatives GPU may steal from the baseline_002 queue after the successor queue; a stolen cell shortens, never lengthens, these dates.",
    }


# --------------------------------------------------------------------------- one cycle
def pull_branch(cfg: dict) -> None:
    """Best-effort pull so other lanes' committed inputs (e.g. the FS-CLOSE refit export) arrive."""
    result = subprocess.run(["git", "-C", str(cfg["_repo_root"]), "pull", "--no-rebase", "--no-edit", "-q"], capture_output=True, text=True, timeout=300)
    with (cfg["_state_dir"] / "git_log.jsonl").open("a") as handle:
        handle.write(json.dumps({"at": iso(utc_now()), "pull_rc": result.returncode, "stderr": result.stderr[-300:]}, sort_keys=True) + "\n")


def run_cycle(cfg: dict, *, git: bool) -> dict:
    started = utc_now()
    out_dir, state_dir = cfg["_output_dir"], cfg["_state_dir"]
    if git and cfg.get("pull_each_cycle", True):
        state_dir.mkdir(parents=True, exist_ok=True)
        pull_branch(cfg)
    cache_dir = state_dir / "terminal_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    heavy = load_heavy_candidates(cfg["_heavy"])
    if len(heavy) != cfg["heavy_denominator"]:
        raise ConfigError(f"heavy denominator {len(heavy)} != declared {cfg['heavy_denominator']}")
    per_feature: dict[str, dict[str, dict]] = defaultdict(dict)
    plan_count = 0
    for plan in cfg["plans"].values():
        for cell in load_plan(plan["_tsv"]):
            plan_count += 1
            if cell["feature_id"] not in heavy:
                continue
            if cell["batch"] != heavy[cell["feature_id"]]:
                per_feature[cell["feature_id"]][cell["role"]] = {"state": "FAILED", "reason": "PLAN_BATCH_MISMATCH"}
                continue
            result = examine_cell(cell, plan["_roots"], cfg, cache_dir)
            prior = per_feature[cell["feature_id"]].get(cell["role"])
            if prior is not None and prior["state"] == "COMPLETED" and result["state"] == "COMPLETED" and prior["terminal"]["results_sha256"] != result["terminal"]["results_sha256"]:
                result = {"state": "FAILED", "reason": "CONTRADICTORY_TERMINAL", "receipt_sha256": prior["terminal"]["results_sha256"] + "," + result["terminal"]["results_sha256"]}
            if prior is None or prior["state"] != "COMPLETED" or result["state"] == "FAILED":
                per_feature[cell["feature_id"]][cell["role"]] = result
    refit = load_refit(cfg["_refit_input"])
    rows: list[dict] = []
    catalog: list[list] = []
    decisions = Counter()
    candidates = {}
    role_states = {role: Counter() for role in ROLE_FAMILIES}
    families_present_hist = Counter()
    covered = 0
    for feature in sorted(heavy):
        terminals = per_feature.get(feature, {})
        for role in ROLE_FAMILIES:
            terminals.setdefault(role, {"state": "PENDING", "reason": "NOT_PLANNED"})
            role_states[role][terminals[role]["state"]] += 1
        decision = decide(feature, terminals, refit)
        decisions[decision["decision"]] += 1
        present = [f for f in FAMILY_ORDER if any(t["state"] == "COMPLETED" and f in ROLE_FAMILIES[r] for r, t in terminals.items())]
        families_present_hist[len(present)] += 1
        if all(t["state"] == "COMPLETED" for t in terminals.values()):
            covered += 1
        candidates[feature] = {
            "batch": heavy[feature], "decision": decision["decision"], "families_present": present,
            "terminals": {r: {"state": t["state"], "reason": t.get("reason", ""), "receipt_sha256": t["terminal"]["results_sha256"] if t["state"] == "COMPLETED" else t.get("receipt_sha256", "")} for r, t in terminals.items()},
            "flags": decision.get("flags", []),
        }
        rows.extend(disposition_rows(feature, heavy[feature], terminals, decision, catalog))
    dispositions = csv_bytes(DISPOSITION_COLUMNS, rows)
    catalog_rows = sorted(catalog, key=lambda r: (r[0], r[2], r[3], str(r[4]), str(r[5]), r[6]))
    catalog_bytes = csv_bytes(CATALOG_COLUMNS, catalog_rows)
    catalog_gz = gzip.compress(catalog_bytes, mtime=0)
    atomic_write(out_dir / "representation_dispositions.csv", dispositions)
    atomic_write(state_dir / "representation_metric_catalog.csv", catalog_bytes)
    atomic_write(state_dir / "representation_metric_catalog.csv.gz", catalog_gz)
    # milestones: commit the catalog slice into the repo on first run, 50 % and 100 %
    milestones_path = state_dir / "milestones.json"
    milestones = json.loads(milestones_path.read_text()) if milestones_path.is_file() else {}
    coverage = covered / len(heavy)
    reached = [m for m, threshold in (("first_table", 0.0), ("coverage_50", 0.5), ("coverage_100", 1.0)) if coverage >= threshold and m not in milestones]
    for m in reached:
        milestones[m] = {"at": iso(started), "covered": covered, "denominator": len(heavy)}
    if reached:
        if any(m in ("first_table", "coverage_100") for m in reached):
            atomic_write(out_dir / "representation_metric_catalog.csv.gz", catalog_gz)
        atomic_json(milestones_path, milestones)
    decisions_table = [{"feature_id": f, "batch": c["batch"], "decision": c["decision"], "families_present": ";".join(c["families_present"]), "flags": ";".join(c["flags"])} for f, c in candidates.items()]
    atomic_write(out_dir / "candidate_decisions.csv", csv_bytes(["feature_id", "batch", "decision", "families_present", "flags"], decisions_table))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    progress = {
        "schema": SCHEMA_PROGRESS, "rule_version": RULE_VERSION, "updated_at": iso(utc_now()), "cycle_started_at": iso(started),
        "heavy_denominator": len(heavy), "heavy_candidates_covered": covered, "coverage_fraction": round(coverage, 4),
        "candidates_decided": sum(v for k, v in decisions.items() if k != "PENDING"),
        "decisions": dict(decisions),
        "terminals_by_role": {r: dict(c) for r, c in role_states.items()},
        "families_present_histogram": {str(k): v for k, v in sorted(families_present_hist.items())},
        "planned_cells_total": plan_count,
        "refit_input": str(cfg["refit_input"]) if cfg.get("refit_input") else None,
        "refit_gate_applied": refit is not None,
        "eta": eta_block(cfg, started),
        "milestones": milestones,
        "outputs": {
            "representation_dispositions.csv": {"sha256": sha256_bytes(dispositions), "rows": len(rows)},
            "candidate_decisions.csv": {"rows": len(decisions_table)},
            "representation_metric_catalog.csv": {"sha256": sha256_bytes(catalog_bytes), "rows": len(catalog_rows), "location": "state_dir; committed as .csv.gz at the first table and at full coverage"},
            "representation_metric_catalog.csv.gz": {"sha256": sha256_bytes(catalog_gz)},
        },
        "safety": {"split": "train", "targets_read": False, "validation_read": False, "test_read": False, "tensorflow_imported": "tensorflow" in sys.modules},
        "process_peak_rss_bytes": int(peak_rss),
        "candidates": candidates,
    }
    atomic_json(out_dir / "progress.json", progress)
    if git:
        maybe_publish(cfg, progress, reached)
    return progress


def maybe_publish(cfg: dict, progress: dict, reached: list[str]) -> None:
    """Commit only on substantive change (table digest, decisions, milestone) or hourly refresh."""
    marker_path = cfg["_state_dir"] / "publish_state.json"
    try:
        marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        marker = {}
    fingerprint = {
        "dispositions_sha256": progress["outputs"]["representation_dispositions.csv"]["sha256"],
        "decisions": progress["decisions"],
        "covered": progress["heavy_candidates_covered"],
    }
    last_at = marker.get("published_at")
    stale = True
    if last_at:
        try:
            stale = (utc_now() - datetime.strptime(last_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)) >= timedelta(seconds=cfg.get("publish_refresh_seconds", 3600))
        except ValueError:
            stale = True
    if marker.get("fingerprint") == fingerprint and not reached and not stale:
        return
    if publish(cfg, progress, reached):
        atomic_json(marker_path, {"published_at": iso(utc_now()), "fingerprint": fingerprint})


def publish(cfg: dict, progress: dict, reached: list[str]) -> bool:
    repo, out_dir = cfg["_repo_root"], cfg["_output_dir"]
    rel = str(out_dir.relative_to(repo))
    log_path = cfg["_state_dir"] / "git_log.jsonl"

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=300)

    status = run("status", "--porcelain", "--", rel)
    if not status.stdout.strip():
        return True
    message = (
        f"FS-REP: dispositions {progress['heavy_candidates_covered']}/{progress['heavy_denominator']} covered, "
        f"{progress['candidates_decided']} decided"
        + (f" [milestone {', '.join(reached)}]" if reached else "")
        + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
    )
    record = {"at": iso(utc_now()), "steps": []}
    committed = False
    for attempt in range(3):
        add = run("add", "--", rel)
        commit = run("commit", "-q", "-m", message, "--", rel)
        record["steps"].append({"attempt": attempt, "add_rc": add.returncode, "commit_rc": commit.returncode, "stderr": (add.stderr + commit.stderr)[-400:]})
        if commit.returncode == 0:
            committed = True
            pull = run("pull", "--no-rebase", "--no-edit", "-q")
            push = run("push", "-q")
            record["steps"].append({"pull_rc": pull.returncode, "push_rc": push.returncode, "stderr": (pull.stderr + push.stderr)[-400:]})
            break
        if "index.lock" in (add.stderr + commit.stderr):
            time.sleep(5 * (attempt + 1))
            continue
        break
    with log_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return committed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-git", action="store_true", help="do not commit/push the output directory")
    parser.add_argument("--poll-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError, KeyError, json.JSONDecodeError) as error:
        print(f"config refused: {error}", file=sys.stderr)
        return 2
    poll = args.poll_seconds or float(cfg.get("poll_interval_seconds", 120))
    cfg["_state_dir"].mkdir(parents=True, exist_ok=True)
    lock_path = cfg["_state_dir"] / "follower.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another FS-REP follower holds {lock_path}", file=sys.stderr)
            return 75
        while True:
            try:
                progress = run_cycle(cfg, git=not args.no_git)
                print(json.dumps({"covered": progress["heavy_candidates_covered"], "decided": progress["candidates_decided"], "decisions": progress["decisions"], "updated_at": progress["updated_at"]}, sort_keys=True), flush=True)
            except Exception as error:  # noqa: BLE001 - a durable follower reports and keeps polling
                failure = {"schema": SCHEMA_PROGRESS, "rule_version": RULE_VERSION, "updated_at": iso(utc_now()), "fatal_error": f"{type(error).__name__}: {error}"}
                atomic_json(cfg["_state_dir"] / "last_error.json", failure)
                print(json.dumps(failure), file=sys.stderr, flush=True)
                if args.once:
                    raise
            if args.once:
                return 0
            time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
