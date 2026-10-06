#!/usr/bin/env python3
"""Frozen population manifest for feature-selection phases 2 and 3.

The manifest binds, under the phase-1 closure identity, the exact feature population,
the targets/horizons, the TRAIN row count, the chronological inner folds (TRAIN ranges
only: validation ranges are deliberately not retained) and the SHA-256 of every phase-1
artifact the campaign depends on.  It is committed to the repository and contains no
absolute paths, host names or credentials: every file is named by basename and digest,
and the data root is always supplied by argument.

Feature order inside the manifest is canonical (sorted by feature id) so that the
population digest and every derived identity are invariant to the column order of the
source parquet.

CLI::

    fs_phase23_manifest.py freeze --population EURUSD --phase1-state-dir DIR \
        --bundle-root DIR --out fs_phase23/manifest/EURUSD_MANIFEST.json [--contract fs_phase23/data/CONTRACT.json]
    fs_phase23_manifest.py verify --manifest FILE --data-root DIR
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fs_phase23.population_manifest.v1"
CAUSAL_LABEL = "CAUSAL_SUPPORTED"
FORBIDDEN_SPLIT_TOKENS = ("validation", "test", "holdout")

_FEATURE_META_FIELDS = ("feature_id", "family", "source", "transform", "unit", "availability_time", "clock",
                        "support_h", "train_coverage", "source_bytes")


class ManifestError(RuntimeError):
    """A manifest that cannot be frozen or does not verify."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_number(value: Any) -> Any:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f != f:   # NaN
        return None
    return int(f) if f.is_integer() and abs(f) < 1e15 else f


def population_digest(manifest: dict) -> str:
    """Identity of the population: feature ids (canonical order), targets, folds, data digests."""
    return digest({
        "population_id": manifest["population_id"],
        "identity": manifest["identity"],
        "features": [f["feature_id"] for f in manifest["features"]],
        "targets": [(t["target_id"], t["horizon_hours"]) for t in manifest["targets"]],
        "folds": manifest["folds"],
        "train_rows": manifest["data"]["train_rows"],
        "canonical_columns_sha256": manifest["data"].get("canonical_columns_sha256"),
    })


def manifest_digest(manifest: dict) -> str:
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    return digest(body)


def _check_no_private_text(manifest: dict) -> None:
    text = json.dumps(manifest)
    for token in ("/home/", "/Users/", "127.0.0.1", "ssh ", "password", "token"):
        if token in text:
            raise ManifestError(f"manifest would carry private text {token!r}; refuse to freeze")


def canonical_columns_digest(path: Path, features: list[str], timestamp_column: str) -> str:
    """Digest of the TRAIN content in canonical column order: invariant to the parquet's column order."""
    import pandas as pd
    frame = pd.read_parquet(path, columns=[timestamp_column, *sorted(features)])
    h = hashlib.sha256()
    ts = pd.to_datetime(frame[timestamp_column], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    h.update(ts.to_numpy().astype("datetime64[ns]").astype("int64").tobytes())
    for f in sorted(features):
        h.update(f.encode())
        h.update(frame[f].to_numpy(dtype="float64").tobytes())
    return h.hexdigest()


def build_manifest(*, population_id: str, identity: str, identity_source: dict, phase1: dict, data: dict,
                   features: list[dict], targets: list[dict], folds: list[dict], causal_supported: list[dict],
                   contract_source: str, bar_hours: int | None = None, data_root: Path | None = None) -> dict:
    if not features:
        raise ManifestError("a population needs at least one feature")
    ids = [f["feature_id"] for f in features]
    if len(set(ids)) != len(ids):
        raise ManifestError("duplicate feature ids")
    for key in ("features_file", "targets_file"):
        name = str(data[key]).lower()
        if any(tok in name for tok in FORBIDDEN_SPLIT_TOKENS):
            raise ManifestError(f"{key} names a non-TRAIN split: {data[key]}")
    clean_features = []
    for f in sorted(features, key=lambda r: r["feature_id"]):
        row = {k: f.get(k) for k in _FEATURE_META_FIELDS}
        for k in ("support_h", "train_coverage", "source_bytes"):
            row[k] = _clean_number(row[k])
        clean_features.append(row)
    clean_folds = []
    for i, fold in enumerate(folds):
        a, b = int(fold["train_rows"][0]), int(fold["train_rows"][1])
        if not (0 <= a < b <= int(data["train_rows"])):
            raise ManifestError(f"fold {i} train range {a}:{b} outside TRAIN 0:{data['train_rows']}")
        clean_folds.append({"fold_id": str(fold.get("fold_id", f"fold_{i}")), "train_rows": [a, b]})
    if [f["fold_id"] for f in clean_folds] != sorted({f["fold_id"] for f in clean_folds}, key=[f["fold_id"] for f in clean_folds].index):
        raise ManifestError("duplicate fold ids")
    clean_targets = [{"target_id": t["target_id"], "column": t.get("column", t["target_id"]), "family": t.get("family"),
                      "head": t.get("head"), "horizon_hours": int(t["horizon_hours"])} for t in targets]
    known = set(ids)
    tids = {t["target_id"] for t in clean_targets}
    causal = []
    for c in causal_supported:
        if c["feature_id"] not in known or c["target_id"] not in tids:
            raise ManifestError(f"causal-supported pair outside population: {c}")
        causal.append({"feature_id": c["feature_id"], "target_id": c["target_id"], "horizon_hours": int(c["horizon_hours"])})
    causal.sort(key=lambda c: (c["feature_id"], c["target_id"]))
    n = len(ids)
    if data_root is not None and not data.get("canonical_columns_sha256"):
        data = dict(data, canonical_columns_sha256=canonical_columns_digest(Path(data_root) / data["features_file"], ids, data["timestamp_column"]))
    manifest = {
        "schema": SCHEMA,
        "population_id": population_id,
        "identity": identity,
        "identity_source": identity_source,
        "contract_source": contract_source,
        "phase1": dict(phase1),
        "data": {
            "features_file": Path(data["features_file"]).name,
            "features_sha256": data["features_sha256"],
            "targets_file": Path(data["targets_file"]).name,
            "targets_sha256": data["targets_sha256"],
            "folds_file": Path(data["folds_file"]).name if data.get("folds_file") else None,
            "folds_sha256": data.get("folds_sha256"),
            "timestamp_column": data["timestamp_column"],
            "row_id_column": data["row_id_column"],
            "bar_hours": int(bar_hours or data["bar_hours"]),
            "train_rows": int(data["train_rows"]),
            "canonical_columns_sha256": data.get("canonical_columns_sha256"),
            "split": "TRAIN",
        },
        "features": clean_features,
        "targets": clean_targets,
        "folds": clean_folds,
        "causal_supported": causal,
        "causal_supported_label": CAUSAL_LABEL,
        "causal_supported_meaning": "phase-1 causal-rule support; candidates for validation, NOT a final selection",
        "final_selection": False,
        "expected_pairs": n * (n - 1) // 2,
        "feature_count": n,
        "target_count": len(clean_targets),
        "fold_count": len(clean_folds),
    }
    manifest["population_sha256"] = population_digest(manifest)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    _check_no_private_text(manifest)
    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["population_sha256"] = population_digest(manifest)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ManifestError(f"unexpected manifest schema {manifest.get('schema')!r}")
    if manifest_digest(manifest) != manifest["manifest_sha256"]:
        raise ManifestError(f"manifest digest mismatch in {path}")
    for key in ("features_file", "targets_file"):
        if any(tok in manifest["data"][key].lower() for tok in FORBIDDEN_SPLIT_TOKENS):
            raise ManifestError(f"manifest {key} names a non-TRAIN (validation/test) split: {manifest['data'][key]}")
    return manifest


def verify_data(manifest: dict, data_root: Path) -> dict:
    """Digest-check the TRAIN files the manifest names; never opens anything else."""
    data_root = Path(data_root)
    out = {}
    for key in ("features", "targets"):
        path = data_root / manifest["data"][f"{key}_file"]
        if not path.exists():
            raise ManifestError(f"{key} file missing under data root: {path.name}")
        observed = sha256_file(path)
        if observed != manifest["data"][f"{key}_sha256"]:
            raise ManifestError(f"{key} file digest {observed[:16]} != manifest {manifest['data'][f'{key}_sha256'][:16]}")
        out[key] = observed
    return out


# ----------------------------------------------------------------------------- freezing from phase 1

def _read_inventory(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r["feature_id"]: r for r in rows}


def _phase1_targets(deployment: dict, population: str) -> list[dict]:
    pack = deployment["populations"][population]["target_pack"]
    return [{"target_id": d["name"], "column": d["column"], "family": d.get("family"), "head": d.get("head"),
             "horizon_hours": d["horizon_hours"]} for d in pack["definitions"]]


def _phase1_folds(folds_doc: dict) -> list[dict]:
    out = []
    for i, fold in enumerate(folds_doc["folds"]):
        out.append({"fold_id": fold.get("name", f"fold_{i}"), "train_rows": list(fold["train_rows"])})
    return out


def _selected_pairs(state_dir: Path, population: str, identity: str) -> list[dict]:
    """Phase-1 SELECTED feature-target pairs (causal-rule support) from the closure readback."""
    readback = state_dir / "selected-features-readback.json"
    if readback.exists():
        doc = json.loads(readback.read_text())
        rows = [r for r in doc["rows"] if r.get("decision") == "SELECTED" and r.get("run_id") == identity]
        return [{"feature_id": r["feature_id"], "target_id": r["target_id"], "horizon_hours": r["horizon"]} for r in rows]
    result = state_dir / "finalizer" / "result.json"
    if result.exists():
        doc = json.loads(result.read_text())
        rows = doc["envelope"]["rows"].get("selection_decisions", [])
        return [{"feature_id": r["feature_id"], "target_id": r["target_id"], "horizon_hours": r["horizon"]}
                for r in rows if r.get("decision") == "SELECTED"]
    raise ManifestError("no phase-1 selection readback found")


def freeze_from_phase1(*, population: str, phase1_state_dir: Path, bundle_root: Path, identity: str,
                       contract_path: Path | None = None) -> dict:
    """Freeze the manifest from the phase-1 closure artifacts (read-only; nothing in phase 1 is edited)."""
    state = Path(phase1_state_dir)
    bundle = Path(bundle_root)
    pop_lower = population.lower()
    plan = json.loads((state / "PLAN.json").read_text())
    complete = json.loads((state / "PHASE_1_COMPLETE.json").read_text())
    if plan.get("population_id") != population:
        raise ManifestError(f"phase-1 plan population {plan.get('population_id')} != {population}")
    if complete.get("state") != "PHASE_1_COMPLETE":
        raise ManifestError("phase-1 closure state is not PHASE_1_COMPLETE")
    deployment = json.loads((bundle / "worker" / f"{pop_lower}_deployment.json").read_text())
    folds_path = bundle / "data" / f"{pop_lower}_folds.json"
    folds_doc = json.loads(folds_path.read_text())
    inventory = _read_inventory(bundle / "inventory" / f"{pop_lower}_inventory.csv")
    feature_ids = [item["feature_id"] for item in plan["items"]]
    if len(feature_ids) != plan["inventory_total"]:
        raise ManifestError("phase-1 plan items do not match its inventory total")
    features = []
    for fid in feature_ids:
        row = inventory.get(fid)
        if row is None:
            raise ManifestError(f"feature {fid} absent from the phase-1 inventory")
        if row.get("admissibility") not in (None, "", "ADMISSIBLE"):
            raise ManifestError(f"feature {fid} is not ADMISSIBLE in the phase-1 inventory")
        features.append({
            "feature_id": fid, "family": row.get("family"), "source": row.get("source"), "transform": row.get("transform"),
            "unit": row.get("unit"), "availability_time": row.get("availability_time"), "clock": row.get("clock"),
            "support_h": row.get("support_h"), "train_coverage": row.get("train_coverage"), "source_bytes": row.get("source_bytes"),
        })
    identity_source: dict[str, Any]
    phase1: dict[str, Any] = {
        "plan_sha256": plan["plan_sha256"],
        "inventory_sha256": plan.get("inventory_sha256"),
        "campaign_sha256": plan.get("campaign_sha256"),
        "config_sha256": plan.get("config_sha256"),
        "target_pack": plan.get("target_pack"),
        "completion_file": "PHASE_1_COMPLETE.json",
        "completion_sha256": sha256_file(state / "PHASE_1_COMPLETE.json"),
        "deployment_sha256": deployment.get("deployment_sha256"),
        "deployment_file": f"{pop_lower}_deployment.json",
    }
    envelope_path = state / f"{pop_lower}-phase1-final-envelope.json"
    if envelope_path.exists():
        run = json.loads(envelope_path.read_text())["run"]
        if run["run_id"] != identity:
            raise ManifestError(f"envelope run_id {run['run_id']} != requested identity {identity}")
        identity_source = {"file": envelope_path.name, "field": "run.run_id", "input_sha256": run.get("input_sha256"),
                           "campaign_sha256": run.get("campaign_sha256"), "code_sha256": run.get("code_sha256")}
        phase1.update({k: run.get(k) for k in ("input_sha256", "code_sha256")})
        phase1["envelope_sha256"] = json.loads(envelope_path.read_text()).get("envelope_sha256")
    else:
        suffix = identity.rsplit(":", 1)[-1]
        if not plan["plan_sha256"].startswith(suffix):
            raise ManifestError(f"identity suffix {suffix} is not a prefix of the phase-1 plan_sha256")
        identity_source = {"file": "PLAN.json", "field": "plan_sha256[:16]"}
        for key in ("final_envelope_sha256", "final_warehouse_receipt_sha256", "terminal_set_sha256",
                    "warehouse_identity_set_sha256", "warehouse_receipt_set_sha256", "gate_sha256"):
            if key in complete:
                phase1[key] = complete[key]
    for key in ("selected_feature_target_count", "selected_distinct_feature_count", "completion_sha256", "feature_count",
                "target_count", "selection_decision_count"):
        if key in complete:
            phase1[f"closure_{key}"] = complete[key]
    recon = state / "warehouse_reconciliation.json"
    if recon.exists():
        doc = json.loads(recon.read_text())
        phase1["warehouse_reconciliation_state"] = doc.get("state", "RECONCILED" if doc.get("complete") else None)
        phase1["warehouse_identities_sha256"] = doc.get("identities_sha256")
    features_file = bundle / "data" / f"{pop_lower}_features_train.parquet"
    targets_file = bundle / "data" / f"{pop_lower}_targets_train.parquet"
    import pyarrow.parquet as pq  # local import: freezing is the only reader of parquet metadata here
    meta = pq.read_metadata(features_file)
    train_rows = meta.num_rows
    schema_names = set(pq.read_schema(features_file).names)
    missing = [f for f in feature_ids if f not in schema_names]
    if missing:
        raise ManifestError(f"{len(missing)} phase-1 features absent from the TRAIN parquet, first {missing[:3]}")
    resources = deployment.get("resources", {})
    freq = next((r.get("frequency") for r in resources.values() if r.get("frequency")), None)
    bar_hours = {"1h": 1, "4h": 4}.get(freq)
    if bar_hours is None:
        raise ManifestError(f"cannot derive bar_hours from resource frequency {freq!r}")
    contract_source = "frozen_from_phase1_artifacts"
    if contract_path and Path(contract_path).exists():
        contract = json.loads(Path(contract_path).read_text())
        pop_contract = contract.get("populations", {}).get(population)
        if pop_contract:
            contract_source = f"adopted:{Path(contract_path).name}:{sha256_file(Path(contract_path))[:16]}"
            if pop_contract.get("identity") not in (None, identity):
                raise ManifestError("DATA contract identity disagrees with the phase-1 identity")
            if pop_contract.get("feature_count") not in (None, len(feature_ids)):
                raise ManifestError("DATA contract feature count disagrees with the phase-1 plan")
    manifest = build_manifest(
        population_id=population, identity=identity, identity_source=identity_source, phase1=phase1,
        data={"features_file": features_file.name, "features_sha256": sha256_file(features_file),
              "targets_file": targets_file.name, "targets_sha256": sha256_file(targets_file),
              "folds_file": folds_path.name, "folds_sha256": sha256_file(folds_path),
              "timestamp_column": "t_decision_utc", "row_id_column": "row_id", "bar_hours": bar_hours,
              "train_rows": train_rows},
        features=features, targets=_phase1_targets(deployment, population), folds=_phase1_folds(folds_doc),
        causal_supported=_selected_pairs(state, population, identity), contract_source=contract_source,
        data_root=bundle / "data",
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--population", required=True)
    f.add_argument("--identity", required=True)
    f.add_argument("--phase1-state-dir", required=True, type=Path)
    f.add_argument("--bundle-root", required=True, type=Path)
    f.add_argument("--contract", type=Path, default=None)
    f.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True, type=Path)
    v.add_argument("--data-root", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verb == "freeze":
        manifest = freeze_from_phase1(population=args.population, phase1_state_dir=args.phase1_state_dir,
                                      bundle_root=args.bundle_root, identity=args.identity, contract_path=args.contract)
        write_manifest(manifest, args.out)
        print(json.dumps({"population_id": manifest["population_id"], "identity": manifest["identity"],
                          "features": manifest["feature_count"], "targets": manifest["target_count"],
                          "folds": manifest["fold_count"], "expected_pairs": manifest["expected_pairs"],
                          "contract_source": manifest["contract_source"], "manifest_sha256": manifest["manifest_sha256"]}))
        return 0
    manifest = load_manifest(args.manifest)
    print(json.dumps(verify_data(manifest, args.data_root)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
