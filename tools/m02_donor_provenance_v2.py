"""Declare schema-2 provenance ALONGSIDE M02's schema-1 donor sidecars, derived only from M02's own records.

    python tools/m02_donor_provenance_v2.py --donors <DONORS_FOR_R1_R2_ecl_v2_s7> --engine-provenance
        predictor_plugins/modular_temporal/provenance.py --out <amendment.json> [--dry-run]

Standard library only (the engine's provenance module is loaded by file path; no TensorFlow), so it runs
under a 512M cap. For every donor listed in DONOR_INDEX.json (pinned sha256 8a6bb203...):

1. BEFORE: the .keras, the schema-1 .manifest.json and M02's .provenance.json must match the sha256 in the
   index; otherwise that donor is refused (not declared).
2. RULE R-AE-TRAINONLY-1 (stated, not inferred). OPERATIONAL + TRAIN_ONLY is declared only if ALL hold in
   M02's records: provenance schema modular.{branch,core}_donor.provenance.v1; label GOVERNED_RESOURCE;
   input_provenance governed_resource; input_manifest_sha256 == the index's engine_pin.manifest_sha256
   (eca31ec1, the governed ECL TRAIN resource); admissible_declaration_sha256 == engine_pin.declaration_sha256
   (ca1098ed); stage in {branch_ae, core_ae} (an autoencoder whose training target is its own input window,
   so no future target conditions it); source_config_sha256 == engine_pin.model_config_sha256 (4d5402a0, the
   R0 config: fitted from initialization, no pretrained source); train_input_sha256 and
   train_validation_input_sha256 present. Branches: the record's internal-validation reconstruction MAE/MSE
   gives reconstruction MEASURED in the model-input space; core: PRETRAIN.json core.reconstruction
   train_validation, in the fused-branch-latent space. A missing value leaves that field UNKNOWN.
   Any failed condition leaves the whole donor UNKNOWN with the reason; nothing is hand-edited.
3. AFTER: every original file's sha256 is re-checked; the amendment lists each new alongside sidecar with
   its sha256.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

INDEX_SHA256 = "8a6bb20389bdb2b03ad0dbc2e39feb973efc05462f4919720624b2dbeb7025a4"
RULE = "R-AE-TRAINONLY-1"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_engine(path):
    spec = importlib.util.spec_from_file_location("engine_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def originals(pre, entry):
    name = entry["name"]
    return {"keras": (pre / f"{name}.keras", entry["keras_sha256"]),
            "manifest": (pre / f"{name}.manifest.json", entry["manifest_json_sha256"]),
            "provenance": (pre / f"{name}.provenance.json", entry["provenance_json_sha256"])}


def derive(pre, entry, pin, pretrain):
    """(declared, sources, reasons): declared is None when the rule does not hold."""
    name = entry["name"]
    prov_path = pre / f"{name}.provenance.json"
    prov = json.loads(prov_path.read_text())
    sources = [{"file": prov_path.name, "sha256": sha(prov_path)}]
    expected_schema = "modular.branch_donor.provenance.v1" if entry["stage"] == "branch_ae" \
        else "modular.core_donor.provenance.v1"
    checks = {
        "schema": prov.get("schema") == expected_schema,
        "label": prov.get("label") == "GOVERNED_RESOURCE",
        "input_provenance": prov.get("input_provenance") == "governed_resource",
        "input_manifest": prov.get("input_manifest_sha256") == pin["manifest_sha256"],
        "admissible_declaration": prov.get("admissible_declaration_sha256") == pin["declaration_sha256"],
        "stage": prov.get("stage") in ("branch_ae", "core_ae") and prov.get("stage") == entry["stage"],
        "r0_source_config": prov.get("source_config_sha256") == pin["model_config_sha256"],
    }
    train_sha = prov.get("train_input_sha256")
    val_sha = prov.get("train_validation_input_sha256")
    if entry["stage"] == "core_ae":
        fused = prov.get("fused_materialization") or {}
        train_sha = train_sha or fused.get("train")
        val_sha = val_sha or fused.get("train_validation")
    checks["train_input_sha"] = bool(train_sha) and len(train_sha) == 64
    checks["train_validation_input_sha"] = bool(val_sha) and len(val_sha) == 64
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        return None, sources, failed
    reconstruction = {"state": "UNKNOWN"}
    if entry["stage"] == "branch_ae":
        record_path = pre / f"{name}.record.json"
        sources.append({"file": record_path.name, "sha256": sha(record_path)})
        rv = (json.loads(record_path.read_text()).get("reconstruction") or {}).get("train_validation") or {}
        space = "model_input (ECL z_train, the governed NPZ's metric space)"
    else:
        sources.append({"file": "PRETRAIN.json", "sha256": sha(pre / "PRETRAIN.json")})
        rv = ((pretrain.get("core") or {}).get("reconstruction") or {}).get("train_validation") or {}
        space = "fused branch latent (materialized fusion output)"
    if isinstance(rv.get("MAE"), (int, float)) and isinstance(rv.get("MSE"), (int, float)):
        reconstruction = {"state": "MEASURED", "mae_z": rv["MAE"], "mse_z": rv["MSE"], "rows": rv.get("rows"),
                          "split": "train_validation (purged internal tail of TRAIN)", "space": space}
    declared = {"conditioning_contract": "OPERATIONAL",
                "learned_corpus": {"kind": "TRAIN_ONLY", "pretrained_weights_source": None,
                                   "dataset_id": f"governed_resource:manifest:{pin['manifest_sha256']}",
                                   "data_sha256": train_sha,
                                   "support": "TRAIN only: AE-train windows + purged internal-validation tail "
                                              f"(train_validation input sha256 {val_sha})",
                                   "admissible_declaration_sha256": pin["declaration_sha256"],
                                   "source_config_sha256": pin["model_config_sha256"]},
                "reconstruction": reconstruction}
    return declared, sources, []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--donors", required=True)
    p.add_argument("--engine-provenance", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    root = Path(a.donors)
    if sha(root / "DONOR_INDEX.json") != INDEX_SHA256:
        raise SystemExit("DONOR_INDEX.json is not the pinned index")
    index = json.loads((root / "DONOR_INDEX.json").read_text())
    pre, pin = Path(index["directory"]), index["engine_pin"]
    pretrain = json.loads((pre / "PRETRAIN.json").read_text())
    pv = load_engine(a.engine_provenance)
    rows, refused = [], []
    for entry in index["donors"]:
        files = originals(pre, entry)
        before = {k: sha(path) for k, (path, _) in files.items()}
        mismatched = [k for k, (path, expected) in files.items() if before[k] != expected]
        if mismatched:
            refused.append({"name": entry["name"], "reason": f"BEFORE_SHA_MISMATCH {mismatched}"})
            continue
        declared, sources, failed = derive(pre, entry, pin, pretrain)
        row = {"name": entry["name"], "stage": entry["stage"], "rule": RULE,
               "originals_before": before, "sources": sources}
        if declared is None:
            row.update(status="UNKNOWN_KEPT", reason=f"rule conditions not met: {failed}")
            rows.append(row)
            continue
        if not a.dry_run:
            result = pv.write_alongside(files["keras"][0], declared, {"rule": RULE, "sources": sources,
                                                                     "index_sha256": INDEX_SHA256})
            target = Path(result["path"])
            row.update(status=result["status"], alongside=target.name, alongside_sha256=sha(target),
                       conditioning_contract="OPERATIONAL", corpus_kind="TRAIN_ONLY",
                       reconstruction_state=declared["reconstruction"]["state"])
        else:
            row.update(status="WOULD_WRITE", reconstruction_state=declared["reconstruction"]["state"])
        after = {k: sha(path) for k, (path, _) in files.items()}
        row["originals_unchanged"] = after == before
        if after != before:
            raise SystemExit(f"ORIGINAL_CHANGED for {entry['name']}: {before} -> {after}")
        rows.append(row)
    amendment = {"schema": "m02.donor_index.amendment.v1", "amendment": 1, "base_index_sha256": INDEX_SHA256,
                 "base_index_path": str(root / "DONOR_INDEX.json"), "rule": RULE,
                 "rule_text": __doc__.split("2. RULE")[1].split("3. AFTER")[0].strip(),
                 "engine_provenance_module_sha256": sha(a.engine_provenance),
                 "donors": rows, "refused": refused,
                 "counts": {"listed": len(index["donors"]),
                            "declared": sum(r["status"] in ("WRITTEN", "ALREADY_CURRENT", "WOULD_WRITE") for r in rows),
                            "unknown_kept": sum(r["status"] == "UNKNOWN_KEPT" for r in rows),
                            "refused": len(refused)},
                 "statement": "Originals (.keras, schema-1 .manifest.json, M02 .provenance.json) not modified; "
                              "sha256 checked before and after each write. Countersign: M02, M04."}
    Path(a.out).write_text(json.dumps(amendment, indent=1, sort_keys=True) + "\n")
    print(json.dumps(amendment["counts"]))


if __name__ == "__main__":
    main()
