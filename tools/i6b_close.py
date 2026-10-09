#!/usr/bin/env python3
"""Authenticate and combine disjoint I6-B branch-pretraining shards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(document):
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _records(shard):
    return sorted(Path(shard).glob("*.record.json"))


def close(config_path, shards, output_dir):
    """Close shards only when every configured branch has one authentic donor."""
    config_path = Path(config_path)
    config = _read(config_path)
    expected = {branch["name"]: tuple(branch["features"]) for branch in config["branches"]}
    if len(expected) != len(config["branches"]):
        raise ValueError("DUPLICATE_CONFIG_BRANCH")
    expected_config_sha = config_sha256(config)
    found = {}
    train_sha = None

    for shard in map(Path, shards):
        record_paths = _records(shard)
        shard_names = [path.name.removesuffix(".record.json") for path in record_paths]
        shard_config = copy.deepcopy(config)
        shard_config["branches"] = [branch for branch in config["branches"]
                                     if branch["name"] in shard_names]
        expected_shard_config_sha = config_sha256(shard_config)
        for record_path in record_paths:
            record = _read(record_path)
            name = record.get("name")
            if name in found:
                raise ValueError(f"DUPLICATE_BRANCH:{name}")
            if name not in expected:
                raise ValueError(f"UNEXPECTED_BRANCH:{name}")
            if record.get("stage") != "branch_ae" or tuple(record.get("features", ())) != expected[name]:
                raise ValueError(f"BRANCH_IDENTITY_MISMATCH:{name}")
            if record.get("source_config_sha256") != expected_shard_config_sha:
                raise ValueError(f"CONFIG_DIGEST_MISMATCH:{name}")
            if record.get("reload_parity", {}).get("passed") is not True:
                raise ValueError(f"RELOAD_PARITY_FAILED:{name}")

            model = shard / f"{name}.keras"
            decoder = shard / f"{name}.decoder.keras"
            manifest_path = shard / f"{name}.manifest.json"
            provenance_path = shard / f"{name}.provenance.json"
            for path in (model, decoder, manifest_path, provenance_path):
                if not path.is_file():
                    raise ValueError(f"MISSING_BRANCH_ARTIFACT:{name}:{path.name}")
            if file_sha256(model) != record.get("donor_sha256"):
                raise ValueError(f"DONOR_DIGEST_MISMATCH:{name}")
            if file_sha256(decoder) != record.get("decoder_sha256"):
                raise ValueError(f"DECODER_DIGEST_MISMATCH:{name}")
            manifest = _read(manifest_path)
            provenance = _read(provenance_path)
            manifest_identity_sha = config_sha256(manifest.get("manifest", {}))
            if manifest_identity_sha != manifest.get("manifest_sha256") \
                    or manifest_identity_sha != record.get("donor_manifest_sha256"):
                raise ValueError(f"MANIFEST_DIGEST_MISMATCH:{name}")
            if manifest.get("model_sha256") != record["donor_sha256"]:
                raise ValueError(f"MANIFEST_MODEL_MISMATCH:{name}")
            if manifest.get("manifest", {}).get("name") != name:
                raise ValueError(f"MANIFEST_BRANCH_MISMATCH:{name}")
            learned = manifest.get("provenance", {}).get("learned_corpus", {})
            candidate_train_sha = record.get("train_input_sha256")
            if learned.get("kind") != "TRAIN_ONLY" or learned.get("data_sha256") != candidate_train_sha:
                raise ValueError(f"TRAIN_PROVENANCE_MISMATCH:{name}")
            if provenance.get("name") != name \
                    or provenance.get("source_config_sha256") != expected_shard_config_sha \
                    or provenance.get("train_input_sha256") != candidate_train_sha:
                raise ValueError(f"PROVENANCE_MISMATCH:{name}")
            if train_sha is None:
                train_sha = candidate_train_sha
            elif train_sha != candidate_train_sha:
                raise ValueError("MIXED_TRAIN_INPUTS")
            found[name] = {"shard": shard, "record": record}

    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError("INCOMPLETE_BRANCH_POPULATION:" + ",".join(missing))

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in expected:
        source = found[name]["shard"]
        copied = {}
        for suffix in (".keras", ".decoder.keras", ".manifest.json", ".provenance.json", ".record.json"):
            src = source / f"{name}{suffix}"
            dst = output / src.name
            shutil.copy2(src, dst)
            copied[src.name] = file_sha256(dst)
        rows.append({"name": name, "features": list(expected[name]), "artifacts": copied,
                     "best_validation_loss": found[name]["record"]["training"]["best_validation_loss"],
                     "seed": found[name]["record"]["training"]["settings"]["seed"]})

    report = {
        "schema": "predictor.i6b.branch_pretrain_closure.v1",
        "status": "COMPLETE",
        "configured_branches": len(expected),
        "completed_branches": len(rows),
        "source_config_sha256": expected_config_sha,
        "train_input_sha256": train_sha,
        "learned_corpus": "TRAIN_ONLY",
        "validation_read": False,
        "test_read": False,
        "fusion_state": "R0_NO_DONOR",
        "core_state": "R0_NO_DONOR",
        "branches": rows,
    }
    (output / "BRANCH_PRETRAIN.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(close(args.config, args.shard, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
