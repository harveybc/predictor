#!/usr/bin/env python3
"""C128 x C129: a materialized synthetic unit as a common dataset contract.

The unit's arrays are verified against the digests its generator recorded
before anything is described. Every file is bound by the sha256 of its bytes,
with a role that tells a profiler what it may read:

  OBSERVED        observed_signal.npy: the only array a profile may consume
  MISSING_MASK    missing_mask.npy
  CLEAN_TRUTH     clean_signal.npy: known truth, for calibration only
  NOISE_TRUTH     additive_noise.npy: known truth, for calibration only
  METRIC_SUPPORT  metric_support.npy
  EVENTS          events.json: known event locations, for preservation tests
  UNIT_RECORD     UNIT.json

Semantics and unit are known by construction; their evidence is the
generator's code digest. License is NOT_APPLICABLE_GENERATED. Time is the
sample index. Partitions are the unit's own, materialized before any
statistic. The unit record is kept verbatim in original_fields.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


C = _load("df_contract")
BANK = _load("df_synthetic_bank")

ROLES = {"observed_signal.npy": "OBSERVED", "missing_mask.npy": "MISSING_MASK",
         "clean_signal.npy": "CLEAN_TRUTH", "additive_noise.npy": "NOISE_TRUTH",
         "metric_support.npy": "METRIC_SUPPORT", "events.json": "EVENTS", "UNIT.json": "UNIT_RECORD"}
NA = "NOT_APPLICABLE"


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_unit(unit_dir: Path) -> dict:
    unit_dir = Path(unit_dir)
    rec = json.loads((unit_dir / "UNIT.json").read_text())
    for name, digest in rec["digests"].items():
        arr = np.load(unit_dir / f"{name}.npy", allow_pickle=False)
        if BANK.array_digest(arr) != digest:
            raise C.ContractRefusal([f"{rec['unit_id']}: {name} does not match its recorded digest"])
    if rec.get("events_sha256") and _sha_file(unit_dir / "events.json") != rec["events_sha256"]:
        raise C.ContractRefusal([f"{rec['unit_id']}: events.json does not match its recorded digest"])
    if rec.get("partitions_materialized_before_any_statistic") is not True:
        raise C.ContractRefusal([f"{rec['unit_id']}: partitions were not materialized before statistics"])
    return rec


def unit_contract(unit_dir: Path) -> dict:
    unit_dir = Path(unit_dir)
    rec = verify_unit(unit_dir)
    facts = BANK.contract_fields(rec)
    gen = facts["generator"]
    dataset_id = f"synthetic.{gen['version']}.{rec['unit_id']}"
    evidence = [{"source": f"{gen['module']} {gen['version']}", "sha256": gen["code_sha256"]}]
    files = []
    for name, role in ROLES.items():
        p = unit_dir / name
        if not p.is_file():
            raise C.ContractRefusal([f"{rec['unit_id']}: missing {name}"])
        files.append({"name": name, "bytes": p.stat().st_size, "sha256": _sha_file(p), "role": role})
    variables = [
        C.variable(dataset_id, name,
                   semantics={"type": f"SYNTHETIC_{rec['family'].upper()}_OBSERVED_WITH_{rec['perturbation'].upper()}",
                              "description": "generated clean component plus separately generated perturbation; "
                                             "parameters in original_fields.unit_record",
                              "evidence": evidence},
                   unit={"value": facts["unit"], "evidence": evidence},
                   producer={"kind": "GENERATOR", "reference": f"{gen['module']}@{gen['code_sha256']}"},
                   physical_type="float64", frequency_nominal_seconds=NA, event_time="SAMPLE_INDEX",
                   available_time_rule="SAMPLE_INDEX",
                   missingness={"encoding": facts["missingness_encoding"], "policy": facts["missingness_policy"]},
                   sentinels={"values": [], "policy": "NONE_GENERATED"},
                   role="INPUT_CANDIDATE", license_state="NOT_APPLICABLE_GENERATED",
                   original_fields={"variable_index": i})
        for i, name in enumerate(facts["variable_names"])]
    fractions = dict(BANK.PARTITION_FRACTIONS)
    doc = {
        "schema": C.DATASET_SCHEMA, "dataset_id": dataset_id, "version": gen["version"], "bank": "SYNTHETIC",
        "files": files, "content_sha256": "", "contract_sha256": "",
        "source": {"provider": "predictor synthetic generator", "official_url": NA, "citation": NA, "doi": NA,
                   "upstream_owner": NA},
        "license": {"state": "NOT_APPLICABLE_GENERATED", "id": NA, "url": NA, "text_sha256": "UNAVAILABLE",
                    "attribution_required": "NO", "redistribution": NA, "derivatives": NA, "evidence": []},
        "time": {"frequency_nominal_seconds": NA, "timezone": NA, "timestamp_meaning": "SAMPLE_INDEX",
                 "range_start": "0", "range_end": str(rec["n_samples"] - 1), "availability_rule": "SAMPLE_INDEX",
                 "availability_delay_seconds": NA},
        "panel": {"aligned_common_grid": True, "n_series": rec["n_variables"],
                  "alignment_rule": "one generated sample grid shared by every variable"},
        "partitions": {"scheme": "CHRONOLOGICAL_FRACTIONS",
                       "fractions": {k: float(fractions[k]) for k in C.PARTITIONS},
                       "boundaries": {k: list(rec["partitions"][k]) for k in C.PARTITIONS},
                       "sealed_periods_excluded": [], "frozen_before_profile": True},
        "dependence": [], "variables": variables,
        "original_fields": {"unit_record": rec},
    }
    return C.seal(doc)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("unit_dirs", nargs="+", type=Path)
    a = ap.parse_args(argv)
    for d in a.unit_dirs:
        c = unit_contract(d)
        print(json.dumps({"dataset_id": c["dataset_id"], "contract_sha256": c["contract_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
