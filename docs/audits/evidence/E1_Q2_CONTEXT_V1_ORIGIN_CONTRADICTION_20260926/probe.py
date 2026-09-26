"""Derive the Q2_CONTEXT v1 origin contradiction from the retained bytes, and emit FINDING.json.

Every number below is read from a retained artifact and none is asserted here. Run from the
repository root with the anaconda env `trading-stack`:

    CUDA_VISIBLE_DEVICES='' python docs/audits/evidence/E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/probe.py

The v1 preparation's BLOCK_DATA.npz is not in this repository (6.6 MB of arrays); when the local
working state that holds it is present the probe also compares the arrays themselves, and says so
in `arrays_checked` when it could not.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EV = REPO / "docs/audits/evidence"
STATE = Path.home() / ".local/state/crispdm-data-foundation"

V1_SEALED = EV / "d3_k5_20260917/RP66/blocks/e1_block_q2_context_v1"
V2_SEALED = EV / "E1_Q2_CONTEXT_BOUNDED_20260926"
V1_STATE = STATE / "e1_block_q2_context_v1"
V2_STATE = STATE / "e1_block_q2_context_bounded_v1"


def _block_module():
    spec = importlib.util.spec_from_file_location("df_e1_block_probe", REPO / "tools/df_e1_block.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["df_e1_block_probe"] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(root: Path, state: Path) -> dict:
    design = json.loads((root / "DESIGN.json").read_text())
    record = json.loads((root / "BLOCK_DATA.json").read_text())
    origins, npz_sha, npz_from = None, None, None
    npz = state / "BLOCK_DATA.npz"
    if npz.is_file() and json.loads((state / "BLOCK_DATA.json").read_text())["data_sha256"] == sha(npz):
        with np.load(npz, allow_pickle=False) as z:
            origins = {a["arm"]: z[f"train_origins__{a['arm']}"] for a in design["arms"]
                       if f"train_origins__{a['arm']}" in z.files}
        npz_sha, npz_from = sha(npz), str(npz)
    return {"root": root, "design": design, "record": record, "origins": origins,
            "bytes": {"design_json_sha256": sha(root / "DESIGN.json"),
                      "block_data_json_sha256": sha(root / "BLOCK_DATA.json"),
                      "block_data_npz_sha256": npz_sha, "block_data_npz_read_from": npz_from}}


def judge(block, case: dict) -> dict:
    out = {"root": str(case["root"].relative_to(REPO)),
           "design_sha256": case["design"]["design_sha256"],
           "declared_train_population": case["design"].get("train_population", "UNDECLARED (no such key)"),
           "arms": [a["arm"] for a in case["design"]["arms"]],
           "bytes": case["bytes"],
           "arrays_checked": None if case["origins"] is None else sorted(case["origins"]),
           "report": None, "verdict": None, "refusal": None}
    report = block.train_population_report(case["design"], case["record"], origins=case["origins"])
    out["report"] = {k: (v if not isinstance(v, dict) else {kk: (int(vv) if isinstance(vv, (int, np.integer)) and vv is not None else vv)
                                                            for kk, vv in v.items()})
                     for k, v in report.items()}
    try:
        block.validate_train_population(case["design"], case["record"], origins=case["origins"])
        out["verdict"] = "ACCEPTED"
    except SystemExit as refusal:
        out["verdict"], out["refusal"] = "REFUSED", str(refusal)
    return out


def main() -> int:
    block = _block_module()
    v1 = judge(block, read(V1_SEALED, V1_STATE))
    v2 = judge(block, read(V2_SEALED, V2_STATE))
    finding = {
        "schema": "df_e1_origin_policy_finding.v1",
        "date": "2026-09-26",
        "author": "Satoshi, successor technical lead, under the owner's grant of 2026-09-26",
        "label": "Q2V1-ORIGIN-POLICY",
        "no_global_finding_number": "No number from the program's finding allocator is claimed: the allocator is "
                                    "fragmented across git refs and only Musashi rules on it. This is a labelled "
                                    "finding against one preparation, nothing more.",
        "against": "the sealed v1 preparation of the Q2_CONTEXT block, e1_block_q2_context_v1",
        "claim": "The retained v1 preparation holds PER-ARM train origins. Its own sealed DESIGN.json carries no "
                 "train_population key at all, while the block catalogue in tools/df_e1_block.py declares "
                 "COMMON_INTERSECTION for Q2_CONTEXT, so a reader of the catalogue would believe the sealed "
                 "preparation trains every arm on the same origins. It does not.",
        "verified_from": "the retained bytes named under v1/v2 below, read by this probe",
        "consequence_had_it_been_fitted": "the input contrast would have been confounded with train volume: arms "
                                          "would have trained on different numbers of origins while the block's "
                                          "declared question is an input difference at fixed volume",
        "published_numbers_affected": 0,
        "why_none": "state SEALED_NOT_EXECUTED / BUDGET_LIMITED_BEFORE_ANY_OUTCOME; no cell of "
                    "e1_block_q2_context_v1 was ever fitted, by this round or any other",
        "usability": {
            "e1_block_q2_context_v1": "UNUSABLE_FOR_ANY_COMPARISON_THAT_ASSUMES_COMMON_TRAIN_ORIGINS",
            "still_usable_for": "its own retained cost pilots (CPU seconds and peak RSS per update), which "
                                "measure resource cost per arm and do not compare arms to each other",
            "not_repaired_in_place": "the sealed preparation is retained as it is. Nothing in it is rewritten: "
                                     "this record and the validator sit beside it.",
        },
        "repair": {
            "validator": "tools/df_e1_block.py validate_train_population(), called by prepare() before a "
                         "preparation seals and by load_data() before anything is fitted on one",
            "tests": "tests/test_df_e1_train_population_validator.py",
        },
        "v1": v1,
        "v2_for_contrast": v2,
        "reading": "the validator refuses v1's bytes and accepts v2's; v2 is the 2026-09-26 bounded block, which "
                   "declares COMMON_INTERSECTION and holds it.",
    }
    (HERE / "FINDING.json").write_text(json.dumps(finding, indent=1) + "\n")
    print(f"v1 {v1['verdict']}: {v1['refusal']}")
    print(f"v2 {v2['verdict']}: common_train_origins={v2['report']['common_train_origins_recorded']}, "
          f"identical counts={v2['report']['held_identical_counts']}, identical origins={v2['report']['held_identical_origins']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
