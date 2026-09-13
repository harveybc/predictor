"""Bank scenarios for the C113/C117 population attacks.

Copied verbatim (by AST span) from the frozen PRE
docs/audits/evidence/repro_runs/c106_c121_pre_2026_09_12.py, lines
242-392, so that tests and the POST write exactly the bytes the PRE
attacked; test_fixture_bytes_equal_pre checks each input digest.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


PANELS, COLS = 6, 5


def _dsha(tag: str) -> str:
    return hashlib.sha256(f"fixture-bytes:{tag}".encode()).hexdigest()


def bank_rows() -> dict:
    """A bank in which every member satisfies all eight C118 conditions."""
    dag, terms, census, temporal = [], [], [], []
    for p in range(PANELS):
        ds = f"panel_{p}"
        dsha = _dsha(ds)
        for c in range(COLS):
            col = f"x_{c}"
            vid = f"{ds}:{col}"
            dag.append({"dataset_id": ds, "dataset_sha256": dsha, "column": col,
                        "class": "CAUSAL_ACTIVE", "binding": {"complete": True}})
            terms.append({"dataset_id": ds, "dataset_sha256": dsha, "column": col,
                          "variable_id": vid, "layer": "INDEPENDENTLY_RECOMPUTED",
                          "semantic_state": "NUMERIC_MEASURABLE", "physical_type": "float64",
                          "observations": 4096, "missing_fraction": 0.0})
            census.append({"variable_id": vid, "dataset_id": ds, "dataset_sha256": dsha,
                           "column": col, "physical_type": "float64",
                           "semantic_type": "continuous_measurement",
                           "semantics": "continuous_measurement", "role": "input_feature",
                           "unit": "1", "license": "CC-BY-4.0", "license_source": "fixture",
                           "missing_policy": "EXCLUDE_ROW_NO_IMPUTATION",
                           "sentinel_policy": "NO_SENTINELS_DECLARED", "producer": "fixture",
                           "symbol": "fixture", "lookback_bars": 1,
                           "evidence_sha256": _dsha("evidence:" + vid)})
        temporal.append({"dataset": {"dataset_id": ds, "dataset_sha256": dsha},
                         "contract_sha256": _dsha("contract:" + ds),
                         "mask_artifact": {"dataset_id": ds, "dataset_sha256": dsha,
                                           "sha256": _dsha("mask:" + ds)}})
    return {"dag": dag, "terminals": terms, "census": census, "temporal": temporal,
            "census_extra": {}}


def _find(rows, ds="panel_0", col="x_0"):
    return next(r for r in rows if r.get("dataset_id") == ds and r.get("column") == col)


def _mut_defect(b):
    b["terminals"] = []
    b["census"] = [{"variable_id": "not_a_member", "semantics": "UNKNOWN", "role": "UNKNOWN",
                    "license": "UNKNOWN", "unit": "UNKNOWN"}]


def _mut_other_variable(b):
    t = _find(b["terminals"]); t["column"] = "x_9"; t["variable_id"] = "panel_0:x_9"


def _mut_census_other_dataset(b):
    c = _find(b["census"]); c["dataset_id"] = "panel_other"; c["dataset_sha256"] = _dsha("panel_other")


def _mut_source_digest(b):
    _find(b["terminals"])["dataset_sha256"] = _dsha("different-bytes")


def _mut_duplicate_column(b):
    b["census"].remove(_find(b["census"], col="x_4"))
    b["census"].append(copy.deepcopy(_find(b["census"])))


def _mut_license_absent(b):
    del _find(b["census"])["license"]


def _mut_role_absent(b):
    del _find(b["census"])["role"]


def _mut_temporal_other_digest(b):
    b["temporal"][0]["dataset"]["dataset_sha256"] = _dsha("other-contract-dataset")


def _mut_mask_other_dataset(b):
    b["temporal"][0]["mask_artifact"]["dataset_id"] = "panel_5"
    b["temporal"][0]["mask_artifact"]["dataset_sha256"] = _dsha("panel_5")


def _mut_only_in_aggregates(b):
    # panel_0:x_4 keeps its DAG node and an aggregate claim, but has no
    # terminal and no census row; cardinalities are kept with rows for a
    # column that is in no DAG.
    for key in ("terminals", "census"):
        row = _find(b[key], col="x_4")
        row.update(dataset_id="panel_5", dataset_sha256=_dsha("panel_5"), column="x_9",
                   variable_id="panel_5:x_9")
    b["census_extra"] = {"aggregates": {"panel_0": {"members": 5}}}


def _mut_role_unknown(b):
    _find(b["census"])["role"] = "UNKNOWN"


def _mut_license_unknown(b):
    _find(b["census"])["license"] = "UNKNOWN"


def _mut_semantically_unresolved(b):
    t = _find(b["terminals"]); t["layer"] = "SEMANTICALLY_UNRESOLVED"
    t["semantic_state"] = "SEMANTIC_TYPE_UNRESOLVED"


def _mut_date_as_integer(b):
    t = _find(b["terminals"]); t["physical_type"] = "int64"
    t["semantic_state"] = "SEMANTIC_TYPE_UNRESOLVED"
    c = _find(b["census"]); c["physical_type"] = "int64"; c["semantic_type"] = "datetime"


SCENARIOS = {
    "control_complete_bank": None,
    "C117.defect_zero_terminals_zero_semantics": _mut_defect,
    "C117.terminal_of_other_variable": _mut_other_variable,
    "C117.census_of_other_dataset": _mut_census_other_dataset,
    "C117.source_digest_distinct": _mut_source_digest,
    "C117.duplicate_column": _mut_duplicate_column,
    "C117.license_absent": _mut_license_absent,
    "C117.role_absent": _mut_role_absent,
    "C117.temporal_contract_other_digest": _mut_temporal_other_digest,
    "C117.mask_other_dataset": _mut_mask_other_dataset,
    "C117.member_only_in_aggregates": _mut_only_in_aggregates,
    "C113.recomputed_role_unknown": _mut_role_unknown,
    "C113.recomputed_license_unknown": _mut_license_unknown,
    "C113.semantically_unresolved_terminal": _mut_semantically_unresolved,
    "C113.date_stored_as_integer": _mut_date_as_integer,
}


def write_bank(root: Path, scenario: str) -> tuple[dict, str]:
    b = bank_rows()
    if SCENARIOS[scenario]:
        SCENARIOS[scenario](b)
    (root / "terminals").mkdir()
    for i, t in enumerate(b["terminals"]):
        (root / "terminals" / f"{i:03d}.json").write_text(json.dumps(t, sort_keys=True))
    (root / "dag.json").write_text(json.dumps({"nodes": b["dag"]}, sort_keys=True))
    (root / "census.json").write_text(json.dumps(dict(b["census_extra"], variables=b["census"]), sort_keys=True))
    contracts = []
    for i, c in enumerate(b["temporal"]):
        p = root / f"temporal_{i}.json"
        p.write_text(json.dumps(c, sort_keys=True))
        contracts.append(p)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.json")):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return {"terminals_v4": root / "terminals", "dag_v4": root / "dag.json",
            "census": root / "census.json", "temporal_contracts": contracts}, h.hexdigest()
