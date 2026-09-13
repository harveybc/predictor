#!/usr/bin/env python3
"""C140 (order 2026-09-12): coverage of the data foundation, cell by cell.

The expected grid is declared first: dataset x variable x metric x operator,
from the contracts and the declared metric and operator lists. Every cell
then gets exactly one state, derived from the rows that exist for it:

  RESULT        at least one COMPLETED row
  FAILED        no result, and at least one FAILED or REFUSED row
  INCONCLUSIVE  no result or failure, and at least one INCONCLUSIVE row
  UNAVAILABLE   only UNAVAILABLE or REJECTED rows
  NOT_RUN       no row at all

A row that matches no declared cell is reported as UNDECLARED, never
silently absorbed. The aggregate counts are derived from the cell ledger and
bound to its digest: no count stands in for the member-by-member ledger.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

NO_OPERATOR = "NONE"
STATES = ("RESULT", "FAILED", "INCONCLUSIVE", "UNAVAILABLE", "NOT_RUN")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_grid(contracts: list[dict], metrics: list[str], operators: list[str] | None = None) -> list[tuple]:
    ops = list(operators) if operators else [NO_OPERATOR]
    if len(set(metrics)) != len(metrics) or len(set(ops)) != len(ops):
        raise ValueError("REFUSED: duplicate metric or operator in the declared grid")
    cells = []
    for c in contracts:
        for v in c["variables"]:
            for m in metrics:
                for o in ops:
                    cells.append((c["dataset_id"], v["variable_id"], m, o))
    if len(set(cells)) != len(cells):
        raise ValueError("REFUSED: duplicate cell in the declared grid")
    return cells


def _cell_state(statuses: list[str]) -> str:
    s = set(statuses)
    if "COMPLETED" in s:
        return "RESULT"
    if s & {"FAILED", "REFUSED"}:
        return "FAILED"
    if "INCONCLUSIVE" in s:
        return "INCONCLUSIVE"
    if s & {"UNAVAILABLE", "REJECTED", "NOT_RUN"}:
        return "UNAVAILABLE" if s & {"UNAVAILABLE", "REJECTED"} else "NOT_RUN"
    return "NOT_RUN"


def build_matrix(cells: list[tuple], rows: list[dict]) -> dict:
    """rows: dicts with dataset_id, variable_id, metric, status and an optional operator."""
    declared = set(cells)
    seen: dict[tuple, list[str]] = {}
    undeclared = []
    for r in rows:
        key = (r["dataset_id"], r["variable_id"], r["metric"], r.get("operator") or NO_OPERATOR)
        if key not in declared:
            undeclared.append({"cell": list(key), "status": r["status"]})
            continue
        seen.setdefault(key, []).append(r["status"])
    ledger = [{"dataset_id": d, "variable_id": v, "metric": m, "operator": o,
               "state": _cell_state(seen.get((d, v, m, o), [])), "rows": len(seen.get((d, v, m, o), []))}
              for (d, v, m, o) in sorted(declared)]
    counts = Counter(c["state"] for c in ledger)
    ledger_sha = _sha(ledger)
    return {"schema": "crispdm.data_foundation.coverage_matrix.v1",
            "cells": len(ledger), "ledger": ledger, "ledger_sha256": ledger_sha,
            "counts_derived_from_ledger": {s: counts.get(s, 0) for s in STATES},
            "counts_bound_to": ledger_sha,
            "undeclared_rows": undeclared}


def verify_counts(matrix: dict) -> bool:
    counts = Counter(c["state"] for c in matrix["ledger"])
    return (_sha(matrix["ledger"]) == matrix["ledger_sha256"] == matrix["counts_bound_to"]
            and matrix["counts_derived_from_ledger"] == {s: counts.get(s, 0) for s in STATES})


def coverage_rows(matrix: dict, *, run_id: str, code_sha256: str) -> list[dict]:
    """The ledger as df_fact_coverage rows."""
    return [{"run_id": run_id, "dataset_id": c["dataset_id"], "variable_id": c["variable_id"],
             "metric": c["metric"], "operator": c["operator"], "state": c["state"], "code_sha256": code_sha256}
            for c in matrix["ledger"]]


# =============================================================================
# C170 (order 2026-09-13): coverage v2. Everything above is v1, kept unchanged
# as the superseded history (schema coverage_matrix.v1, table df_fact_coverage).
# =============================================================================
"""Coverage v2.

Cell = dataset x variable x partition x metric x operator x policy. The grid
DECLARES applicability before any row is read, from the contract, the variable
type and the policy, never from the absence of a row:

  R_TYPE                 numeric metrics do not apply to TIMESTAMP or TEXT columns;
                         an UNKNOWN type (dataset never classified) is UNDETERMINED
  R_SHIFT_TRAIN          *_vs_train metrics do not apply to the train partition
  R_MATRIX_SHARE         pc1_loading / pc1_common_variance_share /
                         private_residual_variance_share apply only to train, to
                         datasets with at least two numeric variables, and to
                         variables inside the declared matrix cap (the first
                         MAX_VARIABLES_MATRIX numeric variables by variable_id)
  R_UNIT_ROOT_BLOCK      *_block_{start,middle,end,spread} apply only when the
                         partition's longest finite run exceeds exact_max_n; a
                         partition no longer than exact_max_n rows (contract) makes
                         them NOT_APPLICABLE without reading any row; a longer
                         partition with no recorded run length is UNDETERMINED
  R_UNIT_ROOT_EXACT      adf_/kpss_ statistic and pvalue apply only when the run is
                         no longer than exact_max_n (beyond it the policy tests blocks)
  R_DEFAULT              everything else applies

Row status -> row state (ROW_STATE_V2):
  COMPLETED -> RESULT; INCONCLUSIVE -> INCONCLUSIVE; UNAVAILABLE -> UNAVAILABLE;
  NOT_RUN -> NOT_RUN, except NOT_RUN with reason NOT_RUN_RESOURCE_BOUND(_PREREQUISITE)
  -> RESOURCE_EXCEEDED (the declared estimate exceeded the budget; the library never ran);
  REFUSED -> REFUSED (a typed abstention, never a numeric failure);
  FAILED -> FAILED; RESOURCE_EXCEEDED -> RESOURCE_EXCEEDED; UNCERTAIN -> UNCERTAIN;
  REJECTED -> RESULT: a rejection is a scientific decision that was reached, so the
  cell has a result; it is never UNAVAILABLE. The ledger keeps the raw status counts,
  so a REJECTED member stays visible.

Cell state when several row states coexist (PRECEDENCE_V2, tested):
  1. RESULT together with FAILED, RESOURCE_EXCEEDED or UNCERTAIN -> UNCERTAIN
     (execution evidence contradicts itself);
  2. otherwise the first present of RESULT > INCONCLUSIVE > REFUSED >
     RESOURCE_EXCEEDED > FAILED > UNCERTAIN > UNAVAILABLE > NOT_RUN.

Declared NOT_APPLICABLE cell: NOT_APPLICABLE (source DECLARATION) when it has no
rows or only NOT_RUN / UNAVAILABLE rows; UNCERTAIN (source ROWS) when any row
carries another state, because rows contradict the declaration.
Applicable or UNDETERMINED cell: from its rows (source ROWS); with no rows, the
dataset's non-completed terminal state when there is one (source
DATASET_TERMINAL); otherwise NOT_RUN (source NO_EVIDENCE).

Totals are derived from the ledger member by member and bound to its digest.
"""
import importlib.util as _ilu
import re as _re
import sys as _sys
from pathlib import Path as _Path


def _plan():
    if "df_memory_plan" in _sys.modules:
        return _sys.modules["df_memory_plan"]
    spec = _ilu.spec_from_file_location("df_memory_plan", _Path(__file__).with_name("df_memory_plan.py"))
    mod = _ilu.module_from_spec(spec)
    _sys.modules["df_memory_plan"] = mod
    spec.loader.exec_module(mod)
    return mod


STATES_V2 = ("RESULT", "INCONCLUSIVE", "UNAVAILABLE", "NOT_APPLICABLE", "NOT_RUN", "REFUSED", "FAILED",
             "RESOURCE_EXCEEDED", "UNCERTAIN")
ROW_STATE_V2 = {"COMPLETED": "RESULT", "INCONCLUSIVE": "INCONCLUSIVE", "UNAVAILABLE": "UNAVAILABLE",
                "NOT_RUN": "NOT_RUN", "REFUSED": "REFUSED", "FAILED": "FAILED",
                "RESOURCE_EXCEEDED": "RESOURCE_EXCEEDED", "UNCERTAIN": "UNCERTAIN", "REJECTED": "RESULT"}
RESOURCE_BOUND_REASONS = ("NOT_RUN_RESOURCE_BOUND", "NOT_RUN_RESOURCE_BOUND_PREREQUISITE")
PRECEDENCE_V2 = ("RESULT", "INCONCLUSIVE", "REFUSED", "RESOURCE_EXCEEDED", "FAILED", "UNCERTAIN", "UNAVAILABLE",
                 "NOT_RUN")
EXECUTION_FAILURES = frozenset({"FAILED", "RESOURCE_EXCEEDED", "UNCERTAIN"})
TERMINAL_STATE_V2 = {"INCONCLUSIVE": "INCONCLUSIVE", "REFUSED": "REFUSED", "FAILED": "FAILED",
                     "RESOURCE_EXCEEDED": "RESOURCE_EXCEEDED", "UNCERTAIN": "UNCERTAIN"}
APPLICABILITY = ("APPLICABLE", "NOT_APPLICABLE", "UNDETERMINED")
VARIABLE_TYPES = ("NUMERIC", "TEXT", "TIMESTAMP", "UNKNOWN")
PARTITIONS_V2 = ("train", "calibration", "confirmation")
NOT_PARTITIONED = "NOT_PARTITIONED"
NO_POLICY = "NONE"
MATRIX_SHARE_METRICS = ("pc1_loading", "pc1_common_variance_share", "private_residual_variance_share")
UNIT_ROOT_EXACT_METRICS = tuple(f"{t}_{w}" for t in ("adf", "kpss") for w in ("statistic", "pvalue"))
UNIT_ROOT_BLOCK_METRICS = tuple(f"{t}_{w}_block_{o}" for t in ("adf", "kpss") for w in ("statistic", "pvalue")
                                for o in ("start", "middle", "end", "spread"))
_BLOCK_RE = _re.compile(r"^(adf|kpss)_(statistic|pvalue)_block_(start|middle|end|spread)$")
RULES = {
    "R_TYPE": "numeric metrics do not apply to TIMESTAMP or TEXT columns; UNKNOWN type is UNDETERMINED",
    "R_SHIFT_TRAIN": "a *_vs_train comparison does not apply to the train partition",
    "R_MATRIX_SHARE": "PC1 shares apply only to train, V_numeric >= 2, variables inside the matrix cap",
    "R_UNIT_ROOT_BLOCK": "block unit-root metrics apply only when the longest finite run exceeds exact_max_n",
    "R_UNIT_ROOT_EXACT": "exact unit-root metrics apply only when the longest finite run is <= exact_max_n",
    "R_DEFAULT": "applies",
}


def row_state_v2(status: str, reason: str = "") -> str:
    if status not in ROW_STATE_V2:
        raise ValueError(f"REFUSED: unknown row status {status!r}")
    if status == "NOT_RUN" and (reason or "") in RESOURCE_BOUND_REASONS:
        return "RESOURCE_EXCEEDED"
    return ROW_STATE_V2[status]


def cell_state_v2(row_states) -> str:
    s = set(row_states)
    unknown = s - set(STATES_V2)
    if unknown:
        raise ValueError(f"REFUSED: unknown row states {sorted(unknown)}")
    if not s:
        return "NOT_RUN"
    if "RESULT" in s and s & EXECUTION_FAILURES:
        return "UNCERTAIN"
    return next(st for st in PRECEDENCE_V2 if st in s)


def policy_for(metric: str) -> str:
    if metric in UNIT_ROOT_EXACT_METRICS or _BLOCK_RE.match(metric):
        return _plan().UNIT_ROOT_POLICY_SHA256
    return NO_POLICY


def variable_type(variable: dict, skipped_ids=None, profiled_ids=None) -> str:
    """TIMESTAMP by name or role; TEXT when the runner skipped it as non-numeric; NUMERIC when it was profiled;
    UNKNOWN when the dataset was never classified or the column could not be read."""
    if variable.get("name") == "timestamp" or variable.get("role") == "TIMESTAMP":
        return "TIMESTAMP"
    vid = variable["variable_id"]
    if skipped_ids is not None and vid in skipped_ids:
        return "TEXT" if skipped_ids[vid] == "NON_NUMERIC_NOT_PROFILED" else "UNKNOWN"
    if profiled_ids is not None and vid in profiled_ids:
        return "NUMERIC"
    return "UNKNOWN"


def applicability(metric: str, vtype: str, partition: str, facts: dict) -> tuple[str, str]:
    """facts: partition_rows, run_length, numeric_variables, in_matrix_cap (each may be None)."""
    if vtype in ("TIMESTAMP", "TEXT"):
        return "NOT_APPLICABLE", "R_TYPE"
    if vtype not in VARIABLE_TYPES:
        raise ValueError(f"REFUSED: unknown variable type {vtype!r}")
    exact_max = _plan().UNIT_ROOT_EXACT_MAX_N
    if metric.endswith("_vs_train") and partition == "train":
        return "NOT_APPLICABLE", "R_SHIFT_TRAIN"
    if metric in MATRIX_SHARE_METRICS:
        if partition != "train":
            return "NOT_APPLICABLE", "R_MATRIX_SHARE"
        nv, cap = facts.get("numeric_variables"), facts.get("in_matrix_cap")
        if nv is not None and nv < 2 or cap is False:
            return "NOT_APPLICABLE", "R_MATRIX_SHARE"
        if nv is None or cap is None or vtype == "UNKNOWN":
            return "UNDETERMINED", "R_MATRIX_SHARE"
        return "APPLICABLE", "R_MATRIX_SHARE"
    pr, run = facts.get("partition_rows"), facts.get("run_length")
    if _BLOCK_RE.match(metric):
        if pr is not None and pr <= exact_max:
            return "NOT_APPLICABLE", "R_UNIT_ROOT_BLOCK"
        if run is not None:
            return ("APPLICABLE" if run > exact_max else "NOT_APPLICABLE"), "R_UNIT_ROOT_BLOCK"
        return "UNDETERMINED", "R_UNIT_ROOT_BLOCK"
    if metric in UNIT_ROOT_EXACT_METRICS:
        if run is not None:
            return ("NOT_APPLICABLE" if run > exact_max else "APPLICABLE"), "R_UNIT_ROOT_EXACT"
        if pr is not None and pr <= exact_max:
            return ("UNDETERMINED" if vtype == "UNKNOWN" else "APPLICABLE"), "R_UNIT_ROOT_EXACT"
        return "UNDETERMINED", "R_UNIT_ROOT_EXACT"
    if vtype == "UNKNOWN":
        return "UNDETERMINED", "R_TYPE"
    return "APPLICABLE", "R_DEFAULT"


def expected_grid_v2(contracts: list[dict], metrics: list[str], operators: list[str] | None = None, *,
                     variable_types: dict, facts_for=None, partitions=PARTITIONS_V2) -> list[dict]:
    """variable_types: {(dataset_id, variable_id): type}; facts_for(dataset_id, variable_id, partition) -> dict.
    Operator cells (operators given) are NOT_PARTITIONED."""
    ops = list(operators) if operators else [NO_OPERATOR]
    if len(set(metrics)) != len(metrics) or len(set(ops)) != len(ops):
        raise ValueError("REFUSED: duplicate metric or operator in the declared grid")
    parts = [NOT_PARTITIONED] if operators else list(partitions)
    facts_for = facts_for or (lambda d, v, p: {})
    cells, keys = [], set()
    for c in contracts:
        for v in c["variables"]:
            vt = variable_types.get((c["dataset_id"], v["variable_id"]), "UNKNOWN")
            for p in parts:
                facts = facts_for(c["dataset_id"], v["variable_id"], p)
                for m in metrics:
                    app, rule = applicability(m, vt, p, facts)
                    for o in ops:
                        key = (c["dataset_id"], v["variable_id"], p, m, o, policy_for(m))
                        if key in keys:
                            raise ValueError("REFUSED: duplicate cell in the declared grid")
                        keys.add(key)
                        cells.append({"key": key, "variable_type": vt, "applicability": app, "rule": rule})
    return cells


def _row_key_v2(r: dict) -> tuple:
    return (r["dataset_id"], r["variable_id"], r.get("partition") or NOT_PARTITIONED, r["metric"],
            r.get("operator") or NO_OPERATOR, r.get("policy") or policy_for(r["metric"]))


def build_matrix_v2(cells: list[dict], rows: list[dict], dataset_terminals: dict | None = None) -> dict:
    """rows: dataset_id, variable_id, partition, metric, status, reason, optional operator and policy.
    dataset_terminals: {dataset_id: terminal status} (COMPLETED or a non-completed state)."""
    declared = {c["key"]: c for c in cells}
    if len(declared) != len(cells):
        raise ValueError("REFUSED: duplicate cell in the declared grid")
    seen: dict[tuple, list] = {}
    undeclared = []
    for r in rows:
        key = _row_key_v2(r)
        if key not in declared:
            undeclared.append({"cell": list(key), "status": r["status"]})
            continue
        seen.setdefault(key, []).append((r["status"], r.get("reason") or ""))
    terms = dataset_terminals or {}
    ledger = []
    for key in sorted(declared):
        c = declared[key]
        members = seen.get(key, [])
        states = Counter(row_state_v2(s, why) for s, why in members)
        raw = Counter(s for s, _ in members)
        if c["applicability"] == "NOT_APPLICABLE":
            if set(states) <= {"NOT_RUN", "UNAVAILABLE"}:
                state, source = "NOT_APPLICABLE", "DECLARATION"
            else:
                state, source = "UNCERTAIN", "ROWS"
        elif members:
            state, source = cell_state_v2(states), "ROWS"
        elif terms.get(key[0]) in TERMINAL_STATE_V2:
            state, source = TERMINAL_STATE_V2[terms[key[0]]], "DATASET_TERMINAL"
        else:
            state, source = "NOT_RUN", "NO_EVIDENCE"
        d, v, p, m, o, pol = key
        ledger.append({"dataset_id": d, "variable_id": v, "partition": p, "metric": m, "operator": o, "policy": pol,
                       "variable_type": c["variable_type"], "applicability": c["applicability"],
                       "applicability_rule": c["rule"], "state": state, "state_source": source,
                       "rows": len(members), "row_states": dict(sorted(states.items())),
                       "row_statuses": dict(sorted(raw.items()))})
    ledger_sha = _sha(ledger)
    return {"schema": "crispdm.data_foundation.coverage_matrix.v2", "cells": len(ledger), "ledger": ledger,
            "ledger_sha256": ledger_sha, "counts_derived_from_ledger": _counts_v2(ledger),
            "applicability_counts": {a: sum(1 for c in ledger if c["applicability"] == a) for a in APPLICABILITY},
            "counts_bound_to": ledger_sha, "precedence": list(PRECEDENCE_V2), "rules": RULES,
            "undeclared_rows": undeclared}


def _counts_v2(ledger) -> dict:
    counts = Counter(c["state"] for c in ledger)
    return {s: counts.get(s, 0) for s in STATES_V2}


def verify_counts_v2(matrix: dict) -> bool:
    """Recount member by member: the totals are exactly the ledger's."""
    led = matrix["ledger"]
    return (_sha(led) == matrix["ledger_sha256"] == matrix["counts_bound_to"]
            and matrix["counts_derived_from_ledger"] == _counts_v2(led)
            and sum(matrix["counts_derived_from_ledger"].values()) == matrix["cells"] == len(led)
            and all(c["rows"] == sum(c["row_statuses"].values()) for c in led))


def coverage_rows_v2(matrix: dict, *, run_id: str, code_sha256: str) -> list[dict]:
    return [{"run_id": run_id, "dataset_id": c["dataset_id"], "variable_id": c["variable_id"],
             "variable_type": c["variable_type"], "partition": c["partition"], "metric": c["metric"],
             "operator": c["operator"], "policy": c["policy"], "applicability": c["applicability"],
             "applicability_rule": c["applicability_rule"], "state": c["state"], "state_source": c["state_source"],
             "rows": c["rows"], "row_states": c["row_statuses"], "code_sha256": code_sha256}
            for c in matrix["ledger"]]


def v1_v2_map(v1: dict, v2: dict) -> dict:
    """One entry per v2 cell with the state of the v1 cell it refines (dataset, variable, metric, operator);
    every v1 cell must be refined by at least one v2 cell."""
    v1_state = {(c["dataset_id"], c["variable_id"], c["metric"], c["operator"]): c["state"] for c in v1["ledger"]}
    entries, used = [], set()
    for c in v2["ledger"]:
        k = (c["dataset_id"], c["variable_id"], c["metric"], c["operator"])
        used.add(k)
        entries.append({"dataset_id": k[0], "variable_id": k[1], "metric": k[2], "operator": k[3],
                        "v1_state": v1_state.get(k, "NOT_IN_V1_GRID"), "partition": c["partition"],
                        "policy": c["policy"], "v2_state": c["state"]})
    transitions = Counter((e["v1_state"], e["v2_state"]) for e in entries)
    return {"schema": "crispdm.data_foundation.coverage_v1_v2_map.v1", "entries": entries,
            "entries_sha256": _sha(entries), "v1_ledger_sha256": v1["ledger_sha256"],
            "v2_ledger_sha256": v2["ledger_sha256"],
            "unmapped_v1_cells": sorted(list(k) for k in set(v1_state) - used),
            "transitions": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())}}


def map_rows(mapping: dict, *, run_id: str, v1_run_id: str, code_sha256: str) -> list[dict]:
    return [dict(e, run_id=run_id, v1_run_id=v1_run_id, code_sha256=code_sha256) for e in mapping["entries"]]
