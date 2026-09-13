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
