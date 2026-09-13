"""C140: every declared cell gets one state from its own rows; counts come
from the ledger; undeclared rows are reported, not absorbed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("df_coverage", ROOT / "tools/df_coverage.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

CONTRACTS = [{"dataset_id": "d1", "variables": [{"variable_id": "a"}, {"variable_id": "b"}]}]


def test_states_by_precedence_and_not_run():
    cells = C.expected_grid(CONTRACTS, ["acf", "entropy"], ["ewma", "identity"])
    assert len(cells) == 8
    rows = [
        {"dataset_id": "d1", "variable_id": "a", "metric": "acf", "operator": "ewma", "status": "FAILED"},
        {"dataset_id": "d1", "variable_id": "a", "metric": "acf", "operator": "ewma", "status": "COMPLETED"},
        {"dataset_id": "d1", "variable_id": "a", "metric": "entropy", "operator": "ewma", "status": "INCONCLUSIVE"},
        {"dataset_id": "d1", "variable_id": "b", "metric": "acf", "operator": "ewma", "status": "REFUSED"},
        {"dataset_id": "d1", "variable_id": "b", "metric": "entropy", "operator": "ewma", "status": "UNAVAILABLE"},
        {"dataset_id": "d1", "variable_id": "z", "metric": "acf", "operator": "ewma", "status": "COMPLETED"},
    ]
    m = C.build_matrix(cells, rows)
    state = {(c["variable_id"], c["metric"], c["operator"]): c["state"] for c in m["ledger"]}
    assert state[("a", "acf", "ewma")] == "RESULT"
    assert state[("a", "entropy", "ewma")] == "INCONCLUSIVE"
    assert state[("b", "acf", "ewma")] == "FAILED"
    assert state[("b", "entropy", "ewma")] == "UNAVAILABLE"
    assert state[("a", "acf", "identity")] == "NOT_RUN"
    assert m["counts_derived_from_ledger"] == {"RESULT": 1, "FAILED": 1, "INCONCLUSIVE": 1, "UNAVAILABLE": 1, "NOT_RUN": 4}
    assert m["undeclared_rows"] == [{"cell": ["d1", "z", "acf", "ewma"], "status": "COMPLETED"}]
    assert C.verify_counts(m)


def test_counts_cannot_be_edited_apart_from_the_ledger():
    m = C.build_matrix(C.expected_grid(CONTRACTS, ["acf"]), [])
    m["counts_derived_from_ledger"]["RESULT"] = 2
    assert not C.verify_counts(m)


def test_duplicate_declarations_refuse():
    with pytest.raises(ValueError, match="duplicate"):
        C.expected_grid(CONTRACTS, ["acf", "acf"])


def test_coverage_rows_shape():
    m = C.build_matrix(C.expected_grid(CONTRACTS, ["acf"]), [])
    rows = C.coverage_rows(m, run_id="r", code_sha256="c" * 64)
    assert len(rows) == 2 and all(r["state"] == "NOT_RUN" and r["operator"] == "NONE" for r in rows)
