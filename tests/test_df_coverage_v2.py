"""C170: coverage v2 separates REFUSED, FAILED, NOT_APPLICABLE and NOT_RUN in
code and in the cube; applicability is declared, never inferred from a missing
row; precedence is specified; v1 is kept and mapped cell by cell."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("df_coverage")
L = _load("load_data_foundation")
SHA = "c" * 64
EXACT_MAX = C._plan().UNIT_ROOT_EXACT_MAX_N


def test_the_four_states_are_distinct_in_code_and_cube():
    s = {"REFUSED", "FAILED", "NOT_APPLICABLE", "NOT_RUN"}
    assert s <= set(C.STATES_V2) and len(C.STATES_V2) == len(set(C.STATES_V2)) == 9
    assert tuple(L.COVERAGE_STATES_V2) == C.STATES_V2
    assert C.row_state_v2("REFUSED") == "REFUSED" and C.row_state_v2("FAILED") == "FAILED"
    assert C.row_state_v2("NOT_RUN", "") == "NOT_RUN"
    assert C.row_state_v2("NOT_RUN", "NOT_RUN_RESOURCE_BOUND") == "RESOURCE_EXCEEDED"
    assert C.row_state_v2("REJECTED") == "RESULT"            # a scientific decision, not unavailability
    with pytest.raises(ValueError):
        C.row_state_v2("DONE")
    ddl = L.ddl()
    table = ddl[ddl.index("public.df_fact_coverage_v2 ("):]
    table = table[:table.index(");")]
    assert "CHECK (state IN ('RESULT', 'INCONCLUSIVE', 'UNAVAILABLE', 'NOT_APPLICABLE', 'NOT_RUN', 'REFUSED', " \
           "'FAILED', 'RESOURCE_EXCEEDED', 'UNCERTAIN'))" in table
    assert "state_source = 'DECLARATION'" in table
    # v1 is preserved as history: it still merges REFUSED into FAILED
    assert C._cell_state(["REFUSED"]) == "FAILED" and C.STATES == ("RESULT", "FAILED", "INCONCLUSIVE",
                                                                   "UNAVAILABLE", "NOT_RUN")


@pytest.mark.parametrize("states, expected", [
    ([], "NOT_RUN"),
    (["RESULT", "INCONCLUSIVE"], "RESULT"),
    (["RESULT", "REFUSED"], "RESULT"),
    (["RESULT", "UNAVAILABLE", "NOT_RUN"], "RESULT"),
    (["RESULT", "FAILED"], "UNCERTAIN"),
    (["RESULT", "RESOURCE_EXCEEDED"], "UNCERTAIN"),
    (["RESULT", "UNCERTAIN"], "UNCERTAIN"),
    (["INCONCLUSIVE", "REFUSED", "FAILED"], "INCONCLUSIVE"),
    (["REFUSED", "FAILED"], "REFUSED"),
    (["REFUSED", "RESOURCE_EXCEEDED"], "REFUSED"),
    (["RESOURCE_EXCEEDED", "FAILED"], "RESOURCE_EXCEEDED"),
    (["FAILED", "UNCERTAIN", "UNAVAILABLE"], "FAILED"),
    (["UNCERTAIN", "NOT_RUN"], "UNCERTAIN"),
    (["UNAVAILABLE", "NOT_RUN"], "UNAVAILABLE"),
    (["NOT_RUN"], "NOT_RUN"),
])
def test_precedence_is_specified_and_order_free(states, expected):
    for perm in itertools.permutations(states):
        assert C.cell_state_v2(list(perm)) == expected


CONTRACTS = [{"dataset_id": "d1", "bank": "PUBLIC",
              "partitions": {"boundaries": {"train": [0, 600], "calibration": [600, 800], "confirmation": [800, 1000]}},
              "variables": [{"variable_id": "d1.ts", "name": "timestamp", "role": "TIMESTAMP"},
                            {"variable_id": "d1.label", "name": "label", "role": "INPUT_CANDIDATE"},
                            {"variable_id": "d1.x", "name": "x", "role": "INPUT_CANDIDATE"},
                            {"variable_id": "d1.y", "name": "y", "role": "INPUT_CANDIDATE"}]}]
TYPES = {("d1", "d1.ts"): "TIMESTAMP", ("d1", "d1.label"): "TEXT", ("d1", "d1.x"): "NUMERIC",
         ("d1", "d1.y"): "NUMERIC"}


def facts(ds, vid, p, cap=("d1.x",)):
    b = CONTRACTS[0]["partitions"]["boundaries"][p]
    return {"partition_rows": b[1] - b[0], "run_length": None, "numeric_variables": 2, "in_matrix_cap": vid in cap}


def row(vid, p, metric, status, reason="", **kw):
    return dict({"dataset_id": "d1", "variable_id": vid, "partition": p, "metric": metric, "status": status,
                 "reason": reason}, **kw)


def ledger_of(m):
    return {(c["variable_id"], c["partition"], c["metric"]): c for c in m["ledger"]}


def test_not_applicable_is_declared_never_inferred_from_a_missing_row():
    metrics = ["acf_lag_1", "ks_statistic_vs_train", "pc1_loading"]
    cells = C.expected_grid_v2(CONTRACTS, metrics, variable_types=TYPES, facts_for=facts)
    m = C.build_matrix_v2(cells, [row("d1.x", "calibration", "acf_lag_1", "COMPLETED")])
    led = ledger_of(m)
    # applicable and absent: NOT_RUN from no evidence, never NOT_APPLICABLE
    assert (led[("d1.x", "train", "acf_lag_1")]["state"], led[("d1.x", "train", "acf_lag_1")]["state_source"]) == \
        ("NOT_RUN", "NO_EVIDENCE")
    assert led[("d1.x", "calibration", "acf_lag_1")]["state"] == "RESULT"
    # declared: timestamp and text columns get no numeric metric
    assert {led[(v, p, "acf_lag_1")]["state"] for v in ("d1.ts", "d1.label") for p in C.PARTITIONS_V2} == \
        {"NOT_APPLICABLE"}
    assert led[("d1.label", "train", "acf_lag_1")]["applicability_rule"] == "R_TYPE"
    # declared: shift metrics on train, PC1 shares outside train or outside the matrix cap
    assert led[("d1.x", "train", "ks_statistic_vs_train")]["state"] == "NOT_APPLICABLE"
    assert led[("d1.x", "calibration", "ks_statistic_vs_train")]["state"] == "NOT_RUN"
    assert led[("d1.x", "train", "pc1_loading")]["state"] == "NOT_RUN"
    assert led[("d1.y", "train", "pc1_loading")]["state"] == "NOT_APPLICABLE"
    assert led[("d1.x", "calibration", "pc1_loading")]["state"] == "NOT_APPLICABLE"
    assert all(c["state"] != "NOT_APPLICABLE" or c["applicability"] == "NOT_APPLICABLE" for c in m["ledger"])
    rows = C.coverage_rows_v2(m, run_id="r", code_sha256=SHA)
    assert all(L.validate_row("df_fact_coverage_v2", r) == [] for r in rows)
    applicable_missing = next(r for r in rows if r["state"] == "NOT_RUN")
    for over in ({"state": "NOT_APPLICABLE"}, {"state": "NOT_APPLICABLE", "state_source": "DECLARATION"},
                 {"state_source": "DECLARATION"}, {"state": "REFUSED"}, {"rows": 1}):
        assert L.validate_row("df_fact_coverage_v2", dict(applicable_missing, **over)), over
    assert C.verify_counts_v2(m)
    m["counts_derived_from_ledger"]["NOT_APPLICABLE"] += 1
    assert not C.verify_counts_v2(m)


def test_rows_contradicting_a_declaration_are_uncertain_not_hidden():
    cells = C.expected_grid_v2(CONTRACTS, ["acf_lag_1"], variable_types=TYPES, facts_for=facts)
    m = C.build_matrix_v2(cells, [row("d1.label", "train", "acf_lag_1", "COMPLETED"),
                                  row("d1.ts", "train", "acf_lag_1", "NOT_RUN", "SAMPLE_INDEX")])
    led = ledger_of(m)
    assert (led[("d1.label", "train", "acf_lag_1")]["state"], led[("d1.label", "train", "acf_lag_1")]["state_source"]) \
        == ("UNCERTAIN", "ROWS")
    assert led[("d1.ts", "train", "acf_lag_1")]["state"] == "NOT_APPLICABLE"
    assert all(L.validate_row("df_fact_coverage_v2", r) == [] for r in C.coverage_rows_v2(m, run_id="r",
                                                                                          code_sha256=SHA))


def test_refused_lab_runs_and_refused_datasets_are_refused_not_failed():
    synth = [{"dataset_id": "u1", "bank": "SYNTHETIC", "variables": [{"variable_id": "u1.v0", "name": "v0"}]}]
    types = {("u1", "u1.v0"): "NUMERIC"}
    ops = ["kalman:{}", "ewma:{}"]
    lab = [{"dataset_id": "u1", "variable_id": "u1.v0", "partition": C.NOT_PARTITIONED, "metric": "operator_evaluation",
            "operator": "kalman:{}", "status": "REFUSED", "reason": "MLE_AT_BOUND"}]
    v1 = C.build_matrix(C.expected_grid(synth, ["operator_evaluation"], ops), lab)
    v2 = C.build_matrix_v2(C.expected_grid_v2(synth, ["operator_evaluation"], ops, variable_types=types), lab)
    assert {c["operator"]: c["state"] for c in v1["ledger"]}["kalman:{}"] == "FAILED"          # the v1 defect
    assert {c["operator"]: c["state"] for c in v2["ledger"]} == {"kalman:{}": "REFUSED", "ewma:{}": "NOT_RUN"}
    mp = C.v1_v2_map(v1, v2)
    assert mp["transitions"] == {"FAILED->REFUSED": 1, "NOT_RUN->NOT_RUN": 1} and mp["unmapped_v1_cells"] == []
    # a dataset whose terminal is REFUSED: its applicable cells without rows are REFUSED, not NOT_RUN
    cells = C.expected_grid_v2(CONTRACTS, ["acf_lag_1"], variable_types=TYPES, facts_for=facts)
    m = C.build_matrix_v2(cells, [], {"d1": "REFUSED"})
    led = ledger_of(m)
    assert (led[("d1.x", "train", "acf_lag_1")]["state"], led[("d1.x", "train", "acf_lag_1")]["state_source"]) == \
        ("REFUSED", "DATASET_TERMINAL")
    assert led[("d1.ts", "train", "acf_lag_1")]["state"] == "NOT_APPLICABLE"
    assert all(L.validate_row("df_fact_coverage_v2", r) == [] for r in C.coverage_rows_v2(m, run_id="r",
                                                                                          code_sha256=SHA))


def test_c164_block_cells_become_not_applicable_by_declaration():
    """The C164 situation: one long dataset emits block ADF/KPSS metrics, so the metric list contains them for
    every dataset; the short datasets were tested exactly. v1 counted their block cells NOT_RUN; v2 declares
    them NOT_APPLICABLE from the contract (partition <= exact_max_n) and maps each one."""
    big_T = 3 * EXACT_MAX
    big = {"dataset_id": "big", "bank": "FINANCIAL",
           "partitions": {"boundaries": {"train": [0, int(big_T * 0.6)], "calibration": [int(big_T * 0.6), int(big_T * 0.8)],
                                         "confirmation": [int(big_T * 0.8), big_T]}},
           "variables": [{"variable_id": "big.v", "name": "v"}]}
    smalls = [{"dataset_id": f"s{i}", "bank": "PUBLIC",
               "partitions": {"boundaries": {"train": [0, 6000], "calibration": [6000, 8000],
                                             "confirmation": [8000, 10000]}},
               "variables": [{"variable_id": f"s{i}.a", "name": "a"}, {"variable_id": f"s{i}.b", "name": "b"}]}
              for i in range(3)]
    contracts = [big] + smalls
    types = {(c["dataset_id"], v["variable_id"]): "NUMERIC" for c in contracts for v in c["variables"]}
    runs = {("big", "big.v", "train"): int(big_T * 0.6), ("big", "big.v", "calibration"): int(big_T * 0.2),
            ("big", "big.v", "confirmation"): int(big_T * 0.2)}
    bounds = {c["dataset_id"]: c["partitions"]["boundaries"] for c in contracts}
    facts_for = lambda d, v, p: {"partition_rows": bounds[d][p][1] - bounds[d][p][0],  # noqa: E731
                                 "run_length": runs.get((d, v, p), bounds[d][p][1] - bounds[d][p][0]),
                                 "numeric_variables": len(next(c for c in contracts if c["dataset_id"] == d)["variables"]),
                                 "in_matrix_cap": True}
    metrics = sorted(set(C.UNIT_ROOT_EXACT_METRICS) | set(C.UNIT_ROOT_BLOCK_METRICS))
    pol = C.policy_for("adf_statistic")
    rows = []
    for m in C.UNIT_ROOT_BLOCK_METRICS:                             # the long train run emitted block rows
        rows.append({"dataset_id": "big", "variable_id": "big.v", "partition": "train", "metric": m,
                     "status": "COMPLETED", "reason": "", "policy": pol})
    for m in C.UNIT_ROOT_EXACT_METRICS:
        rows.append({"dataset_id": "big", "variable_id": "big.v", "partition": "train", "metric": m,
                     "status": "NOT_RUN", "reason": "RUN_LENGTH_EXCEEDS_EXACT_MAX_N_BLOCK_APPROX_ROWS_REPORTED",
                     "policy": pol})
        for p in ("calibration", "confirmation"):
            rows.append({"dataset_id": "big", "variable_id": "big.v", "partition": p, "metric": m,
                         "status": "COMPLETED", "reason": "", "policy": pol})
        for c in smalls:
            for v in c["variables"]:
                for p in C.PARTITIONS_V2:
                    rows.append({"dataset_id": c["dataset_id"], "variable_id": v["variable_id"], "partition": p,
                                 "metric": m, "status": "COMPLETED", "reason": "", "policy": pol})
    v1_rows = [{k: r[k] for k in ("dataset_id", "variable_id", "metric", "status")} for r in rows]
    v1 = C.build_matrix(C.expected_grid(contracts, metrics), v1_rows)
    v1_block_not_run = [c for c in v1["ledger"] if c["metric"] in C.UNIT_ROOT_BLOCK_METRICS and c["state"] == "NOT_RUN"]
    assert len(v1_block_not_run) == 3 * 2 * len(C.UNIT_ROOT_BLOCK_METRICS)          # the false pending cells
    v2 = C.build_matrix_v2(C.expected_grid_v2(contracts, metrics, variable_types=types, facts_for=facts_for), rows)
    assert v2["undeclared_rows"] == [] and C.verify_counts_v2(v2)
    small_block = [c for c in v2["ledger"] if c["dataset_id"] != "big" and c["metric"] in C.UNIT_ROOT_BLOCK_METRICS]
    assert len(small_block) == len(v1_block_not_run) * 3
    assert {(c["state"], c["state_source"], c["applicability_rule"]) for c in small_block} == \
        {("NOT_APPLICABLE", "DECLARATION", "R_UNIT_ROOT_BLOCK")}
    led = {(c["dataset_id"], c["partition"], c["metric"]): c["state"] for c in v2["ledger"]}
    assert led[("big", "train", "adf_statistic_block_middle")] == "RESULT"
    assert led[("big", "calibration", "adf_statistic_block_middle")] == "NOT_APPLICABLE"   # 20% partition <= max
    assert led[("big", "train", "adf_statistic")] == "NOT_APPLICABLE"                      # blocks replace exact
    assert led[("big", "calibration", "adf_statistic")] == "RESULT"
    assert "NOT_RUN" not in {c["state"] for c in v2["ledger"]}
    mp = C.v1_v2_map(v1, v2)
    assert mp["unmapped_v1_cells"] == []
    assert mp["transitions"]["NOT_RUN->NOT_APPLICABLE"] == len(small_block)
    maps = C.map_rows(mp, run_id="r2", v1_run_id="r1", code_sha256=SHA)
    assert len(maps) == v2["cells"] and all(L.validate_row("df_fact_coverage_v1_v2_map", r) == [] for r in maps)
    assert all(L.validate_row("df_fact_coverage_v2", r) == [] for r in C.coverage_rows_v2(v2, run_id="r2",
                                                                                          code_sha256=SHA))


def test_undetermined_applicability_stays_not_run():
    long_c = [{"dataset_id": "L", "bank": "FINANCIAL",
               "partitions": {"boundaries": {"train": [0, 2 * EXACT_MAX], "calibration": [2 * EXACT_MAX, 2 * EXACT_MAX + 10],
                                             "confirmation": [2 * EXACT_MAX + 10, 2 * EXACT_MAX + 20]}},
               "variables": [{"variable_id": "L.v", "name": "v"}]}]
    f = lambda d, v, p: {"partition_rows": long_c[0]["partitions"]["boundaries"][p][1]  # noqa: E731
                         - long_c[0]["partitions"]["boundaries"][p][0], "run_length": None}
    cells = C.expected_grid_v2(long_c, ["adf_statistic_block_start", "adf_statistic"],
                               variable_types={("L", "L.v"): "NUMERIC"}, facts_for=f)
    m = C.build_matrix_v2(cells, [])
    led = ledger_of(m)
    assert led[("L.v", "train", "adf_statistic_block_start")]["applicability"] == "UNDETERMINED"
    assert led[("L.v", "train", "adf_statistic_block_start")]["state"] == "NOT_RUN"
    assert led[("L.v", "calibration", "adf_statistic_block_start")]["state"] == "NOT_APPLICABLE"
    unknown = C.expected_grid_v2(long_c, ["acf_lag_1"], variable_types={}, facts_for=f)
    assert {c["applicability"] for c in unknown} == {"UNDETERMINED"}
