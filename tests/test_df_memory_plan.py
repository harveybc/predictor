"""C147: the memory planner is metadata-only and import-light, its rows have
the declared schema, its constants mirror the modules, its estimates are upper
bounds of every calibration measurement, and its decisions refuse the C146
incident cell before any array exists."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/df_memory_calibration.v1.json"
GiB = 1 << 30


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"tools/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("df_memory_plan")
SHA = "e" * 64


def planner(budget, rows=None, ctx=None):
    sink = rows.append if rows is not None else None
    return P.Planner(run_id="r", bank="FINANCIAL", dataset_id="d", budget_bytes=budget, code_sha256=SHA,
                     context=ctx or {"T": 100, "n_train": 60, "has_ts": False}, sink=sink)


def test_planner_is_import_light():
    code = ("import importlib.util, sys; s = importlib.util.spec_from_file_location('p', 'tools/df_memory_plan.py'); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(sorted(k for k in ('numpy', 'scipy', 'pandas', 'pyarrow', 'statsmodels') if k in sys.modules))")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_mirrored_constants_equal_the_modules():
    U, I, M, S = (_load(m) for m in ("df_profile_univariate", "df_profile_information", "df_profile_multivariate",
                                     "df_sampling"))
    assert P.UNIT_ROOT_EXACT_MAX_N == U.UNIT_ROOT_EXACT_MAX_N
    assert (P.PE_ORDERS, P.SPEC_WINDOW, P.SPEC_HOP, P.WELCH_NPERSEG_INFO) == \
        (I.PE_ORDERS, I.SPEC_WINDOW, I.SPEC_HOP, I.WELCH_NPERSEG)
    assert (P.MAX_PAIRS, P.MAX_VARIABLES_MATRIX, P.MAX_LAG, P.COH_NPERSEG, P.COH_SURROGATES, P.MI_SHUFFLES,
            P.CLUSTER_BOOTSTRAPS) == (M.MAX_PAIRS, M.MAX_VARIABLES_MATRIX, M.MAX_LAG, M.COH_NPERSEG,
                                      M.COH_SURROGATES, M.MI_SHUFFLES, M.CLUSTER_BOOTSTRAPS)
    assert P.WELCH_NPERSEG_SAMPLING == S.WELCH_NPERSEG
    assert P.pair_counts(60) == (50, 60) and P.ADF_PEAK_FACTOR >= 5.04


def test_every_calibration_case_is_under_its_current_estimate():
    doc = json.loads(FIXTURE.read_text())
    cases = [c for c in doc["cases"] if c["case"] != "base"]
    assert not [c for c in doc["cases"] if c.get("error")]
    groups = {f"{c['module']}.{P._family(c['group'])}" for c in cases}
    # every group the runner can invoke has calibration cases; the declared exceptions are never invoked by it
    assert groups == set(P.FACTOR) - set(P.UNCALIBRATED_GROUPS) - set(P.COMPOSITE_GROUPS), set(P.FACTOR) ^ groups
    assert "aliasing_control" not in (ROOT / "tools/df_profile_run.py").read_text()
    for composite, parts in P.COMPOSITE_GROUPS.items():
        assert set(parts) <= groups
    for c in cases:                       # the composite pair block dominates the calibrated pair correlations
        if c["group"] == "pair_correlations":
            n = c["sizes"]["n"]
            block, _, _ = P.estimate("df_profile_multivariate", "pair_block", {"rows": n, "k": 2}, c["context"])
            corr, _, _ = P.estimate("df_profile_multivariate", "pair_correlations", {"n": n}, c["context"])
            assert block >= corr >= c["absolute_maxrss_bytes"]
    for c in cases:
        est, formula, params = P.estimate(c["module"], c["group"], c["sizes"], c["context"])
        without_base = est - P.BASE_PROCESS_BYTES - P.SERIALIZATION_BYTES
        assert c["measured_peak_delta_bytes"] <= without_base, (c["case"], c["measured_peak_delta_bytes"], without_base)
        assert c["absolute_maxrss_bytes"] <= est, (c["case"], c["absolute_maxrss_bytes"], est)
        assert formula and params["factor"] >= 1.0
    base = next(c for c in doc["cases"] if c["case"] == "base")
    assert base["absolute_maxrss_bytes"] <= P.BASE_PROCESS_BYTES


def test_rows_have_exactly_the_declared_schema():
    rows = []
    pl = planner(4 * GiB, rows, {"T": 1000, "n_train": 600, "has_ts": True})
    assert pl.gate("df_profile_univariate", "acf", "v1", "train", n=600)["decision"] == "RUN_EXACT"
    (r,) = rows
    assert tuple(sorted(r)) == tuple(sorted(P.RESOURCE_ESTIMATE_KEYS)) and P.validate_resource_row(r) == []
    assert r["metric"] == "acf" and r["variable_id"] == "v1" and r["budget_bytes"] == 4 * GiB
    assert "nfft" in r["formula"] and r["params"]["sizes"] == {"n": 600}
    assert P.validate_resource_row(dict(r, decision="MAYBE"))
    assert P.validate_resource_row(dict(r, estimated_peak_bytes=5 * GiB))      # admitted above budget is refused


def test_the_c146_incident_cell_is_bounded_before_any_allocation():
    n, lag = 7_952_256, 201
    exact, _, _ = P.estimate("df_profile_univariate", "unit_root_adf", {"n_run": n, "lag": lag},
                             {"T": 13_253_761, "n_train": n, "has_ts": True})
    assert exact > 60 * GiB
    rows = []
    pl = planner(8 * GiB, rows, {"T": 13_253_761, "n_train": n, "has_ts": True})
    assert pl.gate("df_profile_univariate", "unit_root_adf", "v", "train", n_run=n, lag=lag)["decision"] == \
        "NOT_RUN_RESOURCE_BOUND"
    m = P.UNIT_ROOT_EXACT_MAX_N
    d = pl.gate("df_profile_univariate", "unit_root_adf", "v", "train", n_run=m, lag=P.schwert_lag(m),
                variant="BLOCK_APPROX")
    assert d["decision"] == "RUN_BOUNDED" and rows[-1]["estimator"] == "BLOCK_APPROX"


def test_ladder_bounds_a_matrix_group_and_refuses_when_nothing_fits():
    rows = []
    pl = planner(2 * GiB, rows, {"T": 4_000_000, "n_train": 2_400_000, "k_pairs": 9, "pair_rows": 2_400_000})
    d = pl.gate("df_profile_multivariate", "pair_coherence", ["a", "b"], "train", L=2_400_000)
    assert d["decision"] == "RUN_BOUNDED" and d["window"][1] == 2_400_000
    length = d["window"][1] - d["window"][0]
    assert length in P.BOUNDED_ROW_LADDER and rows[-1]["params"]["exact_estimated_peak_bytes"] > 2 * GiB
    assert rows[-1]["variable_id"] == "PAIR:a|b"
    tiny = planner(300 * (1 << 20))
    assert tiny.gate("df_profile_multivariate", "pair_coherence", ["a", "b"], "train", L=2_400_000)["decision"] == \
        "NOT_RUN_RESOURCE_BOUND"


def test_preflight_of_the_worst_dataset_metadata_fits_the_task_budget():
    rows = []
    T = 13_253_761
    b0, b1 = int(T * 0.6), int(T * 0.8)
    meta = {"T": T, "variables": ["value"], "partitions": {"train": [0, b0], "calibration": [b0, b1],
                                                           "confirmation": [b1, T]},
            "has_ts": True, "rg_rows": 1_048_576, "text_timestamp": False}
    pl = P.Planner(run_id="r", bank="FINANCIAL", dataset_id="worst", budget_bytes=int(12 * GiB * 0.9 * 0.8),
                   code_sha256=SHA, context={"T": T, "n_train": b0, "has_ts": True, "rg_rows": 1_048_576},
                   sink=rows.append, stage="PREFLIGHT_METADATA_UPPER_BOUND")
    plan = P.preflight(meta, pl)
    assert plan["reader_decision"] == "RUN_EXACT"
    assert 0 < plan["planned_peak_bytes"] <= pl.budget
    adf = [r for r in rows if r["metric"] == "unit_root_adf"]
    assert {r["decision"] for r in adf} == {"RUN_BOUNDED"}
    assert all(r["params"]["stage"] == "PREFLIGHT_METADATA_UPPER_BOUND" for r in rows)
    assert all(P.validate_resource_row(r) == [] for r in rows)
