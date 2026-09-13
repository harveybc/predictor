"""C152-C159: the executable causal boundary of the operator bank.

Every causal operator use and every trailing_haar_threshold level against the
exhaustive battery (prefix at every t, batch/step/chunk/restart, adversarial
suffixes, independent prefix-only reference) on every case; one negative
control per forbidden class; every guard removed in turn; snapshot refusals;
temporal fit modes; naming decisions.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, directory):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BAT = _load("df_causal_battery", ROOT / "tools")
OPS, SNAP, SYNC, REF = BAT.OPS, BAT.SNAP, BAT.SYNC, BAT.REF
USES = BAT.operator_uses()


@pytest.mark.parametrize("case_id", BAT.ALL_CASES)
@pytest.mark.parametrize("use", USES, ids=lambda u: BAT.use_id(*u))
def test_operator_use_passes_the_exhaustive_battery(use, case_id):
    spec, mode = use
    rows = BAT.operator_rows(spec, mode, case_id, "pytest", "pytest")
    levels = 1 + (spec["params"]["levels"] if spec["kind"] == "trailing_haar_threshold" else 0)
    assert len(rows) == 4 * levels
    bad = [(r["test_class"], r["level"], r["reason"]) for r in rows if r["outcome"] != "PASS"]
    assert not bad
    small = case_id != "large"
    prefix = [r for r in rows if r["test_class"] == "PREFIX_ALL_T"]
    assert all(r["cuts_tested"] == (r["n"] if small else r["cuts_tested"]) for r in prefix)


def test_uses_cover_every_causal_spec_and_both_decomposition_modes():
    specs = [s for s in OPS.bank_specs() if s["kind"] in OPS.CAUSAL_KINDS]
    assert {BAT.use_id(s, m)[:-len(m) - 1] for s, m in USES} == {BAT.use_id(s, "x")[:-2] for s in specs}
    modes = {m for s, m in USES if s["kind"] == "causal_decomposition"}
    assert modes == {OPS.EXPANDING_PREFIX, OPS.FROZEN_PREVIOUS_PARTITION}


def test_small_cases_include_every_boundary_length():
    for cid in BAT.SMALL_CASES:
        X, _ = BAT.cases()[cid]
        assert X.shape[0] == BAT.SMALL_N
    # every t is a prefix, so lengths 2^j-1, 2^j, 2^j+1 up to 65 and every warm-up are covered
    for spec, mode in USES:
        assert OPS.warmup_length(spec["kind"], spec["params"], mode) + 1 < BAT.SMALL_N


# ------------------------------------------------------- negative controls
CONTROLS = list(BAT.OUTPUT_CONTROLS) + list(BAT.SNAPSHOT_CONTROLS)


def test_one_control_per_class():
    assert len(CONTROLS) == 10


@pytest.mark.parametrize("name", CONTROLS)
def test_negative_control_is_detected(name):
    detected, msg = BAT.negative_control_detection(name)
    assert detected, msg
    if name in BAT.OUTPUT_CONTROLS:
        assert msg.startswith("PREFIX_ALL_T")


# ---------------------------------------------------------- guard mutation
GUARDS = [p[0] for p in BAT.GUARD_PROBES]


def test_every_added_guard_has_a_mutation_probe():
    names = {g.split(".")[0] for g in GUARDS}
    assert set(SNAP.GUARDS) <= names and set(OPS.GUARDS) <= names and "exhaustive_cuts" in names


@pytest.mark.parametrize("name", GUARDS)
def test_guard_refuses_and_its_removal_is_detected(name):
    res = BAT.guard_mutation(name)
    # guard on: the probe refuses for the expected reason
    assert res["on"] is not None and res["expected"] in res["on"], res
    # guard off: the same assertion would fail
    assert res["off"] is None or res["expected"] not in res["off"], res


def test_removing_fit_mode_enforcement_leaks_later_train_parameters():
    assert BAT.fit_mode_leak_under_mutation().endswith("True")


def test_seven_cuts_miss_a_one_sample_leak_that_every_t_detects():
    make = BAT._probe_exhaustive_battery
    with pytest.raises(OPS.OperatorRefusal, match="future leak: prefix differs at t=100"):
        make()()
    with BAT.guard_disabled(BAT.BATTERY_GUARDS, "exhaustive_cuts"):
        make()()


# ------------------------------------------------ snapshots and fit modes
@pytest.mark.parametrize("case", BAT._refusal_cases(), ids=lambda c: c[0])
def test_snapshot_refusal(case):
    _, fn, expected = case
    with pytest.raises((OPS.OperatorRefusal, SNAP.SnapshotRefusal), match=expected):
        fn()


def test_fit_modes():
    rows = BAT.fit_mode_rows("pytest", "pytest")
    bad = [(r["operator_kind"], r["case_id"], r["reason"]) for r in rows if r["outcome"] != "PASS"]
    assert not bad
    cases = {r["case_id"] for r in rows}
    assert {"pre_seasonal_case_expanding_prefix_unchanged", "pre_seasonal_case_frozen_in_sample_refused",
            "frozen_transform_of_fit_partition_refused", "expanding_prefix_not_implemented_refused",
            "offline_analysis_never_emits"} <= cases
    refused = {r["operator_kind"] for r in rows if r["case_id"] == "expanding_prefix_not_implemented_refused"}
    assert refused == {"local_level_kalman", "local_linear_trend_kalman", "trailing_hampel",
                       "trailing_haar_threshold"}


def test_expanding_decomposition_is_warmup_until_every_phase_has_data():
    X = BAT.base_series(120, 2, 3)
    X[3, 0] = np.nan                      # phase 3 of column 0 first seen at row 27
    spec = {"kind": "causal_decomposition", "params": {"period": 24, "season_alpha": 0.1, "trend_alpha": 0.1}}
    f = OPS._fit_kernel(spec, BAT.base_series(300, 2, 99), OPS.EXPANDING_PREFIX)
    _, A, R = OPS._transform_kernel(f, X)
    assert (R[:24, 1][np.isfinite(X[:24, 1])] == "WARMUP").all() and A[24:, 1].all()
    assert R[3, 0] == "MISSING_INPUT" and (R[24:28, 0] == "WARMUP").all() and A[28:, 0].all()


# ---------------------------------------------------------- reference
def test_reference_is_independent_of_production():
    tree = ast.parse((ROOT / "tests" / "df_causal_reference.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"math", "__future__"}, imported
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"np", "numpy", "sliding_window_view", "OPS", "df_operators"}


def test_reference_declares_a_tolerance_for_every_licensed_kind():
    assert set(REF.REFERENCE_TOLERANCE) == set(OPS.CAUSAL_KINDS)
    assert all(v[0] == REF.BITWISE and v[1] for v in REF.REFERENCE_TOLERANCE.values())


# ------------------------------------------------------------ naming
def test_naming_decisions():
    assert "wavelet_haar_atrous" not in OPS.KINDS and "trailing_haar_threshold" in OPS.KINDS
    with pytest.raises(OPS.OperatorRefusal, match="unknown operator kind"):
        OPS.validate_spec({"kind": "wavelet_haar_atrous", "params": {"levels": 2, "threshold_k": 3.0}})
    by = {d["subject"]: d for d in OPS.NAMING_DECISIONS}
    assert by["trailing_haar_threshold"]["previous_name"] == "wavelet_haar_atrous"
    assert by["trailing_haar_threshold"]["decision"] == "RENAMED"
    assert by["T06_CAUSAL_SWT"]["decision"] == "DESIGN_ONLY_NOT_IMPLEMENTED"
    assert "T06_CAUSAL_SWT" not in OPS.KINDS
    assert OPS.KIND_META["trailing_haar_threshold"]["assumptions"]["claims_standard_swt_or_a_trous"] is False
    rows = BAT.naming_rows("pytest", "pytest")
    assert {(r["subject"], r["subject_kind"], r["decision"]) for r in rows} == {
        ("trailing_haar_threshold", "OPERATOR", "RENAMED"),
        ("T06_CAUSAL_SWT", "DESIGN_ARM", "DESIGN_ONLY_NOT_IMPLEMENTED"),
        ("wavelet_mad", "ESTIMATOR", "OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL")}


def test_no_code_or_test_uses_the_old_name_outside_naming_records():
    for p in list((ROOT / "tools").glob("*.py")) + list((ROOT / "tests").glob("test_df_*.py")):
        text = p.read_text()
        if "wavelet_haar_atrous" in text:
            assert p.name in ("df_operators.py", "test_df_operators_causality.py"), p.name
    ops_text = (ROOT / "tools" / "df_operators.py").read_text()
    assert ops_text.count("wavelet_haar_atrous") == 2        # docstring history and NAMING_DECISIONS


# ------------------------------------------------------------------ CLI
def test_cli_is_write_once_and_never_writes_under_local(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    assert BAT.main(["--out", str(existing)]) == 2
    assert list(existing.iterdir()) == []
    assert BAT.main(["--out", str(Path.home() / ".local" / "never_created_by_battery")]) == 2
    assert not (Path.home() / ".local" / "never_created_by_battery").exists()
