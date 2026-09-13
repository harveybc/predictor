"""C152-C168: the executable causal boundary of the operator bank.

Every causal operator use and every trailing_haar_threshold level against the
exhaustive battery (prefix at every t, batch/step/chunk/restart, adversarial
suffixes, independent prefix-only reference) on every case; one negative
control per forbidden class; snapshot refusals; temporal fit modes; naming
decisions. C167: no switch can skip a causal check, and a mechanics sample never
passes. C168: every check removed structurally from an isolated copy, in a new
process (tools/df_structural_mutation.py).
"""
from __future__ import annotations

import ast
import functools
import importlib.util
import subprocess
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
SM = _load("df_structural_mutation", ROOT / "tools")
OPS, SNAP, SYNC, REF, PROBES = BAT.OPS, BAT.SNAP, BAT.SYNC, BAT.REF, BAT.PROBES
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


# ------------------------------------------------ C167: no switch anywhere
PRODUCTION = (ROOT / "tools" / "df_snapshot.py", ROOT / "tools" / "df_operators.py")
ENV_NAMES = {"os", "environ", "getenv", "putenv", "environb"}


def _docstring_nodes(tree) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _identifiers(node) -> list:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.alias):
        return [node.name, node.asname or ""]
    if isinstance(node, ast.keyword):
        return [node.arg or ""]
    return []


def _boolean_public_parameters(tree) -> set:
    out = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name.startswith("_"):
            continue
        a = fn.args
        pos = a.posonlyargs + a.args
        pairs = list(zip(pos, [None] * (len(pos) - len(a.defaults)) + list(a.defaults)))
        pairs += list(zip(a.kwonlyargs, a.kw_defaults))
        for arg, default in pairs:
            if ((isinstance(default, ast.Constant) and isinstance(default.value, bool))
                    or (arg.annotation is not None and ast.unparse(arg.annotation) == "bool")):
                out.add((fn.name, arg.arg))
    return out


def switch_findings(path: Path, strings: bool = True) -> list:
    """GUARD-like names (identifiers and non-docstring strings), environment reads, global statements, module
    flags or flag tables, variadic parameters on public functions."""
    text = path.read_text()
    tree = ast.parse(text)
    docs = _docstring_nodes(tree)
    found = []
    for node in ast.walk(tree):
        names = _identifiers(node)
        if strings and isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            names.append(node.value)
        for n in names:
            if "guard" in n.lower():
                found.append(("GUARD_NAME", getattr(node, "lineno", None), n[:60]))
        for n in _identifiers(node):
            if n in ENV_NAMES:
                found.append(("ENVIRONMENT", getattr(node, "lineno", None), n))
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            found.append(("GLOBAL_STATEMENT", node.lineno, ",".join(node.names)))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_") \
                and (node.args.vararg or node.args.kwarg):
            found.append(("VARIADIC_PUBLIC_PARAMETER", node.lineno, node.name))
    # a module-level name holding a bool (or a table holding bools) is a switch when a condition reads it;
    # declarative tables of facts (e.g. KIND_META assumptions) that no condition reads are not
    holders = {}
    for st in tree.body:
        if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            if (isinstance(st.value, ast.Constant) and isinstance(st.value.value, bool)
                    and any(isinstance(t, ast.Name) for t in targets)):
                found.append(("MODULE_FLAG", st.lineno, ast.unparse(st)[:60]))
            if any(isinstance(c, ast.Constant) and isinstance(c.value, bool) for c in ast.walk(st.value)):
                holders.update({t.id: st.lineno for t in targets if isinstance(t, ast.Name)})
    tests = [n.test for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While, ast.IfExp, ast.Assert))]
    for test in tests:
        for n in ast.walk(test):
            if isinstance(n, ast.Name) and n.id in holders:
                found.append(("FLAG_TABLE_READ_BY_A_CONDITION", n.lineno, n.id))
    return found


def test_no_switch_in_production_modules():
    findings = {p.name: switch_findings(p) for p in PRODUCTION}
    assert findings == {"df_snapshot.py": [], "df_operators.py": []}, findings


def test_no_boolean_parameter_gates_a_causal_check():
    """The only boolean public parameter is `oracle_mode` (frozen public signature): it opens the declared
    NON_CAUSAL negative-control kind and refuses for every causal kind. It is referenced by no condition of
    the 17 causal checks nor by any function that holds one."""
    params = set()
    for p in PRODUCTION:
        params |= _boolean_public_parameters(ast.parse(p.read_text()))
    assert params == {("transform_batch_components", "oracle_mode"), ("transform_batch", "oracle_mode"),
                      ("probe_transform", "oracle_mode")}, params
    bool_names = {n for _, n in params}
    for guard, target, function, locator, transformation in SM.MUTATIONS:
        if transformation != SM.REMOVE_IF_RAISE:
            continue
        tree = ast.parse((ROOT / target).read_text())
        fn = SM._function(tree, function)
        assert not {a.arg for a in ast.walk(fn.args) if isinstance(a, ast.arg)} & bool_names, guard
        (lst, i), = SM._if_raise_sites(fn, locator)
        assert not {n.id for n in ast.walk(lst[i].test) if isinstance(n, ast.Name)} & bool_names, guard
    ops = ast.parse(PRODUCTION[1].read_text())
    for node in ast.walk(ops):
        if isinstance(node, ast.If) and any(isinstance(n, ast.Name) and n.id == "oracle_mode" for n in ast.walk(node.test)):
            assert all(isinstance(s, ast.Raise) and "oracle" in ast.unparse(s) for s in node.body), ast.unparse(node)


def test_battery_has_no_switch_and_emits_no_guard_mutation():
    assert switch_findings(ROOT / "tools" / "df_causal_battery.py", strings=False) == []
    # the probe module names the check it probes (`--guard NAME`); it must still hold no switch
    switch_kinds = {"ENVIRONMENT", "GLOBAL_STATEMENT", "MODULE_FLAG", "FLAG_TABLE_READ_BY_A_CONDITION"}
    assert [f for f in switch_findings(ROOT / "tools" / "df_guard_probes.py", strings=False)
            if f[0] in switch_kinds] == []
    for gone in ("BATTERY_GUARDS", "guard_disabled", "GUARD_PROBES", "guard_mutation", "guard_mutation_rows",
                 "SEVEN_CUTS"):
        assert not hasattr(BAT, gone), gone
    for gone in ("GUARDS", "_guard", "_refuse_if"):
        assert not hasattr(OPS, gone) and not hasattr(SNAP, gone), gone
    assert "GUARD_MUTATION" not in BAT.TEST_CLASSES


@pytest.mark.parametrize("guard", sorted(PROBES.PROBES))
def test_no_attribute_table_or_environment_switches_a_check_off(guard, monkeypatch):
    off = {name: False for name in list(PROBES.PROBES) + ["monotonic_timestamps", "exhaustive_cuts"]}
    for mod in (SNAP, OPS, BAT, PROBES):
        for attr in ("GUARDS", "BATTERY_GUARDS", "CAUSAL_GUARDS", "_GUARDS", "GUARD"):
            monkeypatch.setattr(mod, attr, dict(off), raising=False)
    monkeypatch.setattr(OPS.SNAP, "GUARDS", dict(off), raising=False)
    for var in ("GUARDS", "DF_GUARDS", "CRISPDM_DISABLE_GUARDS", "DISABLE_CAUSAL_CHECKS"):
        monkeypatch.setenv(var, "0")
    res = PROBES.run_probe(guard)
    assert res["expected_refusal"], res
    if guard == "fit_mode_enforcement":
        assert res["wavelet"]["outcome"] == "REFUSED", res
    if guard == "exhaustive_cuts":
        assert BAT.cuts_for_prefix(240, PROBES.EWMA, PROBES.FROZEN) == list(range(240))


@pytest.mark.parametrize("guard", ["monotonic_timestamps", "availability", "artifact_bound", "dataset_binding",
                                   "column_identity", "transform_partition_license", "fit_mode_enforcement"])
def test_oracle_mode_does_not_lift_a_causal_check(guard, monkeypatch):
    monkeypatch.setattr(OPS, "transform_batch", functools.partial(OPS.transform_batch, oracle_mode=True))
    res = PROBES.run_probe(guard)
    assert res["expected_refusal"], res


def test_extra_configuration_is_not_accepted_and_tables_are_read_only():
    X, c, L = PROBES.fixture(seed=90)
    fs = SNAP.FitSnapshot.from_contract(c, "TRAIN", L)
    ts = SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100)
    f = OPS.fit(PROBES.EWMA, fs, PROBES.FROZEN)
    for call in (lambda: OPS.fit(PROBES.EWMA, fs, PROBES.FROZEN, guards=False),
                 lambda: OPS.transform_batch(f, ts, checks=False),
                 lambda: OPS.init_state(f, ts, guards={}),
                 lambda: SNAP.verify_fit_snapshot(fs, guards=False),
                 lambda: SNAP.verify_transform_snapshot(ts, availability=False),
                 lambda: SNAP.FitSnapshot.from_contract(c, "TRAIN", L, guards=False),
                 lambda: SNAP.TransformSnapshot.from_contract(c, L, start=60, end=100, guards=False)):
        with pytest.raises(TypeError):
            call()
    with pytest.raises(TypeError):
        OPS.KIND_FIT_MODES["trailing_haar_threshold"] = (OPS.EXPANDING_PREFIX,)
    with pytest.raises(OPS.OperatorRefusal, match="is not implemented for kind"):
        OPS.fit(PROBES.HAAR, fs, PROBES.EXPANDING)


# ------------------------------------------- C167: official battery vs sample
def test_official_prefix_cuts_are_every_t():
    assert BAT.cuts_for_prefix(240, PROBES.EWMA, PROBES.FROZEN) == list(range(240))
    assert BAT.declared_cuts("PREFIX_ALL_T", PROBES.EWMA, PROBES.FROZEN, "univariate", 100) == list(range(100))


def test_mechanics_sample_never_passes_and_never_publishes():
    spec, mode = PROBES.EWMA, PROBES.FROZEN
    rows = BAT.operator_rows(spec, mode, "univariate", "pytest", "pytest", mechanics_sample=5)
    assert rows and {r["outcome"] for r in rows} == {BAT.MECHANICS_SAMPLE}, rows
    assert all(r["cuts_tested"] <= 5 for r in rows)
    for summary in (BAT.summarize("pytest", rows, [], 0.0, 5), BAT.summarize("pytest", rows, [], 0.0)):
        assert summary["battery_scope"] == BAT.MECHANICS_SAMPLE and summary["all_pass"] is False
    full = BAT.operator_rows(spec, mode, "univariate", "pytest", "pytest")
    assert {r["outcome"] for r in full} == {"PASS"}
    assert BAT.summarize("pytest", full, [], 0.0)["all_pass"] is True
    assert BAT.summarize("pytest", full, [], 0.0, 5)["all_pass"] is False


def test_a_one_sample_leak_is_caught_by_the_official_cuts_and_missed_by_seven():
    with pytest.raises(OPS.OperatorRefusal, match="future leak: prefix differs at t=100"):
        PROBES.probe_exhaustive_battery()()
    leaking_rows = PROBES.base_series(240, 2, 49)
    assert 100 not in SM.HISTORICAL_SEVEN_CUTS and leaking_rows.shape[0] == 240


# -------------------------------------------- C168: structural mutations
def test_every_check_has_one_structural_mutation_and_one_probe():
    assert len(SM.MUTATIONS) == 17
    assert set(SM.GUARDS_BY_NAME) == set(PROBES.PROBES)


def test_mutation_targets_are_found_exactly_once():
    for guard, target, function, locator, transformation in SM.MUTATIONS:
        text = (ROOT / target).read_text()
        mut = SM.mutate_source(text, function, locator, transformation)
        assert mut["removed"] and mut["mutant"] != text, guard
        if transformation == SM.REMOVE_IF_RAISE:
            assert SM._if_raise_sites(SM._function(ast.parse(mut["mutant"]), function), locator) == [], guard
    ops = (ROOT / "tools" / "df_operators.py").read_text()
    with pytest.raises(SM.MutationTargetError, match="found 2 times"):
        SM.mutate_source(ops, "_check_bound_stream", "columns differ from the", SM.REMOVE_IF_RAISE)
    with pytest.raises(SM.MutationTargetError, match="found 0 times"):
        SM.mutate_source(ops, "_check_license", "no such refusal text", SM.REMOVE_IF_RAISE)
    with pytest.raises(SM.MutationTargetError, match="function 'no_such_function'"):
        SM.mutate_source(ops, "no_such_function", "x", SM.REMOVE_IF_RAISE)


@pytest.mark.parametrize("guard", [m[0] for m in SM.MUTATIONS])
def test_structural_mutation_bites_in_isolation(guard):
    rec = SM.run_mutation(guard, python=sys.executable)
    assert rec["intact"]["outcome"] == "REFUSED" and rec["intact"]["expected_refusal"], rec["intact"]
    assert rec["children_loaded_the_right_bytes"], rec
    assert rec["original_sha256"] != rec["mutant_sha256"]
    assert rec["bit"] and rec["detected"], rec["mutant"]
    # the unbound kernel artifact has no binding left to check once its refusal is removed
    assert rec["mutant"]["outcome"] == ("EXCEPTION" if guard == "artifact_bound" else "ACCEPTED"), rec["mutant"]
    if guard == "fit_mode_enforcement":
        assert rec["intact"]["wavelet"]["outcome"] == "REFUSED"
        assert rec["mutant"]["wavelet"]["outputs_t_le_30_moved"] is True


def test_evidence_process_never_imports_a_tools_module():
    code = ("import importlib.util,sys,json;sys.dont_write_bytecode=True;"
            f"s=importlib.util.spec_from_file_location('sm',{str(ROOT / 'tools' / 'df_structural_mutation.py')!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "r=m.run_mutation('availability',python=sys.executable);"
            "print(json.dumps({'detected':r['detected'],'loaded':[x for x in m.MODULES_NEVER_IMPORTED_HERE if x in sys.modules]}))")
    p = subprocess.run(SM.default_child_prefix("mut-evidence-process") + [sys.executable, "-B", "-c", code],
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-800:]
    assert p.stdout.strip().splitlines()[-1] == '{"detected": true, "loaded": []}'


def test_structural_cli_is_write_once_and_never_writes_under_local(tmp_path):
    existing = tmp_path / "exists"
    existing.mkdir()
    assert SM.main(["--out", str(existing)]) == 2
    assert list(existing.iterdir()) == []
    assert SM.main(["--out", str(Path.home() / ".local" / "never_created_by_mutation")]) == 2
    assert not (Path.home() / ".local" / "never_created_by_mutation").exists()


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
