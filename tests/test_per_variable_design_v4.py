"""C100-C104: every C100 PRE item is fixed and checked field by field;
Holm and the bounds are executable; the population is honest; scoring
refuses before importing anything."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import per_variable_design_v4 as D  # noqa: E402
import per_variable_screen_v4 as S  # noqa: E402


def resign(d):
    d["design_sha256"] = D.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    return d


def test_the_built_design_validates():
    assert D.validate(D.build_design()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["hypotheses"]["H1"].update(contrast="A1_PER_VARIABLE minus A3_CAPACITY_CONTROL"), "A3"),
    (lambda d: d["panels"].update(minimum_independent_panels=3), "six panels"),
    (lambda d: d["population"].update(requires_all=["terminal v4 layer INDEPENDENTLY_RECOMPUTED"]), "semantic type"),
    (lambda d: d["temporal_quality"].update(rule="none"), "temporal"),
    (lambda d: d["inference"].update(multiplicity="HOLM over the family"), "executable"),
    (lambda d: d["arms"]["A3_CAPACITY_CONTROL"].update(never_uses=["labels"]), "training fold"),
    (lambda d: d["hypotheses"]["H2"].update(requires_both=False), "H2"),
    (lambda d: d["license"].update(scoring="GRANTED"), "scoring"),
])
def test_each_c100_defect_is_refused(mutate, needle):
    d = copy.deepcopy(D.build_design()); mutate(d); resign(d)
    assert any(needle.lower() in x.lower() for x in D.validate(d)), D.validate(d)


def test_a_delta_after_review_refuses():
    d = D.build_design(); rv = d["design_sha256"]
    d["rules"]["harm"] = "softer"; resign(d)
    assert "DELTA_AFTER_REVIEW" in D.validate(d, reviewed_sha256=rv)


def test_holm_matches_a_hand_worked_example():
    adj = D.holm_adjust({"H1": 0.01, "H2_vs_A0": 0.04, "H2_vs_A3": 0.03})
    assert adj["H1"]["adjusted_p"] == pytest.approx(0.03)
    assert adj["H2_vs_A3"]["adjusted_p"] == pytest.approx(0.06)
    assert adj["H2_vs_A0"]["adjusted_p"] == pytest.approx(0.06)
    assert D.holm_rejections({"H1": 0.01, "H2_vs_A0": 0.04, "H2_vs_A3": 0.03}, 0.05) == \
        {"H1": True, "H2_vs_A0": False, "H2_vs_A3": False}
    lv = D.holm_bound_levels(["H1", "H2_vs_A3", "H2_vs_A0"], 0.05)
    assert lv["H1"] == pytest.approx(1 - 0.05 / 3) and lv["H2_vs_A0"] == pytest.approx(0.95)


def test_t_quantile_matches_known_values():
    assert D.t_quantile_one_sided(0.95, 5) == pytest.approx(2.015048, abs=1e-4)
    assert D.t_quantile_one_sided(0.975, 10) == pytest.approx(2.228139, abs=1e-4)


def test_fewer_than_six_panels_is_descriptive_only():
    assert D.panel_contrast([0.1] * 5, 0.95)["state"] == "DESCRIPTIVE_INCONCLUSIVE"
    r = D.panel_contrast([0.01, 0.02, 0.03, 0.02, 0.01, 0.03], 0.95)
    assert r["state"] == "INFERENTIAL" and r["lower_bound"] < r["mean"]


def test_population_is_undetermined_without_dag_v4(tmp_path):
    pop = D.derive_population(terminals_v4=tmp_path, dag_v4=tmp_path / "missing.json",
                              temporal_contracts=[], census=tmp_path / "c.json")
    assert pop["state"] == "UNDETERMINED"


def test_undeclared_semantics_leave_the_bank_insufficient(tmp_path):
    (tmp_path / "t4").mkdir()
    (tmp_path / "t4" / "a.json").write_text(json.dumps({"variable_id": "v", "layer": "INDEPENDENTLY_RECOMPUTED"}))
    (tmp_path / "dag.json").write_text(json.dumps({"nodes": [
        {"dataset_id": "eth", "column": f"c{i}", "class": "CAUSAL_ACTIVE"} for i in range(5)]}))
    (tmp_path / "c.json").write_text(json.dumps({"variables": [
        {"variable_id": "v", "semantics": "UNKNOWN", "role": "UNKNOWN", "license": "UNKNOWN", "unit": "UNKNOWN"}]}))
    (tmp_path / "tq.json").write_text(json.dumps({"contract": {"dataset_id": "eth"}}))
    pop = D.derive_population(terminals_v4=tmp_path / "t4", dag_v4=tmp_path / "dag.json",
                              temporal_contracts=[tmp_path / "tq.json"], census=tmp_path / "c.json")
    assert pop["verdict"] == "BANK_INSUFFICIENT" and pop["members"] == 0
    assert pop["census_variables_with_declared_semantics_role_unit_license"] == 0


def test_the_design_module_imports_no_numeric_library():
    import ast
    tree = ast.parse((REPO / "tools/per_variable_design_v4.py").read_text())
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not mods & {"numpy", "pandas", "scipy", "sklearn", "torch", "pyarrow"}


def test_the_v4_score_entry_point_refuses_before_importing_anything():
    code = ("import sys; sys.path.insert(0, %r)\nbefore=set(sys.modules)\n"
            "import per_variable_screen_v4 as S\ntry:\n    S.main()\nexcept SystemExit as e:\n"
            "    heavy={m for m in set(sys.modules)-before if m.split('.')[0] in "
            "('numpy','pandas','pyarrow','sklearn','torch','tensorflow')}\n"
            "    print(e.code_name, sorted(heavy))\n") % str(REPO / "tools")
    out = subprocess.run((sys.executable, "-c", code), capture_output=True, text=True).stdout.strip()
    assert out == f"{S.REFUSAL} []"
