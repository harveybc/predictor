"""C113, C117-C119, C121: design v5 joins member by member, refuses
duplicates, parses strictly, types every field it checks, keeps v4's
science and closes scoring.

The bank scenarios are the PRE's own bytes (tests/population_join_fixture.py,
checked against the digests the frozen PRE printed)."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import population_join_fixture as F  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("per_variable_design_v5",
                                                  ROOT / "tools/per_variable_design_v5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()
PRE_OUT = ROOT / "docs/audits/evidence/repro_runs/c106_c121_pre_2026_09_12.out"
PANEL0 = ("panel_0", F._dsha("panel_0"))


def derive(kw, **extra):
    return D.derive_population(terminals_dir=kw["terminals_v4"], dag=kw["dag_v4"],
                               census=kw["census"],
                               temporal_contracts=kw["temporal_contracts"], **extra)


def population(tmp_path, scenario):
    kw, digest = F.write_bank(tmp_path, scenario)
    return derive(kw), digest


# ----------------------------------------------------------- same bytes as PRE
def test_fixture_bytes_equal_pre(tmp_path):
    text = PRE_OUT.read_text()
    for name in F.SCENARIOS:
        m = re.search(r"\[" + re.escape(name) + r"\][^\n]*\n\s+input=([0-9a-f]{16})", text)
        assert m, name
        d = tmp_path / name.replace(".", "_")
        d.mkdir()
        _, digest = F.write_bank(d, name)
        assert digest[:16] == m.group(1), name


# ------------------------------------------------------------------ population
def test_control_bank_is_sufficient(tmp_path):
    pop, _ = population(tmp_path, "control_complete_bank")
    assert (pop["verdict"], pop["panel_count"], pop["members"], pop["eligible_variables"]) == \
        ("BANK_SUFFICIENT_FOR_REVIEW", 6, 30, 30)
    assert pop["candidates"] == 30 and all(r["member"] for r in pop["ledger"])
    assert pop["deficit"]["panels_missing"] == 0


EXPECTED_REASON = {
    "C117.terminal_of_other_variable": ("x_0", "NO_TERMINAL_FOR_KEY"),
    "C117.census_of_other_dataset": ("x_0", "NO_CENSUS_ROW_FOR_KEY"),
    "C117.source_digest_distinct": ("x_0", "NO_TERMINAL_FOR_KEY"),
    "C117.license_absent": ("x_0", "UNDECLARED_LICENSE"),
    "C117.role_absent": ("x_0", "ROLE_ABSENT"),
    "C117.temporal_contract_other_digest": ("x_0", "NO_TEMPORAL_CONTRACT_FOR_DATASET_AND_DIGEST"),
    "C117.mask_other_dataset": ("x_0", "MASK_NOT_BOUND_TO_SAME_DATASET"),
    "C117.member_only_in_aggregates": ("x_4", "NO_TERMINAL_FOR_KEY"),
    "C113.recomputed_role_unknown": ("x_0", "ROLE_UNKNOWN"),
    "C113.recomputed_license_unknown": ("x_0", "UNDECLARED_LICENSE"),
    "C113.semantically_unresolved_terminal": ("x_0", "TERMINAL_LAYER_SEMANTICALLY_UNRESOLVED"),
    "C113.date_stored_as_integer": ("x_0", "TIMESTAMP_EXCLUDED_BY_RULE"),
}


@pytest.mark.parametrize("scenario", sorted(EXPECTED_REASON))
def test_pre_attack_is_not_a_sufficient_bank(tmp_path, scenario):
    pop, _ = population(tmp_path, scenario)
    assert pop["verdict"] == "BANK_INSUFFICIENT" and pop["members"] == 0
    assert pop["panel_count"] == 5 and pop["deficit"]["panels_missing"] == 1
    col, reason = EXPECTED_REASON[scenario]
    row = next(r for r in pop["ledger"]
               if (r["dataset_id"], r["dataset_sha256"], r["column"]) == (*PANEL0, col))
    assert row["member"] is False and reason in row["reasons"], row


@pytest.mark.parametrize("scenario", sorted(s for s in EXPECTED_REASON if s.startswith("C113.")))
def test_c113_unknown_or_unresolved_terminal_enters_no_population(tmp_path, scenario):
    pop, _ = population(tmp_path, scenario)
    members = {(r["dataset_id"], r["column"]) for r in pop["ledger"] if r["member"]}
    assert ("panel_0", "x_0") not in members
    (tmp_path / "control").mkdir()
    control, _ = population(tmp_path / "control", "control_complete_bank")
    assert pop["population_sha256"] != control["population_sha256"]


def test_auditor_counterexample_refuses_an_unkeyed_census_row(tmp_path):
    kw, _ = F.write_bank(tmp_path, "C117.defect_zero_terminals_zero_semantics")
    with pytest.raises(D.PopulationRefusal, match="KEY_INCOMPLETE"):
        derive(kw)


def test_auditor_counterexample_with_an_empty_census_is_insufficient(tmp_path):
    kw, _ = F.write_bank(tmp_path, "C117.defect_zero_terminals_zero_semantics")
    Path(kw["census"]).write_text(json.dumps({"variables": []}))
    pop = derive(kw)
    assert pop["verdict"] == "BANK_INSUFFICIENT" and pop["panel_count"] == 0
    assert pop["eligible_variables"] == 0 and pop["candidates"] == 30
    assert all("NO_TERMINAL_FOR_KEY" in r["reasons"] for r in pop["ledger"])
    assert pop["deficit"]["panels_missing"] == 6


def test_duplicate_column_refuses(tmp_path):
    kw, _ = F.write_bank(tmp_path, "C117.duplicate_column")
    with pytest.raises(D.PopulationRefusal, match="DUPLICATE"):
        derive(kw)


@pytest.mark.parametrize("where", ["dag", "terminal", "temporal"])
def test_duplicates_refuse_everywhere(tmp_path, where):
    kw, _ = F.write_bank(tmp_path, "control_complete_bank")
    if where == "dag":
        doc = json.loads(Path(kw["dag_v4"]).read_text())
        doc["nodes"].append(doc["nodes"][0])
        Path(kw["dag_v4"]).write_text(json.dumps(doc))
    elif where == "terminal":
        shutil.copy(Path(kw["terminals_v4"]) / "000.json", Path(kw["terminals_v4"]) / "999.json")
    else:
        kw["temporal_contracts"].append(kw["temporal_contracts"][0])
    with pytest.raises(D.PopulationRefusal, match="DUPLICATE"):
        derive(kw)


def test_aggregates_are_derived_from_ledger_rows_only(tmp_path):
    pop, _ = population(tmp_path, "C117.member_only_in_aggregates")
    assert pop["ignored_input_keys"] == ["aggregates"] and pop["input_aggregates"] == "IGNORED"
    members = [r for r in pop["ledger"] if r["member"]]
    assert pop["eligible_variables"] == len(members)
    assert sum(pop["datasets_below_minimum"].values()) + sum(len(c) for c in pop["panels"].values()) \
        == len(members)
    assert pop["population_sha256"] == D.sha_obj(sorted(
        [r["dataset_id"], r["dataset_sha256"], r["column"], r["variable_id"]] for r in members))


def test_strict_json_refuses_duplicate_keys_and_non_finite_constants(tmp_path):
    with pytest.raises(D.StrictJsonRefusal, match="DUPLICATE_KEY"):
        D.strict_json_loads('{"a": 1, "a": 2}')
    with pytest.raises(D.StrictJsonRefusal, match="NON_FINITE_CONSTANT"):
        D.strict_json_loads('{"a": NaN}')
    kw, _ = F.write_bank(tmp_path, "control_complete_bank")
    text = Path(kw["census"]).read_text()
    Path(kw["census"]).write_text('{"variables": [], ' + text[1:])
    with pytest.raises(D.StrictJsonRefusal, match="DUPLICATE_KEY"):
        derive(kw)


def test_dataset_digest_comes_from_exactly_one_binding(tmp_path):
    kw, _ = F.write_bank(tmp_path, "control_complete_bank")
    doc = json.loads(Path(kw["dag_v4"]).read_text())
    bindings = [{"dataset_id": n["dataset_id"], "output_column": n["column"],
                 "dataset_sha256": n.pop("dataset_sha256")} for n in doc["nodes"]]
    Path(kw["dag_v4"]).write_text(json.dumps(doc))
    unbound = derive(kw)
    assert unbound["verdict"] == "BANK_INSUFFICIENT" and unbound["eligible_variables"] == 0
    # The real keys keep their terminal and census row but lose the DAG
    # node; the DAG nodes become keys of their own with no digest, joined
    # to nothing.
    real = [r for r in unbound["ledger"] if r["dataset_sha256"] != "UNBOUND_NO_DATASET_DIGEST"]
    orphans = [r for r in unbound["ledger"] if r["dataset_sha256"] == "UNBOUND_NO_DATASET_DIGEST"]
    assert len(real) == 30 and len(orphans) == 30
    assert all("NO_DAG_NODE_FOR_KEY" in r["reasons"] for r in real)
    assert all({"NO_TERMINAL_FOR_KEY", "NO_CENSUS_ROW_FOR_KEY"} <= set(r["reasons"]) for r in orphans)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"bindings": bindings}))
    assert derive(kw, binding_manifest=manifest)["verdict"] == "BANK_SUFFICIENT_FOR_REVIEW"
    manifest.write_text(json.dumps({"bindings": bindings + bindings[:1]}))
    with pytest.raises(D.PopulationRefusal, match="DUPLICATE_BINDING"):
        derive(kw, binding_manifest=manifest)


def test_limits_refuse_booleans_and_out_of_range(tmp_path):
    kw, _ = F.write_bank(tmp_path, "control_complete_bank")
    t = Path(kw["terminals_v4"]) / "000.json"
    doc = json.loads(t.read_text())
    for obs, mf in ((True, 0.0), (1999, 0.0), (4096, 0.21), (4096, True)):
        doc.update(observations=obs, missing_fraction=mf)
        t.write_text(json.dumps(doc))
        row = next(r for r in derive(kw)["ledger"] if r["variable_id"] == doc["variable_id"])
        assert "MISSINGNESS_OR_OBSERVATION_LIMIT" in row["reasons"], (obs, mf)


# ---------------------------------------------------------------------- design
def resealed(d, mut):
    d = copy.deepcopy(d)
    mut(d)
    d["design_sha256"] = D.sha_obj({k: v for k, v in d.items() if k != "design_sha256"})
    return d


def setp(path, value):
    def mut(d):
        node = d
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return mut


def test_design_v5_keeps_v4_science_and_validates():
    d = D.build_design()
    assert D.validate(d) == [] and D.typed_problems(d) == []
    v4, v4_file_sha = D.v4_reference()
    for block in D.SCIENTIFIC_BLOCKS:
        assert d[block] == v4[block]
    assert d["population"]["requires_all"] == v4["population"]["requires_all"]
    sup = d["supersedes"]
    assert (sup["file_sha256"], sup["design_sha256"], sup["scientific_change"], sup["rewritten"]) == \
        (v4_file_sha, v4["design_sha256"], "NONE", False)
    assert v4_file_sha.startswith("109cdc505e985c88")
    assert d["license"] == {"scoring": "NOT_GRANTED", "required": D.LICENSE_REQUIRED,
                            "allowed": v4["license"]["allowed"]}


BASE = D.build_design()
C119 = {
    "bool_as_margin": (setp(("hypotheses", "H1", "margin"), True), True),
    "nan_margin": (setp(("hypotheses", "H1", "margin"), float("nan")), True),
    "inf_margin": (setp(("hypotheses", "H2", "margins"), [0.01, float("inf")]), True),
    "margin_out_of_domain": (setp(("hypotheses", "H1", "margin"), -5.0), True),
    "float_as_integer_minimum": (setp(("panels", "minimum_independent_panels"), 6.0), True),
    "duplicate_seed": (setp(("evaluation", "seeds"), [101, 101, 303]), True),
    "duplicate_operator": (setp(("operators",), BASE["operators"] + ["O0_IDENTITY"]), True),
    "duplicate_contrast_in_family": (setp(("inference", "family"), ["H1", "H1", "H2_vs_A3"]), True),
    "lopo_declared_not_derived": (setp(("inference", "sensitivity"), "LOPO: PASSED"), False),
}


@pytest.mark.parametrize("name", sorted(C119))
def test_pre_design_attacks_are_refused(name):
    mut, typed = C119[name]
    d = resealed(BASE, mut)
    assert D.validate(d), name
    if typed:
        assert D.typed_problems(d), f"{name}: the typed guard must catch it without v4"


def test_a_declared_result_in_the_design_is_refused():
    d = resealed(BASE, setp(("implementation", "lopo_result"), "PASSED"))
    assert any("DECLARED_RESULT_IN_DESIGN" in p for p in D.typed_problems(d))


def test_scientific_change_is_refused_block_by_block():
    d = resealed(BASE, setp(("hypotheses", "H1", "margin"), 0.02))
    assert any("SCIENTIFIC_CHANGE: hypotheses" in p for p in D.validate(d))
    d = resealed(BASE, lambda x: x["population"]["requires_all"].pop())
    assert any("population.requires_all" in p for p in D.validate(d))


def test_validate_cli_parses_strictly(tmp_path, capsys):
    text = json.dumps(BASE, sort_keys=True)
    dup = tmp_path / "dup.json"
    dup.write_text('{"schema": "foreign.schema", ' + text[1:])
    assert D.main(["--validate", str(dup)]) == 1
    assert "STRICT_JSON: DUPLICATE_KEY" in capsys.readouterr().out
    nan = tmp_path / "nan.json"
    nan.write_text(json.dumps(resealed(BASE, C119["nan_margin"][0]), sort_keys=True))
    assert D.main(["--validate", str(nan)]) == 1
    assert "NON_FINITE_CONSTANT" in capsys.readouterr().out
    ok = tmp_path / "ok.json"
    ok.write_text(text)
    assert D.main(["--validate", str(ok)]) == 0


def test_write_is_write_once(tmp_path):
    out = tmp_path / "design.json"
    assert D.main(["--write", str(out)]) == 0
    with pytest.raises(SystemExit, match="write-once"):
        D.main(["--write", str(out)])


# ------------------------------------------------------------------- inference
FULL = {"H1": 0.01, "H2_vs_A0": 0.02, "H2_vs_A3": 0.03}


@pytest.mark.parametrize("pv, match", [
    (dict(FULL, H1=1.5), "P_OUT_OF_DOMAIN"),
    (dict(FULL, H1=-0.1), "P_OUT_OF_DOMAIN"),
    (dict(FULL, H1=float("nan")), "P_OUT_OF_DOMAIN"),
    (dict(FULL, H1=True), "P_OUT_OF_DOMAIN"),
    ({"H1": 0.01}, "INCOMPLETE"),
    (dict(FULL, H4=0.01), "INCOMPLETE"),
])
def test_holm_refuses(pv, match):
    with pytest.raises(D.InferenceRefusal, match=match):
        D.holm_adjust_family(pv)


def test_holm_refuses_duplicate_family_and_matches_v4_on_a_complete_one():
    with pytest.raises(D.InferenceRefusal, match="DUPLICATE"):
        D.holm_adjust_family(FULL, family=["H1", "H1", "H2_vs_A3"])
    assert D.holm_adjust_family(FULL) == D.V4.holm_adjust(FULL)


S, S2 = "a" * 64, "b" * 64
VALUES = (0.01, 0.02, 0.03, 0.015, 0.025, 0.005)


def rows(contrast="H1", sample=S, values=VALUES):
    return [{"panel": f"p{i}", "contrast": contrast, "sample_sha256": sample, "value": v}
            for i, v in enumerate(values)]


def test_panel_contrast_binds_one_frozen_sample():
    out = D.panel_contrast_rows(rows(), contrast="H1", sample_sha256=S, level=0.95)
    assert out["state"] == "INFERENTIAL" and out["sample_sha256"] == S and len(out["panels_used"]) == 6
    mixed = rows()
    mixed[2]["sample_sha256"] = S2
    with pytest.raises(D.InferenceRefusal, match="SAMPLE_NOT_FROZEN"):
        D.panel_contrast_rows(mixed, contrast="H1", sample_sha256=S, level=0.95)
    with pytest.raises(D.InferenceRefusal, match="FROZEN_SAMPLE_DIGEST_REQUIRED"):
        D.panel_contrast_rows(rows(sample="x"), contrast="H1", sample_sha256="x", level=0.95)


@pytest.mark.parametrize("bad, match", [
    ((True, False, True, True, False, True), "PANEL_VALUE_NOT_A_FINITE_NUMBER"),
    ((0.01, math.inf, 0.0, 0.0, 0.0, 0.0), "PANEL_VALUE_NOT_A_FINITE_NUMBER"),
])
def test_panel_values_must_be_finite_numbers(bad, match):
    with pytest.raises(D.InferenceRefusal, match=match):
        D.panel_contrast_rows(rows(values=bad), contrast="H1", sample_sha256=S, level=0.95)


def test_duplicate_panel_refuses():
    r = rows()
    r[1]["panel"] = "p0"
    with pytest.raises(D.InferenceRefusal, match="DUPLICATE_OR_UNNAMED_PANEL"):
        D.panel_contrast_rows(r, contrast="H1", sample_sha256=S, level=0.95)


def test_family_must_share_one_sample_and_be_complete():
    fam = {c: rows(contrast=c) for c in D.FAMILY}
    assert D.family_on_one_sample(fam) == S
    fam["H2_vs_A3"] = rows(contrast="H2_vs_A3", sample=S2)
    with pytest.raises(D.InferenceRefusal, match="CONTRASTS_ON_DIFFERENT_SAMPLES"):
        D.family_on_one_sample(fam)
    with pytest.raises(D.InferenceRefusal, match="INCOMPLETE"):
        D.family_on_one_sample({"H1": rows()})


def test_lopo_is_derived_from_panel_rows():
    out = D.lopo_from_panel_rows(rows(), contrast="H1", sample_sha256=S, level=0.95)
    assert out["derived_from"] == "panel_rows" and sorted(out["leave_out"]) == [f"p{i}" for i in range(6)]
    for p, res in out["leave_out"].items():
        assert p not in res["panels_used"] and len(res["panels_used"]) == 5


def test_scoring_refuses():
    with pytest.raises(D.ScoringRefusal, match="EXTERNAL_V5_DESIGN_REVIEW_AND_LICENSE_REQUIRED"):
        D.score()
