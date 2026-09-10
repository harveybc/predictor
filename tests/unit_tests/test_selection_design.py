"""P5: the selection design is sealed, and the preflight cannot
conclude anything.

A design is only a design if it was fixed before the results
existed, and a preflight is only a preflight if it is incapable of
producing the number everyone wants to see early.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from olap import selection_design as sd  # noqa: E402

DESIGN = REPO / "examples/research/crispdm_selection_design.v1.json"
PREFLIGHT = (REPO /
             "examples/research/crispdm_selection_preflight.v1"
             ".json")


@pytest.fixture(scope="module")
def design():
    if not DESIGN.is_file():
        pytest.skip("sealed design not built in this checkout")
    return json.loads(DESIGN.read_text())


def _task(**over):
    t = {"task_id": "t", "objective": "o", "baseline": "b",
         "outer_origins": ["origin0"]}
    t.update(over)
    return t


def _build(**over):
    kw = dict(sealed_at="2026-09-10T00:00:00Z",
              tasks=[_task()],
              budget_rule={"rule": "fixed"},
              bank_index_sha256="a" * 64,
              eligibility_manifest_sha256=None)
    kw.update(over)
    return sd.build_design(**kw)


# ==============================================================
# sealing
# ==============================================================

def test_sealed_design_verifies(design):
    sd.verify_design(design)
    assert design["status"] == "SEALED_BEFORE_ANY_SCORE"


def test_editing_a_sealed_design_breaks_it(design):
    mutated = json.loads(json.dumps(design))
    mutated["frozen_analysis"]["global_non_inferiority"][
        "margin"] = 0.5
    with pytest.raises(SystemExit, match="not sealed"):
        sd.verify_design(mutated)


def test_design_binds_the_bank_index_it_was_written_against(
        design):
    idx = REPO / "examples/research/crispdm_bank_index.v1.json"
    if not idx.is_file():
        pytest.skip("bank index absent")
    bank = json.loads(idx.read_text())
    assert design["binds"]["bank_index_sha256"] == \
        bank["index_sha256"]


# ==============================================================
# the outer unit is never a seed
# ==============================================================

@pytest.mark.parametrize("bad", sd.FORBIDDEN_OUTER_UNITS)
def test_a_seed_can_never_be_the_outer_unit(bad):
    with pytest.raises(SystemExit,
                       match="never an independent observation"):
        _build(tasks=[_task(outer_unit=bad)])


def test_default_outer_unit_is_task_origin_series(design):
    assert design["outer_unit"]["unit"] == "task_origin_series"
    for t in design["tasks"]:
        assert t.get("outer_unit",
                     "task_origin_series") == "task_origin_series"


def test_task_without_objective_or_baseline_refuses():
    for field in ("objective", "baseline", "outer_origins"):
        t = _task()
        del t[field]
        with pytest.raises(SystemExit, match="frozen per task"):
            _build(tasks=[t])


# ==============================================================
# comparators
# ==============================================================

def test_required_comparators_cannot_be_excluded():
    for cid in ("all_mechanically_admissible",
                "random_same_size_control",
                "mutual_information_train_only",
                "regularised_linear",
                "stability_redundancy_filter"):
        with pytest.raises(SystemExit,
                           match="cannot be excluded"):
            _build(excluded_comparators=[
                {"comparator_id": cid, "reason": "x"}])


def test_random_control_is_always_present(design):
    ids = [c["comparator_id"] for c in design["comparators"]]
    assert "random_same_size_control" in ids
    assert "all_mechanically_admissible" in ids


def test_agent_multi_selector_is_excluded_with_a_reason(design):
    excluded = {c["comparator_id"]: c
                for c in design["excluded_comparators"]}
    assert "agent_multi_current_selector" in excluded
    reason = excluded["agent_multi_current_selector"]["reason"]
    assert "LEGACY_NON_AUTHORITATIVE" in reason
    assert "P4" in reason
    ids = [c["comparator_id"] for c in design["comparators"]]
    assert "agent_multi_current_selector" not in ids


def test_optional_comparator_carries_its_admission_condition():
    entry = [c for c in sd.COMPARATORS
             if c["comparator_id"] ==
             "agent_multi_current_selector"][0]
    assert entry["required"] is False
    assert entry["admission_condition"] == \
        "P4_BOUNDARY_AUDIT_PASSED"


# ==============================================================
# what is frozen
# ==============================================================

def test_everything_the_order_requires_is_frozen_as_data(design):
    f = design["frozen_analysis"]
    assert f["global_non_inferiority"]["margin"] == 0.02
    assert f["multiplicity"]["family_frozen_before_scoring"] \
        is True
    assert f["stability_floor"] == 0.5
    for key in ("selection_cost", "stability_across_origins",
                "extreme_preservation", "inconclusive_rule",
                "withdrawal_criterion"):
        assert key in f and f[key]
    assert design["budget_rule"]


def test_only_training_is_fitted_on(design):
    d = design["split_discipline"]
    assert d["outer_split"].startswith("UNTOUCHED")
    assert set(d["fitted_inside_training_only"]) == {
        "imputation", "scaling", "transformations",
        "the selector itself"}
    assert "forbidden" in d["validation_reuse"]


def test_withdrawn_comparators_do_not_re_enter(design):
    w = design["frozen_analysis"]["withdrawal_criterion"]
    assert "not re-entered by a later run" in w


# ==============================================================
# the preflight cannot conclude
# ==============================================================

def test_preflight_declares_no_conclusion(design):
    out = sd.mechanical_preflight(
        design, available_units={
            t["task_id"]: {"series": ["s1", "s2"]}
            for t in design["tasks"]})
    assert out["mode"] == "MECHANICAL_ONLY_NO_SCORE"
    assert out["conclusion"].startswith("NONE")
    assert "external review" in out["next_gate"]


@pytest.mark.parametrize("word", sd.SCORE_WORDS)
def test_a_preflight_carrying_a_score_refuses(word):
    fake = {"schema": "crispdm.selection_preflight.v1",
            "mode": "MECHANICAL_ONLY_NO_SCORE",
            "tasks": [], word: 0.42}
    with pytest.raises(SystemExit, match="never scores"):
        sd.assert_no_conclusion(fake)


def test_a_preflight_that_relabels_its_mode_refuses():
    with pytest.raises(SystemExit, match="mode is not"):
        sd.assert_no_conclusion({"mode": "FULL_RUN",
                                 "tasks": []})


def test_preflight_reports_infeasibility_instead_of_pretending(
        design):
    out = sd.mechanical_preflight(
        design, available_units={})
    for t in out["tasks"]:
        assert t["feasible"] is False
        assert t["outer_units_constructible"] == 0
        assert "no outer unit" in t["reason"]


def test_committed_preflight_is_the_sealed_design(design):
    if not PREFLIGHT.is_file():
        pytest.skip("preflight not built")
    pre = json.loads(PREFLIGHT.read_text())
    assert pre["design_sha256"] == design["design_sha256"]
    sd.assert_no_conclusion(pre)
    assert all(t["feasible"] for t in pre["tasks"])


def test_preflight_refuses_an_unsealed_design(design):
    """Two independent guards, and the digest one bites first —
    so a downgraded status can never slip through unnoticed."""
    mutated = json.loads(json.dumps(design))
    mutated["status"] = "DRAFT"
    with pytest.raises(SystemExit, match="not sealed"):
        sd.mechanical_preflight(mutated, available_units={})
    # a design whose digest is REPAIRED around the downgrade is
    # still refused, on the status itself
    mutated["design_sha256"] = sd._self_sha(mutated)
    with pytest.raises(SystemExit, match="design status is"):
        sd.mechanical_preflight(mutated, available_units={})
