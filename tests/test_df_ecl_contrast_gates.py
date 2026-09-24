"""RP144-RP151 continuation: the contrast's learning and evaluation path, F1 to F4.

These tests were written against the REAL runner before its correction, and their pre-correction failure is frozen in
docs/audits/evidence/d3_k5_20260917/RP152/. Each one states a behaviour: a summary that cannot be vacuously true, a design
identity that separates scientific inputs from operational timestamps, an allocation that refuses rather than rounds up to one
epoch, an evaluation population that loses no window, a trained model that can actually be restored, and a step count that is
an observed optimizer iteration rather than a multiplication.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / ".cache/data-gov/sota_benchmarks/7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad.csv"
needs_data = pytest.mark.skipif(not DATA.is_file(), reason="the governed ECL delivery is not on this host")


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("df_ecl_modular")


# --- F2: a summary that no population can make vacuously true -----------------------------------------------------------

def test_F2_an_empty_population_cannot_satisfy_the_regime_checks():
    """Musashi executed the exact assignment with cells={} and got four green flags. A verdict over nothing is not a verdict."""
    checks = M.regime_checks({}, seeds=(2021, 2022, 2023))
    assert checks["complete"] is False
    assert checks["missing_cells"], "an empty population must name what is missing"
    assert checks["verdict"] == "INCOMPLETE_POPULATION"
    for key in ("R1_detector_unchanged_by_its_fit", "R2_detector_changed_by_its_fit"):
        assert checks[key] is None, "a check with no evidence reports None, never True"


def test_F2_one_regime_of_one_seed_is_reported_as_partial():
    cells = {"R0_s2021": {"steps": 10, "observed_updates": 10, "detector_unchanged_by_the_fit": False, "donor": None}}
    checks = M.regime_checks(cells, seeds=(2021, 2022, 2023))
    assert checks["complete"] is False and checks["verdict"] == "INCOMPLETE_POPULATION"
    assert "R1_s2021" in checks["missing_cells"] and "AE_s2022" in checks["missing_cells"]


def test_F2_a_complete_population_is_judged_on_its_own_evidence():
    cells = {}
    for seed in (2021, 2022, 2023):
        cells[f"AE_s{seed}"] = {"donor_sha256": f"d{seed}"}
        cells[f"R0_s{seed}"] = {"steps": 10, "observed_updates": 10, "detector_unchanged_by_the_fit": False, "donor": None}
        cells[f"R1_s{seed}"] = {"steps": 10, "observed_updates": 10, "detector_unchanged_by_the_fit": True, "donor": f"d{seed}"}
        cells[f"R2_s{seed}"] = {"steps": 10, "observed_updates": 10, "detector_unchanged_by_the_fit": False, "donor": f"d{seed}"}
    checks = M.regime_checks(cells, seeds=(2021, 2022, 2023))
    assert checks["complete"] is True and checks["verdict"] == "COMPLETE"
    assert checks["R1_detector_unchanged_by_its_fit"] is True
    assert checks["R2_detector_changed_by_its_fit"] is True
    assert checks["R1_and_R2_share_the_donor_per_seed"] is True
    assert checks["same_update_allowance"] is True
    # and a single wrong cell flips exactly the check it belongs to
    broken = dict(cells)
    broken["R1_s2022"] = {**cells["R1_s2022"], "detector_unchanged_by_the_fit": False}
    assert M.regime_checks(broken, seeds=(2021, 2022, 2023))["R1_detector_unchanged_by_its_fit"] is False


# --- F3: a design identity that a mutation actually moves ----------------------------------------------------------------

@needs_data
def test_F3_the_design_digest_is_stable_across_seals_and_moves_with_a_scientific_change():
    """The previous assertion ended in `or True`, so it could not fail. A canonical identity must ignore the clock and must
    not ignore the science."""
    a = M.seal_contrast(DATA, pred_len=96)
    b = M.seal_contrast(DATA, pred_len=96)
    assert a["design_sha256"] == b["design_sha256"], "two seals of the same design must share one identity"
    for changed in (M.seal_contrast(DATA, pred_len=192),
                    M.seal_contrast(DATA, pred_len=96, seeds=(2021, 2022)),
                    M.seal_contrast(DATA, pred_len=96, internal_validation_fraction=0.3)):
        assert changed["design_sha256"] != a["design_sha256"], "a scientific change must move the identity"
    assert a["at"] is not None and "at" not in a["identity_covers"]
    assert "task" in a["identity_covers"] and "architecture" in a["identity_covers"]


# --- F4: allocation refused rather than rounded up ------------------------------------------------------------------------

@needs_data
def test_F4_an_infeasible_allocation_is_refused_not_rounded_up_to_one_epoch(tmp_path):
    """The previous code forced at least one epoch even when the usable budget was zero."""
    with pytest.raises(M.AllocationError) as exc:
        M.run_contrast(DATA, tmp_path / "infeasible", pred_len=96, seeds=(2021,),
                       cpu_budget_seconds=1.0, wall_budget_seconds=1.0, limit_train_windows=64)
    assert "allocation" in str(exc.value).lower()
    assert not (tmp_path / "infeasible" / "CONTRAST.json").exists()


# --- the evaluation population loses no window ----------------------------------------------------------------------------

@needs_data
def test_the_window_sequence_delivers_every_origin_including_an_uneven_last_batch():
    """`len(idx) // batch` silently dropped the remainder, so a scoring population could be short without saying so."""
    E = _load("df_mod_e0")
    tf = E._tf()
    d = M.author_datasets(DATA, pred_len=96)
    ds = d["splits"]["train"]["dataset"]
    W = M._windows_class(tf)
    origins = list(range(10))
    seq = W(ds, origins, seq_len=M.SEQ_LEN, pred_len=96, batch=4, seed=0, complete=True)
    rows = sum(len(seq[i][0]) for i in range(len(seq)))
    assert rows == len(origins), f"the sequence delivered {rows} of {len(origins)} windows"
    assert len(seq) == 3, "10 windows at batch 4 is three batches, the last of two"


# --- F1 and F4 through the real runner on a tiny population ----------------------------------------------------------------

@needs_data
def test_F1_each_regime_cell_persists_a_restorable_model_and_a_selected_checkpoint(tmp_path):
    """The declared contract selects a checkpoint on the outer validation and restores it. The run must therefore leave a
    model that can be loaded again and say which epoch it chose."""
    out = M.run_contrast(DATA, tmp_path / "tiny", pred_len=96, seeds=(2021,), batch=8,
                         cpu_budget_seconds=900, wall_budget_seconds=900,
                         limit_train_windows=64, limit_validation_windows=32, max_epochs=2)
    for regime in ("R0", "R1", "R2"):
        cell = out["cells"][f"{regime}_s2021"]
        model_path = Path(cell["model_path"])
        assert model_path.is_file(), f"{regime} left no restorable model"
        assert cell["model_sha256"] and len(cell["model_sha256"]) == 64
        assert isinstance(cell["selected_epoch"], int) and cell["selected_epoch"] >= 1
        assert cell["selected_epoch"] <= cell["epochs"]
        assert cell["restored_matches_selection"] is True


@needs_data
def test_F4_the_reported_updates_are_observed_optimizer_iterations(tmp_path):
    """`epochs * len(sequence)` is an arithmetic claim. The optimizer's own iteration counter is the measurement."""
    out = M.run_contrast(DATA, tmp_path / "tiny_steps", pred_len=96, seeds=(2021,), batch=8,
                         cpu_budget_seconds=900, wall_budget_seconds=900,
                         limit_train_windows=64, limit_validation_windows=32, max_epochs=2)
    for regime in ("R0", "R1", "R2"):
        cell = out["cells"][f"{regime}_s2021"]
        assert cell["observed_updates"] > 0
        assert cell["observed_updates"] == cell["steps"], "the observed count must match the prescribed one or be reported"
    assert out["measured_cost"]["cpu_seconds"] > 0 and out["measured_cost"]["wall_seconds"] > 0
    assert out["measured_cost"]["cpu_seconds"] != out["measured_cost"]["wall_seconds"]


@needs_data
def test_the_monitoring_subset_is_declared_apart_from_the_full_scoring_population(tmp_path):
    """A twenty-batch validation reading is a monitor, not the reference metric, and the record must say so."""
    out = M.run_contrast(DATA, tmp_path / "tiny_pop", pred_len=96, seeds=(2021,), batch=8,
                         cpu_budget_seconds=900, wall_budget_seconds=900,
                         limit_train_windows=64, limit_validation_windows=32, max_epochs=2)
    pop = out["populations"]
    assert pop["validation_monitor_windows"] <= pop["validation_windows_available"]
    assert pop["scoring_population"] == "NOT_SCORED_IN_THIS_RUN"
    cell = out["cells"]["R0_s2021"]
    assert cell["validation_monitor_loss_is_not_the_reference_metric"] is True


# --- Musashi's integration review, finding 5: missing evidence is not a positive finding ------------------------------------

def test_F5_missing_R2_change_evidence_is_never_a_positive_finding():
    """The exact counterexample: every expected cell name present, but no R2 change measurement. `not v.get(...)` turned the
    absence into True. A check needs typed evidence, not the absence of a False."""
    cells = {}
    for seed in (2021,):
        cells[f"AE_s{seed}"] = {"donor_sha256": "d"}
        cells[f"R0_s{seed}"] = {"steps": 10, "observed_updates": 10, "detector_unchanged_by_the_fit": False, "donor": None}
        cells[f"R1_s{seed}"] = {"steps": 10, "observed_updates": 10, "detector_unchanged_by_the_fit": True, "donor": "d"}
        cells[f"R2_s{seed}"] = {"steps": 10, "observed_updates": 10, "donor": "d"}          # no measurement at all
    checks = M.regime_checks(cells, seeds=(2021,))
    assert checks["R2_detector_changed_by_its_fit"] is None
    assert checks["complete"] is False
    assert checks["verdict"] in ("INCOMPLETE_POPULATION", "INCOMPLETE_EVIDENCE")
    assert any("R2_s2021" in str(x) for x in checks.get("cells_without_evidence", []))


@pytest.mark.parametrize("bad", [None, "yes", 1, 0])
def test_F5_an_untyped_detector_flag_is_not_evidence(bad):
    cells = {"AE_s2021": {"donor_sha256": "d"},
             "R0_s2021": {"steps": 1, "observed_updates": 1, "detector_unchanged_by_the_fit": False, "donor": None},
             "R1_s2021": {"steps": 1, "observed_updates": 1, "detector_unchanged_by_the_fit": bad, "donor": "d"},
             "R2_s2021": {"steps": 1, "observed_updates": 1, "detector_unchanged_by_the_fit": False, "donor": "d"}}
    checks = M.regime_checks(cells, seeds=(2021,))
    assert checks["R1_detector_unchanged_by_its_fit"] is None
    assert checks["complete"] is False


def test_F5_an_unexpected_cell_cannot_expand_the_denominator():
    """Adding R0 from a seed nobody asked for used to leave the verdict COMPLETE."""
    cells = {}
    for seed in (2021,):
        cells[f"AE_s{seed}"] = {"donor_sha256": "d"}
        for regime, unchanged in (("R0", False), ("R1", True), ("R2", False)):
            cells[f"{regime}_s{seed}"] = {"steps": 1, "observed_updates": 1,
                                          "detector_unchanged_by_the_fit": unchanged,
                                          "donor": None if regime == "R0" else "d"}
    clean = M.regime_checks(cells, seeds=(2021,))
    assert clean["complete"] is True and clean["verdict"] == "COMPLETE"
    intruder = dict(cells)
    intruder["R0_s9999"] = {"steps": 1, "observed_updates": 1, "detector_unchanged_by_the_fit": False, "donor": None}
    checks = M.regime_checks(intruder, seeds=(2021,))
    assert checks["unexpected_cells"] == ["R0_s9999"]
    assert checks["complete"] is False and checks["verdict"] == "UNEXPECTED_CELLS"


# --- RP157: the closure authenticates its design and binds every returned child ------------------------------------------------

def _fake_run(tmp_path, cells, *, design_sha="d" * 64, monitor=640, pred_len=96, seeds=(2021, 2022, 2023)):
    run_dir = tmp_path / "run"; run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "CONTRAST.json").write_text(json.dumps({
        "design_sha256": design_sha, "seeds": list(seeds), "pred_len": pred_len,
        "populations": {"validation_monitor_windows": monitor}, "cells": cells}))
    return run_dir


def _cell_record(regime, seed, **over):
    record = {"regime": regime, "seed": seed, "selected_epoch": 1, "model_path": "/nonexistent",
              "model_sha256": "a" * 64, "steps": 1, "observed_updates": 1,
              "detector_unchanged_by_the_fit": regime == "R1", "donor": None if regime == "R0" else "d"}
    record.update(over)
    return record


def _full_design(cells, *, sha="d" * 64, pred_len=96, seeds=(2021,)):
    """A design carrying every canonical field, so a test can exercise one factor at a time."""
    design = {"reference": {"model": "TimeFilter"}, "task": {"pred_len": pred_len, "channels": 321},
              "channel_order": {"sha256": "c" * 64}, "architecture": {"arch": "B"}, "regimes": {"R0": "", "R1": "", "R2": ""},
              "factorial": {"cells": sorted(cells), "seeds": list(seeds)}, "pretraining": {"mask_ratio": 0.25},
              "optimisation": {"loss": "mse"}, "exposure": {"outer_test": "NO_ACCESS"}}
    design["design_sha256"] = sha
    return design


def test_RP158_a_partial_object_cannot_authenticate_as_a_design(tmp_path):
    """An object that carries only a digest and a cell list is not this design; the SCHEMA names what identifies it."""
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in (2021,) for r in ("R0", "R1", "R2")}
    run_dir = _fake_run(tmp_path, cells, seeds=(2021,))
    partial = {"design_sha256": "d" * 64, "factorial": {"cells": sorted(cells)}, "task": {"pred_len": 96}}
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(Path("/nonexistent"), run_dir, design=partial)
    assert "canonical" in str(exc.value).lower()


def test_RP158_a_design_cannot_choose_which_of_its_fields_identify_it(tmp_path):
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in (2021,) for r in ("R0", "R1", "R2")}
    run_dir = _fake_run(tmp_path, cells, seeds=(2021,))
    design = _full_design(cells)
    design["identity_covers"] = ["task"]
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(Path("/nonexistent"), run_dir, design=design)
    assert "identifies it" in str(exc.value) or "canonical fields" in str(exc.value)


def test_RP157_a_design_whose_digest_is_not_the_runs_is_refused(tmp_path):
    """The two digests used to be reported and never compared; a supplied design could claim anything."""
    import hashlib, json as _json
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in (2021,) for r in ("R0", "R1", "R2")}
    run_dir = _fake_run(tmp_path, cells, seeds=(2021,))
    foreign = _full_design(cells)
    foreign["design_sha256"] = hashlib.sha256(_json.dumps(
        {k: foreign[k] for k in M.CANONICAL_DESIGN_FIELDS}, sort_keys=True, default=str).encode()).hexdigest()
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(Path("/nonexistent"), run_dir, design=foreign)
    assert "not the one this run recorded" in str(exc.value)


def test_RP158_a_declared_horizon_that_is_not_the_runs_is_refused(tmp_path):
    """A design declaring 192 while the run produced its cells at 96 used to complete."""
    import hashlib, json as _json
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in (2021,) for r in ("R0", "R1", "R2")}
    run_dir = _fake_run(tmp_path, cells, seeds=(2021,))
    design = _full_design(cells, pred_len=192)
    design["design_sha256"] = "d" * 64
    design["task"]["pred_len"] = 192
    # make the canonical digest agree with itself so only the horizon differs from the run
    design["design_sha256"] = hashlib.sha256(_json.dumps(
        {k: design[k] for k in M.CANONICAL_DESIGN_FIELDS}, sort_keys=True, default=str).encode()).hexdigest()
    run = _fake_run(tmp_path / "b", cells, design_sha=design["design_sha256"], seeds=(2021,), pred_len=96)
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(Path("/nonexistent"), run, design=design)
    assert "horizon" in str(exc.value).lower()


def test_RP158_nine_children_agreeing_on_a_wrong_count_do_not_make_it_right():
    """Setting BOTH window counts to 1 in ALL nine children used to complete, because the first sibling defined the reference."""
    expected = [f"{r}_s{s}" for s in (2021, 2022, 2023) for r in ("R0", "R1", "R2")]
    children = {}
    for cell in expected:
        seed = int(cell.split("_s")[1]); regime = cell.split("_s")[0]
        populations = {name: {"windows": 1, "elements": 1 * 96 * 321,
                              "author_float32": {"mae": 0.37, "mse": 0.29},
                              "independent_float64": {"mae": 0.37, "mse": 0.29},
                              "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
                              "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}
                       for name in ("complete_validation", "label_disjoint_from_selection")}
        children[cell] = {"cell": cell, "seed": seed, "regime": regime, "pred_len": 96,
                          "model_identity_reconciled": True, "populations": populations}
    report = M.validate_children(children, expected=expected, pred_len=96,
                                 populations=("complete_validation", "label_disjoint_from_selection"),
                                 expected_windows={"complete_validation": 2537, "label_disjoint_from_selection": 1802})
    assert report["bound"] is False
    assert any("support contract declares" in p for p in report["problems"])


@pytest.mark.parametrize("flag", ["false", "true", 1, 0, None])
def test_RP158_a_truthy_identity_flag_is_not_a_reconciliation(flag):
    expected = ["R0_s2021"]
    children = {"R0_s2021": {"cell": "R0_s2021", "seed": 2021, "regime": "R0", "pred_len": 96,
                             "model_identity_reconciled": flag,
                             "populations": {"complete_validation": {
                                 "windows": 2537, "elements": 2537 * 96 * 321,
                                 "author_float32": {"mae": 0.37, "mse": 0.29},
                                 "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
                                 "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}}}}
    report = M.validate_children(children, expected=expected, pred_len=96, populations=("complete_validation",),
                                 expected_windows={"complete_validation": 2537})
    assert report["bound"] is False
    assert any("boolean True" in p for p in report["problems"])


def test_RP158_a_non_finite_baseline_or_skill_is_refused():
    expected = ["R0_s2021"]

    def child(**over):
        population = {"windows": 2537, "elements": 2537 * 96 * 321,
                      "author_float32": {"mae": 0.37, "mse": 0.29},
                      "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
                      "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}
        population.update(over)
        return {"R0_s2021": {"cell": "R0_s2021", "seed": 2021, "regime": "R0", "pred_len": 96,
                             "model_identity_reconciled": True, "populations": {"complete_validation": population}}}

    for mutation, needle in ((
            {"matched_persistence_author_float32": {"mae": float("nan"), "mse": 1.5}}, "persistence"),
            ({"skill_mae_vs_persistence": float("inf")}, "skill"),
            ({"matched_persistence_author_float32": None}, "persistence")):
        report = M.validate_children(child(**mutation), expected=expected, pred_len=96,
                                     populations=("complete_validation",),
                                     expected_windows={"complete_validation": 2537})
        assert report["bound"] is False, mutation
        assert any(needle in p.lower() for p in report["problems"]), report["problems"]


def test_RP158_a_child_that_scored_another_checkpoint_is_refused():
    expected = ["R0_s2021"]
    children = {"R0_s2021": {"cell": "R0_s2021", "seed": 2021, "regime": "R0", "pred_len": 96,
                             "model_identity_reconciled": True, "model_sha256_on_disk": "b" * 64,
                             "populations": {"complete_validation": {
                                 "windows": 2537, "elements": 2537 * 96 * 321,
                                 "author_float32": {"mae": 0.37, "mse": 0.29},
                                 "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
                                 "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}}}}
    report = M.validate_children(children, expected=expected, pred_len=96, populations=("complete_validation",),
                                 expected_windows={"complete_validation": 2537}, checkpoints={"R0_s2021": "a" * 64})
    assert report["bound"] is False
    assert any("retained record declares" in p for p in report["problems"])


def test_RP157_an_empty_factorial_is_refused_rather_than_trivially_complete(tmp_path):
    run_dir = _fake_run(tmp_path, {}, seeds=(2021,))
    empty = {"design_sha256": "d" * 64, "factorial": {"cells": []}, "task": {"pred_len": 96}}
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(DATA if DATA.is_file() else Path("/nonexistent"), run_dir, design=empty)
    assert "empty" in str(exc.value).lower() or "no cells" in str(exc.value).lower()


def test_RP157_a_design_that_does_not_cover_the_runs_cells_is_refused(tmp_path):
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in (2021,) for r in ("R0", "R1", "R2")}
    run_dir = _fake_run(tmp_path, cells, seeds=(2021,))
    short = {"design_sha256": "d" * 64, "factorial": {"cells": ["R0_s2021"]}, "task": {"pred_len": 96}}
    with pytest.raises(M.DesignError):
        M.score_contrast(DATA if DATA.is_file() else Path("/nonexistent"), run_dir, design=short)


@pytest.mark.parametrize("mutation,reason", [
    ({"populations_drop": "label_disjoint_from_selection"}, "population"),
    ({"pred_len": 192}, "horizon"),
    ({"nan_mae": True}, "finite"),
    ({"cell_name": "R0_s9999"}, "cell"),
    ({"regime": "R9"}, "regime"),
    ({"drop_reduction": "author_float32"}, "reduction"),
])
def test_RP157_a_child_that_does_not_bind_to_its_expected_cell_suppresses_the_summary(mutation, reason):
    """Every returned child is bound to the cell it was asked for: its name, seed, regime, horizon, populations and typed
    finite reductions. A child that drifts on any of them is a problem, not an average over fewer seeds."""
    def child(cell):
        seed = int(cell.split("_s")[1]); regime = cell.split("_s")[0]
        populations = {}
        for name, windows in (("complete_validation", 2537), ("label_disjoint_from_selection", 1802)):
            populations[name] = {"windows": windows, "elements": windows * 96 * 321,
                                 "author_float32": {"mae": 0.37, "mse": 0.29},
                                 "independent_float64": {"mae": 0.37, "mse": 0.29},
                                 "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
                                 "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}
        record = {"cell": cell, "seed": seed, "regime": regime, "pred_len": 96,
                  "model_identity_reconciled": True, "populations": populations}
        if cell == "R2_s2023":
            if "populations_drop" in mutation:
                record["populations"].pop(mutation["populations_drop"])
            if "pred_len" in mutation:
                record["pred_len"] = mutation["pred_len"]
            if mutation.get("nan_mae"):
                record["populations"]["complete_validation"]["author_float32"]["mae"] = float("nan")
            if "cell_name" in mutation:
                record["cell"] = mutation["cell_name"]
            if "regime" in mutation:
                record["regime"] = mutation["regime"]
            if "drop_reduction" in mutation:
                record["populations"]["complete_validation"].pop(mutation["drop_reduction"])
        return record

    expected = [f"{r}_s{s}" for s in (2021, 2022, 2023) for r in ("R0", "R1", "R2")]
    report = M.validate_children({c: child(c) for c in expected}, expected=expected, pred_len=96,
                                 populations=("complete_validation", "label_disjoint_from_selection"))
    assert report["problems"], f"the {reason} mutation must be reported"
    assert any(reason in p.lower() for p in report["problems"]), report["problems"]
    assert report["bound"] is False


def test_RP157_a_clean_population_binds_and_permits_a_summary():
    expected = [f"{r}_s{s}" for s in (2021, 2022, 2023) for r in ("R0", "R1", "R2")]
    children = {}
    for cell in expected:
        seed = int(cell.split("_s")[1]); regime = cell.split("_s")[0]
        populations = {name: {"windows": w, "elements": w * 96 * 321,
                              "author_float32": {"mae": 0.37, "mse": 0.29},
                              "independent_float64": {"mae": 0.37, "mse": 0.29},
                              "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
                              "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}
                       for name, w in (("complete_validation", 2537), ("label_disjoint_from_selection", 1802))}
        children[cell] = {"cell": cell, "seed": seed, "regime": regime, "pred_len": 96,
                          "model_identity_reconciled": True, "populations": populations}
    report = M.validate_children(children, expected=expected, pred_len=96,
                                 populations=("complete_validation", "label_disjoint_from_selection"))
    assert report["bound"] is True and not report["problems"]
    assert report["windows_per_population"] == {"complete_validation": 2537, "label_disjoint_from_selection": 1802}


# --- RP158 continuation: the remaining acceptance gaps, and a closure that cannot run inference --------------------------------

def _population(**over):
    """A child population that is internally consistent: the skill FOLLOWS from the MAE and the baseline beside it."""
    entry = {"windows": 2537, "elements": 2537 * 96 * 321,
             "author_float32": {"mae": 0.37, "mse": 0.29},
             "independent_float64": {"mae": 0.37, "mse": 0.29},
             "matched_persistence_author_float32": {"mae": 0.86, "mse": 1.5},
             "skill_mae_vs_persistence": 1.0 - 0.37 / 0.86}
    entry.update(over)
    return entry


def _one_child(**over):
    return _child_with(_population(**over))


def _child_with(population):
    """Takes the population AS GIVEN, so a test can remove a field as well as change one."""
    return {"R0_s2021": {"cell": "R0_s2021", "seed": 2021, "regime": "R0", "pred_len": 96,
                         "model_identity_reconciled": True,
                         "populations": {"complete_validation": population}}}


def _check(children):
    return M.validate_children(children, expected=["R0_s2021"], pred_len=96, populations=("complete_validation",),
                               reductions=("author_float32", "independent_float64"),
                               expected_windows={"complete_validation": 2537})


def test_RP158_the_valid_child_still_passes_every_new_gate():
    """The gates below must reject what they claim to reject WITHOUT rejecting a correct record."""
    report = _check(_one_child())
    assert report["bound"] is True, report["problems"]


@pytest.mark.parametrize("elements", [None, "812373", 812373.0, True])
def test_RP158_an_element_count_that_is_absent_or_untyped_is_refused(elements):
    """It was validated only WHEN it was an integer, so omitting it passed. The count ties a metric to its own array."""
    population = _population()
    if elements is None:
        population.pop("elements")
    else:
        population["elements"] = elements
    report = _check(_child_with(population))
    assert report["bound"] is False
    assert any("element count" in p for p in report["problems"]), report["problems"]


@pytest.mark.parametrize("field,where", [("author_float32", "mae"), ("author_float32", "mse"),
                                         ("independent_float64", "mae"),
                                         ("matched_persistence_author_float32", "mae")])
def test_RP158_a_negative_loss_is_refused(field, where):
    """A mean of absolute or squared errors is never negative. Finiteness alone accepted -0.37 as a very good result."""
    population = _population()
    population[field] = {**population[field], where: -abs(population[field][where])}
    if field == "author_float32":
        population["skill_mae_vs_persistence"] = 1.0 - population["author_float32"]["mae"] / 0.86
    if field == "matched_persistence_author_float32" and where == "mae":
        population["skill_mae_vs_persistence"] = 1.0 - 0.37 / population[field]["mae"]
    report = _check(_child_with(population))
    assert report["bound"] is False
    assert any("never negative" in p for p in report["problems"]), report["problems"]


def test_RP158_a_skill_that_does_not_follow_from_its_own_record_is_refused():
    """0.9 is finite, in range and plausible. It is still not 1 - 0.37/0.86, and both numbers sit in the same record."""
    report = _check(_one_child(skill_mae_vs_persistence=0.9))
    assert report["bound"] is False
    assert any("does not follow from its own record" in p for p in report["problems"]), report["problems"]
    assert _check(_one_child())["bound"] is True


def test_RP158_a_null_skill_is_correct_only_against_a_zero_baseline():
    absent = _check(_one_child(skill_mae_vs_persistence=None))
    assert absent["bound"] is False
    assert any("only correct against a zero baseline" in p for p in absent["problems"]), absent["problems"]
    zero = _population(matched_persistence_author_float32={"mae": 0.0, "mse": 0.0},
                       author_float32={"mae": 0.0, "mse": 0.0},
                       independent_float64={"mae": 0.0, "mse": 0.0},
                       skill_mae_vs_persistence=None)
    assert _check(_child_with(zero))["bound"] is True, "an undefined quotient IS reported as null"
    report = _check(_child_with(dict(zero, skill_mae_vs_persistence=1.0)))
    assert report["bound"] is False
    assert any("zero baseline" in p for p in report["problems"]), report["problems"]


def _retained_run(tmp_path, *, seeds=(2021, 2022, 2023)):
    """A run as it survives on disk: its contrast, its design and the children a scoring pass already wrote."""
    import hashlib, json as _json
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in seeds for r in ("R0", "R1", "R2")}
    design = _full_design(cells, seeds=seeds)
    design["design_sha256"] = hashlib.sha256(_json.dumps(
        {k: design[k] for k in M.CANONICAL_DESIGN_FIELDS}, sort_keys=True, default=str).encode()).hexdigest()
    run_dir = _fake_run(tmp_path, cells, design_sha=design["design_sha256"], seeds=seeds)
    (run_dir / "DESIGN.json").write_text(json.dumps(design))
    children = {}
    for cell in cells:
        regime, _, seed = cell.partition("_s")
        children[cell] = {"cell": cell, "seed": int(seed), "regime": regime, "pred_len": 96,
                          "model_identity_reconciled": True, "model_sha256_on_disk": "a" * 64,
                          "populations": {"complete_validation": _population(),
                                          "label_disjoint_from_selection": _population(windows=1802,
                                                                                       elements=1802 * 96 * 321)}}
    (run_dir / "SCORING.json").write_text(json.dumps({
        "schema": "df_ecl_modular_scoring.v3", "cells": children,
        "child_binding": {"expected_windows_from_contract": {"complete_validation": 2537,
                                                             "label_disjoint_from_selection": 1802}}}))
    return run_dir


def test_RP158_the_closure_over_retained_records_runs_no_inference_and_starts_no_process(tmp_path, monkeypatch):
    """The requested pure closure. If it can still reach a subprocess, 'no inference' is a description of intent, not a fact."""
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("the closure started a process; it must read the retained records and nothing else")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    monkeypatch.setattr(M, "score_cell", forbidden)
    run_dir = _retained_run(tmp_path)
    before = (run_dir / "SCORING.json").read_bytes()
    closed = M.close_retained_run(run_dir)
    assert closed["status"] == "COMPLETE", closed["problems"]
    assert closed["source"] == "RETAINED_RECORDS_NO_INFERENCE"
    assert closed["fresh_process_per_cell"] is False
    assert sorted(closed["by_regime"]) == ["R0", "R1", "R2"]
    assert closed["reconciliation"]["cells_scored"] == 9
    assert (run_dir / "SCORING.json").read_bytes() == before, "a re-closure must not overwrite the record it read"
    assert (run_dir / "CLOSURE.json").is_file()


def test_RP158_the_pure_closure_applies_the_same_gates_as_the_dispatching_one(tmp_path, monkeypatch):
    """One set of rules in one place: a corrupted retained child must fail the closure exactly as it fails a fresh scoring."""
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no process")))
    run_dir = _retained_run(tmp_path)
    scoring = json.loads((run_dir / "SCORING.json").read_text())
    scoring["cells"]["R1_s2022"]["populations"]["complete_validation"]["skill_mae_vs_persistence"] = 0.9
    (run_dir / "SCORING.json").write_text(json.dumps(scoring))
    closed = M.close_retained_run(run_dir)
    assert closed["status"] == "INCOMPLETE_EVIDENCE"
    assert closed["by_regime"] is None
    assert any("does not follow from its own record" in p for p in closed["problems"])


def test_RP158_a_closure_without_the_contract_counts_refuses_instead_of_deriving_them(tmp_path):
    """Recomputing the expectations would mean rebuilding the datasets, and a closure that does that can only agree with itself."""
    run_dir = _retained_run(tmp_path)
    scoring = json.loads((run_dir / "SCORING.json").read_text())
    scoring["child_binding"] = {}
    (run_dir / "SCORING.json").write_text(json.dumps(scoring))
    with pytest.raises(M.DesignError) as exc:
        M.close_retained_run(run_dir)
    assert "agree with itself" in str(exc.value)


def test_RP158_a_scoring_retains_the_design_it_authenticated_so_a_later_closure_needs_no_data(tmp_path, monkeypatch):
    """A run whose design was only re-derived can never be closed again without the datasets. Retaining it once ends that."""
    import hashlib, json as _json
    cells = {f"{r}_s{s}": _cell_record(r, s) for s in (2021,) for r in ("R0", "R1", "R2")}
    design = _full_design(cells, seeds=(2021,))
    design["design_sha256"] = hashlib.sha256(_json.dumps(
        {k: design[k] for k in M.CANONICAL_DESIGN_FIELDS}, sort_keys=True, default=str).encode()).hexdigest()
    run_dir = _fake_run(tmp_path / "r", cells, design_sha=design["design_sha256"], seeds=(2021,))
    monkeypatch.setattr(M, "author_datasets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("stop after retention")))
    with pytest.raises(AssertionError):
        M.score_contrast(Path("/nonexistent"), run_dir, design=design)
    retained = json.loads((run_dir / "DESIGN.json").read_text())
    assert retained["design_sha256"] == design["design_sha256"], "the design it authenticated is the one it retained"
    original = (run_dir / "DESIGN.json").read_bytes()
    with pytest.raises(AssertionError):
        M.score_contrast(Path("/nonexistent"), run_dir, design=design)
    assert (run_dir / "DESIGN.json").read_bytes() == original, "a retained design is never overwritten by a later pass"
