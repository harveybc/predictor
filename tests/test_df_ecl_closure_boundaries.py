"""RP159: Musashi's five closure boundaries (MUSASHI_RP157_REVIEW_2026_09_24.md), each with its positive case and EVERY
mandatory refusal he enumerated.

His table is the map, and the test names carry it:

    B1 design       retained accepted design, required canonical fields, recomputed content, accepted run linkage
                    refuse: absent/empty/altered coverage, missing canonical field, stale or re-digested foreign design,
                            no retained authority
    B2 dispatch     horizon, seed/regime axes, data/scaler/channel order and monitor support from that validated contract
                    refuse: caller overrides conflicting with the design; duplicate, missing or unexpected cell; silent reseal
    B3 population   exact origin/target support and the resulting windows/elements per population
                    refuse: all children consistently wrong; inconsistent elements; omitted or extra population;
                            bool/string counts
    B4 checkpoint   the exact recorded AND observed digest, bound to the expected selected model
                    refuse: missing/mismatched digest, a truthy string or number in place of a literal boolean, wrong
                            cell/model
    B5 metrics      the named model AND naive reductions, finite nonnegative losses, a recomputed skill with a defined
                    zero-naive behaviour
                    refuse: NaN/inf/bool/string in any measured numeric field, omitted reduction, contradictory derived skill

Nothing here fits, loads or scores a model: every case is a record at the contract boundary. The last section closes the
RETAINED nine-cell run from evidence on disk with the subprocess transport, `score_cell` and `author_datasets` all replaced by
raising doubles, so a closure that reached inference would fail rather than quietly cost nine replays.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/audits/evidence/d3_k5_20260917"


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


M = _load("df_ecl_modular")

SEEDS = (2021, 2022, 2023)
REGIMES = ("R0", "R1", "R2")
CELLS = [f"{r}_s{s}" for r in REGIMES for s in SEEDS]
CHANNEL_ORDER = {"n_channels": 321, "first": ["0", "1", "2"], "last": ["318", "319", "OT"], "sha256": "c" * 64}
COMPLETE, DISJOINT = "complete_validation", "label_disjoint_from_selection"


# --- fixtures: a run, its design, and the children a scoring pass wrote ---------------------------------------------------

def _digest(design: dict) -> str:
    return hashlib.sha256(json.dumps({k: design[k] for k in M.CANONICAL_DESIGN_FIELDS},
                                     sort_keys=True, default=str).encode()).hexdigest()


def _design(*, cells=None, pred_len=96, seeds=SEEDS, channel_order=None, **over) -> dict:
    """A design carrying every canonical field and declaring the coverage this schema requires, so one factor moves at a time."""
    design = {"reference": {"model": "TimeFilter", "revision": "pinned"},
              "task": {"pred_len": pred_len, "channels": 321, "seq_len": 96, "label_len": 48,
                       "scaler": "the author's StandardScaler fitted on TRAIN rows only"},
              "channel_order": dict(channel_order or CHANNEL_ORDER),
              "architecture": {"arch": "B", "fusion": "sequence"},
              "regimes": {r: {} for r in REGIMES},
              "factorial": {"cells": sorted(cells or CELLS), "seeds": list(seeds), "regimes": list(REGIMES)},
              "pretraining": {"objective": "masked reconstruction"},
              "optimisation": {"loss": "MSE on the normalized target"},
              "exposure": {"outer_test": "NO_ACCESS during any fit or selection"}}
    design.update(over)
    design["identity_covers"] = list(M.CANONICAL_DESIGN_FIELDS)
    design["design_sha256"] = _digest(design)
    return design


def _contrast(design: dict, *, pred_len=96, seeds=SEEDS, available=2537, monitor=640, cells=None) -> dict:
    names = sorted(cells or design["factorial"]["cells"])
    return {"schema": "df_ecl_modular_contrast.v1", "design_sha256": design["design_sha256"], "pred_len": pred_len,
            "seeds": list(seeds),
            "populations": {"validation_windows_available": available, "validation_monitor_windows": monitor},
            "cells": {c: {"regime": c.partition("_s")[0], "seed": int(c.partition("_s")[2]), "selected_epoch": 3,
                          "model_path": f"/nonexistent/{c}.weights.h5", "model_sha256": _model_digest(c)} for c in names}}


def _model_digest(cell: str) -> str:
    return hashlib.sha256(cell.encode()).hexdigest()


def _population(*, windows=2537, mae=0.37, naive=0.86, **over) -> dict:
    entry = {"windows": windows, "elements": windows * 96 * 321,
             "author_float32": {"mae": mae, "mse": 0.29},
             "independent_float64": {"mae": mae, "mse": 0.29},
             "matched_persistence_author_float32": {"mae": naive, "mse": 1.5},
             "skill_mae_vs_persistence": (1.0 - mae / naive) if naive else None}
    entry.update(over)
    return entry


def _child(cell: str, **over) -> dict:
    regime, _, seed = cell.partition("_s")
    record = {"cell": cell, "seed": int(seed), "regime": regime, "pred_len": 96, "selected_epoch": 3,
              "model_sha256_recorded": _model_digest(cell), "model_sha256_on_disk": _model_digest(cell),
              "model_identity_reconciled": True,
              "populations": {COMPLETE: _population(), DISJOINT: _population(windows=1802)}}
    record.update(over)
    return record


def _children(cells=None) -> dict:
    return {c: _child(c) for c in (cells or CELLS)}


def _retained_run(tmp_path, *, design=None, contrast=None, children=None, write_design=True, scoring_extra=None) -> Path:
    """A run as it survives on disk after a scoring pass: its contrast, the design it authenticated, and the child records."""
    design = design or _design()
    contrast = contrast or _contrast(design)
    run_dir = Path(tmp_path) / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "CONTRAST.json").write_text(json.dumps(contrast))
    if write_design:
        (run_dir / "DESIGN.json").write_text(json.dumps(design))
    scoring = {"schema": "df_ecl_modular_scoring.v3", "cells": children if children is not None else _children(),
               "child_binding": {"expected_windows_from_contract": {COMPLETE: 2537, DISJOINT: 1802}}}
    scoring.update(scoring_extra or {})
    (run_dir / "SCORING.json").write_text(json.dumps(scoring))
    return run_dir


def _no_inference(monkeypatch):
    """Every route to a model is a raising double: a boundary test that silently replayed nine models would be the bug."""
    def forbidden(*args, **kwargs):
        raise AssertionError("this path reached inference; the closure must read retained records and nothing else")

    for name in ("run", "Popen", "check_output", "call", "check_call"):
        monkeypatch.setattr(subprocess, name, forbidden, raising=False)
    monkeypatch.setattr(M, "score_cell", forbidden)
    monkeypatch.setattr(M, "author_datasets", forbidden)
    return forbidden


def _check(children, *, expected=None, checkpoints=None, populations=(COMPLETE, DISJOINT),
           expected_windows=None) -> dict:
    expected = expected if expected is not None else sorted(children)
    return M.validate_children(children, expected=expected, pred_len=96, populations=populations,
                               reductions=("author_float32", "independent_float64"),
                               expected_windows=expected_windows if expected_windows is not None
                               else {COMPLETE: 2537, DISJOINT: 1802},
                               checkpoints=checkpoints if checkpoints is not None
                               else {c: _model_digest(c) for c in expected})


def _problems(report) -> str:
    return " | ".join(report["problems"]).lower()


# =============================================================================================================================
# B1 DESIGN -- retained accepted design, required canonical fields, recomputed content, accepted run linkage
# =============================================================================================================================

def test_B1_design_positive_a_retained_accepted_design_authenticates_against_the_run_that_produced_the_cells():
    design = _design()
    authenticated = M.authenticate_design(_contrast(design), design)
    assert authenticated["design_sha256"] == design["design_sha256"]
    assert authenticated["cells"] == sorted(CELLS) and authenticated["pred_len"] == 96
    assert authenticated["seeds"] == list(SEEDS)
    assert "recomputed" in authenticated["authenticated"]


def test_B1_design_refuses_absent_coverage():
    """Musashi's `missing_canonical_cover`: with no declared coverage the recomputation used to be skipped entirely."""
    design = _design()
    design.pop("identity_covers")
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(_contrast(design), design)
    assert "identity" in str(exc.value).lower()


def test_B1_design_refuses_empty_coverage():
    """`empty_canonical_cover`: an empty list is not 'covers everything'; it is an object declining to be identified."""
    design = _design()
    design["identity_covers"] = []
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(_contrast(design), design)
    assert "identity" in str(exc.value).lower()


def test_B1_design_refuses_altered_coverage():
    design = _design()
    design["identity_covers"] = ["task", "factorial"]
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(_contrast(design), design)
    assert "does not choose what identifies it" in str(exc.value)


@pytest.mark.parametrize("field", list(M.CANONICAL_DESIGN_FIELDS))
def test_B1_design_refuses_a_missing_canonical_field(field):
    """Every canonical scientific field is required by THIS schema; a partial object cannot authenticate as this design."""
    design = _design()
    contrast = _contrast(design)
    design.pop(field)
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert field in str(exc.value)


def test_B1_design_refuses_a_re_digested_foreign_design():
    """A foreign design re-hashed so it agrees with ITSELF still is not the digest the run recorded when it made these cells."""
    run_design = _design()
    contrast = _contrast(run_design)
    foreign = _design(pretraining={"objective": "a different pre-training objective entirely"})
    assert foreign["design_sha256"] == _digest(foreign) != run_design["design_sha256"]
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, foreign)
    assert "not the one this run recorded" in str(exc.value)


def test_B1_design_refuses_a_stale_design_whose_content_no_longer_hashes_to_its_claim():
    """The stale case: a design edited after it was sealed, still carrying the digest it had before the edit."""
    design = _design()
    contrast = _contrast(design)
    design["task"]["scaler"] = "a scaler fitted on all the rows"          # the content moved, the claim did not
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert "hashes to" in str(exc.value)


def test_B1_design_refuses_when_no_retained_authority_exists(tmp_path):
    """The pure closure has no data to re-derive from and must say so, not manufacture the authority it then checks against."""
    run_dir = _retained_run(tmp_path, write_design=False)
    with pytest.raises(M.DesignError) as exc:
        M.close_retained_run(run_dir)
    assert "DESIGN.json" in str(exc.value)


def test_B1_design_refuses_a_re_derivation_that_is_not_the_runs_recorded_authority(tmp_path, monkeypatch):
    """No retained design and no supplied one: a fresh local seal is not authority unless it reproduces the recorded digest."""
    design = _design()
    run_dir = _retained_run(tmp_path, design=design, write_design=False)
    monkeypatch.setattr(M, "seal_contrast", lambda *a, **k: _design(pretraining={"objective": "something else"}))
    monkeypatch.setattr(M, "author_datasets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no data work")))
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(Path("/nonexistent"), run_dir)
    assert "not the one this run recorded" in str(exc.value)


# =============================================================================================================================
# B2 DISPATCH TASK -- horizon, seed/regime axes, data/scaler/channel order and monitor support from the validated contract
# =============================================================================================================================

def test_B2_dispatch_task_positive_is_taken_from_the_validated_contract():
    design = _design()
    contrast = _contrast(design)
    task = M.bind_dispatch_task(contrast, M.authenticate_design(contrast, design), design,
                                requested_pred_len=96, observed_channel_order=dict(CHANNEL_ORDER))
    assert task["pred_len"] == 96 and task["seeds"] == list(SEEDS)
    assert task["cells"] == sorted(CELLS)
    assert task["channel_order_sha256"] == CHANNEL_ORDER["sha256"]
    assert task["monitor_windows"] == 640
    assert task["expected_windows"] == {COMPLETE: 2537, DISJOINT: 1802}


def test_B2_dispatch_refuses_a_caller_horizon_that_conflicts_with_the_design():
    """Musashi's `contradictory_design_horizon` in the other direction: the caller's 192 used to be silently replaced by 96."""
    design = _design()
    contrast = _contrast(design)
    with pytest.raises(M.DesignError) as exc:
        M.bind_dispatch_task(contrast, M.authenticate_design(contrast, design), design, requested_pred_len=192,
                             observed_channel_order=dict(CHANNEL_ORDER))
    assert "192" in str(exc.value) and "caller" in str(exc.value).lower()


def test_B2_dispatch_refuses_a_design_horizon_that_is_not_the_runs():
    design = _design(pred_len=192)
    contrast = _contrast(design, pred_len=96)
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert "horizon" in str(exc.value).lower()


def test_B2_dispatch_refuses_a_seed_axis_that_is_not_the_runs():
    design = _design(seeds=(2021, 2022, 2099))
    contrast = _contrast(design, seeds=SEEDS)
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert "seeds" in str(exc.value).lower()


def test_B2_dispatch_refuses_a_duplicate_declared_cell():
    """A cell declared twice is not a larger population; it used to collapse into the set and change the denominator."""
    design = _design()
    design["factorial"]["cells"] = sorted(CELLS) + ["R2_s2023"]
    design["design_sha256"] = _digest(design)
    contrast = _contrast(design, cells=CELLS)
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert "more than once" in str(exc.value)


def test_B2_dispatch_refuses_a_cell_the_run_does_not_hold():
    design = _design(cells=CELLS + ["R3_s2024"])
    contrast = _contrast(design, cells=CELLS)
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert "does not hold" in str(exc.value)


def test_B2_dispatch_refuses_a_cell_the_design_does_not_declare():
    design = _design(cells=[c for c in CELLS if c != "R2_s2023"])
    contrast = _contrast(design, cells=CELLS)
    with pytest.raises(M.DesignError) as exc:
        M.authenticate_design(contrast, design)
    assert "does not declare" in str(exc.value)


def test_B2_dispatch_refuses_data_whose_channel_order_is_not_the_designs():
    """A re-sorted or subset column list is a different task, however similar the file name is."""
    design = _design()
    contrast = _contrast(design)
    resorted = dict(CHANNEL_ORDER, sha256="e" * 64)
    with pytest.raises(M.DesignError) as exc:
        M.bind_dispatch_task(contrast, M.authenticate_design(contrast, design), design, requested_pred_len=96,
                             observed_channel_order=resorted)
    assert "channel" in str(exc.value).lower()


def test_B2_dispatch_refuses_a_monitor_support_the_run_record_does_not_declare():
    """The monitor support comes from the accepted run record; an undeclared one is refused, never invented."""
    design = _design()
    contrast = _contrast(design)
    contrast["populations"].pop("validation_monitor_windows")
    with pytest.raises(M.DesignError) as exc:
        M.bind_dispatch_task(contrast, M.authenticate_design(contrast, design), design, requested_pred_len=96,
                             observed_channel_order=dict(CHANNEL_ORDER))
    assert "will not invent it" in str(exc.value)


def test_B2_dispatch_refuses_a_silent_reseal_of_the_retained_design(tmp_path, monkeypatch):
    """A supplied design that is not the retained one must be refused rather than quietly taking its place."""
    retained = _design()
    run_dir = _retained_run(tmp_path, design=retained)
    supplied = _design(optimisation={"loss": "a different loss"})
    monkeypatch.setattr(M, "author_datasets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no data work")))
    with pytest.raises(M.DesignError) as exc:
        M.score_contrast(Path("/nonexistent"), run_dir, design=supplied)
    assert "retained" in str(exc.value).lower()
    assert json.loads((run_dir / "DESIGN.json").read_text())["design_sha256"] == retained["design_sha256"]


# =============================================================================================================================
# B3 POPULATION -- exact origin/target support and the resulting windows/elements per population
# =============================================================================================================================

def test_B3_population_positive_exact_support_and_elements_in_every_population():
    report = _check(_children())
    assert report["bound"] is True, report["problems"]
    assert report["windows_per_population"] == {COMPLETE: 2537, DISJOINT: 1802}


def test_B3_population_the_support_is_computed_from_the_accepted_record_not_assumed():
    """2537 available windows behind a 640-window monitor leave 1802 label-disjoint origins at horizon 96."""
    assert M.contract_windows_from_run({"populations": {"validation_windows_available": 2537,
                                                        "validation_monitor_windows": 640}}, 96) == {COMPLETE: 2537,
                                                                                                     DISJOINT: 1802}


def test_B3_population_refuses_all_children_consistently_wrong():
    """Musashi's `consistent_wrong_counts`: nine children agreeing on 1 window are nine children being wrong together."""
    children = _children()
    for child in children.values():
        for population in child["populations"].values():
            population.update(windows=1, elements=96 * 321)
    report = _check(children)
    assert report["bound"] is False
    assert "support contract declares" in _problems(report)


def test_B3_population_refuses_inconsistent_elements():
    children = _children()
    children["R1_s2022"]["populations"][COMPLETE]["elements"] = 2537 * 96 * 320
    report = _check(children)
    assert report["bound"] is False
    assert "elements" in _problems(report)


def test_B3_population_refuses_an_omitted_population():
    children = _children()
    children["R2_s2023"]["populations"].pop(DISJOINT)
    report = _check(children)
    assert report["bound"] is False
    assert f"omits the {DISJOINT} population" in _problems(report)


def test_B3_population_refuses_an_extra_population():
    """A population nobody asked for is not a bonus: the aggregate is defined over the declared ones, and a summary that
    silently ignores an extra reading cannot say which support its numbers came from."""
    children = _children()
    children["R0_s2021"]["populations"]["validation_monitor_only"] = _population(windows=640)
    report = _check(children)
    assert report["bound"] is False
    assert "nobody asked for" in _problems(report) or "unexpected population" in _problems(report)


@pytest.mark.parametrize("count", [True, False, "2537", 2537.0, None])
def test_B3_population_refuses_bool_or_string_window_counts(count):
    children = _children()
    population = children["R0_s2021"]["populations"][COMPLETE]
    if count is None:
        population.pop("windows")
    else:
        population["windows"] = count
    report = _check(children)
    assert report["bound"] is False
    assert "window count" in _problems(report)


@pytest.mark.parametrize("elements", [True, "78180192", 78180192.0, None])
def test_B3_population_refuses_bool_or_string_element_counts(elements):
    children = _children()
    population = children["R0_s2021"]["populations"][COMPLETE]
    if elements is None:
        population.pop("elements")
    else:
        population["elements"] = elements
    report = _check(children)
    assert report["bound"] is False
    assert "element count" in _problems(report)


# =============================================================================================================================
# B4 CHECKPOINT -- the exact recorded AND observed digest, bound to the expected selected model
# =============================================================================================================================

def test_B4_checkpoint_positive_recorded_and_observed_digests_bind_the_expected_model():
    report = _check(_children())
    assert report["bound"] is True, report["problems"]


@pytest.mark.parametrize("field", ["model_sha256_recorded", "model_sha256_on_disk"])
def test_B4_checkpoint_refuses_a_missing_digest(field):
    children = _children()
    children["R1_s2021"].pop(field)
    report = _check(children)
    assert report["bound"] is False
    assert "digest" in _problems(report)


def test_B4_checkpoint_refuses_a_run_record_that_declares_no_digest_for_an_expected_cell():
    """With no digest in the accepted record there is nothing to bind the scored bytes to, and silence is not agreement."""
    checkpoints = {c: _model_digest(c) for c in CELLS}
    checkpoints["R2_s2022"] = None
    report = _check(_children(), checkpoints=checkpoints)
    assert report["bound"] is False
    assert "declares no checkpoint digest" in _problems(report)


def test_B4_checkpoint_refuses_a_digest_that_does_not_match_the_retained_record():
    children = _children()
    children["R0_s2022"]["model_sha256_on_disk"] = "b" * 64
    children["R0_s2022"]["model_sha256_recorded"] = "b" * 64
    report = _check(children)
    assert report["bound"] is False
    assert "retained record declares" in _problems(report)


def test_B4_checkpoint_refuses_a_recorded_digest_that_is_not_the_observed_one():
    """The child hashed the file it actually loaded; a record that disagrees with its own observation is not reconciled."""
    children = _children()
    children["R1_s2023"]["model_sha256_recorded"] = "b" * 64
    report = _check(children)
    assert report["bound"] is False
    assert "observed" in _problems(report) or "on disk" in _problems(report)


@pytest.mark.parametrize("flag", ["true", "false", 1, 0, None, "True"])
def test_B4_checkpoint_refuses_a_truthy_string_or_number_in_place_of_a_literal_boolean(flag):
    children = _children()
    children["R2_s2021"]["model_identity_reconciled"] = flag
    report = _check(children)
    assert report["bound"] is False
    assert "boolean true" in _problems(report)


def test_B4_checkpoint_refuses_the_wrong_cell():
    children = _children()
    children["R2_s2023"]["cell"] = "R0_s2021"
    report = _check(children)
    assert report["bound"] is False
    assert "names cell" in _problems(report)


def test_B4_checkpoint_refuses_a_child_for_a_cell_nobody_asked_for():
    children = _children()
    children["R9_s2099"] = _child("R9_s2099")
    report = _check(children, expected=sorted(CELLS))
    assert report["bound"] is False
    assert "nobody asked for" in _problems(report)


# =============================================================================================================================
# B5 METRICS -- named model AND naive reductions, finite nonnegative losses, recomputed skill, defined zero-naive behaviour
# =============================================================================================================================

def test_B5_metrics_positive_named_model_and_naive_reductions_with_a_skill_that_follows_from_them():
    report = _check(_children())
    assert report["bound"] is True, report["problems"]


@pytest.mark.parametrize("reduction", ["author_float32", "independent_float64", "matched_persistence_author_float32"])
@pytest.mark.parametrize("metric", ["mae", "mse"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.37", None])
def test_B5_metrics_refuses_a_nonfinite_or_untyped_measured_value(reduction, metric, value):
    """Musashi's `nonfinite_naive` generalised: the naive reduction is a measured number like any other, and a NaN there used
    to ride into a COMPLETE summary because only the two model reductions were checked."""
    children = _children()
    population = children["R2_s2023"]["populations"][DISJOINT]
    population[reduction] = dict(population[reduction])
    population[reduction][metric] = value
    population["skill_mae_vs_persistence"] = None                      # the skill cannot follow from a value like this
    report = _check(children)
    assert report["bound"] is False
    assert "finite" in _problems(report) or "zero baseline" in _problems(report)


@pytest.mark.parametrize("reduction", ["author_float32", "independent_float64", "matched_persistence_author_float32"])
def test_B5_metrics_refuses_a_negative_loss(reduction):
    children = _children()
    population = children["R0_s2023"]["populations"][COMPLETE]
    population[reduction] = dict(population[reduction], mse=-0.29)
    report = _check(children)
    assert report["bound"] is False
    assert "never negative" in _problems(report)


@pytest.mark.parametrize("reduction", ["author_float32", "independent_float64"])
def test_B5_metrics_refuses_an_omitted_model_reduction(reduction):
    children = _children()
    children["R1_s2021"]["populations"][COMPLETE].pop(reduction)
    report = _check(children)
    assert report["bound"] is False
    assert f"omits the {reduction} reduction" in _problems(report)


def test_B5_metrics_refuses_an_omitted_naive_reduction():
    """Without the matched baseline there is no skill and no scale: 0.37 MAE is neither good nor bad on its own."""
    children = _children()
    children["R1_s2021"]["populations"][COMPLETE].pop("matched_persistence_author_float32")
    report = _check(children)
    assert report["bound"] is False
    assert "persistence" in _problems(report)


def test_B5_metrics_refuses_a_contradictory_derived_skill():
    children = _children()
    children["R0_s2021"]["populations"][COMPLETE]["skill_mae_vs_persistence"] = 0.9
    report = _check(children)
    assert report["bound"] is False
    assert "does not follow from its own record" in _problems(report)


def test_B5_metrics_defines_the_zero_naive_case_and_refuses_a_number_against_it():
    zero = _population(mae=0.0, naive=0.0)
    zero["skill_mae_vs_persistence"] = None
    children = _children()
    children["R0_s2021"]["populations"][COMPLETE] = copy.deepcopy(zero)
    assert _check(children)["bound"] is True, "an undefined quotient IS reported as null"
    children["R0_s2021"]["populations"][COMPLETE]["skill_mae_vs_persistence"] = 1.0
    report = _check(children)
    assert report["bound"] is False
    assert "zero baseline" in _problems(report)


def test_B5_metrics_refuses_a_null_skill_against_a_baseline_that_is_not_zero():
    children = _children()
    children["R0_s2021"]["populations"][COMPLETE]["skill_mae_vs_persistence"] = None
    report = _check(children)
    assert report["bound"] is False
    assert "only correct against a zero baseline" in _problems(report)


# =============================================================================================================================
# The offline validation-and-aggregation path: the retained evidence closes with no subprocess inference at all
# =============================================================================================================================

def test_the_offline_closure_of_a_retained_nine_cell_run_starts_no_process_and_loads_no_dataset(tmp_path, monkeypatch):
    _no_inference(monkeypatch)
    run_dir = _retained_run(tmp_path)
    before = (run_dir / "SCORING.json").read_bytes()
    closed = M.close_retained_run(run_dir)
    assert closed["status"] == "COMPLETE", closed["problems"]
    assert closed["source"] == "RETAINED_RECORDS_NO_INFERENCE" and closed["fresh_process_per_cell"] is False
    assert closed["reconciliation"]["cells_expected"] == 9 and closed["reconciliation"]["cells_scored"] == 9
    assert sorted(closed["by_regime"]) == ["R0", "R1", "R2"]
    assert (run_dir / "SCORING.json").read_bytes() == before, "a re-closure never rewrites the record it read"
    assert (run_dir / "CLOSURE.json").is_file()


def test_the_offline_closure_applies_every_boundary_it_was_given(tmp_path, monkeypatch):
    """One set of rules in one place: each boundary's counterexample must fail the offline path exactly as it fails a dispatch."""
    _no_inference(monkeypatch)
    for mutate, needle in (
            (lambda s: s["cells"]["R0_s2021"]["populations"][COMPLETE].update(windows=1, elements=96 * 321), "support"),
            (lambda s: s["cells"]["R1_s2022"].update(model_identity_reconciled="true"), "boolean true"),
            (lambda s: s["cells"]["R2_s2023"].update(model_sha256_on_disk="b" * 64), "retained record declares"),
            (lambda s: s["cells"]["R0_s2022"]["populations"][DISJOINT].update(skill_mae_vs_persistence=0.9), "follow"),
            (lambda s: s["cells"]["R1_s2023"]["populations"][DISJOINT].pop("matched_persistence_author_float32"),
             "persistence")):
        run_dir = _retained_run(tmp_path / f"case{needle[:4]}")
        scoring = json.loads((run_dir / "SCORING.json").read_text())
        mutate(scoring)
        (run_dir / "SCORING.json").write_text(json.dumps(scoring))
        closed = M.close_retained_run(run_dir)
        assert closed["status"] == "INCOMPLETE_EVIDENCE", needle
        assert closed["by_regime"] is None
        assert any(needle in p.lower() for p in closed["problems"]), (needle, closed["problems"])


RETAINED = {"contrast": EVIDENCE / "RP155/CONTRAST.json", "scoring": EVIDENCE / "RP158_SCORING.json",
            "design": EVIDENCE / "RP159/DESIGN.json"}
needs_retained = pytest.mark.skipif(not all(p.is_file() for p in RETAINED.values()),
                                    reason="the retained nine-cell contrast evidence is not in this checkout")


@needs_retained
def test_the_retained_nine_cell_contrast_closes_offline_with_no_inference_and_the_published_numbers(tmp_path, monkeypatch):
    """The real thing: the nine retained cells of the ECL modular contrast, closed from evidence on disk.

    `subprocess`, `score_cell` and `author_datasets` are raising doubles, so this test fails rather than replays nine models.
    The design is the one retained beside the evidence; it reproduces the digest the run recorded when it produced the cells."""
    _no_inference(monkeypatch)
    run_dir = tmp_path / "retained"
    run_dir.mkdir()
    (run_dir / "CONTRAST.json").write_text(RETAINED["contrast"].read_text())
    (run_dir / "DESIGN.json").write_text(RETAINED["design"].read_text())
    scoring = json.loads(RETAINED["scoring"].read_text())
    (run_dir / "SCORING.json").write_text(json.dumps(scoring))

    closed = M.close_retained_run(run_dir)

    assert closed["status"] == "COMPLETE", closed["problems"]
    assert closed["source"] == "RETAINED_RECORDS_NO_INFERENCE"
    assert closed["design_sha256"] == "b5d5eee1b5fce981f4627352ed516d5745b1e16996507a8c63fffc53518cadd9"
    assert closed["child_binding"]["expected_windows_from_contract"] == {COMPLETE: 2537, DISJOINT: 1802}
    assert closed["reconciliation"]["cells_scored"] == 9 and not closed["reconciliation"]["identity_failures"]
    published = {"R0": 0.371174, "R1": 0.374584, "R2": 0.368596}
    for regime, mae in published.items():
        block = closed["by_regime"][regime][DISJOINT]
        assert block["seeds"] == 3
        assert round(block["author_float32"]["mae_mean"], 6) == mae
        assert round(block["matched_persistence_author_float32_mae"], 6) == 0.868283
    assert closed["by_regime"]["R2"][DISJOINT]["author_float32"]["mae_mean"] < \
           closed["by_regime"]["R0"][DISJOINT]["author_float32"]["mae_mean"], "the retained development reading is unchanged"


@needs_retained
def test_the_retained_evidence_refuses_a_foreign_design_even_when_every_child_is_valid(tmp_path, monkeypatch):
    """Reuse of the retained numbers is permitted only after the AUTHORITY passes, not because the rows look consistent."""
    _no_inference(monkeypatch)
    run_dir = tmp_path / "retained"
    run_dir.mkdir()
    (run_dir / "CONTRAST.json").write_text(RETAINED["contrast"].read_text())
    design = json.loads(RETAINED["design"].read_text())
    design["task"] = dict(design["task"], pred_len=192)
    design["design_sha256"] = _digest(design)
    (run_dir / "DESIGN.json").write_text(json.dumps(design))
    (run_dir / "SCORING.json").write_text(RETAINED["scoring"].read_text())
    with pytest.raises(M.DesignError) as exc:
        M.close_retained_run(run_dir)
    assert "not the one this run recorded" in str(exc.value) or "horizon" in str(exc.value).lower()
