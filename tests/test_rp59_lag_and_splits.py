"""RP59's two repairs as executable rules: the restated lag table, and materialization per split.

Written under the owner's grant of 2026-09-26 by Satoshi, successor technical lead. These are the
rules the two repairs must not lose, and they are deliberately adversarial about the two ways a
repair of this kind rots:

  the table loses a column or its supersession note
      A restatement that keeps only the corrected column has silently replaced the published
      evidence; one that keeps only the published column has not repaired anything; one that keeps
      both but drops the estimator names or the shrinkage has published two numbers a reader cannot
      tell apart. ``test_the_restated_table_must_carry_*`` fail on each of those, on the DOCUMENT as
      well as on the tool's output, and mutants of both are built here and watched failing.

  a split's check is skipped, or silently passes on absent bytes
      The failure mode that matters is not a wrong number, it is a check that never ran and reads
      clean. ``_verdict`` refuses any split whose check dictionary carries a None, the run root is a
      hard requirement for the split rules rather than a skip, and mutants with a missing root, a
      dropped split, an emptied check set and a None check are all built here and each must refuse.

Nothing here fits, trains or replays a model, contacts governance, or writes into a run root.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
DOC = REPO / "docs" / "audits" / "work_plan" / \
    "SATOSHI_RP59_LAG_TABLE_AND_MATERIALIZATION_2026_09_26.md"
EVIDENCE = REPO / "docs" / "audits" / "evidence" / "RP59_LAG_AND_SPLITS_20260926" / \
    "LAG_TABLE_AND_SPLIT_MATERIALIZATION.json"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name: str):
    """Load the tool WITHOUT registering it in ``sys.modules`` (the defect the RP49-RP64 audit
    repaired in itself: a shared module object let one battery pick up another's copy)."""
    spec = importlib.util.spec_from_file_location(f"_rp59t_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("df_rp59_lag_and_splits")
VERIFIED, REFUTED, UNCHECKABLE, BY_DESIGN = T.VERIFIED, T.REFUTED, T.UNCHECKABLE, T.BY_DESIGN
SPLITS = ("train", "validation", "pretrain_train", "pretrain_internal_validation", "test")

needs_root = pytest.mark.skipif(not (T.SUCCESSOR_ROOT / "DATA.npz").is_file(),
                                reason="the successor run root is not on this host")


@pytest.fixture(scope="module")
def report():
    if not (T.SUCCESSOR_ROOT / "DATA.npz").is_file():
        pytest.skip("the successor run root is not on this host")
    return T.run(T.SUCCESSOR_ROOT)


@pytest.fixture(scope="module")
def evidence():
    if not EVIDENCE.is_file():
        pytest.skip(f"{EVIDENCE.name} is not retained on this checkout")
    return json.loads(EVIDENCE.read_text())


@pytest.fixture(scope="module")
def document():
    assert DOC.is_file(), f"the restatement document is missing: {DOC}"
    return DOC.read_text()


# --- repair one: the table may lose neither column, nor its supersession note ------------------------

def test_the_restated_table_must_carry_both_estimator_columns_in_the_document(document):
    """Both columns, named, in the document itself -- not only in a JSON a reader will not open."""
    for needle in ("as published", "biased", "lag-truncated Pearson", "shrinkage"):
        assert needle.lower() in document.lower(), f"the restated table lost {needle!r}"
    for value in ("0.317186", "0.265593", "0.335157", "0.366864", "0.7492"):
        assert value in document, (f"the restated table lost the value {value}: both the "
                                  f"as-published and the bias-corrected column must be printed, "
                                  f"with the shrinkage that explains the difference")


def test_the_document_must_state_what_it_supersedes_and_why(document):
    low = document.lower()
    assert "supersede" in low, "the restatement does not say what it supersedes"
    assert "DATA_TARGET_PREPROCESSING_AUDIT.json" in document, \
        "the restatement does not name the artifact whose block it supersedes"
    assert "RP57_RP64_RETURN_2026_09_21" in document, \
        "the restatement does not name the return whose printed list it supersedes"
    assert "nothing" in low and "withdraw" in low, \
        "the restatement must say which published values are withdrawn (none) rather than leave it open"


def test_the_document_must_state_which_reading_survives_and_which_reverses(document):
    low = document.lower()
    assert "survives" in low, "the restatement does not say which reading survives"
    assert "revers" in low, "the restatement does not say which reading reverses"
    assert "0.335157 < 0.403283" in document or "0.335 < 0.403" in document, \
        "the surviving reading must be shown as the inequality that carries it"


@needs_root
def test_both_columns_re_derive_and_only_the_order_flips(report):
    t = report["lag_table"]
    assert t["state"] == VERIFIED, t
    assert t["as_published_re_derives"], "the as-published column no longer re-derives"
    assert t["train_rows_match_the_published_support"], t
    assert t["ordering_by_strength"]["the_order_changes"], \
        "the whole finding is that the order flips; it no longer does"
    assert t["readings"]["survives"]["state"] == VERIFIED
    assert t["readings"]["reverses"]["state"] == REFUTED
    assert t["bias_corrected"]["10080"] > t["bias_corrected"]["1440"], \
        "bias-corrected, the week must stand ABOVE the day"
    assert t["as_published"]["10080"] < t["as_published"]["1440"], \
        "as published, the week must stand BELOW the day"


@needs_root
def test_every_lag_row_carries_both_columns_and_its_shrinkage(report):
    rows = report["lag_table"]["rows"]
    assert [r["lag_minutes"] for r in rows] == list(T.LAGS)
    for r in rows:
        for field in ("as_published_biased_acf", "bias_corrected_lag_truncated_pearson",
                      "shrinkage_n_minus_k_over_n"):
            assert isinstance(r[field], float), f"lag {r['lag_minutes']} lost {field}"
    weekly = next(r for r in rows if r["lag_minutes"] == 10080)
    assert abs(weekly["shrinkage_n_minus_k_over_n"] - 0.7492) < 5e-5, \
        "the weekly shrinkage factor is the reason the order flips; it must be printed"


@needs_root
def test_the_biased_column_is_the_tool_that_published_it_and_not_a_third_implementation(report):
    """The as-published column must come from ``df_e1_data_audit.autocorrelation`` itself."""
    audit = _load("df_e1_data_audit")
    D = np.load(T.SUCCESSOR_ROOT / "DATA.npz")
    data = {k: D[k] for k in D.files}
    theirs = audit.autocorrelation(data, lags=T.LAGS)["autocorrelation_by_lag_minutes"]
    mine = report["lag_table"]["recomputed_as_published"]
    assert set(mine) == {str(k) for k in T.LAGS}
    for k, v in theirs.items():
        assert abs(float(v) - mine[str(k)]) <= 0.0, \
            "the restatement's as-published column is not the estimator that produced it"


def test_a_table_that_drops_either_column_or_the_note_is_rejected(evidence):
    """The mutants: one column missing, then the supersession note missing. Each must be detectable."""
    good = evidence["lag_table"]
    for field in ("as_published", "bias_corrected", "shrinkage_n_minus_k_over_n", "supersession",
                  "estimators", "readings"):
        mutant = copy.deepcopy(good)
        mutant.pop(field)
        assert not _table_is_complete(mutant), \
            f"a restated table missing {field!r} was accepted; the rule does not bite"
    assert _table_is_complete(good), "the retained restatement fails its own completeness rule"
    half = copy.deepcopy(good)
    half["rows"] = [{k: v for k, v in r.items() if k != "bias_corrected_lag_truncated_pearson"}
                    for r in half["rows"]]
    assert not _table_is_complete(half), "a table whose ROWS lost a column was accepted"


def _table_is_complete(t: dict) -> bool:
    """A restated table is complete only with both columns, the shrinkage, the estimator names,
    the two readings and the supersession note -- in the table and in every row."""
    for field in ("as_published", "bias_corrected", "shrinkage_n_minus_k_over_n", "estimators",
                  "readings", "supersession", "rows"):
        if field not in t:
            return False
    if set(t["estimators"]) != {"as_published", "bias_corrected"}:
        return False
    if set(t["readings"]) != {"survives", "reverses"}:
        return False
    if not all(k in t["supersession"] for k in ("supersedes", "why", "what_is_withdrawn")):
        return False
    return all(all(f in r for f in ("as_published_biased_acf",
                                    "bias_corrected_lag_truncated_pearson",
                                    "shrinkage_n_minus_k_over_n")) for r in t["rows"])


def test_the_retained_evidence_carries_both_columns_with_the_published_values(evidence):
    t = evidence["lag_table"]
    assert _table_is_complete(t)
    assert abs(t["as_published"]["1440"] - 0.31718580057541035) < 1e-12
    assert abs(t["as_published"]["10080"] - 0.2655930481297192) < 1e-12
    assert abs(t["bias_corrected"]["1440"] - 0.3351571550096535) < 1e-10
    assert abs(t["bias_corrected"]["10080"] - 0.36686408713442276) < 1e-10


# --- repair two: a split's check may not be skipped, and may not pass on absent bytes ---------------

def test_a_check_that_could_not_run_refuses_and_never_passes():
    """``_verdict`` is the rule: a None check refuses the split. This is the anti-silence rule."""
    assert T._verdict({"a": True, "b": True}) == VERIFIED
    assert T._verdict({"a": True, "b": False}) == REFUTED
    assert T._verdict({"a": True, "b": None}) == UNCHECKABLE, \
        "a check that could not run was folded into a pass"
    assert T._verdict({"a": None}) == UNCHECKABLE
    assert T._verdict({}) == UNCHECKABLE, "a split with NO checks must refuse, not pass vacuously"


def test_an_absent_run_root_refuses_every_split_by_name(tmp_path):
    """On absent bytes the tool must name every split as unverifiable, not omit them."""
    out = T.materialization(tmp_path / "nothing")
    assert out["state"] == UNCHECKABLE
    assert {s["split"] for s in out["splits"]} == set(SPLITS), \
        "a split disappeared from the record instead of being refused by name"
    for s in out["splits"]:
        assert s["state"] == UNCHECKABLE
        assert any(v is None for v in s["checks"].values())
    lag = T.lag_table(tmp_path / "nothing")
    assert lag["state"] == UNCHECKABLE and "missing" in lag, \
        "the lag table repeated published values it could not re-derive"


def test_a_report_missing_a_split_or_a_check_is_rejected(evidence):
    """The mutants for repair two: a dropped split, an emptied check set, a None check."""
    good = evidence["materialization"]
    assert _every_split_checked(good), "the retained per-split verification fails its own rule"
    for split in SPLITS:
        mutant = copy.deepcopy(good)
        mutant["splits"] = [s for s in mutant["splits"] if s["split"] != split]
        assert not _every_split_checked(mutant), \
            f"a report with no entry for {split!r} was accepted; a skipped split is invisible"
    emptied = copy.deepcopy(good)
    emptied["splits"][0]["checks"] = {}
    assert not _every_split_checked(emptied), "a split with no checks at all was accepted"
    noned = copy.deepcopy(good)
    first = next(s for s in noned["splits"] if s["state"] == VERIFIED)
    first["checks"][next(iter(first["checks"]))] = None
    assert not _every_split_checked(noned), \
        "a VERIFIED split carrying a check that never ran was accepted"


def _every_split_checked(m: dict) -> bool:
    """Every split present, every one with a state, and no split VERIFIED on a check that is None
    or on an empty check set. A refusal is acceptable; silence is not."""
    by_name = {s["split"]: s for s in m.get("splits", [])}
    if set(by_name) != set(SPLITS):
        return False
    for s in by_name.values():
        checks = s.get("checks")
        if checks is None:
            return False
        if s["state"] == VERIFIED and (not checks or any(v is None for v in checks.values())):
            return False
        if s["state"] not in (VERIFIED, REFUTED, UNCHECKABLE, BY_DESIGN):
            return False
    return True


@needs_root
def test_every_split_is_verified_or_refused_by_name_from_the_bytes(report):
    m = report["materialization"]
    assert _every_split_checked(m)
    states = m["per_split_state"]
    assert states["train"] == VERIFIED and states["validation"] == VERIFIED
    assert states["pretrain_train"] == VERIFIED
    assert states["pretrain_internal_validation"] == VERIFIED
    assert states["test"] == BY_DESIGN, \
        "the reserve must be REFUSED by name, never reported as a verified materialization"
    assert m["refusals"]["test"].startswith("REFUSED")


@needs_root
def test_the_declared_origin_ranges_re_derive_rather_than_being_quoted(report):
    for s in report["materialization"]["splits"]:
        if s["split"] not in ("train", "validation"):
            continue
        d = s["declared"]
        assert d["origins_declared"] == d["enumerator_origins"], \
            f"{s['split']}: the declared origin count does not re-derive from the design's own rule"
        assert s["checks"]["the_materialized_origins_are_exactly_the_admissible_declared_origins"]
        assert s["checks"]["the_withdrawal_arithmetic_re_derives"]
        assert s["checks"]["the_materialized_target_rows_ARE_the_panel_rows_this_split_declares"]


@needs_root
def test_no_splits_rows_appear_in_another_unless_the_nesting_is_declared(report):
    ex = report["materialization"]["exclusivity"]
    assert ex["state"] == VERIFIED
    assert ex["pairs"]["train|validation"]["shared_origins"] == 0
    assert ex["pairs"]["train|validation"]["shared_touched_rows"] == 0
    for key, pair in ex["pairs"].items():
        if pair["nesting_declared"]:
            assert "train" in key, "only the pre-training splits nest, and only inside `train`"
            continue
        assert pair["origins_disjoint"] and pair["rows_disjoint"], key
    for key in ("pretrain_internal_validation|validation", "pretrain_train|validation"):
        assert ex["pairs"][key]["shared_touched_rows"] == 0, \
            "pre-training reached a row of the supervised validation split"


@needs_root
def test_the_declared_purge_boundary_is_the_observed_one(report):
    p = report["materialization"]["purge"]
    assert p["state"] == VERIFIED
    assert p["declared_purge_task"] == p["declared_purge_dev_subpartition"] == 120
    assert p["observed_origin_gap"] == 121
    assert p["observed_untouched_rows_between_the_splits"] >= 0
    assert p["train_rows_touched_inclusive"][1] < p["validation_rows_touched_inclusive"][0]


@needs_root
def test_the_reserve_absence_is_verified_not_assumed(report):
    t = next(s for s in report["materialization"]["splits"] if s["split"] == "test")
    assert t["absence_state"] == VERIFIED
    assert t["absence_verified"]["the_consumed_slice_ends_before_the_declared_test_block"]
    assert t["absence_verified"]["no_retained_array_carries_a_test_population"]
    assert t["absence_verified"]["every_retained_cell_declares_NO_TEST_ACCESS"]
    assert t["materialized"]["cells_scanned"] > 0, \
        "the absence check scanned nothing and would have passed on an empty run root"
    assert t["checks"]["materialization_of_this_split"] is None, \
        "a split with nothing materialized must not carry a passing materialization check"


@needs_root
def test_the_within_slice_boundary_is_the_family_s_own_split_edge(report):
    b = report["materialization"]["declared_blocks"]
    assert b["the_within_slice_boundary_IS_the_family_train_validation_edge"], \
        "the dev split boundary no longer coincides with the family's train/validation edge"
    assert b["family_split_rule"]["edges_in_panel_rows"]["train"][1] == 1452681
    assert report["materialization"]["identity"]["re_derives"]
    assert report["materialization"]["delivered_bytes"]["path_digest_matches_the_delivery"]


@needs_root
def test_the_sibling_run_root_consumed_the_same_prepared_bytes(report):
    sib = report["sibling_root"]
    if sib["state"] == UNCHECKABLE:
        pytest.skip("the phase-1 run root is not on this host")
    assert sib["identical_to_the_audited_root"], \
        "the phase-1 root's prepared bytes differ; the per-split verdicts do not carry to it"
