"""The D2 totals: what the records say, and the grain that makes the question answerable.

Musashi's B1 (2026-09-14). The declared rules are both arithmetical (the tool counts what a
small fixture contains) and factual (the conserved records give 47 + 6, 39, and five plus
two), so a future edit of the prose cannot drift from the records again.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

TOOLS = Path(os.environ.get("D2_TOOLS_DIR") or (Path(__file__).resolve().parents[1] / "tools"))
STATE = Path(os.environ.get("D2_STATE_DIR")
             or Path.home() / ".local/state/crispdm-data-foundation")
PUBLISHED = STATE / "d2_fresh_c174_v1_collected_tables" / "df_fact_d2_decision.jsonl"
SUCCESSOR = STATE / "d2_r3_review_v1" / "DECISIONS_SUCCESSOR.jsonl"

spec = importlib.util.spec_from_file_location("df_d2_summary_under_test", TOOLS / "df_d2_summary.py")
summary_module = importlib.util.module_from_spec(spec)
sys.modules["df_d2_summary_under_test"] = summary_module
spec.loader.exec_module(summary_module)

REGIME_A = {"declared_snr_db": "5", "family": "motif", "length": 2048,
            "missingness": "none", "perturbation": "white"}
REGIME_B = dict(REGIME_A, family="steps")


def row(**over):
    base = {"subject_kind": "OPERATOR", "subject": "an_operator", "operator_params": {},
            "regime": dict(REGIME_A), "arm_role": "CANDIDATE", "decision": "LAB_CALIBRATED",
            "n_seeds_valid": 30}
    base.update(over)
    return base


def write(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_rows_regimes_and_methods_are_counted_separately(tmp_path):
    rows = [row(), row(regime=dict(REGIME_B)), row(subject="another", operator_params={"k": 1})]
    published = summary_module.read(write(tmp_path, "p.jsonl", rows))
    totals = summary_module.totals(published)["CANDIDATE"]["OPERATOR"]["LAB_CALIBRATED"]
    assert totals == {"rows": 3, "regimes": 2, "methods": 2}


def test_the_same_grain_twice_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="repeats a decision grain"):
        summary_module.read(write(tmp_path, "dup.jsonl", [row(), row()]))


def test_a_pass_that_moves_is_lost_and_one_that_arrives_is_gained(tmp_path):
    published = summary_module.read(write(tmp_path, "p.jsonl", [
        row(), row(subject="b", decision="LAB_REJECTED")]))
    successor = summary_module.read(write(tmp_path, "s.jsonl", [
        row(decision="NOT_IDENTIFIABLE"), row(subject="b", decision="REGIME_LIMITED")]))
    moved = summary_module.transitions(published, successor)
    assert len(moved["passes_lost"]) == 1 and moved["passes_lost"][0]["new"] == "NOT_IDENTIFIABLE"
    assert len(moved["passes_gained"]) == 1
    assert moved["changed_rows"] == 2


def test_operator_params_are_part_of_the_grain(tmp_path):
    """Two arms of the same operator in the same regime are two decisions, not one."""
    published = summary_module.read(write(tmp_path, "p.jsonl", [
        row(operator_params={"w": 1}), row(operator_params={"w": 2})]))
    successor = summary_module.read(write(tmp_path, "s.jsonl", [
        row(operator_params={"w": 1}), row(operator_params={"w": 2}, decision="NOT_IDENTIFIABLE")]))
    moved = summary_module.transitions(published, successor)
    assert moved["compared_grains"] == 2
    assert len(moved["passes_lost"]) == 1


@pytest.mark.skipif(not (PUBLISHED.is_file() and SUCCESSOR.is_file()),
                    reason="the conserved D2 records are not on this host")
def test_the_conserved_records_give_the_corrected_totals():
    summary = summary_module.summarise(summary_module.read(PUBLISHED), summary_module.read(SUCCESSOR))
    operators = summary["operator_passes"]
    assert operators["published"]["CANDIDATE"] == {"LAB_CALIBRATED": 51, "REGIME_LIMITED": 7}
    assert operators["successor"]["CANDIDATE"] == {"LAB_CALIBRATED": 47, "REGIME_LIMITED": 6}
    assert summary["snr_passes"]["published"]["SNR_CALIBRATED_FOR_REGIME"] == 39
    assert summary["snr_passes"]["successor"]["SNR_CALIBRATED_FOR_REGIME"] == 39
    moved = summary["transitions"]
    assert moved["passes_lost_by_arm"] == {"CANDIDATE": 5, "IDENTITY_RAW_CONTROL": 2}
    assert moved["passes_gained"] == []
    assert moved["changed_rows"] == 138
    assert moved["only_in_published"] == [] and moved["only_in_successor"] == []


@pytest.mark.skipif(not (PUBLISHED.is_file() and SUCCESSOR.is_file()),
                    reason="the conserved D2 records are not on this host")
def test_a_coarser_grain_would_have_invented_losses():
    """Why the grain matters: on (subject, regime) the same records appear to lose 36 passes."""
    published = summary_module.read(PUBLISHED)
    successor = summary_module.read(SUCCESSOR)
    coarse_old, coarse_new = {}, {}
    for rows, target in ((published, coarse_old), (successor, coarse_new)):
        for key, value in rows.items():
            target.setdefault((key[0], key[1], key[3]), []).append(value["decision"])
    inflated = 0
    for key, olds in coarse_old.items():
        for old in olds:
            for new in coarse_new.get(key, []):
                if old in summary_module.PASSES and new not in summary_module.PASSES:
                    inflated += 1
    assert inflated > 7, "the coarse join is the one that invented extra losses"
