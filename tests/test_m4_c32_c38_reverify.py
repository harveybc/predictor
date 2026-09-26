"""The M4 C32-C38 re-verification receipt must not be vacuous.

`docs/audits/evidence/M4_C32_C38_REVERIFY_20260926/reverify_m4_c32_c38.py`
re-derives the frozen M4 CONFIRMATION plan from the `agent-multi` object
store and asserts it is still unexecuted.  A receipt that passes no matter
what is worse than no receipt, so these tests do two things:

1. pin the receipt's own constants against the words of the order
   (agent-multi@889320ee, "P2 - M4 C32-C38 confirmation preparation") and
   Musashi's accepting audit, so a silently edited pin fails here;
2. mutate each pin in turn and require the receipt to go RED, so the
   checks are proved load-bearing.

The receipt reads git only; nothing is fitted, loaded or scored.  When the
`agent-multi` repository is not present on the host the mutation tests skip
and the constant tests still run.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (REPO_ROOT / "docs/audits/evidence/M4_C32_C38_REVERIFY_20260926"
           / "reverify_m4_c32_c38.py")
AGENT_MULTI = Path("/home/harveybc/Documents/GitHub/agent-multi")


def _load():
    spec = importlib.util.spec_from_file_location(
        "m4_c32_c38_receipt", RECEIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def receipt():
    assert RECEIPT.is_file(), f"the receipt script is missing: {RECEIPT}"
    mod = _load()
    mod._results.clear()
    return mod


def _agent_multi_available() -> bool:
    return (AGENT_MULTI / ".git").exists() or AGENT_MULTI.joinpath(
        "HEAD").exists()


needs_repo = pytest.mark.skipif(
    not _agent_multi_available(),
    reason="the agent-multi repository is not present on this host")


# --------------------------------------------------------------- pins ----

def test_the_order_pinned_identities_are_the_ones_the_order_names(receipt):
    """C32 lists four identities. The receipt must carry exactly those."""
    assert receipt.REVIEWED_TIP == (
        "5e7a8fd430c8231a049baf03f00e720ba24ec994")
    by_name = {Path(p).name: v for p, v in receipt.SELF_ID.items()}
    assert by_name["M4_SEALED_DESIGN_V5_2026_09_09.json"] == (
        "design_sha256",
        "d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9")
    assert by_name["M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_2026_09_09.json"] == (
        "amendment_sha256",
        "43e0804e1e6e583b10ddbe46b7d4cd752838b0473ccbc6496f0e458c49aedd4b")
    assert by_name[
        "M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_GOVERNING_2026_09_09.json"
    ] == (
        "record_sha256",
        "b35b6fd969aa162047bdfb55b8f9fcce01aa76864c388d29a1c36642ab051ade")


def test_the_order_facts_are_the_ones_the_order_states(receipt):
    """C32: "21/28 eligible slots, two incomplete generators, zero
    calibration-incomplete cells, and M2 gain -0.41982887"."""
    assert receipt.ORDER_FACTS == {
        "eligible_slots": 21,
        "total_slots": 28,
        "ineligible_slots": 7,
        "incomplete_generators": 2,
        "calibration_incomplete_cells": 0,
        "m2_gain": -0.41982887,
    }
    assert (receipt.ORDER_FACTS["eligible_slots"]
            + receipt.ORDER_FACTS["ineligible_slots"]
            == receipt.ORDER_FACTS["total_slots"])


def test_the_c33_policy_numbers_are_the_ones_the_order_freezes(receipt):
    """C33: ">=12 of 16 ... 48 CONFIRMATION generators per eligible slot
    ... the existing 20-percent attrition allowance and minimum complete
    count ... M2 as DOES_NOT_ADVANCE_FROM_CALIBRATION"."""
    c = receipt.C33
    assert (c["min_learnable"], c["of_generators"],
            c["max_numerically_invalid"]) == (12, 16, 0)
    assert c["generators_per_eligible_slot"] == 48
    assert c["attrition_allowance"] == 0.20
    assert c["nested_seeds"] == 3
    assert c["selection_rule_label"] == "CALIBRATION_DERIVED_AND_REVIEWED"
    assert c["classification"] == "SCIENTIFIC_ANALYSIS_FREEZE"
    assert c["m2_status"] == "DOES_NOT_ADVANCE_FROM_CALIBRATION"


def test_the_attrition_floor_follows_from_the_formula_not_from_a_field(
        receipt):
    """The floor is max(3, ceil(planned * (1 - allowance))), never a
    number someone typed."""
    c = receipt.C33
    planned = c["generators_per_eligible_slot"]
    floor = max(3, math.ceil(planned * (1 - c["attrition_allowance"])))
    assert floor == 39
    assert c["min_complete_required"] == floor


def test_the_census_arithmetic_closes(receipt):
    """3024 units = 21 eligible slots x 48 generators x 3 nested seeds."""
    f, c = receipt.ORDER_FACTS, receipt.C33
    assert (f["eligible_slots"] * c["generators_per_eligible_slot"]
            * c["nested_seeds"]) == 3024


# ------------------------------------------------- the canonical digest --

def test_the_self_identity_rule_ignores_only_the_self_key(receipt):
    doc = {"a": 1, "b": [2, 3], "self": "whatever"}
    assert receipt.selfsha(doc, "self") == receipt.selfsha(
        {**doc, "self": "something else"}, "self")


def test_the_self_identity_rule_is_sensitive_to_every_other_field(receipt):
    doc = {"a": 1, "b": [2, 3], "self": "x"}
    base = receipt.selfsha(doc, "self")
    assert receipt.selfsha({**doc, "a": 2}, "self") != base
    assert receipt.selfsha({**doc, "b": [3, 2]}, "self") != base
    assert receipt.selfsha({**doc, "c": None}, "self") != base


# ------------------------------------------------------ not vacuous ------

@needs_repo
def test_the_receipt_passes_against_the_real_object_store(receipt):
    assert receipt.main(["--repo", str(AGENT_MULTI)]) == 0
    assert receipt._results, "the receipt ran no checks"
    assert all(ok for ok, _ in receipt._results)


@needs_repo
@pytest.mark.parametrize("pin", [
    "eligible_slots", "total_slots", "ineligible_slots",
    "incomplete_generators", "calibration_incomplete_cells", "m2_gain",
])
def test_a_mutated_order_fact_turns_the_receipt_red(pin):
    mod = _load()
    mod._results.clear()
    old = mod.ORDER_FACTS[pin]
    mod.ORDER_FACTS[pin] = (old + 1 if isinstance(old, int)
                            else old + 0.001)
    assert mod.main(["--repo", str(AGENT_MULTI)]) == 1, (
        f"mutating ORDER_FACTS[{pin!r}] did not fail the receipt -- "
        "that check is vacuous")


@needs_repo
@pytest.mark.parametrize("pin", [
    "min_learnable", "of_generators", "generators_per_eligible_slot",
    "min_complete_required", "nested_seeds", "selection_rule_label",
    "classification", "m2_status",
])
def test_a_mutated_c33_policy_number_turns_the_receipt_red(pin):
    mod = _load()
    mod._results.clear()
    old = mod.C33[pin]
    mod.C33[pin] = old + 1 if isinstance(old, int) else f"{old}_MUTATED"
    assert mod.main(["--repo", str(AGENT_MULTI)]) == 1, (
        f"mutating C33[{pin!r}] did not fail the receipt -- "
        "that check is vacuous")


@needs_repo
def test_a_mutated_self_identity_pin_turns_the_receipt_red():
    mod = _load()
    mod._results.clear()
    key, want = mod.SELF_ID[mod.SUCCESSOR_PATH]
    mod.SELF_ID[mod.SUCCESSOR_PATH] = (key, "0" * 64)
    assert mod.main(["--repo", str(AGENT_MULTI)]) == 1


@needs_repo
def test_a_mutated_deliverable_digest_turns_the_receipt_red():
    mod = _load()
    mod._results.clear()
    mod.DELIVERABLE_SHA["tools/m4_confirmation_protocol.py"] = "0" * 64
    assert mod.main(["--repo", str(AGENT_MULTI)]) == 1


@needs_repo
def test_a_missing_deliverable_is_a_hard_failure_not_a_silent_pass():
    mod = _load()
    mod._results.clear()
    mod.DELIVERABLE_SHA["tools/does_not_exist_anywhere.py"] = "0" * 64
    with pytest.raises(mod.Failed):
        mod.main(["--repo", str(AGENT_MULTI)])
