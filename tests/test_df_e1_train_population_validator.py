"""A preparation may not hold origins its design does not declare.

Q2_CONTEXT v1 is the case this was written for. Its sealed design carries no `train_population`
key at all, while the block catalogue declares `COMMON_INTERSECTION` for Q2_CONTEXT, and its
retained preparation holds PER-ARM train origins (40 020 / 38 700 / 38 700 / 38 700 / 40 080).
Had a cell been fitted on those bytes the input contrast would have been confounded with train
volume. None was, so nothing published moves -- and these tests are what keep it that way: the
validator refuses the v1 bytes and accepts the 2026-09-26 bounded block's, which declared
`COMMON_INTERSECTION` and held it with 38 700 identical train origins across every arm.

The retained preparation itself is NOT rewritten. The refusal is computed from its bytes, at
`prepare()` and again at `load_data()`, so the contradiction can neither seal nor be fitted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "docs/audits/evidence"
STATE = Path.home() / ".local/state/crispdm-data-foundation"

#: the sealed v1 preparation (design 6d1aaecaf27c581c..., state BUDGET_LIMITED_BEFORE_ANY_OUTCOME)
V1 = EV / "d3_k5_20260917/RP66/blocks/e1_block_q2_context_v1"
#: the 2026-09-26 bounded block (design 47a270eec01f203c...), which honoured COMMON_INTERSECTION
V2 = EV / "E1_Q2_CONTEXT_BOUNDED_20260926"
FINDING = EV / "E1_Q2_CONTEXT_V1_ORIGIN_CONTRADICTION_20260926/FINDING.json"


def _load():
    name = "df_e1_block_train_population_subject"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "df_e1_block.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = _load()


def _bytes_of(root: Path) -> tuple[dict, dict]:
    return json.loads((root / "DESIGN.json").read_text()), json.loads((root / "BLOCK_DATA.json").read_text())


def _arrays(state: Path, design: dict):
    """The per-arm train-origin arrays, when the working state that holds the .npz is present."""
    npz = state / "BLOCK_DATA.npz"
    if not npz.is_file():
        return None
    with np.load(npz, allow_pickle=False) as z:
        return {a["arm"]: z[f"train_origins__{a['arm']}"] for a in design["arms"]
                if f"train_origins__{a['arm']}" in z.files}


# --- the two retained preparations -----------------------------------------------------------------

def test_the_sealed_q2_context_v1_preparation_is_refused_on_its_own_bytes():
    design, record = _bytes_of(V1)
    assert design["design_sha256"].startswith("6d1aaecaf27c581c")
    assert "train_population" not in design, "v1's sealed design carries no origin policy at all"
    held = {a["arm"]: record["counts_from_identities"][a["arm"]]["labels"] for a in design["arms"]}
    assert sorted(held.values()) == [38700, 38700, 38700, 40020, 40080], held

    with pytest.raises(SystemExit) as refusal:
        B.validate_train_population(design, record)
    message = str(refusal.value)
    assert "REFUSED" in message and "per-arm train origins" in message
    assert "confounded with train volume" in message
    for arm, count in held.items():
        assert f"{arm}={count}" in message, "the refusal must name the origins each arm holds"


def test_the_sealed_v1_arrays_are_refused_too_when_the_working_state_holds_them():
    design, record = _bytes_of(V1)
    origins = _arrays(STATE / "e1_block_q2_context_v1", design)
    if origins is None:
        pytest.skip("the v1 BLOCK_DATA.npz is not retained in this repository and the working state is absent here")
    assert B.train_population_report(design, record, origins=origins)["held_identical_origins"] is False
    with pytest.raises(SystemExit):
        B.validate_train_population(design, record, origins=origins)


def test_the_bounded_block_of_2026_09_26_is_accepted_on_its_own_bytes():
    design, record = _bytes_of(V2)
    assert design["design_sha256"].startswith("47a270eec01f203c")
    assert design["train_population"] == "COMMON_INTERSECTION"
    report = B.validate_train_population(design, record)
    assert report["held_identical_counts"] is True
    assert set(report["held_train_origins"].values()) == {38700}
    assert report["common_train_origins_recorded"] == 38700
    assert all(value is not None for value in report["before_intersection_recorded"].values())


def test_the_bounded_blocks_arms_hold_the_very_same_origin_ARRAYS():
    design, record = _bytes_of(V2)
    origins = _arrays(STATE / "e1_block_q2_context_bounded_v1", design)
    if origins is None:
        pytest.skip("the bounded BLOCK_DATA.npz is not retained in this repository and the working state is absent here")
    report = B.validate_train_population(design, record, origins=origins)
    assert report["held_identical_origins"] is True
    first = origins[design["arms"][0]["arm"]]
    assert first.size == 38700
    for arm, array in origins.items():
        assert np.array_equal(first, array), arm


def test_the_finding_against_the_v1_preparation_is_retained():
    """The record, not the repair: it must keep naming the preparation, the digest and the verdict."""
    finding = json.loads(FINDING.read_text())
    assert finding["label"] == "Q2V1-ORIGIN-POLICY"
    assert finding["v1"]["design_sha256"].startswith("6d1aaecaf27c581c")
    assert finding["v1"]["verdict"] == "REFUSED"
    assert finding["v2_for_contrast"]["verdict"] == "ACCEPTED"
    assert finding["published_numbers_affected"] == 0
    assert finding["usability"]["e1_block_q2_context_v1"] == \
        "UNUSABLE_FOR_ANY_COMPARISON_THAT_ASSUMES_COMMON_TRAIN_ORIGINS"


# --- every branch of the refusal, on synthetic records ----------------------------------------------

def _record(counts: dict, *, before: bool = True, common: int | None = None) -> dict:
    return {
        "counts_from_identities": {arm: {"labels": n, "distinct_windows": n} for arm, n in counts.items()},
        "feasibility": {arm: ({"train_admissible_before_intersection": n + 7} if before else {})
                        for arm, n in counts.items()},
        "binding_to_source": {} if common is None else {"common_train_origins": common},
    }


def _design(policy: str | None, arms: list[str]) -> dict:
    design = {"arms": [{"arm": a} for a in arms]}
    if policy is not None:
        design["train_population"] = policy
    return design


def test_common_intersection_with_differing_counts_is_refused():
    with pytest.raises(SystemExit, match="COMMON_INTERSECTION but the preparation holds per-arm"):
        B.validate_train_population(_design("COMMON_INTERSECTION", ["a", "b"]),
                                    _record({"a": 100, "b": 90}, common=90))


def test_common_intersection_with_equal_counts_of_DIFFERENT_origins_is_refused():
    origins = {"a": np.arange(10), "b": np.arange(10) + 1}
    with pytest.raises(SystemExit, match="the origin SETS are not identical"):
        B.validate_train_population(_design("COMMON_INTERSECTION", ["a", "b"]),
                                    _record({"a": 10, "b": 10}, common=10), origins=origins)


def test_common_intersection_without_the_pre_intersection_counts_is_refused():
    with pytest.raises(SystemExit, match="no evidence the intersection was applied"):
        B.validate_train_population(_design("COMMON_INTERSECTION", ["a", "b"]),
                                    _record({"a": 90, "b": 90}, before=False, common=90))


def test_common_intersection_without_a_recorded_common_count_is_refused():
    with pytest.raises(SystemExit, match="records no common_train_origins"):
        B.validate_train_population(_design("COMMON_INTERSECTION", ["a", "b"]), _record({"a": 90, "b": 90}))


def test_a_recorded_common_count_that_is_not_the_one_held_is_refused():
    with pytest.raises(SystemExit, match="is not the count the arms hold"):
        B.validate_train_population(_design("COMMON_INTERSECTION", ["a", "b"]),
                                    _record({"a": 90, "b": 90}, common=38700))


def test_an_origin_policy_outside_the_vocabulary_is_refused():
    with pytest.raises(SystemExit, match="unknown train_population"):
        B.validate_train_population(_design("WHATEVER_IS_ADMISSIBLE", ["a", "b"]),
                                    _record({"a": 90, "b": 90}, common=90))
    assert B.TRAIN_POPULATIONS == ("COMMON_INTERSECTION", "PER_ARM_ADMISSIBLE")


def test_a_preparation_that_records_no_held_count_for_an_arm_is_refused():
    record = _record({"a": 90, "b": 90}, common=90)
    del record["counts_from_identities"]["b"]
    with pytest.raises(SystemExit, match="records no held train-origin count"):
        B.validate_train_population(_design("COMMON_INTERSECTION", ["a", "b"]), record)


def test_per_arm_admissible_may_legitimately_differ_and_an_undeclared_policy_may_not():
    """The validator judges a preparation against the policy its design DECLARES, and nothing else."""
    differing = _record({"a": 100, "b": 90})
    B.validate_train_population(_design("PER_ARM_ADMISSIBLE", ["a", "b"]), differing)
    with pytest.raises(SystemExit, match="declares no train_population"):
        B.validate_train_population(_design(None, ["a", "b"]), differing)
    # undeclared but factually common: readable, and reported as undeclared rather than refused
    report = B.validate_train_population(_design(None, ["a", "b"]), _record({"a": 90, "b": 90}))
    assert report["declared_policy"] == "UNDECLARED" and report["held_identical_counts"] is True


def test_the_validator_guards_both_production_and_consumption():
    """prepare() may not seal a contradicting preparation, and load_data() may not serve one."""
    source = (ROOT / "tools/df_e1_block.py").read_text()
    prepare = source.split("def prepare(", 1)[1].split("\ndef ", 1)[0]
    load = source.split("def load_data(", 1)[1].split("\ndef ", 1)[0]
    assert "validate_train_population(design, rec" in prepare
    assert "validate_train_population(design, rec" in load
