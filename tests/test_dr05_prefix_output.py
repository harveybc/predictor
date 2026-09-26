"""Rules for the frozen-prefix OUTPUT materialization: the cut, the five stages, the four proofs.

Every rule is a mutant that must be REJECTED. Nothing here fits a model, opens a reserve or writes to a
run root: the store rules run on a tiny store built in ``tmp_path``, and the graph rules build one small
prefix on CPU.

    CUDA_VISIBLE_DEVICES='' python -m pytest tests/test_dr05_prefix_output.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("_t_dr05",
                                                  REPO / "tools" / "df_dr05_prefix_output.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = _tool()


# --- the record rules ---------------------------------------------------------------------------------

def _stage(name: str) -> dict:
    return {"stage": name, "kind": T.STAGE_KINDS[name], "what": "x",
            "learned_state": {"digest": "d"}, "clock": {"state": T.VERIFIED},
            "rows_and_splits": {"train": {"split": "train"}}, "shape": {"in": [1], "out": [1]},
            "version": "v1"}


def _record() -> dict:
    return {
        "schema": "df_dr05_prefix_output.v1",
        "what_parity_of_inputs_is_not": T.WHAT_PARITY_OF_INPUTS_IS_NOT,
        "prefix_ends_at": T.PREFIX_ENDS_AT,
        "reserve": "NOT_OPENED",
        "stages": [_stage(n) for n in T.STAGE_KINDS],
        "proofs": {p: {"proof": p, "state": T.VERIFIED} for p in T.REQUIRED_PROOFS},
        "splits": {"train": {"n_origins": 1}, "validation": {"n_origins": 1},
                   "test": {"state": T.BY_DESIGN, "materialization": None}},
        "arrays": {"train": {"array": "a"}, "validation": {"array": "b"}},
        "what_this_does_not_deliver": ["a claim that this prefix is the right prefix"],
        "checks": {"a": True},
        "state": T.VERIFIED,
    }


def test_the_reference_record_is_admissible():
    assert T.validate(_record()) == []


@pytest.mark.parametrize("proof", list(T.REQUIRED_PROOFS))
def test_a_record_missing_any_of_the_four_proofs_is_rejected(proof):
    r = _record()
    del r["proofs"][proof]
    assert any(proof in m and "absent" in m for m in T.validate(r))


@pytest.mark.parametrize("stage", list(T.STAGE_KINDS))
def test_a_record_missing_any_of_the_five_stages_is_rejected(stage):
    r = _record()
    r["stages"] = [s for s in r["stages"] if s["stage"] != stage]
    assert any("five stages" in m for m in T.validate(r))


@pytest.mark.parametrize("field", ["learned_state", "clock", "rows_and_splits", "shape", "version"])
def test_a_stage_that_drops_its_state_clock_rows_shape_or_version_is_rejected(field):
    for stage in T.STAGE_KINDS:
        r = _record()
        for s in r["stages"]:
            if s["stage"] == stage:
                s.pop(field)
        assert any(field in m and stage in m for m in T.validate(r)), (stage, field)


def test_the_stages_must_be_in_prefix_order():
    r = _record()
    r["stages"] = list(reversed(r["stages"]))
    assert any("in order" in m for m in T.validate(r))


def test_a_stage_that_misdeclares_learned_for_configuration_is_rejected():
    r = _record()
    for s in r["stages"]:
        if s["stage"] == "fusion":
            s["kind"] = "LEARNED"
    assert any("wrong kind" in m for m in T.validate(r))
    r = _record()
    for s in r["stages"]:
        if s["stage"] == "detector":
            s["kind"] = "CONFIGURATION"
    assert any("wrong kind" in m for m in T.validate(r))


def test_the_reserve_may_not_be_reported_as_verified():
    r = _record()
    r["splits"]["test"] = {"state": T.VERIFIED, "materialization": True}
    assert any("refused by name" in m for m in T.validate(r))


def test_an_array_written_for_the_reserve_is_rejected():
    r = _record()
    r["arrays"]["test"] = {"array": "c"}
    assert any("reserve" in m for m in T.validate(r))


def test_a_record_that_does_not_say_the_reserve_stayed_shut_is_rejected():
    r = _record()
    r["reserve"] = "OPENED"
    assert any("was not opened" in m for m in T.validate(r))


def test_a_record_that_omits_what_parity_of_inputs_is_not_is_rejected():
    r = _record()
    r.pop("what_parity_of_inputs_is_not")
    assert any("INPUTS" in m for m in T.validate(r))


def test_a_record_that_omits_what_it_does_not_deliver_is_rejected():
    r = _record()
    r.pop("what_this_does_not_deliver")
    assert any("not a delivery" in m for m in T.validate(r))


def test_a_cut_past_the_fusion_is_rejected():
    r = _record()
    r["prefix_ends_at"] = "core_tcn1_conv"
    assert any("must end at" in m for m in T.validate(r))


def test_a_verified_record_may_not_carry_a_check_that_could_not_run():
    r = _record()
    r["checks"]["clock_verified"] = None
    assert any("could not run" in m for m in T.validate(r))


# --- the fresh-process verifier catches a tampered store ----------------------------------------------

def _tiny_store(tmp_path: Path) -> Path:
    store = tmp_path / "store"
    store.mkdir()
    Z = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    o = np.array([10, 11], dtype=np.int64)
    np.save(store / "prefix_output_train.npy", Z)
    np.save(store / "origins_train.npy", o)
    man = {"schema": "df_dr05_prefix_store.v1", "version": {"version": "dr05.prefix.test"},
           "arrays": {"train": {"array": "prefix_output_train.npy",
                                "origins": "origins_train.npy",
                                "shape": [2, 3, 4], "dtype": "float32",
                                "sha256": T.sha_file(store / "prefix_output_train.npy"),
                                "origins_sha256": T.sha_file(store / "origins_train.npy")}}}
    (store / "MANIFEST.json").write_text(json.dumps(man))
    return store


def test_an_untampered_store_verifies(tmp_path):
    out = T.verify_store(_tiny_store(tmp_path))
    assert out["all_digests_match"] is True


def test_one_flipped_float_in_the_store_is_caught(tmp_path):
    store = _tiny_store(tmp_path)
    Z = np.load(store / "prefix_output_train.npy")
    Z[1, 2, 3] = np.float32(Z[1, 2, 3] + 1e-3)
    np.save(store / "prefix_output_train.npy", Z)
    out = T.verify_store(store)
    assert out["all_digests_match"] is False
    assert out["per_split"]["train"]["file_sha256_matches_manifest"] is False


def test_a_reshaped_store_is_caught(tmp_path):
    store = _tiny_store(tmp_path)
    man = json.loads((store / "MANIFEST.json").read_text())
    man["arrays"]["train"]["shape"] = [2, 4, 3]
    (store / "MANIFEST.json").write_text(json.dumps(man))
    out = T.verify_store(store)
    assert out["per_split"]["train"]["shape_matches_manifest"] is False
    assert out["all_digests_match"] is False


def test_a_swapped_origins_array_is_caught(tmp_path):
    store = _tiny_store(tmp_path)
    np.save(store / "origins_train.npy", np.array([10, 12], dtype=np.int64))
    out = T.verify_store(store)
    assert out["per_split"]["train"]["origins_sha256_matches_manifest"] is False


# --- the reserve is never scheduled for materialization -----------------------------------------------

def test_an_absent_run_root_refuses_and_writes_nothing(tmp_path):
    rec = T.run(tmp_path, "R1_s1", tmp_path / "store")
    assert rec["state"] == T.UNCHECKABLE
    assert "arrays" not in rec and "proofs" not in rec
    assert rec["missing"]
    assert not (tmp_path / "store").exists()


# --- the graph rule: the cut is at the fusion ---------------------------------------------------------

def test_a_prefix_that_reaches_past_the_fusion_is_flagged():
    """Built small, on CPU, from the repository's own builder: no run root and no weights."""
    spec = importlib.util.spec_from_file_location("_t_e0", REPO / "tools" / "df_mod_e0.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    tf = E._tf()
    full = E.build_modular([0, 0, 1], 16, 3, fusion="sequence", seed=1)
    good = tf.keras.Model(full.input, full.get_layer(T.PREFIX_ENDS_AT).output)
    bad = tf.keras.Model(full.input, full.get_layer("core_conv").output)
    assert T.prefix_contains_nothing_downstream({"prefix": good})["clean"] is True
    leaked = T.prefix_contains_nothing_downstream({"prefix": bad})
    assert leaked["clean"] is False
    assert "core_conv" in leaked["downstream_layers_found_inside"]


def test_the_five_stage_kinds_are_fixed():
    assert T.STAGE_KINDS == {"preprocessing": "LEARNED", "groups": "CONSTANT_BY_DESIGN",
                             "detector": "LEARNED", "adapter": "LEARNED",
                             "fusion": "CONFIGURATION"}
    assert copy.copy(T.REQUIRED_PROOFS) == ("direct_against_cache", "fresh_process_reload",
                                            "mutation", "causality")
