"""Behavioral contract for weekly branch-pretraining regimes."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from predictor_plugins.modular_temporal import build_modular, save_donor
from predictor_plugins.modular_temporal.common import weights_hash
from tools import i6d_weekly_walk_forward as parent
from tools import i7_weekly_branch_regimes as regimes


def _config():
    members = [f"feature_{index:02d}" for index in range(20)]
    feature_names = [channel for member in members
                     for channel in (f"{member}__value", f"{member}__observed")]
    return {
        "schema": "predictor.i6d.matched_control.v1",
        "experiment_id": "i7-regime-fixture",
        "feature_names": feature_names,
        "feature_groups": [
            {"name": member,
             "channels": [f"{member}__value", f"{member}__observed"]}
            for member in members
        ],
        "window": 24,
        "sample_hours": 1,
        "horizons": [1],
        "target_names": ["Y_s_1h"],
        "seed": 17,
        "branch": {
            "dense": {"hidden_units": [4], "latent_units": 2,
                      "activation": "gelu", "use_bias": True},
            "conv": {"channels": 2, "kernel_size": 3},
        },
        "core": {
            "vector_width": 12,
            "dense_hidden_units": [8],
            "conv": {"d_model": 8, "heads": 1, "blocks": 1, "ff_dim": 8,
                     "dropout": 0.0, "stage_channels": [6, 4, 2],
                     "time_factors": [2, 2, 1], "kernel_size": 3},
            "conv_output_steps": 6,
            "conv_output_channels": 2,
        },
        "head": {"initializer_seed": 991},
        "fit": {"max_epochs": 2, "patience": 1, "batch_size": 4,
                "learning_rate": 0.001, "weight_decay": 0.0,
                "loss": "huber", "huber_delta": 1.0, "min_delta": 0.0,
                "max_updates": 10, "max_seconds": 60.0,
                "monitor": "validation_loss", "monitor_every": 1},
    }


def _one_week_parent():
    design = parent.build_weekly_design(
        _config(), 2024, target_transform="ROBUST_Z_FIT", control_kind="BRANCH_ONLY"
    )
    design["weeks"] = design["weeks"][:1]
    design["expected_cells"] = 2
    return parent._seal(design, "design_sha256")


@pytest.fixture(scope="module")
def campaign_material(tmp_path_factory):
    root = tmp_path_factory.mktemp("i7-donors")
    parent_design = _one_week_parent()
    base = parent._branch_only_base_config(parent_design["config"])
    bundle = build_modular(base)
    week_entries = {}
    provenance = {
        "conditioning_contract": "OPERATIONAL",
        "learned_corpus": {
            "kind": "TRAIN_ONLY", "dataset_id": "fixture",
            "data_sha256": "a" * 64, "support": "four-year rolling TRAIN fixture",
        },
        "reconstruction": {"state": "MEASURED", "mae_z": 0.1, "mse_z": 0.02},
    }
    for ordinal, member in enumerate(parent_design["members"]):
        name = f"branch_{ordinal:03d}"
        path = root / f"{name}.keras"
        saved = save_donor(
            bundle.branch_models[name], path, bundle.donor_manifest("branch", name),
            provenance=provenance,
        )
        week_entries[member] = {
            "week_ordinal": 0,
            "member": member,
            "parent_design_sha256": parent_design["design_sha256"],
            "path": str(path),
            "donor_manifest_sha256": saved["manifest_sha256"],
            "model_sha256": saved["model_sha256"],
            "weights_sha256": saved["weights_sha256"],
        }
    donor_index = regimes.seal_donor_index({
        "schema": regimes.DONOR_INDEX_SCHEMA,
        "parent_design_sha256": parent_design["design_sha256"],
        "members": parent_design["members"],
        "weeks": {"0": week_entries},
    })
    design = regimes.build_design(parent_design, donor_index)
    return root, parent_design, donor_index, design


def _hashes(bundle):
    return {
        "branches": {name: weights_hash(model)
                     for name, model in bundle.branch_models.items()},
        "core": weights_hash(bundle.core_model),
        "head": weights_hash(bundle.forecast_model.get_layer("forecast_head")),
    }


def test_design_consumes_verified_parent_and_exact_twenty_member_week_index(campaign_material):
    _, parent_design, donor_index, design = campaign_material

    assert design["parent_design_sha256"] == parent_design["design_sha256"]
    assert design["donor_index_sha256"] == donor_index["index_sha256"]
    assert design["arms"] == ["R0", "R1_B", "R2_B"]
    assert len(design["members"]) == 20
    assert design["expected_cells"] == 3
    assert design["target_transform"] == "ROBUST_Z_FIT"
    assert design["test_paths"] is None
    assert design["test_read"] is False


def test_r1_and_r2_load_identical_donors_and_all_arms_share_core_head_initialization(
        campaign_material):
    _, _, _, design = campaign_material
    r0 = regimes.build_arm_bundle(design, "R0", 0, 17)
    r1 = regimes.build_arm_bundle(design, "R1_B", 0, 17)
    r2 = regimes.build_arm_bundle(design, "R2_B", 0, 17)
    h0, h1, h2 = _hashes(r0), _hashes(r1), _hashes(r2)

    assert h1["branches"] == h2["branches"]
    assert h0["core"] == h1["core"] == h2["core"]
    assert h0["head"] == h1["head"] == h2["head"]
    assert not any(model.trainable for model in r1.branch_models.values())
    assert all(model.trainable for model in r2.branch_models.values())


def test_one_optimizer_update_keeps_r1_branches_frozen_and_moves_r2(campaign_material):
    _, _, _, design = campaign_material
    rng = np.random.default_rng(9)
    x = rng.normal(size=(4, 24, 40)).astype("float32")
    y = rng.normal(size=(4, 1, 1)).astype("float32")

    r1 = regimes.build_arm_bundle(design, "R1_B", 0, 17)
    r2 = regimes.build_arm_bundle(design, "R2_B", 0, 17)
    frozen = regimes.one_optimizer_update_evidence(r1, x, y)
    trainable = regimes.one_optimizer_update_evidence(r2, x, y)

    assert frozen["observed_updates"] == trainable["observed_updates"] == 1
    assert frozen["branch_changed_count"] == 0
    assert trainable["branch_changed_count"] > 0
    assert frozen["trainable_branch_variables"] == 0
    assert trainable["trainable_branch_variables"] > 0


def test_missing_or_wrong_week_donor_refuses_before_fit(monkeypatch, campaign_material):
    _, _, donor_index, _ = campaign_material
    damaged = copy.deepcopy(donor_index)
    victim = damaged["weeks"]["0"][damaged["members"][0]]
    victim["path"] = str(Path(victim["path"]).with_name("missing.keras"))
    damaged = regimes.seal_donor_index(
        {key: value for key, value in damaged.items() if key != "index_sha256"}
    )
    design = regimes.build_design(_one_week_parent(), damaged)
    called = []
    monkeypatch.setattr(regimes.weekly_contract, "run_task",
                        lambda *args, **kwargs: called.append(True))

    with pytest.raises(ValueError, match="DONOR_MISSING"):
        regimes.run_cell(design, "R1_B", 0, 17, object())
    assert called == []

    wrong_bytes = copy.deepcopy(donor_index)
    wrong_bytes["weeks"]["0"][wrong_bytes["members"][0]]["model_sha256"] = "b" * 64
    wrong_bytes = regimes.seal_donor_index(
        {key: value for key, value in wrong_bytes.items() if key != "index_sha256"}
    )
    design = regimes.build_design(_one_week_parent(), wrong_bytes)
    with pytest.raises(ValueError, match="DONOR_BYTES_OR_MANIFEST_MISMATCH"):
        regimes.run_cell(design, "R2_B", 0, 17, object())
    assert called == []

    mismatched = copy.deepcopy(donor_index)
    entry = mismatched["weeks"]["0"][mismatched["members"][0]]
    entry["week_ordinal"] = 1
    mismatched = regimes.seal_donor_index(
        {key: value for key, value in mismatched.items() if key != "index_sha256"}
    )
    with pytest.raises(ValueError, match="DONOR_IDENTITY_MISMATCH"):
        regimes.build_design(_one_week_parent(), mismatched)


def _result(design, week, arm, *, rows="same", branch_changed=None):
    before_value = "random" if arm == "R0" else "donor"
    branch_before = {f"branch_{index:03d}": before_value for index in range(20)}
    branch_after = dict(branch_before)
    changed = (0 if arm == "R1_B" else 1) if branch_changed is None else branch_changed
    if changed:
        for index in range(changed):
            branch_after[f"branch_{index:03d}"] = f"moved-{index}"
    return {
        "disposition": "COMPLETED", "split": "validation",
        "week_start": week["start"], "week_end": week["end"],
        "fit_start": week["fit_start"], "cutoff": week["cutoff"],
        "seed": 17, "n_scored": 5, "rows_sha256": rows,
        "fit_population_digest": "fit", "inner_population_digest": "inner",
        "input_sha256": "input", "standardiser_sha256": "scaler",
        "naive": {"rows_sha256": rows, "rule": "zero-return"},
        "metrics": {"mae": 0.2, "mse": 0.05,
                    "naive_mae": 0.4, "naive_mse": 0.2},
        "cost": {"fit_seconds": 1.0, "n_params": 10},
        "model_sha256": f"model-{arm}",
        "regime_evidence": {
            "observed_updates": 1,
            "trainable_variable_count": 2,
            "trainable_branch_variables": 0 if arm == "R1_B" else 1,
            "branch_changed_count": changed,
            "branch_weights_before": branch_before,
            "branch_weights_after": branch_after,
            "core_weights_before": "same-core",
            "core_weights_after": "updated-core",
            "head_weights_before": "same-head",
            "head_weights_after": "updated-head",
            "donor_set_sha256": None if arm == "R0" else regimes.digest(
                design["donor_index"]["weeks"]["0"]
            ),
        },
        "target_transform": {"method": "ROBUST_Z_FIT", "center": 0.0,
                             "scale": 1.0, "fit_rows_sha256": "fit"},
    }


def test_closure_is_row_weighted_reports_paired_deltas_and_suppresses_on_parity_mismatch(
        campaign_material):
    _, _, _, design = campaign_material
    week = design["weeks"][0]
    cells = [regimes.seal_cell(design, regimes.make_task(design, 0, 17), arm,
                               _result(design, week, arm))
             for arm in regimes.ARMS]

    closure = regimes.close_cells(design, cells)
    assert closure["state"] == "COMPLETE"
    assert closure["annual"]["R1_B"]["mae"] == pytest.approx(0.2)
    assert closure["annual"]["R2_B"]["skill_mae"] == pytest.approx(0.5)
    assert closure["paired_deltas_to_R0"]["R1_B"]["mae"] == pytest.approx(0.0)
    assert closure["regime_evidence"]["R1_B"]["branch_changed_count"] == 0

    broken = copy.deepcopy(cells)
    broken[1]["result"]["rows_sha256"] = "different"
    broken[1]["result"]["result_sha256"] = regimes.digest(
        {key: value for key, value in broken[1]["result"].items()
         if key != "result_sha256"}
    )
    broken[1] = regimes.reseal_cell(broken[1])
    refused = regimes.close_cells(design, broken)
    assert refused["state"] == "INCOMPLETE_EVIDENCE"
    assert "annual" not in refused
    assert any("ARM_PARITY_MISMATCH:rows_sha256" in item["reason"]
               for item in refused["problems"])


def test_test_is_unreachable_and_campaign_files_are_atomic_and_resumable(
        tmp_path, campaign_material):
    _, parent_design, donor_index, design = campaign_material
    with pytest.raises(ValueError, match="TEST_IS_SEALED"):
        regimes.make_task(design, 0, 17, split="test")

    parent_path = tmp_path / "WEEKLY_DESIGN.json"
    index_path = tmp_path / "DONOR_INDEX.json"
    parent_path.write_text(json.dumps(parent_design))
    index_path.write_text(json.dumps(donor_index))
    output = tmp_path / "campaign"
    created = regimes.initialize_campaign(parent_path, index_path, output)
    status = regimes.campaign_status(output)

    assert created["design_sha256"] == design["design_sha256"]
    assert status["state"] == "READY"
    assert status["pending_cells"] == 3
    assert not list(output.rglob("*.tmp"))
