"""Governed H72 input-downsampling successor.

The experiment changes only the input grid.  Its physical endpoint, business
support and paired 72-hour naive remain those of the sealed H72 fold binding.
"""

from __future__ import annotations

import copy
import json
import random

from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_doin_sequence_config as sequence
from tools import i6n_island_plan as island_plan
from tools import i6n_search_space as space
from tests.test_i6n_historical_cell import fixture  # noqa: F401
from tests.test_i6n_island_optimizer import RecordingEvaluator, _config
from tests.test_i6n_stlfsi4_exclusion import GENESIS, _data_root, _nodes


CONSTRAINTS = {
    "schema": "i6n.genotype_constraints.v1",
    "delta_x_min_hours": 2,
    "delta_x_max_hours": 72,
    "supervisions": ["endpoint"],
    "input_transforms": ["identity", "difference"],
}


def test_h72_ds_in_plan_is_a_distinct_sealed_input_only_identity():
    raw = island_plan.build_plan_v3()
    plan = island_plan.build_plan_v4_ds_in()

    assert plan["plan_sha256"] != raw["plan_sha256"]
    assert plan["search_constraints"] == CONSTRAINTS
    assert plan["sequence_scope"] == "H72-DS-IN only; hourly/raw H72 is a later control"
    assert plan["feature_manifest"] == raw["feature_manifest"]


def test_h72_ds_in_generator_preserves_fold_support_and_naive(fixture, tmp_path):  # noqa: F811
    root, source = _data_root(tmp_path, fixture[0])
    expected_folds = copy.deepcopy(source["domains"][0]["optimization_config"]["i6n_folds"])

    manifest = sequence.generate_h72_ds_in_v4(
        source, _nodes(), tmp_path / "out", "<predictor-root>", "<python>",
        campaign_root="<campaign-root>", data_root=root, genesis_hash=GENESIS,
    )

    config = json.loads(
        (tmp_path / "out" / "h72-ds-in" / "node-0" / "doin_node.json").read_text()
    )
    domain = config["domains"][0]
    opt = domain["optimization_config"]
    assert manifest["order"] == ["H72-DS-IN"]
    assert domain["domain_id"] == island_plan.DOMAIN_ID_V4_DS_IN
    assert opt["i6n_genotype_constraints"] == CONSTRAINTS
    for fold, expected in expected_folds.items():
        actual = opt["i6n_folds"][fold]
        assert actual["i6n_support"] == expected["i6n_support"]
        assert actual.get("i6n_origins_sha256") == expected.get("i6n_origins_sha256")
        assert actual["i6n_evaluation_context"]["base_config"] == expected[
            "i6n_evaluation_context"
        ]["base_config"]
        assert actual["i6n_evaluation_context"]["frozen_support"] == expected[
            "i6n_evaluation_context"
        ]["frozen_support"]
        assert actual["i6n_evaluation_context"]["experiment"] == island_plan.EXPERIMENT_V4_DS_IN
    assert opt["i6n_admitted_classes"] == ["S1", "S2"]


def test_h72_ds_in_population_and_mutation_never_escape_constraints():
    plugin = Plugin()
    cfg = _config(
        RecordingEvaluator(), horizon=72,
        i6n_plan=island_plan.build_plan_v4_ds_in(),
        i6n_genotype_constraints=copy.deepcopy(CONSTRAINTS),
        i6n_admitted_classes=["S1", "S2"],
        i6n_control_island_id="omega-4070",
    )
    settings = plugin._island_settings(cfg)
    constraints = plugin._genotype_constraints(cfg, cfg["i6n_plan"])
    population = plugin._initial_island_population(
        settings, {"S1", "S2"}, random.Random(17), constraints
    )

    assert len(population) == settings["population"]
    assert all(g["delta_x_hours"] >= 2 for g in population)
    assert all(g["supervision"] == "endpoint" for g in population)
    for genotype in population:
        derived = space.derived(genotype)
        assert derived["input_offsets_hours"][-1] == 0
        assert derived["output_offsets_hours"] == [72]

    ranked = [
        {"genotype": g, "fitness": float(index), "generation": 0}
        for index, g in enumerate(population)
    ]
    successor = plugin._next_island_population(
        ranked, settings, {"S1", "S2"}, random.Random(19), constraints
    )
    assert all(g["delta_x_hours"] >= 2 for g in successor)
    assert all(g["supervision"] == "endpoint" for g in successor)


def test_h72_ds_in_rejects_hourly_or_trajectory_network_champions():
    plan = island_plan.build_plan_v4_ds_in()
    envelope = {
        "schema": "i6n.island.champion.v1",
        "plan_sha256": plan["plan_sha256"],
        "context_sha256": "9" * 64,
        "horizon": 72,
        "training_seed": 6102026,
        "genotype_schema": space.GENOTYPE_SCHEMA,
    }
    constraints = Plugin._genotype_constraints(
        {"i6n_genotype_constraints": copy.deepcopy(CONSTRAINTS)}, plan
    )
    allowed = space.seed_genotype(72)
    hourly = copy.deepcopy(allowed)
    hourly.update(history_hours=72, delta_x_hours=1)
    trajectory = copy.deepcopy(allowed)
    trajectory.update(supervision="trajectory", trajectory_k=2)

    def payload(genotype):
        return dict(space.to_evaluator_genome(space.canonicalize(genotype)),
                    _i6n_envelope=envelope)

    assert Plugin._network_genotype(payload(allowed), 72, envelope, constraints) is not None
    assert Plugin._network_genotype(payload(hourly), 72, envelope, constraints) is None
    assert Plugin._network_genotype(payload(trajectory), 72, envelope, constraints) is None
