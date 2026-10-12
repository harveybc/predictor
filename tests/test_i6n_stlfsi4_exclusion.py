"""Owner order 2026-10-12: exclude STLFSI4 before H72 and isolate the new identity.

Invented CPU fixtures only.  These tests prove that (1) a 19-feature cell must
pass the availability gate, (2) the H72 successor identity differs from every
STLFSI4-bearing identity, and (3) checkpoints, migrants, fit caches, rankings
and start receipts of the contaminated identities are refused under it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from optimizer_plugins.i6n_optimizer import Plugin
from tools import i6n_contamination as contamination
from tools import i6n_doin_sequence_config as sequence
from tools import i6n_historical_cell as cell
from tools import i6n_island_plan as island_plan
from tools import i6n_island_successor as successor
from tools import i6n_search_space as space
from tools import i6n_weekly_rollup as rollup
from tools.i6m_data import file_hash, load_input
from tools.i6n_contracts import feature_availability as fa

from tests.test_i6n_historical_cell import fixture  # noqa: F401  (pytest fixture)
from tests.test_i6n_island_optimizer import FOLDS, RecordingEvaluator, _config, _optimize, _support

LEGACY_PLAN = "1b83490b54af927033a93105e3be85b8b7323c89672b147745bbeb1d3efee983"
LEGACY_H4 = "fb70328d8b057f110006e0713d304b5c42fec5419a8dcee35527add6ca19a4cf"
LEGACY_H72 = "c22b67e456c9bcc7156024dd5318587d4c876f53220686c41e28e5c9d9aeb97a"
LEGACY_EXPERIMENT = "i6n-swarm-weekly-2024-v1"
GENESIS = "4e19257e8941caec2ec6a4d581a981a533ab5728c11e58c3d13040049e3cd6d5"


# --------------------------------------------------------------------------- cells

def _v2_cell(cfg, tmp_path, features):
    """Rewrite the invented v1 fixture as a v2 cell with ``len(features)`` columns."""
    cfg = copy.deepcopy(cfg)
    with np.load(cfg["input"]["path"]) as archive:
        ts, price = archive["ts"], archive["price"]
    x = np.arange(len(ts) * len(features), dtype=float).reshape(-1, len(features))
    path = tmp_path / f"input{len(features)}.npz"
    np.savez(path, ts=ts, x=x, price=price)
    clock = json.loads(Path(cfg["source_clock"]["path"]).read_text())
    clock.update(input_sha256=file_hash(path), features=list(features))
    clock_path = tmp_path / f"clock{len(features)}.json"
    clock_path.write_text(json.dumps(clock))
    cfg.update(schema=cell.SCHEMA_V2, features=list(features),
               input=dict(path=str(path), sha256=file_hash(path)),
               source_clock=dict(path=str(clock_path), sha256=file_hash(clock_path)),
               feature_manifest_sha256=fa.manifest_sha256(features),
               availability_contract_sha256=fa.load_contract()["contract_sha256"])
    return cfg


def test_v2_cell_with_the_gated_nineteen_features_prepares_19_channels(fixture, tmp_path):  # noqa: F811
    cfg = _v2_cell(fixture[0], tmp_path, fa.MANIFEST_NO_STLFSI4)
    cell.authenticated(cfg)
    data = cell.prepare(cfg)
    assert data["scaled"].shape[1] == 2 * 19
    origins = np.array(data["ts"][-3:], dtype=np.int64)
    assert cell.windows(data["ts"], data["scaled"], origins, 1, 3).shape == (3, 3, 2 * 19 + 1)
    ts, x, _ = load_input(cfg["input"]["path"], cfg["input"]["sha256"], n_features=19)
    assert x.shape == (len(ts), 19)
    with pytest.raises(ValueError, match="invalid hourly"):
        load_input(cfg["input"]["path"], cfg["input"]["sha256"])  # legacy default stays 20


@pytest.mark.parametrize(("swap", "code"), [
    ("fred.new.series.level", "UNDECLARED_FEATURE"),
    ("fred.stress.stlfsi4.level", "EXCLUDED_FEATURE"),
])
def test_v2_cell_refuses_undeclared_or_excluded_features(fixture, tmp_path, swap, code):  # noqa: F811
    features = list(fa.MANIFEST_NO_STLFSI4)
    features[0] = swap
    cfg = _v2_cell(fixture[0], tmp_path, features)
    with pytest.raises(fa.AvailabilityGateError, match=code):
        cell.authenticated(cfg)


@pytest.mark.parametrize("field", ["feature_manifest_sha256", "availability_contract_sha256"])
def test_v2_cell_binds_manifest_and_contract_digests(fixture, tmp_path, field):  # noqa: F811
    cfg = _v2_cell(fixture[0], tmp_path, fa.MANIFEST_NO_STLFSI4)
    cfg[field] = "0" * 64
    with pytest.raises(ValueError, match=field):
        cell.authenticated(cfg)


def test_v1_cell_still_requires_exactly_twenty_features(fixture, tmp_path):  # noqa: F811
    cfg = _v2_cell(fixture[0], tmp_path, fa.MANIFEST_NO_STLFSI4)
    cfg["schema"] = "i6n.historical_cell.v1"
    for key in ("feature_manifest_sha256", "availability_contract_sha256"):
        cfg.pop(key)
    with pytest.raises(ValueError, match="exactly 20"):
        cell.authenticated(cfg)


# --------------------------------------------------------------------------- plan

def test_successor_plan_is_sealed_binds_the_manifest_and_supersedes_the_legacy_plan():
    legacy = island_plan.build_plan()
    plan = island_plan.build_plan_v3()
    assert legacy["plan_sha256"] == LEGACY_PLAN  # the H4 identity is untouched
    assert plan["plan_sha256"] != LEGACY_PLAN
    island_plan.legacy.verify_plan(plan)
    manifest = plan["feature_manifest"]
    assert manifest["features"] == list(fa.MANIFEST_NO_STLFSI4)
    assert manifest["sha256"] == fa.manifest_sha256(fa.MANIFEST_NO_STLFSI4)
    assert manifest["excluded"] == ["fred.stress.stlfsi4.level"]
    assert manifest["availability_contract_sha256"] == fa.load_contract()["contract_sha256"]
    assert plan["point_in_time"]["contamination_label_sha256"] == contamination.load_label()["label_sha256"]
    assert plan["supersedes"][-1]["plan_sha256"] == LEGACY_PLAN
    assert "19" in plan["features"] and "STLFSI4" in plan["features"]


# --------------------------------------------------------------------------- generator

def _nodes():
    return [{"island_id": f"node-{i}", "endpoint": f"node-{i}.invalid:{18570 + i}",
             "candidate_archive_store_url": "http://127.0.0.1:15061"} for i in range(4)]


def _data_root(tmp_path, fixture_cfg, features=fa.MANIFEST_NO_STLFSI4):  # noqa: F811
    """Four portable v2 fold cells and frozen supports under one invented data root."""
    root = tmp_path / "data"
    folds = {}
    for index, fold in enumerate(FOLDS):
        cells = root / "c" / "plan-v6-no-stlfsi4" / "cells"
        cells.mkdir(parents=True, exist_ok=True)
        cfg = dict(_v2_cell(fixture_cfg, tmp_path, features), horizon_hours=72)
        path = cells / f"{fold}_h72.json"
        path.write_text(json.dumps(cfg))
        frozen_dir = root / "bindings" / "i6nw3-no-stlfsi4" / "freeze_folds_h72" / fold
        frozen_dir.mkdir(parents=True, exist_ok=True)
        support = _support(index)
        (frozen_dir / "FROZEN.json").write_text(json.dumps({"schema": "i6n.historical_evaluator.v3",
                                                            "support": support}))
        folds[fold] = {
            "i6n_support": support,
            "i6n_evaluation_context": {
                "schema": "i6n.historical_evaluator.v3",
                "base_config": {"relpath": path.relative_to(root).as_posix(), "sha256": file_hash(path)},
                "frozen_support": {"relpath": (frozen_dir / "FROZEN.json").relative_to(root).as_posix(),
                                   "sha256": file_hash(frozen_dir / "FROZEN.json")},
                "output_root": {"relpath": frozen_dir.relative_to(root).as_posix()},
                "experiment": island_plan.EXPERIMENT_V3, "fold": fold,
            },
        }
    source = {"domains": [{"optimization_config": {
        "i6n_horizon": 72, "i6n_folds": folds, "i6n_admitted_classes": ["S1", "S2"]}}]}
    return root, source


def _generate(tmp_path, root, source):
    return sequence.generate_h72_v3(
        source, _nodes(), tmp_path / "out", "<predictor-root>", "<python>",
        campaign_root="<campaign-root>", data_root=root, genesis_hash=GENESIS)


def test_v3_generator_writes_a_new_h72_identity_and_never_reuses_legacy_paths(fixture, tmp_path):  # noqa: F811
    root, source = _data_root(tmp_path, fixture[0])
    manifest = _generate(tmp_path, root, source)
    assert manifest["horizons_hours"] == [72] and manifest["order"] == ["H72"]
    digest = manifest["optimization_config_sha256_by_horizon"]["72"]
    assert digest not in (LEGACY_H4, LEGACY_H72)
    assert manifest["supersedes"] == {"h4": LEGACY_H4, "h72": LEGACY_H72, "plan_sha256": LEGACY_PLAN}
    assert manifest["domain_id"] == island_plan.DOMAIN_ID_V3
    assert manifest["network_id"] == sequence.network_id(GENESIS, [island_plan.DOMAIN_ID_V3])
    assert manifest["network_id"] not in contamination.load_label()["identities"]["network_id"]
    assert manifest["feature_manifest_sha256"] == fa.manifest_sha256(fa.MANIFEST_NO_STLFSI4)
    config = json.loads((tmp_path / "out" / "h72" / "node-0" / "doin_node.json").read_text())
    domain = config["domains"][0]
    assert domain["domain_id"] == island_plan.DOMAIN_ID_V3
    assert sequence.canonical_sha256(domain["optimization_config"]) == digest
    opt = domain["optimization_config"]
    assert opt["i6n_feature_manifest"]["features"] == list(fa.MANIFEST_NO_STLFSI4)
    tag = island_plan.RUN_TAG_V3
    for value in (config["data_dir"], config["candidate_archive_root"], config["archive_reference_spool"],
                  opt["optimization_resume_file"], opt["optimization_parameters_file"]):
        assert f"/{tag}/" in value and "/h72/" not in value
    assert not contamination.is_contaminated({"scientific_config_sha256": digest,
                                              "plan_sha256": opt["i6n_plan"]["plan_sha256"],
                                              "experiment": island_plan.EXPERIMENT_V3})


def test_v3_generator_refuses_a_fold_whose_features_fail_the_gate(fixture, tmp_path):  # noqa: F811
    features = list(fa.MANIFEST_NO_STLFSI4)
    features[3] = "fred.undeclared.level"
    root, source = _data_root(tmp_path, fixture[0], features)
    with pytest.raises(fa.AvailabilityGateError, match="UNDECLARED_FEATURE"):
        _generate(tmp_path, root, source)
    assert not (tmp_path / "out").exists()


def test_v3_generator_refuses_the_legacy_experiment_and_digest_drift(fixture, tmp_path):  # noqa: F811
    root, source = _data_root(tmp_path, fixture[0])
    legacy = copy.deepcopy(source)
    for fold in legacy["domains"][0]["optimization_config"]["i6n_folds"].values():
        fold["i6n_evaluation_context"]["experiment"] = LEGACY_EXPERIMENT
    with pytest.raises(contamination.ContaminatedResult, match="POINT_IN_TIME_CONTAMINATED"):
        _generate(tmp_path, root, legacy)
    drift = copy.deepcopy(source)
    drift["domains"][0]["optimization_config"]["i6n_folds"][FOLDS[1]][
        "i6n_evaluation_context"]["base_config"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="base_config digest"):
        _generate(tmp_path, root, drift)


# --------------------------------------------------------------------------- isolation

def test_fit_cache_identity_differs_between_legacy_and_successor_bindings():
    genotype = space.seed_genotype(72)
    old = space.fit_id(genotype, {"experiment": LEGACY_EXPERIMENT, "fold_id": "2023Q4",
                                  "context_sha256": "1" * 64, "support_sha256": "2" * 64})
    new = space.fit_id(genotype, {"experiment": island_plan.EXPERIMENT_V3, "fold_id": "2023Q4",
                                  "context_sha256": "1" * 64, "support_sha256": "2" * 64})
    assert old != new


def _v3_config(evaluator, **changes):
    plan = island_plan.build_plan_v3()
    cfg = _config(evaluator, horizon=72, i6n_plan=plan,
                  i6n_feature_manifest=copy.deepcopy(plan["feature_manifest"]), **changes)
    for fold in cfg["i6n_folds"].values():
        fold["i6n_evaluation_context"]["experiment"] = island_plan.EXPERIMENT_V3
    return cfg


def test_successor_island_refuses_a_legacy_checkpoint_before_any_evaluation(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    _optimize(_config(RecordingEvaluator(), horizon=72, i6n_island_id="island-a",
                      optimization_resume_file=str(checkpoint)))
    assert json.loads(checkpoint.read_text())["checkpoint_identity"]["plan_sha256"] == LEGACY_PLAN

    def must_not_run(_request):
        raise AssertionError("a contaminated checkpoint must be refused before evaluation")

    with pytest.raises(ValueError, match="POINT_IN_TIME_CONTAMINATED"):
        _optimize(_v3_config(must_not_run, i6n_island_id="island-a",
                             optimization_resume_file=str(checkpoint)))


def test_successor_island_refuses_contaminated_contexts_and_manifest_drift():
    def must_not_run(_request):
        raise AssertionError("must be refused before evaluation")

    legacy_context = _v3_config(must_not_run)
    legacy_context["i6n_folds"][FOLDS[2]]["i6n_evaluation_context"]["experiment"] = LEGACY_EXPERIMENT
    with pytest.raises(ValueError, match="POINT_IN_TIME_CONTAMINATED"):
        _optimize(legacy_context)
    drift = _v3_config(must_not_run)
    drift["i6n_feature_manifest"]["sha256"] = fa.manifest_sha256(fa.LEGACY_MANIFEST_20)
    with pytest.raises(ValueError, match="feature manifest"):
        _optimize(drift)


def test_successor_island_never_adopts_a_legacy_migrant():
    plan = island_plan.build_plan_v3()
    new_envelope = {"schema": "i6n.island.champion.v1", "plan_sha256": plan["plan_sha256"],
                    "context_sha256": "9" * 64, "horizon": 72, "training_seed": 6102026,
                    "genotype_schema": space.GENOTYPE_SCHEMA}
    genome = space.to_evaluator_genome(space.seed_genotype(72))
    legacy_params = dict(genome, _i6n_envelope=dict(new_envelope, plan_sha256=LEGACY_PLAN))
    assert Plugin._network_genotype(legacy_params, 72, new_envelope) is None
    legacy_context = dict(genome, _i6n_envelope=dict(new_envelope, context_sha256="8" * 64))
    assert Plugin._network_genotype(legacy_context, 72, new_envelope) is None
    assert Plugin._network_genotype(dict(genome, _i6n_envelope=new_envelope), 72, new_envelope) is not None


def test_contaminated_results_cannot_be_ranked_or_compared_as_valid_skill():
    label = contamination.load_label()
    assert label["label"] == "DEVELOPMENT_TUNING_POINT_IN_TIME_CONTAMINATED"
    assert LEGACY_H4 in label["identities"]["scientific_config_sha256"]
    h4 = {"plan_sha256": LEGACY_PLAN, "scientific_config_sha256": LEGACY_H4, "fitness": -0.0106}
    new = {"plan_sha256": island_plan.build_plan_v3()["plan_sha256"], "fitness": -0.001}
    assert contamination.is_contaminated(h4)
    with pytest.raises(contamination.ContaminatedResult, match="POINT_IN_TIME_CONTAMINATED"):
        contamination.rank_valid_skill([new, h4])
    with pytest.raises(contamination.ContaminatedResult):
        contamination.compare_skill(new, h4)
    assert contamination.rank_valid_skill([new]) == [new]
    row = {k: None for k in rollup.IDENTITY}
    a = dict(row, scope=rollup.INNER_SCOPE, experiment=LEGACY_EXPERIMENT)
    b = dict(row, scope=rollup.INNER_SCOPE, experiment=island_plan.EXPERIMENT_V3)
    with pytest.raises(contamination.ContaminatedResult):
        rollup.assert_comparable(a, b)
    rollup.assert_comparable(b, dict(b))


# --------------------------------------------------------------------------- successor spec

def _v2_spec(tmp_path, **expected_changes):
    terminal_paths = [tmp_path / "terminals" / f"island-{i}.json" for i in range(4)]
    expected = {
        "seed": 6102026, "h4_scientific_config_sha256": "4" * 64,
        "h72_scientific_config_sha256": "7" * 64, "network_id": "network", "genesis_hash": "genesis",
        "component_versions": {"doin-node": "n", "doin-core": "c", "doin-plugins": "p", "predictor": "old"},
        "h72_network_id": "h72-network", "h72_genesis_hash": "genesis",
        "h72_component_versions": {"doin-node": "n", "doin-core": "c", "doin-plugins": "p",
                                   "predictor": "gated"},
    }
    expected.update(expected_changes)
    spec = {
        "schema": successor.SPEC_SCHEMA_V2, "state_dir": str(tmp_path / "state"),
        "poll_interval_seconds": 60, "launch_retry_seconds": 300,
        "terminal_receipts": [str(p) for p in terminal_paths], "expected": expected,
        "archive_closure_receipt": str(tmp_path / "closure.json"),
        "h72_start": {"command": ["start"], "receipt": str(tmp_path / "START.json"),
                      "working_directory": str(tmp_path), "environment": {},
                      "verification_timeout_seconds": 2.0,
                      "nodes": [{"node_id": f"island-{i}", "base_url": f"http://node-{i}.invalid"}
                                for i in range(4)]},
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_successor_v2_spec_carries_separate_h72_identity(tmp_path):
    spec = successor.load_spec(_v2_spec(tmp_path))
    assert spec["h72_expected"]["network_id"] == "h72-network"
    assert spec["h72_expected"]["component_versions"]["predictor"] == "gated"
    assert spec["expected"]["network_id"] == "network"


def test_successor_v2_spec_refuses_a_contaminated_h72_identity(tmp_path):
    with pytest.raises(successor.SpecError, match="POINT_IN_TIME_CONTAMINATED"):
        successor.load_spec(_v2_spec(tmp_path, h72_scientific_config_sha256=LEGACY_H72))
    with pytest.raises(successor.SpecError, match="POINT_IN_TIME_CONTAMINATED"):
        successor.load_spec(_v2_spec(tmp_path, h72_network_id=contamination.load_label()[
            "identities"]["network_id"][0]))


# --------------------------------------------------------------------------- data preparation

def _legacy_root(tmp_path):
    from tools import i6n_h72_no_stlfsi4_prepare as prep  # noqa: F401
    root = tmp_path / "legacy"
    source = root / "inputs" / "primary"
    source.mkdir(parents=True)
    ts = np.arange(10, dtype=np.int64) * 3600 + 1_700_000_000 // 3600 * 3600
    x = np.arange(200, dtype=float).reshape(10, 20)
    x[2, 5] = np.nan
    x[4, 7] = np.nan
    np.savez(source / "INPUT.npz", ts=ts, x=x, price=np.linspace(1.0, 1.1, 10))
    clock = {"schema": "i6n.source_clock.v1", "features": list(fa.LEGACY_MANIFEST_20),
             "input_sha256": file_hash(source / "INPUT.npz"), "clock_policy": "invented"}
    (source / "CLOCK.json").write_text(json.dumps(clock))
    for name in ("CALENDAR.json", "CLOCK_SCOPE.json", "PREPARATION.json"):
        (source / name).write_text("{}")
    cells = root / "c" / "plan-v6" / "cells"
    cells.mkdir(parents=True)
    cell_cfg = {"schema": "i6n.historical_cell.v1", "features": list(fa.LEGACY_MANIFEST_20),
                "horizon_hours": 72,
                "input": {"path": "../../../inputs/primary/INPUT.npz", "sha256": clock["input_sha256"]},
                "source_clock": {"path": "../../../inputs/primary/CLOCK.json",
                                 "sha256": file_hash(source / "CLOCK.json")},
                "training": {"seed": 6102026}}
    for unit in ("2023Q4", "W2024-01-07"):
        (cells / f"{unit}_h72.json").write_text(json.dumps(cell_cfg))
    (cells / "2023Q4_h4.json").write_text(json.dumps(dict(cell_cfg, horizon_hours=4)))
    return root, x


def test_prepare_drops_only_the_stlfsi4_column_and_rewrites_only_h72_cells(tmp_path):
    from tools import i6n_h72_no_stlfsi4_prepare as prep
    root, x = _legacy_root(tmp_path)
    stage = tmp_path / "stage"
    report = prep.prepare(root, stage, 72)
    assert report["cells"] == 2 and report["folds"] == ["2023Q4"]
    derived = stage / "inputs" / "primary-no-stlfsi4"
    with np.load(derived / "INPUT.npz") as archive:
        new = archive["x"]
    assert new.shape == (10, 19)
    assert np.array_equal(new, np.delete(x, 5, axis=1), equal_nan=True)
    clock = json.loads((derived / "CLOCK.json").read_text())
    assert clock["features"] == list(fa.MANIFEST_NO_STLFSI4)
    assert clock["input_sha256"] == file_hash(derived / "INPUT.npz")
    derivation = json.loads((derived / "DERIVATION.json").read_text())
    assert derivation["excluded_column_indices"] == [5] and derivation["shifted_by_assumed_lag"] is False
    cell_v2 = json.loads((stage / "c" / prep.PLAN_DIR / "cells" / "2023Q4_h72.json").read_text())
    assert cell_v2["schema"] == cell.SCHEMA_V2 and cell_v2["features"] == list(fa.MANIFEST_NO_STLFSI4)
    assert cell_v2["input"] == {"path": "../../../inputs/primary-no-stlfsi4/INPUT.npz",
                                "sha256": clock["input_sha256"]}
    assert cell_v2["training"] == {"seed": 6102026}
    assert not (stage / "c" / prep.PLAN_DIR / "cells" / "2023Q4_h4.json").exists()
    with pytest.raises(FileExistsError):
        prep.prepare(root, stage, 72)
