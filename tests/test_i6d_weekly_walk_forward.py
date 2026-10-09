"""Contract tests for the I6-D BUSINESS weekly walk-forward successor."""

from __future__ import annotations

import copy
from datetime import datetime, timezone

import pytest

from tools import fs4_weekly_wrapper as weekly_contract
from tools import i6d_weekly_walk_forward as weekly


UTC = timezone.utc


def config():
    return {
        "schema": "predictor.i6d.matched_control.v1",
        "experiment_id": "i6d-weekly-fixture",
        "feature_names": ["a__value", "a__observed", "b__value", "b__observed"],
        "feature_groups": [
            {"name": "a", "channels": ["a__value", "a__observed"]},
            {"name": "b", "channels": ["b__value", "b__observed"]},
        ],
        "window": 24,
        "sample_hours": 1,
        "horizons": [1],
        "target_names": ["Y_s_1h"],
        "seed": 7,
        "branch": {
            "dense": {"hidden_units": [8], "latent_units": 4,
                      "activation": "gelu", "use_bias": True},
            "conv": {"channels": 4, "kernel_size": 3},
        },
        "core": {
            "vector_width": 24,
            "dense_hidden_units": [16],
            "conv": {"d_model": 16, "heads": 2, "blocks": 1, "ff_dim": 24,
                     "dropout": 0.0, "stage_channels": [12, 10, 4],
                     "time_factors": [2, 2, 1], "kernel_size": 3},
            "conv_output_steps": 6,
            "conv_output_channels": 4,
        },
        "head": {"initializer_seed": 991},
        "fit": {"max_epochs": 3, "patience": 1, "batch_size": 4,
                "learning_rate": 0.001, "weight_decay": 0.0001,
                "loss": "huber", "huber_delta": 1.0, "min_delta": 0.0,
                "max_updates": 20, "max_seconds": 60.0,
                "monitor": "validation_loss", "monitor_every": 1},
    }


def _cell(design, week_ordinal, arm, *, n=10, mae=0.2, naive=0.4):
    task = weekly.make_weekly_task(design, week_ordinal, design["seeds"][0])
    result = {
        "disposition": "COMPLETED",
        "split": "validation",
        "week_start": task["week"]["start"],
        "week_end": task["week"]["end"],
        "fit_start": task["week"]["fit_start"],
        "cutoff": task["week"]["cutoff"],
        "seed": task["seed"],
        "n_scored": n,
        "rows_sha256": f"rows-{week_ordinal}",
        "fit_population_digest": f"fit-{week_ordinal}",
        "inner_population_digest": f"inner-{week_ordinal}",
        "input_sha256": "input",
        "standardiser_sha256": f"scaler-{week_ordinal}",
        "naive": {"rows_sha256": f"rows-{week_ordinal}", "rule": "zero-return"},
        "metrics": {"mae": mae, "mse": mae * mae,
                    "naive_mae": naive, "naive_mse": naive * naive},
        "cost": {"fit_seconds": 1.0, "n_params": 10},
        "model_sha256": f"model-{arm}-{week_ordinal}",
    }
    return weekly.seal_weekly_cell(design, task, arm, result)


def test_default_is_one_seed_and_every_validation_week_has_exact_rolling_four_year_boundary():
    design = weekly.build_weekly_design(config(), validation_year=2024)

    assert design["seeds"] == [7]
    assert design["evaluation_mode"] == "BUSINESS_WEEKLY_WALK_FORWARD"
    assert design["update_mode"] == "FULL_RETRAIN_ROLLING_4Y"
    assert design["test_read"] is False
    assert design["test_paths"] is None
    assert len(design["weeks"]) == 52
    for ordinal, item in enumerate(design["weeks"]):
        task = weekly.make_weekly_task(design, ordinal, 7)
        start = datetime.fromisoformat(item["start"])
        fit_start = datetime.fromisoformat(item["fit_start"])
        assert start.tzinfo == UTC
        assert fit_start == start.replace(year=start.year - 4)
        assert item["cutoff"] == item["start"]
        assert task["split"] == "validation"


def test_seed_population_defaults_to_one_and_refuses_more_than_three_or_implicit_test():
    assert weekly.build_weekly_design(config(), 2024)["seeds"] == [7]
    assert weekly.build_weekly_design(config(), 2024, seeds=[1, 2, 3])["seeds"] == [1, 2, 3]
    with pytest.raises(ValueError, match="at most three"):
        weekly.build_weekly_design(config(), 2024, seeds=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="TEST_IS_SEALED"):
        weekly.make_weekly_task(weekly.build_weekly_design(config(), 2024), 0, 7,
                                split="test")


def test_run_cell_delegates_week_population_to_existing_weekly_contract(monkeypatch):
    design = weekly.build_weekly_design(config(), 2024)
    captured = {}

    def stand_in(task, store, **kwargs):
        captured.update(task=task, store=store, kwargs=kwargs)
        return {
            "disposition": "COMPLETED", "split": "validation",
            "week_start": task["week"]["start"], "week_end": task["week"]["end"],
            "cutoff": task["week"]["cutoff"], "fit_start": task["week"]["fit_start"],
            "seed": task["seed"], "n_scored": 2, "rows_sha256": "rows",
            "fit_population_digest": "fit", "inner_population_digest": "inner",
            "input_sha256": "input", "standardiser_sha256": "scaler",
            "naive": {"rows_sha256": "rows"},
            "metrics": {"mae": 0.2, "mse": 0.04, "naive_mae": 0.4,
                        "naive_mse": 0.16},
            "cost": {"fit_seconds": 1.0, "n_params": 10},
            "model_sha256": "model",
        }

    monkeypatch.setattr(weekly_contract, "run_task", stand_in)
    cell = weekly.run_weekly_cell(design, "DENSE", 0, 7, object())

    assert captured["task"]["split"] == "validation"
    assert captured["task"]["week"]["fit_start"].startswith("2020-")
    assert captured["task"]["week"]["cutoff"] == captured["task"]["week"]["start"]
    assert captured["kwargs"]["expected_seed"] == 7
    assert callable(captured["kwargs"]["trainer"])
    assert cell["test_read"] is False


@pytest.mark.parametrize("arm", weekly.ARMS)
def test_weekly_model_adapter_keeps_i6d_graph_and_exposes_one_scalar_target(arm):
    spec = weekly._predictor_spec(config())
    model = weekly._model_builder(config(), arm, 7)(spec, 2, "RAW", None)

    assert model.input_shape == (None, 24, 4)
    assert model.output_shape == (None, 1)
    assert isinstance(model.fs4_architecture, str)
    assert len(model.fs4_architecture) == 64


def test_inherited_weekly_contract_records_an_explicit_nonzero_seed(monkeypatch):
    class Store:
        population = "EURUSD"

    task = {"schema": weekly_contract.TASK_SCHEMA, "task_id": "task",
            "population_id": "EURUSD", "seed": 9, "split": "invalid"}
    with pytest.raises(weekly_contract.Refusal, match="SPLIT_INVALID"):
        weekly_contract.run_task(task, Store(), expected_seed=9)
    with pytest.raises(weekly_contract.Refusal, match="SEED_MISMATCH"):
        weekly_contract.run_task(task, Store(), expected_seed=0)


def test_closure_requires_same_rows_for_both_arms_and_suppresses_aggregate_on_mismatch():
    design = weekly.build_weekly_design(config(), 2024)
    cells = [_cell(design, week, arm) for week in range(len(design["weeks"]))
             for arm in weekly.ARMS]
    damaged = copy.deepcopy(cells)
    victim = next(item for item in damaged if item["arm"] == "CONV")
    victim["result"]["rows_sha256"] = "other-rows"
    victim["result"]["result_sha256"] = weekly_contract.digest(
        {key: value for key, value in victim["result"].items()
         if key != "result_sha256"}
    )
    victim = weekly.reseal_weekly_cell(victim)
    damaged[damaged.index(next(item for item in damaged
                               if item["arm"] == "CONV" and item["week_ordinal"] == 0))] = victim

    closure = weekly.close_weekly_cells(design, damaged)

    assert closure["state"] == "INCOMPLETE_EVIDENCE"
    assert any(problem["reason"] == "ARM_PARITY_MISMATCH:rows_sha256"
               for problem in closure["problems"])
    assert "annual" not in closure


def test_annual_decision_population_is_row_weighted_and_uses_same_row_naive():
    design = weekly.build_weekly_design(config(), 2024)
    cells = []
    for week in range(len(design["weeks"])):
        n = 1 if week == 0 else 3
        mae = 1.0 if week == 0 else 0.0
        naive = 2.0 if week == 0 else 1.0
        cells.extend(_cell(design, week, arm, n=n, mae=mae, naive=naive)
                     for arm in weekly.ARMS)

    closure = weekly.close_weekly_cells(design, cells)
    expected_model = 1.0 / (1 + 3 * 51)
    expected_naive = (2.0 + 3 * 51) / (1 + 3 * 51)

    assert closure["state"] == "COMPLETE"
    assert closure["annual"]["DENSE"]["mae"] == pytest.approx(expected_model)
    assert closure["annual"]["CONV"]["naive_mae"] == pytest.approx(expected_naive)
    assert closure["annual"]["DENSE"]["same_row_naive"] is True
    assert closure["decision_population"]["split"] == "validation"
    assert closure["test_read"] is False


def test_static_i6d_cli_commands_remain_available():
    from tools.i6d_matched_control import _parser

    choices = _parser()._subparsers._group_actions[0].choices
    for command in ("plan", "init", "run-arm", "status", "close"):
        assert command in choices
    for command in ("weekly-plan", "weekly-init", "weekly-run-cell",
                    "weekly-status", "weekly-close"):
        assert command in choices
