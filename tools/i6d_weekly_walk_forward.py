"""BUSINESS weekly walk-forward successor for the I6-D matched control.

This module composes the existing I6-A weekly task contract with the I6-D
Dense-versus-Conv model pair.  Every validation week receives a newly fitted
model whose population is the rolling four calendar years ending before that
week.  TEST has no path, command, or implicit fallback in this workflow.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import resource

import numpy as np

from predictor_plugins.modular_temporal.matched_control import (
    build_matched_control,
    canonical_sha256,
    normalize_matched_config,
)
from tools import business_weekly_training as business_training
from tools import fs4_temporal_predictor as predictor_contract
from tools import fs4_weekly_wrapper as weekly_contract
from tools.i6a_weekly_arch_pilot import target_horizon_hours


ARMS = ("DENSE", "CONV")
DESIGN_SCHEMA = "predictor.i6d.weekly_campaign.v1"
CELL_SCHEMA = "predictor.i6d.weekly_cell.v1"
CLOSURE_SCHEMA = "predictor.i6d.weekly_closure.v1"
STATUS_SCHEMA = "predictor.i6d.weekly_status.v1"
EVALUATION_MODE = business_training.EvaluationMode.BUSINESS_WEEKLY_WALK_FORWARD.value
UPDATE_MODE = business_training.UpdateMode.FULL_RETRAIN_ROLLING_4Y.value


def _atomic_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, destination)


def _seal(value, field):
    body = {key: item for key, item in value.items() if key != field}
    return {**body, field: canonical_sha256(body)}


def _verify_seal(value, field):
    expected = canonical_sha256({key: item for key, item in value.items() if key != field})
    if value.get(field) != expected:
        raise ValueError(f"{field} mismatch")
    return value


def _plain_seeds(config_seed, seeds):
    values = [config_seed] if seeds is None else list(seeds)
    if not values or len(values) > 3:
        raise ValueError("weekly campaigns require one seed by default and at most three explicitly")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in values):
        raise ValueError("seeds must be nonnegative integers, not booleans")
    if len(set(values)) != len(values):
        raise ValueError("seeds must be unique")
    return values


def _member_contract(config):
    """Bind I6-D channels to the value/mask layout emitted by FS4 windows."""
    groups = sorted(config["feature_groups"], key=lambda item: item["name"])
    members = [item["name"] for item in groups]
    expected_channels = [channel for member in members
                         for channel in (f"{member}__value", f"{member}__observed")]
    actual_channels = [channel for item in groups for channel in item["channels"]]
    if actual_channels != expected_channels or config["feature_names"] != expected_channels:
        raise ValueError("I6-D weekly features must be sorted value/observed pairs per member")
    return members


def _predictor_spec(config):
    fit = config["fit"]
    return predictor_contract.PredictorSpec(
        window=24,
        latent_steps=config["core"]["conv_output_steps"],
        max_epochs=fit["max_epochs"],
        batch_size=fit["batch_size"],
        patience=fit["patience"],
        learning_rate=fit["learning_rate"],
        loss=fit["loss"],
    )


def build_weekly_design(config, validation_year, *, seeds=None):
    """Seal the annual validation population without reading any split bytes."""
    normalized = normalize_matched_config(config)
    if isinstance(validation_year, bool) or not isinstance(validation_year, int):
        raise ValueError("validation_year must be an integer")
    if len(normalized["horizons"]) != 1 or len(normalized["target_names"]) != 1:
        raise ValueError("one weekly campaign binds exactly one target and horizon")
    target = normalized["target_names"][0]
    horizon = target_horizon_hours(target)
    if normalized["horizons"] != [horizon]:
        raise ValueError("target name and configured horizon disagree")
    members = _member_contract(normalized)
    seed_population = _plain_seeds(normalized["seed"], seeds)
    identity = {
        "schema": DESIGN_SCHEMA,
        "config_sha256": canonical_sha256(normalized),
        "validation_year": validation_year,
        "target_id": target,
        "horizon_hours": horizon,
        "members": members,
        "seeds": seed_population,
        "evaluation_mode": EVALUATION_MODE,
        "update_mode": UPDATE_MODE,
        "rolling_calendar_years": 4,
    }
    procedure_sha = canonical_sha256(identity)
    protocol = weekly_contract.W.build_protocol(validation_year, procedure_sha)
    weeks = [weekly_contract._week_dict(week) for week in protocol.weeks()
             if week.split is weekly_contract.EvaluationSplit.VALIDATION]
    spec = _predictor_spec(normalized)
    body = {
        **identity,
        "status": "READY",
        "procedure_sha256": procedure_sha,
        "config": normalized,
        "predictor_spec": spec.to_dict(),
        "predictor_spec_sha256": spec.sha256(),
        "encoder_spec_sha256": predictor_contract.EncoderSpec().sha256(),
        "weeks": weeks,
        "expected_cells": len(weeks) * len(seed_population) * len(ARMS),
        "arms": list(ARMS),
        "annual_decision_population": (
            "all finite rows from every complete validation week; annual metrics are "
            "row-weighted and paired to the same-row zero-return persistence naive"
        ),
        "test_paths": None,
        "test_read": False,
    }
    return _seal(body, "design_sha256")


def verify_weekly_design(design):
    if design.get("schema") != DESIGN_SCHEMA:
        raise ValueError("weekly design schema mismatch")
    _verify_seal(design, "design_sha256")
    if design.get("test_paths") is not None or design.get("test_read") is not False:
        raise ValueError("TEST_IS_SEALED")
    if design.get("evaluation_mode") != EVALUATION_MODE or design.get("update_mode") != UPDATE_MODE:
        raise ValueError("weekly BUSINESS protocol mismatch")
    if design.get("rolling_calendar_years") != 4:
        raise ValueError("weekly BUSINESS protocol requires four rolling calendar years")
    _plain_seeds(design["config"]["seed"], design.get("seeds"))
    return design


def make_weekly_task(design, week_ordinal, seed, *, split="validation"):
    """Create the same weekly task identity used by I6-A; TEST is unreachable."""
    design = verify_weekly_design(design)
    if split != "validation":
        raise ValueError("TEST_IS_SEALED: I6-D weekly successor exposes validation only")
    if seed not in design["seeds"]:
        raise ValueError("seed is outside the sealed campaign population")
    if isinstance(week_ordinal, bool) or not isinstance(week_ordinal, int):
        raise ValueError("week_ordinal must be an integer")
    if not 0 <= week_ordinal < len(design["weeks"]):
        raise ValueError("week_ordinal is outside the annual validation population")
    week = design["weeks"][week_ordinal]
    task = {
        "schema": weekly_contract.TASK_SCHEMA,
        "plan_sha256": design["procedure_sha256"],
        "population_id": "EURUSD",
        "identity": design["config_sha256"],
        "set_id": design["config"]["experiment_id"],
        "target_id": design["target_id"],
        "horizon_hours": design["horizon_hours"],
        "members": design["members"],
        "n_features": len(design["members"]),
        "input_mode": "RAW",
        "stage": 1,
        "split": "validation",
        "validation_year": design["validation_year"],
        "week": {key: week[key] for key in
                 ("ordinal", "start", "end", "cutoff", "fit_start")},
        "seed": seed,
        "predictor_spec_sha256": design["predictor_spec_sha256"],
        "encoder_spec_sha256": design["encoder_spec_sha256"],
    }
    task["task_id"] = weekly_contract.digest(task)
    return task


def _model_builder(config, arm, seed):
    def build(spec, n_features, input_mode, latent_dim):
        if input_mode != "RAW" or latent_dim is not None:
            raise predictor_contract.Refusal("I6D_WEEKLY_RAW_ONLY")
        if n_features != len(config["feature_groups"]):
            raise predictor_contract.Refusal("I6D_WEEKLY_FEATURE_COUNT_MISMATCH")
        seeded = copy.deepcopy(config)
        seeded["seed"] = seed
        harness = build_matched_control(seeded)
        selected = harness.dense if arm == "DENSE" else harness.conv
        keras = predictor_contract._keras()
        scalar = keras.layers.Reshape(
            (1,), name="weekly_scalar_target"
        )(selected.model.output)
        model = keras.Model(selected.model.input, scalar,
                            name=f"i6d_weekly_{arm.lower()}")
        model.fs4_architecture = canonical_sha256({
            "schema": "predictor.i6d.weekly_architecture.v1",
            "arm": arm,
            "config_sha256": canonical_sha256(seeded),
            "architecture": harness.report["arms"][arm],
        })
        return model
    return build


def _trainer(design, arm, store, seed, task):
    config = design["config"]
    spec = _predictor_spec(config)
    members = tuple(design["members"])

    def train(model_spec, X, y, fit_idx, inner_idx, input_mode, encoder, actual_seed):
        if actual_seed != seed:
            raise predictor_contract.Refusal("I6D_WEEKLY_SEED_MISMATCH")
        if input_mode != "RAW" or encoder is not None:
            raise predictor_contract.Refusal("I6D_WEEKLY_RAW_ONLY")
        order = sorted(range(len(members)), key=lambda index: members[index])
        ordered_members = tuple(members[index] for index in order)
        values = np.asarray(X, dtype="float64")[:, order]
        targets = np.asarray(y, dtype="float64")
        fit_idx = np.asarray(fit_idx, dtype="int64")
        inner_idx = np.asarray(inner_idx, dtype="int64")
        standardiser = predictor_contract.Standardiser.fit(values[fit_idx])
        minimum = int(weekly_contract._parse(task["week"]["fit_start"]).timestamp())
        fit_windows, kept_fit = predictor_contract._inputs_for(
            model_spec, None, values, standardiser, fit_idx, store.ts, minimum
        )
        inner_windows, kept_inner = predictor_contract._inputs_for(
            model_spec, None, values, standardiser, inner_idx, store.ts, minimum
        )
        if kept_fit.size < config["fit"]["batch_size"] or kept_inner.size == 0:
            raise predictor_contract.Refusal(
                f"TOO_FEW_WINDOWS: fit {kept_fit.size} inner {kept_inner.size}"
            )
        fit_targets = targets[kept_fit].astype("float32").reshape(-1, 1)
        inner_targets = targets[kept_inner].astype("float32").reshape(-1, 1)
        if not (np.all(np.isfinite(fit_targets)) and np.all(np.isfinite(inner_targets))):
            raise predictor_contract.Refusal("TARGET_NOT_FINITE_ON_FIT_OR_INNER_ROWS")
        keras = predictor_contract._keras()
        keras.utils.set_random_seed(actual_seed)
        model = _model_builder(config, arm, seed)(model_spec, len(members), "RAW", None)
        initial_sha = predictor_contract.model_weights_sha256(model)
        from tools.modular_candidate_evaluator import fit_with_early_stopping
        training = fit_with_early_stopping(
            model, fit_windows, fit_targets, inner_windows, inner_targets,
            {**config["fit"], "seed": actual_seed},
        )
        return predictor_contract.FitReport(
            spec=model_spec,
            input_mode="RAW",
            features=ordered_members,
            standardiser=standardiser,
            encoder=None,
            model=model,
            weights_sha256=predictor_contract.model_weights_sha256(model),
            initial_weights_sha256=initial_sha,
            epochs_run=int(training["epochs_completed"]),
            best_epoch=int(training["best_epoch"]),
            updates=int(training["observed_updates"]),
            fit_seconds=float(training["elapsed_seconds"]),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            n_params=predictor_contract.count_params(model),
            budget_sha256=canonical_sha256(config["fit"]),
            architecture_sha256=predictor_contract.architecture_sha256(model),
            encoder_sha256=None,
            input_identity=canonical_sha256({
                "members": ordered_members,
                "window": 24,
                "mode": "RAW",
            }),
            fit_windows=int(kept_fit.size),
            inner_windows=int(kept_inner.size),
            seed=actual_seed,
            history=training,
            timestamps=np.asarray(store.ts, dtype="int64"),
            min_timestamp=minimum,
        )
    return spec, train


def seal_weekly_cell(design, task, arm, result):
    if arm not in ARMS:
        raise ValueError("unknown I6-D arm")
    result = copy.deepcopy(result)
    result["result_sha256"] = weekly_contract.digest(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    body = {
        "schema": CELL_SCHEMA,
        "status": "COMPLETED" if result.get("disposition") == "COMPLETED" else "FAILED",
        "design_sha256": design["design_sha256"],
        "arm": arm,
        "seed": task["seed"],
        "week_ordinal": task["week"]["ordinal"],
        "task": task,
        "result": result,
        "test_read": False,
    }
    return _seal(body, "cell_sha256")


def reseal_weekly_cell(cell):
    return _seal({key: value for key, value in cell.items() if key != "cell_sha256"},
                 "cell_sha256")


def run_weekly_cell(design, arm, week_ordinal, seed, store):
    """Fully retrain one arm for one week through the established FS4/I6-A contract."""
    design = verify_weekly_design(design)
    task = make_weekly_task(design, week_ordinal, seed)
    spec, trainer = _trainer(design, arm, store, seed, task)
    result = weekly_contract.run_task(
        task, store, trainer=trainer, spec=spec, expected_seed=seed
    )
    result["trainer"] = "I6D_MATCHED_CONTROL_REAL_KERAS"
    result["architecture_arm"] = arm
    result["result_sha256"] = weekly_contract.digest(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    return seal_weekly_cell(design, task, arm, result)


def _parity_problem(left, right):
    for field in (
        "rows_sha256", "n_scored", "fit_population_digest",
        "inner_population_digest", "input_sha256", "standardiser_sha256",
        "seed", "week_start", "week_end", "fit_start", "cutoff",
    ):
        if left.get(field) != right.get(field):
            return f"ARM_PARITY_MISMATCH:{field}"
    if left.get("naive", {}).get("rows_sha256") != right.get("naive", {}).get("rows_sha256"):
        return "ARM_PARITY_MISMATCH:naive.rows_sha256"
    for metric in ("naive_mae", "naive_mse"):
        if left.get("metrics", {}).get(metric) != right.get("metrics", {}).get(metric):
            return f"ARM_PARITY_MISMATCH:metrics.{metric}"
    return None


def _annual_for_seed(records):
    total = sum(int(result["n_scored"]) for result in records)
    if total <= 0:
        raise ValueError("annual validation population is empty")
    metric = lambda name: float(sum(float(result["metrics"][name]) * int(result["n_scored"])
                                    for result in records) / total)
    mae, naive_mae = metric("mae"), metric("naive_mae")
    return {
        "n_scored": total,
        "mae": mae,
        "mse": metric("mse"),
        "naive_mae": naive_mae,
        "naive_mse": metric("naive_mse"),
        "skill_mae": (naive_mae - mae) / naive_mae if naive_mae > 0 else None,
        "beats_naive": mae < naive_mae,
        "same_row_naive": True,
        "weeks": len(records),
        "fit_seconds_total": float(sum(float(result["cost"]["fit_seconds"])
                                       for result in records)),
    }


def close_weekly_cells(design, cells):
    """Close only a complete paired annual validation population."""
    design = verify_weekly_design(design)
    indexed, problems = {}, []
    for candidate in cells:
        try:
            cell = _verify_seal(candidate, "cell_sha256")
            if (cell.get("schema") != CELL_SCHEMA
                    or cell.get("design_sha256") != design["design_sha256"]
                    or cell.get("test_read") is not False):
                raise ValueError("cell identity mismatch")
            if cell.get("arm") not in ARMS:
                raise ValueError("unknown cell arm")
            result = cell.get("result", {})
            expected_result_sha = weekly_contract.digest(
                {key: value for key, value in result.items() if key != "result_sha256"}
            )
            if result.get("result_sha256") != expected_result_sha:
                raise ValueError("result_sha256 mismatch")
            key = (cell["seed"], cell["week_ordinal"], cell["arm"])
            if key in indexed:
                raise ValueError("duplicate cell")
            expected = make_weekly_task(design, cell["week_ordinal"], cell["seed"])
            if cell["task"] != expected:
                raise ValueError("task identity mismatch")
            result_identity = {
                "split": "validation",
                "week_start": expected["week"]["start"],
                "week_end": expected["week"]["end"],
                "fit_start": expected["week"]["fit_start"],
                "cutoff": expected["week"]["cutoff"],
                "seed": expected["seed"],
            }
            if any(result.get(field) != value for field, value in result_identity.items()):
                raise ValueError("result weekly identity mismatch")
            indexed[key] = cell
        except (KeyError, TypeError, ValueError) as exc:
            problems.append({"reason": f"INVALID_CELL:{exc}"})
    expected_keys = {(seed, week, arm) for seed in design["seeds"]
                     for week in range(len(design["weeks"])) for arm in ARMS}
    for key in sorted(expected_keys - set(indexed)):
        problems.append({"seed": key[0], "week_ordinal": key[1], "arm": key[2],
                         "reason": "MISSING_CELL"})
    for seed in design["seeds"]:
        for week in range(len(design["weeks"])):
            if all((seed, week, arm) in indexed for arm in ARMS):
                dense = indexed[(seed, week, "DENSE")]["result"]
                conv = indexed[(seed, week, "CONV")]["result"]
                if dense.get("disposition") != "COMPLETED" or conv.get("disposition") != "COMPLETED":
                    problems.append({"seed": seed, "week_ordinal": week,
                                     "reason": "NON_COMPLETED_CELL"})
                else:
                    reason = _parity_problem(dense, conv)
                    if reason:
                        problems.append({"seed": seed, "week_ordinal": week,
                                         "reason": reason})
    reference_seed = design["seeds"][0]
    for seed in design["seeds"][1:]:
        for week in range(len(design["weeks"])):
            if ((reference_seed, week, "DENSE") in indexed
                    and (seed, week, "DENSE") in indexed):
                reference = indexed[(reference_seed, week, "DENSE")]["result"]
                candidate = indexed[(seed, week, "DENSE")]["result"]
                for field in ("rows_sha256", "n_scored", "input_sha256"):
                    if reference.get(field) != candidate.get(field):
                        problems.append({"seed": seed, "week_ordinal": week,
                                         "reason": f"SEED_POPULATION_MISMATCH:{field}"})
                for metric in ("naive_mae", "naive_mse"):
                    if (reference.get("metrics", {}).get(metric)
                            != candidate.get("metrics", {}).get(metric)):
                        problems.append({"seed": seed, "week_ordinal": week,
                                         "reason": f"SEED_NAIVE_MISMATCH:{metric}"})
    base = {
        "schema": CLOSURE_SCHEMA,
        "design_sha256": design["design_sha256"],
        "expected_cells": len(expected_keys),
        "verified_cells": len(indexed),
        "problems": problems,
        "test_read": False,
    }
    if problems:
        return _seal({**base, "state": "INCOMPLETE_EVIDENCE"}, "closure_sha256")
    annual_by_seed = {}
    annual = {}
    for arm in ARMS:
        per_seed = {}
        for seed in design["seeds"]:
            results = [indexed[(seed, week, arm)]["result"]
                       for week in range(len(design["weeks"]))]
            per_seed[str(seed)] = _annual_for_seed(results)
        annual_by_seed[arm] = per_seed
        annual[arm] = {
            key: float(np.mean([item[key] for item in per_seed.values()]))
            for key in ("mae", "mse", "naive_mae", "naive_mse", "skill_mae")
        }
        annual[arm].update({
            "seeds": len(per_seed),
            "n_scored_per_seed": next(iter(per_seed.values()))["n_scored"],
            "beats_naive": all(item["beats_naive"] for item in per_seed.values()),
            "same_row_naive": True,
        })
    population = [{"week_ordinal": week,
                   "rows_sha256": indexed[(design["seeds"][0], week, "DENSE")]["result"]["rows_sha256"],
                   "n_scored": indexed[(design["seeds"][0], week, "DENSE")]["result"]["n_scored"]}
                  for week in range(len(design["weeks"]))]
    body = {
        **base,
        "state": "COMPLETE",
        "annual": annual,
        "annual_by_seed": annual_by_seed,
        "decision_population": {
            "split": "validation",
            "year": design["validation_year"],
            "weeks": len(design["weeks"]),
            "rows_sha256": canonical_sha256(population),
            "weekly_rows": population,
            "naive": "same-row zero-return persistence naive",
        },
    }
    return _seal(body, "closure_sha256")


def initialize_weekly_campaign(config, output, validation_year, *, seeds=None):
    root = Path(output)
    if root.exists():
        raise ValueError("weekly campaign output already exists")
    design = build_weekly_design(config, validation_year, seeds=seeds)
    root.mkdir(parents=True)
    _atomic_json(root / "WEEKLY_DESIGN.json", design)
    _atomic_json(root / "WEEKLY_STATUS.json", weekly_status(root))
    return design


def read_weekly_design(output):
    return verify_weekly_design(json.loads((Path(output) / "WEEKLY_DESIGN.json").read_text()))


def _cell_path(root, seed, week_ordinal, arm):
    return Path(root) / "weekly_cells" / f"seed_{seed}" / f"week_{week_ordinal:03d}" / f"{arm}.json"


def weekly_status(output):
    root = Path(output)
    design = read_weekly_design(root)
    complete, invalid = 0, 0
    for seed in design["seeds"]:
        for week in range(len(design["weeks"])):
            for arm in ARMS:
                path = _cell_path(root, seed, week, arm)
                if path.exists():
                    try:
                        _verify_seal(json.loads(path.read_text()), "cell_sha256")
                        complete += 1
                    except (ValueError, json.JSONDecodeError):
                        invalid += 1
    state = "INVALID_EVIDENCE" if invalid else (
        "COMPLETE" if complete == design["expected_cells"]
        else "READY" if complete == 0 else "RUNNING"
    )
    return {"schema": STATUS_SCHEMA, "state": state, "complete_cells": complete,
            "expected_cells": design["expected_cells"], "invalid_cells": invalid,
            "pending_cells": design["expected_cells"] - complete,
            "design_sha256": design["design_sha256"], "test_read": False}


def run_weekly_cell_to_disk(output, arm, week_ordinal, seed, store):
    root = Path(output)
    design = read_weekly_design(root)
    path = _cell_path(root, seed, week_ordinal, arm)
    if path.exists():
        return _verify_seal(json.loads(path.read_text()), "cell_sha256")
    cell = run_weekly_cell(design, arm, week_ordinal, seed, store)
    _atomic_json(path, cell)
    _atomic_json(root / "WEEKLY_STATUS.json", weekly_status(root))
    return cell


def close_weekly_campaign(output):
    root = Path(output)
    design = read_weekly_design(root)
    cells = []
    for seed in design["seeds"]:
        for week in range(len(design["weeks"])):
            for arm in ARMS:
                path = _cell_path(root, seed, week, arm)
                if path.exists():
                    cells.append(json.loads(path.read_text()))
    closure = close_weekly_cells(design, cells)
    _atomic_json(root / "WEEKLY_CLOSURE.json", closure)
    _atomic_json(root / "WEEKLY_STATUS.json", weekly_status(root))
    return closure


def add_cli_commands(commands):
    plan = commands.add_parser("weekly-plan")
    plan.add_argument("--config", required=True)
    plan.add_argument("--validation-year", type=int, required=True)
    plan.add_argument("--seed", type=int, action="append")
    init = commands.add_parser("weekly-init")
    init.add_argument("--config", required=True)
    init.add_argument("--output", required=True)
    init.add_argument("--validation-year", type=int, required=True)
    init.add_argument("--seed", type=int, action="append")
    run = commands.add_parser("weekly-run-cell")
    run.add_argument("--output", required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--week-ordinal", type=int, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--feature-parquet", action="append", required=True)
    run.add_argument("--target-parquet", required=True)
    run.add_argument("--validation-feature-parquet", action="append", required=True)
    run.add_argument("--validation-target-parquet", required=True)
    status = commands.add_parser("weekly-status")
    status.add_argument("--output", required=True)
    close = commands.add_parser("weekly-close")
    close.add_argument("--output", required=True)


def run_cli(args):
    if args.command == "weekly-plan":
        return build_weekly_design(json.loads(Path(args.config).read_text()),
                                   args.validation_year, seeds=args.seed)
    if args.command == "weekly-init":
        return initialize_weekly_campaign(json.loads(Path(args.config).read_text()),
                                          args.output, args.validation_year,
                                          seeds=args.seed)
    if args.command == "weekly-status":
        return weekly_status(args.output)
    if args.command == "weekly-close":
        return close_weekly_campaign(args.output)
    design = read_weekly_design(args.output)
    store = weekly_contract.DataStore.from_paths(
        "EURUSD", args.feature_parquet, args.target_parquet,
        args.validation_feature_parquet, args.validation_target_parquet,
        bar_hours=1,
    )
    return run_weekly_cell_to_disk(args.output, args.arm, args.week_ordinal,
                                   args.seed, store)
