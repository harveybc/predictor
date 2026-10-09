"""Weekly R0/R1_B/R2_B harness over a verified I6-D branch-only design.

The harness never exposes TEST.  Each validation week is fitted on the same
rolling four-year TRAIN population used by I6-D.  R1_B and R2_B require one
sealed, exact modular branch donor for every one of the twenty selected members
and validate the complete weekly donor set before entering the fit contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource

import numpy as np

from predictor_plugins.modular_temporal import build_modular
from predictor_plugins.modular_temporal.common import weights_hash
from tools import fs4_temporal_predictor as predictor_contract
from tools import fs4_weekly_wrapper as weekly_contract
from tools import i6d_weekly_walk_forward as parent


ARMS = ("R0", "R1_B", "R2_B")
DONOR_INDEX_SCHEMA = "predictor.i7.weekly_branch_donor_index.v1"
DESIGN_SCHEMA = "predictor.i7.weekly_branch_regimes.v1"
CELL_SCHEMA = "predictor.i7.weekly_branch_regime_cell.v1"
STATUS_SCHEMA = "predictor.i7.weekly_branch_regime_status.v1"
CLOSURE_SCHEMA = "predictor.i7.weekly_branch_regime_closure.v1"


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _seal(value, field):
    body = copy.deepcopy({key: item for key, item in value.items() if key != field})
    return {**body, field: digest(body)}


def _verify_seal(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} document must be an object")
    expected = digest({key: item for key, item in value.items() if key != field})
    if value.get(field) != expected:
        raise ValueError(f"{field} mismatch")
    return value


def seal_donor_index(index):
    """Seal an index; semantic validation is intentionally performed separately."""
    return _seal(index, "index_sha256")


def verify_donor_index(index, parent_design):
    index = _verify_seal(copy.deepcopy(index), "index_sha256")
    required = {"schema", "parent_design_sha256", "members", "weeks", "index_sha256"}
    if set(index) != required or index.get("schema") != DONOR_INDEX_SCHEMA:
        raise ValueError("DONOR_INDEX_SCHEMA_MISMATCH")
    members = parent_design["members"]
    if index.get("parent_design_sha256") != parent_design["design_sha256"]:
        raise ValueError("DONOR_PARENT_DESIGN_MISMATCH")
    if index.get("members") != members or len(members) != 20 or len(set(members)) != 20:
        raise ValueError("DONOR_INDEX_REQUIRES_EXACT_PARENT_20_MEMBERS")
    if [item.get("ordinal") for item in parent_design["weeks"]] != list(
            range(len(parent_design["weeks"]))):
        raise ValueError("PARENT_WEEK_ORDINALS_MUST_BE_CONTIGUOUS")
    expected_weeks = {str(item["ordinal"]) for item in parent_design["weeks"]}
    if set(index.get("weeks", {})) != expected_weeks:
        raise ValueError("DONOR_WEEK_POPULATION_MISMATCH")
    entry_fields = {
        "week_ordinal", "member", "parent_design_sha256", "path",
        "donor_manifest_sha256", "model_sha256", "weights_sha256",
    }
    seen_paths = set()
    for week_key in sorted(expected_weeks, key=int):
        entries = index["weeks"][week_key]
        if not isinstance(entries, dict) or set(entries) != set(members):
            raise ValueError(f"DONOR_MEMBER_POPULATION_MISMATCH:{week_key}")
        for member in members:
            entry = entries[member]
            if not isinstance(entry, dict) or set(entry) != entry_fields:
                raise ValueError(f"DONOR_ENTRY_SCHEMA_MISMATCH:{week_key}:{member}")
            if (entry["week_ordinal"] != int(week_key)
                    or entry["member"] != member
                    or entry["parent_design_sha256"] != parent_design["design_sha256"]):
                raise ValueError(f"DONOR_IDENTITY_MISMATCH:{week_key}:{member}")
            if (not isinstance(entry["path"], str)
                    or not entry["path"].endswith(".keras")):
                raise ValueError(f"DONOR_PATH_INVALID:{week_key}:{member}")
            if entry["path"] in seen_paths:
                raise ValueError(f"DONOR_PATH_REUSED:{week_key}:{member}")
            seen_paths.add(entry["path"])
            for field in ("donor_manifest_sha256", "model_sha256", "weights_sha256"):
                value = entry[field]
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError(f"DONOR_DIGEST_INVALID:{week_key}:{member}:{field}")
    return index


def build_design(parent_design, donor_index):
    """Create the child campaign only from a verified branch-only robust parent."""
    parent_design = parent.verify_weekly_design(copy.deepcopy(parent_design))
    if parent.design_control_kind(parent_design) != "BRANCH_ONLY":
        raise ValueError("PARENT_MUST_BE_BRANCH_ONLY")
    if parent_design.get("target_transform") != "ROBUST_Z_FIT":
        raise ValueError("PARENT_MUST_USE_ROBUST_Z_FIT")
    if len(parent_design.get("members", [])) != 20:
        raise ValueError("PARENT_MUST_HAVE_EXACTLY_20_MEMBERS")
    donor_index = verify_donor_index(donor_index, parent_design)
    identity = {
        "schema": DESIGN_SCHEMA,
        "parent_design_sha256": parent_design["design_sha256"],
        "donor_index_sha256": donor_index["index_sha256"],
        "members": list(parent_design["members"]),
        "weeks": copy.deepcopy(parent_design["weeks"]),
        "seeds": list(parent_design["seeds"]),
        "arms": list(ARMS),
        "target_transform": "ROBUST_Z_FIT",
        "evaluation_mode": parent_design["evaluation_mode"],
        "update_mode": parent_design["update_mode"],
        "rolling_calendar_years": 4,
    }
    body = {
        **identity,
        "status": "READY",
        "procedure_sha256": digest(identity),
        "expected_cells": len(identity["weeks"]) * len(identity["seeds"]) * len(ARMS),
        "parent_design": parent_design,
        "donor_index": donor_index,
        "annual_decision_population": (
            "all finite rows from every complete validation week, paired across "
            "R0/R1_B/R2_B and against the same-row zero-return persistence naive"
        ),
        "test_paths": None,
        "test_read": False,
    }
    return _seal(body, "design_sha256")


def verify_design(design):
    design = _verify_seal(copy.deepcopy(design), "design_sha256")
    if design.get("schema") != DESIGN_SCHEMA:
        raise ValueError("I7 design schema mismatch")
    if design.get("test_paths") is not None or design.get("test_read") is not False:
        raise ValueError("TEST_IS_SEALED")
    parent_design = parent.verify_weekly_design(design.get("parent_design", {}))
    donor_index = verify_donor_index(design.get("donor_index", {}), parent_design)
    if (design.get("parent_design_sha256") != parent_design["design_sha256"]
            or design.get("donor_index_sha256") != donor_index["index_sha256"]
            or design.get("members") != parent_design["members"]
            or design.get("weeks") != parent_design["weeks"]
            or design.get("seeds") != parent_design["seeds"]
            or design.get("arms") != list(ARMS)
            or design.get("target_transform") != "ROBUST_Z_FIT"
            or design.get("rolling_calendar_years") != 4):
        raise ValueError("I7 design contradicts its parent or donor index")
    expected = len(design["weeks"]) * len(design["seeds"]) * len(ARMS)
    if design.get("expected_cells") != expected:
        raise ValueError("I7 expected cell population mismatch")
    return design


def make_task(design, week_ordinal, seed, *, split="validation"):
    design = verify_design(design)
    return _make_task(design, week_ordinal, seed, split=split)


def _make_task(design, week_ordinal, seed, *, split="validation"):
    if split != "validation":
        raise ValueError("TEST_IS_SEALED: I7 exposes validation only")
    return parent.make_weekly_task(design["parent_design"], week_ordinal, seed,
                                   split=split)


def _sidecar_for(path):
    return Path(path).with_suffix(".manifest.json")


def _preflight_week_donors(design, week_ordinal):
    """Authenticate all twenty files before Keras builds or fitting can begin."""
    entries = design["donor_index"]["weeks"][str(week_ordinal)]
    verified = {}
    for member in design["members"]:
        entry = entries[member]
        path = Path(entry["path"])
        sidecar = _sidecar_for(path)
        if not path.is_file() or not sidecar.is_file():
            raise ValueError(f"DONOR_MISSING:{week_ordinal}:{member}")
        try:
            document = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError(f"DONOR_SIDECAR_INVALID:{week_ordinal}:{member}") from exc
        if (document.get("manifest_sha256") != entry["donor_manifest_sha256"]
                or document.get("model_sha256") != entry["model_sha256"]
                or document.get("weights_sha256") != entry["weights_sha256"]
                or digest(document.get("manifest")) != entry["donor_manifest_sha256"]
                or _file_digest(path) != entry["model_sha256"]):
            raise ValueError(f"DONOR_BYTES_OR_MANIFEST_MISMATCH:{week_ordinal}:{member}")
        if (document.get("provenance") or {}).get("conditioning_contract") != "OPERATIONAL":
            raise ValueError(f"DONOR_NOT_OPERATIONAL:{week_ordinal}:{member}")
        verified[member] = copy.deepcopy(entry)
    return verified, digest(verified)


def _head_model(bundle):
    return bundle.forecast_model.get_layer("forecast_head")


def _component_hashes(bundle):
    return {
        "branches": {name: weights_hash(model)
                     for name, model in bundle.branch_models.items()},
        "core": weights_hash(bundle.core_model),
        "head": weights_hash(_head_model(bundle)),
    }


def _arm_config(design, arm, donors=None):
    config = parent._branch_only_base_config(design["parent_design"]["config"])
    config["donor_contract"] = "OPERATIONAL"
    if arm == "R0":
        return config
    regime = "R1" if arm == "R1_B" else "R2"
    expected = {
        member: [f"{member}__value", f"{member}__observed"]
        for member in design["members"]
    }
    for branch in config["branches"]:
        matches = [member for member, features in expected.items()
                   if branch["features"] == features]
        if len(matches) != 1:
            raise ValueError("BRANCH_MEMBER_MAPPING_INVALID")
        member = matches[0]
        branch.update(regime=regime, donor=donors[member]["path"])
    return config


def build_arm_bundle(design, arm, week_ordinal, seed):
    """Build one arm with donor and downstream initialization parity enforced."""
    design = verify_design(design)
    _make_task(design, week_ordinal, seed)
    if arm not in ARMS:
        raise ValueError("unknown I7 arm")
    keras = predictor_contract._keras()
    keras.utils.set_random_seed(seed)
    pristine = build_modular(_arm_config(design, "R0"))
    pristine_hashes = _component_hashes(pristine)
    if arm == "R0":
        pristine.i7_donor_set_sha256 = None
        pristine.i7_initial_hashes = pristine_hashes
        return pristine

    donors, donor_set_sha256 = _preflight_week_donors(design, week_ordinal)
    keras.utils.set_random_seed(seed)
    bundle = build_modular(_arm_config(design, arm, donors))
    bundle.core_model.set_weights(pristine.core_model.get_weights())
    _head_model(bundle).set_weights(_head_model(pristine).get_weights())
    hashes = _component_hashes(bundle)
    if hashes["core"] != pristine_hashes["core"] or hashes["head"] != pristine_hashes["head"]:
        raise ValueError("DOWNSTREAM_INITIALIZATION_PARITY_MISMATCH")
    bundle.i7_donor_set_sha256 = donor_set_sha256
    bundle.i7_initial_hashes = hashes
    return bundle


def one_optimizer_update_evidence(bundle, windows, targets, *, learning_rate=1e-3):
    """Run exactly one real optimizer update and report component movement."""
    keras = predictor_contract._keras()
    x = np.asarray(windows, dtype="float32")
    y = np.asarray(targets, dtype="float32")
    before = _component_hashes(bundle)
    branch_variables = [variable for model in bundle.branch_models.values()
                        for variable in model.trainable_variables]
    trainable = list(bundle.forecast_model.trainable_variables)
    import tensorflow as tf
    with tf.GradientTape() as tape:
        prediction = bundle.forecast_model(x, training=True)
        loss = tf.reduce_mean(tf.square(prediction - y))
    gradients = tape.gradient(loss, trainable)
    pairs = [(gradient, variable) for gradient, variable in zip(gradients, trainable)
             if gradient is not None]
    if not pairs:
        raise ValueError("NO_TRAINABLE_GRADIENTS")
    keras.optimizers.Adam(learning_rate=learning_rate).apply_gradients(pairs)
    after = _component_hashes(bundle)
    changed = sum(before["branches"][name] != after["branches"][name]
                  for name in before["branches"])
    return {
        "observed_updates": 1,
        "loss": float(loss.numpy()),
        "trainable_variable_count": len(trainable),
        "trainable_branch_variables": len(branch_variables),
        "branch_changed_count": changed,
        "branch_weights_before": before["branches"],
        "branch_weights_after": after["branches"],
        "core_weights_before": before["core"],
        "core_weights_after": after["core"],
        "head_weights_before": before["head"],
        "head_weights_after": after["head"],
    }


def _trainer(design, arm, store, seed, task):
    config = design["parent_design"]["config"]
    spec = parent._predictor_spec(config)
    members = tuple(design["members"])

    def train(model_spec, X, y, fit_idx, inner_idx, input_mode, encoder, actual_seed):
        if actual_seed != seed:
            raise predictor_contract.Refusal("I7_WEEKLY_SEED_MISMATCH")
        if input_mode != "RAW" or encoder is not None:
            raise predictor_contract.Refusal("I7_WEEKLY_RAW_ONLY")
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
        fit_targets = targets[kept_fit].astype("float64").reshape(-1, 1)
        inner_targets = targets[kept_inner].astype("float64").reshape(-1, 1)
        if not (np.all(np.isfinite(fit_targets)) and np.all(np.isfinite(inner_targets))):
            raise predictor_contract.Refusal("TARGET_NOT_FINITE_ON_FIT_OR_INNER_ROWS")
        center = float(np.median(fit_targets))
        scale = float(1.4826 * np.median(np.abs(fit_targets - center)))
        if not np.isfinite(scale) or scale <= np.finfo("float32").eps:
            scale = float(np.std(fit_targets))
        if not np.isfinite(scale) or scale <= np.finfo("float32").eps:
            raise predictor_contract.Refusal("TARGET_SCALE_DEGENERATE")
        fit_targets = ((fit_targets - center) / scale).astype("float32")
        inner_targets = ((inner_targets - center) / scale).astype("float32")

        bundle = build_arm_bundle(design, arm, task["week"]["ordinal"], seed)
        keras = predictor_contract._keras()
        scalar = keras.layers.Reshape((1,), name="weekly_scalar_target")(
            bundle.forecast_model.output
        )
        model = keras.Model(bundle.forecast_model.input, scalar,
                            name=f"i7_weekly_{arm.lower()}")
        model.fs4_architecture = digest({
            "schema": "predictor.i7.weekly_architecture.v1",
            "design_sha256": design["design_sha256"], "arm": arm,
            "seed": seed, "week_ordinal": task["week"]["ordinal"],
            "model_json": model.to_json(),
        })
        before = _component_hashes(bundle)
        initial_sha = predictor_contract.model_weights_sha256(model)
        from tools.modular_candidate_evaluator import fit_with_early_stopping
        training = fit_with_early_stopping(
            model, fit_windows, fit_targets, inner_windows, inner_targets,
            {**config["fit"], "seed": actual_seed},
        )
        after = _component_hashes(bundle)
        branch_changed = sum(before["branches"][name] != after["branches"][name]
                             for name in before["branches"])
        evidence = {
            "observed_updates": int(training["observed_updates"]),
            "trainable_variable_count": len(model.trainable_variables),
            "trainable_branch_variables": sum(
                len(branch.trainable_variables) for branch in bundle.branch_models.values()
            ),
            "branch_changed_count": branch_changed,
            "branch_weights_before": before["branches"],
            "branch_weights_after": after["branches"],
            "core_weights_before": before["core"],
            "core_weights_after": after["core"],
            "head_weights_before": before["head"],
            "head_weights_after": after["head"],
            "donor_set_sha256": bundle.i7_donor_set_sha256,
        }
        if arm == "R1_B" and branch_changed != 0:
            raise predictor_contract.Refusal("R1_B_BRANCH_WEIGHTS_MOVED")
        if arm == "R2_B" and training["observed_updates"] > 0 and branch_changed == 0:
            raise predictor_contract.Refusal("R2_B_BRANCH_WEIGHTS_DID_NOT_MOVE")
        train.regime_evidence = evidence
        train.target_transform_metadata = {
            "method": "ROBUST_Z_FIT", "center": center, "scale": scale,
            "fit_rows_sha256": digest(kept_fit.tolist()),
        }
        return predictor_contract.FitReport(
            spec=model_spec, input_mode="RAW", features=ordered_members,
            standardiser=standardiser, encoder=None, model=model,
            weights_sha256=predictor_contract.model_weights_sha256(model),
            initial_weights_sha256=initial_sha,
            epochs_run=int(training["epochs_completed"]),
            best_epoch=int(training["best_epoch"]),
            updates=int(training["observed_updates"]),
            fit_seconds=float(training["elapsed_seconds"]),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            n_params=predictor_contract.count_params(model),
            budget_sha256=digest(config["fit"]),
            architecture_sha256=predictor_contract.architecture_sha256(model),
            encoder_sha256=None,
            input_identity=digest({"members": ordered_members, "window": 24, "mode": "RAW"}),
            fit_windows=int(kept_fit.size), inner_windows=int(kept_inner.size),
            seed=actual_seed, history=training,
            timestamps=np.asarray(store.ts, dtype="int64"), min_timestamp=minimum,
            target_center=center, target_scale=scale,
        )

    train.regime_evidence = None
    train.target_transform_metadata = None
    return spec, train


def seal_cell(design, task, arm, result):
    if arm not in ARMS:
        raise ValueError("unknown I7 arm")
    result = copy.deepcopy(result)
    result["result_sha256"] = digest(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
    body = {
        "schema": CELL_SCHEMA,
        "status": "COMPLETED" if result.get("disposition") == "COMPLETED" else "FAILED",
        "design_sha256": design["design_sha256"],
        "arm": arm,
        "seed": task["seed"],
        "week_ordinal": task["week"]["ordinal"],
        "task": copy.deepcopy(task),
        "result": result,
        "test_read": False,
    }
    return _seal(body, "cell_sha256")


def reseal_cell(cell):
    return _seal(cell, "cell_sha256")


def run_cell(design, arm, week_ordinal, seed, store):
    """Run one cell; donor failures occur before the weekly fit function is called."""
    design = verify_design(design)
    task = _make_task(design, week_ordinal, seed)
    if arm not in ARMS:
        raise ValueError("unknown I7 arm")
    if arm != "R0":
        _preflight_week_donors(design, week_ordinal)
    spec, trainer = _trainer(design, arm, store, seed, task)
    result = weekly_contract.run_task(
        task, store, trainer=trainer, spec=spec, expected_seed=seed
    )
    if trainer.regime_evidence is None or trainer.target_transform_metadata is None:
        raise ValueError("trainer did not produce regime evidence")
    result.update({
        "trainer": "I7_WEEKLY_BRANCH_REGIMES_REAL_KERAS",
        "architecture_arm": arm,
        "regime_evidence": trainer.regime_evidence,
        "target_transform": trainer.target_transform_metadata,
    })
    return seal_cell(design, task, arm, result)


def _validate_cell(design, candidate):
    cell = _verify_seal(copy.deepcopy(candidate), "cell_sha256")
    if (cell.get("schema") != CELL_SCHEMA
            or cell.get("design_sha256") != design["design_sha256"]
            or cell.get("arm") not in ARMS
            or cell.get("test_read") is not False):
        raise ValueError("cell identity mismatch")
    expected_task = _make_task(design, cell["week_ordinal"], cell["seed"])
    if cell.get("task") != expected_task:
        raise ValueError("task identity mismatch")
    result = cell.get("result", {})
    if result.get("result_sha256") != digest(
            {key: value for key, value in result.items() if key != "result_sha256"}):
        raise ValueError("result_sha256 mismatch")
    required_identity = {
        "split": "validation", "week_start": expected_task["week"]["start"],
        "week_end": expected_task["week"]["end"],
        "fit_start": expected_task["week"]["fit_start"],
        "cutoff": expected_task["week"]["cutoff"], "seed": expected_task["seed"],
    }
    if any(result.get(key) != value for key, value in required_identity.items()):
        raise ValueError("result weekly identity mismatch")
    metrics = result.get("metrics", {})
    if any(not np.isfinite(metrics.get(key, np.nan))
           for key in ("mae", "mse", "naive_mae", "naive_mse")):
        raise ValueError("metrics must be finite")
    evidence = result.get("regime_evidence", {})
    required_evidence = {
        "observed_updates", "trainable_variable_count", "trainable_branch_variables",
        "branch_changed_count", "donor_set_sha256", "branch_weights_before",
        "branch_weights_after", "core_weights_before", "core_weights_after",
        "head_weights_before", "head_weights_after",
    }
    if not required_evidence <= set(evidence):
        raise ValueError("regime evidence incomplete")
    branch_names = {f"branch_{index:03d}" for index in range(20)}
    before, after = evidence["branch_weights_before"], evidence["branch_weights_after"]
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != branch_names or set(after) != branch_names:
        raise ValueError("branch weight evidence population mismatch")
    measured_changed = sum(before[name] != after[name] for name in branch_names)
    if evidence["branch_changed_count"] != measured_changed:
        raise ValueError("branch changed count contradicts weight hashes")
    expected_donor_set = None if cell["arm"] == "R0" else digest(
        design["donor_index"]["weeks"][str(cell["week_ordinal"])]
    )
    if evidence["donor_set_sha256"] != expected_donor_set:
        raise ValueError("weekly donor set identity mismatch")
    if cell["arm"] == "R1_B" and (evidence["trainable_branch_variables"] != 0
                                   or evidence["branch_changed_count"] != 0):
        raise ValueError("R1_B frozen-branch evidence mismatch")
    if cell["arm"] == "R2_B" and (evidence["trainable_branch_variables"] <= 0
                                   or evidence["observed_updates"] <= 0
                                   or evidence["branch_changed_count"] <= 0):
        raise ValueError("R2_B trainable-branch evidence mismatch")
    return cell


def _parity_problem(reference, candidate):
    for field in (
        "rows_sha256", "n_scored", "fit_population_digest",
        "inner_population_digest", "input_sha256", "standardiser_sha256",
        "seed", "week_start", "week_end", "fit_start", "cutoff",
    ):
        if reference.get(field) != candidate.get(field):
            return f"ARM_PARITY_MISMATCH:{field}"
    if reference.get("naive", {}).get("rows_sha256") != candidate.get("naive", {}).get("rows_sha256"):
        return "ARM_PARITY_MISMATCH:naive.rows_sha256"
    for metric in ("naive_mae", "naive_mse"):
        if reference["metrics"].get(metric) != candidate["metrics"].get(metric):
            return f"ARM_PARITY_MISMATCH:metrics.{metric}"
    if reference.get("target_transform") != candidate.get("target_transform"):
        return "ARM_PARITY_MISMATCH:target_transform"
    return None


def _initialization_parity_problem(reference, candidate):
    left = reference["regime_evidence"]
    right = candidate["regime_evidence"]
    for field in ("core_weights_before", "head_weights_before"):
        if left[field] != right[field]:
            return f"INITIALIZATION_PARITY_MISMATCH:{field}"
    return None


def _annual(records):
    total = sum(int(result["n_scored"]) for result in records)
    if total <= 0:
        raise ValueError("annual validation population is empty")
    def weighted(name):
        return float(sum(float(result["metrics"][name]) * int(result["n_scored"])
                         for result in records) / total)
    mae, mse = weighted("mae"), weighted("mse")
    naive_mae, naive_mse = weighted("naive_mae"), weighted("naive_mse")
    return {
        "n_scored": total, "weeks": len(records), "mae": mae, "mse": mse,
        "naive_mae": naive_mae, "naive_mse": naive_mse,
        "skill_mae": (naive_mae - mae) / naive_mae if naive_mae > 0 else None,
        "skill_mse": (naive_mse - mse) / naive_mse if naive_mse > 0 else None,
        "beats_naive": mae < naive_mae, "same_row_naive": True,
        "fit_seconds_total": float(sum(float(item["cost"]["fit_seconds"])
                                       for item in records)),
    }


def close_cells(design, cells):
    """Close only the complete, row-paired annual arm population."""
    design = verify_design(design)
    indexed, problems = {}, []
    for candidate in cells:
        try:
            cell = _validate_cell(design, candidate)
            key = (cell["seed"], cell["week_ordinal"], cell["arm"])
            if key in indexed:
                raise ValueError("duplicate cell")
            indexed[key] = cell
        except (KeyError, TypeError, ValueError) as exc:
            problems.append({"reason": f"INVALID_CELL:{exc}"})
    expected = {(seed, week, arm) for seed in design["seeds"]
                for week in range(len(design["weeks"])) for arm in ARMS}
    for seed, week, arm in sorted(expected - set(indexed)):
        problems.append({"seed": seed, "week_ordinal": week, "arm": arm,
                         "reason": "MISSING_CELL"})
    for seed in design["seeds"]:
        for week in range(len(design["weeks"])):
            if not all((seed, week, arm) in indexed for arm in ARMS):
                continue
            reference = indexed[(seed, week, "R0")]["result"]
            for arm in ARMS:
                result = indexed[(seed, week, arm)]["result"]
                if result.get("disposition") != "COMPLETED":
                    problems.append({"seed": seed, "week_ordinal": week, "arm": arm,
                                     "reason": "NON_COMPLETED_CELL"})
                elif arm != "R0":
                    reason = _parity_problem(reference, result)
                    if reason:
                        problems.append({"seed": seed, "week_ordinal": week,
                                         "arm": arm, "reason": reason})
                    init_reason = _initialization_parity_problem(reference, result)
                    if init_reason:
                        problems.append({"seed": seed, "week_ordinal": week,
                                         "arm": arm, "reason": init_reason})
            if all((seed, week, arm) in indexed for arm in ("R1_B", "R2_B")):
                r1 = indexed[(seed, week, "R1_B")]["result"]["regime_evidence"]
                r2 = indexed[(seed, week, "R2_B")]["result"]["regime_evidence"]
                if r1["branch_weights_before"] != r2["branch_weights_before"]:
                    problems.append({"seed": seed, "week_ordinal": week,
                                     "reason": "INITIALIZATION_PARITY_MISMATCH:donor_branches"})
    reference_seed = design["seeds"][0]
    for seed in design["seeds"][1:]:
        for week in range(len(design["weeks"])):
            reference_key = (reference_seed, week, "R0")
            candidate_key = (seed, week, "R0")
            if reference_key not in indexed or candidate_key not in indexed:
                continue
            reference = indexed[reference_key]["result"]
            candidate = indexed[candidate_key]["result"]
            for field in ("rows_sha256", "n_scored", "input_sha256"):
                if reference.get(field) != candidate.get(field):
                    problems.append({"seed": seed, "week_ordinal": week,
                                     "reason": f"SEED_POPULATION_MISMATCH:{field}"})
            for metric in ("naive_mae", "naive_mse"):
                if reference["metrics"].get(metric) != candidate["metrics"].get(metric):
                    problems.append({"seed": seed, "week_ordinal": week,
                                     "reason": f"SEED_NAIVE_MISMATCH:{metric}"})
    base = {
        "schema": CLOSURE_SCHEMA, "design_sha256": design["design_sha256"],
        "expected_cells": len(expected), "verified_cells": len(indexed),
        "problems": problems, "test_read": False,
    }
    if problems:
        return _seal({**base, "state": "INCOMPLETE_EVIDENCE"}, "closure_sha256")

    by_seed, annual = {}, {}
    for arm in ARMS:
        per_seed = {
            str(seed): _annual([indexed[(seed, week, arm)]["result"]
                                for week in range(len(design["weeks"]))])
            for seed in design["seeds"]
        }
        by_seed[arm] = per_seed
        annual[arm] = {}
        for key in ("mae", "mse", "naive_mae", "naive_mse", "skill_mae", "skill_mse"):
            values = [value[key] for value in per_seed.values()]
            annual[arm][key] = None if any(value is None for value in values) else float(np.mean(values))
        annual[arm].update({
            "seeds": len(per_seed),
            "n_scored_per_seed": next(iter(per_seed.values()))["n_scored"],
            "beats_naive": all(value["beats_naive"] for value in per_seed.values()),
            "same_row_naive": True,
        })
    deltas = {
        arm: {
            metric: (None if annual[arm][metric] is None or annual["R0"][metric] is None
                     else annual[arm][metric] - annual["R0"][metric])
            for metric in ("mae", "mse", "skill_mae", "skill_mse")
        }
        for arm in ("R1_B", "R2_B")
    }
    regime_evidence = {}
    for arm in ARMS:
        evidence = [indexed[key]["result"]["regime_evidence"]
                    for key in sorted(indexed) if key[2] == arm]
        regime_evidence[arm] = {
            "observed_updates": sum(int(item["observed_updates"]) for item in evidence),
            "trainable_variable_count_min": min(int(item["trainable_variable_count"])
                                                for item in evidence),
            "trainable_branch_variables_min": min(int(item["trainable_branch_variables"])
                                                  for item in evidence),
            "branch_changed_count": sum(int(item["branch_changed_count"])
                                        for item in evidence),
            "donor_sets_sha256": sorted({item["donor_set_sha256"] for item in evidence
                                          if item["donor_set_sha256"]}),
        }
    population = [
        {"week_ordinal": week,
         "rows_sha256": indexed[(design["seeds"][0], week, "R0")]["result"]["rows_sha256"],
         "n_scored": indexed[(design["seeds"][0], week, "R0")]["result"]["n_scored"]}
        for week in range(len(design["weeks"]))
    ]
    return _seal({
        **base, "state": "COMPLETE", "annual": annual, "annual_by_seed": by_seed,
        "paired_deltas_to_R0": deltas, "regime_evidence": regime_evidence,
        "decision_population": {
            "split": "validation", "year": design["parent_design"]["validation_year"],
            "weeks": len(design["weeks"]), "weekly_rows": population,
            "rows_sha256": digest(population),
            "naive": "same-row zero-return persistence naive",
        },
    }, "closure_sha256")


def initialize_campaign(parent_design_path, donor_index_path, output):
    root = Path(output)
    if root.exists():
        raise ValueError("I7 campaign output already exists")
    parent_design = json.loads(Path(parent_design_path).read_text(encoding="utf-8"))
    donor_index = json.loads(Path(donor_index_path).read_text(encoding="utf-8"))
    design = build_design(parent_design, donor_index)
    root.mkdir(parents=True)
    _atomic_json(root / "I7_DESIGN.json", design)
    _atomic_json(root / "I7_STATUS.json", campaign_status(root))
    return design


def read_design(output):
    return verify_design(json.loads(
        (Path(output) / "I7_DESIGN.json").read_text(encoding="utf-8")
    ))


def _cell_path(root, seed, week_ordinal, arm):
    return (Path(root) / "cells" / f"seed_{seed}" /
            f"week_{week_ordinal:03d}" / f"{arm}.json")


def campaign_status(output):
    root = Path(output)
    design = read_design(root)
    complete, invalid = 0, 0
    for seed in design["seeds"]:
        for week in range(len(design["weeks"])):
            for arm in ARMS:
                path = _cell_path(root, seed, week, arm)
                if path.exists():
                    try:
                        _validate_cell(design, json.loads(path.read_text(encoding="utf-8")))
                        complete += 1
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                        invalid += 1
    state = "INVALID_EVIDENCE" if invalid else (
        "COMPLETE" if complete == design["expected_cells"]
        else "READY" if complete == 0 else "RUNNING"
    )
    return {
        "schema": STATUS_SCHEMA, "state": state, "complete_cells": complete,
        "expected_cells": design["expected_cells"], "invalid_cells": invalid,
        "pending_cells": design["expected_cells"] - complete,
        "design_sha256": design["design_sha256"], "test_read": False,
    }


def run_cell_to_disk(output, arm, week_ordinal, seed, store):
    root = Path(output)
    design = read_design(root)
    path = _cell_path(root, seed, week_ordinal, arm)
    if path.exists():
        return _validate_cell(design, json.loads(path.read_text(encoding="utf-8")))
    cell = run_cell(design, arm, week_ordinal, seed, store)
    _atomic_json(path, cell)
    _atomic_json(root / "I7_STATUS.json", campaign_status(root))
    return cell


def close_campaign(output):
    root = Path(output)
    design = read_design(root)
    cells = []
    for seed in design["seeds"]:
        for week in range(len(design["weeks"])):
            for arm in ARMS:
                path = _cell_path(root, seed, week, arm)
                if path.exists():
                    cells.append(json.loads(path.read_text(encoding="utf-8")))
    closure = close_cells(design, cells)
    _atomic_json(root / "I7_CLOSURE.json", closure)
    _atomic_json(root / "I7_STATUS.json", campaign_status(root))
    return closure


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--parent-design", required=True)
    init.add_argument("--donor-index", required=True)
    init.add_argument("--output", required=True)
    run = commands.add_parser("run-cell")
    run.add_argument("--output", required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--week-ordinal", type=int, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--feature-parquet", action="append", required=True)
    run.add_argument("--target-parquet", required=True)
    run.add_argument("--validation-feature-parquet", action="append", required=True)
    run.add_argument("--validation-target-parquet", required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", required=True)
    close = commands.add_parser("close")
    close.add_argument("--output", required=True)
    return parser


def run_cli(args):
    if args.command == "init":
        return initialize_campaign(args.parent_design, args.donor_index, args.output)
    if args.command == "status":
        return campaign_status(args.output)
    if args.command == "close":
        return close_campaign(args.output)
    store = weekly_contract.DataStore.from_paths(
        "EURUSD", args.feature_parquet, args.target_parquet,
        args.validation_feature_parquet, args.validation_target_parquet,
        bar_hours=1,
    )
    return run_cell_to_disk(args.output, args.arm, args.week_ordinal, args.seed, store)


def main(argv=None):
    result = run_cli(_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
