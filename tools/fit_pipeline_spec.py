#!/usr/bin/env python
"""Fit one ``m5phet.pipeline.v1`` spec on one CSV and score it on a holdout sealed BEFORE the fit.

WP18 step 7 / WP06 stages 3-4 of the M5PHET work plan (2026-09-24, revision 2). Steps 1-6 produced a *configuration*
chosen by Laya and never fitted; this tool is the step that turns such a configuration into (a) a trained graph with the
artifacts an export needs and (b) an evaluation report in the exact shape ``M5PHET/evaluation`` reads
(``m5phet-evaluation-report/1``), so ``evaluation/compare_stages.py`` can build the owner's closure table from artifacts
instead of from anybody's typing.

What this tool is, and what it refuses to be:

* it reads a spec and builds the predictor configuration the spec *implies*. It never chooses anything the spec did not
  declare: an encoder the spec does not name, a core that is not in the spec, a window that is not in its
  representation. Anything it cannot honour is written into the run manifest by name, not silently replaced;
* **the holdout is sealed first.** The population is fixed before a single weight is fitted, its digest is computed from
  the rows and their labels, and every stage run against the same CSV, holdout fraction, sealing window and horizon
  recomputes the identical seal. A stage whose seal differs is not comparable and ``compare_stages`` says so;
* the naive reference (last observed value of the target at the origin) is computed by the evaluation package on
  **exactly** the sealed rows, never on a convenient subset;
* it never writes a quality claim anywhere but the report, and the report carries the protocol digest, the seal and the
  counts that every ratio in it rests on.

The population deserves its own sentence. The sealed rows are the forecast **origins** inside the holdout for which the
whole *sealing window* of history and the whole horizon of future lie inside the holdout. That is stricter than each
model needs -- a 60-step model could start earlier -- and it is deliberate: two stages with different windows must be
scored on the same rows or their errors are two different questions. ``--seal-window`` is therefore declared once for a
comparison and passed to every stage of it.

Per-feature preprocessing. A spec names one preprocessor per feature, chosen out of the registered
``preprocessor.plugins`` entry points of ``predictor`` and of the standalone ``preprocessor`` application. Those plugins
are whole-dataset pipeline stages (they trim, split a dataset into D1-D6 and write normalisation files for a declared
column order), not per-feature transforms, and this harness has no place to apply one to a single column. So the run
manifest records, per feature, ``preprocessing_applied: DECLARED_NOT_APPLIED`` with the plugin the spec named and the
reason, and the model is fitted on the standard scaling every bundle in this stack uses: per-column z-score with mean
and standard deviation fitted on the TRAIN rows only. That is a statement about this harness, not a claim that the
spec's choice was wrong or right.

Usage (one invocation per stage; the same ``--seal-window``, ``--sealed-at``, ``--epochs`` and ``--patience`` for all of
them, or the stages are not comparable):

    python tools/fit_pipeline_spec.py --spec spec.json --data slice.csv --stage laya_chosen \\
        --seal-window 197 --horizon 60 --epochs 30 --patience 5 --sealed-at 2026-09-25T00:00:00Z \\
        --out-dir /path/to/stage --evaluation-src /path/to/M5PHET/evaluation/src
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PIPELINE_SCHEMA = "m5phet.pipeline.v1"
RUN_SCHEMA = "predictor.fitted_forecast.v1"

#: what the run manifest says about a preprocessor the spec named and this harness did not apply
DECLARED_NOT_APPLIED = "DECLARED_NOT_APPLIED"

#: the reason, once, so it cannot be re-explained more softly per feature
PREPROCESSOR_NOT_PER_FEATURE = (
    "the registered preprocessor plugins of this stack are whole-dataset pipeline stages (they trim a dataset, split it "
    "into D1-D6 and write normalisation files for a declared column order); none of them exposes a per-feature "
    "transform, and this harness has no place to apply one to a single column. The feature is fitted on the standard "
    "scaling every bundle here uses instead: per-column z-score with mean and standard deviation fitted on the TRAIN "
    "rows only")


class SpecError(ValueError):
    """The spec does not declare something this tool refuses to invent."""


# --------------------------------------------------------------------------------------------------- plugin resolution

def declared_plugins(repo_root: Path) -> dict:
    """The ``predictor.plugins`` entry points as ``setup.py`` declares them.

    Resolved from the checkout's own declaration rather than from whatever is installed in the interpreter, because a
    stage must be able to say which module it fitted; an entry point resolved out of another checkout would put a
    different graph behind the same name.
    """
    text = (repo_root / "setup.py").read_text()
    block = re.search(r"'predictor\.plugins'\s*:\s*\[(.*?)\]", text, re.S)
    if block is None:
        raise SpecError("setup.py declares no 'predictor.plugins' entry-point group")
    found = {}
    for name, module, attribute in re.findall(r"'([\w.]+)=([\w.]+):(\w+)'", block.group(1)):
        found[name] = (module, attribute)
    return found


def load_plugin_class(repo_root: Path, key: str):
    import importlib

    declared = declared_plugins(repo_root)
    if key not in declared:
        raise SpecError(f"UNKNOWN_CORE: {key!r} is not a declared predictor.plugins entry point; "
                        f"declared: {sorted(declared)}")
    module_name, attribute = declared[key]
    module = importlib.import_module(module_name)
    return getattr(module, attribute), f"{module_name}:{attribute}"


# ------------------------------------------------------------------------------------------------------- the spec read

def read_spec(path: Path) -> dict:
    spec = json.loads(path.read_text())
    if spec.get("schema") != PIPELINE_SCHEMA:
        raise SpecError(f"{path.name}: schema {spec.get('schema')!r} is not {PIPELINE_SCHEMA!r}")
    for key in ("representation", "grouping", "core", "features"):
        if key not in spec:
            raise SpecError(f"{path.name}: a {PIPELINE_SCHEMA} spec declares {key!r}")
    return spec


def branches_from_spec(spec: dict) -> list:
    """One branch per declared group, with the encoder the spec's core mapping gives that group.

    A group whose encoder the mapping could not resolve (``status`` other than ``MAPPED``, or a null family) is refused
    by name: fitting it with some default encoder would publish a graph nobody chose under the name of one they did.
    """
    mapping = ((spec.get("core") or {}).get("encoder_mapping") or {}).get("branches") or {}
    branches = []
    for group in spec["grouping"]["groups"]:
        gid = group["group_id"]
        entry = mapping.get(gid)
        if not isinstance(entry, dict) or entry.get("status") != "MAPPED" or not entry.get("encoder"):
            raise SpecError(f"BRANCH_ENCODER_NOT_MAPPED: group {gid!r} has no mapped encoder in the spec "
                            f"({entry!r}); this tool does not choose one")
        branches.append({"name": gid, "columns": list(group["members"]), "encoder": str(entry["encoder"]),
                         "declared_extractor": entry.get("extractor")})
    if not branches:
        raise SpecError("NO_BRANCHES: the spec declares no feature group")
    return branches


def preprocessing_record(spec: dict, features) -> dict:
    """Per feature: the plugin the spec named, and whether this harness applied it. Never silently dropped."""
    declared = spec.get("preprocessing") or {}
    record = {}
    for feature in features:
        entry = declared.get(feature) or {}
        plugin = entry.get("plugin")
        record[feature] = {
            "declared_plugin": plugin,
            "decision": entry.get("decision"),
            "preprocessing_applied": DECLARED_NOT_APPLIED if plugin else "NONE_DECLARED",
            "why": PREPROCESSOR_NOT_PER_FEATURE if plugin else "the spec names no preprocessor for this feature",
            "applied_instead": "train_only_per_column_zscore",
        }
    return record


# ------------------------------------------------------------------------------------------------------------ the data

def read_csv(path: Path):
    """Timestamps, the feature matrix in the file's own column order, and the finite-row mask."""
    import pandas as pd

    frame = pd.read_csv(path)
    time_column = frame.columns[0]
    stamps = pd.to_datetime(frame[time_column], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if stamps.isna().any():
        stamps = pd.to_datetime(frame[time_column], errors="coerce")
    if stamps.isna().any():
        raise SpecError(f"{path.name}: {int(stamps.isna().sum())} timestamps could not be parsed")
    columns = [c for c in frame.columns if c != time_column]
    values = frame[columns].to_numpy(dtype=np.float64)
    # measured, never assumed, and read through pandas' own accessor: the underlying resolution of a datetime column is
    # not fixed across pandas versions, so an integer view of it would silently publish nanoseconds as seconds
    steps = stamps.diff().dropna().dt.total_seconds().to_numpy()
    if steps.size == 0 or not np.isfinite(steps).all() or (steps <= 0).any():
        raise SpecError(f"{path.name}: the timestamps are not strictly increasing; the grid cannot be measured")
    finite = np.isfinite(values).all(axis=1)
    return {"time_column": time_column, "stamps": stamps, "columns": columns, "values": values,
            "finite": finite, "steps": steps}


def sealed_population(data, *, holdout_fraction: float, seal_window: int, horizon: int):
    """The forecast origins inside the holdout whose whole sealing window and whole horizon lie inside the holdout.

    Identities are the origin timestamps in ISO form: readable in a refusal, and stable across two runs of this tool.
    """
    n = len(data["values"])
    start = int(round(n * (1.0 - holdout_fraction)))
    first = start + seal_window - 1
    last = n - horizon - 1
    if first > last:
        raise SpecError(f"the holdout holds no origin with {seal_window} history rows and {horizon} future rows")
    target_index = data["target_index"]
    rows, labels, origins = [], {}, []
    for origin in range(first, last + 1):
        if not data["finite"][origin - seal_window + 1:origin + horizon + 1].all():
            continue           # a window or a label with a non-finite value is dropped, and counted in the manifest
        identity = data["stamps"].iloc[origin].isoformat()
        rows.append(identity)
        labels[identity] = float(data["values"][origin + horizon, target_index])
        origins.append(origin)
    return {"holdout_start": start, "rows": tuple(rows), "labels": labels, "origins": np.asarray(origins, dtype=int),
            "dropped": (last - first + 1) - len(rows)}


def training_origins(data, *, holdout_start: int, window: int, horizon: int):
    """Origins whose window and label lie entirely before the holdout. Nothing from the holdout is ever fitted on."""
    first = window - 1
    last = holdout_start - horizon - 1
    out = [origin for origin in range(first, last + 1)
           if data["finite"][origin - window + 1:origin + horizon + 1].all()]
    return np.asarray(out, dtype=int)


def windows_for(values, origins, window: int):
    """(n, window, channels) float32, gathered per origin. The scaled matrix is materialised once, the windows once."""
    out = np.empty((len(origins), window, values.shape[1]), dtype=np.float32)
    for position, origin in enumerate(origins):
        out[position] = values[origin - window + 1:origin + 1]
    return out


# ------------------------------------------------------------------------------------------------------------ the fit

def build_config(spec: dict, *, columns, horizon: int, epochs: int, patience: int, batch_size: int,
                 seed: int, head: str, quantiles) -> dict:
    """The predictor configuration this spec implies. Every value here comes from the spec or from a declared flag."""
    representation = spec["representation"]
    windows = [int(value) for value in (representation.get("windows") or [])]
    if not windows:
        raise SpecError("the representation declares no window")
    # The LONGEST window the representation declares is the memory the candidate claims, and it is the one fitted here.
    # A design candidate may name several -- WP06's `seasonal_lag_1443` declares [197, 1443], the seasonal peak with the
    # decay window kept beside it -- and this harness builds ONE input block, so taking the first would silently fit the
    # shorter memory and publish three different candidates as the same graph. What is not fitted is declared: every
    # window the spec named is written into the run manifest beside the one used.
    window = max(windows)
    core = (spec.get("core") or {}).get("key")
    if not core:
        raise SpecError("the spec declares no core")
    # a core is handed branches only when it DECLARES that it takes them; a core whose plugin_params has no `branches`
    # key is a single-window core, and the spec's grouping is recorded as not applied rather than silently ignored
    plugin_class, _ = load_plugin_class(REPO_ROOT, core)
    takes_branches = "branches" in getattr(plugin_class, "plugin_params", {})
    branches = branches_from_spec(spec) if takes_branches else []
    target = representation["target"]["column"]
    if target not in columns:
        raise SpecError(f"the target {target!r} is not a column of the data")
    # A representation may declare derived or calendar features by name. This harness fits the columns of the sealed
    # dataset and builds none of them: building one would produce a different file, and the holdout identity -- the
    # protocol digest and the corpus seal -- is computed over the file the population was sealed from. So a declared
    # feature that is not already a column is refused by name instead of being quietly dropped.
    declared_features = [name for name in (representation.get("features") or ()) if name not in columns]
    if declared_features:
        raise SpecError(f"FEATURE_NOT_IN_DATA: the representation declares {declared_features}, which the sealed "
                        f"dataset does not carry; this harness builds no feature, and adding a column would change "
                        f"the file the holdout was sealed from")
    config = {
        "predictor_plugin": core,
        "target_column": target,
        "window_size": window,
        "predicted_horizons": [horizon],
        "plotted_horizon": horizon,
        "feature_names": list(columns),
        "encoder_units": 32,
        "encoder_layers": 1,
        "encoder_kernel_size": 3,
        "head_units": [64],
        "head_activation": "relu",
        "activation": "relu",
        "dropout_rate": 0.0,
        # The Bayesian head is turned OFF for both stages of this comparison, and the reason is declared rather than
        # assumed: a flipout head samples its weights at inference, so the same window would produce a different number
        # on every call and neither the score nor the exported graph would be reproducible. The uncertainty of this
        # comparison is the quantile head of WP07, not MC noise.
        "bayesian_head": False,
        "learning_rate": 0.001,
        "batch_size": batch_size,
        "early_patience": patience,
        "epochs": epochs,
        "seed": seed,
        "quiet": True,
        "disable_postfit_uncertainty": True,
    }
    if takes_branches:
        config["branches"] = [{k: v for k, v in branch.items() if k != "declared_extractor"} for branch in branches]
        config["fusion"] = "concat"
        config["grouping_applied"] = "APPLIED"
    else:
        config["grouping_applied"] = "NOT_APPLICABLE_SINGLE_WINDOW_CORE"
    if head == "quantile":
        config["quantiles"] = list(quantiles)
    return config


#: why the plugin's own `train` is not the loop used here
SINGLE_OUTPUT_LOOP = (
    "keras `fit` driven by this tool with the plugin's OWN callbacks (`BaseKerasPredictor._build_callbacks`: early "
    "stopping on val_loss with restore_best_weights, and ReduceLROnPlateau). `BaseKerasPredictor.train` takes a dict of "
    "per-horizon targets, and Keras 3 does not unpack a dict for a single-output graph; this comparison declares one "
    "horizon, so the dict path is not available. Nothing about the optimiser, the loss or the callbacks differs")


def fit_stage(config, x_train, y_train, x_val, y_val, *, repo_root: Path, epochs: int, head: str):
    plugin_class, module_ref = load_plugin_class(repo_root, config["predictor_plugin"])
    plugin = plugin_class(dict(config))
    output_name = f"output_horizon_{config['predicted_horizons'][0]}"
    plugin.build_model((config["window_size"], x_train.shape[2]), x_train, dict(config))
    if plugin.output_names != [output_name]:
        raise SpecError(f"the plugin declares outputs {plugin.output_names}, not [{output_name!r}]")
    history = plugin.model.fit(x_train, y_train, epochs=epochs, batch_size=config["batch_size"],
                               validation_data=(x_val, y_val), callbacks=plugin._build_callbacks(),
                               verbose=2, shuffle=False)
    epochs_run = len(history.history.get("loss", []))
    return plugin, module_ref, epochs_run, {k: [float(v) for v in vals] for k, vals in history.history.items()}


# ------------------------------------------------------------------------------------------------------- the reporting

def build_protocol_and_seal(evaluation_src: Path, population, labels, *, data_path: Path, sealed_at: str,
                            seal_window: int, horizon: int, holdout_fraction: float, minimum_rows: int):
    if str(evaluation_src) not in sys.path:
        sys.path.insert(0, str(evaluation_src))
    from m5phet_evaluation import EvaluationProtocol, seal_corpus

    protocol = EvaluationProtocol(
        family="forecast",
        population=population,
        label_source=f"{data_path.name} (household DEV slice, 1-minute grid, original units)",
        label_producer=("the electric meter record of the household power dataset, as sliced by the governed resource "
                        "e1_household_dev_pilot_v1 of the CRISP-DM data foundation"),
        label_provenance="REALISED_OUTCOME",
        annotation_rules=(
            f"No row is annotated. The label of an origin is the realised value of the target column {horizon} steps "
            f"after it, read from the same file.",),
        ambiguity_adjudication=("No row is ambiguous: a realised measurement has one value, and rows whose window or "
                                "label carried a non-finite value were dropped from the population before sealing."),
        split={"test": population},
        split_frozen_at=sealed_at,
        split_frozen_by=(f"the last {holdout_fraction:.0%} of the file by time, restricted to origins whose "
                         f"{seal_window}-row sealing window and {horizon}-row horizon both lie inside it; fixed before "
                         f"any weight was fitted"),
        metrics=("mae", "rmse", "skill_mae", "skill_rmse"),
        baseline="last_value",
        minimum_rows=minimum_rows,
    )
    seal = seal_corpus(labels, protocol=protocol, sealed_at=sealed_at)
    return protocol, seal


def score_and_report(evaluation_src: Path, protocol, seal, *, truth, predictions, baseline, extra_sets=()):
    if str(evaluation_src) not in sys.path:
        sys.path.insert(0, str(evaluation_src))
    from m5phet_evaluation import build_report, score_forecast

    metrics = score_forecast(protocol=protocol, seal=seal, truth=truth, predictions=predictions,
                             baseline_predictions=baseline, baseline_name=protocol.baseline)
    return build_report(protocol=protocol, seal=seal, metric_sets=(metrics, *extra_sets)), metrics


def coverage_set(evaluation_src: Path, protocol, *, truth, lower, upper, level):
    """Measured coverage of one fitted interval on the sealed rows: the honest quality claim WP07 can make."""
    if str(evaluation_src) not in sys.path:
        sys.path.insert(0, str(evaluation_src))
    from m5phet_evaluation.scoring import MetricSet

    rows = protocol.population
    inside = sum(1 for row in rows if lower[row] <= truth[row] <= upper[row])
    widths = [upper[row] - lower[row] for row in rows]
    below = sum(1 for row in rows if truth[row] < lower[row])
    above = sum(1 for row in rows if truth[row] > upper[row])
    return MetricSet(
        name="interval_coverage", family="forecast", population=protocol.population,
        values={"nominal_level": float(level), "empirical_coverage": inside / len(rows),
                "mean_interval_width": float(np.mean(widths)), "median_interval_width": float(np.median(widths)),
                "below_lower": below / len(rows), "above_upper": above / len(rows)},
        counts={"declared_rows": len(rows), "scored_rows": len(rows), "inside_rows": inside},
        baseline=None,
        notes=("Coverage is the share of sealed rows whose realised value fell inside the fitted interval. It is a "
               "measurement of this interval on these rows and nothing else: it is not a guarantee for other rows, and "
               "a nominal level is a claim the fit makes, never one this number confirms.",))


# ------------------------------------------------------------------------------------------------------------- the run

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--evaluation-src", type=Path,
                        default=Path(os.environ.get("M5PHET_EVALUATION_SRC", "")) or None,
                        help="the src/ directory of M5PHET's evaluation package")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--seal-window", type=int, required=True,
                        help="the history length the sealed population is defined by; the same for every stage of one "
                             "comparison, and never smaller than any stage's own window")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--sealed-at", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--minimum-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--head", choices=("point", "quantile"), default="point")
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.5, 0.95])
    parser.add_argument("--device", choices=("gpu", "cpu"), default="cpu")
    args = parser.parse_args(argv)

    if args.evaluation_src is None or not (args.evaluation_src / "m5phet_evaluation").is_dir():
        parser.error("--evaluation-src must point at M5PHET/evaluation/src")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf

    if args.device == "gpu":
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    # A comparison whose stages cannot be refitted to the same numbers is not reviewable: without this, two runs of the
    # SAME stage on the GPU differed in the fourth decimal of the MAE and in the epoch early stopping fired, which is
    # the size of the gap between the stages themselves.
    tf.config.experimental.enable_op_determinism()

    started = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started))
    spec = read_spec(args.spec)
    data = read_csv(args.data)
    config = build_config(spec, columns=data["columns"], horizon=args.horizon, epochs=args.epochs,
                          patience=args.patience, batch_size=args.batch_size, seed=args.seed,
                          head=args.head, quantiles=args.quantiles)
    if config["window_size"] > args.seal_window:
        raise SpecError(f"the stage's window {config['window_size']} exceeds the sealing window {args.seal_window}; "
                        f"the sealed population would not hold its history")
    data["target_index"] = data["columns"].index(config["target_column"])

    population = sealed_population(data, holdout_fraction=args.holdout_fraction,
                                   seal_window=args.seal_window, horizon=args.horizon)
    protocol, seal = build_protocol_and_seal(
        args.evaluation_src, population["rows"], population["labels"], data_path=args.data,
        sealed_at=args.sealed_at, seal_window=args.seal_window, horizon=args.horizon,
        holdout_fraction=args.holdout_fraction, minimum_rows=args.minimum_rows)

    # the scaler: TRAIN rows only, by construction -- the holdout rows are never read for it
    train_rows = data["values"][:population["holdout_start"]]
    usable = data["finite"][:population["holdout_start"]]
    mean = train_rows[usable].mean(axis=0)
    sd = train_rows[usable].std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or (sd <= 0).any():
        raise SpecError("a train column has a non-finite or zero spread; it cannot be standardised")
    scaled = np.where(data["finite"][:, None], (data["values"] - mean) / sd, np.nan).astype(np.float32)

    window = config["window_size"]
    origins = training_origins(data, holdout_start=population["holdout_start"], window=window, horizon=args.horizon)
    cut = int(len(origins) * (1.0 - args.validation_fraction))
    fit_origins, val_origins = origins[:cut], origins[cut:]
    target_index = data["target_index"]
    y_all = ((data["values"][:, target_index] - mean[target_index]) / sd[target_index]).astype(np.float32)

    x_train = windows_for(scaled, fit_origins, window)
    x_val = windows_for(scaled, val_origins, window)
    y_train = y_all[fit_origins + args.horizon].reshape(-1, 1)
    y_val = y_all[val_origins + args.horizon].reshape(-1, 1)

    plugin, module_ref, epochs_run, history = fit_stage(
        config, x_train, y_train, x_val, y_val, repo_root=REPO_ROOT, epochs=args.epochs, head=args.head)

    x_hold = windows_for(scaled, population["origins"], window)
    raw = plugin.model.predict(x_hold, batch_size=args.batch_size, verbose=0)
    raw = np.asarray(raw[0] if isinstance(raw, list) else raw, dtype=np.float64)
    values = raw * float(sd[target_index]) + float(mean[target_index])

    rows = protocol.population
    truth = dict(population["labels"])
    baseline = {row: float(data["values"][origin, target_index])
                for row, origin in zip(rows, population["origins"])}

    extra_sets, coverage_payload = (), None
    if args.head == "quantile":
        quantiles = list(config["quantiles"])
        if values.shape[1] != len(quantiles):
            raise SpecError(f"the quantile head emitted {values.shape[1]} columns for {len(quantiles)} quantiles")
        median = quantiles.index(0.5)
        point = {row: float(values[i, median]) for i, row in enumerate(rows)}
        low, high = 0, len(quantiles) - 1
        level = round(quantiles[high] - quantiles[low], 6)
        lower = {row: float(values[i, low]) for i, row in enumerate(rows)}
        upper = {row: float(values[i, high]) for i, row in enumerate(rows)}
        covered = coverage_set(args.evaluation_src, protocol, truth=truth, lower=lower, upper=upper, level=level)
        extra_sets = (covered,)
        coverage_payload = dict(covered.values, quantile_pair=[quantiles[low], quantiles[high]],
                                rows=len(rows), seal=seal.seal)
    else:
        point = {row: float(values[i, 0]) for i, row in enumerate(rows)}

    report, metrics = score_and_report(args.evaluation_src, protocol, seal, truth=truth, predictions=point,
                                       baseline=baseline, extra_sets=extra_sets)

    sys.path.insert(0, str(Path(args.evaluation_src).parent))
    from compare_stages import annotate

    payload = annotate(report, stage=args.stage, target=config["target_column"], horizon=f"h+{args.horizon}",
                       scale=f"kW (mean absolute error in kW, {args.horizon} steps of "
                             f"{int(np.median(data['steps']))} s ahead)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (out / "seal.json").write_text(json.dumps(seal.as_dict(), indent=2, sort_keys=True) + "\n")
    (out / "protocol.json").write_text(json.dumps(protocol.to_dict(), indent=2, sort_keys=True) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if coverage_payload is not None:
        (out / "interval_coverage.json").write_text(json.dumps(coverage_payload, indent=2, sort_keys=True) + "\n")

    fitted = out / "fitted"
    fitted.mkdir(exist_ok=True)
    plugin.model.save(fitted / "model.keras")
    manifest = {
        "schema": RUN_SCHEMA,
        "stage": args.stage,
        "fitted_at": started_at,
        # The spec is carried by VALUE, not only by path: an export has to be able to say which representation produced
        # the bundle it serves, and a path is not an answer -- the file may have changed, or may not exist on the host
        # that reads the manifest. The digest binds the file this run actually read.
        "spec": {"path": str(args.spec), "schema": spec["schema"],
                 "representation_id": spec.get("representation_id"),
                 "decisions": list(spec.get("decisions") or ()),
                 "provenance": spec.get("provenance"),
                 "sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                 "representation": json.loads(json.dumps(spec["representation"], sort_keys=True))},
        "config": config,
        "plugin_module": module_ref,
        "head": args.head,
        "quantiles": list(config["quantiles"]) if args.head == "quantile" else None,
        "columns": list(data["columns"]),
        "windows_declared": [int(value) for value in (spec["representation"].get("windows") or ())],
        "window_selected_by": "max(representation.windows): the longest memory the candidate declares; a shorter window "
                              "in the same list is not fitted by this single-block harness and is recorded here",
        "target": config["target_column"],
        "horizons": [args.horizon],
        "window": window,
        "step_seconds": int(np.median(data["steps"])),
        "scaler": {"kind": "per_column_zscore", "mean": mean.tolist(), "sd": sd.tolist(),
                   "fitted_on": f"the first {population['holdout_start']} rows of {args.data.name} (TRAIN only; the "
                                f"sealed holdout is never read for it)"},
        "data": {"path": str(args.data), "rows": int(len(data["values"])),
                 "holdout_start_row": population["holdout_start"],
                 "non_finite_rows": int((~data["finite"]).sum())},
        "population": {"sealed_rows": len(rows), "dropped_for_non_finite": population["dropped"],
                       "seal": seal.seal, "sealed_at": args.sealed_at, "seal_window": args.seal_window,
                       "protocol_digest": protocol.digest},
        "training": {"fit_origins": int(len(fit_origins)), "validation_origins": int(len(val_origins)),
                     "epochs_declared": args.epochs, "epochs_run": epochs_run, "patience": args.patience,
                     "batch_size": args.batch_size, "seed": args.seed, "device": args.device,
                     "deterministic_ops": True, "loop": SINGLE_OUTPUT_LOOP,
                     "final_val_loss": history.get("val_loss", [None])[-1]},
        "preprocessing": preprocessing_record(spec, data["columns"]),
        "branches": config.get("branches") or config["grouping_applied"],
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "wall_seconds": round(time.time() - started, 3),
        "execution_authorized": False,
    }
    (out / "fitted" / "fit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True) + "\n")

    with (out / "predictions.csv").open("w") as stream:
        header = ["row", "truth", "naive_last_value", "prediction"]
        if args.head == "quantile":
            header += [f"q{q}" for q in config["quantiles"]]
        stream.write(",".join(header) + "\n")
        for i, row in enumerate(rows):
            line = [row, repr(truth[row]), repr(baseline[row]), repr(point[row])]
            if args.head == "quantile":
                line += [repr(float(values[i, j])) for j in range(values.shape[1])]
            stream.write(",".join(line) + "\n")

    print(json.dumps({"stage": args.stage, "seal": seal.seal, "sealed_rows": len(rows),
                      "mae": metrics.values["mae"], "rmse": metrics.values["rmse"],
                      "naive_mae": metrics.baseline["mae"], "skill_mae": metrics.values["skill_mae"],
                      "epochs_run": epochs_run, "coverage": coverage_payload, "out": str(out)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
