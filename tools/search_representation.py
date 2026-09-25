#!/usr/bin/env python
"""WP06 stage 5: search the representation space, with the closure table's own number as the objective.

Stages 1-4 built one representation by hand, four by a design job, and scored each of them once on a sealed holdout.
This is the stage that asks whether *anything* in the declared space does better -- not by proposing two more
candidates, but by spending a declared budget of fits over a declared space and letting every point that was
evaluated become a stage of the table.

What makes it a search and not a tuning run:

* **the objective is the closure table's own number.** Each point is fitted through ``tools/fit_pipeline_spec.py``
  with everything but the representation held at ``baseline_hand``'s configuration -- the same core, the same
  encoder, the same grouping shape, the same epochs, patience, batch size, seed and deterministic ops -- and scored
  on the identical seal, which the fit refuses to run without (``--expect-seal``). The scalar the search minimises is
  the held-out MAE in kW of that report, read from the report, never recomputed here;
* **every evaluated point is a stage.** Its report (``m5phet-evaluation-report/1``), its run manifest, its config and
  its training history are written under ``--out-dir`` and are what ``evaluation/compare_stages.py`` reads. A point
  is never scored privately and thrown away;
* **a point that does not validate is refused by name and counted, never repaired.** The representation goes through
  ``feature_eng_m5phet.representation.validate_spec`` and the harness's own bounds, and a genome that names a lag
  beyond its window, a differencing order under a transform that already differences, or no column at all, produces a
  refusal with the code and the genome that caused it. Moving it to the nearest legal point would make the search
  report a space it did not search;
* **the card is checked before every dispatch.** ``--device gpu`` means "use the GPU if nothing else is holding it":
  ``device_for`` reads the driver before each fit and falls back to the CPU, recording why in the ledger, when another
  job has the card or the card cannot be asked. (Added after the two rounds of 2026-09-25, which checked the card once
  per round instead; their ledgers therefore carry no ``device`` field.)
* **nothing is tuned to make a candidate win.** The only thing a genome changes is the representation. Every number
  that decides how long a fit runs or how it is optimised is declared once, on the command line, and is the same for
  every point and for the stages measured before this one.

The optimiser is DEAP, driven exactly as ``optimizer_plugins/default_optimizer.py`` drives it -- integer genes over
declared bounds, ``cxTwoPoint``, per-gene mutation at ``indpb``, ``selTournament(tournsize=3)``, elitism of one -- and
for the same reason that plugin exists: this repository already has a genetic optimizer and a second one would be a
second answer to a question that has one. It is not that plugin's ``optimize()`` because that method's objective is
the predictor pipeline's own validation loss over its own data loading, and this search's objective must be the
sealed-holdout number of the closure table and nothing else; see ``docs`` in the accompanying evidence for why the
DOIN interface was not used either.

Usage::

    python tools/search_representation.py --hand-spec <baseline_hand pipeline spec> --data slice.csv \\
        --out-dir <dir> --evaluation-src <M5PHET/evaluation/src> --feature-eng-src <feature-eng worktree> \\
        --seal-window 197 --horizon 60 --expect-seal 33820b552ddf --sealed-at 2026-09-25T00:00:00Z \\
        --epochs 200 --patience 15 --population 12 --generations 5 --seed 1

Restarting the same command with the same ``--out-dir`` loses no fit: every evaluation is written to the ledger as it
finishes and a genome whose representation was already evaluated is read from it instead of being fitted again.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

LEDGER_SCHEMA = "predictor.representation_search.v1"

#: the lags the design job (WP06 stage 2) motivated on this series, and nothing else. `1` is the one-step lag every
#: candidate carries; `74`, `1443` and `2892` are the autocorrelation peaks it reported outside the white-noise band;
#: `197` is the decay lag -- the first lag whose autocorrelation falls inside the band -- which is also the sealing
#: window of this comparison. A search that invented lags of its own would be searching a space no test motivated.
MOTIVATED_LAGS = (1, 74, 197, 1443, 2892)

#: the lags a genome may name. The two long peaks are NOT in the space, and they are excluded by the refusal that
#: already removed the two design candidates naming them from the sealed comparison: a window that reaches lag 1443 or
#: 2892 exceeds the sealing window, which seals a different population (8578 and 7129 origins under different seals),
#: so no point naming them could be scored on this holdout at all. That refusal is reused as the bound of the space --
#: the same way the window's upper bound is the sealing window -- rather than being rediscovered once per genome.
#: Within the space it is still live: a genome naming lag 197 or 74 under a shorter window is refused by name.
SEARCHABLE_LAGS = (1, 74, 197)
EXCLUDED_LAGS = {
    1443: "LAG_EXCEEDS_SEALING_WINDOW: the design candidate seasonal_lag_1443 was refused for this reason (sealing at "
          "1443 leaves 8578 origins under seal a8c7f07f6d6c, not 9824 under 33820b552ddf)",
    2892: "LAG_EXCEEDS_SEALING_WINDOW: the design candidate seasonal_lag_2892 was refused for this reason (sealing at "
          "2892 leaves 7129 origins under seal 97155a063e25, not 9824 under 33820b552ddf)",
}

#: `feature_eng_m5phet.representation.TRANSFORMS`, in the order the spec declares them
TRANSFORMS = ("level", "diff", "log_return")

#: `feature_eng_m5phet.representation.MAX_ORDER`
MAX_ORDER = 2

#: the shortest window a genome may name: the encoder's kernel size. A window shorter than the kernel is read entirely
#: through the causal padding, so it is not a shorter memory but a degenerate graph.
MIN_WINDOW = 3

REFUSED = "REFUSED"
OK = "OK"

#: refusals this harness raises before a fit is dispatched. Each names what was declared, never what it should be.
LAG_EXCEEDS_WINDOW = "LAG_EXCEEDS_WINDOW"
NO_LAG_DECLARED = "NO_LAG_DECLARED"
NO_FEATURE_SELECTED = "NO_FEATURE_SELECTED"


#: how much of the GPU another process may hold before a dispatch falls back to the CPU. A display server sits on a
#: few hundred MiB of an otherwise idle card; a training job does not.
GPU_BUSY_MIB = 1536


def gpu_holders(nvidia_smi="nvidia-smi"):
    """(used MiB, compute processes) as the driver reports them, or `None` when the card cannot be asked.

    Read before every dispatch, not once per round: the owner's other work may start at any time, and a fit that has
    to share the card is both slower and no longer running under the conditions the other stages ran under.
    """
    try:
        used = subprocess.run([nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30)
        apps = subprocess.run([nvidia_smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if used.returncode != 0:
        return None
    first = used.stdout.strip().splitlines()
    if not first:
        return None
    try:
        megabytes = int(first[0].strip())
    except ValueError:
        return None
    processes = [line for line in apps.stdout.strip().splitlines() if line.strip()]
    return megabytes, processes


def device_for(requested, *, nvidia_smi="nvidia-smi"):
    """The device this dispatch may use, and why. A card another job is holding is left alone, not shared."""
    if requested != "gpu":
        return requested, "the command asked for the CPU"
    reading = gpu_holders(nvidia_smi)
    if reading is None:
        return "cpu", "the GPU could not be asked what it is holding, so this dispatch does not assume it is free"
    megabytes, processes = reading
    if megabytes > GPU_BUSY_MIB:
        return "cpu", (f"the GPU holds {megabytes} MiB and {len(processes)} compute process(es), above the declared "
                       f"{GPU_BUSY_MIB} MiB: another job has it and this dispatch does not share it")
    return "gpu", f"the GPU holds {megabytes} MiB, below the declared {GPU_BUSY_MIB} MiB"


class SearchError(RuntimeError):
    """The search cannot be set up: a missing input, not a refused candidate."""


# ------------------------------------------------------------------------------------------------- the search space

def space(*, seal_window: int, columns, window_low: int = MIN_WINDOW, window_high: int = 0) -> dict:
    """The declared space, as data: what each gene means and the bounds it is drawn from. Written into the ledger."""
    window_high = int(window_high or seal_window)
    return {
        "window": {"gene": 0, "type": "int", "low": int(window_low), "high": window_high,
                   "seal_window": int(seal_window), "shortest_possible": MIN_WINDOW,
                   "why": f"the sealing window of this comparison is {seal_window} rows: a stage whose window exceeds "
                          f"it cannot be scored on the sealed population at all, and that refusal is reused here as "
                          f"the bound of the space rather than worked around. The shortest possible window is the "
                          f"encoder's kernel size. A round may draw from a declared SUB-RANGE of that interval "
                          f"({window_low}..{window_high} here), which is a second search over a declared sub-space and "
                          f"is written here rather than left to be inferred from the points that came out"},
        "lags": {"genes": [1, 2, 3], "type": "bit", "values": list(SEARCHABLE_LAGS),
                 "motivated_set": list(MOTIVATED_LAGS),
                 "excluded": {str(lag): why for lag, why in EXCLUDED_LAGS.items()},
                 "why": "the design job's own motivated set -- the one-step lag, the autocorrelation peaks outside the "
                        "white-noise band (74, 1443, 2892) and the decay lag (197) -- less the two peaks no point of "
                        "this comparison could be scored under, which are excluded by the refusal that already "
                        "removed the candidates naming them. A lag beyond the genome's own window is still refused"},
        "transform": {"gene": 4, "type": "int", "low": 0, "high": len(TRANSFORMS) - 1, "values": list(TRANSFORMS),
                      "why": "the values m5phet.representation.v1 declares, and no others"},
        "differencing_order": {"gene": 5, "type": "int", "low": 0, "high": MAX_ORDER,
                               "why": "0..MAX_ORDER of the spec; an order above 0 under a transform that already "
                                      "differences is refused by the spec as AMBIGUOUS_DIFFERENCING"},
        "features": {"genes": [6, 7, 8, 9, 10, 11, 12], "type": "bit", "values": list(columns),
                     "why": "the seven meter columns of the sealed file; the subset a branch reads is the subset the "
                            "graph gathers, and the empty subset is refused"},
    }


GENOME_LENGTH = 13


def bounds(*, seal_window: int, window_low: int = MIN_WINDOW, window_high: int = 0):
    low = [int(window_low)] + [0] * len(SEARCHABLE_LAGS) + [0, 0] + [0] * 7
    high = [int(window_high or seal_window)] + [1] * len(SEARCHABLE_LAGS) + [len(TRANSFORMS) - 1, MAX_ORDER] + [1] * 7
    return low, high


def decode(genome, columns):
    """A genome as the five declared quantities. No clipping, no repair: the caller refuses what is not legal."""
    window = int(genome[0])
    lags = [lag for bit, lag in zip(genome[1:4], SEARCHABLE_LAGS) if int(bit)]
    transform = TRANSFORMS[int(genome[4])]
    order = int(genome[5])
    features = [name for bit, name in zip(genome[6:13], columns) if int(bit)]
    return {"window": window, "lags": lags, "transform": transform, "differencing_order": order,
            "features": features}


# ----------------------------------------------------------------------------------------------- the spec it builds

def representation_of(point, *, hand, candidate_id):
    """One `m5phet.representation.v1` from a decoded point, with the hand stage's sampling, clock and holdout."""
    base = hand["representation"]
    target = base["target"]["column"]
    return {
        "schema": base["schema"],
        "candidate_id": candidate_id,
        "provenance": base["provenance"],
        "target": {"column": target, "transform": point["transform"]},
        "windows": [point["window"]],
        "lags": sorted(point["lags"]),
        "differencing": {"order": point["differencing_order"]},
        "features": [],
        "exogenous": sorted(name for name in point["features"] if name != target),
        "calendar": dict(base["calendar"]),
        "sampling": dict(base["sampling"]),
        "holdout": dict(base["holdout"]),
        "why": {"windows": "searched: the window is a gene of the declared space, bounded by the sealing window",
                "lags": "searched: a subset of the design job's motivated lags, each no longer than the window",
                "transform": "searched: one of the transforms the representation schema declares",
                "features": "none: this search adds no derived or calendar feature, only chooses among the file's "
                            "own columns"},
    }


def pipeline_spec_of(point, *, hand, stage, candidate_id, budget_sentence):
    """The `m5phet.pipeline.v1` the fit reads: the searched representation, everything else held at the hand stage."""
    spec = json.loads(json.dumps(hand))
    spec["stage"] = stage
    spec["chosen_by"] = "SEARCH"
    spec["why"] = (f"WP06 stage 5: the searched representation {candidate_id!r}, fitted with the rest of the pipeline "
                   f"held at exactly what the baseline_hand stage uses (the same core, the same encoder, no "
                   f"per-feature preprocessor declared, the same epochs, patience, seed, batch size and deterministic "
                   f"ops). The representation is the only thing that differs from the baseline, and the only thing a "
                   f"difference in the table can be attributed to. {budget_sentence}")
    spec["representation"] = representation_of(point, hand=hand, candidate_id=candidate_id)
    spec["decisions"] = []
    spec["grouping"] = json.loads(json.dumps(hand["grouping"]))
    spec["grouping"]["chosen_by"] = "SEARCH"
    spec["grouping"]["groups"] = [{"group_id": "all", "members": list(point["features"])}]
    spec["grouping"]["why"] = ("the baseline_hand grouping shape -- one block -- holding the columns the search "
                               "selected; the cut k is not a gene of this space")
    return spec


def validate(spec_representation, *, point, seal_window, feature_eng_src):
    """`None` when the representation is legal, else `(code, why)`. Nothing here repairs anything."""
    if str(feature_eng_src) not in sys.path:
        sys.path.insert(0, str(feature_eng_src))
    from feature_eng_m5phet import representation as representation_module

    if not point["features"]:
        return (NO_FEATURE_SELECTED, "the genome selects no column; a branch that reads nothing is not a "
                                     "representation of anything")
    if not point["lags"]:
        return (NO_LAG_DECLARED, "the genome selects no lag; the design job motivated five, three of them are in this "
                                 "space, and a representation that names none of them declares no memory to read")
    if max(point["lags"]) > point["window"]:
        return (LAG_EXCEEDS_WINDOW,
                f"the genome declares the lag {max(point['lags'])} and the window {point['window']}; this harness "
                f"fits one contiguous window, so a lag beyond it is history the graph never sees. It is the refusal "
                f"that kept seasonal_lag_1443 and seasonal_lag_2892 out of the sealed comparison and it is reused, "
                f"not worked around")
    if point["window"] + point["differencing_order"] > seal_window:
        return ("WINDOW_PLUS_DIFFERENCING_EXCEEDS_SEALING_WINDOW",
                f"the genome needs {point['window'] + point['differencing_order']} rows of history before each origin "
                f"and the sealing window is {seal_window}")
    try:
        representation_module.validate_spec(spec_representation)
    except representation_module.SpecError as error:
        return (error.code, error.why)
    return None


def identity(spec_representation, *, feature_eng_src):
    """The representation's own identity, from feature-eng: two genomes with the same representation are one point."""
    if str(feature_eng_src) not in sys.path:
        sys.path.insert(0, str(feature_eng_src))
    from feature_eng_m5phet import representation as representation_module

    return representation_module.spec_id(spec_representation)


# ---------------------------------------------------------------------------------------------------- the objective

def fit(spec_path: Path, *, stage: str, out_dir: Path, args) -> dict:
    """One evaluation: the fit harness, under the memory guard, reporting the report's own MAE. Never recomputed here."""
    device, why = device_for(args.device)
    command = [args.crispdm_run, "-m", args.memory, "-t", str(args.wall_seconds), "-n", args.guard_name, "--",
               args.python, str(REPO_ROOT / "tools" / "fit_pipeline_spec.py"),
               "--spec", str(spec_path), "--data", str(args.data), "--stage", stage,
               "--out-dir", str(out_dir), "--evaluation-src", str(args.evaluation_src),
               "--seal-window", str(args.seal_window), "--horizon", str(args.horizon),
               "--holdout-fraction", str(args.holdout_fraction), "--sealed-at", args.sealed_at,
               "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--batch-size", str(args.batch_size), "--seed", str(args.seed), "--device", device,
               "--expect-seal", args.expect_seal]
    started = time.time()
    finished = subprocess.run(command, capture_output=True, text=True)
    seconds = round(time.time() - started, 3)
    if finished.returncode != 0:
        tail = (finished.stderr or finished.stdout or "").strip().splitlines()
        reason = next((line for line in reversed(tail) if ":" in line), "the fit exited non-zero with no message")
        return {"status": REFUSED, "refusal": "FIT_REFUSED", "why": reason.strip()[:600], "seconds": seconds}
    line = [row for row in finished.stdout.strip().splitlines() if row.startswith("{")]
    if not line:
        return {"status": REFUSED, "refusal": "NO_SUMMARY", "why": "the fit wrote no summary line", "seconds": seconds}
    summary = json.loads(line[-1])
    report = json.loads((out_dir / "report.json").read_text())
    # the scalar is READ from the report the table will read, not recomputed from predictions here
    metric_set = report["metric_sets"][0]
    mae = float(metric_set["values"]["mae"])
    if not summary["seal"].startswith(args.expect_seal):
        return {"status": REFUSED, "refusal": "NOT_COMPARABLE",
                "why": f"the fit sealed {summary['seal'][:12]}, not {args.expect_seal}", "seconds": seconds}
    return {"status": OK, "device": device, "device_why": why, "mae": mae,
            "rmse": float(metric_set["values"]["rmse"]),
            "skill_mae": float(metric_set["values"]["skill_mae"]),
            "naive_mae": float(metric_set["baseline"]["mae"]), "seal": summary["seal"],
            "sealed_rows": int(summary["sealed_rows"]), "epochs_run": int(summary["epochs_run"]),
            "seconds": seconds, "report": str(out_dir / "report.json")}


# -------------------------------------------------------------------------------------------------------- the ledger

def load_ledger(path: Path):
    if not path.exists():
        return {}
    ledger = json.loads(path.read_text())
    return {entry["representation_id"]: entry for entry in ledger.get("evaluations", ())}


def write_ledger(path: Path, *, entries, header):
    payload = dict(header)
    payload["evaluations"] = [entries[key] for key in sorted(entries)]
    payload["evaluated"] = sum(1 for entry in entries.values() if entry["status"] == OK)
    payload["refused"] = sum(1 for entry in entries.values() if entry["status"] == REFUSED)
    refusals = {}
    for entry in entries.values():
        if entry["status"] == REFUSED:
            refusals[entry["refusal"]] = refusals.get(entry["refusal"], 0) + 1
    payload["refusals_by_name"] = dict(sorted(refusals.items()))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------------------------------------ the run

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand-spec", required=True, type=Path, help="the baseline_hand m5phet.pipeline.v1 spec")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--evaluation-src", required=True, type=Path)
    parser.add_argument("--feature-eng-src", required=True, type=Path,
                        help="the checkout holding feature_eng_m5phet (its representation module validates the specs)")
    parser.add_argument("--expect-seal", required=True, help="the seal every stage of this comparison was scored on")
    parser.add_argument("--seal-window", type=int, required=True)
    parser.add_argument("--window-low", type=int, default=MIN_WINDOW,
                        help="the shortest window this round draws; declared in the ledger's space")
    parser.add_argument("--window-high", type=int, default=0,
                        help="the longest window this round draws; 0 means the sealing window")
    parser.add_argument("--sealed-at", required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--initial-population", type=int, default=0,
                        help="genomes drawn for generation 0; 0 means --population. A larger first draw is declared "
                             "rather than repaired: the space couples the window to the lags, so a uniform draw is "
                             "mostly refused, and sampling more of it is the honest way to start a generation with "
                             "several legal points instead of moving illegal ones to the nearest legal place")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--cxpb", type=float, default=0.5)
    parser.add_argument("--mutpb", type=float, default=0.3)
    parser.add_argument("--indpb", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--crispdm-run", default=os.path.expanduser("~/.local/bin/crispdm-run"))
    parser.add_argument("--memory", default="10G")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--guard-name", default="wp06s")
    parser.add_argument("--max-evaluations", type=int, default=0,
                        help="a hard stop on fits actually dispatched; 0 means the generational budget alone")
    args = parser.parse_args(argv)

    from deap import base, creator, tools

    hand = json.loads(args.hand_spec.read_text())
    columns = list(hand["features"])
    if len(columns) != 7:
        raise SearchError(f"the feature genes are declared over seven columns; the hand spec names {len(columns)}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages_dir = out_dir / "stages"
    specs_dir = out_dir / "specs"
    stages_dir.mkdir(exist_ok=True)
    specs_dir.mkdir(exist_ok=True)
    ledger_path = out_dir / "ledger.json"
    entries = load_ledger(ledger_path)

    initial_population = args.initial_population or args.population
    budget_sentence = (f"The budget of this search is {initial_population} genomes drawn for generation 0 and "
                       f"{args.population} individuals for each of the {args.generations - 1} generations after it, "
                       f"at most {args.max_evaluations or args.population * args.generations} fits dispatched; every "
                       f"legal point is fitted once and every refused point is counted and never fitted.")
    header = {
        "schema": LEDGER_SCHEMA,
        "objective": (f"the held-out mean absolute error in kW of the report the closure table reads, on the corpus "
                      f"sealed {args.expect_seal} (sealing window {args.seal_window}, horizon {args.horizon}); "
                      f"minimised"),
        "budget": {"population": args.population, "initial_population": initial_population,
                   "generations": args.generations,
                   "maximum_evaluations": args.max_evaluations or args.population * args.generations,
                   "sentence": budget_sentence},
        "operators": {"library": "deap", "selection": "selTournament(tournsize=3)", "crossover": "cxTwoPoint",
                      "mutation": "per-gene uniform redraw within the declared bounds",
                      "cxpb": args.cxpb, "mutpb": args.mutpb, "indpb": args.indpb, "elitism": 1,
                      "why": "the operator set optimizer_plugins/default_optimizer.py already uses in this "
                             "repository; a second genetic optimizer would be a second answer to a question that "
                             "has one"},
        "held_fixed": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                       "seed": args.seed, "device": args.device, "horizon": args.horizon,
                       "seal_window": args.seal_window, "core": hand["core"]["key"],
                       "encoder": hand["core"]["encoder_mapping"]["branches"]["all"]["encoder"],
                       "why": "everything but the representation is the baseline_hand stage's own configuration; a "
                              "hyper-parameter moved to make a point win would make the table unreadable"},
        "space": space(seal_window=args.seal_window, columns=columns, window_low=args.window_low,
                       window_high=args.window_high),
        "data": str(args.data),
        "hand_spec": str(args.hand_spec),
        "seed": args.seed,
    }

    random.seed(args.seed)
    if not hasattr(creator, "FitnessMinMAE"):
        creator.create("FitnessMinMAE", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "SearchIndividual"):
        creator.create("SearchIndividual", list, fitness=creator.FitnessMinMAE)
    low, high = bounds(seal_window=args.seal_window, window_low=args.window_low, window_high=args.window_high)

    toolbox = base.Toolbox()
    toolbox.register("individual", lambda: creator.SearchIndividual(
        [random.randint(low[i], high[i]) for i in range(GENOME_LENGTH)]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def mutate(individual, indpb):
        for i in range(GENOME_LENGTH):
            if random.random() < indpb:
                individual[i] = random.randint(low[i], high[i])
        return individual,

    toolbox.register("mutate", mutate, indpb=args.indpb)

    dispatched = [0]

    def evaluate(individual):
        point = decode(individual, columns)
        spec = pipeline_spec_of(point, hand=hand, stage="pending", candidate_id="pending",
                                budget_sentence=budget_sentence)
        problem = validate(spec["representation"], point=point, seal_window=args.seal_window,
                           feature_eng_src=args.feature_eng_src)
        if problem is not None:
            key = f"refused:{problem[0]}:{json.dumps(list(individual))}"
            if key not in entries:
                entries[key] = {"representation_id": key, "status": REFUSED, "refusal": problem[0], "why": problem[1],
                                "genome": list(individual), "point": point}
                write_ledger(ledger_path, entries=entries, header=header)
            return (float("inf"),)

        representation_id = identity(spec["representation"], feature_eng_src=args.feature_eng_src)
        if representation_id in entries:
            known = entries[representation_id]
            return (known["mae"] if known["status"] == OK else float("inf"),)

        if args.max_evaluations and dispatched[0] >= args.max_evaluations:
            entries[representation_id] = {"representation_id": representation_id, "status": REFUSED,
                                          "refusal": "BUDGET_EXHAUSTED",
                                          "why": f"the declared hard stop of {args.max_evaluations} dispatched fits "
                                                 f"was reached before this point was evaluated",
                                          "genome": list(individual), "point": point}
            write_ledger(ledger_path, entries=entries, header=header)
            return (float("inf"),)

        candidate_id = f"searched_{representation_id[:12]}"
        stage = f"searched_{representation_id[:12]}"
        spec = pipeline_spec_of(point, hand=hand, stage=stage, candidate_id=candidate_id,
                                budget_sentence=budget_sentence)
        spec_path = specs_dir / f"spec_{stage}.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
        dispatched[0] += 1
        outcome = fit(spec_path, stage=stage, out_dir=stages_dir / stage, args=args)
        entry = {"representation_id": representation_id, "stage": stage, "candidate_id": candidate_id,
                 "genome": list(individual), "point": point, "spec": str(spec_path), **outcome}
        entries[representation_id] = entry
        write_ledger(ledger_path, entries=entries, header=header)
        print(json.dumps({"dispatched": dispatched[0], "stage": stage, "point": point,
                          "status": entry["status"], "mae": entry.get("mae"), "seconds": entry.get("seconds")},
                         sort_keys=True), flush=True)
        return (entry["mae"] if entry["status"] == OK else float("inf"),)

    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=initial_population)
    for individual, fitness in zip(population, map(toolbox.evaluate, population)):
        individual.fitness.values = fitness
    generations = [{"generation": 0, "best": min(float(ind.fitness.values[0]) for ind in population)}]

    for generation in range(1, args.generations):
        elite = min(population, key=lambda ind: ind.fitness.values[0])
        offspring = [toolbox.clone(ind) for ind in toolbox.select(population, args.population - 1)]
        for first, second in zip(offspring[::2], offspring[1::2]):
            if random.random() < args.cxpb:
                toolbox.mate(first, second)
                del first.fitness.values
                del second.fitness.values
        for mutant in offspring:
            if random.random() < args.mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        offspring.append(toolbox.clone(elite))               # elitism of one: the best point is never lost
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness
        population = offspring
        generations.append({"generation": generation,
                            "best": min(float(ind.fitness.values[0]) for ind in population)})

    header["generations"] = generations
    header["dispatched_fits"] = dispatched[0]
    write_ledger(ledger_path, entries=entries, header=header)

    measured = [entry for entry in entries.values() if entry["status"] == OK]
    best = min(measured, key=lambda entry: entry["mae"]) if measured else None
    print(json.dumps({"dispatched_fits": dispatched[0], "measured": len(measured),
                      "refused": sum(1 for entry in entries.values() if entry["status"] == REFUSED),
                      "best_stage": best and best["stage"], "best_mae": best and best["mae"],
                      "best_point": best and best["point"], "ledger": str(ledger_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
