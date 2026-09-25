#!/usr/bin/env python
"""Re-fit declared stages on the inner region of a frozen nested split and score them ONCE on its outer holdout.

WP26 of the M5PHET work plan (2026-09-24, revision 3). This is the confirmation step, and it is deliberately small:
it chooses nothing, searches nothing and tunes nothing. It takes the stages a previous round selected, the split
``tools/nested_split.py`` froze before any of these fits existed, and a declared number of seeds, and it reports the
difference between each stage and the declared reference with an interval.

What it does, and what it refuses to do:

* **it does not select.** The stages are given on the command line; the winner of the earlier search is a stage like
  any other here, and nothing in this tool can promote one;
* **it scores once.** Every fit runs against the same frozen outer seal, passed to the fitting harness as
  ``--expect-seal``, which refuses before a weight is fitted if the population it would seal is another one;
* **the interval is named.** The primary interval is a paired Student-t interval over the seeds -- n is the number of
  seeds and nothing else -- and it measures **fit-to-fit variability under this one split**. It is not a sampling
  interval over datasets, over weeks, or over households, and this tool writes that sentence into its own output so it
  cannot be quoted without it. A second, different interval is reported beside it: a paired bootstrap over the sealed
  ROWS of the seed-averaged absolute errors, which measures the row-sampling uncertainty of the same difference. Two
  questions, two intervals, both declared;
* **the GPU is asked before every dispatch**, by the same ``device_for`` the search uses, and a held card sends the fit
  to the CPU with the reason recorded rather than sharing the card.

Usage::

    python tools/confirm_representation.py --split <evidence>/nested_split.json \\
        --stage baseline_hand=<spec.json> --stage searched_b373275495e2=<spec.json> \\
        --reference baseline_hand --seeds 1 2 3 4 5 --out-dir <evidence> \\
        --evaluation-src <M5PHET/evaluation/src> --epochs 200 --patience 15 --batch-size 256
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "predictor.outer_confirmation.v1"

#: Student-t two-sided 95 % quantiles for the small degrees of freedom this design can have. Written out rather than
#: pulled from scipy so the interval can be recomputed by hand from the table this tool prints.
T95 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622}


def load_search_tool():
    spec = importlib.util.spec_from_file_location("search_representation",
                                                  REPO_ROOT / "tools" / "search_representation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_predictions(path: Path):
    rows, truth, prediction, naive = [], {}, {}, {}
    with path.open() as stream:
        header = stream.readline().rstrip("\n").split(",")
        index = {name: position for position, name in enumerate(header)}
        for line in stream:
            parts = line.rstrip("\n").split(",")
            row = parts[index["row"]]
            rows.append(row)
            truth[row] = float(parts[index["truth"]])
            naive[row] = float(parts[index["naive_last_value"]])
            prediction[row] = float(parts[index["prediction"]])
    return rows, truth, naive, prediction


def t_interval(values):
    """Mean and two-sided 95 % Student-t interval of a paired difference. Refused, by name, below two observations."""
    n = len(values)
    if n < 2:
        return {"refused": "NOT_ENOUGH_OBSERVATIONS", "n": n}
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    standard_error = math.sqrt(variance / n)
    quantile = T95.get(n - 1)
    if quantile is None:
        quantile = 1.9600
    return {"n": n, "mean": mean, "sd": math.sqrt(variance), "standard_error": standard_error,
            "t_quantile_95": quantile, "low": mean - quantile * standard_error,
            "high": mean + quantile * standard_error, "df": n - 1}


def bootstrap_interval(differences, *, draws: int, seed: int):
    """Percentile bootstrap over the sealed rows of a per-row paired difference.

    Resampled with replacement row by row, which treats the sealed rows as exchangeable. They are a contiguous week of
    one household and are not: the interval is therefore narrower than the truth by an amount this tool does not
    estimate, and says so in its own output.
    """
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    n = values.size
    generator = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    block = 200
    for start in range(0, draws, block):
        count = min(block, draws - start)
        index = generator.integers(0, n, size=(count, n))
        means[start:start + count] = values[index].mean(axis=1)
    means.sort()
    return {"n_rows": int(n), "draws": int(draws), "seed": int(seed), "mean": float(values.mean()),
            "low": float(means[int(0.025 * draws)]), "high": float(means[min(draws - 1, int(0.975 * draws))])}


def fit_once(*, stage: str, spec: Path, seed: int, out_dir: Path, split: dict, args, search_tool) -> dict:
    device, why = search_tool.device_for(args.device)
    command = [args.crispdm_run, "-m", args.memory, "-t", str(args.wall_seconds), "-n", args.guard_name, "--",
               args.python, str(REPO_ROOT / "tools" / "fit_pipeline_spec.py"),
               "--spec", str(spec), "--data", split["derived"]["path"], "--stage", f"{stage}__seed{seed}",
               "--out-dir", str(out_dir), "--evaluation-src", str(args.evaluation_src),
               "--seal-window", str(split["outer"]["seal_window"]), "--horizon", str(split["outer"]["horizon"]),
               "--holdout-fraction", repr(split["derived"]["holdout_fraction"]),
               "--sealed-at", split["outer"]["sealed_at"],
               "--epochs", str(args.epochs), "--patience", str(args.patience),
               "--batch-size", str(args.batch_size), "--seed", str(seed), "--device", device,
               "--expect-seal", split["outer"]["seal"][:12]]
    started = time.time()
    finished = subprocess.run(command, capture_output=True, text=True)
    seconds = round(time.time() - started, 3)
    if finished.returncode != 0:
        tail = (finished.stderr or finished.stdout or "").strip().splitlines()
        reason = next((line for line in reversed(tail) if ":" in line), "the fit exited non-zero with no message")
        return {"status": "REFUSED", "refusal": "FIT_REFUSED", "why": reason.strip()[:600], "seconds": seconds,
                "device": device, "device_why": why}
    report = json.loads((out_dir / "report.json").read_text())
    metric_set = report["metric_sets"][0]
    return {"status": "OK", "device": device, "device_why": why, "seconds": seconds,
            "mae": float(metric_set["values"]["mae"]), "rmse": float(metric_set["values"]["rmse"]),
            "skill_mae": float(metric_set["values"]["skill_mae"]),
            "naive_mae": float(metric_set["baseline"]["mae"]),
            "sealed_rows": int(metric_set["counts"]["scored_rows"]),
            "seal": report.get("corpus_seal"), "protocol_digest": report.get("protocol_digest"),
            "out": str(out_dir)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--stage", action="append", required=True, metavar="NAME=SPEC",
                        help="a stage to re-fit; repeatable, order preserved")
    parser.add_argument("--reference", required=True, help="the stage every difference is taken against")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--evaluation-src", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260925)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--crispdm-run", default=os.path.expanduser("~/.local/bin/crispdm-run"))
    parser.add_argument("--memory", default="10G")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--guard-name", default="wp26")
    args = parser.parse_args(argv)

    split = json.loads(args.split.read_text())
    if split.get("schema") != "predictor.nested_split.v1":
        raise SystemExit(f"{args.split} is not a predictor.nested_split.v1 declaration")
    stages = []
    for entry in args.stage:
        name, _, path = entry.partition("=")
        if not path:
            raise SystemExit(f"--stage takes NAME=SPEC; got {entry!r}")
        stages.append((name, Path(path)))
    names = [name for name, _ in stages]
    if args.reference not in names:
        raise SystemExit(f"the reference {args.reference!r} is not among the stages {names}")

    search_tool = load_search_tool()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.out_dir / "stages"
    runs_dir.mkdir(exist_ok=True)

    results, predictions = {}, {}
    for name, spec in stages:
        results[name], predictions[name] = {}, {}
        for seed in args.seeds:
            out_dir = runs_dir / f"{name}__seed{seed}"
            record = fit_once(stage=name, spec=spec, seed=seed, out_dir=out_dir, split=split, args=args,
                              search_tool=search_tool)
            results[name][seed] = record
            print(json.dumps({"stage": name, "seed": seed, **{k: v for k, v in record.items() if k != "out"}},
                             sort_keys=True), flush=True)
            if record["status"] == "OK":
                predictions[name][seed] = read_predictions(out_dir / "predictions.csv")

    reference_maes = [results[args.reference][seed]["mae"] for seed in args.seeds
                      if results[args.reference][seed]["status"] == "OK"]
    per_stage, comparisons = {}, {}
    for name, _ in stages:
        maes = [results[name][seed]["mae"] for seed in args.seeds if results[name][seed]["status"] == "OK"]
        skills = [results[name][seed]["skill_mae"] for seed in args.seeds if results[name][seed]["status"] == "OK"]
        mean = sum(maes) / len(maes) if maes else None
        sd = (math.sqrt(sum((value - mean) ** 2 for value in maes) / (len(maes) - 1))
              if maes and len(maes) > 1 else None)
        per_stage[name] = {"seeds": args.seeds, "mae_per_seed": maes, "mae_mean": mean, "mae_sd": sd,
                           "mae_min": min(maes) if maes else None, "mae_max": max(maes) if maes else None,
                           "skill_mae_mean": sum(skills) / len(skills) if skills else None,
                           "epochs_or_refusals": [results[name][seed].get("refusal", "OK") for seed in args.seeds]}
        if name == args.reference:
            continue
        paired = [results[name][seed]["mae"] - results[args.reference][seed]["mae"] for seed in args.seeds
                  if results[name][seed]["status"] == results[args.reference][seed]["status"] == "OK"]
        over_seeds = t_interval(paired)
        # the row-level question: the seed-averaged absolute error of each stage, differenced row by row
        rows = predictions[args.reference][args.seeds[0]][0]
        per_row = []
        for row in rows:
            here = sum(abs(predictions[name][seed][3][row] - predictions[name][seed][1][row]) for seed in args.seeds)
            there = sum(abs(predictions[args.reference][seed][3][row] - predictions[args.reference][seed][1][row])
                        for seed in args.seeds)
            per_row.append((here - there) / len(args.seeds))
        comparisons[name] = {
            "reference": args.reference,
            "paired_over_seeds": {
                **over_seeds,
                "pairing": "by seed: the two stages are fitted with the same seed, the same inner rows, the same "
                           "declared epochs, patience, batch size and deterministic ops; only the representation "
                           "differs",
                "what_it_measures": "fit-to-fit variability under THIS split and THIS outer holdout. It is not a "
                                    "sampling interval over datasets, weeks or households, and a seed is not a draw "
                                    "from a population of problems",
                "interval": "Student-t, two-sided 95 %, df = seeds - 1",
                "excludes_zero": bool(over_seeds.get("low", 0) > 0 or over_seeds.get("high", 0) < 0),
                "direction": ("the stage beats the reference" if over_seeds.get("mean", 0) < 0
                              else "the reference beats the stage"),
            },
            "paired_over_rows": {
                **bootstrap_interval(per_row, draws=args.bootstrap_draws, seed=args.bootstrap_seed),
                "quantity": "per sealed row, the seed-averaged absolute error of this stage minus that of the "
                            "reference, in kW",
                "interval": "percentile bootstrap over the sealed rows, two-sided 95 %",
                "what_it_measures": "row-sampling uncertainty of the same difference on this outer holdout; the rows "
                                    "are a contiguous week of one household and are not independent, so this interval "
                                    "is optimistic by an amount this tool does not estimate",
            },
        }
        comparisons[name]["paired_over_rows"]["excludes_zero"] = bool(
            comparisons[name]["paired_over_rows"]["low"] > 0 or comparisons[name]["paired_over_rows"]["high"] < 0)

    payload = {
        "schema": SCHEMA,
        "split": {"path": str(args.split), "outer_seal": split["outer"]["seal"],
                  "protocol_digest": split["outer"]["protocol_digest"],
                  "outer_origins": split["outer"]["sealed_origins"], "inner_rows": split["inner"]["rows"],
                  "residual": split["rule"]["residual"]["code"]},
        "held_fixed": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                       "core": "as each spec declares (all four declare fused_branches with the tcn encoder)",
                       "deterministic_ops": True, "seeds": args.seeds,
                       "why": "everything but the representation and the seed is identical across stages; a "
                              "hyper-parameter moved for one stage would make the difference unreadable"},
        "reference": args.reference,
        "stages": {name: {"spec": str(spec)} for name, spec in stages},
        "per_stage": per_stage,
        "comparisons": comparisons,
        "runs": {name: {str(seed): record for seed, record in seeds.items()} for name, seeds in results.items()},
        "execution_authorized": False,
    }
    (args.out_dir / "confirmation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out_dir / "confirmation.json"),
                      "reference_mean_mae": sum(reference_maes) / len(reference_maes) if reference_maes else None,
                      "comparisons": {name: {"mean_difference": value["paired_over_seeds"].get("mean"),
                                             "excludes_zero": value["paired_over_seeds"]["excludes_zero"]}
                                      for name, value in comparisons.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
