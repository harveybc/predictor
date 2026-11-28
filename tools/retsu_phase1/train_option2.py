#!/usr/bin/env python3
"""Bounded DEAP search for the seasonal encoder. Reads OPTION2_PLAN.md.

Horizon 6 is scored on the inner train cut. H2 2023 and 2024 are used only
after the winner is chosen. The transfer to 24h is one refit, not a second search.
"""
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from deap import base, creator, tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/var/tmp/retsu-phase1")
import train_h12_h36 as h12  # noqa: E402
import train_phase2_step4 as step4  # noqa: E402

step3 = step4.step3
from app.structural_refit import build_model  # noqa: E402

INNER = "inner train [2023-01-01, 2023-07-01)"
MONITOR = "H2 2023 monitor"


def load_plan(path: Path) -> dict:
    text = path.read_text()
    marker = "```option2\n"
    start = text.index(marker) + len(marker)
    plan = json.loads(text[start : text.index("\n```", start)])
    if plan.get("schema") != "retsu_option2.v1":
        raise SystemExit("PLAN_SCHEMA")
    if plan.get("search_horizon") != 6 or plan.get("transfer_horizon") != 24:
        raise SystemExit("PLAN_HORIZONS")
    if plan.get("population") != 8 or plan.get("generations") != 2 or plan.get("max_trainings") != 24:
        raise SystemExit("PLAN_BUDGET")
    if plan.get("baseline") != [1, 0, 1, 0, 0, 0, 0, 0]:
        raise SystemExit("PLAN_BASELINE")
    return plan


def stamp(text: str) -> int:
    return int(np.datetime64(text.replace("Z", ""), "ns").astype(np.int64))


def decode(plan: dict, genes: list[int]) -> dict:
    decoded = {}
    for index, (name, values) in enumerate(plan["genes"]):
        decoded[name] = values[genes[index]]
    if decoded["core_kind"] == "pointwise":
        decoded["effective_kernel"] = 1
    else:
        decoded["effective_kernel"] = decoded["core_kernel"]
    return decoded


def effective_for(base: dict, recipe: dict, decoded: dict) -> dict:
    effective = copy.deepcopy(base)
    arch = effective["architecture"]
    blocks = int(decoded["core_blocks"])
    arch["core_blocks"] = blocks
    arch["core_width"] = 16
    arch["core_dropout"] = float(decoded["core_dropout"])
    arch["core_config"] = {
        "width": 16,
        "blocks": blocks,
        "kind": decoded["core_kind"],
        "kernel": int(decoded["effective_kernel"]),
        "dilation_base": int(decoded["core_dilation_base"]),
        "activation": decoded["core_activation"],
    }
    if decoded["branch_mode"] == "pointwise":
        count = len(recipe["representation"]["resolved"]["indices"])
        arch["group_configs"] = [
            {"width": 8, "blocks": 1, "kind": "pointwise", "kernel": 1, "dilation_base": 1, "activation": "silu"}
            for _ in range(count)
        ]
    effective["critic_regularization"] = {"kind": "none"}
    return effective


def build_fresh(recipe, effective):
    state = random.getstate()
    numpy_state = np.random.get_state()
    model = build_model(recipe, effective, seed=7, device="cuda")
    random.setstate(state)
    np.random.set_state(numpy_state)
    width = model.actor_net.encoder.features_dim
    if width != 208:
        raise SystemExit(f"FEATURES_DIM {width}")
    return model


def key_of(genes: list[int]) -> str:
    return "-".join(str(value) for value in genes)


def assert_disjoint(label: str, parts: dict) -> None:
    named = [(name, set(np.asarray(idx).tolist())) for name, idx in parts.items()]
    for i, (left, a) in enumerate(named):
        for right, b in named[i + 1 :]:
            if a & b:
                raise SystemExit(f"SPLIT_OVERLAP {label} {left} {right}")


def assert_finite(token: str, values: np.ndarray, idx: np.ndarray) -> None:
    if len(idx) == 0 or not np.isfinite(values[idx]).all():
        raise SystemExit(token)


def better(left: dict, right: dict, min_delta: float) -> bool:
    if left["fitness"] > right["fitness"] + min_delta:
        return True
    if abs(left["fitness"] - right["fitness"]) > min_delta:
        return False
    return (
        left["decoded"]["core_blocks"],
        left["decoded"]["encoder_lr"],
        left["decoded"]["core_dropout"],
        0 if left["decoded"]["core_kind"] == "tcn" else 1,
        0 if left["decoded"]["branch_mode"] == "shared" else 1,
    ) < (
        right["decoded"]["core_blocks"],
        right["decoded"]["encoder_lr"],
        right["decoded"]["core_dropout"],
        0 if right["decoded"]["core_kind"] == "tcn" else 1,
        0 if right["decoded"]["branch_mode"] == "shared" else 1,
    )


def train_genome(recipe, base_effective, windows, y, fit_idx, es_idx, plan, out: Path, genes, horizon: int, encoder_lr, max_epochs, split_name) -> dict:
    # build_model and make_head call torch.manual_seed, which also reseeds
    # Python and NumPy. Put that draw back so DEAP keeps one stream and a
    # resume walks the same genomes.
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        decoded = decode(plan, genes)
        effective = effective_for(base_effective, recipe, decoded)
        model = build_fresh(recipe, effective)
        trainable = step4.encoder_trainable(model, "full_encoder")
        head = step4.make_head("linear", model.actor_net.encoder.features_dim, [horizon], float(horizon))
        cell_plan = dict(plan)
        cell_plan["max_epochs"] = max_epochs
        done = step4.train_cell(
            model, head, trainable, windows, y, fit_idx, es_idx, cell_plan, out, out.name,
            [horizon], float(encoder_lr), split_name,
        )
        del model, head
        torch.cuda.empty_cache()
        return done
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def evaluate(recipe, base_effective, windows, y, fit_idx, es_idx, plan, root: Path, genes, naive: float) -> dict:
    decoded = decode(plan, genes)
    folder = root / "search" / key_of(genes)
    if not step3.job_done(folder):
        train_genome(
            recipe, base_effective, windows, y, fit_idx, es_idx, plan, folder, genes,
            int(plan["search_horizon"]), decoded["encoder_lr"], int(plan["search_max_epochs"]), INNER,
        )
    done = json.loads((folder / "STATUS.json").read_text())
    fitness = float(naive - done["champion_val_mae"])
    record = {"genes": list(genes), "decoded": decoded, "fitness": fitness, "val_mae": done["champion_val_mae"], "best_epoch": done["best_epoch"]}
    print(f"EVAL {key_of(genes)} fitness {fitness:.8f} val {done['champion_val_mae']:.8f} epoch {done['best_epoch']}", flush=True)
    return record


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUIRED")
    plan = load_plan(Path(sys.argv[sys.argv.index("--plan") + 1]))
    out = Path(plan["out_dir"]).resolve()
    for finished in plan["finished_dirs"]:
        finished_path = Path(finished).resolve()
        if out == finished_path or str(out).startswith(str(finished_path) + "/"):
            raise SystemExit("REFUSING_FINISHED_DIR")
    if (out / "OPTION2_DONE").exists():
        print("OPTION2_DONE", flush=True)
        return
    phase = Path(plan["phase2_dir"])
    recipe = json.loads((phase / "RECIPE.json").read_text())
    if recipe["identity"] != plan["recipe_identity"] or recipe["list"]["n"] != 164:
        raise SystemExit("RECIPE_IDENTITY")
    resolved = recipe["representation"]["resolved"]
    if len(resolved["indices"]) != 77 or len(resolved["names"]) != 77:
        raise SystemExit("RECIPE_GROUPS")
    data = np.load(phase / "dataset.npz")
    loaded = np.load(phase / "windows.npz")
    if not np.array_equal(loaded["ts"], data["ts"]):
        raise SystemExit("WINDOWS_TS")
    windows = loaded["windows"]
    nonzero = np.any(windows != 0, axis=(1, 2))
    if not np.all(nonzero[data["ok_short"]]):
        raise SystemExit("ZERO_WINDOW_ON_SHORT")
    ts = data["ts"].astype(np.int64)
    score_start = stamp(plan["score_start"])
    monitor_start = stamp(plan["monitor_start"])
    holdout_start = stamp(plan["holdout_start"])
    short = data["ok_short"] & nonzero
    if int((data["monitor"] & short).sum()) != 1506:
        raise SystemExit("SHORT_MONITOR_ROWS")
    # y_short columns are horizons 1..6, so column 5 is the 6h increment.
    y6 = data["y_short"][:, 5:6]
    search_fit = np.flatnonzero(data["pool"] & short & (ts < score_start))
    search_es = np.flatnonzero(data["pool"] & short & (ts >= score_start) & (ts < monitor_start))
    final_fit = np.flatnonzero(data["pool"] & short)
    monitor = np.flatnonzero(data["monitor"] & short)
    hold6 = np.flatnonzero(data["holdout"] & short)
    if len(search_fit) < int(plan["min_pool"]) or len(search_es) < int(plan["min_monitor"]):
        raise SystemExit(f"ROWS_SEARCH {len(search_fit)} {len(search_es)}")
    series = np.load(plan["stl_series"])
    by_avail = {int(value): i for i, value in enumerate(series["avail_ns"].astype(np.int64))}
    bundle = h12.exp.ROOT / "model/61/bundle"
    joined, real = h12.exp.assemble(h12.exp.PARENT, bundle, h12.exp.ROOT / "history397-61-r2")
    risk = json.loads((h12.exp.PARENT / "RISK.json").read_text())
    bounds = {
        **h12.exp.segment_bounds(joined, risk, "2016-01-01", "2024-01-01"),
        **h12.exp.segment_bounds(real, risk, "2024-01-01", "2025-01-01"),
    }
    increment, admitted = h12.labels_for(ts, series["seasonal"].astype(np.float64), by_avail, bounds, 24)
    y24 = increment[:, None].astype(np.float32)
    mask24 = admitted & nonzero
    parts24 = {
        "fit": np.flatnonzero(data["pool"] & mask24),
        "monitor": np.flatnonzero(data["monitor"] & mask24),
        "hold": np.flatnonzero(data["holdout"] & mask24),
    }
    keep = float(plan["row_keep_fraction"])
    declared24 = {"fit": int(plan["alone_pool_24"]), "monitor": int(plan["alone_monitor_24"]), "hold": int(plan["alone_holdout_24"])}
    counts24 = {name: int(len(idx)) for name, idx in parts24.items()}
    print(f"ROWS h6 search_fit {len(search_fit)} search_es {len(search_es)} refit {len(final_fit)} monitor {len(monitor)} hold {len(hold6)}", flush=True)
    print(f"ROWS h24 fit {counts24['fit']} monitor {counts24['monitor']} hold {counts24['hold']}", flush=True)
    if counts24["fit"] > declared24["fit"] or counts24["fit"] < int(plan["min_pool"]):
        raise SystemExit(f"ROW_DRIFT 24 fit {counts24['fit']}")
    for name in ("monitor", "hold"):
        if counts24[name] > declared24[name] or counts24[name] < declared24[name] * keep:
            raise SystemExit(f"ROW_DRIFT 24 {name} {counts24[name]} {declared24[name]}")
    assert_disjoint("h6", {"search_fit": search_fit, "search_es": search_es, "monitor": monitor, "hold": hold6})
    assert_disjoint("h6refit", {"fit": final_fit, "monitor": monitor, "hold": hold6})
    assert_disjoint("h24", parts24)
    for name, idx in (("search_fit", search_fit), ("search_es", search_es), ("refit", final_fit), ("monitor", monitor), ("hold", hold6)):
        assert_finite(f"Y6_NONFINITE {name}", y6, idx)
    for name, idx in parts24.items():
        assert_finite(f"Y24_NONFINITE {name}", y24, idx)
    naive6 = float(np.mean(np.abs(y6[search_es, 0])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "DECLARED.json").write_text(json.dumps({
        "schema": plan["schema"],
        "naive_search": naive6,
        "rows_h6": {"search_fit": int(len(search_fit)), "search_es": int(len(search_es)), "refit": int(len(final_fit)), "monitor": int(len(monitor)), "hold": int(len(hold6))},
        "rows_h24": counts24,
        "holdout_2024": "not predicted until both refits exist",
        "score_start": plan["score_start"],
        "monitor_start": plan["monitor_start"],
        "holdout_start": plan["holdout_start"],
    }, indent=2) + "\n")
    print("OPTION2_DECLARED", flush=True)
    del holdout_start
    base_effective = json.loads(step3.MODEL_CONFIG.read_text())
    if not (out / "search" / "SEARCH_DONE").exists():
        random.seed(int(plan["search_seed"]))
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        cache = {}
        trained = 0

        def cached(genes: list[int]) -> dict:
            nonlocal trained
            token = key_of(genes)
            if token not in cache:
                folder = out / "search" / token
                if not step3.job_done(folder):
                    if trained >= int(plan["max_trainings"]):
                        raise SystemExit("PLAN_BUDGET_EXHAUSTED")
                    trained += 1
                cache[token] = evaluate(recipe, base_effective, windows, y6, search_fit, search_es, plan, out, genes, naive6)
                with (out / "search" / "evals.jsonl").open("a") as handle:
                    handle.write(json.dumps(cache[token]) + "\n")
            return cache[token]

        (out / "search").mkdir(parents=True, exist_ok=True)
        population = [creator.Individual(plan["baseline"])]
        while len(population) < int(plan["population"]):
            population.append(creator.Individual(random.randrange(len(values)) for _, values in plan["genes"]))
        for ind in population:
            ind.fitness.values = (cached(list(ind))["fitness"],)
        for _generation in range(int(plan["generations"])):
            offspring = tools.selTournament(population, len(population), tournsize=int(plan["tournament"]))
            offspring = [creator.Individual(list(ind)) for ind in offspring]
            for left, right in zip(offspring[::2], offspring[1::2]):
                if random.random() < float(plan["crossover_probability"]):
                    tools.cxTwoPoint(left, right)
                    del left.fitness.values, right.fitness.values
            for ind in offspring:
                if random.random() < float(plan["mutation_probability"]):
                    gene = random.randrange(len(plan["genes"]))
                    choices = [value for value in range(len(plan["genes"][gene][1])) if value != ind[gene]]
                    ind[gene] = random.choice(choices)
                    if ind.fitness.valid:
                        del ind.fitness.values
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = (cached(list(ind))["fitness"],)
            population = offspring
        records = list(cache.values())
        winner = records[0]
        for record in records[1:]:
            if better(record, winner, float(plan["min_delta"])):
                winner = record
        (out / "search" / "WINNER.json").write_text(json.dumps(winner, indent=2) + "\n")
        (out / "search" / "SEARCH_DONE").write_text(f"trainings {trained}\n")
        print(f"SEARCH_DONE {key_of(winner['genes'])} fitness {winner['fitness']:.8f}", flush=True)
    winner = json.loads((out / "search" / "WINNER.json").read_text())
    genes = list(winner["genes"])
    decoded = decode(plan, genes)
    refits = {
        6: (y6, final_fit, monitor, hold6, float(decoded["encoder_lr"])),
        24: (y24, parts24["fit"], parts24["monitor"], parts24["hold"], float(plan["transfer_encoder_lr"])),
    }
    for horizon, (target, fit_idx, es_idx, _hold_idx, rate) in refits.items():
        folder = out / "refit" / f"h{horizon}"
        if step3.job_done(folder):
            continue
        train_genome(
            recipe, base_effective, windows, target, fit_idx, es_idx, plan, folder, genes,
            horizon, rate, int(plan["refit_max_epochs"]), MONITOR,
        )
    for horizon, (target, _fit_idx, _es_idx, hold_idx, _rate) in refits.items():
        hold_path = out / "refit" / f"h{horizon}" / "holdout_2024.json"
        if hold_path.exists():
            print(f"HOLDOUT_2024 h{horizon} already written, not predicted again", flush=True)
            continue
        bundle_pt = torch.load(out / "refit" / f"h{horizon}" / "champion.pt", map_location="cpu", weights_only=False)
        model = build_fresh(recipe, effective_for(base_effective, recipe, decoded))
        head = step4.make_head("linear", model.actor_net.encoder.features_dim, [horizon], float(horizon))
        head.load_state_dict(bundle_pt["head"])
        step4.restore_encoder(model, bundle_pt["encoder"])
        record = bundle_pt["record"]
        measured = step4.predict_holdout(
            model, model.loss.actor_network_params, head, float(record["embedding_scale"]),
            windows, hold_idx, target[hold_idx], np.asarray(record["target_mean"], dtype=np.float64),
            np.asarray(record["target_std"], dtype=np.float64), int(plan["batch"]),
        )
        hold_path.write_text(json.dumps({
            "horizon": horizon,
            "definition": "MAE of the seasonal increment in price units. The naive is zero increment, seasonal-level persistence.",
            "split": "2024 holdout, one predict after both refits",
            "rows": measured["rows"],
            "mae": measured["mae"],
            "naive": measured["naive"][0],
            "champion_epoch": record["best_epoch"],
            "validation_h2_2023_mae": record["champion_val_mae"],
            "validation_rows": record["validation_rows"],
            "genes": genes,
            "encoder_lr": record["encoder_lr"],
        }, indent=2) + "\n")
        print(f"HOLDOUT_2024 h{horizon} mae {measured['mae']:.8f} rows {measured['rows']}", flush=True)
        del model, head
        torch.cuda.empty_cache()
    (out / "OPTION2_DONE").write_text("one 2024 predict per refit horizon\n")
    print("OPTION2_DONE", flush=True)


if __name__ == "__main__":
    main()
