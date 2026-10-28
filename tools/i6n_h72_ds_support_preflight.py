#!/usr/bin/env python3
"""Verify every H72-DS-IN window against the frozen physical support.

This command performs no fit and writes no campaign state.  It authenticates
the four fold cells, materializes each admitted input grid and requires exact
equality with the frozen endpoint support and paired 72-hour naive.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

from tools import i6n_historical_evaluator as evaluator
from tools import i6n_island_plan as island_plan
from tools import i6n_search as search
from tools import i6n_search_space as space


def _write_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def verify(config_path, data_root):
    document = json.loads(Path(config_path).read_text(encoding="utf-8"))
    domains = document.get("domains")
    if not isinstance(domains, list) or len(domains) != 1:
        raise ValueError("config must contain exactly one domain")
    opt = domains[0]["optimization_config"]
    if opt.get("i6n_genotype_constraints") != island_plan.GENOTYPE_CONSTRAINTS_V4_DS_IN:
        raise ValueError("config is not the sealed H72-DS-IN constraint")
    folds = opt.get("i6n_folds")
    if not isinstance(folds, dict) or len(folds) != 4:
        raise ValueError("config must bind exactly four folds")

    windows = []
    for history in space.history_domain(72):
        for delta_x in space.delta_x_domain(72, history):
            samples = 1 + history // delta_x
            if 2 <= delta_x <= 72 and samples <= 55:
                windows.append((history, delta_x))

    previous = os.environ.get(evaluator.DATA_ROOT_ENV)
    os.environ[evaluator.DATA_ROOT_ENV] = str(Path(data_root).resolve())
    checked = 0
    try:
        for binding in folds.values():
            context = copy.deepcopy(binding["i6n_evaluation_context"])
            if context.get("experiment") != island_plan.EXPERIMENT_V4_DS_IN:
                raise ValueError("fold is not bound to the H72-DS-IN experiment")
            for history, delta_x in windows:
                for transform in space.INPUT_TRANSFORMS:
                    genotype = copy.deepcopy(space.seed_genotype(72))
                    genotype.update(
                        history_hours=history,
                        delta_x_hours=delta_x,
                        input_transform=transform,
                    )
                    genotype = space.canonicalize(genotype)
                    effective = search.effective_config(space.to_evaluator_genome(genotype))
                    request = {
                        "config": effective,
                        "generation": 0,
                        "context": context,
                        "support": binding["i6n_support"],
                    }
                    cfg, frozen = evaluator.load_request(request)
                    data = evaluator.prepare_candidate(cfg, request)
                    evaluator.check_support(data, cfg, frozen)
                    checked += 1
    finally:
        if previous is None:
            os.environ.pop(evaluator.DATA_ROOT_ENV, None)
        else:
            os.environ[evaluator.DATA_ROOT_ENV] = previous
    return {
        "schema": "i6n.h72_ds_in.support_preflight.v1",
        "status": "PASS",
        "folds": len(folds),
        "windows": [{"history_hours": h, "delta_x_hours": dx} for h, dx in windows],
        "input_transforms": list(space.INPUT_TRANSFORMS),
        "cells_checked": checked,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = verify(args.config, args.data_root)
    if args.output:
        _write_report(args.output, report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
