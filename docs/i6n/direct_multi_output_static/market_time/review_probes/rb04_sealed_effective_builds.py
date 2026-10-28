#!/usr/bin/env python3
"""RB04 - can the SEALED effective config actually build its model?

The trainer builds the network with exactly ``model.build_dmo(effective["model"])``
(tools/i6dmo/train.py, train_unit). So the sealed effective config of each
market-time family must produce a model block that ``normalize_model_config``
accepts. This probe asks only that, for both sealed designs, and carries the
positive control that proves the refusal is about the sealed block and not
about market-time models in general.

CPU only, no fit, no TEST row, no network. Exit 0; JSON on stdout is the verdict.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.getcwd())

from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4  # noqa: E402
from tools.i6dmo import design as D, model as M                           # noqa: E402

ROOT = "docs/i6n/direct_multi_output_static"
R = {"probe": "RB04_sealed_effective_builds", "families": {}}

for name, family in (("EXPERIMENT.mg2.json", "short_mg2"),
                     ("EXPERIMENT.mg12.json", "long_mg12")):
    row = {}
    eff = D.effective_config(D.load_design(f"{ROOT}/{name}"))
    row["effective_config_sha256"] = eff["effective_config_sha256"]
    row["family"] = eff.get("family")
    row["top_level_offsets_hours"] = eff.get("offsets_hours")
    block = eff["model"]
    row["model_block_keys"] = sorted(block)
    row["keys_not_accepted_by_normalize"] = sorted(set(block) - M.MARKET_KEYS)
    for label, call in (("normalize_model_config", lambda: M.normalize_model_config(block)),
                        ("build_dmo", lambda: M.build_dmo(block))):
        try:
            call()
            row[label] = "ACCEPTED"
        except Exception as exc:                        # noqa: BLE001
            row[label] = type(exc).__name__ + ":" + str(exc)[:180]
    # positive control: the model lane's own constructor for the same family
    try:
        built = M.build_dmo(M.model_config(list(MANIFEST_NO_STLFSI4), family=family))
        row["control_model_config_builds"] = "ACCEPTED"
        row["control_core_input_channels"] = int(built.core_model.input_shape[-1])
        row["control_block_keys"] = sorted(built.config)
    except Exception as exc:                            # noqa: BLE001
        row["control_model_config_builds"] = type(exc).__name__ + ":" + str(exc)[:180]
    row["verdict"] = "PASS" if (row["build_dmo"] == "ACCEPTED"
                                and row["control_model_config_builds"] == "ACCEPTED") else "FAIL"
    R["families"][name] = row

R["verdict"] = {k: v["verdict"] for k, v in R["families"].items()}
R["consequence"] = (
    "train_unit builds the network with build_dmo(effective['model']); if that is "
    "refused, the sealed market-time identity cannot produce a fit at all, "
    "independently of the data bundle it is given.")
print(json.dumps(R, indent=2, sort_keys=True, default=str))
