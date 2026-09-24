"""Actual RP157 parent scorer, published child fixtures, no inference or live data.

Tests acceptance at the subprocess transport boundary; deliberately malformed
requests/results are not claims about the retained scientific rows.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("candidate", args.candidate / "tools/df_ecl_modular.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evidence = args.candidate / "docs/audits/evidence/d3_k5_20260917"
    run = json.loads((evidence / "RP155/CONTRAST.json").read_text())
    scored = json.loads((evidence / "RP157_SCORING.json").read_text())
    expected = sorted(scored["cells"])
    reduced_design = {"design_sha256": run["design_sha256"],
                      "factorial": {"cells": expected}, "task": {"pred_len": 96}}
    results = {"candidate": "3c25ecab", "scope": __doc__, "cases": {}}
    for case in ("missing_canonical_cover", "contradictory_design_horizon", "empty_canonical_cover",
                 "consistent_wrong_counts", "nonfinite_naive", "string_false_identity"):
        design = copy.deepcopy(reduced_design)
        if case == "contradictory_design_horizon":
            design["task"]["pred_len"] = 192
        if case == "empty_canonical_cover":
            design["identity_covers"] = []

        def child(command, **kwargs):
            name = next(k for k in expected if f",{k!r}," in command[-1])
            record = copy.deepcopy(scored["cells"][name])
            if case == "consistent_wrong_counts":
                for population in record["populations"].values():
                    population["windows"] = 1
            if name == "R2_s2023":
                if case == "nonfinite_naive":
                    record["populations"]["label_disjoint_from_selection"]["matched_persistence_author_float32"]["mae"] = float("nan")
                if case == "string_false_identity":
                    record["model_identity_reconciled"] = "false"
            return subprocess.CompletedProcess(command, 0, json.dumps(record) + "\n", "")

        with tempfile.TemporaryDirectory(prefix="rp157-review-") as temp:
            root = Path(temp)
            (root / "CONTRAST.json").write_text(json.dumps(run))
            with patch("subprocess.run", side_effect=child):
                result = module.score_contrast(Path("not-read.csv"), root, design=design)
        assert result["status"] == "COMPLETE"
        results["cases"][case] = {
            "status": result["status"], "problems": result["problems"],
            "authentication_claim": result["design_authentication"]["authenticated"],
            "authenticated_horizon": result["design_authentication"]["pred_len"],
            "requested_and_returned_horizon": 96,
            "windows_accepted": result["child_binding"]["windows_per_population"],
            "naive_mean_finite": bool(np.isfinite(result["by_regime"]["R2"]["label_disjoint_from_selection"]["matched_persistence_author_float32_mae"]))}

    bound = module.validate_children(scored["cells"], expected=expected, pred_len=96,
                                     populations=("complete_validation", "label_disjoint_from_selection"),
                                     reductions=("author_float32", "independent_float64"))
    assert bound["bound"]
    previous = json.loads((evidence / "RP156_SCORING.json").read_text())
    assert all(previous["cells"][c]["populations"] == scored["cells"][c]["populations"] for c in expected)
    checks = {}
    for name, windows in (("complete_validation", 2537), ("label_disjoint_from_selection", 1802)):
        rows = [scored["cells"][c] for c in expected]
        assert all(r["populations"][name]["windows"] == windows for r in rows)
        assert all(r["populations"][name]["elements"] == windows * 96 * 321 for r in rows)
        assert all(type(r["model_identity_reconciled"]) is bool and r["model_identity_reconciled"] for r in rows)
        assert all(r["model_sha256_recorded"] == r["model_sha256_on_disk"] == run["cells"][r["cell"]]["model_sha256"] for r in rows)
        assert all(np.isfinite(r["populations"][name][reduction][metric])
                   for r in rows for reduction in ("author_float32", "independent_float64", "matched_persistence_author_float32")
                   for metric in ("mae", "mse"))
        checks[name] = {"windows": windows, "elements_per_cell": windows * 96 * 321,
                       "nine_cells_consistent": True}
    results["retained_published_rows"] = {"matches_RP156_population_metrics_exactly": True,
                                           "model_digest_strings_match_prior_run_records": True,
                                           "population_checks": checks,
                                           "scope": "Published JSON records, not disk checkpoints or live authority"}
    text = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
