"""Retained-child validator checks only; no subprocess inference, GPU or data read."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("candidate", args.candidate / "tools/df_ecl_modular.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = args.candidate / "docs/audits/evidence/d3_k5_20260917"
    current = json.loads((root / "RP158_SCORING.json").read_text())
    previous = json.loads((root / "RP157_SCORING.json").read_text())
    run = json.loads((root / "RP155/CONTRAST.json").read_text())
    cells = current["cells"]
    kwargs = dict(expected=sorted(cells), pred_len=96,
                  populations=("complete_validation", "label_disjoint_from_selection"),
                  reductions=("author_float32", "independent_float64"),
                  expected_windows={"complete_validation": 2537, "label_disjoint_from_selection": 1802},
                  checkpoints={c: run["cells"][c]["model_sha256"] for c in cells})
    results = {"scope": __doc__, "candidate": "e36c1c6d", "cases": {}}
    results["retained_positive"] = module.validate_children(cells, **kwargs)
    assert results["retained_positive"]["bound"]
    results["population_metrics_equal_RP157"] = all(
        cells[c]["populations"] == previous["cells"][c]["populations"] for c in cells)
    assert results["population_metrics_equal_RP157"]
    for case in ("all_counts_one", "naive_nan", "identity_string_false",
                 "elements_absent", "elements_string", "skill_wrong_finite", "negative_loss"):
        altered = copy.deepcopy(cells)
        row = altered["R2_s2023"]
        entry = row["populations"]["label_disjoint_from_selection"]
        if case == "all_counts_one":
            for child in altered.values():
                for population in child["populations"].values():
                    population["windows"] = 1
        elif case == "naive_nan":
            entry["matched_persistence_author_float32"]["mae"] = float("nan")
        elif case == "identity_string_false":
            row["model_identity_reconciled"] = "false"
        elif case == "elements_absent":
            entry.pop("elements")
        elif case == "elements_string":
            entry["elements"] = "not-an-element-count"
        elif case == "skill_wrong_finite":
            entry["skill_mae_vs_persistence"] = 0.999
        elif case == "negative_loss":
            entry["author_float32"]["mae"] = -0.1
        result = module.validate_children(altered, **kwargs)
        results["cases"][case] = result
        assert result["bound"] is (case in {
            "elements_absent", "elements_string", "skill_wrong_finite", "negative_loss"})
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"retained_valid": results["retained_positive"]["bound"],
                      "unchanged": results["population_metrics_equal_RP157"],
                      "accepted_mutations": [k for k, v in results["cases"].items() if v["bound"]]}))


if __name__ == "__main__":
    main()
