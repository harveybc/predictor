"""RP156 independent CPU acceptance probes; no model inference or live resources.

Real scoring orchestration with published child fixtures at the subprocess
boundary. Assertions characterize repairs and remaining acceptance defects.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("candidate", args.candidate / "tools/df_ecl_modular.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    evidence = args.candidate / "docs/audits/evidence/d3_k5_20260917"
    contrast = json.loads((evidence / "RP155/CONTRAST.json").read_text())
    scored = json.loads((evidence / "RP156_SCORING.json").read_text())
    design = {"design_sha256": scored["registered_design_sha256"],
              "factorial": {"cells": scored["expected_cells"]}}
    output = {"candidate": "b3a064d5", "scope": __doc__, "cases": {}}
    support = m.label_disjoint_origins(2537, 640, pred_len=96)
    assert support["first_disjoint_origin"] == 735 and support["n_disjoint"] == 1802
    monitor_targets = {o + 96 + h for o in range(640) for h in range(96)}
    evaluation_targets = {o + 96 + h for o in support["disjoint_origins"] for h in range(96)}
    assert not monitor_targets & evaluation_targets
    output["support"] = {k: v for k, v in support.items() if not k.endswith("origins")}

    for case in ("baseline", "missing_cell", "identity_false", "foreign_design_digest",
                 "empty_declared_design", "missing_population", "foreign_horizon", "nan_metric"):
        run, declared = copy.deepcopy(contrast), copy.deepcopy(design)
        if case == "missing_cell":
            run["cells"].pop("R2_s2023")
        if case == "foreign_design_digest":
            declared["design_sha256"] = "f" * 64
        if case == "empty_declared_design":
            declared["factorial"]["cells"] = []

        def child(command, **kwargs):
            name = next(k for k in scored["cells"] if f",{k!r}," in command[-1])
            row = copy.deepcopy(scored["cells"][name])
            if name == "R2_s2023":
                if case == "identity_false":
                    row["model_identity_reconciled"] = False
                elif case == "missing_population":
                    row["populations"].pop("label_disjoint_from_selection")
                elif case == "foreign_horizon":
                    row["pred_len"] = 192
                elif case == "nan_metric":
                    row["populations"]["label_disjoint_from_selection"]["author_float32"]["mae"] = float("nan")
            return subprocess.CompletedProcess(command, 0, json.dumps(row) + "\n", "")

        with tempfile.TemporaryDirectory(prefix="rp156-review-") as root:
            directory = Path(root)
            (directory / "CONTRAST.json").write_text(json.dumps(run))
            with patch("subprocess.run", side_effect=child):
                result = m.score_contrast(Path("not-read.csv"), directory, design=declared)
        expected = "INCOMPLETE_EVIDENCE" if case in ("missing_cell", "identity_false") else "COMPLETE"
        assert result["status"] == expected
        summary = result["by_regime"]
        output["cases"][case] = {
            "status": result["status"], "problems": result["problems"],
            "design_digests_equal": result["design_sha256"] == result["registered_design_sha256"],
            "reconciliation": result["reconciliation"], "summary_present": summary is not None,
            "label_disjoint_seeds": {r: v.get("label_disjoint_from_selection", {}).get("seeds")
                                     for r, v in (summary or {}).items()},
            "summary_contains_nan": bool(summary and any(
                not np.isfinite(v["label_disjoint_from_selection"]["author_float32"]["mae_mean"])
                for v in summary.values()))}
        if case in ("missing_cell", "identity_false"):
            assert summary is None
        if case == "missing_population":
            assert summary["R2"]["label_disjoint_from_selection"]["seeds"] == 2

    with tempfile.TemporaryDirectory(prefix="rp156-checkpoint-") as root:
        directory = Path(root)
        file = directory / "weights.h5"
        file.write_bytes(b"deliberately not model weights")
        run = copy.deepcopy(contrast)
        run["cells"]["R0_s2021"].update(model_path=str(file), model_sha256="0" * 64)
        (directory / "CONTRAST.json").write_text(json.dumps(run))
        touched = []
        engine = SimpleNamespace(_tf=lambda: touched.append(True))
        with patch.object(m, "_sota", return_value=None), patch.object(m, "_e0", return_value=engine), patch.object(m, "_regimes", return_value=None):
            try:
                m.score_cell(Path("not-read.csv"), directory, "R0_s2021")
            except ValueError as error:
                assert "REFUSED" in str(error) and not touched
                output["checkpoint_mismatch_before_model"] = True
            else:
                raise AssertionError("mismatched weights were not refused")

    values = {r: np.array([scored["cells"][f"{r}_s{s}"]["populations"]["label_disjoint_from_selection"]["author_float32"]["mae"]
                          for s in (2021, 2022, 2023)]) for r in ("R0", "R1", "R2")}
    output["published_arithmetic_not_array_reduction"] = {
        r: {"mean": float(v.mean()), "sd_ddof1": float(v.std(ddof=1))} for r, v in values.items()}
    output["published_arithmetic_not_array_reduction"].update(
        paired_R2_minus_R0=(values["R2"] - values["R0"]).tolist(),
        paired_R1_minus_R0=(values["R1"] - values["R0"]).tolist(),
        R2_relative_reduction=float(1 - values["R2"].mean() / values["R0"].mean()))
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
