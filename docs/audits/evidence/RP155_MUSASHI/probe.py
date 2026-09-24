"""Bounded CPU review of actual RP155 scoring orchestration and window adapter.

Child results are published fixtures, not new model inference. The subprocess
transport is replaced to attack acceptance, while the real score_contrast runs.
Window values encode row identity to measure target-support intersection through
the actual _TrainWindows implementation. No GPU, services or original run writes.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("rp155_candidate", args.candidate / "tools/df_ecl_modular.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = args.candidate / "docs/audits/evidence/d3_k5_20260917/RP155"
    contrast = json.loads((source / "CONTRAST.json").read_text())
    scoring = json.loads((source / "SCORING.json").read_text())
    n = contrast["populations"]["validation_windows_available"]
    monitor = contrast["populations"]["validation_monitor_windows"]
    horizon, width = contrast["pred_len"], module.SEQ_LEN
    results = {"candidate": "ebe35fb3", "scope": __doc__, "cases": {}}

    class IdentityDataset:
        def __getitem__(self, origin):
            x = np.arange(origin, origin + width, dtype=np.float32)[:, None]
            y = np.arange(origin + width - module.LABEL_LEN,
                          origin + width + horizon, dtype=np.float32)[:, None]
            return x, y, None, None

    window = module._windows_class(SimpleNamespace(keras=SimpleNamespace(utils=SimpleNamespace(PyDataset=object))))

    def targets(indices):
        seq = window(IdentityDataset(), indices, seq_len=width, pred_len=horizon,
                     batch=len(indices), seed=0, complete=True)
        return seq[0][1].astype(np.int64)[:, :, 0]

    selected = targets(list(range(monitor)))
    complement = targets(list(range(monitor, n)))
    common = set(selected.ravel()) & set(complement.ravel())
    intersections = [len(set(row) & common) for row in complement]
    assert len(common) == 95 and sum(c > 0 for c in intersections) == 95
    disjoint_start = next(monitor + i for i, count in enumerate(intersections) if count == 0)
    assert not (set(selected.ravel()) & set(targets(list(range(disjoint_start, n))).ravel()))
    results["target_support"] = {
        "monitor_windows": monitor, "reported_complement_windows": n - monitor,
        "shared_target_timestamps": len(common), "complement_windows_with_shared_labels": sum(c > 0 for c in intersections),
        "repeated_target_channel_elements_in_complement": sum(intersections) * module.CHANNELS,
        "first_label_disjoint_local_origin": disjoint_start,
        "label_disjoint_windows": n - disjoint_start,
        "scope": "Row/channel identity, not statistical independence; causal input-context reuse need not be removed"}

    original_rows = scoring["cells"]
    for case in ("positive", "missing_registered_cell", "changed_checkpoint", "empty_population"):
        run = copy.deepcopy(contrast)
        if case == "missing_registered_cell":
            run["cells"].pop("R2_s2023")
        if case == "empty_population":
            run["cells"] = {}

        def child(command, **kwargs):
            code = command[-1]
            name = next(k for k in original_rows if f",{k!r}," in code)
            row = copy.deepcopy(original_rows[name])
            if case == "changed_checkpoint" and name == "R2_s2023":
                row["model_identity_reconciled"] = False
                row["model_sha256_on_disk"] = "0" * 64
            return subprocess.CompletedProcess(command, 0, json.dumps(row) + "\n", "")

        with tempfile.TemporaryDirectory(prefix="rp155-review-") as tmp:
            directory = Path(tmp)
            (directory / "CONTRAST.json").write_text(json.dumps(run))
            with patch("subprocess.run", side_effect=child):
                output = module.score_contrast(Path("not-read.csv"), directory)
            results["cases"][case] = {
                "problems": output["problems"], "reconciliation": output["reconciliation"],
                "regimes_summarized": sorted(output["by_regime"]),
                "seeds_summarized": {r: v["complete_validation"]["seeds"] for r, v in output["by_regime"].items()}}
            assert output["problems"] == []
            if case == "missing_registered_cell":
                assert output["reconciliation"]["cells_expected"] == 8
                assert output["by_regime"]["R2"]["complete_validation"]["seeds"] == 2
            if case == "changed_checkpoint":
                assert output["reconciliation"]["all_model_identities_match"] is False
                assert output["by_regime"]["R2"]["complete_validation"]["seeds"] == 3
            if case == "empty_population":
                assert output["reconciliation"]["all_model_identities_match"] is True

    population = "validation_never_used_for_selection"
    values = {r: np.array([original_rows[f"{r}_s{s}"]["populations"][population]["model"]["mae"]
                          for s in contrast["seeds"]]) for r in ("R0", "R1", "R2")}
    results["reported_metric_recalculation"] = {
        r: {"mean": float(v.mean()), "sd_ddof1": float(v.std(ddof=1))} for r, v in values.items()}
    results["reported_metric_recalculation"]["paired_R2_minus_R0"] = (values["R2"] - values["R0"]).tolist()
    results["reported_metric_recalculation"]["paired_R1_minus_R0"] = (values["R1"] - values["R0"]).tolist()
    results["reported_metric_recalculation"]["R2_relative_MAE_reduction"] = float(1 - values["R2"].mean() / values["R0"].mean())
    results["reported_metric_recalculation"]["scope"] = "Recomputed from published per-cell scores, not from prediction arrays"
    text = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
