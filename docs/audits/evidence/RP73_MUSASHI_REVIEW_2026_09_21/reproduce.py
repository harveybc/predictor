"""Read-only audit of 1880caa; synthetic counterexamples live in temporary directories.

Run with the trading-stack interpreter and --repo pointing to the audited checkout.
No services, financial resource, reserved data or scientific training are used.
The four-update constant model is a numerical test of the production training loop.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = args.repo.resolve()


def load(name, relative=None):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, repo / (relative or f"tools/{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


import numpy as np

B = load("df_benchmark_contract")
K = load("df_e1_block")
F = load("df_fin_runner")
results = {}
with tempfile.TemporaryDirectory(prefix="rp73-audit-") as directory:
    tmp = Path(directory)
    ref = tmp / "reference"
    ref.mkdir()
    (ref / "DESIGN.json").write_text(json.dumps({
        "benchmark_contract": {"contract_sha256": B.household_ours().sha256()}}))
    (ref / "TERMINAL_RECEIPTS.json").write_text(json.dumps({
        "units": {"prepare": {"status": "COMPLETED"}}}))
    outcome = B.reference_evidence(B.household_ours(), ref)
    results["reference_without_model_or_predictions"] = {
        "state": outcome["state"], "units": outcome.get("units"),
        "has_design_digest": bool(outcome.get("design_sha256")),
        "expected": "not VERIFIED_COMPARATOR"}

    table_fixture = load("audit_table_fixture", "tests/test_df_closure_table.py")
    example = table_fixture.synthetic_root.__wrapped__(tmp / "table")
    unit = table_fixture.UNIT
    (example["root"] / "TERMINALS" / f"{unit}.json").unlink()
    def warehouse_without_artifacts(campaign):
        return {"current": {unit: {"terminal_sha256": "t" * 64,
                                  "status": "COMPLETED", "artifacts": []}}}
    before = table_fixture._table(example, warehouse=warehouse_without_artifacts)
    changed_sha = table_fixture._write_arrays(example["root"], example["Y"][example["origins"] + table_fixture.H],
        example["Y"][example["origins"] + table_fixture.H], example["origins"], Y=example["Y"])
    table_fixture._bind(example["root"], unit, changed_sha, mae=0., terminal=False)
    after = table_fixture._table(example, warehouse=warehouse_without_artifacts)
    results["unanchored_record_rewrite"] = {
        "before_mae": before["rows"][0]["model_error"],
        "after_mae": after["rows"][0]["model_error"],
        "before_verified": before["rows"][0]["verified"],
        "after_verified": after["rows"][0]["verified"],
        "problems": after["problems"],
        "warehouse_and_receipt_unchanged": True,
        "expected": "no end-to-end verified score from a mutable unanchored record"}

    fixture = load("audit_fin_fixture", "tests/test_fin_loss_opt_acceptance.py")
    frame = fixture._bars(drop_frac=0)
    design = fixture._design(frame)
    F.prepare(design, tmp / "clean", frame=frame)
    data, record = F.load_data(tmp / "clean", design)
    origin = int(data["f0_train_origins"][len(data["f0_train_origins"]) // 2])
    bad = frame.copy()
    bad.loc[origin, "volume"] = np.nan
    F.prepare(design, tmp / "nan", frame=bad)
    damaged, record = F.load_data(tmp / "nan", design)
    w = design["receiver"]["window"]
    origins = damaged["f0_train_origins"]
    affected = origins[(origins >= origin) & (origins < origin + w)]
    fold = record["folds"][0]
    scaled = ((damaged["X"] - np.asarray(fold["scaler"]["mean"])) /
              np.asarray(fold["scaler"]["sd"]))
    batch = F.PairBatches(scaled, damaged["y"], affected[:1], affected[:1] + 6,
                         w, 1, shuffle=False, seed=1)
    inputs, _ = batch[0]
    results["financial_nonfinite_window"] = {
        "retained_affected_train_windows": int(affected.size),
        "tensor_contains_nan": bool(np.isnan(inputs).any()),
        "expected": "typed refusal or exclusion before model input; common population across arms"}

    # Two folds with a block length of two have only one resampling start.
    results["two_fold_bootstrap"] = F.block_bootstrap(np.array([-0.1, 0.3]), block_len=2, n_boot=100)

    # A declared two-seed population: selection currently chooses a seed, not a paired family estimate.
    select_root = tmp / "select"
    cells, candidates = [], [{"loss": "mae"}, {"loss": "huber"}]
    for loss, values in {"mae": [(0.1, 0.4), (0.9, 0.4)], "huber": [(0.2, 0.2), (0.3, 0.2)]}.items():
        for seed, (validation, test) in enumerate(values, 1):
            unit = f"{loss}_s{seed}"
            cell = {"cell_id": unit, "fold": 0, "seed": seed}
            cells.append(cell)
            folder = select_root / "attempts" / unit
            folder.mkdir(parents=True)
            (folder / "cell.json").write_text(json.dumps({
                "cell": cell, "candidate": {"loss": loss},
                "scores": {"validation": {"mae_z": validation}, "test": {"mae_z": test}}}))
    selection = F.select(select_root, {"cells": cells, "candidates": candidates,
                         "folds": {"dev_weeks": 1}, "design_sha256": "synthetic-audit"})
    results["selection_chooses_replica"] = selection["per_fold"][0]

    tf = load("df_mod_e0")._tf()

    class Batches:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, i):
            return np.zeros((1, 1), np.float32), np.array([[self.values[i]]], np.float32)

        def on_epoch_end(self):
            pass

    model = tf.keras.Sequential([tf.keras.Input((1,)), tf.keras.layers.Dense(
        1, kernel_initializer="zeros", bias_initializer="zeros")])
    observed = K.fit_by_updates(model, Batches([4., 0., 0., 0.]), Batches([1.]),
        max_updates=4, validate_every=1, patience=3, lr=0., seed=1)
    results["training_loop"] = {k: observed[k] for k in (
        "updates", "optimizer_iterations", "stop_reason", "censoring", "events")}
    results["training_loop"]["expected_batch_losses"] = [4., 0., 0., 0.]
    results["training_loop"]["expected_budget_status"] = "CENSORED_BY_BUDGET at update 4"

# Independent accounting from retained cost pilots; excludes validation from update cost.
block = repo / "docs/audits/evidence/d3_k5_20260917/RP66/blocks/e1_block_q2_context_v1"
design = json.loads((block / "DESIGN.json").read_text())
projection = json.loads((block / "REPORT.pilot.json").read_text())["projection"]
corrected, per_arm = 0., {}
for p in sorted((block / "attempts").glob("*/cell.json")):
    r = json.loads(p.read_text())
    training = r["training"]
    # Includes final restore validation and loop overhead once, conservatively amortized.
    update_cost = (training["fit_cpu_seconds"] - training["validation_cpu_seconds"]) / training["updates"]
    arm = r["cell"]["arm"]
    n = sum(c["arm"] == arm for c in design["cells"])
    val_cost = projection["per_arm"][arm]["seconds_per_validation_event_pilot"]
    val_scale = design["source_run"]["evaluation_origins"] / r["population"]["evaluation_origins"]
    events = int(np.ceil(design["recipe"]["max_updates"] / design["recipe"]["validate_every_updates"]))
    value = update_cost * design["recipe"]["max_updates"] + val_cost * val_scale * events
    per_arm[arm] = {"training_plus_overhead_per_update": update_cost, "corrected_per_cell": value}
    corrected += n * value
results["q2_cost_double_count"] = {
    "reported_projection": projection["total_at_ceiling_seconds"],
    "validation_counted_once_projection": corrected,
    "still_above_14400": corrected > 14400,
    "per_arm": per_arm,
    "scope": "arithmetic correction, not a new cost measurement or execution authorization"}
args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
print(json.dumps(results, indent=2, allow_nan=False))
