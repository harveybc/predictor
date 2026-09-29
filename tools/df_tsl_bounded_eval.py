#!/usr/bin/env python3
"""TRAFFIC BOUNDED EVALUATOR: separate the evaluation IMPLEMENTATION's footprint from the MODEL's, and measure the bounded path.

The claim this module exists to settle is that Traffic's two long horizons are impossible on this fleet. That claim was made from
`df_tsl_execute.eval_path_memory_derivation`, which prices the AUTHOR's unchunked `Exp_Long_Term_Forecast.test()`: three float32
Python lists, three `np.concatenate` copies and `utils.metrics.metric`'s full-size temporaries. Every one of those terms is a
property of ONE evaluation implementation. None of them is a property of the model. This module:

  split     re-derives that figure TERM BY TERM for Traffic and labels each term EVALUATION_IMPLEMENTATION or MODEL, beside the
            bounded path's own derived terms and its DISK budget. Pure arithmetic over the sealed populations; reads no bytes.
  characterize   the sealed `df_tsl_repro.characterize` on the delivered Traffic bytes, so the element counts the split uses are
            reconciled against the author's own loader rather than assumed.
  reducer-parity  the bounded float32 reduction against `utils.metrics.metric` itself, BITWISE, on populations that fit, over
            shapes chosen to exercise the two things that can break a chunked mean: an element count not exactly representable in
            float32, and a leaf boundary that falls inside a window. No model, no GPU, no data.
  probe     ONE horizon through the EXISTING bounded evaluator, with the whole-cgroup peak read from INSIDE the child and the disk
            high-water read under the work directory. With `--native-witness` the author's OWN unchunked `test()` runs FIRST in
            the same child, from the SAME on-disk checkpoint, and the two paths are compared on the digest of the arrays and on
            the metric values, bitwise. It stores NO score: the model is untrained and an untrained model's error is not a result.

  target-pairing  the CHEAP half of the writer's parity, which does NOT need the author's unchunked path to fit: a separate pass
            of the author's own test loader hashes the target windows as they stream, and that digest is compared to the one the
            bounded writer produced. It is the same identity the sealed scored-cell path already refuses a cell over.

WHAT IS NOT CHANGED, and is asserted by `tools/test_tsl_bounded_eval.py` against this module's own source:

  * the bounded evaluator itself. `df_sota_repro.bounded_test` (`df_sota_bounded_eval.v1`) and `df_sota_repro.author_metric_exact`
    (`df_sota_author_metric_exact.v2`) are IMPORTED and CALLED. This module defines no reduction, no chunked mean, no downcast and
    no sub-sampling of its own. The bounded reduction is NOT a mean of chunk means: it is numpy's own pairwise summation tree
    replayed over the same flattened element index space, with numpy's own final `float32(float64(sum)/int64(N))`.
  * the input population (every sealed test window), the model, the batch semantics (the author's own test loader, same order,
    same batch sizes), the target (`--inverse` False, the z-normalized space), the dtype (float32), the reduction and the
    checkpoint rule (`bounded_test` reloads the checkpoint from disk exactly as the author's `test(test=1)` does).
  * the declared cap. It is passed once by the caller and never shrunk here. A peak is never reduced to make a cap fit.

A memmap is NOT a proof of bounded resident memory: `memory.peak` counts file-backed pages, so the number this module reports is
the whole cgroup's high-water mark INCLUDING page cache, which is the conservative reading. A bounded evaluator that fills the
disk has moved the failure rather than removed it, so the disk high-water under the work directory and the free space before and
after are recorded beside the memory peak, and the arrays are deleted under a stated retention rule once their digests and
metrics are recorded.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402  the author bridge AND the existing bounded evaluator
import df_tsl_repro as R                                               # noqa: E402  the sealed design, characterization and horizon clock
import df_tsl_execute as X                                             # noqa: E402  the author-path derivation this module re-derives against

SPLIT_SCHEMA = "df_tsl_eval_memory_split.v1"
PARITY_SCHEMA = "df_tsl_reducer_parity.v1"
PROBE_SCHEMA = "df_tsl_bounded_probe.v1"

#: The evaluation path this module measures. It is the EXISTING one, named so a reader can check it against its own source.
BOUNDED_EVAL_PATH = ("df_sota_repro.bounded_test (df_sota_bounded_eval.v1): the author's own test loader, model, batches, order, "
                     "slicing and dtype, with the three accumulating Python lists and the three np.concatenate copies replaced by "
                     "two ordered float32 .npy files written block by block with the written pages released behind, the unused "
                     "`inputs` not retained, and the author's float32 MAE/MSE evaluated by df_sota_repro.author_metric_exact "
                     "(df_sota_author_metric_exact.v2), numpy's pairwise summation tree replayed over the same flattened element "
                     "index space. No chunked averaging, no downcast, no sub-sampling")

#: What a bounded evaluator must NOT change for its number to answer the same question. Asserted by test against this source.
INVARIANTS = ("input population: every sealed test window, the author's own loader",
              "model: the author's own, built from the sealed argv",
              "batch semantics: the author's own test loader, same order and same batch sizes",
              "target: the z-normalized target space, --inverse False",
              "dtype: float32 throughout the reduction",
              "reduction: the author's float32 np.mean over window x step x channel, replayed in numpy's own summation order",
              "checkpoint rule: the weights are reloaded from the checkpoint on disk, as the author's test(test=1) does")


class BoundedRefusal(SystemExit):
    """Nothing is measured, compared or published from a state that cannot be shown."""


# --- term by term: which terms belong to the implementation and which to the model ----------------------------------------

def parameter_bytes(design: dict, horizon: int, *, dtype_bytes: int = 4) -> dict:
    """The MODEL term that can be counted without a measurement: the author's own model built on the CPU from the sealed argv,
    its parameter count and the bytes those parameters occupy. Activations and training memory are NOT this and are not derived
    here; they are measured or they are named UNMEASURED."""
    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon))
    S.author_env()
    import torch
    args = S.build_args(cell["argv"], data_dir=Path("/nonexistent"), data_name="x.csv", checkpoints=Path("/nonexistent"),
                        gpu=0, use_gpu=False)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        exp = exp_module.Exp_Long_Term_Forecast(args)
    n = int(sum(p.numel() for p in exp.model.parameters()))
    trainable = int(sum(p.numel() for p in exp.model.parameters() if p.requires_grad))
    del exp
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"parameters": n, "trainable_parameters": trainable, "dtype_bytes": dtype_bytes,
            "weights_bytes": n * dtype_bytes, "weights_gib": n * dtype_bytes / float(1 << 30),
            "term_class": "MODEL",
            "reading": "the author's own model instantiated from the sealed argv and counted. It is the only model-side term "
                       "this module derives; activations, optimizer state and training memory are measured or named UNMEASURED"}


def bounded_path_derivation(design: dict, characterization: dict, horizon: int, *, baseline_bytes: int,
                            batch_size: int | None = None, leaf_elements: int = 1 << 22,
                            flush_every_batches: int = 16) -> dict:
    """The bounded path's derived RESIDENT terms and its DISK terms, from the same sealed population the author-path derivation
    reads. Every term names the line of `df_sota_repro.bounded_test` or `author_metric_exact` it comes from. It is a DERIVATION
    and it is labelled one; the probe never consults it."""
    R.validate_design(design)
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]
    windows, channels = int(sets["test"]["windows"]), int(sets["test"]["channels"])
    elements = windows * int(horizon) * channels
    if elements != int(sets["elements_test"]):
        raise BoundedRefusal(f"REFUSED: the characterization's element count for {key} does not reconcile: "
                             f"{sets['elements_test']} against windows {windows} x steps {horizon} x channels {channels}")
    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon))
    b = int(batch_size if batch_size is not None else cell["effective_args"]["batch_size"])
    f32 = 4
    window_bytes = int(horizon) * channels * f32
    block = b * window_bytes
    # resident, per batch: the outputs and targets copied off the device, plus the contiguous copy each is written from
    per_batch = {"bytes": 4 * block,
                 "from": "bounded_test: `outputs`/`batch_y` copied off the device plus their np.ascontiguousarray write buffers",
                 "term_class": "EVALUATION_IMPLEMENTATION"}
    # page cache written since the last posix_fadvise(DONTNEED): the writer releases every `flush_every_batches` batches
    dirty_window = {"bytes": 2 * flush_every_batches * block,
                    "from": f"bounded_test: pages written for preds and trues between the DONTNEED calls it makes every "
                            f"{flush_every_batches} batches. They are file-backed and RECLAIMABLE, and memory.peak counts them",
                    "term_class": "EVALUATION_IMPLEMENTATION"}
    # the reducer: one leaf's worth of windows, read twice (abs and square passes) and held as a, b, d, |d| and d*d
    leaf_windows = max(1, -(-int(leaf_elements) // max(1, int(horizon) * channels)) + 1)
    reducer = {"bytes": 5 * leaf_windows * window_bytes,
               "from": f"author_metric_exact: one pairwise leaf of {int(leaf_elements)} elements spans at most {leaf_windows} "
                       f"windows; `rows()` holds the float32 slice of preds and trues, their difference, its absolute value and "
                       f"its square. Independent of the population size",
               "term_class": "EVALUATION_IMPLEMENTATION"}
    f64 = {"bytes": 2 * 64 * window_bytes * 2,
           "from": "float64_metrics_files: 64 windows of each operand promoted to float64, pages released behind",
           "term_class": "EVALUATION_IMPLEMENTATION"}
    terms = {"per_batch_blocks": per_batch, "writer_dirty_page_window": dirty_window,
             "reducer_leaf_working_set": reducer, "independent_float64_chunk": f64}
    arrays_peak = max(per_batch["bytes"] + dirty_window["bytes"], reducer["bytes"], f64["bytes"])
    array_bytes = elements * f32
    disk = {"preds_bytes": array_bytes, "trues_bytes": array_bytes, "total_bytes": 2 * array_bytes,
            "total_gib": 2 * array_bytes / float(1 << 30),
            "retained_after_metrics": 0,
            "retention_rule": ("the two arrays are deleted by this probe once the metrics, the element count and the digests "
                               "that identify the population are recorded. What is retained is the record, not the arrays"),
            "term_class": "EVALUATION_IMPLEMENTATION"}
    return {"kind": "DERIVED_NOT_MEASURED", "path": BOUNDED_EVAL_PATH, "horizon_steps": int(horizon),
            "horizon_seconds": R.horizon_seconds(design["dataset"], horizon),
            "windows": windows, "channels": channels, "elements": elements, "element_dtype": "float32",
            "batch_size": b, "leaf_elements": int(leaf_elements), "flush_every_batches": int(flush_every_batches),
            "terms": terms, "arrays_peak_bytes": int(arrays_peak),
            "baseline_bytes": int(baseline_bytes),
            "baseline_from": "the whole-cgroup peak a TRAIN-only pilot measured for this design on the admitted device",
            "derived_peak_bytes": int(arrays_peak + baseline_bytes),
            "derived_peak_gib": (arrays_peak + baseline_bytes) / float(1 << 30),
            "disk": disk,
            "reading": "a derivation. The resident terms do not contain the population size; the DISK term does, which is why a "
                       "disk budget is a prerequisite and not a footnote"}


def memory_split(design: dict, characterization: dict, *, baseline_bytes: int, cap_bytes: int,
                 parameters: dict | None = None) -> dict:
    """The deliverable of the correction: for every horizon, the author-path terms and the bounded-path terms, each labelled with
    what it is a property of, against one declared cap. Nothing here is measured and nothing here is a verdict."""
    R.validate_design(design)
    rows = {}
    for h in design["horizons"]:
        author = X.eval_path_memory_derivation(design, characterization, h, baseline_bytes=baseline_bytes)
        bounded = bounded_path_derivation(design, characterization, h, baseline_bytes=baseline_bytes)
        author_terms = {k: {**v, "term_class": "EVALUATION_IMPLEMENTATION"} for k, v in author["terms"].items()}
        rows[f"h{h}"] = {
            "horizon_steps": int(h), "horizon_seconds": R.horizon_seconds(design["dataset"], h),
            "windows": author["windows"], "channels": author["channels"], "elements": author["elements"],
            "author_path": {"terms": author_terms, "arrays_peak_bytes": author["arrays_peak_bytes"],
                            "derived_peak_bytes": author["derived_peak_bytes"],
                            "derived_peak_gib": author["derived_peak_gib"],
                            "fits_cap": author["derived_peak_bytes"] <= int(cap_bytes)},
            "bounded_path": {"terms": bounded["terms"], "arrays_peak_bytes": bounded["arrays_peak_bytes"],
                             "derived_peak_bytes": bounded["derived_peak_bytes"],
                             "derived_peak_gib": bounded["derived_peak_gib"],
                             "disk": bounded["disk"],
                             "fits_cap": bounded["derived_peak_bytes"] <= int(cap_bytes)},
            "ratio_author_over_bounded": (author["derived_peak_bytes"] / bounded["derived_peak_bytes"]),
        }
    out = {"schema": SPLIT_SCHEMA, "kind": "DERIVATION_NOT_MEASUREMENT",
           "dataset": design["dataset"], "design_sha256": design["design_sha256"],
           "protocol_sha256": R.protocol_sha256(design), "seq_len": design["seq_len"],
           "declared_cap_bytes": int(cap_bytes), "baseline_bytes": int(baseline_bytes),
           "author_path": X.AUTHOR_EVAL_PATH, "bounded_path": BOUNDED_EVAL_PATH, "invariants_held": list(INVARIANTS),
           "term_classes": {
               "EVALUATION_IMPLEMENTATION": "a property of ONE way of evaluating: how predictions are accumulated, how many "
                                            "copies the concatenation makes, how many full-size temporaries the metric builds. "
                                            "Changing the implementation changes it and changes NO scientific quantity",
               "MODEL": "a property of the model itself: its weights, its activations and the memory its training needs. No "
                        "evaluation implementation can reduce it"},
           "model_side": (parameters or {"state": "NOT_DERIVED_HERE",
                                         "reading": "run `split --with-parameters` to count the author's own parameters"}),
           "model_side_unmeasured": {"activations": "UNMEASURED", "optimizer_state": "UNMEASURED",
                                     "training_host_and_device_memory": "UNMEASURED",
                                     "reading": "null is NOT small. These are named unmeasured rather than derived, and no "
                                                "Traffic training cost is claimed or requested by this module"},
           "horizons": rows}
    out["record_sha256"] = S.sha_obj(out)
    return out


# --- the reducer's parity, bitwise, with no model and no GPU ---------------------------------------------------------------

def reducer_parity(shapes: list, *, seed: int = 20260929, work: Path | None = None) -> dict:
    """The bounded float32 reduction against `utils.metrics.metric` ITSELF, on populations small enough for the author's own
    function to run unchunked. Equality is asserted BITWISE on the float64 repr of the returned float32 scalars; a tolerance is
    not claimed where equality can be demonstrated, and where it is not exact the measured difference is reported rather than
    assumed to be zero."""
    S.author_env()
    MET = importlib.import_module("utils.metrics")
    work = Path(work or Path(os.environ.get("TMPDIR", "/tmp")) / f"reducer_parity_{os.getpid()}")
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cases = []
    try:
        for shape in shapes:
            w, t, c = (int(x) for x in shape)
            preds = rng.standard_normal((w, t, c), dtype=np.float32)
            trues = (preds + rng.standard_normal((w, t, c), dtype=np.float32) * np.float32(0.3)).astype(np.float32)
            mae, mse, _, _, _ = MET.metric(preds, trues)
            p_path, t_path = work / "p.npy", work / "t.npy"
            np.save(p_path, preds); np.save(t_path, trues)
            exact = S.author_metric_exact(S.StoredArray(p_path), S.StoredArray(t_path))
            n = w * t * c
            # the counter-example the trap names: a mean of chunk means, which is NOT what the bounded route does
            k = max(1, w // 7)
            chunk_means_mae = float(np.mean([float(np.mean(np.abs(trues[i:i + k] - preds[i:i + k]))) for i in range(0, w, k)]))
            cases.append({
                "shape": [w, t, c], "elements": n,
                "denominator_exact_in_float32": int(np.float32(n)) == n,
                "author_function": {"mae": float(mae), "mse": float(mse)},
                "bounded_route": {"mae": exact["mae"], "mse": exact["mse"], "route": exact["route"]},
                "bitwise_equal": bool(float(mae) == exact["mae"] and float(mse) == exact["mse"]),
                "absolute_difference": {"mae": abs(float(mae) - exact["mae"]), "mse": abs(float(mse) - exact["mse"])},
                "naive_chunk_mean_of_means_mae": chunk_means_mae,
                "naive_chunk_mean_of_means_is_equal": bool(chunk_means_mae == float(mae)),
                "naive_chunk_mean_of_means_difference": abs(chunk_means_mae - float(mae)),
            })
            p_path.unlink(missing_ok=True); t_path.unlink(missing_ok=True)
            del preds, trues
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(work, ignore_errors=True)
    out = {"schema": PARITY_SCHEMA, "kind": "BITWISE_PARITY_OF_THE_REDUCTION",
           "scorer": X.author_scorer_identity(), "route": S.AUTHOR_SCORER_ROUTE,
           "cases": cases,
           "all_bitwise_equal": all(c["bitwise_equal"] for c in cases),
           "max_absolute_difference": max([max(c["absolute_difference"].values()) for c in cases] or [0.0]),
           "chunk_mean_of_means_ever_equal": any(c["naive_chunk_mean_of_means_is_equal"] for c in cases),
           "reading": "the bounded route is not a chunked average. The `naive_chunk_mean_of_means_*` columns are the thing it is "
                      "NOT, computed here so the difference is demonstrated rather than asserted: an average of per-chunk means "
                      "is a different number in float32 and is reported beside the equal one",
           "environment": S.environment(), "numpy": np.__version__}
    out["record_sha256"] = S.sha_obj(out)
    return out


# --- the probe: one horizon through the existing bounded evaluator ---------------------------------------------------------

PROBE_VERDICTS = {
    "ADMISSIBLE": "the bounded evaluation path fits the cap this child was admitted under, on this horizon's complete sealed "
                  "population, and its parity with the author's own reduction is established",
    "ADMISSIBLE_PARITY_INHERITED": "the bounded path fits the cap, but the author's own unchunked function could not run beside "
                                   "it at this size; parity is inherited from the horizons where it WAS demonstrated and from "
                                   "the reducer battery, and is labelled inherited rather than measured here",
    "CAPACITY_DEFICIT": "the bounded path does not fit the cap this child was admitted under. The cap is not shrunk and the "
                        "population is not reduced; this is a named deficit",
    "UNDETERMINED": "the cap or the peak could not be read; nothing is admitted on an unread number",
    "REFUSED_INCOMPLETE_POPULATION": "the bounded path did not cover the sealed number of test windows; an incomplete "
                                     "evaluation population is not a measurement",
    "REFUSED_PARITY_BROKEN": "the bounded reduction and the author's own function disagree. Nothing is admitted on that",
}


def probe_verdict(*, measured_peak_bytes: int | None, cap_bytes: int | None, population_complete: bool,
                  parity_holds: bool | None, author_function_bit_equal: bool | None) -> str:
    """The verdict is a comparison of measured numbers and demonstrated equalities, and nothing else. A cap is never shrunk to
    make a peak fit and a peak is never reduced to make a cap fit: when either is unreadable the answer is UNDETERMINED, not
    admission. A horizon at which the author's own function could not run does not thereby become 'in parity'; it becomes
    ADMISSIBLE_PARITY_INHERITED, which says where the parity came from."""
    if measured_peak_bytes is None or cap_bytes is None:
        return "UNDETERMINED"
    if not population_complete:
        return "REFUSED_INCOMPLETE_POPULATION"
    if parity_holds is False or author_function_bit_equal is False:
        return "REFUSED_PARITY_BROKEN"
    if int(measured_peak_bytes) > int(cap_bytes):
        return "CAPACITY_DEFICIT"
    if parity_holds is True or author_function_bit_equal is True:
        return "ADMISSIBLE"
    return "ADMISSIBLE_PARITY_INHERITED"


def _disk(path: Path) -> dict:
    u = shutil.disk_usage(path)
    return {"total_bytes": int(u.total), "used_bytes": int(u.used), "free_bytes": int(u.free)}


def _tree_bytes(path: Path) -> int:
    return int(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()))


def bounded_probe(design: dict, characterization: dict, *, data_path: Path, work: Path, horizon: int,
                  gpu: int = 0, require_gpu_uuid: str | None = None, dataloader_workers: int | None = 0,
                  native_witness: bool = False, author_metric_budget_bytes: int | None = None,
                  retain_arrays: bool = False) -> dict:
    """The bounded evaluation path at `horizon` on the complete sealed test population, with an UNTRAINED model, measuring the
    WHOLE-CGROUP peak from INSIDE this child and the disk high-water under `work`. It produces NO score.

    With `native_witness` the author's OWN unchunked `test()` runs FIRST, in this same child, from the SAME on-disk checkpoint,
    and the two paths are compared on the sha256 of the arrays and on the metric values, BITWISE. That comparison is the parity
    proof; it is only attempted at a horizon whose native path fits, and its absence elsewhere is reported, never assumed away.
    """
    R.validate_design(design)
    if characterization.get("design_sha256") != design["design_sha256"]:
        raise BoundedRefusal("REFUSED: this characterization was not produced under this design")
    facts = R.dataset_facts(design["dataset"])
    got = S.sha_file(data_path)
    if got != facts["sha256"]:
        raise BoundedRefusal(f"REFUSED: the probe would read bytes that are not the registered resource ({got[:12]})")
    drift = S.code_drift(design)
    if drift:
        raise BoundedRefusal(f"REFUSED: the author files have drifted from the sealed digests: {sorted(drift)}")
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]
    expected_windows = int(sets["test"]["windows"])
    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon))
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cap = X.declared_cap_bytes()
    S.author_env()
    import torch
    device_check = S.assert_child_device(require_gpu_uuid, gpu)
    S.fix_seeds(cell["seed"])
    args = S.build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=work / "checkpoints",
                        gpu=gpu, dataloader_workers=dataloader_workers)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    started = S.now_iso()
    stages = {"entry": X.cgroup_peak_bytes()}
    disk_before = _disk(work)
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0, cpu0 = time.time(), time.process_time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cwd = os.getcwd()
    os.chdir(work)
    native = None
    try:
        with open(work / "bounded_author_stdout.log", "a") as fh, contextlib.redirect_stdout(fh):
            exp = exp_module.Exp_Long_Term_Forecast(args)
            setting = S.setting_of(args)
            stages["after_model_build"] = X.cgroup_peak_bytes()
            # ONE set of weights on disk, used by BOTH paths: the checkpoint rule is the author's own, and identity is a digest
            ckpt_dir = Path(args.checkpoints) / setting
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(exp.model.state_dict(), ckpt_dir / "checkpoint.pth")
            checkpoint_sha = S.sha_file(ckpt_dir / "checkpoint.pth")
            if native_witness:
                native = _native_witness(exp_module, exp, setting, expected_windows, stages)
            t0 = time.time()
            out = S.bounded_test(exp, setting, work, author_metric_budget_bytes=author_metric_budget_bytes)
            bounded_seconds = time.time() - t0
            stages["after_bounded_evaluation"] = X.cgroup_peak_bytes()
            preds_path, trues_path = Path(out["preds_path"]), Path(out["trues_path"])
            disk_high_water = _tree_bytes(work)
            disk_at_high_water = _disk(work)
            pred_sha = S.sha_array(np.load(preds_path, mmap_mode="r"))
            stages["after_digests"] = X.cgroup_peak_bytes()
    finally:
        os.chdir(cwd)
    wall, cpu = time.time() - wall0, time.process_time() - cpu0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    measured_peak = X.cgroup_peak_bytes()
    shape = [int(x) for x in out["shape"]]
    population_ok = shape[0] == expected_windows and shape[1] == int(horizon) and shape[2] == int(sets["test"]["channels"])
    derived = bounded_path_derivation(design, characterization, horizon, baseline_bytes=0)
    author_derived = X.eval_path_memory_derivation(design, characterization, horizon, baseline_bytes=0)
    parity = {"native_path_executed_here": bool(native is not None),
              "author_function_beside_the_bounded_route": out.get("author_scorer_parity")}
    if native is not None:
        parity["arrays"] = {"preds_sha256_native": native["preds_sha256"], "preds_sha256_bounded": pred_sha,
                            "preds_identical": native["preds_sha256"] == pred_sha,
                            "trues_sha256_native": native["trues_sha256"],
                            "trues_sha256_bounded": out["finalized"]["true_sha256"],
                            "trues_identical": native["trues_sha256"] == out["finalized"]["true_sha256"],
                            "reading": "the same bytes in the same order: the bounded writer's file against the array the "
                                       "author's own np.concatenate built, from the SAME forward pass configuration and the "
                                       "SAME on-disk checkpoint"}
        parity["metrics"] = {"native_author_function": native["metric"],
                             "bounded_route": out["author_metric"],
                             "mae_bitwise_equal": native["metric"]["mae"] == out["author_metric"]["mae"],
                             "mse_bitwise_equal": native["metric"]["mse"] == out["author_metric"]["mse"],
                             "mae_absolute_difference": abs(native["metric"]["mae"] - out["author_metric"]["mae"]),
                             "mse_absolute_difference": abs(native["metric"]["mse"] - out["author_metric"]["mse"])}
        parity["holds"] = bool(parity["arrays"]["preds_identical"] and parity["arrays"]["trues_identical"]
                               and parity["metrics"]["mae_bitwise_equal"] and parity["metrics"]["mse_bitwise_equal"])
    else:
        parity["holds"] = None
        parity["why"] = ("the author's unchunked path was not run at this horizon: its own derivation puts it at "
                         f"{author_derived['arrays_peak_bytes'] / float(1 << 30):.2f} GiB of arrays alone. Parity at this "
                         "horizon is INHERITED from the horizons where it was demonstrated, and is labelled inherited")
    verdict = probe_verdict(measured_peak_bytes=measured_peak, cap_bytes=cap, population_complete=population_ok,
                            parity_holds=parity["holds"],
                            author_function_bit_equal=(out.get("author_scorer_parity") or {}).get("bit_equal"))
    retention = {"arrays_deleted": not retain_arrays,
                 "rule": derived["disk"]["retention_rule"],
                 "deleted": [], "retained": []}
    for p in (preds_path, trues_path):
        if retain_arrays:
            retention["retained"].append({"path": str(p), "bytes": p.stat().st_size if p.is_file() else None})
        else:
            with contextlib.suppress(OSError):
                p.unlink()
            retention["deleted"].append(str(p))
    record = {
        "schema": PROBE_SCHEMA, "kind": "BOUNDED_EVALUATION_PATH_MEMORY_AND_PARITY_PROBE",
        "reading": "a host-memory capacity measurement of the EXISTING bounded evaluation path with an UNTRAINED model on the "
                   "complete sealed test population. It stores NO metric as a result: an untrained model's error is not a "
                   "result, and the metric values appear here only as the two sides of a numerical parity comparison",
        "gate": "no scored Traffic cell is authorized by this record. It prepares an alternative; it does not run a campaign",
        "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
        "dataset": design["dataset"], "horizon_steps": int(horizon),
        "horizon_seconds": R.horizon_seconds(design["dataset"], horizon), "seq_len": design["seq_len"],
        "cell_id_geometry_from": cell["cell_id"], "trained": False,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_rule": "the untrained weights were written to disk once and RELOADED by both paths, exactly as the author's "
                           "test(test=1) reloads a trained checkpoint. The rule is unchanged; only the weights are untrained",
        "evaluation_path": {"bounded": BOUNDED_EVAL_PATH, "adapter": out["adapter"],
                            "author_path_for_contrast": X.AUTHOR_EVAL_PATH,
                            "invariants_held": list(INVARIANTS)},
        "population": {"expected_windows": expected_windows, "scored_shape": shape, "dtype": "float32",
                       "complete": population_ok, "elements": int(np.prod(shape)),
                       "elements_sealed": int(sets["elements_test"]),
                       "batches": out["finalized"]["batches"], "batch_sizes": out["finalized"]["batch_sizes"],
                       "preds_sha256": pred_sha, "trues_sha256": out["finalized"]["true_sha256"]},
        "measured": {"whole_cgroup_peak_bytes_in_child": measured_peak,
                     "whole_cgroup_peak_gib": (measured_peak / float(1 << 30)) if measured_peak else None,
                     "cgroup_stage_peaks_bytes": stages, "cgroup": S._cgroup_memory(),
                     "peak_rss_bytes_self": int(ru1.ru_maxrss) * 1024,
                     "user_cpu_seconds": ru1.ru_utime - ru0.ru_utime, "system_cpu_seconds": ru1.ru_stime - ru0.ru_stime,
                     "bounded_evaluation_seconds": bounded_seconds, "wall_seconds": wall, "cpu_seconds": cpu,
                     "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                     "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
                     "host_memavailable_bytes_at_exit": X.meminfo_available_bytes()},
        "disk": {"high_water_under_work_bytes": disk_high_water,
                 "high_water_under_work_gib": disk_high_water / float(1 << 30),
                 "filesystem_before": disk_before, "filesystem_at_high_water": disk_at_high_water,
                 "filesystem_after_retention": _disk(work),
                 "derived_arrays_bytes": derived["disk"]["total_bytes"],
                 "reading": "a bounded evaluator that fills the disk has moved the failure, not removed it. This is the measured "
                            "high-water of the work directory, beside the filesystem's own free space at that moment"},
        "retention": retention,
        "parity": parity,
        "author_metric_state": out["author_metric_state"],
        "independent_float64_beside_the_float32_route": out["independent_metric_float64"],
        "derived_for_contrast": {"bounded": derived, "author_path": author_derived,
                                 "note": "both derivations carry baseline_bytes 0: the baseline is what this measurement "
                                         "includes and does not need to assume"},
        "measured_over_derived_bounded_arrays": ((measured_peak / derived["arrays_peak_bytes"]) if measured_peak else None),
        "declared_cap_bytes": cap,
        "headroom_bytes": ((cap - measured_peak) if (cap and measured_peak) else None),
        "verdict": verdict, "verdict_reading": PROBE_VERDICTS[verdict],
        "device": str(exp.device), "device_uuid_measured_inside_child": S.actual_device_uuid(gpu),
        "device_assertion": device_check, "gpu_state_after": S.gpu_state(),
        "environment": S.environment(), "host_thermals_c": S.host_thermals(),
        "author_clone": S.author_git(), "source_drift": drift, "file_sha256": got,
        "boot_id": (Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                    if Path("/proc/sys/kernel/random/boot_id").is_file() else None),
        "pid": os.getpid(), "started_at": started, "finished_at": S.now_iso(),
        "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                       "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")},
    }
    record["record_sha256"] = S.sha_obj(record)
    return record


TARGET_PAIRING_SCHEMA = "df_tsl_target_pairing.v1"


def target_pairing(design: dict, characterization: dict, *, data_path: Path, horizon: int, probe: dict) -> dict:
    """The CHEAP half of the writer's parity, and the one that does not need the author's unchunked path to fit.

    `df_sota_repro.naive_and_trues` rebuilds the author's OWN test loader from the same argv, the same borders and the same
    scaler, and hashes the target windows AS THEY STREAM, in loader order. The sealed scored-cell path already refuses a cell
    whose naive was computed on a different target population than the model was scored on, by exactly this digest. Comparing it
    to the digest the bounded WRITER produced is therefore an independent statement that the bounded file holds the same bytes,
    in the same rows, in the same order as the author's loader yields them — established without holding either array.

    Predictions and targets are written inside ONE loop body at ONE running offset, so a row placement that is right for the
    targets is right for the predictions. This does not, by itself, establish that the model's outputs are identical; that is
    what the native witness establishes where it fits."""
    R.validate_design(design)
    if probe.get("design_sha256") != design["design_sha256"]:
        raise BoundedRefusal("REFUSED: the probe record was not produced under this design")
    if int(probe.get("horizon_steps", -1)) != int(horizon):
        raise BoundedRefusal(f"REFUSED: the probe record is for h{probe.get('horizon_steps')}, not h{horizon}")
    facts = R.dataset_facts(design["dataset"])
    got = S.sha_file(data_path)
    if got != facts["sha256"]:
        raise BoundedRefusal(f"REFUSED: these are not the registered bytes of {facts['resource']} ({got[:12]})")
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]
    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon))
    started = S.now_iso()
    wall0 = time.time()
    nv = S.naive_and_trues(design, {**cell, "horizon": int(horizon), "seq_len": design["seq_len"]}, data_path)
    written = probe["population"]["trues_sha256"]
    out = {"schema": TARGET_PAIRING_SCHEMA, "kind": "INDEPENDENT_TARGET_POPULATION_IDENTITY",
           "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
           "dataset": design["dataset"], "horizon_steps": int(horizon),
           "horizon_seconds": R.horizon_seconds(design["dataset"], horizon),
           "probe_record_sha256": probe.get("record_sha256"),
           "targets": {"sha256_from_the_authors_loader": nv["true_sha256"],
                       "sha256_written_by_the_bounded_writer": written,
                       "identical": nv["true_sha256"] == written,
                       "windows_from_the_authors_loader": int(nv["windows"]),
                       "windows_written": int(probe["population"]["scored_shape"][0]),
                       "windows_sealed": int(sets["test"]["windows"]),
                       "elements_from_the_authors_loader": int(nv["n_elements"]),
                       "elements_sealed": int(sets["elements_test"])},
           "paired_persistence_naive": {"mse": float(nv["naive"]["mse"]), "mae": float(nv["naive"]["mae"]),
                                        "definition": nv["naive_definition"],
                                        "reading": "the persistence baseline on EXACTLY these rows, computed here because the "
                                                   "same pass produces it. It is a baseline for a future scored cell, not a "
                                                   "result: no model was trained and nothing was scored against it"},
           "reading": "an independent identity of the evaluated target population, from a separate pass of the author's own "
                      "loader. It is the half of the writer's parity that does not need the unchunked path to fit",
           "wall_seconds": time.time() - wall0, "started_at": started, "finished_at": S.now_iso(),
           "environment": S.environment(), "file_sha256": got,
           "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                          "CRISPDM_JOB_NAME", "CRISPDM_SLICE")}}
    out["verdict"] = "IDENTICAL" if out["targets"]["identical"] else "REFUSED_DIFFERENT_TARGET_POPULATION"
    out["record_sha256"] = S.sha_obj(out)
    return out


def _native_witness(exp_module, exp, setting: str, expected_windows: int, stages: dict) -> dict:
    """The author's OWN unchunked `test()`, run once so the bounded path has something to be identical TO. The author's arrays
    and return value pass through untouched; what is taken from them is a digest and the metric values, and no reference to the
    arrays outlives this function."""
    original = exp_module.metric
    seen = {}

    def witnessed(preds, trues):
        stages["native_before_scoring"] = X.cgroup_peak_bytes()
        seen["shape"] = [int(x) for x in np.shape(preds)]
        seen["preds_sha256"] = S.sha_array(np.asarray(preds))
        seen["trues_sha256"] = S.sha_array(np.asarray(trues))
        value = original(preds, trues)
        stages["native_after_scoring"] = X.cgroup_peak_bytes()
        seen["metric"] = {"mae": float(value[0]), "mse": float(value[1])}
        return value

    exp_module.metric = witnessed
    try:
        t0 = time.time()
        exp.test(setting, test=1)                                       # test=1: the on-disk checkpoint, the author's own reload
        seen["seconds"] = time.time() - t0
    finally:
        exp_module.metric = original
    if seen.get("shape", [None])[0] != expected_windows:
        raise BoundedRefusal(f"REFUSED: the author's native path saw {seen.get('shape')} against the sealed {expected_windows} "
                             "windows; an incomplete population is not a witness")
    stages["after_native_evaluation"] = X.cgroup_peak_bytes()
    return seen


# --- CLI -------------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("characterize", help="the sealed characterization of the delivered bytes")
    c.add_argument("--design", required=True)
    c.add_argument("--data-path", required=True)
    c.add_argument("--out", required=True)

    s = sub.add_parser("split", help="the term-by-term memory split, author path against bounded path")
    s.add_argument("--design", required=True)
    s.add_argument("--characterization", required=True)
    s.add_argument("--baseline-bytes", type=int, default=0)
    s.add_argument("--cap-bytes", type=int, required=True)
    s.add_argument("--with-parameters", action="store_true")
    s.add_argument("--out", required=True)

    r = sub.add_parser("reducer-parity", help="the bounded reduction against utils.metrics.metric, bitwise")
    r.add_argument("--shape", action="append", required=True, help="W,T,C — repeatable")
    r.add_argument("--out", required=True)

    b = sub.add_parser("probe", help="one horizon through the existing bounded evaluator, measured")
    b.add_argument("--design", required=True)
    b.add_argument("--characterization", required=True)
    b.add_argument("--data-path", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--horizon", type=int, required=True)
    b.add_argument("--gpu", type=int, default=0)
    b.add_argument("--require-gpu-uuid", default=os.environ.get("CRISPDM_REQUIRED_GPU_UUID"))
    b.add_argument("--dataloader-workers", type=int, default=0)
    b.add_argument("--native-witness", action="store_true")
    b.add_argument("--author-metric-budget-gib", type=float, default=None)
    b.add_argument("--retain-arrays", action="store_true")
    b.add_argument("--out", required=True)

    t = sub.add_parser("target-pairing", help="the evaluated target population's identity, from a separate loader pass")
    t.add_argument("--design", required=True)
    t.add_argument("--characterization", required=True)
    t.add_argument("--data-path", required=True)
    t.add_argument("--horizon", type=int, required=True)
    t.add_argument("--probe", required=True)
    t.add_argument("--out", required=True)

    a = p.parse_args(argv)
    if a.cmd == "characterize":
        design = json.loads(Path(a.design).read_text())
        char, arrays = R.characterize(design, Path(a.data_path))
        _write(Path(a.out), char)
        np.savez(Path(a.out).with_suffix(".scalers.npz"), **arrays)
        print(json.dumps({"out": a.out, "sets": sorted(char["sets"]),
                          "test_windows": {k: v["test"]["windows"] for k, v in char["sets"].items()},
                          "elements_test": {k: v["elements_test"] for k, v in char["sets"].items()}}, indent=1))
        return 0
    if a.cmd == "split":
        design = json.loads(Path(a.design).read_text())
        char = json.loads(Path(a.characterization).read_text())
        params = parameter_bytes(design, design["horizons"][0]) if a.with_parameters else None
        out = memory_split(design, char, baseline_bytes=a.baseline_bytes, cap_bytes=a.cap_bytes, parameters=params)
        _write(Path(a.out), out)
        for k, row in out["horizons"].items():
            print(f"{k:>5} windows {row['windows']:>6}  author {row['author_path']['derived_peak_gib']:8.3f} GiB "
                  f"(fits {row['author_path']['fits_cap']})  bounded {row['bounded_path']['derived_peak_gib']:8.3f} GiB "
                  f"(fits {row['bounded_path']['fits_cap']})  disk {row['bounded_path']['disk']['total_gib']:7.3f} GiB")
        return 0
    if a.cmd == "reducer-parity":
        shapes = [tuple(int(x) for x in s.split(",")) for s in a.shape]
        out = reducer_parity(shapes)
        _write(Path(a.out), out)
        print(json.dumps({"all_bitwise_equal": out["all_bitwise_equal"],
                          "max_absolute_difference": out["max_absolute_difference"],
                          "chunk_mean_of_means_ever_equal": out["chunk_mean_of_means_ever_equal"],
                          "cases": [{"shape": c["shape"], "bitwise_equal": c["bitwise_equal"],
                                     "denominator_exact_in_float32": c["denominator_exact_in_float32"],
                                     "chunk_mean_of_means_difference": c["naive_chunk_mean_of_means_difference"]}
                                    for c in out["cases"]]}, indent=1))
        return 0
    if a.cmd == "probe":
        design = json.loads(Path(a.design).read_text())
        char = json.loads(Path(a.characterization).read_text())
        budget = int(a.author_metric_budget_gib * (1 << 30)) if a.author_metric_budget_gib is not None else None
        out = bounded_probe(design, char, data_path=Path(a.data_path), work=Path(a.work), horizon=a.horizon,
                            gpu=a.gpu, require_gpu_uuid=a.require_gpu_uuid, dataloader_workers=a.dataloader_workers,
                            native_witness=a.native_witness, author_metric_budget_bytes=budget,
                            retain_arrays=a.retain_arrays)
        _write(Path(a.out), out)
        print(json.dumps({"verdict": out["verdict"], "horizon_steps": out["horizon_steps"],
                          "whole_cgroup_peak_gib": out["measured"]["whole_cgroup_peak_gib"],
                          "declared_cap_bytes": out["declared_cap_bytes"],
                          "disk_high_water_gib": out["disk"]["high_water_under_work_gib"],
                          "population_complete": out["population"]["complete"],
                          "parity_holds": out["parity"]["holds"],
                          "author_function_beside": (out["parity"]["author_function_beside_the_bounded_route"] or {}).get("bit_equal"),
                          "record_sha256": out["record_sha256"]}, indent=1))
        return 0 if out["verdict"].startswith("ADMISSIBLE") else 75
    if a.cmd == "target-pairing":
        design = json.loads(Path(a.design).read_text())
        char = json.loads(Path(a.characterization).read_text())
        probe = json.loads(Path(a.probe).read_text())
        out = target_pairing(design, char, data_path=Path(a.data_path), horizon=a.horizon, probe=probe)
        _write(Path(a.out), out)
        print(json.dumps({"verdict": out["verdict"], "horizon_steps": out["horizon_steps"],
                          "targets": out["targets"], "record_sha256": out["record_sha256"]}, indent=1))
        return 0 if out["verdict"] == "IDENTICAL" else 75
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
