#!/usr/bin/env python3
"""RB02 EXECUTION: gate one (the evaluation-path memory probe) and the twelve scored Weather cells.

This module introduces NO model, loader, loss, scorer, recipe or margin. Everything scientific it uses is read from objects
that already exist and are not touched here:

  * `df_tsl_repro` (SEALED by the matched-reference delivery) — the design, the characterization, the receipt builder and the
    PRODUCER GATE. Its behaviour is not modified, wrapped or re-derived. This module imports it and calls it.
  * `df_sota_repro` — the author bridge: the author's own argparse, `Dataset_Custom`, `Exp_Long_Term_Forecast`, optimizer,
    loss, early stopping, and `utils.metrics.metric`. The evaluation path executed here is the AUTHOR's OWN, unchunked.

What this module adds is three things, all of them measurement or bookkeeping:

  probe   GATE ONE. The author's UNCHUNKED evaluation path (`Exp_Long_Term_Forecast.test`: three float32 lists, three
          `np.concatenate` copies, then `utils.metrics.metric` with full-size temporaries) run once with an UNTRAINED model at
          the longest horizon, measuring the WHOLE-CGROUP peak from INSIDE the child. It is a capacity measurement and stores
          no score. If it does not fit the declared cap the answer is a NAMED CAPACITY DEFICIT: this module contains no
          chunking, no downcast and no sub-sampling of the author's metric population, so there is nothing here that could
          silently make an oversized evaluation fit.
  cell    One sealed cell through the author's own train-then-test, paired with the persistence naive on EXACTLY the same
          rows (proved by a digest of the targets, not asserted), emitted as a `tsl_literature_metrics.v1` receipt through
          `df_tsl_repro.validate_receipt`.
  close   The comparison rows and the closure table, composed ONLY from retained cell records: model error with its space and
          reduction, the paired naive, the skill, the published value with its source, the comparability class against THIS
          dataset's own published dispersion, the seed dispersion, and the resources.

The agreement margin is read from `design["lock"]["agreement"]["std_paper"]`, which the seal built from Weather's own Table 7
row. `df_sota_repro.AGREEMENT` holds ELECTRICITY's margin and is never used here; a test asserts the two differ and that this
module reads the design.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import inspect
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402  the author bridge, reused verbatim
import df_tsl_repro as R                                               # noqa: E402  the sealed design, receipt and producer gate

PROBE_SCHEMA = "df_tsl_eval_probe.v1"
CELL_SCHEMA = "df_tsl_cell_record.v1"
ROW_SCHEMA = "df_tsl_comparison_row.v1"
CLOSURE_SCHEMA = "df_tsl_closure.v1"

#: The evaluation path this module measures and executes is the author's own, unmodified. Named so a reader can check it.
AUTHOR_EVAL_PATH = ("exp.exp_long_term_forecasting.Exp_Long_Term_Forecast.test: per-batch float32 lists for inputs, preds and "
                    "trues, three np.concatenate copies, then utils.metrics.metric on the whole arrays. No chunking, no "
                    "memmap, no downcast, no sub-sampling")

#: What a governed unit would need and what this lane holds instead. Stated, never worked around.
TRANSPORT_CLASS = "DECLARED_TRANSPORT_OF_GOVERNED_BYTES_NOT_A_NEW_GOVERNED_UNIT"
TRANSPORT_READING = ("a new governed unit for a scored cell needs the data-gov service key, which this lane does not hold and "
                     "did not go looking for. The bytes read here were governed-DELIVERED by the adoption campaign named in "
                     "`transport`, were verified there as VERIFIED_TRANSFER, and are re-verified by sha256 INSIDE this child "
                     "before the loader sees them. The receipt is therefore a declared transport of governed bytes, not a new "
                     "governed unit, and it is labelled as such in its own tags")


class ExecutionRefusal(SystemExit):
    """Nothing is measured, scored or published from a state that cannot be shown."""


# --- the cgroup, read from inside the child ------------------------------------------------------------------------------

def cgroup_peak_bytes() -> int | None:
    """The kernel's high-water mark for THIS process's whole cgroup — the job's own transient scope under crispdm-run, so it
    covers every process in the job, file-backed pages included. It is monotone, which is what makes a stage reading possible.
    The launcher's periodic sampler undershoots sub-second loads; this is the authoritative number."""
    cg = S._cgroup_memory()
    peak = cg.get("memory.peak")
    return int(peak) if isinstance(peak, int) else None


def declared_cap_bytes() -> int | None:
    """The cap this child was admitted under: the reservation the launcher holds, else the cgroup's own MemoryMax. Never a
    number this module chooses, and never shrunk to evade a refusal."""
    env = os.environ.get("CRISPDM_RESERVATION_BYTES")
    if env and str(env).strip().isdigit():
        return int(env)
    cg = S._cgroup_memory()
    m = cg.get("memory.max")
    return int(m) if isinstance(m, int) else None


def meminfo_available_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


# --- the DERIVED figure this probe exists to replace ----------------------------------------------------------------------

def eval_path_memory_derivation(design: dict, characterization: dict, horizon: int, *, baseline_bytes: int) -> dict:
    """The host-memory peak of the author's unchunked evaluation path, DERIVED term by term from the sealed population — the
    quantity the matched-reference delivery could only derive, reproduced here so the probe's measurement has something
    explicit to be measured against. Every term names the line of the author's `test()` it comes from. It is a DERIVATION and
    it is labelled one; it never substitutes for the measurement and the probe never consults it."""
    R.validate_design(design)
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]
    windows, channels = int(sets["test"]["windows"]), int(sets["test"]["channels"])
    elements = windows * int(horizon) * channels
    if elements != int(sets["elements_test"]):
        raise ExecutionRefusal(f"REFUSED: the characterization's element count for {key} does not reconcile: "
                               f"{sets['elements_test']} against windows {windows} x steps {horizon} x channels {channels}")
    f32 = 4
    pred = true = elements * f32
    inputs = windows * int(design["seq_len"]) * channels * f32
    terms = {
        "accumulated_lists": {"bytes": pred + true + inputs,
                              "from": "the per-batch float32 blocks held in the `preds`, `trues` and `inputs` lists"},
        "concatenate_stage_peak": {"bytes": inputs + pred + true + true,
                                   "from": "inputs and preds already concatenated while the trues list is still alive and its "
                                           "own concatenated copy is being written"},
        "retained_after_concatenate": {"bytes": inputs + pred + true, "from": "the three concatenated arrays"},
        "metric_temporaries": {"bytes": 3 * true,
                               "from": "utils.metrics: MAPE/MSPE build (true - pred), divide it by true and then take abs/square, "
                                       "so up to three full-size float32 temporaries are live at once; MAE and MSE need two"},
    }
    scoring_stage = terms["retained_after_concatenate"]["bytes"] + terms["metric_temporaries"]["bytes"]
    arrays_peak = max(terms["concatenate_stage_peak"]["bytes"], scoring_stage)
    return {"kind": "DERIVED_NOT_MEASURED", "horizon_steps": int(horizon), "windows": windows, "channels": channels,
            "elements": elements, "element_dtype": "float32", "terms": terms,
            "scoring_stage_bytes": scoring_stage, "arrays_peak_bytes": arrays_peak,
            "baseline_bytes": int(baseline_bytes),
            "baseline_from": "the whole-cgroup peak the TRAIN-only pilot measured for this design on the admitted device",
            "derived_peak_bytes": int(arrays_peak + baseline_bytes),
            "derived_peak_gib": (arrays_peak + baseline_bytes) / float(1 << 30),
            "author_path": AUTHOR_EVAL_PATH,
            "reading": "a derivation from element counts. The constraint that killed six Electricity cells was exactly this "
                       "path's host memory, so it is measured before any cell runs and never assumed"}


# --- gate one: the evaluation-path memory probe ---------------------------------------------------------------------------

PROBE_VERDICTS = {
    "ADMISSIBLE": "the author's own evaluation path fits the cap this child was admitted under, so the scored cells may run "
                  "it unaltered",
    "CAPACITY_DEFICIT": "the author's evaluation path does not fit. That is an honest capacity deficit and the run STOPS: "
                        "chunking, downcasting or sub-sampling the metric population would silently change what is being "
                        "compared",
    "UNDETERMINED": "the cap or the peak could not be read; nothing is admitted on an unread number",
    "REFUSED_INCOMPLETE_POPULATION": "the author's scorer did not see the sealed number of test windows; an incomplete "
                                     "evaluation population is not a measurement",
}


def probe_verdict(measured_peak_bytes: int | None, cap_bytes: int | None) -> str:
    """The verdict is a comparison of two measured numbers and nothing else. A cap is never shrunk to make a peak fit and a
    peak is never reduced to make a cap fit: when either is unreadable the answer is UNDETERMINED, not admission."""
    if measured_peak_bytes is None or cap_bytes is None:
        return "UNDETERMINED"
    return "ADMISSIBLE" if int(measured_peak_bytes) <= int(cap_bytes) else "CAPACITY_DEFICIT"


def eval_probe(design: dict, characterization: dict, *, data_path: Path, work: Path, horizon: int, gpu: int = 0,
               require_gpu_uuid: str | None = None, dataloader_workers: int | None = 0,
               baseline_bytes: int = 0) -> dict:
    """GATE ONE. The author's own unchunked `test()` with an UNTRAINED model at `horizon`, on the real test population, with
    the whole-cgroup peak read from inside this child. It produces NO score: an untrained model's error is not a result, and
    nothing here is stored as one. Its only output is a capacity measurement and a verdict."""
    R.validate_design(design)
    facts = R.dataset_facts(design["dataset"])
    got = S.sha_file(data_path)
    if got != facts["sha256"]:
        raise ExecutionRefusal(f"REFUSED: the probe would read bytes that are not the registered resource ({got[:12]})")
    cell = next(c for c in design["cells"] if c["horizon_steps"] == horizon)
    work.mkdir(parents=True, exist_ok=True)
    S.author_env()
    import torch
    device_check = S.assert_child_device(require_gpu_uuid, gpu)
    S.fix_seeds(cell["seed"])
    args = S.build_args(cell["argv"], data_dir=data_path.parent, data_name=data_path.name, checkpoints=work / "checkpoints",
                        gpu=gpu, dataloader_workers=dataloader_workers)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    cap = declared_cap_bytes()
    started = S.now_iso()
    stages = {"entry": cgroup_peak_bytes()}
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0, cpu0 = time.time(), time.process_time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # a measurement hook on the author's own scorer: it reads the cgroup's high-water mark on entry and on exit and passes the
    # author's arrays and return value through untouched. The lock already declares this wrapper as an operational patch with
    # no arithmetic effect; here it is what separates "accumulate and concatenate" from "score".
    original = exp_module.metric
    seen = {}

    def measured_metric(preds, trues):
        seen["before_scoring_peak"] = cgroup_peak_bytes()
        seen["shape"] = list(np.shape(preds))
        seen["dtype"] = str(np.asarray(preds).dtype)
        value = original(preds, trues)
        seen["after_scoring_peak"] = cgroup_peak_bytes()
        seen["author_returned_finite"] = [bool(np.isfinite(v)) for v in value]
        return value

    exp_module.metric = measured_metric
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open(work / "probe_author_stdout.log", "a") as fh, contextlib.redirect_stdout(fh):
            exp = exp_module.Exp_Long_Term_Forecast(args)
            setting = S.setting_of(args)
            stages["after_model_build"] = cgroup_peak_bytes()
            t0 = time.time()
            exp.test(setting, test=0)                                  # test=0: the freshly initialised weights, no checkpoint
            eval_seconds = time.time() - t0
    finally:
        os.chdir(cwd)
        exp_module.metric = original
    stages["after_evaluation"] = cgroup_peak_bytes()
    wall, cpu = time.time() - wall0, time.process_time() - cpu0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    measured_peak = cgroup_peak_bytes()
    derived = eval_path_memory_derivation(design, characterization, horizon, baseline_bytes=baseline_bytes)
    key = f"L{design['seq_len']}_h{horizon}"
    expected_windows = int(characterization["sets"][key]["test"]["windows"])
    population_ok = seen.get("shape", [None])[0] == expected_windows
    verdict = probe_verdict(measured_peak, cap)
    out = {"schema": PROBE_SCHEMA, "kind": "EVALUATION_PATH_MEMORY_PROBE",
           "reading": "a host-memory capacity measurement of the AUTHOR's unchunked evaluation path with an UNTRAINED model. "
                      "It stores NO metric: an untrained model's error is not a result. It alters nothing about the path it "
                      "measures, and this module holds no chunking, downcast or sub-sampling that could make an oversized "
                      "evaluation fit",
           "gate": "GATE_ONE: no scored cell may run before this measurement admits the path",
           "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
           "dataset": design["dataset"], "horizon_steps": int(horizon),
           "horizon_seconds": R.horizon_seconds(design["dataset"], horizon), "seq_len": design["seq_len"],
           "cell_id_geometry_from": cell["cell_id"], "trained": False, "author_eval_path": AUTHOR_EVAL_PATH,
           "population": {"expected_windows": expected_windows, "scored_shape": seen.get("shape"),
                          "dtype": seen.get("dtype"), "complete": population_ok,
                          "elements": derived["elements"]},
           "measured": {"whole_cgroup_peak_bytes_in_child": measured_peak,
                        "whole_cgroup_peak_gib": (measured_peak / float(1 << 30)) if measured_peak else None,
                        "cgroup_stage_peaks_bytes": {**stages, "before_scoring": seen.get("before_scoring_peak"),
                                                     "after_scoring": seen.get("after_scoring_peak")},
                        "cgroup": S._cgroup_memory(),
                        "peak_rss_bytes_self": int(ru1.ru_maxrss) * 1024,
                        "user_cpu_seconds": ru1.ru_utime - ru0.ru_utime, "system_cpu_seconds": ru1.ru_stime - ru0.ru_stime,
                        "evaluation_seconds": eval_seconds, "wall_seconds": wall, "cpu_seconds": cpu,
                        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
                        "host_memavailable_bytes_at_exit": meminfo_available_bytes()},
           "derived_for_contrast": derived,
           "measured_over_derived": ((measured_peak / derived["derived_peak_bytes"]) if measured_peak else None),
           "declared_cap_bytes": cap,
           "headroom_bytes": ((cap - measured_peak) if (cap and measured_peak) else None),
           "verdict": verdict, "verdict_reading": PROBE_VERDICTS[verdict],
           "device": str(exp.device), "device_uuid_measured_inside_child": S.actual_device_uuid(gpu),
           "device_assertion": device_check, "gpu_state_after": S.gpu_state(),
           "environment": S.environment(), "host_thermals_c": S.host_thermals(),
           "boot_id": (Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                       if Path("/proc/sys/kernel/random/boot_id").is_file() else None),
           "pid": os.getpid(), "started_at": started, "finished_at": S.now_iso(),
           "file_sha256": got, "author_scorer_finite_on_untrained": seen.get("author_returned_finite"),
           "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                          "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")}}
    if not population_ok:
        out["verdict"] = "REFUSED_INCOMPLETE_POPULATION"
        out["verdict_reading"] = (PROBE_VERDICTS["REFUSED_INCOMPLETE_POPULATION"] +
                                  f": {seen.get('shape')} against the sealed {expected_windows} windows")
    out["record_sha256"] = S.sha_obj(out)
    return out


# --- one scored cell ------------------------------------------------------------------------------------------------------

def author_scorer_identity() -> dict:
    """The identity of the function that actually produced the number: the author's own `utils.metrics` module source."""
    S.author_env()
    MET = importlib.import_module("utils.metrics")
    src = inspect.getsource(MET)
    return {"module": "utils.metrics", "function": "metric",
            "sha256": hashlib.sha256(src.encode()).hexdigest(),
            "reading": "MAE and MSE as the author computes them: float32 np.mean over EVERY test window x forecast step x "
                       "target channel element of the concatenated arrays"}


def transport_record(transport: dict) -> dict:
    """The governed delivery these bytes came from, read from the adoption campaign's own retained record."""
    need = ("campaign_key", "campaign_sha256", "delivery_id", "availability_contract_sha256", "verification_state",
            "adoption_record_sha256")
    missing = [k for k in need if not str(transport.get(k) or "").strip()]
    if missing:
        raise ExecutionRefusal(f"REFUSED: the declared transport of governed bytes is missing {missing}")
    if transport["verification_state"] != "VERIFIED_TRANSFER":
        raise ExecutionRefusal(f"REFUSED: the delivery's state is {transport['verification_state']!r}, not VERIFIED_TRANSFER")
    return {**transport, "class": TRANSPORT_CLASS, "reading": TRANSPORT_READING}


def run_scored_cell(design: dict, characterization: dict, *, data_path: Path, work: Path, horizon: int, seed: int,
                    transport: dict, gpu: int = 0, require_gpu_uuid: str | None = None,
                    dataloader_workers: int | None = 0, probe: dict | None = None) -> dict:
    """One sealed cell: the author's train-then-test, the paired persistence naive on the same rows, and the receipt through
    the sealed producer gate. Nothing scientific is decided here and no number is chosen here."""
    R.validate_design(design)
    # gate one first: it is the cheapest check and the most binding one
    if probe is not None:
        if probe.get("design_sha256") != design["design_sha256"]:
            raise ExecutionRefusal("REFUSED: the evaluation-path probe was not measured under this design")
        if probe.get("verdict") != "ADMISSIBLE":
            raise ExecutionRefusal(f"REFUSED: gate one did not admit the author's evaluation path ({probe.get('verdict')}); "
                                   "no scored cell runs behind a gate that has not passed")
    if characterization.get("design_sha256") != design["design_sha256"]:
        raise ExecutionRefusal("REFUSED: this characterization was not produced under this design")
    facts = R.dataset_facts(design["dataset"])
    got = S.sha_file(data_path)                                        # re-verified INSIDE the child, before the loader
    if got != facts["sha256"]:
        raise ExecutionRefusal(f"REFUSED: these are not the registered bytes of {facts['resource']} ({got[:12]})")
    drift = S.code_drift(design)
    if drift:
        raise ExecutionRefusal(f"REFUSED: the author files have drifted from the sealed digests: {sorted(drift)}")
    cell = next(c for c in design["cells"] if c["horizon_steps"] == horizon and c["seed"] == seed)
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]
    tr = transport_record(transport)
    scorer = author_scorer_identity()
    work.mkdir(parents=True, exist_ok=True)
    S.author_env()
    import torch
    device_check = S.assert_child_device(require_gpu_uuid, gpu)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gpu_before, thermals_before = S.gpu_state(), S.host_thermals()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    started, wall0 = S.now_iso(), time.time()
    res = S.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=data_path.parent, data_name=data_path.name,
                             work=work, gpu=gpu, log=work / "author_stdout.log", train=True, bounded=False,
                             dataloader_workers=dataloader_workers)
    cgroup_peak_after_cell = cgroup_peak_bytes()
    preds, trues = res["preds"], res["trues"]
    if np.shape(preds) != np.shape(trues):
        raise ExecutionRefusal(f"REFUSED: the author's scorer saw {np.shape(preds)} against {np.shape(trues)}")
    windows, steps, channels = (int(x) for x in np.shape(preds))
    if windows != int(sets["test"]["windows"]) or steps != int(horizon) or channels != int(sets["test"]["channels"]):
        raise ExecutionRefusal(f"REFUSED: scored population {np.shape(preds)} against the sealed "
                               f"{(sets['test']['windows'], horizon, sets['test']['channels'])}")
    elements = windows * steps * channels
    if elements != int(sets["elements_test"]):
        raise ExecutionRefusal(f"REFUSED: {elements} scored elements against the sealed {sets['elements_test']}")
    author = {"mae": float(res["author_metric"]["mae"]), "mse": float(res["author_metric"]["mse"])}
    independent = S.float64_metrics(np.asarray(preds), np.asarray(trues))
    pred_sha, true_sha = S.sha_array(np.asarray(preds)), S.sha_array(np.asarray(trues))
    finite = bool(S.all_finite(np.asarray(preds)))
    # the paired naive, on the SAME rows: the author's own test loader rebuilt from the same argv, and the pairing PROVED by
    # the digest of the targets rather than asserted in prose
    naive = S.naive_and_trues(design, {**cell, "horizon": horizon}, data_path)
    if naive["true_sha256"] != true_sha:
        raise ExecutionRefusal("REFUSED: the naive was computed on a different target population than the model was scored on "
                               f"({naive['true_sha256'][:12]} != {true_sha[:12]}); an unpaired baseline is not a baseline")
    if int(naive["n_elements"]) != elements or int(naive["windows"]) != windows:
        raise ExecutionRefusal(f"REFUSED: the naive covered {naive['n_elements']} elements over {naive['windows']} windows "
                               f"against the model's {elements} over {windows}")
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.time() - wall0
    finished = S.now_iso()
    training = S.parse_author_log((work / "author_stdout.log").read_text())
    costs = {"wall_seconds": wall, "cpu_seconds": (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)}
    population = {"sha256": true_sha, "windows": windows, "target_channels": channels, "elements": elements}
    receipt = R.build_receipt(
        design=design, characterization=characterization, horizon=horizon, seed=seed,
        metrics={"mse": author["mse"], "mae": author["mae"]},
        naive={"mse": float(naive["naive"]["mse"]), "mae": float(naive["naive"]["mae"])},
        population=population, model_commit=R.PINNED_COMMIT, scorer_sha256=scorer["sha256"],
        campaign_key=tr["campaign_key"], campaign_sha256=tr["campaign_sha256"], unit_id=cell["cell_id"],
        delivery_id=tr["delivery_id"], availability_contract_sha256=tr["availability_contract_sha256"],
        costs=costs, started_at=started, finished_at=finished,
        comparison_class="MATCHED_PUBLISHED_RECIPE_EXECUTED", metric_dtype="float32")
    gate = R.validate_receipt(receipt)                                 # the SEALED producer gate, unmodified
    record = {"schema": CELL_SCHEMA, "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
              "dataset": design["dataset"], "cell_id": cell["cell_id"], "horizon_steps": int(horizon),
              "horizon_seconds": R.horizon_seconds(design["dataset"], horizon), "seed": int(seed),
              "seq_len": design["seq_len"], "configuration_sha256": cell["configuration_sha256"],
              "setting": res["setting"], "argv": cell["argv"],
              "effective_args": {k: v for k, v in sorted(res["args"].items())},
              "metric": {"author_float32": author, "independent_float64": independent,
                         "scorer": scorer, "space": "z_train (the training-scaler normalized target space; --inverse False)",
                         "reduction": R.contract()["reduction"],
                         "author_float32_vs_independent_float64": {"mae": author["mae"] - independent["mae"],
                                                                   "mse": author["mse"] - independent["mse"]}},
              "naive": {"mse": float(naive["naive"]["mse"]), "mae": float(naive["naive"]["mae"]),
                        "definition": naive["naive_definition"],
                        "paired_on_the_same_rows_proved_by": {"target_population_sha256": true_sha,
                                                              "naive_target_sha256": naive["true_sha256"],
                                                              "equal": True}},
              "population": {**population, "predictions_sha256": pred_sha, "all_finite": finite,
                             "expected_from_characterization": {"windows": int(sets["test"]["windows"]),
                                                                "channels": int(sets["test"]["channels"]),
                                                                "elements": int(sets["elements_test"])}},
              "evaluation_path": {"author_test": True, "chunked": False, "adapter": None, "path": AUTHOR_EVAL_PATH,
                                  "admitted_by_probe": (probe or {}).get("record_sha256"),
                                  "probe_verdict": (probe or {}).get("verdict")},
              "training": training,
              "convergence": {"epochs_sealed": int(res["args"]["train_epochs"]),
                              "epochs_run": training["epochs_run"], "early_stopped": training["early_stopped"],
                              "best_epoch_by_validation": training["best_epoch_by_vali"],
                              "status": ("EARLY_STOPPED_ON_VALIDATION" if training["early_stopped"] else
                                         ("EPOCH_BUDGET_CEILING_BEST_AT_LAST_EPOCH"
                                          if training["best_epoch_by_vali"] == int(res["args"]["train_epochs"]) else
                                          "RAN_THE_SEALED_EPOCH_BUDGET_BEST_BEFORE_THE_END")),
                              "reading": "the checkpoint scored is the lowest-validation-MSE epoch's, reloaded by the author's "
                                         "train() before test(). A best epoch at the sealed budget's last epoch means the "
                                         "budget, not convergence, ended the run"},
              "operational_patches": res.get("operational_patches", []),
              "transport": tr,
              "resources": {"wall_seconds": wall, "cpu_seconds": costs["cpu_seconds"],
                            "user_cpu_seconds": ru1.ru_utime - ru0.ru_utime,
                            "system_cpu_seconds": ru1.ru_stime - ru0.ru_stime,
                            "peak_rss_bytes_self": int(ru1.ru_maxrss) * 1024,
                            "whole_cgroup_peak_bytes_in_child": cgroup_peak_bytes(),
                            "whole_cgroup_peak_after_author_run_bytes": cgroup_peak_after_cell,
                            "cgroup": S._cgroup_memory(), "declared_cap_bytes": declared_cap_bytes(),
                            "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated())
                                                         if torch.cuda.is_available() else 0),
                            "peak_gpu_reserved_bytes": (int(torch.cuda.max_memory_reserved())
                                                        if torch.cuda.is_available() else 0),
                            "gpu_before": gpu_before, "gpu_after": S.gpu_state(),
                            "host_thermals_before_c": thermals_before, "host_thermals_after_c": S.host_thermals(),
                            "host_memavailable_bytes_at_exit": meminfo_available_bytes()},
              "device": res["device"], "device_uuid_measured_inside_child": S.actual_device_uuid(gpu),
              "device_assertion": device_check, "n_parameters": res["n_parameters"],
              "checkpoint_sha256": S.sha_file(Path(res["checkpoint"])) if Path(res["checkpoint"]).is_file() else None,
              "environment": S.environment(), "source_files_sha256": S.source_digests(),
              "source_drift": drift, "author_clone": S.author_git(), "file_sha256": got,
              "boot_id": (Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                          if Path("/proc/sys/kernel/random/boot_id").is_file() else None),
              "pid": os.getpid(), "started_at": started, "finished_at": finished,
              "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                             "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")},
              "receipt": receipt, "receipt_gate": gate}
    record["record_sha256"] = S.sha_obj(record)
    return record


# --- the comparison row and the closure -----------------------------------------------------------------------------------

def published_row(design: dict) -> dict:
    """THIS dataset's published row, from the sealed lock — never from a number typed here."""
    pub = design["lock"]["published"]
    if not pub:
        raise ExecutionRefusal("REFUSED: this design pins no published row, so no comparison may be composed")
    return pub


def margin(design: dict) -> dict:
    """THIS dataset's own predeclared operational margin, from the sealed lock. Electricity's margin is not Weather's and
    df_sota_repro.AGREEMENT (which holds Electricity's) is never read here."""
    ag = design["lock"]["agreement"]
    if ag.get("state") == "NO_MARGIN_EXISTS":
        raise ExecutionRefusal("REFUSED: no published dispersion is pinned for this dataset, so no comparability class exists")
    return ag


def classify(design: dict, values: list, published_value: float, metric: str) -> dict:
    """The predeclared class for a seed mean against a published value, under THIS dataset's margin."""
    ag = margin(design)
    if not values:
        return {"class": "NO_NEW_MEASUREMENT", "mean": None, "published": float(published_value),
                "reading": "no cell of this contrast produced a value"}
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
    std_paper = float(ag["std_paper"][metric])
    tol_a = float(ag["k_agree"]) * std_paper + float(ag["rounding"])
    tol_p = float(ag["k_partial"]) * std_paper + float(ag["rounding"])
    diff = mean - float(published_value)
    cls = ("OPERATIONAL_AGREEMENT" if abs(diff) <= tol_a else
           ("OPERATIONAL_PARTIAL" if abs(diff) <= tol_p else "OUTSIDE_OPERATIONAL_MARGIN"))
    return {"class": cls, "mean": mean, "seed_sd_ddof1": sd, "values": [float(v) for v in values],
            "n_seeds": len(values), "published": float(published_value), "difference": diff,
            "tolerance_agreement": tol_a, "tolerance_partial": tol_p,
            "margin_source": ag["source"], "margin_std_paper": std_paper,
            "reading": ag["rule"]}


def comparison_row(design: dict, record: dict) -> dict:
    """One cell's FULL comparison row, composed only from that cell's retained record and the sealed lock."""
    R.validate_design(design)
    if record.get("design_sha256") != design["design_sha256"]:
        raise ExecutionRefusal("REFUSED: this cell record was not produced under this design")
    if record.get("record_sha256") != S.sha_obj({k: v for k, v in record.items() if k != "record_sha256"}):
        raise ExecutionRefusal("REFUSED: this cell record's digest does not recompute from its content")
    h = str(record["horizon_steps"])
    pub = published_row(design)["per_horizon"][h]
    m, nv = record["metric"]["author_float32"], record["naive"]
    res = record["resources"]
    row = {"schema": ROW_SCHEMA, "cell_id": record["cell_id"], "dataset": record["dataset"],
           "split": R.contract()["split"], "horizon_steps": record["horizon_steps"],
           "horizon_seconds": record["horizon_seconds"],
           "horizon_elapsed": f"{record['horizon_seconds'] / 3600:g} h",
           "input_window_steps": record["seq_len"], "seed": record["seed"],
           "model_error": {"mse": m["mse"], "mae": m["mae"]},
           "metric": {"names": ["sota.test.mse_normalized", "sota.test.mae_normalized"],
                      "space": record["metric"]["space"], "reduction": record["metric"]["reduction"],
                      "dtype": "float32 (the author's own reduction); a float64 reduction over the same arrays is reported beside it",
                      "independent_float64": record["metric"]["independent_float64"],
                      "scorer_sha256": record["metric"]["scorer"]["sha256"]},
           "paired_naive": {"mse": nv["mse"], "mae": nv["mae"], "definition": nv["definition"],
                            "same_rows": nv["paired_on_the_same_rows_proved_by"]},
           "skill_vs_naive": {"mse": 1.0 - (m["mse"] / nv["mse"]) if nv["mse"] else None,
                              "mae": 1.0 - (m["mae"] / nv["mae"]) if nv["mae"] else None,
                              "definition": "1 - model_error / paired_naive_error on the same rows, same space, same reduction"},
           "published": {"mse": float(pub["mse"]), "mae": float(pub["mae"]),
                         "source": f"{design['lock']['paper']} — {published_row(design)['table']}",
                         "read_at": design["lock"]["paper_read_at"],
                         "not_current_sota": design["lock"]["published_is_not_current_sota"]},
           "difference_vs_published": {"mse": m["mse"] - float(pub["mse"]), "mae": m["mae"] - float(pub["mae"])},
           "comparability": {"class": record["receipt"]["tags"]["comparison_class"],
                             "single_seed_not_a_class": "a per-seed row carries a difference, not an agreement class: the "
                                                        "predeclared class is defined on the three-seed mean and is assigned "
                                                        "in the closure, never here",
                             "margin_source": margin(design)["source"],
                             "never": design["lock"]["comparability"]["never"]},
           "dataset_identity": {"resource": record["receipt"]["tags"]["resource"],
                                "dataset_sha256": record["file_sha256"],
                                "population_sha256": record["population"]["sha256"],
                                "windows": record["population"]["windows"],
                                "elements": record["population"]["elements"],
                                "scaler_sha256": record["receipt"]["tags"]["scaler_sha256"],
                                "scaler_fit_population_sha256": record["receipt"]["tags"]["scaler_fit_population_sha256"]},
           "convergence": record["convergence"],
           "resources": {"wall_seconds": res["wall_seconds"], "cpu_seconds": res["cpu_seconds"],
                         "peak_gpu_allocated_bytes": res["peak_gpu_allocated_bytes"],
                         "whole_cgroup_peak_bytes": res["whole_cgroup_peak_bytes_in_child"],
                         "declared_cap_bytes": res["declared_cap_bytes"],
                         "device_uuid": record["device_uuid_measured_inside_child"]},
           "evidence_class": "measured under the frozen design, receipts retained, NOT_INDEPENDENTLY_VERIFIED (one host, one "
                             "execution per cell, not replayed in a fresh process)",
           "transport": {"class": record["transport"]["class"], "delivery_id": record["transport"]["delivery_id"],
                         "campaign_key": record["transport"]["campaign_key"],
                         "reading": record["transport"]["reading"]},
           "from_record_sha256": record["record_sha256"], "receipt_terminal_sha256": record["receipt"]["terminal_sha256"]}
    row["row_sha256"] = S.sha_obj(row)
    return row


def closure(design: dict, records: list, *, probe: dict | None = None) -> dict:
    """The closure table: per horizon the three-seed mean with its dispersion beside the published value and its comparability
    class, then the four-horizon average formed WITHIN each seed first. Composed only from retained records."""
    R.validate_design(design)
    rows = [comparison_row(design, r) for r in records]
    pub = published_row(design)
    seeds = sorted({int(r["seed"]) for r in records})
    horizons = [int(h) for h in design["horizons"]]
    per_horizon, missing = {}, []
    for h in horizons:
        got = [r for r in records if int(r["horizon_steps"]) == h]
        if len(got) != len(design["seeds"]):
            missing.append({"horizon_steps": h, "cells_present": len(got), "cells_sealed": len(design["seeds"])})
        mse = [r["metric"]["author_float32"]["mse"] for r in got]
        mae = [r["metric"]["author_float32"]["mae"] for r in got]
        nmse = [r["naive"]["mse"] for r in got]
        nmae = [r["naive"]["mae"] for r in got]
        p = pub["per_horizon"][str(h)]
        per_horizon[str(h)] = {
            "horizon_steps": h, "horizon_seconds": R.horizon_seconds(design["dataset"], h),
            "horizon_elapsed": f"{R.horizon_seconds(design['dataset'], h) / 3600:g} h",
            "seeds": [int(r["seed"]) for r in got],
            "mse": classify(design, mse, p["mse"], "mse"), "mae": classify(design, mae, p["mae"], "mae"),
            "paired_naive": {"mse": float(np.mean(nmse)) if nmse else None, "mae": float(np.mean(nmae)) if nmae else None,
                             "identical_across_seeds": (len(set(nmse)) == 1 if nmse else None),
                             "reading": "the naive does not depend on the seed: the same rows, so one value per horizon"},
            "skill_vs_naive": {"mse": (1.0 - float(np.mean(mse)) / float(np.mean(nmse))) if nmse else None,
                               "mae": (1.0 - float(np.mean(mae)) / float(np.mean(nmae))) if nmae else None},
            "cells": [r["cell_id"] for r in got]}
    within_seed = {}
    for s in seeds:
        got = {int(r["horizon_steps"]): r for r in records if int(r["seed"]) == s}
        if sorted(got) == sorted(horizons):
            within_seed[str(s)] = {"mse": float(np.mean([got[h]["metric"]["author_float32"]["mse"] for h in horizons])),
                                   "mae": float(np.mean([got[h]["metric"]["author_float32"]["mae"] for h in horizons]))}
    avg = {"mse": classify(design, [v["mse"] for v in within_seed.values()], pub["average"]["mse"], "mse"),
           "mae": classify(design, [v["mae"] for v in within_seed.values()], pub["average"]["mae"], "mae"),
           "within_seed_averages": within_seed,
           "formed": "the four-horizon average is formed WITHIN each seed first, then across seeds — the paper's own quantity"}
    out = {"schema": CLOSURE_SCHEMA, "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
           "dataset": design["dataset"], "seq_len": design["seq_len"], "protocol": design["protocol"],
           "cells_sealed": len(design["cells"]), "cells_present": len(records), "missing": missing,
           "published_row": {**pub, "source": design["lock"]["paper"], "read_at": design["lock"]["paper_read_at"],
                             "margin": margin(design)["std_paper"], "margin_source": margin(design)["source"],
                             "not_current_sota": design["lock"]["published_is_not_current_sota"]},
           "per_horizon": per_horizon, "four_horizon_average": avg,
           "gate_one": ({"verdict": probe["verdict"], "measured_whole_cgroup_peak_bytes":
                         probe["measured"]["whole_cgroup_peak_bytes_in_child"],
                         "derived_peak_bytes": probe["derived_for_contrast"]["derived_peak_bytes"],
                         "declared_cap_bytes": probe["declared_cap_bytes"],
                         "record_sha256": probe["record_sha256"]} if probe else
                        {"verdict": "NOT_PRESENTED", "reading": "no probe record was composed into this closure"}),
           "rows": rows,
           "evidence_classes": {
               "model_error_and_paired_naive": "measured under the frozen design on the registered bytes; receipts retained; "
                                               "NOT_INDEPENDENTLY_VERIFIED (one execution per cell, one host, no replay)",
               "published_row": "VERIFIED as a reading of the named paper's own tables; not a verification of the paper's "
                                "numbers and not a claim of current best SOTA",
               "comparability": "a predeclared OPERATIONAL class against THIS dataset's own published dispersion; not "
                                "statistical equivalence. A missing comparability is never cured by rescaling",
               "governance": TRANSPORT_CLASS},
           "composed_at": S.now_iso(), "composed_from_records": [r["record_sha256"] for r in records]}
    out["closure_sha256"] = S.sha_obj(out)
    return out


def markdown(clo: dict) -> str:
    """The closure table as the owner's standing rule requires it: model error with its space, the naive on the same rows, the
    skill, the published value with its source, and the comparability class — generated from the artifacts, never typed."""
    L = []
    pub = clo["published_row"]
    L.append(f"| horizon | elapsed | model MSE (z, mean of {len(clo['four_horizon_average']['within_seed_averages'])} seeds) "
             "| published MSE | class | model MAE (z) | published MAE | class | naive MSE (same rows) | skill MSE |")
    L.append("|---|---:|---:|---:|---|---:|---:|---|---:|---:|")
    for h in sorted(clo["per_horizon"], key=lambda x: int(x)):
        r = clo["per_horizon"][h]
        mse, mae = r["mse"], r["mae"]
        L.append(f"| {h} | {r['horizon_elapsed']} | {mse['mean']:.4f} ± "
                 f"{(mse['seed_sd_ddof1'] or 0):.4f} | {mse['published']:.3f} | {mse['class']} | "
                 f"{mae['mean']:.4f} ± {(mae['seed_sd_ddof1'] or 0):.4f} | {mae['published']:.3f} | {mae['class']} | "
                 f"{r['paired_naive']['mse']:.4f} | {r['skill_vs_naive']['mse']:.4f} |")
    a = clo["four_horizon_average"]
    L.append(f"| **avg** | | **{a['mse']['mean']:.4f} ± {(a['mse']['seed_sd_ddof1'] or 0):.4f}** | "
             f"**{a['mse']['published']:.3f} ± {pub['margin']['mse']}** | **{a['mse']['class']}** | "
             f"**{a['mae']['mean']:.4f} ± {(a['mae']['seed_sd_ddof1'] or 0):.4f}** | "
             f"**{a['mae']['published']:.3f} ± {pub['margin']['mae']}** | **{a['mae']['class']}** | | |")
    L.append("")
    L.append(f"Space: {clo['rows'][0]['metric']['space']}. Reduction: {clo['rows'][0]['metric']['reduction']}.")
    L.append(f"Published source: {pub['source']} — {pub['table']}; margin {pub['margin']} from {pub['margin_source']}.")
    L.append(f"Comparability: {clo['evidence_classes']['comparability']}")
    L.append(f"Governance: {clo['evidence_classes']['governance']}")
    return "\n".join(L)


# --- CLI ------------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["derive", "probe", "cell", "close"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", default="weather")
    ap.add_argument("--protocol", default="L96")
    ap.add_argument("--data-path", type=Path, default=None)
    ap.add_argument("--horizon", type=int, default=720)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dataloader-workers", type=int, default=0)
    ap.add_argument("--require-gpu-uuid", default=os.environ.get(S.REQUIRED_GPU_ENV) or None)
    ap.add_argument("--baseline-bytes", type=int, default=0,
                    help="the TRAIN-only pilot's measured whole-cgroup peak, used ONLY as the derivation's baseline term")
    ap.add_argument("--pilot", type=Path, default=None, help="the retained pilot record the baseline is read from")
    ap.add_argument("--transport", type=Path, default=None, help="the retained adoption record of the governed delivery")
    ap.add_argument("--probe", type=Path, default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    design = json.loads((root / f"DESIGN.{a.dataset}.{a.protocol}.json").read_text())
    chars = json.loads((root / f"CHARACTERIZATION.{a.dataset}.json").read_text())
    baseline = a.baseline_bytes
    if a.pilot is not None:
        pilot = json.loads(Path(a.pilot).read_text())
        if pilot.get("design_sha256") != design["design_sha256"]:
            raise ExecutionRefusal("REFUSED: that pilot record was not measured under this design")
        baseline = int((pilot.get("cgroup") or {}).get("memory.peak") or 0)

    if a.command == "derive":
        d = eval_path_memory_derivation(design, chars, a.horizon, baseline_bytes=baseline)
        print(json.dumps(d, indent=1, default=str))
        return 0
    if a.command == "probe":
        rec = eval_probe(design, chars, data_path=Path(a.data_path), work=root / "probe_work", horizon=a.horizon,
                         gpu=a.gpu, require_gpu_uuid=a.require_gpu_uuid, dataloader_workers=a.dataloader_workers,
                         baseline_bytes=baseline)
        path = _write(root / f"EVAL_PROBE.{a.dataset}.h{a.horizon}.json", rec)
        print(json.dumps({"path": str(path), "verdict": rec["verdict"],
                          "measured_whole_cgroup_peak_bytes": rec["measured"]["whole_cgroup_peak_bytes_in_child"],
                          "measured_whole_cgroup_peak_gib": rec["measured"]["whole_cgroup_peak_gib"],
                          "derived_peak_bytes": rec["derived_for_contrast"]["derived_peak_bytes"],
                          "derived_peak_gib": rec["derived_for_contrast"]["derived_peak_gib"],
                          "declared_cap_bytes": rec["declared_cap_bytes"], "headroom_bytes": rec["headroom_bytes"],
                          "evaluation_seconds": rec["measured"]["evaluation_seconds"]}, indent=1))
        return 0 if rec["verdict"] == "ADMISSIBLE" else 3
    if a.command == "cell":
        probe = json.loads(Path(a.probe).read_text()) if a.probe else None
        transport = json.loads(Path(a.transport).read_text())
        rec = run_scored_cell(design, chars, data_path=Path(a.data_path),
                              work=root / f"cell_{a.dataset}_h{a.horizon}_s{a.seed}", horizon=a.horizon, seed=a.seed,
                              transport=transport, gpu=a.gpu, require_gpu_uuid=a.require_gpu_uuid,
                              dataloader_workers=a.dataloader_workers, probe=probe)
        path = _write(root / "CELLS" / f"{rec['cell_id']}.json", rec)
        print(json.dumps({"path": str(path), "cell_id": rec["cell_id"],
                          "mse": rec["metric"]["author_float32"]["mse"], "mae": rec["metric"]["author_float32"]["mae"],
                          "naive_mse": rec["naive"]["mse"], "naive_mae": rec["naive"]["mae"],
                          "epochs_run": rec["training"]["epochs_run"], "early_stopped": rec["training"]["early_stopped"],
                          "wall_seconds": rec["resources"]["wall_seconds"],
                          "whole_cgroup_peak_bytes": rec["resources"]["whole_cgroup_peak_bytes_in_child"],
                          "receipt_gate": rec["receipt_gate"]["checks"]}, indent=1))
        return 0
    if a.command == "close":
        records = [json.loads(p.read_text()) for p in sorted((root / "CELLS").glob("*.json"))]
        probe_path = a.probe or (root / f"EVAL_PROBE.{a.dataset}.h{max(design['horizons'])}.json")
        probe = json.loads(Path(probe_path).read_text()) if Path(probe_path).is_file() else None
        clo = closure(design, records, probe=probe)
        _write(root / f"CLOSURE.{a.dataset}.{a.protocol}.json", clo)
        (root / f"CLOSURE.{a.dataset}.{a.protocol}.md").write_text(markdown(clo) + "\n")
        print(markdown(clo))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
