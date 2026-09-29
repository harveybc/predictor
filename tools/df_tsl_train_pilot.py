#!/usr/bin/env python3
"""TRAFFIC TRAIN-ONLY COST PILOT: the training footprint the bounded evaluator does not carry.

`tools/df_tsl_bounded_eval.py` measured Traffic's EVALUATION path at 3.600 GiB whole-cgroup on the complete h720 test
population. That number is an evaluation of an UNTRAINED model: no optimizer was built, no gradient was allocated, no
optimizer slot was instantiated and no checkpoint was written. It is therefore not a training footprint and nothing here
reads it as one.

What this module measures, in ONE cell scope, on a TRAIN-only fixture of the planned shape (the author's own train split of
the registered bytes, at the sealed geometry, the sealed batch size and the sealed dtype):

  * the model built from the sealed argv, on the device asserted by physical UUID from INSIDE the child;
  * the author's own train loader and validation loader --- and NOTHING else: `_get_data` is wrapped and REFUSES the test
    split, so a cost pilot cannot become a peek at the held-out data by accident;
  * forward, backward and `optimizer.step()`, the author's own body verbatim, with the optimizer's slots MEASURED after the
    first step --- a record whose optimizer state is zero bytes is refused, because a training footprint without the slots is
    not a training footprint;
  * WARMUP steps and STEADY steps kept apart, because the first steps pay for allocator growth and cuDNN autotuning and
    their cost is not the per-step cost;
  * the LARGEST FINAL-BATCH shape. The author's `data_provider` sets `drop_last=False` for every flag, so each loader ends on
    a partial batch whose shape the steady loop never sees; a new shape forces fresh allocations while the steady blocks are
    still cached, so it is measured rather than assumed away;
  * the checkpoint WRITE and the checkpoint RELOAD, the author's own `EarlyStopping.save_checkpoint` and the
    `load_state_dict` at the end of `train()`;
  * the VALIDATION mechanics: the author's `vali()` body over the COMPLETE validation loader --- the per-epoch cost, measured,
    not projected. Its loss VALUE is deliberately not recorded: this is a cost pilot, and a validation loss is a selection
    signal.

What it does NOT do, and what no sentence here should be read as doing:

  * it does not score anything. No test window is opened. No metric is stored. There is no accuracy number to select on.
  * it does not shorten, shrink or alter the sealed recipe. The batch size, dtype, optimizer, loader, model and geometry are
    the design's own; only the NUMBER OF STEPS is bounded, and that bound is named in the record.
  * it does not transfer another dataset's footprint. `refuse_foreign_pilot` refuses to price one design from another
    design's pilot record: Weather's 862-channel-less model is a different width and its footprint is not Traffic's.
  * it does not claim a maximum. `projection` reports a peak and a projection WITH EXPLICIT HEADROOM, and says in the record
    that a measured peak on one host in one execution is not a proof of the maximum memory the path can take.
  * it does not read the elapsed horizon off the step count. `df_tsl_repro.horizon_seconds` is the only clock: Traffic's step
    is 3600 s, so its 96 steps are 96 HOURS where Weather's 96 steps are 16.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import os
import re
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402  the author bridge (model, argv, device, env)
import df_tsl_repro as R                                               # noqa: E402  the sealed design, characterization and clock
import df_tsl_execute as X                                             # noqa: E402  cgroup peak and declared cap readers

PILOT_SCHEMA = "df_tsl_train_pilot.v2"
PROJECTION_SCHEMA = "df_tsl_train_projection.v1"

#: v1 (`df_tsl_repro.train_only_pilot`) is SEALED and unchanged. It timed K optimizer steps and read one cgroup peak. It did
#: not open the validation loader, did not measure the optimizer's slots, did not touch a final-batch shape and never wrote or
#: reloaded a checkpoint, so its peak is not the training footprint of a cell. v2 adds those stages; it changes no recipe.
SUPERSEDES = {"df_tsl_train_pilot.v1": "one stage only: model build, train loader, K steps, one peak. No optimizer-slot "
                                       "measurement, no final-batch shape, no checkpoint write or reload, no validation "
                                       "mechanics. Its peak is a floor for a training cell, never the cell's footprint"}

#: Every stage that must appear in a record before it is allowed to be called a training footprint.
REQUIRED_STAGES = ("entry", "after_model_build", "after_train_loader", "after_validation_loader", "after_optimizer_build",
                   "after_first_optimizer_step", "after_warmup_steps", "after_steady_steps", "after_final_batch_shapes",
                   "after_checkpoint_write", "after_checkpoint_reload", "after_validation_pass", "exit")

PILOT_VERDICTS = {
    "MEASURED_WITHIN_CAP": "every required stage ran in one cell scope, the optimizer's slots were instantiated and measured, "
                           "and the whole-cgroup peak of the whole thing is at or under the cap this child was admitted "
                           "under. It is a measurement of THIS execution on THIS host, not a proof of a maximum",
    "CAPACITY_DEFICIT": "the training path's measured peak is above the cap this child was admitted under. The cap is not "
                        "shrunk and no stage is dropped to make it fit; this is a named deficit",
    "UNDETERMINED": "the peak or the cap could not be read. A missing peak is UNKNOWN, never zero and never success",
    "REFUSED_NO_OPTIMIZER_STATE": "the optimizer held no slot tensors after the first step. A footprint measured without the "
                                  "optimizer's state is not a training footprint and is not reported as one",
    "REFUSED_INCOMPLETE_STAGE_COVERAGE": "a required stage did not run in this cell scope. A partial footprint is not the "
                                         "footprint of the path it is supposed to price",
    "REFUSED_TEST_ACCESS": "the test split was opened. A TRAIN-only pilot that reads held-out data is not a TRAIN-only pilot",
}


class TrainPilotRefusal(SystemExit):
    """Nothing is measured, projected or published from a state that cannot be shown."""


# --- pure arithmetic: batch geometry, the clock, and the verdict -----------------------------------------------------------

def final_batch_rows(windows: int, batch_size: int, *, drop_last: bool = False) -> int:
    """The number of rows in a loader's LAST batch. The author's `data_provider` sets `drop_last=False` for every flag, so a
    partial last batch is reached on every split; `drop_last` is a parameter here only so a test can show that the author's
    own setting is the one being used."""
    n, bs = int(windows), int(batch_size)
    if bs <= 0:
        raise TrainPilotRefusal("REFUSED: a batch size of zero has no batch geometry")
    if n <= 0:
        return 0
    rem = n % bs
    if rem == 0:
        return bs
    return bs if drop_last else rem


def n_batches(windows: int, batch_size: int, *, drop_last: bool = False) -> int:
    n, bs = int(windows), int(batch_size)
    if n <= 0:
        return 0
    return (n // bs) if drop_last else -(-n // bs)


def split_geometry(design: dict, characterization: dict, horizon: int) -> dict:
    """Window and batch counts for every split of one horizon, from the sealed characterization's own counts. It reads no
    bytes: the test row is COUNTS ONLY and is marked as such, because a TRAIN-only pilot may price a pass it does not run but
    may not open the data that pass would read."""
    key = f"L{design['seq_len']}_h{int(horizon)}"
    sets = characterization["sets"][key]
    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon))
    bs = int(cell["effective_args"]["batch_size"])
    out = {"key": key, "batch_size": bs, "splits": {}}
    for flag in ("train", "vali", "test"):
        w = int(sets[flag]["windows"])
        out["splits"][flag] = {"windows": w, "channels": int(sets[flag]["channels"]),
                               "batches": n_batches(w, bs), "final_batch_rows": final_batch_rows(w, bs),
                               "opened_by_this_pilot": flag in ("train", "vali"),
                               "source": ("the author's own loader, opened here" if flag in ("train", "vali")
                                          else "COUNTS ONLY, from the sealed characterization. This pilot does not open it")}
    finals = {f: out["splits"][f]["final_batch_rows"] for f in ("train", "vali")}
    out["largest_final_batch_rows_over_opened_splits"] = max(finals.values())
    out["final_batch_rows_by_opened_split"] = finals
    out["drop_last"] = False
    out["drop_last_source"] = "data_provider/data_factory.py sets drop_last=False for every flag at the pinned commit"
    return out


def horizon_clock(design: dict, horizon: int) -> dict:
    """Traffic's steps are hourly. The same step count is not the same elapsed horizon on another dataset, and this is the
    only place the elapsed horizon is produced."""
    ds = design["dataset"]
    sec = R.horizon_seconds(ds, int(horizon))
    return {"dataset": ds, "horizon_steps": int(horizon), "step_seconds": int(R.dataset_facts(ds)["step_seconds"]),
            "horizon_seconds": sec, "horizon_hours": sec / 3600.0, "horizon_days": sec / 86400.0,
            "reading": f"{int(horizon)} steps of {ds} are {sec / 3600.0:.0f} hours. A step count is not an elapsed horizon "
                       f"until it is multiplied by THIS resource's own sample interval"}


def pilot_verdict(*, measured_peak_bytes, cap_bytes, optimizer_slot_bytes, stages_present, test_split_opened: bool) -> str:
    """A comparison of measured numbers, and nothing else. No cap is shrunk to admit a peak and no stage is dropped to admit a
    record. An unread peak is UNDETERMINED, never zero."""
    if test_split_opened:
        return "REFUSED_TEST_ACCESS"
    missing = [s for s in REQUIRED_STAGES if s not in set(stages_present or ())]
    if missing:
        return "REFUSED_INCOMPLETE_STAGE_COVERAGE"
    if not optimizer_slot_bytes:
        return "REFUSED_NO_OPTIMIZER_STATE"
    if measured_peak_bytes is None or cap_bytes is None:
        return "UNDETERMINED"
    if int(measured_peak_bytes) > int(cap_bytes):
        return "CAPACITY_DEFICIT"
    return "MEASURED_WITHIN_CAP"


def refuse_foreign_pilot(design: dict, pilot: dict) -> None:
    """The transfer this module exists to prevent. Weather's model is 42 graph tokens wide at d_model 128; Traffic's is 862 at
    512. A per-step cost or a memory peak measured under one design prices nothing under another, and the refusal is here
    rather than in a reader's memory."""
    if pilot.get("design_sha256") != design.get("design_sha256"):
        raise TrainPilotRefusal("REFUSED: that pilot record was not measured under this design. A training footprint measured "
                                "on one dataset's model width is not transferable to another's")
    if str(pilot.get("dataset")) != str(design.get("dataset")):
        raise TrainPilotRefusal(f"REFUSED: that pilot is a {pilot.get('dataset')!r} record and this design is "
                                f"{design.get('dataset')!r}")
    if pilot.get("schema") != PILOT_SCHEMA:
        raise TrainPilotRefusal(f"REFUSED: {pilot.get('schema')!r} is not {PILOT_SCHEMA}. {SUPERSEDES.get(pilot.get('schema'), '')}")


# --- the measurement -------------------------------------------------------------------------------------------------------

def _gpu_now(torch) -> dict:
    if not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {"allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved())}


def _stage(stages: dict, name: str, torch) -> None:
    """A stage reading is the cgroup's own monotone high-water plus torch's own monotone high-water, taken at a boundary. It is
    not a sample: the kernel maintains it, so a stage shorter than the launcher's 5 s sampler is still measured here."""
    stages[name] = {"cgroup_peak_bytes": X.cgroup_peak_bytes(), **_gpu_now(torch)}


def _optimizer_slots(optim, torch) -> dict:
    """The slots Adam allocates lazily on the FIRST step: one exp_avg and one exp_avg_sq per parameter. Measured, never derived
    from the parameter count, because a record that derives them cannot tell an instantiated slot from an absent one."""
    tensors, names, params = 0, set(), 0
    for st in optim.state.values():
        params += 1
        for k, v in st.items():
            if torch.is_tensor(v):
                names.add(k)
                tensors += int(v.numel()) * int(v.element_size())
    return {"bytes": tensors, "slot_names": sorted(names), "parameters_with_state": params,
            "reading": "measured from the optimizer's own state after the first step. Zero bytes here means the slots were "
                       "never instantiated, and such a record is refused"}


def train_footprint_pilot(design: dict, characterization: dict, *, data_path: Path, work: Path, horizon: int,
                          seed: int | None = None, steps: int = 30, warmup_steps: int = 5, gpu: int = 0,
                          require_gpu_uuid: str | None = None, dataloader_workers: int | None = None,
                          full_validation: bool = True, validation_batches: int | None = None) -> dict:
    """The TRAIN-only cost pilot, all stages in ONE cell scope. Returns a record; stores no score."""
    R.validate_design(design)
    if characterization.get("design_sha256") != design["design_sha256"]:
        raise TrainPilotRefusal("REFUSED: this characterization was not produced under this design")
    if int(steps) < 2 or int(warmup_steps) < 1 or int(warmup_steps) >= int(steps):
        raise TrainPilotRefusal("REFUSED: the step budget must leave at least one warmup step and one steady step")
    facts = R.dataset_facts(design["dataset"])
    got = S.sha_file(Path(data_path))
    if got != facts["sha256"]:
        raise TrainPilotRefusal(f"REFUSED: the pilot would read bytes that are not the registered resource ({got[:12]})")
    drift = S.code_drift(design)
    if drift:
        raise TrainPilotRefusal(f"REFUSED: the author files have drifted from the sealed digests: {sorted(drift)}")

    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon) and (seed is None or c["seed"] == seed))
    geom = split_geometry(design, characterization, horizon)
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cap = X.declared_cap_bytes()

    S.author_env()
    import torch
    device_check = S.assert_child_device(require_gpu_uuid, gpu)
    S.fix_seeds(cell["seed"])
    args = S.build_args(cell["argv"], data_dir=Path(data_path).parent, data_name=Path(data_path).name,
                        checkpoints=work / "checkpoints", gpu=gpu, dataloader_workers=dataloader_workers)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")

    started = S.now_iso()
    stages: dict = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _stage(stages, "entry", torch)
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0, cpu0 = time.time(), time.process_time()
    splits_opened: list = []
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open(work / "train_pilot_author_stdout.log", "a") as fh, contextlib.redirect_stdout(fh):
            t = time.time()
            exp = exp_module.Exp_Long_Term_Forecast(args)
            model_build_seconds = time.time() - t
            setting = S.setting_of(args)
            _stage(stages, "after_model_build", torch)
            n_params = int(sum(p.numel() for p in exp.model.parameters()))
            param_bytes = int(sum(p.numel() * p.element_size() for p in exp.model.parameters()))
            mask_bytes = int(exp.masks.numel() * exp.masks.element_size()) if torch.is_tensor(exp.masks) else None

            # the guard: a TRAIN-only pilot may open train and val and nothing else
            _get_data = exp._get_data

            def guarded(flag):
                splits_opened.append(str(flag))
                if str(flag) == "test":
                    raise TrainPilotRefusal("REFUSED: a TRAIN-only cost pilot does not open the test split")
                return _get_data(flag)
            exp._get_data = guarded

            t = time.time()
            train_data, train_loader = exp._get_data(flag="train")
            train_loader_seconds = time.time() - t
            _stage(stages, "after_train_loader", torch)
            t = time.time()
            vali_data, vali_loader = exp._get_data(flag="val")
            vali_loader_seconds = time.time() - t
            _stage(stages, "after_validation_loader", torch)

            optim, criterion = exp._select_optimizer(), exp._select_criterion()
            _stage(stages, "after_optimizer_build", torch)
            slots_before_first_step = _optimizer_slots(optim, torch)

            f_dim = -1 if args.features == "MS" else 0
            pred_len = int(args.pred_len)
            updates = {"n": 0}

            def one_train_step(batch_x, batch_y):
                """The author's own training body, verbatim in effect (exp_long_term_forecasting.train)."""
                t0 = time.time()
                optim.zero_grad()
                bx = batch_x.float().to(exp.device)
                by = batch_y.float().to(exp.device)
                outputs, moe_loss = exp.model(bx, exp.masks, is_training=True)
                outputs = outputs[:, -pred_len:, f_dim:]
                by = by[:, -pred_len:, f_dim:].to(exp.device)
                loss = criterion(outputs, by) + 0.05 * moe_loss
                loss.backward()
                optim.step()
                updates["n"] += 1
                if str(exp.device).startswith("cuda"):
                    torch.cuda.synchronize()
                return time.time() - t0, [int(x) for x in tuple(bx.shape)]

            exp.model.train()
            per_step, shapes, done = [], [], 0
            grad_bytes = None
            for batch_x, batch_y, _bxm, _bym in train_loader:
                dt, shp = one_train_step(batch_x, batch_y)
                per_step.append(dt); shapes.append(shp); done += 1
                if done == 1:
                    _stage(stages, "after_first_optimizer_step", torch)
                    slots_after_first_step = _optimizer_slots(optim, torch)
                    grad_bytes = int(sum(p.grad.numel() * p.grad.element_size()
                                         for p in exp.model.parameters() if p.grad is not None))
                if done == int(warmup_steps):
                    _stage(stages, "after_warmup_steps", torch)
                if done >= int(steps):
                    break
            if done < int(steps):
                raise TrainPilotRefusal(f"REFUSED: the train loader yielded {done} batches, fewer than the {steps} steps this "
                                        "pilot declared. A short pilot is not a shortened recipe, it is an unmeasured one")
            _stage(stages, "after_steady_steps", torch)

            # the largest final-batch shape: reached by materializing the split's own LAST windows, because the shape is what
            # is priced and iterating a whole epoch to arrive at it is not part of a bounded pilot
            from torch.utils.data import default_collate
            final_shapes = []
            for rows in sorted({geom["final_batch_rows_by_opened_split"]["train"],
                                geom["final_batch_rows_by_opened_split"]["vali"]}):
                idx = list(range(max(0, len(train_data) - int(rows)), len(train_data)))
                batch = default_collate([train_data[i] for i in idx])
                dt, shp = one_train_step(batch[0], batch[1])
                final_shapes.append({"rows": int(rows), "train_step_seconds": dt, "input_shape": shp,
                                     "cgroup_peak_bytes_after": X.cgroup_peak_bytes(), **_gpu_now(torch)})
                del batch
            _stage(stages, "after_final_batch_shapes", torch)

            # the checkpoint the author's EarlyStopping writes, and the reload at the end of train()
            ckpt_dir = Path(args.checkpoints) / setting
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_dir / "checkpoint.pth"
            t = time.time()
            torch.save(exp.model.state_dict(), ckpt)
            ckpt_write_seconds = time.time() - t
            ckpt_bytes = int(ckpt.stat().st_size)
            ckpt_sha = S.sha_file(ckpt)
            _stage(stages, "after_checkpoint_write", torch)
            t = time.time()
            exp.model.load_state_dict(torch.load(ckpt))
            ckpt_reload_seconds = time.time() - t
            _stage(stages, "after_checkpoint_reload", torch)

            # the validation mechanics: the author's vali() body. The LOSS VALUE is not recorded on purpose.
            t = time.time()
            vb, vali_shapes = 0, []
            exp.model.eval()
            with torch.no_grad():
                for bx, by, _a, _b in vali_loader:
                    bx = bx.float().to(exp.device)
                    by = by.float()
                    outputs, _ = exp.model(bx, exp.masks, is_training=False)
                    outputs = outputs[:, -pred_len:, f_dim:]
                    by = by[:, -pred_len:, f_dim:].to(exp.device)
                    _pred = outputs.detach().cpu()
                    _true = by.detach().cpu()
                    _ = criterion(_pred, _true)                 # computed exactly as the author does; the VALUE is discarded
                    vali_shapes.append(int(bx.shape[0]))
                    vb += 1
                    if not full_validation and validation_batches is not None and vb >= int(validation_batches):
                        break
            vali_seconds = time.time() - t
            exp.model.train()
            _stage(stages, "after_validation_pass", torch)
    finally:
        os.chdir(cwd)

    wall, cpu = time.time() - wall0, time.process_time() - cpu0
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    _stage(stages, "exit", torch)
    measured_peak = X.cgroup_peak_bytes()
    warm = per_step[:int(warmup_steps)]
    steady = per_step[int(warmup_steps):]
    slots = slots_after_first_step
    test_opened = "test" in splits_opened
    verdict = pilot_verdict(measured_peak_bytes=measured_peak, cap_bytes=cap, optimizer_slot_bytes=slots["bytes"],
                            stages_present=tuple(stages), test_split_opened=test_opened)

    record = {
        "schema": PILOT_SCHEMA, "kind": "TRAIN_ONLY_COST_PILOT",
        "supersedes": SUPERSEDES,
        "reading": "an optimizer-step cost and a TRAINING memory footprint, measured in one cell scope with the optimizer's "
                   "slots instantiated. NO test window was opened, NO metric is stored, NO validation loss value is recorded "
                   "and nothing here selects anything. A cost is not a result",
        "gate": "no Traffic campaign is authorized by this record. It prices one; it does not run one",
        "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
        "dataset": design["dataset"], "cell_id_geometry_from": cell["cell_id"], "seed": int(cell["seed"]),
        "seq_len": int(design["seq_len"]), "clock": horizon_clock(design, horizon),
        "recipe_unchanged": {"batch_size": int(args.batch_size), "dtype": "float32", "optimizer": "Adam (the author's own "
                             "_select_optimizer)", "criterion": "MSELoss + 0.05 * moe_loss (the author's own train body)",
                             "learning_rate": float(args.learning_rate), "lradj": str(args.lradj),
                             "d_model": int(args.d_model), "e_layers": int(args.e_layers), "patch_len": int(args.patch_len),
                             "enc_in": int(args.enc_in), "train_epochs_sealed": int(args.train_epochs),
                             "patience_sealed": int(args.patience), "num_workers": int(args.num_workers),
                             "what_was_bounded": f"the NUMBER OF STEPS ({steps}), and nothing else"},
        "model": {"n_parameters": n_params, "parameter_bytes": param_bytes, "gradient_bytes_after_first_backward": grad_bytes,
                  "attention_mask_bytes": mask_bytes,
                  "optimizer_slots_before_first_step": slots_before_first_step,
                  "optimizer_slots_after_first_step": slots,
                  "reading": "parameters, gradients and optimizer slots are three separate resident terms. The bounded "
                             "evaluator carried only the first"},
        "geometry": geom,
        "splits_opened": sorted(set(splits_opened)),
        "test_split_opened": test_opened,
        "steps": {"timed": done, "warmup": int(warmup_steps), "steady": len(steady),
                  "warmup_seconds": warm, "steady_seconds_median": float(np.median(steady)),
                  "steady_seconds_mean": float(np.mean(steady)), "steady_seconds_p90": float(np.percentile(steady, 90)),
                  "steady_seconds_min": float(np.min(steady)), "steady_seconds_max": float(np.max(steady)),
                  "first_step_seconds": float(per_step[0]), "input_shapes": sorted({tuple(s) for s in shapes}),
                  "reading": "the warmup steps pay for allocator growth and kernel autotuning; the steady median is the "
                             "per-step cost and the warmup is reported beside it rather than averaged into it"},
        "final_batch_shapes": final_shapes,
        "final_batch_reading": "the author's data_provider sets drop_last=False on every flag, so every split ends on a "
                               "partial batch the steady loop never sees. The shapes are materialized from the train split's "
                               "own last windows: the SHAPE is what is priced, not the composition",
        "checkpoint": {"bytes": ckpt_bytes, "sha256": ckpt_sha, "write_seconds": ckpt_write_seconds,
                       "reload_seconds": ckpt_reload_seconds,
                       "rule": "the author's EarlyStopping.save_checkpoint writes model.state_dict() and train() reloads it "
                               "with load_state_dict at the end. Both were executed here, on the weights this pilot's own "
                               "optimizer updates produced --- NOT untrained weights, and NOT a converged model either",
                       "optimizer_updates_in_these_weights": updates["n"],
                       "weights_class": f"{updates['n']} optimizer updates. Trained, in the sense that every weight has been "
                                        "moved by the optimizer with its slots live; NOT converged and NOT selected on "
                                        "anything. Its error is not a result"},
        "validation": {"batches": vb, "seconds": vali_seconds,
                       "seconds_per_batch": (vali_seconds / vb) if vb else None,
                       "complete_pass": bool(full_validation and vb == geom["splits"]["vali"]["batches"]),
                       "batch_rows": sorted(set(vali_shapes)),
                       "loss_value_recorded": False,
                       "reading": "the author's vali() body over the validation loader, timed. The loss VALUE is deliberately "
                                  "not recorded: this is a cost pilot and a validation loss is a selection signal"},
        "measured": {"whole_cgroup_peak_bytes_in_child": measured_peak,
                     "whole_cgroup_peak_gib": (measured_peak / float(1 << 30)) if measured_peak else None,
                     "cgroup_stage_peaks": stages, "cgroup": S._cgroup_memory(),
                     "peak_rss_bytes_self": int(ru1.ru_maxrss) * 1024,
                     "user_cpu_seconds": ru1.ru_utime - ru0.ru_utime, "system_cpu_seconds": ru1.ru_stime - ru0.ru_stime,
                     "model_build_seconds": model_build_seconds, "train_loader_build_seconds": train_loader_seconds,
                     "validation_loader_build_seconds": vali_loader_seconds,
                     "wall_seconds": wall, "cpu_seconds": cpu,
                     "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None),
                     "peak_gpu_reserved_bytes": (int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None),
                     "host_memavailable_bytes_at_exit": X.meminfo_available_bytes()},
        "peak_scope": "the kernel's own memory.peak for this job's whole transient scope, read from INSIDE the child. It is a "
                      "high-water mark maintained by the kernel, not a sample, so it does not undershoot a short child. The "
                      "launcher's periodic sampler (5 s) is a DIFFERENT number and IS a floor for a child shorter than its "
                      "interval; both are recorded where available and they are never interchanged",
        "launcher_sampler_seconds": 5,
        "child_shorter_than_sampler_interval": bool(wall < 5),
        "declared_cap_bytes": cap,
        "declared_cap_source": ("CRISPDM_RESERVATION_BYTES" if os.environ.get("CRISPDM_RESERVATION_BYTES")
                                else "the cgroup's own memory.max (the launcher did not export the reservation)"),
        "headroom_bytes": ((cap - measured_peak) if (cap and measured_peak) else None),
        "headroom_gib": (((cap - measured_peak) / float(1 << 30)) if (cap and measured_peak) else None),
        "verdict": verdict, "verdict_reading": PILOT_VERDICTS[verdict],
        "not_a_maximum": "one execution, one host, one step budget. A measured peak is what THIS run reached, never a proof "
                         "of the maximum the path can take. It was not replayed in a fresh process",
        "device": str(exp.device), "device_uuid_measured_inside_child": S.actual_device_uuid(gpu),
        "device_assertion": device_check, "gpu_state_after": S.gpu_state(),
        "environment": S.environment(), "host_thermals_c": S.host_thermals(),
        "author_clone": S.author_git(), "source_drift": drift, "file_sha256": got,
        "host": socket.gethostname(),
        "boot_id": (Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                    if Path("/proc/sys/kernel/random/boot_id").is_file() else None),
        "pid": os.getpid(), "started_at": started, "finished_at": S.now_iso(),
        "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                       "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")},
    }
    record["record_sha256"] = S.sha_obj(record)
    return record


# --- parity of the sealed bounded reduction on TRAINED outputs ---------------------------------------------------------------

PARITY_SCHEMA = "df_tsl_trained_reducer_parity.v1"

PARITY_VERDICTS = {
    "BIT_EQUAL_ON_TRAINED_OUTPUTS": "the author's own utils.metrics.metric and the sealed bounded route returned the same "
                                    "float32 bits for MAE and MSE over the complete population of a TRAINED model's outputs",
    "REFUSED_PARITY_BROKEN": "the two reductions disagreed. Nothing is admitted on that",
    "REFUSED_FOREIGN_WEIGHTS": "the weights scored here are not the weights the cited pilot trained",
    "REFUSED_DEGENERATE_OUTPUTS": "the predictions are constant, so an agreement between two reductions of them would prove "
                                  "nothing about either. This is exactly the hole an untrained model leaves",
    "UNDETERMINED": "the author's own function could not be executed within the declared budget, so there was nothing to "
                    "compare the bounded route against here",
}


def trained_reducer_parity(design: dict, characterization: dict, *, data_path: Path, work: Path, horizon: int,
                           checkpoint: Path, pilot: dict, gpu: int = 0, require_gpu_uuid: str | None = None,
                           dataloader_workers: int | None = 0, author_metric_budget_bytes: int | None = None,
                           retain_arrays: bool = False) -> dict:
    """The gap `SATOSHI_TRAFFIC_BOUNDED_EVALUATOR_2026_09_29.md` left open: its reducer parity was proved on an UNTRAINED
    model's outputs, and an untrained model can emit a near-constant field on which two reductions agree for the wrong reason.

    This re-proves it on the outputs of the weights the TRAIN-only pilot's own optimizer produced, over the author's own
    VALIDATION loader --- the complete validation population, at the sealed geometry. The TEST split is not opened: parity of a
    reduction is a property of the arithmetic and the array, not of which split the array came from, and a cost pilot has no
    business reading held-out data. Both sides are somebody else's code: the author's own `utils.metrics.metric` and the sealed
    `df_sota_repro.author_metric_exact`. This module defines no reduction.

    The metric VALUES are recorded only as the two sides of a bitwise comparison. A 32-update model's error is not a result and
    nothing here selects on it.
    """
    from numpy.lib.format import open_memmap
    R.validate_design(design)
    if characterization.get("design_sha256") != design["design_sha256"]:
        raise TrainPilotRefusal("REFUSED: this characterization was not produced under this design")
    refuse_foreign_pilot(design, pilot)
    facts = R.dataset_facts(design["dataset"])
    got = S.sha_file(Path(data_path))
    if got != facts["sha256"]:
        raise TrainPilotRefusal(f"REFUSED: this child would read bytes that are not the registered resource ({got[:12]})")
    drift = S.code_drift(design)
    if drift:
        raise TrainPilotRefusal(f"REFUSED: the author files have drifted from the sealed digests: {sorted(drift)}")
    ckpt_src = Path(checkpoint)
    ckpt_sha = S.sha_file(ckpt_src)
    if ckpt_sha != pilot["checkpoint"]["sha256"]:
        raise TrainPilotRefusal("REFUSED: the checkpoint's sha256 is not the one the cited pilot record wrote. The weights "
                                "scored here must be the weights that pilot trained")
    if int(pilot["clock"]["horizon_steps"]) != int(horizon):
        raise TrainPilotRefusal("REFUSED: that pilot is a different horizon's; its weights have a different head width")

    cell = next(c for c in design["cells"] if c["horizon_steps"] == int(horizon) and c["seed"] == int(pilot["seed"]))
    geom = split_geometry(design, characterization, horizon)
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cap = X.declared_cap_bytes()
    S.author_env()
    import torch
    device_check = S.assert_child_device(require_gpu_uuid, gpu)
    S.fix_seeds(cell["seed"])
    args = S.build_args(cell["argv"], data_dir=Path(data_path).parent, data_name=Path(data_path).name,
                        checkpoints=work / "checkpoints", gpu=gpu, dataloader_workers=dataloader_workers)
    exp_module = importlib.import_module("exp.exp_long_term_forecasting")
    started = S.now_iso()
    stages: dict = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _stage(stages, "entry", torch)
    wall0 = time.time()
    splits_opened: list = []
    cwd = os.getcwd()
    os.chdir(work)
    try:
        with open(work / "parity_author_stdout.log", "a") as fh, contextlib.redirect_stdout(fh):
            exp = exp_module.Exp_Long_Term_Forecast(args)
            setting = S.setting_of(args)
            _stage(stages, "after_model_build", torch)
            # the author's own reload rule, from ONE file on disk, exactly as train() ends and test(test=1) begins
            ckpt_dir = Path(args.checkpoints) / setting
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = ckpt_dir / "checkpoint.pth"
            ckpt.write_bytes(ckpt_src.read_bytes())
            if S.sha_file(ckpt) != ckpt_sha:
                raise TrainPilotRefusal("REFUSED: the checkpoint changed in transport to the work directory")
            exp.model.load_state_dict(torch.load(ckpt, map_location=exp.device))
            _stage(stages, "after_checkpoint_reload", torch)

            _get_data = exp._get_data

            def guarded(flag):
                splits_opened.append(str(flag))
                if str(flag) == "test":
                    raise TrainPilotRefusal("REFUSED: this child does not open the test split either")
                return _get_data(flag)
            exp._get_data = guarded
            vali_data, vali_loader = exp._get_data(flag="val")
            n_windows = len(vali_data)
            if n_windows != geom["splits"]["vali"]["windows"]:
                raise TrainPilotRefusal(f"REFUSED: the validation loader declares {n_windows} windows, the sealed "
                                        f"characterization {geom['splits']['vali']['windows']}")
            _stage(stages, "after_validation_loader", torch)

            preds_path, trues_path = work / "vali_preds.npy", work / "vali_trues.npy"
            f_dim = -1 if args.features == "MS" else 0
            pred_len = int(args.pred_len)
            mp = mt = None
            n_seen, batches = 0, []
            t0 = time.time()
            exp.model.eval()
            with torch.no_grad():
                for bx, by, _a, _b in vali_loader:
                    bx = bx.float().to(exp.device)
                    by = by.float().to(exp.device)
                    outputs, _ = exp.model(bx, exp.masks, is_training=False)
                    outputs = outputs[:, -pred_len:, :]
                    by = by[:, -pred_len:, :].to(exp.device)
                    o = outputs.detach().cpu().numpy()[:, :, f_dim:]
                    y = by.detach().cpu().numpy()[:, :, f_dim:]
                    if mp is None:
                        shape = (n_windows, o.shape[1], o.shape[2])
                        mp = open_memmap(preds_path, mode="w+", dtype=np.float32, shape=shape)
                        mt = open_memmap(trues_path, mode="w+", dtype=np.float32, shape=shape)
                    b = o.shape[0]
                    mp[n_seen:n_seen + b] = o.astype(np.float32, copy=False)
                    mt[n_seen:n_seen + b] = y.astype(np.float32, copy=False)
                    batches.append(b); n_seen += b
            forward_seconds = time.time() - t0
            if mp is not None:
                mp.flush(); mt.flush()
            del mp, mt
            if n_seen != n_windows:
                raise TrainPilotRefusal(f"REFUSED: {n_seen} windows written against {n_windows} declared: INCOMPLETE POPULATION")
            _stage(stages, "after_forward_pass", torch)

            preds = np.load(preds_path, mmap_mode="r")
            trues = np.load(trues_path, mmap_mode="r")
            shape = [int(x) for x in preds.shape]
            array_bytes = int(preds.nbytes)
            # non-degeneracy: two reductions of a constant field agree for a reason that has nothing to do with either
            sample = np.asarray(preds[: min(64, shape[0])], dtype=np.float32)
            nondegenerate = {"sampled_windows": int(sample.shape[0]),
                             "prediction_std_float64": float(np.std(sample, dtype=np.float64)),
                             "prediction_min": float(sample.min()), "prediction_max": float(sample.max()),
                             "distinct_values_in_sample": int(np.unique(sample).size),
                             "reading": "a constant prediction field would make a reduction comparison vacuous. These are the "
                                        "spread statistics of the TRAINED model's own outputs, reported so the comparison "
                                        "below can be read as meaning something"}
            degenerate = not (nondegenerate["prediction_std_float64"] > 0.0
                              and nondegenerate["distinct_values_in_sample"] > 1)

            exact = S.author_metric_exact(S.StoredArray(preds_path), S.StoredArray(trues_path))
            _stage(stages, "after_bounded_route", torch)
            need = 3 * array_bytes
            author_side = None
            if author_metric_budget_bytes is None or need <= int(author_metric_budget_bytes):
                MET = importlib.import_module("utils.metrics")
                mae, mse, _rmse, _mape, _mspe = MET.metric(np.asarray(preds), np.asarray(trues))
                author_side = {"mae": float(mae), "mse": float(mse)}
            _stage(stages, "after_author_function", torch)
            pred_sha = S.sha_array(preds)
            true_sha = S.sha_array(trues)
            f64 = S.float64_metrics_files(preds_path, trues_path, tuple(shape))
            _stage(stages, "after_digests", torch)
    finally:
        os.chdir(cwd)
    wall = time.time() - wall0
    _stage(stages, "exit", torch)
    measured_peak = X.cgroup_peak_bytes()

    if degenerate:
        verdict = "REFUSED_DEGENERATE_OUTPUTS"
        bit_equal = None
    elif author_side is None:
        verdict = "UNDETERMINED"
        bit_equal = None
    else:
        bit_equal = bool(author_side["mae"] == exact["mae"] and author_side["mse"] == exact["mse"])
        verdict = "BIT_EQUAL_ON_TRAINED_OUTPUTS" if bit_equal else "REFUSED_PARITY_BROKEN"

    retention = {"arrays_deleted": not retain_arrays, "deleted": [], "retained": [],
                 "rule": "the arrays are deleted once the metrics, the complete element count and the sha256 digests that "
                         "identify the population are recorded. What is retained is the record, not the arrays"}
    for p in (preds_path, trues_path):
        if retain_arrays:
            retention["retained"].append({"path": str(p), "bytes": p.stat().st_size if p.is_file() else None})
        else:
            with contextlib.suppress(OSError):
                p.unlink()
            retention["deleted"].append(str(p))

    record = {
        "schema": PARITY_SCHEMA, "kind": "REDUCER_PARITY_ON_TRAINED_OUTPUTS",
        "reading": "the author's own utils.metrics.metric beside the sealed bounded route, on the complete validation "
                   "population of a TRAINED model's outputs. The metric values are the two sides of a bitwise comparison and "
                   "are NOT a result: the weights carry a handful of optimizer updates and were selected on nothing",
        "what_this_closes": "the parity in the bounded-evaluator delivery was demonstrated on an UNTRAINED model's outputs. "
                            "An untrained field can be near-degenerate, and two reductions of a near-constant array can agree "
                            "for a reason that has nothing to do with either reduction. This is the same equality on outputs "
                            "the optimizer actually moved",
        "what_this_does_not_close": "the bounded WRITER's row placement and the author's own unchunked test() over the TEST "
                                    "population. Neither is measured here: this child does not open the test split, and the "
                                    "native-witness child that would needs a cap the slice's aggregate budget does not admit",
        "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
        "dataset": design["dataset"], "clock": horizon_clock(design, horizon),
        "cell_id_geometry_from": cell["cell_id"], "seed": int(cell["seed"]),
        "weights": {"source": "the TRAIN-only pilot's own checkpoint", "sha256": ckpt_sha,
                    "pilot_record_sha256": pilot["record_sha256"],
                    "optimizer_updates": pilot["checkpoint"].get("optimizer_updates_in_these_weights"),
                    "trained": True, "converged": False, "selected_on": None,
                    "reload_rule": "written to the author's own checkpoint path and reloaded with load_state_dict, exactly as "
                                   "train() ends and test(test=1) begins"},
        "population": {"split": "validation", "windows": n_seen, "shape": shape, "elements": int(np.prod(shape)),
                       "complete": n_seen == geom["splits"]["vali"]["windows"],
                       "batches": len(batches), "batch_sizes": sorted(set(batches)),
                       "preds_sha256": pred_sha, "trues_sha256": true_sha, "dtype": "float32",
                       "test_split_opened": "test" in splits_opened, "splits_opened": sorted(set(splits_opened))},
        "non_degeneracy": nondegenerate,
        "parity": {"author_function_unchunked": author_side,
                   "bounded_route": {"mae": exact["mae"], "mse": exact["mse"], "route": exact["route"],
                                     "elements": exact["elements"],
                                     "denominator_exact_in_float32": exact["denominator_exact_in_float32"],
                                     "reduction": exact["reduction"]},
                   "mae_bitwise_equal": (None if author_side is None else author_side["mae"] == exact["mae"]),
                   "mse_bitwise_equal": (None if author_side is None else author_side["mse"] == exact["mse"]),
                   "mae_absolute_difference": (None if author_side is None else abs(author_side["mae"] - exact["mae"])),
                   "mse_absolute_difference": (None if author_side is None else abs(author_side["mse"] - exact["mse"])),
                   "bit_equal": bit_equal,
                   "author_function_budget_bytes": author_metric_budget_bytes,
                   "author_function_temporaries_bytes": need,
                   "both_sides_are_other_peoples_code": "utils.metrics.metric is the author's; author_metric_exact is the "
                                                        "sealed df_sota_repro route. This module reduces nothing"},
        "independent_float64_beside_the_float32_route": f64,
        "measured": {"whole_cgroup_peak_bytes_in_child": measured_peak,
                     "whole_cgroup_peak_gib": (measured_peak / float(1 << 30)) if measured_peak else None,
                     "cgroup_stage_peaks": stages, "cgroup": S._cgroup_memory(),
                     "forward_pass_seconds": forward_seconds, "wall_seconds": wall,
                     "array_bytes_each": array_bytes,
                     "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None),
                     "peak_gpu_reserved_bytes": (int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None)},
        "declared_cap_bytes": cap,
        "headroom_bytes": ((cap - measured_peak) if (cap and measured_peak) else None),
        "retention": retention,
        "verdict": verdict, "verdict_reading": PARITY_VERDICTS[verdict],
        "device": str(exp.device), "device_uuid_measured_inside_child": S.actual_device_uuid(gpu),
        "device_assertion": device_check, "environment": S.environment(),
        "author_clone": S.author_git(), "source_drift": drift, "file_sha256": got,
        "host": socket.gethostname(), "pid": os.getpid(), "started_at": started, "finished_at": S.now_iso(),
        "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                       "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")},
    }
    record["record_sha256"] = S.sha_obj(record)
    return record


# --- the projection, with explicit headroom ---------------------------------------------------------------------------------

def early_stopping_state(design: dict) -> dict:
    """Whether the sealed recipe can stop early at the PINNED author commit. It matters to every projection: if it cannot, the
    epoch budget is the actual count and not an upper bound, and the checkpoint is written once per epoch."""
    src = (S.AUTHOR_REPO / "utils/tools.py")
    text = src.read_text() if src.is_file() else ""
    body = text.split("def __call__", 1)[-1].split("def save_checkpoint", 1)[0] if "def __call__" in text else ""
    # a commented-out or triple-quoted branch is not a branch: the early-stop line inside a docstring block cannot fire
    live = re.sub(r"'''.*?'''", "", re.sub(r'"""' + r'.*?' + r'"""', "", body, flags=re.S), flags=re.S)
    live = "\n".join(ln for ln in live.splitlines() if not ln.strip().startswith("#"))
    active = "self.early_stop = True" in live
    return {"early_stopping_can_fire": bool(active),
            "evidence": "utils/tools.py EarlyStopping.__call__ at the pinned commit calls save_checkpoint unconditionally and "
                        "its scoring/counter body is commented out, so early_stop is never set",
            "consequence": ("the epoch budget is an upper bound" if active else
                            "the sealed train_epochs is the ACTUAL epoch count, not an upper bound, and the checkpoint is "
                            "written once per epoch"),
            "checkpoint_writes_per_cell": None if active else "one per epoch"}


def projection(design: dict, characterization: dict, pilots: list, *, headroom_fraction: float = 0.25) -> dict:
    """Price the sealed design from MEASURED pilots. Every term says whether it was measured or derived, and the memory line is
    a projection WITH EXPLICIT HEADROOM --- never a claim that a maximum has been proven."""
    if not pilots:
        raise TrainPilotRefusal("REFUSED: nothing is projected from no measurement. A missing pilot is unknown, never zero")
    for p in pilots:
        refuse_foreign_pilot(design, p)
    by_h = {int(p["clock"]["horizon_steps"]): p for p in pilots}
    es = early_stopping_state(design)
    cells, unmeasured = [], []
    for c in design["cells"]:
        h = int(c["horizon_steps"])
        geom = split_geometry(design, characterization, h)
        src_h = h if h in by_h else min(by_h, key=lambda k: abs(k - h))
        p = by_h[src_h]
        measured_here = h in by_h
        if not measured_here:
            unmeasured.append(h)
        step_s = float(p["steps"]["steady_seconds_median"])
        val_per_batch = p["validation"]["seconds_per_batch"]
        epochs = int(c["effective_args"]["train_epochs"])
        tb = geom["splits"]["train"]["batches"]
        vb = geom["splits"]["vali"]["batches"]
        sb = geom["splits"]["test"]["batches"]
        train_s = step_s * tb
        vali_s = (val_per_batch or 0.0) * vb
        logging_test_s = (val_per_batch or 0.0) * sb
        ckpt_s = float(p["checkpoint"]["write_seconds"])
        epoch_s = train_s + vali_s + logging_test_s + ckpt_s
        cells.append({
            "cell_id": c["cell_id"], "horizon_steps": h, "seed": int(c["seed"]),
            "clock": horizon_clock(design, h),
            "priced_from_horizon": src_h,
            "per_step_seconds_class": "MEASURED at this horizon" if measured_here else
                                      f"DERIVED: the h{src_h} pilot's measured step cost applied to this horizon's batch count",
            "train_batches": tb, "vali_batches": vb, "test_batches_counts_only": sb,
            "epochs": epochs, "epoch_seconds": epoch_s,
            "train_pass_seconds": train_s, "validation_pass_seconds": vali_s,
            "logging_test_pass_seconds_DERIVED": logging_test_s,
            "checkpoint_write_seconds": ckpt_s,
            "cell_seconds": epoch_s * epochs,
            "cell_hours": epoch_s * epochs / 3600.0})
    total_s = sum(c["cell_seconds"] for c in cells)

    # a bracket, not a point. An unmeasured horizon priced at the FASTEST measured rate is a floor and at the SLOWEST a
    # ceiling; the true value lies between them only if the per-step cost is monotone in the prediction length, which the two
    # measured points are consistent with and which is stated here as an ASSUMPTION rather than smuggled into the number.
    def _rates(p):
        return float(p["steps"]["steady_seconds_median"]), float(p["validation"]["seconds_per_batch"] or 0.0)
    fastest = min(pilots, key=lambda p: _rates(p)[0])
    slowest = max(pilots, key=lambda p: _rates(p)[0])

    def _total(pick) -> float:
        s = 0.0
        for c in design["cells"]:
            h = int(c["horizon_steps"])
            g = split_geometry(design, characterization, h)
            p = by_h[h] if h in by_h else pick
            step_s, vpb = _rates(p)
            epoch = (step_s * g["splits"]["train"]["batches"]
                     + vpb * (g["splits"]["vali"]["batches"] + g["splits"]["test"]["batches"])
                     + float(p["checkpoint"]["write_seconds"]))
            s += epoch * int(c["effective_args"]["train_epochs"])
        return s
    lo, hi = _total(fastest), _total(slowest)
    bracket = {"hours_lower": min(lo, hi) / 3600.0, "hours_upper": max(lo, hi) / 3600.0,
               "lower_prices_unmeasured_horizons_at_horizon": int(fastest["clock"]["horizon_steps"]),
               "upper_prices_unmeasured_horizons_at_horizon": int(slowest["clock"]["horizon_steps"]),
               "assumption": "the per-step cost is monotone in the prediction length. The two measured points are consistent "
                             "with that and nothing here proves it; the unmeasured horizons are unmeasured",
               "class": "DERIVED BRACKET over MEASURED endpoints"}
    peaks = {int(p["clock"]["horizon_steps"]): p["measured"]["whole_cgroup_peak_bytes_in_child"] for p in pilots}
    measured_peaks = {k: v for k, v in peaks.items() if v}
    worst = max(measured_peaks.values()) if measured_peaks else None
    out = {
        "schema": PROJECTION_SCHEMA, "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
        "dataset": design["dataset"],
        "measured_from": [{"record_sha256": p["record_sha256"], "horizon_steps": int(p["clock"]["horizon_steps"]),
                           "host_key": p.get("boot_id"), "device_uuid": p["device_uuid_measured_inside_child"],
                           "steps_timed": p["steps"]["timed"], "steady_steps": p["steps"]["steady"],
                           "steady_seconds_median": p["steps"]["steady_seconds_median"],
                           "whole_cgroup_peak_bytes": p["measured"]["whole_cgroup_peak_bytes_in_child"],
                           "declared_cap_bytes": p["declared_cap_bytes"], "verdict": p["verdict"]} for p in pilots],
        "early_stopping": es,
        "cells": cells, "cells_total": len(cells),
        "seconds_total": total_s, "hours_total": total_s / 3600.0,
        "hours_total_class": "a point priced with each unmeasured horizon taken at the NEAREST measured horizon's rate. It is "
                             "not a best estimate; the reportable quantity is the bracket below",
        "bracket": bracket,
        "horizons_measured": sorted(measured_peaks), "horizons_priced_by_interpolation": sorted(set(unmeasured)),
        "time_reading": ("every cell runs its full epoch budget: early stopping cannot fire at the pinned commit, so this is "
                         "not an upper bound that early stopping shortens --- it is the schedule"
                         if not es["early_stopping_can_fire"] else
                         "an upper bound: early stopping can only shorten it"),
        "memory": {
            "measured_peak_bytes_by_horizon": peaks,
            "worst_measured_peak_bytes": worst,
            "worst_measured_peak_gib": (worst / float(1 << 30)) if worst else None,
            "headroom_fraction": float(headroom_fraction),
            "projected_cap_bytes": (int(math.ceil(worst * (1.0 + float(headroom_fraction)))) if worst else None),
            "projected_cap_gib": ((worst * (1.0 + float(headroom_fraction))) / float(1 << 30)) if worst else None,
            "class": "PROJECTION WITH EXPLICIT HEADROOM",
            "reading": "the worst peak MEASURED in this pilot set, multiplied by an explicitly stated headroom fraction. It is "
                       "NOT a proof that the training path cannot exceed it: the pilot ran a bounded number of steps on one "
                       "host in one execution, and a longer run can reach fragmentation states a short one does not",
            "not_measured": ["the horizons no pilot covered", "any state reachable only after many epochs",
                             "the author's own per-epoch logging test pass, which this pilot does not run"]},
        "unpriced_terms": [
            "the author's per-epoch logging test pass over the TEST loader: its batch count is known from the sealed "
            "characterization, so its TIME is derived from the measured validation rate, but its own memory was not measured "
            "here and the pilot did not open that split",
            "the evaluation at the end of the cell, which the bounded evaluator prices separately"],
    }
    out["projection_sha256"] = S.sha_obj(out)
    return out


# --- CLI --------------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pilot", help="the TRAIN-only cost pilot, one horizon, one cell scope")
    p.add_argument("--design", required=True)
    p.add_argument("--characterization", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--require-gpu-uuid", default=os.environ.get(S.REQUIRED_GPU_ENV))
    p.add_argument("--dataloader-workers", type=int, default=None)
    p.add_argument("--validation-batches", type=int, default=None)
    p.add_argument("--out", required=True)

    g = sub.add_parser("geometry", help="batch geometry and the clock; reads no bytes")
    g.add_argument("--design", required=True)
    g.add_argument("--characterization", required=True)
    g.add_argument("--horizon", type=int, required=True)
    g.add_argument("--out", default=None)

    q = sub.add_parser("trained-parity", help="the sealed bounded reduction against the author's own function, on TRAINED "
                                              "outputs over the validation population. The test split is not opened")
    q.add_argument("--design", required=True)
    q.add_argument("--characterization", required=True)
    q.add_argument("--data-path", required=True)
    q.add_argument("--work", required=True)
    q.add_argument("--horizon", type=int, required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--pilot", required=True)
    q.add_argument("--gpu", type=int, default=0)
    q.add_argument("--require-gpu-uuid", default=os.environ.get(S.REQUIRED_GPU_ENV))
    q.add_argument("--dataloader-workers", type=int, default=0)
    q.add_argument("--author-metric-budget-gib", type=float, default=None)
    q.add_argument("--retain-arrays", action="store_true")
    q.add_argument("--out", required=True)

    j = sub.add_parser("project", help="price the sealed design from measured pilots, with explicit headroom")
    j.add_argument("--design", required=True)
    j.add_argument("--characterization", required=True)
    j.add_argument("--pilot", action="append", required=True, help="a TRAIN_PILOT record; repeatable")
    j.add_argument("--headroom-fraction", type=float, default=0.25)
    j.add_argument("--out", required=True)

    a = ap.parse_args(argv)
    design = json.loads(Path(a.design).read_text())
    chars = json.loads(Path(a.characterization).read_text())

    if a.command == "geometry":
        out = {"geometry": split_geometry(design, chars, a.horizon), "clock": horizon_clock(design, a.horizon),
               "early_stopping": early_stopping_state(design)}
        if a.out:
            _write(Path(a.out), out)
        print(json.dumps(out, indent=1, default=str))
        return 0

    if a.command == "pilot":
        rec = train_footprint_pilot(design, chars, data_path=Path(a.data_path), work=Path(a.work), horizon=a.horizon,
                                    seed=a.seed, steps=a.steps, warmup_steps=a.warmup_steps, gpu=a.gpu,
                                    require_gpu_uuid=a.require_gpu_uuid, dataloader_workers=a.dataloader_workers,
                                    full_validation=a.validation_batches is None,
                                    validation_batches=a.validation_batches)
        path = _write(Path(a.out), rec)
        print(json.dumps({"path": str(path), "verdict": rec["verdict"],
                          "whole_cgroup_peak_bytes": rec["measured"]["whole_cgroup_peak_bytes_in_child"],
                          "whole_cgroup_peak_gib": rec["measured"]["whole_cgroup_peak_gib"],
                          "declared_cap_bytes": rec["declared_cap_bytes"], "headroom_gib": rec["headroom_gib"],
                          "steady_seconds_median": rec["steps"]["steady_seconds_median"],
                          "optimizer_slot_bytes": rec["model"]["optimizer_slots_after_first_step"]["bytes"],
                          "peak_gpu_reserved_bytes": rec["measured"]["peak_gpu_reserved_bytes"],
                          "validation_batches": rec["validation"]["batches"],
                          "checkpoint_bytes": rec["checkpoint"]["bytes"],
                          "device_uuid": rec["device_uuid_measured_inside_child"],
                          "record_sha256": rec["record_sha256"]}, indent=1))
        return 0 if rec["verdict"] == "MEASURED_WITHIN_CAP" else 3

    if a.command == "trained-parity":
        budget = None if a.author_metric_budget_gib is None else int(a.author_metric_budget_gib * (1 << 30))
        rec = trained_reducer_parity(design, chars, data_path=Path(a.data_path), work=Path(a.work), horizon=a.horizon,
                                     checkpoint=Path(a.checkpoint), pilot=json.loads(Path(a.pilot).read_text()),
                                     gpu=a.gpu, require_gpu_uuid=a.require_gpu_uuid,
                                     dataloader_workers=a.dataloader_workers, author_metric_budget_bytes=budget,
                                     retain_arrays=a.retain_arrays)
        path = _write(Path(a.out), rec)
        print(json.dumps({"path": str(path), "verdict": rec["verdict"],
                          "population": rec["population"]["shape"], "elements": rec["population"]["elements"],
                          "complete": rec["population"]["complete"],
                          "test_split_opened": rec["population"]["test_split_opened"],
                          "author_function": rec["parity"]["author_function_unchunked"],
                          "bounded_route": {k: rec["parity"]["bounded_route"][k] for k in ("mae", "mse")},
                          "bit_equal": rec["parity"]["bit_equal"],
                          "prediction_std": rec["non_degeneracy"]["prediction_std_float64"],
                          "whole_cgroup_peak_gib": rec["measured"]["whole_cgroup_peak_gib"],
                          "declared_cap_bytes": rec["declared_cap_bytes"],
                          "optimizer_updates_in_weights": rec["weights"]["optimizer_updates"],
                          "record_sha256": rec["record_sha256"]}, indent=1))
        return 0 if rec["verdict"] == "BIT_EQUAL_ON_TRAINED_OUTPUTS" else 3

    if a.command == "project":
        pilots = [json.loads(Path(x).read_text()) for x in a.pilot]
        out = projection(design, chars, pilots, headroom_fraction=a.headroom_fraction)
        path = _write(Path(a.out), out)
        print(json.dumps({"path": str(path), "hours_total": out["hours_total"], "cells": out["cells_total"],
                          "worst_measured_peak_gib": out["memory"]["worst_measured_peak_gib"],
                          "projected_cap_gib": out["memory"]["projected_cap_gib"],
                          "early_stopping_can_fire": out["early_stopping"]["early_stopping_can_fire"]}, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
