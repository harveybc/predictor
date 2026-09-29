#!/usr/bin/env python3
"""RB02 AUDIT: the independently invoked evaluation of the twelve retained Weather checkpoints, and the identity
reconciliation that has to agree before it is allowed to run.

Nothing here trains. Nothing here scores a new recipe, invents a margin, widens a tolerance or opens a governed unit.
This module has exactly two jobs and refuses everything else:

  identity   Re-DERIVE the five identities the execution return asserts, rather than reading them from its claim:
             the RECIPE (the design and protocol digests recomputed from the retained design object and from the
             author clone's own files), the SELECTED WEIGHTS (the sha256 of the checkpoint on disk, and the
             checkpoint-selection rule re-parsed from the retained author log rather than from the record's own
             summary), the TEST POPULATION (the delivered bytes' sha256 and the target population digest re-derived
             by the author's own loader), the SCORER (the sha256 of `utils.metrics`' source as the process that will
             score sees it) and the NAIVE (recomputed on the loader's own rows, not asserted).

  replay     One cell: a FRESH process reloads the retained checkpoint through the author's own `test(setting, 1)`
             path — unchunked, no adapter, no downcast, the same population — and the result is compared with the
             retained record under the EXISTING frozen rule `df_sota_repro.AGREEMENT["replay"]`, which is read here
             and never modified. The stronger BITWISE question ("is the replayed prediction array the same bytes?")
             is answered separately by the sha256 of the array, so a pass under the tolerance can never be mistaken
             for bit-for-bit reproduction.

The frozen rule's declared device is `cpu`, a CROSS-device replay. A same-device replay is a STRICTER setting of the
same rule, not a widened one; every record says which device it ran on and the comparison never changes with it.

Refusals, not repairs: if the data bytes, the author files, the checkpoint digest, the design digest or the scored
population disagree with the retained record, this module stops. A replay of something that is not the audited object
is not a replay.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import df_sota_repro as S                                              # noqa: E402  the author bridge, reused verbatim
import df_tsl_repro as R                                               # noqa: E402  the sealed design and protocol digest
import df_tsl_execute as X                                             # noqa: E402  the executed lane's own definitions

IDENTITY_SCHEMA = "df_tsl_identity_reconciliation.v1"
REPLAY_SCHEMA = "df_tsl_replay_record.v1"
CLOSURE_SCHEMA = "df_tsl_replay_closure.v1"

#: read, never written: the tolerance this audit is judged under already existed before this audit did.
REPLAY_RULE_SOURCE = "df_sota_repro.AGREEMENT['replay'] (frozen before any Electricity or Weather score was read)"


class ReplayRefusal(SystemExit):
    """Nothing is reconciled, replayed or published from a state that cannot be shown."""


def replay_rule() -> dict:
    rule = dict(S.AGREEMENT["replay"])
    rule["metric_recompute"] = dict(S.AGREEMENT["metric_recompute"])
    rule["source"] = REPLAY_RULE_SOURCE
    rule["not_widened"] = ("atol/rtol are read from the module that froze them; this file contains no tolerance of its "
                           "own and no branch that relaxes one")
    return rule


# --- job 1: the five identities, re-derived --------------------------------------------------------------------------------

def recipe_identity(design: dict) -> dict:
    """The recipe digest recomputed from the retained design OBJECT, and the protocol digest recomputed from its lock —
    then the author files re-hashed from the clone, so 'the same recipe' is a statement about bytes on this disk."""
    claimed_design = design.get("design_sha256")
    # the sealer digests the design body BEFORE the protocol digest is written into the lock; both are reproduced that way
    lock_body = {k: v for k, v in design["lock"].items() if k != "protocol_sha256"}
    body = {**{k: v for k, v in design.items() if k != "design_sha256"}, "lock": lock_body}
    recomputed_design = S.sha_obj(body)
    recomputed_protocol = S.sha_obj(lock_body)
    claimed_protocol = design["lock"].get("protocol_sha256")
    S.author_env()
    on_disk = S.source_digests()
    sealed = design["lock"]["source"]["files_sha256"]
    drift = {f: {"sealed": sealed[f], "on_disk": on_disk.get(f)} for f in sealed if on_disk.get(f) != sealed[f]}
    git = S.author_git()
    return {"design_sha256_claimed": claimed_design, "design_sha256_recomputed": recomputed_design,
            "design_agrees": claimed_design == recomputed_design,
            "protocol_sha256_claimed": claimed_protocol, "protocol_sha256_recomputed": recomputed_protocol,
            "protocol_agrees": claimed_protocol == recomputed_protocol,
            "author_files_rehashed": on_disk, "author_file_drift": drift, "author_files_agree": not drift,
            "author_clone": git, "clone_at_pinned_revision": bool(git.get("pinned_matches") and git.get("clean")),
            "method": "the digest is recomputed over the design object with its own digest field removed, exactly as the "
                      "sealer computes it; the protocol digest is recomputed by df_tsl_repro.protocol_sha256 from the lock"}


def weights_identity(record: dict, checkpoint: Path, author_log: Path | None) -> dict:
    """The selected weights: the file's sha256 against the record's, and the SELECTION RULE re-derived by re-parsing the
    retained author log — the record's own `convergence` block is compared with that re-derivation, never trusted as it."""
    if not checkpoint.is_file():
        raise ReplayRefusal(f"REFUSED: no retained checkpoint at {checkpoint}")
    got = S.sha_file(checkpoint)
    claimed = record.get("checkpoint_sha256")
    out = {"checkpoint_path_basename": checkpoint.name, "sha256_recomputed": got, "sha256_claimed": claimed,
           "bytes": checkpoint.stat().st_size, "agrees": got == claimed}
    if author_log is not None and author_log.is_file():
        parsed = S.parse_author_log(author_log.read_text())
        rec_tr = record.get("training") or {}
        best = parsed.get("best_epoch_by_vali")
        out["selection_rule"] = {
            "rule": "the lowest-validation-MSE epoch's state_dict, reloaded by the author's train() before test()",
            "best_epoch_by_vali_reparsed": best,
            "best_epoch_by_vali_in_record": rec_tr.get("best_epoch_by_vali"),
            "epochs_run_reparsed": parsed.get("epochs_run"), "epochs_run_in_record": rec_tr.get("epochs_run"),
            "early_stopped_reparsed": parsed.get("early_stopped"),
            "per_epoch_vali_reparsed": [e["vali_loss"] for e in parsed.get("per_epoch", [])],
            "agrees": (best == rec_tr.get("best_epoch_by_vali")
                       and parsed.get("epochs_run") == rec_tr.get("epochs_run")
                       and parsed.get("early_stopped") == rec_tr.get("early_stopped")),
            "from": "the retained author stdout log of the ORIGINAL cell, re-parsed by df_sota_repro.parse_author_log",
            "limit": "re-parsing the log proves which epoch the rule selected; it does not re-prove that the saved bytes "
                     "are that epoch's, which only retraining could, and retraining would destroy the audited object"}
    else:
        out["selection_rule"] = {"state": "UNAVAILABLE", "why": "the original cell's author log was not retained here"}
    return out


def scorer_identity(record: dict) -> dict:
    """The function that produced the number, hashed in THIS process — not copied from the record."""
    ident = X.author_scorer_identity()
    claimed = ((record.get("metric") or {}).get("scorer") or {}).get("sha256")
    return {**ident, "sha256_claimed": claimed, "agrees": ident["sha256"] == claimed}


def transport_identity(record: dict, data_path: Path) -> dict:
    """The delivered bytes, re-hashed here, against BOTH the record and the sealed design's registered digest."""
    got = S.sha_file(data_path)
    tr = record.get("transport") or {}
    facts = R.dataset_facts(record["dataset"])
    return {"data_sha256_recomputed": got, "data_sha256_in_record": record.get("file_sha256"),
            "data_sha256_registered": facts["sha256"], "delivered_sha256_in_transport": tr.get("delivered_sha256"),
            "bytes": data_path.stat().st_size, "bytes_in_transport": tr.get("delivered_bytes"),
            "agrees": got == record.get("file_sha256") == facts["sha256"] == tr.get("delivered_sha256"),
            "transport_class": tr.get("class") or tr.get("kind")}


# --- job 2: the independently invoked evaluation ---------------------------------------------------------------------------

def stage_checkpoint(source: Path, work: Path, setting: str, expect_sha: str) -> dict:
    """The retained weights placed where the author's own `test(setting, 1)` looks for them — copied, verified again after
    the copy, and never modified. The author's loader takes `./checkpoints/<setting>/checkpoint.pth` under its cwd."""
    dest_dir = work / "checkpoints" / setting
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "checkpoint.pth"
    before = S.sha_file(source)
    if before != expect_sha:
        raise ReplayRefusal(f"REFUSED: the retained checkpoint hashes {before[:12]}, the record says {expect_sha[:12]}")
    dest.write_bytes(source.read_bytes())
    after = S.sha_file(dest)
    if after != expect_sha:
        raise ReplayRefusal(f"REFUSED: the staged checkpoint hashes {after[:12]} after the copy")
    return {"source_sha256": before, "staged_sha256": after, "verified_after_copy": True,
            "loaded_by": "the author's own test(setting, test=1): torch.load('./checkpoints/' + setting + '/checkpoint.pth')"}


def compare_to_record(record: dict, *, author: dict, independent: dict, pred_sha: str, true_sha: str,
                      naive: dict, shape: tuple, rule: dict) -> dict:
    """The comparison, in two independent registers that are never merged: BITWISE (the array digests and exact float
    equality) and the FROZEN TOLERANCE (the existing replay rule). A pass in one is never reported as a pass in the other."""
    ra = record["metric"]["author_float32"]
    ri = record["metric"]["independent_float64"]
    rp = record["population"]
    rn = record["naive"]
    d_mae = float(author["mae"]) - float(ra["mae"])
    d_mse = float(author["mse"]) - float(ra["mse"])
    di_mae = float(independent["mae"]) - float(ri["mae"])
    di_mse = float(independent["mse"]) - float(ri["mse"])
    dn_mae = float(naive["naive"]["mae"]) - float(rn["mae"])
    dn_mse = float(naive["naive"]["mse"]) - float(rn["mse"])
    bitwise = {
        "predictions_sha256_replayed": pred_sha, "predictions_sha256_recorded": rp.get("predictions_sha256"),
        "predictions_bitwise_equal": pred_sha == rp.get("predictions_sha256"),
        "target_population_sha256_replayed": true_sha, "target_population_sha256_recorded": rp.get("sha256"),
        "target_population_bitwise_equal": true_sha == rp.get("sha256"),
        "author_float32_mae_exactly_equal": float(author["mae"]) == float(ra["mae"]),
        "author_float32_mse_exactly_equal": float(author["mse"]) == float(ra["mse"]),
        "naive_mae_exactly_equal": float(naive["naive"]["mae"]) == float(rn["mae"]),
        "naive_mse_exactly_equal": float(naive["naive"]["mse"]) == float(rn["mse"]),
        "reading": "sha256 over the float32 array bytes and exact float equality of the reduced numbers. This is a "
                   "STRICTER question than the frozen tolerance and is reported on its own line",
    }
    bitwise["all"] = bool(bitwise["predictions_bitwise_equal"] and bitwise["target_population_bitwise_equal"]
                          and bitwise["author_float32_mae_exactly_equal"] and bitwise["author_float32_mse_exactly_equal"]
                          and bitwise["naive_mae_exactly_equal"] and bitwise["naive_mse_exactly_equal"])
    # the frozen rule, applied exactly as it is written for a REPLAY: "the metric recomputed from the replayed predictions
    # must be within 1e-5 of the stored one". The rule's other clause (`metric_recompute`) governs a recomputation from the
    # SAME arrays and is reported beside it, never folded into the replay verdict — that would silently move the criterion.
    tol = {
        "rule": rule,
        "criterion_applied": ("the frozen rule's replay criterion: the metric recomputed from the replayed predictions is "
                              "within 1e-5 of the stored one"),
        "author_float32_delta": {"mae": d_mae, "mse": d_mse},
        "author_metric_within_1e-5": abs(d_mae) <= 1e-5 and abs(d_mse) <= 1e-5,
        "stricter_same_arrays_clause": {
            "author_float32_bitwise_equal": (bitwise["author_float32_mae_exactly_equal"]
                                             and bitwise["author_float32_mse_exactly_equal"]),
            "independent_float64_delta": {"mae": di_mae, "mse": di_mse},
            "independent_float64_within_1e-6": abs(di_mae) <= 1e-6 and abs(di_mse) <= 1e-6,
            "applies_when": ("the metric is recomputed from the SAME arrays. A cross-device replay is not that case and is "
                             "not judged by it")},
        "naive_delta": {"mae": dn_mae, "mse": dn_mse},
        "naive_within_1e-5": abs(dn_mae) <= 1e-5 and abs(dn_mse) <= 1e-5,
    }
    tol["met"] = bool(tol["author_metric_within_1e-5"] and tol["naive_within_1e-5"])
    population = {
        "shape_replayed": [int(x) for x in shape],
        "windows_recorded": rp.get("windows"), "channels_recorded": rp.get("target_channels"),
        "elements_replayed": int(np.prod(shape)), "elements_recorded": rp.get("elements"),
        "agrees": int(np.prod(shape)) == int(rp.get("elements")) and int(shape[0]) == int(rp.get("windows")),
    }
    return {"bitwise": bitwise, "frozen_tolerance": tol, "population": population}


def replay_cell(design: dict, characterization: dict, record: dict, *, data_path: Path, work: Path, lane: Path,
                gpu: int = 0, require_gpu_uuid: str | None = None, device: str = "cuda",
                dataloader_workers: int | None = 0) -> dict:
    """One fresh-process evaluation of one retained checkpoint through the author's own test path. No training, no
    adapter, no chunking, no downcast, the complete sealed population."""
    R.validate_design(design)
    if record.get("design_sha256") != design["design_sha256"]:
        raise ReplayRefusal("REFUSED: this record was not produced under the design being replayed")
    if characterization.get("design_sha256") != design["design_sha256"]:
        raise ReplayRefusal("REFUSED: this characterization was not produced under this design")
    horizon, seed = int(record["horizon_steps"]), int(record["seed"])
    cell = next(c for c in design["cells"] if c["horizon_steps"] == horizon and c["seed"] == seed)
    key = f"L{design['seq_len']}_h{horizon}"
    sets = characterization["sets"][key]

    ident = {"recipe": recipe_identity(design),
             "transport": transport_identity(record, data_path),
             "scorer": scorer_identity(record)}
    if not ident["recipe"]["design_agrees"] or not ident["recipe"]["protocol_agrees"]:
        raise ReplayRefusal("REFUSED: the recipe digests do not reconcile")
    if not ident["recipe"]["author_files_agree"]:
        raise ReplayRefusal(f"REFUSED: the author files have drifted: {sorted(ident['recipe']['author_file_drift'])}")
    if not ident["transport"]["agrees"]:
        raise ReplayRefusal("REFUSED: these are not the registered bytes the cell was scored on")
    if not ident["scorer"]["agrees"]:
        raise ReplayRefusal("REFUSED: the scorer in this process is not the scorer that produced the record")

    cell_dir = lane / f"cell_{design['dataset']}_h{horizon}_s{seed}"
    source_ckpt = cell_dir / "checkpoints" / record["setting"] / "checkpoint.pth"
    author_log = cell_dir / "author_stdout.log"
    ident["weights"] = weights_identity(record, source_ckpt, author_log)
    if not ident["weights"]["agrees"]:
        raise ReplayRefusal("REFUSED: the retained checkpoint is not the one the record names")

    work.mkdir(parents=True, exist_ok=True)
    staged = stage_checkpoint(source_ckpt, work, record["setting"], record["checkpoint_sha256"])

    S.author_env()
    import functools
    import torch
    cross_device_patch = []
    if device == "cpu":
        # operational patch, declared exactly as the Electricity lane declares it: the author's test(test=1) calls
        # torch.load(path) with no map_location, so a checkpoint saved from CUDA cannot be READ at all on a CPU-only
        # replay. Placement only; no arithmetic changes and no tensor value changes.
        torch.load = functools.partial(torch.load, map_location=torch.device("cpu"))
        cross_device_patch = [{"what": "torch.load bound to map_location=cpu for the author's own checkpoint reload",
                               "why": "the author's test(test=1) passes no map_location and a CUDA-saved checkpoint is "
                                      "unreadable on a CPU-only process",
                               "effect": "placement only: the same stored float32 values, on the CPU"}]
    device_check = S.assert_child_device(require_gpu_uuid, gpu) if device != "cpu" else {"required": None, "actual": None, "asserted": False}
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    gpu_before, thermals_before = S.gpu_state(), S.host_thermals()
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    started, wall0 = S.now_iso(), time.time()
    res = S.main_like_run_py(cell["argv"], seed=cell["seed"], data_dir=data_path.parent, data_name=data_path.name,
                             work=work, gpu=gpu, use_gpu=(device != "cpu"), log=work / "replay_stdout.log",
                             train=False, bounded=False, dataloader_workers=dataloader_workers)
    cgroup_peak_after = X.cgroup_peak_bytes()
    preds, trues = np.asarray(res["preds"]), np.asarray(res["trues"])
    if preds.shape != trues.shape:
        raise ReplayRefusal(f"REFUSED: the author's scorer saw {preds.shape} against {trues.shape}")
    if preds.shape != (int(sets["test"]["windows"]), int(horizon), int(sets["test"]["channels"])):
        raise ReplayRefusal(f"REFUSED: replayed population {preds.shape} against the sealed "
                            f"{(sets['test']['windows'], horizon, sets['test']['channels'])}")
    author = {"mae": float(res["author_metric"]["mae"]), "mse": float(res["author_metric"]["mse"])}
    independent = S.float64_metrics(preds, trues)
    pred_sha, true_sha = S.sha_array(preds), S.sha_array(trues)
    finite = bool(S.all_finite(preds))
    naive = S.naive_and_trues(design, {**cell, "horizon": horizon}, data_path)
    if naive["true_sha256"] != true_sha:
        raise ReplayRefusal("REFUSED: the replayed naive is not on the replayed model's rows")
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.time() - wall0
    comparison = compare_to_record(record, author=author, independent=independent, pred_sha=pred_sha,
                                   true_sha=true_sha, naive=naive, shape=preds.shape, rule=replay_rule())
    out = {
        "schema": REPLAY_SCHEMA, "kind": "INDEPENDENTLY_INVOKED_EVALUATION_OF_A_RETAINED_CHECKPOINT",
        "reading": ("a fresh process, a fresh interpreter, a fresh CUDA context and a fresh loader reload the retained "
                    "weights through the author's own test(setting, test=1) and score the complete sealed population. "
                    "No training ran; the audited object was not modified"),
        "design_sha256": design["design_sha256"], "protocol_sha256": R.protocol_sha256(design),
        "dataset": design["dataset"], "cell_id": record["cell_id"], "horizon_steps": horizon, "seed": seed,
        "seq_len": design["seq_len"], "setting": res["setting"],
        "replayed_record_sha256": record.get("record_sha256"),
        "identity": ident, "checkpoint_staging": staged,
        "evaluation_path": {"author_test": True, "test_flag": 1, "chunked": False, "adapter": None,
                            "path": X.AUTHOR_EVAL_PATH, "trained_in_this_process": False},
        "metric": {"author_float32": author, "independent_float64": independent,
                   "space": "z_train (the training-scaler normalized target space; --inverse False)",
                   "reduction": R.contract()["reduction"],
                   "author_float32_vs_independent_float64": {"mae": author["mae"] - independent["mae"],
                                                             "mse": author["mse"] - independent["mse"]}},
        "naive": {"mse": float(naive["naive"]["mse"]), "mae": float(naive["naive"]["mae"]),
                  "definition": naive["naive_definition"],
                  "paired_on_the_same_rows_proved_by": {"target_population_sha256": true_sha,
                                                        "naive_target_sha256": naive["true_sha256"], "equal": True}},
        "population": {"sha256": true_sha, "predictions_sha256": pred_sha, "windows": int(preds.shape[0]),
                       "target_channels": int(preds.shape[2]), "elements": int(preds.size), "all_finite": finite},
        "comparison": comparison,
        "operational_patches": res.get("operational_patches", []) + cross_device_patch + [
            {"what": "the retained checkpoint copied into this replay's own work directory",
             "effect": "none: the bytes are verified by sha256 before and after the copy; the author's loader path is unchanged"}],
        "resources": {"wall_seconds": wall, "cpu_seconds": (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime),
                      "peak_rss_bytes_self": int(ru1.ru_maxrss) * 1024,
                      "whole_cgroup_peak_bytes_in_child": X.cgroup_peak_bytes(),
                      "whole_cgroup_peak_after_author_run_bytes": cgroup_peak_after,
                      "cgroup": S._cgroup_memory(), "declared_cap_bytes": X.declared_cap_bytes(),
                      "peak_gpu_allocated_bytes": (int(torch.cuda.max_memory_allocated()) if device != "cpu" and torch.cuda.is_available() else 0),
                      "peak_gpu_reserved_bytes": (int(torch.cuda.max_memory_reserved()) if device != "cpu" and torch.cuda.is_available() else 0),
                      "gpu_before": gpu_before, "gpu_after": S.gpu_state(),
                      "host_thermals_before_c": thermals_before, "host_thermals_after_c": S.host_thermals(),
                      "host_memavailable_bytes_at_exit": X.meminfo_available_bytes()},
        "device": res["device"], "device_requested": device,
        "device_uuid_measured_inside_child": (S.actual_device_uuid(gpu) if device != "cpu" else None),
        "device_assertion": device_check, "n_parameters": res["n_parameters"],
        "environment": S.environment(),
        "environment_of_the_original_cell": record.get("environment"),
        "environment_identical": S.environment() == record.get("environment"),
        "boot_id": (Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                    if Path("/proc/sys/kernel/random/boot_id").is_file() else None),
        "boot_id_of_the_original_cell": record.get("boot_id"),
        "pid": os.getpid(), "started_at": started, "finished_at": S.now_iso(),
        "reservation": {k: os.environ.get(k) for k in ("CRISPDM_RESERVATION_ID", "CRISPDM_RESERVATION_BYTES",
                                                       "CRISPDM_JOB_NAME", "CRISPDM_SLICE", "INVOCATION_ID")},
        "governance": {"class": "NOT_A_GOVERNED_UNIT_AND_NOT_A_PROMOTION",
                       "reading": ("this replay opens no governed unit, submits no receipt and changes no custody class. "
                                   "It is an audit measurement of retained artifacts")},
    }
    out["record_sha256"] = S.sha_obj(out)
    return out


# --- the closure over the twelve -------------------------------------------------------------------------------------------

def closure(design: dict, replays: list, records: list, *, device: str = "cuda") -> dict:
    """The four statuses, kept apart. A green one never carries a red one."""
    by_cell = {r["cell_id"]: r for r in replays if r.get("device_requested") == device}
    cross = [r for r in replays if r.get("device_requested") != device]
    rec = {r["cell_id"]: r for r in records}
    sealed = [c["cell_id"] for c in design["cells"]]
    missing = [c for c in sealed if c not in by_cell]
    rows = []
    for cid in sealed:
        if cid not in by_cell:
            rows.append({"cell_id": cid, "state": "NOT_REPLAYED"})
            continue
        r = by_cell[cid]
        c = r["comparison"]
        rows.append({"cell_id": cid, "horizon_steps": r["horizon_steps"], "seed": r["seed"],
                     "device_requested": r["device_requested"],
                     "predictions_bitwise_equal": c["bitwise"]["predictions_bitwise_equal"],
                     "population_bitwise_equal": c["bitwise"]["target_population_bitwise_equal"],
                     "metric_exactly_equal": (c["bitwise"]["author_float32_mae_exactly_equal"]
                                              and c["bitwise"]["author_float32_mse_exactly_equal"]),
                     "naive_exactly_equal": (c["bitwise"]["naive_mae_exactly_equal"]
                                             and c["bitwise"]["naive_mse_exactly_equal"]),
                     "bitwise_all": c["bitwise"]["all"],
                     "frozen_tolerance_met": c["frozen_tolerance"]["met"],
                     "author_float32_delta": c["frozen_tolerance"]["author_float32_delta"],
                     "identity_all_agree": all([r["identity"]["recipe"]["design_agrees"],
                                                r["identity"]["recipe"]["protocol_agrees"],
                                                r["identity"]["recipe"]["author_files_agree"],
                                                r["identity"]["transport"]["agrees"],
                                                r["identity"]["scorer"]["agrees"],
                                                r["identity"]["weights"]["agrees"],
                                                (r["identity"]["weights"].get("selection_rule") or {}).get("agrees", False)]),
                     "replayed_record_sha256": r["replayed_record_sha256"],
                     "record_sha256_recomputed": (S.sha_obj({k: v for k, v in rec[cid].items() if k != "record_sha256"})
                                                  if cid in rec else None),
                     "wall_seconds": r["resources"]["wall_seconds"],
                     "cgroup_peak_bytes": r["resources"]["whole_cgroup_peak_bytes_in_child"],
                     "declared_cap_bytes": r["resources"]["declared_cap_bytes"]})
    replayed = [r for r in rows if r.get("state") != "NOT_REPLAYED"]
    numerical = {
        "status": ("IDENTITIES_RECONCILED" if replayed and all(r["identity_all_agree"] for r in replayed) and not missing
                   else "NOT_ESTABLISHED"),
        "cells": len(replayed), "sealed": len(sealed), "missing": missing,
        "scope": "recipe, selected weights, test population, scorer and naive identities, each re-derived in the replay child",
    }
    replay_status = {
        "status": ("BITWISE_REPRODUCED" if replayed and not missing and all(r["bitwise_all"] for r in replayed)
                   else ("WITHIN_FROZEN_TOLERANCE_NOT_BITWISE"
                         if replayed and not missing and all(r["frozen_tolerance_met"] for r in replayed)
                         else "NOT_ESTABLISHED")),
        "cells_bitwise": sum(1 for r in replayed if r["bitwise_all"]),
        "cells_within_frozen_tolerance": sum(1 for r in replayed if r["frozen_tolerance_met"]),
        "cells_replayed": len(replayed), "cells_sealed": len(sealed),
        "rule": replay_rule(),
        "scope": "a fresh process per cell reloading the retained checkpoint through the author's own unchunked test path",
    }
    cross_rows = [{"cell_id": r["cell_id"], "device_requested": r["device_requested"], "device": r["device"],
                   "author_float32_delta": r["comparison"]["frozen_tolerance"]["author_float32_delta"],
                   "author_metric_within_1e-5": r["comparison"]["frozen_tolerance"]["author_metric_within_1e-5"],
                   "predictions_bitwise_equal": r["comparison"]["bitwise"]["predictions_bitwise_equal"],
                   "population_bitwise_equal": r["comparison"]["bitwise"]["target_population_bitwise_equal"],
                   "naive_delta": r["comparison"]["frozen_tolerance"]["naive_delta"]} for r in cross]
    return {"schema": CLOSURE_SCHEMA, "design_sha256": design["design_sha256"],
            "protocol_sha256": R.protocol_sha256(design), "dataset": design["dataset"],
            "primary_device": device,
            "rows": rows, "numerical_agreement": numerical, "replay": replay_status,
            "cross_device_replays": {
                "device": (cross[0]["device_requested"] if cross else None), "rows": cross_rows,
                "reading": ("the frozen rule's own declared device is `cpu`, a CROSS-device replay. These rows exercise it "
                            "and bound the kernel-level difference; they are reported apart from the same-device replay and "
                            "never averaged with it")},
            "custody": {"status": "REPORTED_SEPARATELY",
                        "reading": "custody is not established by a replay and is never inferred from one"},
            "scientific": {"status": "REPORTED_SEPARATELY",
                           "reading": "a reproduced number is not a scientific verdict; the comparability class and the "
                                      "one-sided gap are argued on their own evidence"},
            "built_at": S.now_iso()}


# --- CLI -------------------------------------------------------------------------------------------------------------------

def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("replay", help="one fresh-process evaluation of one retained checkpoint")
    p.add_argument("--lane", required=True, help="the executed lane root (design, characterization, CELLS, cell_* folders)")
    p.add_argument("--out", required=True, help="the replay lane root this record is written under")
    p.add_argument("--dataset", default="weather")
    p.add_argument("--protocol", default="L96")
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--require-gpu-uuid", default=os.environ.get("CRISPDM_REQUIRED_GPU_UUID"))
    p.add_argument("--dataloader-workers", type=int, default=0)

    c = sub.add_parser("close", help="the closure over the retained replay records")
    c.add_argument("--lane", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--dataset", default="weather")
    c.add_argument("--protocol", default="L96")
    c.add_argument("--device", default="cuda", choices=("cuda", "cpu"), help="the PRIMARY device the twelve are judged on")

    a = ap.parse_args(argv)
    lane, out = Path(a.lane).expanduser(), Path(a.out).expanduser()
    design = json.loads((lane / f"DESIGN.{a.dataset}.{a.protocol}.json").read_text())
    characterization = json.loads((lane / f"CHARACTERIZATION.{a.dataset}.json").read_text())

    if a.cmd == "replay":
        cid = f"{a.dataset}_{a.protocol}_h{a.horizon}_s{a.seed}"
        record = json.loads((lane / "CELLS" / f"{cid}.json").read_text())
        rec_sha = S.sha_obj({k: v for k, v in record.items() if k != "record_sha256"})
        if rec_sha != record.get("record_sha256"):
            raise ReplayRefusal(f"REFUSED: the retained record's own digest does not reconcile ({rec_sha[:12]})")
        rep = replay_cell(design, characterization, record, data_path=Path(a.data_path).expanduser(),
                          work=out / f"replay_{a.dataset}_h{a.horizon}_s{a.seed}", lane=lane, gpu=a.gpu,
                          require_gpu_uuid=a.require_gpu_uuid, device=a.device,
                          dataloader_workers=a.dataloader_workers)
        path = _write(out / "REPLAYS" / f"{cid}.{a.device}.json", rep)
        cmp_ = rep["comparison"]
        print(json.dumps({"cell": cid, "device": a.device, "bitwise": cmp_["bitwise"]["all"],
                          "frozen_tolerance_met": cmp_["frozen_tolerance"]["met"],
                          "predictions_bitwise_equal": cmp_["bitwise"]["predictions_bitwise_equal"],
                          "mse": rep["metric"]["author_float32"]["mse"], "mae": rep["metric"]["author_float32"]["mae"],
                          "cgroup_peak_bytes": rep["resources"]["whole_cgroup_peak_bytes_in_child"],
                          "declared_cap_bytes": rep["resources"]["declared_cap_bytes"],
                          "wall_seconds": rep["resources"]["wall_seconds"], "record": str(path)}, indent=1))
        return 0

    replays = [json.loads(p.read_text()) for p in sorted((out / "REPLAYS").glob(f"{a.dataset}_{a.protocol}_*.json"))]
    records = [json.loads(p.read_text()) for p in sorted((lane / "CELLS").glob("*.json"))]
    clo = closure(design, replays, records, device=a.device)
    path = _write(out / f"REPLAY_CLOSURE.{a.dataset}.{a.protocol}.json", clo)
    print(json.dumps({"numerical_agreement": clo["numerical_agreement"]["status"],
                      "replay": clo["replay"]["status"],
                      "cells_bitwise": clo["replay"]["cells_bitwise"],
                      "cells_replayed": clo["replay"]["cells_replayed"], "closure": str(path)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
