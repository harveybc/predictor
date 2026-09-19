#!/usr/bin/env python3
"""MOD-E0 closure bound to the registered population (RP10) with task, weight and metric
verification from the files (RP11). No training, no new platform.

Population: derived from the sealed DESIGN.json in the root (schema, self-digest, cells equal to
the design's own enumeration, successor chain recorded) plus the cost pilots the runner derives
from that design, and from the campaign registrations (CAMPAIGNS.json keyed by run id). REPORT.json
is NOT an independent source of what the run did: it is only compared against the files (parent)
and used for the code identity it recorded.

Local closure (files): every member must have an attempt whose job equals the job the design
implies, whose outcome/result/cell digests agree, whose arrays are exactly the generator's output
for (level, r, seed) under the contract (rows, targets, naive, oracle, denominators, train-only
scale, linear reference), whose scores for every baseline and variable are recomputed from the
arrays (a non-finite value is never MEDIDO), whose weights are readable and reproduce the stored
predictions in a fresh CPU process within a declared tolerance, whose restored checkpoint reaches
the recorded best validation loss, and — for an H3 arm — whose donor is the verified attempt of
its declared dependency with every extractor weight and adapter activation equal in both arms.
A stranger attempt, a duplicated or unexpected terminal in the report, a design that does not
seal itself or a report of another run are typed refusals (exit 2). A member without a verified
attempt makes the closure PARTIAL (exit 1) with the population denominator fixed by the design;
`all_verified` is the conjunction of every member's status and the parent comparison.

Live closure (governance + warehouse): the registered population of each campaign is derived
from accounting (reconciliation: missing units plus units holding a terminal; accounting and
lake must agree) and compared with the design; for every unit the current-generation terminal in
the warehouse must carry the local outcome's status, costs, output digest, code identity, design
identity, every metric row and every metric state. Local and live results are reported apart and
neither is presented as the other.

    python tools/df_mod_e0_close.py --root RUN_ROOT [--out-dir DIR] [--no-live]
        [--gov-url URL --api-key-file KEY] [--warehouse-url URL --token-env VAR] [--tolerance 1e-5]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, where: Path = HERE):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


E = _load("df_mod_e0")
D = _load("df_mod_e0_design")
AD = _load("df_mod_e0_arch_design")
V = _load("df_mod_e0_verify")
RUN = _load("df_mod_e0_run")

SCHEMA = "df_mod_e0_close.v2"
TOTAL, PARTIAL, REFUSED = "TOTAL", "PARTIAL", "REFUSED"
VERIFIED, PROBLEMS, MISSING, FAILED, CARRIED, NO_DONOR = "VERIFIED", "PROBLEMS", "MISSING", "FAILED", "CARRIED", "H3_ARM_WITHOUT_VERIFIED_DONOR"
DEFAULT_TOLERANCE = {"prediction_atol": 1e-5, "restore_rel": 1e-5, "linear_rtol": 1e-6, "activation_atol": 1e-6,
                     "why": "float32 inference: convolution accumulation order differs between processes and thread counts "
                            "(observed max |diff| ~1e-6 in the replays of the executed pilot); the tolerance is declared BEFORE "
                            "the check and recorded with every measured difference"}
SPLITS = ("train", "validation", "test")
BASELINES = (("model", "pred"), ("naive", "naive"), ("oracle", "oracle"), ("linear_window", "linear"))


class ClosureRefusal(SystemExit):
    """A closure that cannot be made: the population is not the registered one."""

    def __init__(self, why: str):
        super().__init__(why)
        self.code = 2
        self.why = why


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(x) -> bool:
    """A number that is really a number: not None, not bool, not NaN/inf (RP19: NaN > tol is False)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


REPLAY_REQUIRED = ("schema", "attempt", "problems", "inputs", "prediction_max_abs_diff", "restore_abs_diff", "restored_validation_loss",
                   "recorded_best_validation_loss", "scale_recomputed_equal")


# --- population --------------------------------------------------------------------------------------------

def pilots_of(design: dict) -> list:
    """The cost pilots the runner derives from a design (df_mod_e0_run.run_mod_e0), with their campaign suffix."""
    return [{"cell_id": "pilot__H2_profiles", "hypothesis": "H2", "level": 3, "r": 1, "seed": 1, "arm": "profiles", "campaign": "-mod-e0-cost-pilot"},
            {"cell_id": "pilot__H3_extractor", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "extractor", "campaign": "-mod-e0-cost-pilot"},
            {"cell_id": "pilot__H3_sequence", "hypothesis": "H3", "level": design["h3_level"], "r": 1, "seed": 1, "arm": "sequence", "campaign": "-mod-e0-cost-pilot-arm",
             "depends_on": "pilot__H3_extractor"}]


def population(design: dict) -> dict:
    """Members, pilots, dependencies and campaign membership DERIVED from the sealed design (v1 or v2)."""
    v2 = design.get("schema") == AD.DESIGN_SCHEMA
    if design.get("schema") not in (D.DESIGN_SCHEMA, AD.DESIGN_SCHEMA):
        raise ClosureRefusal(f"REFUSED: design schema {design.get('schema')!r} is not a MOD-E0 design")
    body = {k: v for k, v in design.items() if k != "design_sha256"}
    if E.sha_obj(body) != design.get("design_sha256"):
        raise ClosureRefusal("REFUSED: the design does not seal itself (self-digest differs)")
    if v2 and design.get("kind") == "READOUT_COMPLETION":
        # the successor's enumeration is the parent's readout completion; re-derive it from the design's own fields
        parent_like = {**design, "readout_controls_r": [r for r in design["r_values"] if r not in {c["r"] for c in design["cells"]}] or design["readout_controls_r"]}
        rs0 = sorted({c["r"] for c in design["cells"]})
        expected_cells = [{"cell_id": f"H3__r{r}__s{seed}__{a}__{arm}", "hypothesis": "H3", "level": design["h3_level"], "r": r, "seed": seed, "arm": arm, "arch": a,
                           "depends_on": f"H3__r{r}__s{seed}__{a}__extractor", "donor": "sequence", "host_role": "COORDINATOR"}
                          for a in design["archs"] for r in rs0 for seed in design["replicates"] for arm in ("sequence_gap", "summary_last")]
        expected_inherited = [f"H3__r{r}__s{seed}__{a}__extractor" for a in design["archs"] for r in rs0 for seed in design["replicates"]]
        if [i["cell_id"] for i in design.get("inherited") or []] != expected_inherited or not design.get("successor_of"):
            raise ClosureRefusal("REFUSED: the readout-completion successor does not enumerate its inherited donors")
    else:
        expected_cells = AD.cells(design) if v2 else D.cells(design)
    if v2 and design.get("pilots") != ([] if (design.get("only_hypotheses") or design.get("kind") == "READOUT_COMPLETION") else AD.pilots(design)):
        raise ClosureRefusal("REFUSED: the design's pilots are not its own derivation")
    if expected_cells != design.get("cells") or design.get("cells_total") != len(expected_cells):
        raise ClosureRefusal("REFUSED: the design's cells are not its own enumeration (levels/replicates/assignments/h3_level)")
    if not expected_cells:
        raise ClosureRefusal("REFUSED: the design enumerates no cell (empty population)")
    ids = [c["cell_id"] for c in expected_cells]
    if len(set(ids)) != len(ids):
        raise ClosureRefusal("REFUSED: the design repeats a cell id")
    inherited = {c["cell_id"]: c for c in (design.get("inherited") or [])}
    for c in expected_cells:
        if c.get("depends_on") and c["depends_on"] not in ids and c["depends_on"] not in inherited:
            raise ClosureRefusal(f"REFUSED: {c['cell_id']} depends on {c['depends_on']!r}, not a member nor an inherited donor")
    pilots = (design.get("pilots") or []) if v2 else pilots_of(design)
    return {"design_sha256": design["design_sha256"], "successor_of": design.get("successor_of"), "successor_reason": design.get("successor_reason"), "v2": v2,
            "cells": expected_cells, "members": ids, "pilots": pilots, "pilot_ids": [p["cell_id"] for p in pilots],
            "campaigns": {"-mod-e0-cells": ids, "-mod-e0-cost-pilot": [p["cell_id"] for p in pilots if p["campaign"] == "-mod-e0-cost-pilot"],
                          "-mod-e0-cost-pilot-arm": [p["cell_id"] for p in pilots if p["campaign"] == "-mod-e0-cost-pilot-arm"]},
            "dependencies": {c["cell_id"]: c["depends_on"] for c in expected_cells + pilots if c.get("depends_on")},
            "inherited": list(inherited.values()),
            "training": design["training"], "window": design["window"], "n_total": design["n_total"], "horizon": design["horizon"]}


def expected_job(design: dict, run_id: str, unit: dict, root: Path, pilot_updates: int | None) -> dict:
    """The job the runner builds for a unit (df_mod_e0_run.job_for + role/extractor extras)."""
    pilot = unit["cell_id"].startswith("pilot__")
    job = {"kind": "mod_e0_cell", "cell_id": unit["cell_id"], "hypothesis": unit["hypothesis"], "level": unit["level"], "r": unit["r"],
           "seed": unit["seed"], "arm": unit["arm"], "window": design["window"], "training": design["training"],
           "design_sha256": design["design_sha256"], "run_id": run_id, "role": "COST_PILOT" if pilot else "CELL",
           **{k: unit[k] for k in ("arch", "donor", "diagnostic") if k in unit}}
    if pilot:
        job["max_updates_override"] = pilot_updates
    if unit.get("depends_on"):
        job["extractor_weights"] = str(root / "attempts" / unit["depends_on"] / "weights.weights.h5")
    return job


CONSUMED = ("kind", "cell_id", "hypothesis", "level", "r", "seed", "arm", "window", "training", "design_sha256", "run_id", "role",
            "extractor_weights", "max_updates_override", "arch", "donor", "diagnostic")


# --- one attempt, from the files ----------------------------------------------------------------------------

def _close(a, b, rtol, atol=0.0) -> bool:
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def verify_attempt(attempt: Path, job_expected: dict, replay_doc: dict | None, tolerance: dict, donor_attempt: Path | None = None) -> dict:
    """Every check of one attempt from its files; `replay_doc` is the fresh-process reproduction."""
    entry = {"attempt": attempt.name, "status": None, "problems": [], "record": None, "checks": {}, "measured": {}}
    prob = entry["problems"].append
    if not attempt.is_dir():
        entry["status"] = MISSING
        prob("no attempt directory")
        return entry
    if not (attempt / "outcome.json").is_file():
        entry["status"] = MISSING
        prob("no outcome.json")
        return entry
    outcome = json.loads((attempt / "outcome.json").read_text())
    if outcome.get("status") != "COMPLETED":
        entry["status"] = FAILED
        entry["carried"] = {"status": outcome.get("status"), "outcome": (outcome.get("summary") or {}).get("outcome"),
                            "reason": (outcome.get("summary") or {}).get("reason")}
        prob(f"attempt did not complete: {outcome.get('status')}")
        return entry
    # --- the consumed job equals the job the design implies ---
    job = json.loads((attempt / "job.json").read_text()) if (attempt / "job.json").is_file() else None
    if job is None:
        prob("job.json absent: the consumed parameters cannot be bound to the design")
    else:
        for k in CONSUMED:
            if job.get(k) != job_expected.get(k):
                prob(f"job.{k}: consumed {job.get(k)!r} vs design {job_expected.get(k)!r}")
    entry["checks"]["job_equals_design"] = not entry["problems"]
    # --- receipts and digests ---
    result = json.loads((attempt / "result.json").read_text()) if (attempt / "result.json").is_file() else {}
    if not (attempt / "cell.json").is_file():
        prob("cell.json absent")
        entry["status"] = PROBLEMS
        return entry
    body = (attempt / "cell.json").read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != result.get("output_sha256") or digest != (outcome.get("verified") or {}).get("output_sha256"):
        prob("cell.json bytes differ from the declared/verified digest")
    entry["digests"] = {"cell": digest}
    rec = json.loads(body)
    if rec.get("schema") != E.CELL_SCHEMA:
        prob(f"record schema {rec.get('schema')!r}")
        entry["status"] = PROBLEMS
        return entry
    for k in ("cell_id", "hypothesis", "level", "r", "seed", "arm", "window"):
        if rec.get(k) != job_expected.get(k):
            prob(f"record.{k}: {rec.get(k)!r} vs design {job_expected.get(k)!r}")
    if rec.get("role") != job_expected["role"]:
        prob(f"record.role {rec.get('role')!r} vs {job_expected['role']!r}")
    rule = (rec.get("training") or {}).get("rule") or {}
    expected_rule = dict(job_expected["training"])
    if job_expected.get("max_updates_override"):
        expected_rule["max_updates"] = int(job_expected["max_updates_override"])
    if {k: rule.get(k) for k in expected_rule} != expected_rule:
        prob(f"training rule consumed {rule} differs from the design's {expected_rule}")
    # --- arrays: digests, schema, finiteness ---
    arrays = attempt / "arrays.npz"
    if not arrays.is_file() or sha(arrays) != rec.get("arrays_sha256"):
        prob("arrays absent or altered (digest)")
        entry["status"] = PROBLEMS
        return entry
    with np.load(arrays) as z:
        arr = {k: z[k] for k in z.files}
    pilot = job_expected["role"] == "COST_PILOT"
    parts = [q for q in SPLITS if f"{q}_y" in arr]
    if pilot and "test" in parts:
        prob("a cost pilot holds test arrays")
    if not pilot and parts != list(SPLITS):
        prob(f"splits present {parts} (expected train, validation, test)")
    if rec.get("exposure") != ("NO_TEST_ACCESS" if pilot else "TEST_SCORED_DESCRIPTIVE"):
        prob(f"exposure {rec.get('exposure')!r} for role {job_expected['role']}")
    # --- the task from the generator and the contract ---
    diag = None if (rec.get("diagnostic") or "none") == "none" else rec["diagnostic"]
    if diag != (job_expected.get("diagnostic") or None):
        prob(f"record diagnostic {rec.get('diagnostic')!r} vs design {job_expected.get('diagnostic')!r}")
    if str(rec.get("arch") or E.DEFAULT_ARCH) != str(job_expected.get("arch") or E.DEFAULT_ARCH):
        prob(f"record arch {rec.get('arch')!r} vs design {job_expected.get('arch')!r}")
    gen = E.generate(int(rec["level"]), int(rec["r"]), int(rec["seed"]), diagnostic=diag)
    x = gen["x"]
    p = x.shape[1]
    if hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() != rec.get("x_sha256"):
        prob("consumed data are not the generator's output for (level, r, seed)")
    periods = [gen["params"]["groups"][g]["period"] for g in gen["params"]["latent_groups"]]
    prep = E.prepare(x, gen["oracle"], periods, int(rec["window"]), int(rec["horizon"]), test_access=not pilot)
    if prep["boundaries"] != rec.get("boundaries"):
        prob("boundaries differ from the contract's")
    denom_expected = prep["mase_denominator"]
    denom = arr["denominator"]
    if denom.shape != (p,) or not np.array_equal(denom, np.asarray(denom_expected)):
        prob("denominators are not the train seasonal-naive MAE of the generator's series")
    if list(map(float, rec.get("mase_denominator") or [])) != list(map(float, denom.tolist())):
        prob("record denominators differ from the arrays'")
    scale = rec.get("scale")
    if scale is not None and not (_close(scale["mean"], prep["scale"]["mean"], 1e-12) and _close(scale["sd"], prep["scale"]["sd"], 1e-12)):
        prob("recorded scale is not the train-only scale of the contract")
    for part in parts:
        P = prep["parts"].get(part)
        if P is None:
            prob(f"{part}: the contract has no such split for this role")
            continue
        for key, value in (("rows", P["rows"]), ("y", P["y"]), ("naive", P["naive"]), ("oracle", P["oracle"])):
            a = arr.get(f"{part}_{key}")
            if a is None or a.shape != value.shape or not np.array_equal(a, value):
                prob(f"{part}.{key}: not the generator's under the contract")
        if rec.get("rows", {}).get(part) != int(P["rows"].size):
            prob(f"{part}: recorded rows {rec.get('rows', {}).get(part)} vs contract {int(P['rows'].size)}")
        for key in ("pred", "linear"):
            a = arr.get(f"{part}_{key}")
            if a is None or a.shape != P["y"].shape or not np.issubdtype(a.dtype, np.floating):
                prob(f"{part}.{key}: absent, wrong shape or dtype")
    # linear reference recomputed from the train-only ridge of the contract
    if "train" in parts and all(f"{q}_linear" in arr for q in parts):
        s = prep["scale"]
        Xtr, ytr = E._sx(prep["parts"]["train"]["X"], s), E._sy(prep["parts"]["train"]["y"], s)
        Ftr = Xtr.reshape(Xtr.shape[0], -1)
        A = np.hstack([Ftr, np.ones((Ftr.shape[0], 1))])
        w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)
        for part in parts:
            Xp = E._sx(prep["parts"][part]["X"], s).reshape(prep["parts"][part]["X"].shape[0], -1)
            lin = E._uy(np.hstack([Xp, np.ones((Xp.shape[0], 1))]) @ w, s)
            if not _close(arr[f"{part}_linear"], lin, tolerance["linear_rtol"], 1e-9):
                prob(f"{part}.linear: not the train-only ridge reference of the contract")
    # --- every score of every baseline and variable from the arrays; non-finite is never MEDIDO ---
    entry["recomputed"] = {}
    for part in parts:
        y = arr.get(f"{part}_y")
        scores = (rec.get("scores") or {}).get(part) or {}
        if scores.get("rows") != int(y.shape[0]):
            prob(f"{part}: recorded score rows {scores.get('rows')} vs arrays {int(y.shape[0])}")
        for model, src in BASELINES:
            a = arr.get(f"{part}_{src}")
            got = scores.get(model)
            if a is None or got is None:
                prob(f"{part}.{model}: score or array absent")
                continue
            if a.shape != y.shape:
                prob(f"{part}.{model}: array shape {a.shape} vs targets {y.shape}")
                continue
            new = E.mase(a, y, denom.tolist())
            entry["recomputed"].setdefault(part, {})[model] = {"mase_mean": new["mase_mean"], "mae_mean": new["mae_mean"], "status": new["status"]}
            if new["status"] == E.NO_MEDIDO:
                if got.get("mase_mean") is not None or got.get("mae_mean") is not None or got.get("status", E.NO_MEDIDO) != E.NO_MEDIDO:
                    prob(f"{part}.{model}: non-finite or empty arrays recorded as measured")
                continue
            for agg in ("mase_mean", "mae_mean"):
                a_, b_ = new[agg], got.get(agg)
                if (a_ is None) != (b_ is None) or (a_ is not None and abs(a_ - b_) > 1e-9):
                    prob(f"{part}.{model}.{agg}: record {b_} vs recomputed {a_}")
            per = got.get("per_variable") or {}
            for k in range(y.shape[1]):
                pv, nv = per.get(str(k), per.get(k)) or {}, new["per_variable"][k]
                for f in ("mae", "mase"):
                    a_, b_ = nv[f], pv.get(f)
                    if (a_ is None) != (b_ is None) or (a_ is not None and abs(a_ - b_) > 1e-9):
                        prob(f"{part}.{model}.var{k}.{f}: record {b_} vs recomputed {a_}")
                if pv.get("status") != nv["status"]:
                    prob(f"{part}.{model}.var{k}: state {pv.get('status')} vs {nv['status']}")
    # --- training rule: allowance, stop, restore ---
    tr = rec.get("training") or {}
    if int(tr.get("updates", -1)) > int(expected_rule["max_updates"]):
        prob(f"updates {tr.get('updates')} exceed the allowance {expected_rule['max_updates']}")
    if tr.get("restore_verified") is False:
        prob("the record itself says the best checkpoint was not restored")
    # --- profiles and assignment recomputed from the generator ---
    lo, hi = prep["boundaries"]["train"]
    dv = int((rec.get("profiles") or {}).get("descriptor_version") or rec.get("descriptor_version") or 1)
    prof = E.profiles(x[lo:hi], dv)
    labels = E.average_linkage(prof["scaled"], 2)
    if labels != (rec.get("profiles") or {}).get("labels"):
        prob(f"profile labels recomputed with descriptor v{dv} differ from the record's")
    if prof["kept"] != (rec.get("profiles") or {}).get("kept"):
        prob("profile descriptors kept differ from the record's")
    latent = [0 if g == "A" else 1 for g in gen["params"]["latent_groups"]]
    if abs(E.adjusted_rand(labels, latent) - float((rec.get("profiles") or {}).get("ari_vs_latent", -9))) > 1e-12:
        prob("ARI vs latent groups differs from the record's")
    assignment = rec.get("assignment")
    if rec["hypothesis"] in ("H2", "DX"):                       # DX (RP14 diagnostic) uses the H2 arms
        if rec["arm"] == "profiles":
            if assignment != labels or not rec.get("assignment_is_profile"):
                prob("H2 profiles arm did not use the profile partition")
        else:
            k = int(rec["arm"].split("_")[-1])
            sizes = [labels.count(g) for g in sorted(set(labels))]
            cands = [c for c in E.random_assignments(p, 2, sizes, 8, seed=1000 + int(rec["seed"])) if not E.same_partition(c, labels)]
            if k >= len(cands) or assignment != cands[k] or rec.get("assignment_is_profile"):
                prob("H2 random arm is not the predefined redistribution k of this replicate (or equals the profile partition)")
    else:
        if assignment != labels:
            prob("H3 unit did not use the profile partition")
        if rec.get("fusion") != {"extractor": "sequence", "extractor_summary": "summary", "sequence": "sequence", "sequence_gap": "sequence_gap",
                                 "summary": "summary", "summary_last": "summary_last"}.get(rec["arm"]):
            prob("fusion does not match the arm")
    # --- weights: digest, donor binding ---
    wpath = attempt / "weights.weights.h5"
    if not wpath.is_file() or sha(wpath) != rec.get("weights_sha256"):
        prob("weights absent or altered (digest)")
    arm_h3 = rec["hypothesis"] == "H3" and rec["arm"] in ("sequence", "sequence_gap", "summary", "summary_last")
    no_extractor = str(rec.get("arch") or E.DEFAULT_ARCH) == "0"
    if arm_h3:
        frozen = rec.get("frozen") or {}
        if not frozen or rec.get("extractor_weight_change", 1.0) != 0.0:
            prob("H3 arm: the record does not show a frozen extractor")
        if (job_expected.get("donor") or "sequence") != (rec.get("donor") or "sequence"):
            prob(f"H3 arm: donor kind {rec.get('donor')!r} vs design {job_expected.get('donor')!r}")
        if donor_attempt is None or not (donor_attempt / "weights.weights.h5").is_file():
            prob("H3 arm: the declared donor attempt has no weights")
        elif frozen.get("extractor_sha256") != sha(donor_attempt / "weights.weights.h5"):
            prob("H3 arm: the consumed extractor digest is not the declared donor's weights")
        if job and job.get("extractor_weights") and Path(job["extractor_weights"]).name != "weights.weights.h5":
            prob("H3 arm: job names a foreign extractor file")
    # --- fresh-process replay: predictions, restore, donor equality ---
    if replay_doc is None:
        prob("no fresh-process replay of the weights")
    else:
        entry["measured"]["replay"] = {k: replay_doc.get(k) for k in ("prediction_max_abs_diff", "restore_abs_diff", "restored_validation_loss",
                                                                       "recorded_best_validation_loss", "extractor_weights_unequal_layers",
                                                                       "adapter_activation_max_abs_diff", "scale_recomputed_equal", "process", "cpu_seconds")}
        entry["replay_scope"] = replay_doc.get("_scope") or ("CURRENT_CODE" if replay_doc.get("identity") else "HISTORIC_v1_UNBOUND_CODE")
        entry["replay_identity"] = replay_doc.get("identity")
        for why in replay_doc.get("problems") or []:
            prob(f"replay: {why}")
        missing_keys = [k for k in REPLAY_REQUIRED if k not in replay_doc]
        if missing_keys:
            prob(f"replay: document incomplete, missing {missing_keys}")
        if replay_doc.get("_process_exit") not in (None, 0):
            prob(f"replay: the replay process exited {replay_doc.get('_process_exit')}; a partial document is not a reproduction")
        diffs = replay_doc.get("prediction_max_abs_diff") or {}
        for part in parts:
            d = diffs.get(part)
            if not _finite(d) or d > tolerance["prediction_atol"]:
                prob(f"replay: {part} predictions from the reloaded weights differ by {d} (> {tolerance['prediction_atol']}) or are not a finite number")
        rd, best, rl = replay_doc.get("restore_abs_diff"), replay_doc.get("recorded_best_validation_loss"), replay_doc.get("restored_validation_loss")
        if not (_finite(rd) and _finite(best) and _finite(rl)):
            prob(f"replay: restore evidence is not finite (diff {rd!r}, best {best!r}, restored {rl!r})")
        elif rd > tolerance["restore_rel"] * max(1.0, abs(best)):
            prob(f"replay: the saved weights do not reach the recorded best validation loss (|diff| = {rd})")
        if replay_doc.get("scale_recomputed_equal") is not True:
            prob(f"replay: scale equality is {replay_doc.get('scale_recomputed_equal')!r}, not True")
        if arm_h3:
            if replay_doc.get("extractor_weights_unequal_layers"):
                prob(f"replay: extractor weights differ from the donor in {replay_doc['extractor_weights_unequal_layers']}")
            act = replay_doc.get("adapter_activation_max_abs_diff")
            if no_extractor:
                if act:
                    prob("replay: ARCH-0 has no extractor, yet adapter activations were reported")
            else:
                expected_adapters = sorted({f"g{g}_adapt" for g in set(rec.get("assignment") or [])})
                if not isinstance(act, dict) or sorted(act) != expected_adapters:
                    prob(f"replay: adapter activations missing or not the expected adapters {expected_adapters}: {act!r}")
                elif any(not _finite(v) or v > tolerance["activation_atol"] for v in act.values()):
                    prob(f"replay: adapter activations differ from the donor's or are not finite: {act}")
            unequal = replay_doc.get("extractor_weights_unequal_layers")
            if not isinstance(unequal, list):
                prob(f"replay: extractor weight comparison absent ({unequal!r})")
            if not (replay_doc.get("donor") or {}).get("sha256_equals_record"):
                prob("replay: the donor file read in the fresh process is not the one the record names")
        # the parity flag alone is not evidence: the replayed arrays are
        if not rec.get("prediction_parity_after_reload"):
            prob("record says reload parity failed")
    entry["checks"].update({"receipts": not any("digest" in q for q in entry["problems"]),
                            "task_from_generator": not any(q.startswith(("train.", "validation.", "test.", "boundaries", "denominators", "consumed data", "recorded scale")) for q in entry["problems"]),
                            "scores_from_arrays": not any(".mase_mean" in q or ".mae_mean" in q or ".var" in q or "non-finite" in q for q in entry["problems"]),
                            "weights_and_replay": not any(q.startswith(("replay", "weights", "H3 arm", "record says")) for q in entry["problems"]),
                            "profiles_and_assignment": not any(q.startswith(("profile", "H2 ", "H3 unit", "ARI", "fusion")) for q in entry["problems"])})
    entry["status"] = VERIFIED if not entry["problems"] else PROBLEMS
    entry["record"] = {k: rec.get(k) for k in ("cell_id", "hypothesis", "level", "r", "seed", "arm", "role", "assignment", "assignment_is_profile",
                                              "fusion", "parameters", "exposure", "extractor_weight_change", "receptive_field", "window",
                                              "descriptor_version", "arch", "diagnostic", "donor", "branch_reach", "support_reach")}
    entry["record"]["profiles_ari"] = rec["profiles"]["ari_vs_latent"]
    entry["record"]["updates"] = tr.get("updates")
    entry["record"]["stop_reason"] = tr.get("stop_reason")
    entry["record"]["cost"] = rec.get("cost")
    for name in ("mase", "mae"):
        entry["record"][name] = {q: rec["scores"][q]["model"].get(f"{name}_mean") for q in rec["scores"]}
    entry["record"]["naive_mase"] = {q: rec["scores"][q]["naive"]["mase_mean"] for q in rec["scores"]}
    entry["record"]["oracle_mase"] = {q: rec["scores"][q]["oracle"]["mase_mean"] for q in rec["scores"]}
    entry["record"]["linear_mase"] = {q: rec["scores"][q]["linear_window"]["mase_mean"] for q in rec["scores"]}
    entry["record"]["mse"] = {q: entry["recomputed"].get(q, {}).get("model", {}).get("mase_mean") and float(np.mean((arr[f"{q}_pred"] - arr[f"{q}_y"]) ** 2)) for q in parts}
    entry["record"]["curve"] = tr.get("curve")
    entry["record"]["denominator"] = denom.tolist()
    return entry


# --- replays in fresh processes -----------------------------------------------------------------------------

def current_replay_identity(threads: int) -> dict:
    """The identity a fresh replay would carry now (computed in a child process with the same environment)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(threads), "TF_CPP_MIN_LOG_LEVEL": "3"}
    code = "import json,sys; sys.path.insert(0, sys.argv[1]); import df_mod_e0 as E; print(json.dumps(E.replay_identity()))"
    proc = subprocess.run([sys.executable, "-B", "-c", code, str(HERE)], env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"replay identity failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_replays(attempts: list, out_dir: Path, workers: int = 4, threads: int = 2, historic: Path | None = None, only: set | None = None) -> dict:
    """Reproduce each attempt in its own CPU process (df_mod_e0.py --replay); CPU accounted. A cached document
    is reused only when its inputs AND its replay identity (code + numeric environment) equal the current
    ones. `historic`: a directory of earlier replay documents used, for attempts outside `only`, under the
    explicit scope HISTORIC (labelled, never promoted); `only`: attempt names that must be replayed now."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(threads), "TF_CPP_MIN_LOG_LEVEL": "3"}
    identity = current_replay_identity(threads)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)

    def current_inputs(attempt: Path) -> dict:
        inputs = {n: (sha(attempt / f) if (attempt / f).is_file() else None)
                  for n, f in (("cell", "cell.json"), ("arrays", "arrays.npz"), ("weights", "weights.weights.h5"), ("job", "job.json"))}
        job = json.loads((attempt / "job.json").read_text()) if (attempt / "job.json").is_file() else {}
        if job.get("extractor_weights") and Path(job["extractor_weights"]).is_file():
            inputs["donor"] = sha(Path(job["extractor_weights"]))
        return inputs

    def one(attempt: Path):
        target = out_dir / f"{attempt.name}.json"
        if target.is_file():
            prior = json.loads(target.read_text())
            if prior.get("inputs") == current_inputs(attempt) and prior.get("identity") == identity and prior.get("_process_exit", 0) == 0:
                return attempt.name, prior, "reused"                 # the very same bytes under the very same implementation and environment
            target.unlink()
        if only is not None and attempt.name not in only and historic is not None and (historic / f"{attempt.name}.json").is_file():
            doc = json.loads((historic / f"{attempt.name}.json").read_text())
            if doc.get("inputs") == current_inputs(attempt):
                doc["_scope"] = "HISTORIC_" + ("v2_" + doc["identity"]["df_mod_e0_sha256"][:8] if doc.get("identity") else "v1_UNBOUND_CODE")
                return attempt.name, doc, "historic"
        proc = subprocess.run([sys.executable, "-B", str(HERE / "df_mod_e0.py"), "--replay", str(attempt), "--out", str(target)],
                              env=env, capture_output=True, text=True, timeout=1800)
        if target.is_file():
            doc = json.loads(target.read_text())
            doc["_process_exit"] = proc.returncode
            if proc.returncode != 0:
                doc.setdefault("problems", []).append(f"replay process exited {proc.returncode}: {(proc.stderr or '')[-200:]}")
            target.write_text(json.dumps(doc, indent=1, sort_keys=True, default=float))
            return attempt.name, doc, f"exit {proc.returncode}"
        return attempt.name, {"schema": "df_mod_e0_replay.v2", "attempt": attempt.name, "_process_exit": proc.returncode,
                              "problems": [f"replay process failed: exit {proc.returncode}: {(proc.stderr or '')[-300:]}"]}, f"exit {proc.returncode}"
    docs, notes = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name, doc, note in pool.map(one, attempts):
            docs[name], notes[name] = doc, note
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return {"docs": docs, "notes": notes, "cpu_seconds_children": round(cpu, 3), "workers": workers, "threads_per_replay": threads,
            "identity": identity, "scopes": {n: (d.get("_scope") or "CURRENT_CODE") for n, d in docs.items()}}


# --- local closure -----------------------------------------------------------------------------------------

def local_closure(root: Path, out_dir: Path, tolerance: dict | None = None, replays: bool = True, workers: int = 4,
                  pilot_updates: int | None = None, split: str = "validation", historic_replays: Path | None = None, replay_only: set | None = None) -> dict:
    root, out_dir = Path(root), Path(out_dir)
    tolerance = {**DEFAULT_TOLERANCE, **(tolerance or {})}
    if not (root / "DESIGN.json").is_file():
        raise ClosureRefusal("REFUSED: no DESIGN.json in the root; the population cannot be derived")
    design = json.loads((root / "DESIGN.json").read_text())
    pop = population(design)
    if not (root / "CAMPAIGNS.json").is_file():
        raise ClosureRefusal("REFUSED: no CAMPAIGNS.json; the run's registrations are absent")
    registrations = json.loads((root / "CAMPAIGNS.json").read_text())
    run_ids = {k[: -len(suffix)] for k in registrations for suffix in pop["campaigns"] if k.endswith(suffix)}
    if len(run_ids) != 1:
        raise ClosureRefusal(f"REFUSED: the registrations do not name exactly one run: {sorted(run_ids)}")
    run_id = run_ids.pop()
    report_path = root / ("REPORT.collected.json" if (root / "REPORT.collected.json").is_file() else "REPORT.json")
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    if report is not None and report_path.name == "REPORT.collected.json" and (root / "REPORT.json").is_file():
        first = json.loads((root / "REPORT.json").read_text())
        if first.get("run_id") != report.get("run_id") or first.get("design_sha256") != report.get("design_sha256"):
            raise ClosureRefusal("REFUSED: REPORT.json and REPORT.collected.json disagree on the run or design")
        report["reporting_code_identity"] = report.get("code_identity")   # the collected report was emitted under a later commit
        report["code_identity"] = first.get("code_identity")               # the EXECUTION identity is the first run's (the campaigns' registration)
        for cid, entry in (first.get("cells") or {}).items():         # the first run's own outcomes stay authoritative for its cells
            if entry.get("outcome") != RUN.DELEGATED and cid not in (report.get("cells") or {}):
                report.setdefault("cells", {})[cid] = entry
        report["terminals"] = list(first.get("terminals") or []) + [t for t in report.get("terminals") or [] if t.get("status") != "RESUMED"]
    if report is not None:
        if report.get("run_id") != run_id:
            raise ClosureRefusal(f"REFUSED: REPORT.json is run {report.get('run_id')!r}; the registrations are {run_id!r}")
        if report.get("design_sha256") != design["design_sha256"]:
            raise ClosureRefusal("REFUSED: REPORT.json names another design")
        terminal_ids = [t.get("unit_id") for t in report.get("terminals") or []]
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ClosureRefusal("REFUSED: duplicated terminal in REPORT.json")
        strangers = sorted((set(terminal_ids) | set(report.get("cells") or {})) - set(pop["members"]) - set(pop["pilot_ids"]))
        if strangers:
            raise ClosureRefusal(f"REFUSED: REPORT.json names units that are not members of the design: {strangers}")
    attempts_dir = root / "attempts"
    on_disk = sorted(p.name for p in attempts_dir.iterdir() if p.is_dir()) if attempts_dir.is_dir() else []
    strangers = sorted(set(on_disk) - set(pop["members"]) - set(pop["pilot_ids"]))
    if strangers:
        raise ClosureRefusal(f"REFUSED: attempts that are not members of the design: {strangers}")
    if pilot_updates is None:                                  # the allowance the pilots consumed, read from any pilot's job (v1 or v2 names)
        for pid in pop["pilot_ids"]:
            pj = attempts_dir / pid / "job.json"
            if pj.is_file():
                pilot_updates = json.loads(pj.read_text()).get("max_updates_override")
                break
    units = pop["pilots"] + pop["cells"] + pop.get("inherited", [])
    # replays only for attempts that completed
    replay_docs = {}
    replay_meta = None
    if replays:
        completed = [attempts_dir / u["cell_id"] for u in units if (attempts_dir / u["cell_id"] / "outcome.json").is_file()
                     and json.loads((attempts_dir / u["cell_id"] / "outcome.json").read_text()).get("status") == "COMPLETED"
                     and (attempts_dir / u["cell_id"] / "cell.json").is_file()]
        replay_meta = run_replays(completed, out_dir / "replays", workers=workers, historic=historic_replays, only=replay_only)
        replay_docs = replay_meta["docs"]
    out = {"schema": SCHEMA, "run_id": run_id, "root": str(root), "design_sha256": design["design_sha256"], "successor_of": pop["successor_of"],
           "population": {"cells": len(pop["members"]), "pilots": len(pop["pilot_ids"]), "members": pop["members"], "pilot_ids": pop["pilot_ids"]},
           "tolerance": tolerance, "units": {}, "code_identity_reported": (report or {}).get("code_identity"),
           "reporting_code_identity": (report or {}).get("reporting_code_identity"),
           "replays": None if replay_meta is None else {k: v for k, v in replay_meta.items() if k != "docs"}}
    parent_designs = {}
    for u in units:
        if u.get("inherited_from"):
            inh = u["inherited_from"]
            if inh["design_sha256"] not in parent_designs:
                pd_path = Path(inh["root"]) / "DESIGN.json"
                parent_designs[inh["design_sha256"]] = json.loads(pd_path.read_text()) if pd_path.is_file() else None
            parent = parent_designs[inh["design_sha256"]]
            if parent is None or parent.get("design_sha256") != inh["design_sha256"]:
                raise ClosureRefusal(f"REFUSED: inherited donor {u['cell_id']}: the parent design at {inh['root']} is absent or not {inh['design_sha256'][:12]}")
            job = expected_job(parent, inh["run_id"], u, Path(inh["root"]), None)
        else:
            job = expected_job(design, run_id, u, root, pilot_updates)
        donor = attempts_dir / u["depends_on"] if u.get("depends_on") else None
        entry = verify_attempt(attempts_dir / u["cell_id"], job, replay_docs.get(u["cell_id"]), tolerance, donor)
        entry["role"] = job["role"]
        entry["depends_on"] = u.get("depends_on")
        if u.get("inherited_from"):
            entry["inherited_from"] = u["inherited_from"]
            entry["role"] = "INHERITED"
        out["units"][u["cell_id"]] = entry
    # dependencies: an H3 arm is verified only if its donor is
    for cid, dep in pop["dependencies"].items():
        e, d = out["units"][cid], out["units"].get(dep)
        if e["status"] == VERIFIED and (d is None or d["status"] != VERIFIED):
            e["status"] = NO_DONOR
            e["problems"].append(f"donor {dep} is not a verified attempt ({(d or {}).get('status')})")
    # parent: the report's recorded outcome per member equals the file's
    parent = {}
    if report is not None:
        for cid in pop["members"]:
            rc = (report.get("cells") or {}).get(cid) or {}
            e = out["units"][cid]
            file_mase = (e.get("record") or {}).get("mase", {}).get("validation") if e.get("record") else None
            if rc.get("outcome") == RUN.DELEGATED:
                rc = {}                                                     # never reported after collection: no parent
            equal = bool(rc) and (e["status"] in (VERIFIED, PROBLEMS, NO_DONOR)) and rc.get("mase_validation") is not None and file_mase is not None \
                and abs(rc["mase_validation"] - file_mase) <= 1e-9
            if e["status"] == FAILED and rc and rc.get("outcome") == (e.get("carried") or {}).get("outcome"):
                equal = True
            parent[cid] = {"parent_mase_validation": rc.get("mase_validation"), "file": file_mase, "parent_outcome": rc.get("outcome"), "equal": equal}
            if not equal:
                e["problems"].append("parent: the report's recorded outcome differs from the file (or is absent)")
    out["parent"] = parent
    out["parent_equal"] = bool(parent) and all(v["equal"] for v in parent.values()) if report is not None else None
    counts = {}
    for e in out["units"].values():
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    out["counts"] = counts
    out["not_verified"] = sorted(k for k, e in out["units"].items() if e["status"] != VERIFIED)
    out["replay_scopes"] = {}
    for e in out["units"].values():
        sc = e.get("replay_scope")
        if sc:
            out["replay_scopes"][sc] = out["replay_scopes"].get(sc, 0) + 1
    out["all_verified"] = not out["not_verified"] and (out["parent_equal"] is not False)
    out["closure"] = TOTAL if out["all_verified"] else PARTIAL
    out["denominator"] = {"cells": len(pop["members"]), "pilots": len(pop["pilot_ids"]), "inherited": len(pop.get("inherited", [])), "fixed_by": "DESIGN.json + runner's pilot derivation"}
    usable = {k: {"hypothesis": e["record"]["hypothesis"], "level": e["record"]["level"], "r": e["record"]["r"], "seed": e["record"]["seed"],
                  "arm": e["record"]["arm"], "mase": e["record"]["mase"].get(split)}
              for k, e in out["units"].items() if e["status"] == VERIFIED and e["role"] == "CELL"}
    out["effects_split"] = split
    out["effects_population"] = {"verified_cells": len(usable), "of": len(pop["members"]), "scope": "VERIFIED_MEMBERS_ONLY" if usable else "NONE",
                                 "complete": len(usable) == len(pop["members"])}
    out["effects"] = V.effects(usable) if usable else {}
    out["bootstrap"] = V.bootstrap_effects(usable) if usable else None
    out["effects_test"] = V.effects({k: {**v, "mase": out["units"][k]["record"]["mase"].get("test")} for k, v in usable.items()}) if usable else None
    return out


# --- live closure: accounting population and warehouse content -----------------------------------------------

def _query(url: str, token: str, sql: str) -> list:
    request = urllib.request.Request(f"{url}/api/v1/query?" + urllib.parse.urlencode({"sql": sql}), headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=120) as answer:
        return json.loads(answer.read())["rows"]


def warehouse_terminals(url: str, token: str, campaign_sha256: str) -> dict:
    """Current-generation terminal per unit with its metric rows and artifacts."""
    rows = _query(url, token, f"SELECT unit_id, terminal_sha256, generation, status, reason, costs_json, tags_json, code_identity_json, config_sha256, "
                              f"synthetic_spec_sha256, started_at, finished_at FROM \"main\".\"gov_terminal\" WHERE campaign_sha256 = '{campaign_sha256}' LIMIT 5000")
    current = {}
    for r in rows:
        if r["unit_id"] not in current or int(r["generation"] or 0) > int(current[r["unit_id"]]["generation"] or 0):
            current[r["unit_id"]] = r
    for r in current.values():
        r["metrics"] = _query(url, token, f"SELECT metric, split, horizon, unit, value, std_dev, min_value, max_value FROM \"main\".\"gov_terminal_metric\" "
                                          f"WHERE terminal_sha256 = '{r['terminal_sha256']}' LIMIT 1000")
        r["artifacts"] = _query(url, token, f"SELECT role, sha256, bytes FROM \"main\".\"gov_terminal_artifact\" WHERE terminal_sha256 = '{r['terminal_sha256']}' LIMIT 1000")
    return {"current": current, "rows_all_generations": len(rows)}


def _metric_key(m: dict) -> tuple:
    return (m["metric"], m["split"], int(m["horizon"]) if m.get("horizon") is not None else None, m["unit"],
            None if m.get("value") is None else round(float(m["value"]), 9))


def live_closure(root: Path, local: dict, gov, warehouse_url: str, token: str, backfill_dir: Path | None = None) -> dict:
    """`backfill_dir`: RP13 backfill documents (one per unit); when given, the expected metric rows
    and states of a unit are the run's rows plus the backfilled ones (the successor generation)."""
    root = Path(root)
    backfill = {}
    if backfill_dir is not None:
        for path in Path(backfill_dir).glob("*.json"):
            if path.name == "BACKFILL.json":
                continue
            doc = json.loads(path.read_text())
            if doc.get("schema") == "df_mod_e0_backfill.v1":
                backfill[doc["cell_id"]] = doc
    registrations = json.loads((root / "CAMPAIGNS.json").read_text())
    design = json.loads((root / "DESIGN.json").read_text())
    pop = population(design)
    run_id = local["run_id"]
    out = {"schema": "df_mod_e0_live_closure.v1", "run_id": run_id, "campaigns": {}, "all_equal": True, "population_equal": True, "units": {}}
    for suffix, members in pop["campaigns"].items():
        key = f"{run_id}{suffix}"
        reg = registrations.get(key)
        camp = {"key": key, "registered": bool(reg), "members_by_design": members, "problems": []}
        out["campaigns"][key] = camp
        if not members and not reg:
            camp["note"] = "no member by design (a hypothesis-only successor has no pilots)"
            continue
        if not reg:
            camp["problems"].append("campaign not registered by this run")
            out["population_equal"] = out["all_equal"] = False
            continue
        sha_c = reg["campaign_sha256"]
        status, body = gov.reconcile_campaign(sha_c)
        camp["reconcile"] = {"http": status, **{k: body.get(k) for k in ("missing_units", "accounting_only", "lake_only")}}
        if status != 200:
            camp["problems"].append(f"reconciliation http {status}")
            out["all_equal"] = out["population_equal"] = False
            continue
        if body.get("accounting_only") or body.get("lake_only"):
            camp["problems"].append("accounting and lake diverge")
            out["all_equal"] = False
        wh = warehouse_terminals(warehouse_url, token, sha_c)
        held = wh["current"]
        registered = sorted(set(body.get("missing_units") or []) | set(held))
        camp["population_by_accounting"] = registered
        camp["rows_all_generations"] = wh["rows_all_generations"]
        if registered != sorted(members):
            camp["problems"].append(f"registered population {registered} differs from the design's {sorted(members)}")
            out["population_equal"] = out["all_equal"] = False
        for unit in members:
            e = local["units"].get(unit) or {}
            t = held.get(unit)
            u = {"in_warehouse": t is not None, "problems": []}
            out["units"][unit] = u
            attempt = root / "attempts" / unit
            outcome = json.loads((attempt / "outcome.json").read_text()) if (attempt / "outcome.json").is_file() else None
            if t is None:
                if outcome is not None:
                    u["problems"].append("attempt has an outcome but no terminal in the warehouse")
                elif unit not in (body.get("missing_units") or []):
                    u["problems"].append("neither an attempt nor a registered-missing unit")
                out["all_equal"] &= not u["problems"]
                continue
            if outcome is None:
                u["problems"].append("terminal in the warehouse for a unit without a local attempt")
            else:
                summary = outcome.get("summary") or {}
                expected_status = "COMPLETED" if (outcome.get("status") == "COMPLETED" and e.get("record")) else \
                                  ("FAILED" if summary.get("outcome") == RUN.H.RESOURCE_EXCEEDED else ("INCONCLUSIVE" if outcome.get("status") != "COMPLETED" else "COMPLETED"))
                if t["status"] != expected_status:
                    u["problems"].append(f"status {t['status']} vs local {expected_status}")
                costs = json.loads(t["costs_json"] or "{}")
                lc = summary.get("cost") or {}
                for k in ("cpu_seconds", "wall_seconds"):
                    if lc.get(k) is None or costs.get(k) is None or abs(float(costs[k]) - float(lc[k])) > 1e-6:
                        u["problems"].append(f"cost {k}: warehouse {costs.get(k)} vs local {lc.get(k)}")
                tags = json.loads(t["tags_json"] or "{}")
                if tags.get("output_sha256", "") != (summary.get("output_sha256") or ""):
                    u["problems"].append("output digest in the terminal differs from the local outcome's")
                if tags.get("design_sha256") != design["design_sha256"] or t.get("config_sha256") != design["design_sha256"]:
                    u["problems"].append("design identity in the terminal differs")
                ci = json.loads(t["code_identity_json"] or "{}")
                if local.get("code_identity_reported") and ci != local["code_identity_reported"]:
                    u["problems"].append(f"code identity {ci} vs reported {local['code_identity_reported']}")
                if (attempt / "cell.json").is_file() and outcome.get("status") == "COMPLETED":
                    rec = json.loads((attempt / "cell.json").read_bytes())
                    rows, states = RUN._metrics(rec)
                    bf = backfill.get(unit)
                    if bf is not None:
                        if bf.get("cell_sha256") != hashlib.sha256((attempt / "cell.json").read_bytes()).hexdigest():
                            u["problems"].append("backfill document is not of this attempt's record")
                        known = {(m["metric"], m["split"]) for m in rows}
                        rows = rows + [m for m in bf["rows"] if (m["metric"], m["split"]) not in known]
                        states = {**states, **bf["states"]}
                        u["backfill_rows"] = len(bf["rows"])
                    expected_rows = sorted(_metric_key(m) for m in rows)
                    got_rows = sorted(_metric_key(m) for m in t["metrics"])
                    if expected_rows != got_rows:
                        u["problems"].append(f"metric rows differ: warehouse {len(got_rows)} vs expected {len(expected_rows)}")
                    try:
                        got_states = json.loads(tags.get("metric_states") or "{}")
                    except ValueError:
                        got_states = None
                    if got_states != states:
                        u["problems"].append("metric states differ")
                    u["metric_rows"] = {"warehouse": len(got_rows), "expected": len(expected_rows)}
                u["artifacts"] = len(t["artifacts"])
                u["generation"] = t["generation"]
            out["all_equal"] &= not u["problems"]
    out["covered"] = sum(1 for u in out["units"].values() if u["in_warehouse"])
    return out


# --- CLI -----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None, help="default ROOT/closure_v2; refused if it holds a CLOSE.json")
    parser.add_argument("--no-replay", action="store_true", help="skip the fresh-process replays (every unit then fails the weights check)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=None, help="prediction |diff| tolerance (default 1e-5)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--gov-url", default="http://127.0.0.1:5055")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--warehouse-url", default="http://127.0.0.1:5057")
    parser.add_argument("--token-env", default="DATA_GOV_LAKE_TOKEN")
    parser.add_argument("--backfill-dir", type=Path, default=None, help="RP13 backfill documents whose rows the live terminals must carry")
    parser.add_argument("--historic-replays", type=Path, default=None, help="RP19: earlier replay documents reused under an explicit HISTORIC scope for attempts not in --replay-units")
    parser.add_argument("--replay-units", default=None, help="comma-separated attempt names that must be replayed now under the current code")
    parser.add_argument("--local-from", type=Path, default=None,
                        help="reuse the local closure of a prior CLOSE.json of this very root (the live closure is a separate result)")
    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.root / "closure_v2")
    if (out_dir / "CLOSE.json").exists():
        raise SystemExit(f"REFUSED: {out_dir / 'CLOSE.json'} exists; a closure is never written over")
    out_dir.mkdir(parents=True, exist_ok=True)
    tol = {"prediction_atol": args.tolerance} if args.tolerance is not None else None
    try:
        if args.local_from is not None:
            prior = json.loads(args.local_from.read_text())
            local = prior["local"]
            if Path(local["root"]).resolve() != args.root.resolve() or local["design_sha256"] != json.loads((args.root / "DESIGN.json").read_text())["design_sha256"]:
                raise ClosureRefusal("REFUSED: --local-from is a closure of another root or design")
            local["reused_from"] = str(args.local_from)
        else:
            local = local_closure(args.root, out_dir, tol, replays=not args.no_replay, workers=args.workers, split=args.split,
                                  historic_replays=args.historic_replays, replay_only=set(args.replay_units.split(",")) if args.replay_units else None)
    except ClosureRefusal as e:
        (out_dir / "REFUSAL.json").write_text(json.dumps({"schema": SCHEMA, "closure": REFUSED, "why": e.why}, indent=1) + "\n")
        print(json.dumps({"closure": REFUSED, "why": e.why}))
        return 2
    live = None
    if not args.no_live:
        GR = _load("governed_run")
        if not args.api_key_file:
            raise SystemExit("REFUSED: --api-key-file is required for the live closure (or pass --no-live)")
        token = os.environ.get(args.token_env, "")
        if not token:
            raise SystemExit(f"REFUSED: no warehouse token in ${args.token_env} (or pass --no-live)")
        gov = GR.GovHttp(args.gov_url, GR.load_api_key(args.api_key_file), local["run_id"])
        live = live_closure(args.root, local, gov, args.warehouse_url, token, backfill_dir=args.backfill_dir)
    doc = {"schema": SCHEMA, "local": local, "live": live, "live_requested": not args.no_live,
           "verdict": {"local_closure": local["closure"], "live_all_equal": None if live is None else live["all_equal"],
                       "live_population_equal": None if live is None else live["population_equal"],
                       "note": "local (files) and live (accounting + warehouse) are separate results; neither stands for the other"}}
    (out_dir / "CLOSE.json").write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    summary = {"closure": local["closure"], "all_verified": local["all_verified"], "parent_equal": local["parent_equal"], "counts": local["counts"],
               "not_verified": local["not_verified"][:20], "effects": local["effects"], "effects_population": local["effects_population"],
               "replays": local["replays"], "live": None if live is None else {"all_equal": live["all_equal"], "population_equal": live["population_equal"],
                                                                                 "covered": live["covered"], "problems": {k: v["problems"] for k, v in live["units"].items() if v["problems"]}}}
    print(json.dumps(summary, indent=1, default=str))
    ok = local["closure"] == TOTAL and (live is None or (live["all_equal"] and live["population_equal"]))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
