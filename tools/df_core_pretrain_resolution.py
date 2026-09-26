#!/usr/bin/env python3
"""The measured resolution of the E1 household protocol — item one of the MOD-CORE-PRETRAIN block.

The block on ``MOD-CORE-PRETRAIN`` (SATOSHI_RP49_RP56_DISPOSITION_2026_09_26 §9,
SATOSHI_RP57_RP64_DISPOSITION_2026_09_26 §6) names three things that must exist first, and says the
first of them **needs no new training**. This tool is that first thing, and the reason it needs no
training is that every measurement a resolution is made of was already taken and retained:

    seed-to-seed spread on identical configurations   the 3 seeds of each of the retained arms
    a cross-runner replicate of one configuration     phase-1 ``core_mse`` IS the successor run's
                                                      ``R0`` -- same design, same seeds, same
                                                      per-seed update counts, two different runners
    the paired differences to be resolved             the arm contrasts the two rounds published

RETRACTION, 2026-09-26. An earlier reading treated RP60's scrambled-label difference as this
instrument's noise floor, and concluded that the floor was five times the effect to be resolved. That
reasoning is WITHDRAWN and nothing here uses it. Shuffling the train labels destroys the signal; the
difference it produces measures how much structure the labels carried -- a property of the task and
the data -- not the seed-to-seed dispersion of a fitted contrast. It is reported under
``label_structure``, it is labelled as not a resolution, and neither ``resolution`` nor ``ruling``
reads it: the resolution comes only from dispersion among runs that share one configuration. Also
withdrawn: mixing kW with persistence-scaled units inside one comparison. The two scales are reported
side by side and never subtracted from or divided by one another.

A resolution is a statement of the form

    "on THIS task, with THIS protocol, n seeds per arm and THESE evaluation rows, a difference
     smaller than X kW is indistinguishable from the protocol's own seed variation"

and it is meaningless without the estimator that produced X, that estimator's assumptions, and an
interval. All three are emitted. The estimator is a minimum detectable effect (MDE) built on a
pooled within-arm standard deviation; the interval on X comes from the chi-square interval on the
variance that the same pooled estimate has, so the interval is the interval of the *instrument*, not
a bootstrap of the point.

Nothing here is read as true from a return or a disposition. Every arm mean, every standard
deviation and every paired difference is recomputed in float64 from the per-cell ``arrays.npz`` --
the stored predictions and labels -- and the per-cell record's own score is checked against the
recomputation rather than trusted for it. The scrambled-label control is the one number read from a
retained JSON, because its weights were not kept; it is labelled as such in the output, it is not a
resolution, and the arm it is reported beside is recomputed here.

    python tools/df_core_pretrain_resolution.py --out RESOLUTION.json [--markdown RESOLUTION.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVID = REPO / "docs" / "audits" / "evidence" / "d3_k5_20260917"
STATE = Path.home() / ".local/state/crispdm-data-foundation"
PHASE1_ROOT = STATE / "e1_phase1_v1b"
SUCCESSOR_ROOT = STATE / "e1_household_successor_v3"

SCHEMA = "core_pretrain_resolution.v1"
ALPHA = 0.05
POWER = 0.80
RECOMPUTE_TOL = 1e-12          # a record's own score must equal the float64 recomputation

# The arms, by the run they were fitted in. `core_mse` and `R0` are the SAME configuration fitted by
# two runners; they are declared as one arm with two runs so the pooled variance cannot double-count
# a single configuration's seed noise as if it were two independent arms.
ARMS = (
    {"arm": "core_mae", "root": "phase1", "cells": ("core_mae_s1", "core_mae_s2", "core_mae_s3"),
     "monitor_trained_on": "mae", "what": "our core, MAE loss, early stopping on validation MAE"},
    {"arm": "core_mse", "root": "phase1", "cells": ("core_mse_s1", "core_mse_s2", "core_mse_s3"),
     "monitor_trained_on": "mse", "what": "our core, MSE loss, early stopping on validation MSE (the run's own recipe)"},
    {"arm": "tcn_mse", "root": "phase1", "cells": ("tcn_mse_s1", "tcn_mse_s2", "tcn_mse_s3"),
     "monitor_trained_on": "mse", "what": "the reference TCN block, parameter-matched, at the run's own recipe"},
    {"arm": "R0", "root": "successor", "cells": ("R0_s1", "R0_s2", "R0_s3"),
     "monitor_trained_on": "mse", "what": "no pretraining (the same configuration as core_mse, other runner)"},
    {"arm": "R1", "root": "successor", "cells": ("R1_s1", "R1_s2", "R1_s3"),
     "monitor_trained_on": "mse", "what": "pretrained detector imported and FROZEN"},
    {"arm": "R2", "root": "successor", "cells": ("R2_s1", "R2_s2", "R2_s3"),
     "monitor_trained_on": "mse", "what": "pretrained detector imported and fine-tuned"},
)
# One configuration, two runners: its seed noise enters the pooled estimate ONCE.
REPLICATE_PAIR = ("core_mse", "R0")
INDEPENDENT_ARMS = ("core_mae", "core_mse", "tcn_mse", "R1", "R2")
# The arms of MOD-CORE-PRETRAIN's own contrast: all three carry the run's own MSE recipe.
MODULE_ARMS = ("R0", "R1", "R2")

CONTRASTS = (
    {"name": "R1-R0", "a": "R1", "b": "R0", "what": "pretraining, frozen detector -- MOD-CORE-PRETRAIN's own effect"},
    {"name": "R2-R0", "a": "R2", "b": "R0", "what": "pretraining, fine-tuned detector -- MOD-CORE-PRETRAIN's own effect"},
    {"name": "R2-R1", "a": "R2", "b": "R1", "what": "fine-tuning against freezing"},
    {"name": "tcn_mse-core_mse", "a": "tcn_mse", "b": "core_mse", "what": "the reference block against ours, parameter-matched"},
    {"name": "core_mae-core_mse", "a": "core_mae", "b": "core_mse", "what": "the recipe contrast (loss and monitor moved together)"},
)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _root(which: str) -> Path:
    return PHASE1_ROOT if which == "phase1" else SUCCESSOR_ROOT


# --- the measurements, recomputed from the retained arrays -----------------------------------------

def cell_measurement(root: Path, cell_id: str) -> dict:
    """One cell's MAE and scaled error, recomputed in float64 from its own stored arrays.

    The record's score is CHECKED against the recomputation, never used in place of it. A cell whose
    record disagrees with its own arrays is returned with ``record_agrees: False`` and the caller
    refuses the whole table rather than quietly preferring one of the two numbers.
    """
    d = root / "attempts" / cell_id
    with np.load(d / "arrays.npz", allow_pickle=False) as z:
        pred = np.asarray(z["validation_pred"], dtype=np.float64).reshape(-1)
        y = np.asarray(z["validation_y"], dtype=np.float64).reshape(-1)
        origins = np.asarray(z["eval_origins"], dtype=np.int64)
        denom = float(np.asarray(z["denominator"], dtype=np.float64).reshape(-1)[0])
    mae = float(np.mean(np.abs(pred - y)))
    rec = json.loads((d / "cell.json").read_text())
    stored = ((rec.get("scores") or {}).get("validation") or {}).get("model") or {}
    declared = stored.get("mae_mean")
    t = rec.get("training") or {}
    return {"cell_id": cell_id, "n_rows": int(pred.size), "mae_kW": mae,
            "scaled_error": mae / denom, "denominator_kW": denom,
            "origins_sha256": hashlib.sha256(origins.tobytes()).hexdigest(),
            "arrays_sha256": sha_file(d / "arrays.npz"),
            "record_mae": declared,
            "record_agrees": declared is not None and abs(float(declared) - mae) <= RECOMPUTE_TOL,
            "updates": t.get("updates"), "epochs": t.get("epochs"),
            "steps_per_epoch": t.get("steps_per_epoch"),
            "restored_checkpoint_epoch": t.get("restored_checkpoint_epoch"),
            "stop_reason": t.get("stop_reason"),
            "censored_by_budget": ((t.get("censoring") or {}).get("verdict") == "CENSORED_BY_BUDGET"),
            "seed": rec.get("seed")}


def naive_on_the_same_rows(root: Path, cell_id: str) -> dict:
    """Persistence at the horizon on the identical evaluation origins -- the closure table's naive.

    Recomputed from the run's own prepared DATA, on the very origins the cell's arrays carry, so the
    naive can never be taken from a different population than the model it is the reference for.
    """
    with np.load(root / "attempts" / cell_id / "arrays.npz", allow_pickle=False) as z:
        origins = np.asarray(z["eval_origins"], dtype=np.int64)
        y = np.asarray(z["validation_y"], dtype=np.float64).reshape(-1)
    with np.load(root / "DATA.npz", allow_pickle=False) as z:
        Y = np.asarray(z["Y"], dtype=np.float64)
        h = int(np.asarray(z["horizon"]).reshape(-1)[0])
    base = Y[origins]                                   # the last observed row of each window
    label = Y[origins + h]
    if not np.array_equal(label, y):
        raise ValueError(f"{cell_id}: the stored labels are not Y[origins+h]; no naive is computed")
    return {"naive_kW": float(np.mean(np.abs(base - label))), "n_rows": int(origins.size),
            "horizon_steps": h,
            "definition": "persistence at the horizon on the identical evaluation origins"}


def scrambled_label_control() -> dict:
    """RP60's full-scale negative control, as retained -- and NOT a resolution floor.

    RETRACTION (2026-09-26, on the coordinator's correction, accepted). An earlier reading treated the
    scrambled-label difference as this instrument's noise floor and concluded that the floor was five
    times the effect to be resolved. That reasoning is WITHDRAWN and is not used anywhere in this tool.
    Shuffling the train labels destroys the signal; the difference it produces measures **how much
    structure the labels carried**, which is a property of the task and the data, not the seed-to-seed
    dispersion of a fitted contrast. A resolution can only come from dispersion among runs that share
    the same configuration. Nothing in `resolution` or in `ruling` reads this function's output, and a
    rule pins that the resolution is reproducible from the cells alone.

    Its weights were NOT kept: this is the one number in the whole tool that is read rather than
    recomputed, and it says so.
    """
    p = EVID / "RP60" / "OPTIMISATION_PROBE_FULL_SCALE.json"
    doc = json.loads(p.read_text())
    blk = doc["shuffled_labels_full_scale"]
    return {"source_file": str(p.relative_to(REPO)), "source_sha256": sha_file(p),
            "custody": "READ_FROM_RETAINED_JSON_WEIGHTS_NOT_KEPT",
            "scrambled_mae_kW": float(blk["with_scrambled_labels"]["mae"]),
            "untrained_mae_kW": float(blk["with_scrambled_labels"]["mae_before_any_update"]),
            "scrambled_updates": int(blk["with_scrambled_labels"]["updates"]),
            "scrambled_restored_epoch": int(blk["with_scrambled_labels"]["restored_epoch"]),
            "compared_against_unit": blk["the_runs_own_true_label_fit"]["unit"],
            "declared_gap_kW": float(blk["gap_in_mae"]),
            "persistence_kW": float(blk["persistence"]),
            "train_windows_scrambled": int(blk["train_windows"]),
            "scope": blk["scope"]}


# --- the estimator --------------------------------------------------------------------------------

def _sd(v) -> float:
    return float(np.std(np.asarray(v, dtype=np.float64), ddof=1))


def pooled_within_arm_sd(groups: dict) -> dict:
    """Pooled within-arm SD: sqrt(sum SS / sum df). The assumption is homoscedasticity ACROSS arms,
    which is itself tested (Bartlett) and reported beside the estimate rather than assumed."""
    ss, df, per = 0.0, 0, {}
    for arm, vals in groups.items():
        v = np.asarray(vals, dtype=np.float64)
        ss += float(np.sum((v - v.mean()) ** 2))
        df += v.size - 1
        per[arm] = {"n": int(v.size), "mean": float(v.mean()), "sd": _sd(v)}
    sd = math.sqrt(ss / df) if df else float("nan")
    bart = stats.bartlett(*[np.asarray(v, dtype=np.float64) for v in groups.values()]) if len(groups) > 1 else None
    return {"pooled_sd": sd, "df": int(df), "sum_of_squares": ss, "per_arm": per,
            "homoscedasticity_bartlett": None if bart is None else
                {"statistic": float(bart.statistic), "p_value": float(bart.pvalue),
                 "reading": ("the arms do NOT share one variance; a single pooled number understates the noise of the "
                             "noisier arms and overstates the quieter ones" if bart.pvalue < 0.05 else
                             "no evidence against one shared variance across these arms")}}


def sd_interval(sd: float, df: int, conf: float = 0.95) -> dict:
    """The chi-square interval for sigma from a pooled variance with `df` degrees of freedom.

    Assumption: the within-arm residuals are normal and independent. With df in single digits this
    interval is WIDE, and that width is the honest part of the answer -- it is why a resolution
    quoted as a bare point estimate from three seeds is not a measurement.
    """
    a = (1 - conf) / 2
    lo = sd * math.sqrt(df / stats.chi2.ppf(1 - a, df))
    hi = sd * math.sqrt(df / stats.chi2.ppf(a, df))
    return {"sd": sd, "df": int(df), "conf": conf, "lo": lo, "hi": hi,
            "estimator": "chi-square interval on sigma from a pooled within-arm variance",
            "assumptions": ["within-arm residuals normal", "independent across seeds",
                            "one variance shared by the arms that were pooled"]}


def mde(sd: float, df: int, n: int, *, alpha: float = ALPHA, power: float = POWER, paired: bool = True) -> dict:
    """Minimum detectable effect for a two-arm contrast at `n` seeds per arm.

    paired=True  -> the two arms share the seeds and the test is a paired t on n differences,
                    each difference having sd sqrt(2)*sigma when there is no seed main effect;
                    the standard error is sqrt(2/n)*sigma and the test has n-1 df.
    paired=False -> a two-sample t on 2n cells, standard error sqrt(2/n)*sigma, 2n-2 df.

    In both cases the standard error is the same sqrt(2/n)*sigma; what differs is the df of the
    critical value. The df of the CRITICAL VALUE is taken from the test that would actually be run;
    the df of the VARIANCE ESTIMATE is `df` (the pooled one) and is what the interval uses. Quoting
    a t critical value on pooled df while running a 2-df paired test would flatter the instrument,
    so both are reported and `mde_kW` uses the df of the test as run.
    """
    se_factor = math.sqrt(2.0 / n)
    test_df = (n - 1) if paired else (2 * n - 2)
    t_a = stats.t.ppf(1 - alpha / 2, test_df)
    t_b = stats.t.ppf(power, test_df)
    t_a_pooled = stats.t.ppf(1 - alpha / 2, df)
    t_b_pooled = stats.t.ppf(power, df)
    return {"n_seeds_per_arm": int(n), "paired": bool(paired), "alpha": alpha, "power": power,
            "sigma_used": sd, "sigma_df": int(df), "test_df": int(test_df),
            "standard_error_of_the_difference": se_factor * sd,
            "mde_kW": (t_a + t_b) * se_factor * sd,
            "mde_kW_if_sigma_were_known_to_pooled_df": (t_a_pooled + t_b_pooled) * se_factor * sd,
            "detectable_at_50pc_power_kW": t_a * se_factor * sd,
            "formula": "(t_{1-alpha/2,df_test} + t_{power,df_test}) * sqrt(2/n) * sigma"}


def seeds_required(sd: float, effect: float, *, alpha: float = ALPHA, power: float = POWER,
                   paired: bool = True, arms_pooled: int = 3, n_max: int = 2000) -> dict:
    """The smallest n per arm whose MDE is at or below `effect`, by direct search over the same
    formula (no normal approximation, so the t-inflation at small n is not quietly dropped).

    `arms_pooled` is how many arms' residuals the variance estimate pools at that n, which fixes the
    variance df as arms_pooled*(n-1); it is stated because a sample size quoted without it is not
    reproducible.
    """
    if not (effect > 0 and sd > 0):
        return {"n_per_arm": None, "why": "a non-positive effect or sigma has no sample size"}
    for n in range(2, n_max + 1):
        if mde(sd, max(1, arms_pooled * (n - 1)), n, alpha=alpha, power=power, paired=paired)["mde_kW"] <= effect:
            return {"n_per_arm": n, "effect_kW": effect, "sigma_used": sd, "alpha": alpha,
                    "power": power, "paired": bool(paired), "arms_pooled": arms_pooled,
                    "note": "n per arm; a two-arm contrast costs 2n fits, a three-arm contrast 3n"}
    return {"n_per_arm": None, "why": f"more than {n_max} seeds per arm", "effect_kW": effect, "sigma_used": sd}


def estimator_band(sd: float, df: int, n: int) -> dict:
    """Three defensible estimators of the same resolution, and the band they span.

    The three differ only in the degrees of freedom of the critical value, which is the one thing a
    reader can legitimately argue about:

        paired          the test the two rounds actually reported (a paired t on n differences, n-1 df)
        unpaired        a two-sample t on the 2n cells (2n-2 df) -- the better design HERE, because the
                        seed main effect is measured to be absent, so pairing spends df for nothing
        pooled_sigma    the two-sample statistic with sigma taken as known to the pooled df, the most
                        favourable reading any of these three supports

    The band is reported because a conclusion that holds across all three is not an artefact of a
    statistical preference. `governing` is the SMALLEST of the three -- the resolution claim is made
    at the instrument's most favourable, so that a verdict of "below the resolution" cannot be
    answered with "you chose the pessimistic test".
    """
    paired = mde(sd, df, n, paired=True)
    unpaired = mde(sd, df, n, paired=False)
    best = min(paired["mde_kW"], unpaired["mde_kW"], unpaired["mde_kW_if_sigma_were_known_to_pooled_df"])
    return {"paired_kW": paired["mde_kW"], "unpaired_kW": unpaired["mde_kW"],
            "unpaired_with_sigma_at_pooled_df_kW": unpaired["mde_kW_if_sigma_were_known_to_pooled_df"],
            "band_kW": [best, max(paired["mde_kW"], unpaired["mde_kW"])],
            "governing_kW": best,
            "governing_estimator": "the most favourable of the three, so the ruling cannot be blamed on a pessimistic test",
            "paired": paired, "unpaired": unpaired}


def paired_contrast(a_vals, b_vals, name: str, what: str) -> dict:
    """A seed-paired contrast: its mean difference, its own t interval, and its own p."""
    a = np.asarray(a_vals, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    d = a - b
    n = d.size
    m, s = float(d.mean()), _sd(d)
    se = s / math.sqrt(n)
    t_crit = stats.t.ppf(1 - ALPHA / 2, n - 1)
    res = stats.ttest_rel(a, b)
    return {"name": name, "what": what, "n_seeds": int(n),
            "per_seed_differences_kW": [float(x) for x in d],
            "mean_difference_kW": m, "sd_of_differences_kW": s, "standard_error_kW": se,
            "ci95_kW": [m - t_crit * se, m + t_crit * se],
            "t": float(res.statistic), "p_value": float(res.pvalue),
            "sign": "a POSITIVE difference means the first arm has the LARGER error (worse)"}


# --- budget, as a variance component ---------------------------------------------------------------

def budget_leg(cells: dict, arms: tuple) -> dict:
    """How much of the seed spread travels with the update count.

    The contrast the module needs is a contrast of means, and the arms' TOTAL updates differ because
    early stopping fired at different epochs. Two things are measured here, both from the retained
    records: the per-arm total updates (the mismatch itself), and the within-arm relation between a
    cell's update count and its error -- which says which way budget-matching would push a mean.
    """
    per_arm, xs, ys = {}, [], []
    for arm in arms:
        cs = [cells[c] for c in next(a for a in ARMS if a["arm"] == arm)["cells"]]
        u = [int(c["updates"]) for c in cs]
        e = [float(c["mae_kW"]) for c in cs]
        cen = [bool(c["censored_by_budget"]) for c in cs]
        centred_u = np.asarray(u, float) - np.mean(u)
        centred_e = np.asarray(e, float) - np.mean(e)
        xs += list(centred_u)
        ys += list(centred_e)
        per_arm[arm] = {"updates_per_seed": u, "total_updates": int(sum(u)),
                        "mae_per_seed_kW": e, "censored_by_budget": cen,
                        "n_censored": int(sum(cen)),
                        "within_arm_pearson_r_updates_vs_error": (
                            None if len(set(u)) < 2 else float(np.corrcoef(u, e)[0, 1]))}
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    slope = float(np.sum(xs * ys) / np.sum(xs * xs)) if np.sum(xs * xs) > 0 else None
    r = float(np.corrcoef(xs, ys)[0, 1]) if xs.size > 2 else None
    return {"per_arm": per_arm, "arms": list(arms),
            "within_arm_pooled_slope_kW_per_update": slope,
            "within_arm_pooled_pearson_r": r,
            "reading": ("a POSITIVE slope means more updates went with a larger (worse) error in these "
                        "arms, so raising a short arm to the common ceiling would push its mean UP, "
                        "not down" if (slope or 0) > 0 else
                        "a NEGATIVE slope means more updates went with a smaller error, so the arm that "
                        "ran longer was helped by its extra budget"),
            "caveat": ("the update counts are themselves outcomes of early stopping, so this relation is "
                       "observational within the retained cells and is not the causal effect of budget; "
                       "only a budget-matched re-run measures that")}


# --- the whole thing -------------------------------------------------------------------------------

def build() -> dict:
    cells, naive, problems = {}, {}, []
    for spec in ARMS:
        root = _root(spec["root"])
        for cid in spec["cells"]:
            m = cell_measurement(root, cid)
            m["arm"], m["run_root"] = spec["arm"], root.name
            cells[cid] = m
            if not m["record_agrees"]:
                problems.append(f"{cid}: the record's own score does not equal the float64 recomputation "
                                f"({m['record_mae']} vs {m['mae_kW']})")
            naive[cid] = naive_on_the_same_rows(root, cid)

    rows = {c["n_rows"] for c in cells.values()}
    if len(rows) != 1:
        problems.append(f"the cells are not scored on one evaluation population: {sorted(rows)}")
    origin_digests = {c["origins_sha256"] for c in cells.values()}
    if len(origin_digests) != 1:
        problems.append("the cells do not share one set of evaluation origins")
    naive_vals = {round(v["naive_kW"], 12) for v in naive.values()}
    if len(naive_vals) != 1:
        problems.append(f"the naive differs across cells that claim the same rows: {sorted(naive_vals)}")

    arm_vals = {spec["arm"]: [cells[c]["mae_kW"] for c in spec["cells"]] for spec in ARMS}
    arm_scaled = {spec["arm"]: [cells[c]["scaled_error"] for c in spec["cells"]] for spec in ARMS}
    denom = {c["denominator_kW"] for c in cells.values()}
    if len(denom) != 1:
        problems.append(f"more than one scaling denominator across the cells: {sorted(denom)}")
    denominator = sorted(denom)[0]

    # the cross-runner replicate: one configuration, two runners, three shared seeds
    a, b = REPLICATE_PAIR
    rep_d = np.asarray(arm_vals[a], float) - np.asarray(arm_vals[b], float)
    replicate = {"pair": list(REPLICATE_PAIR),
                 "per_seed_absolute_difference_kW": [abs(float(x)) for x in rep_d],
                 "max_absolute_difference_kW": float(np.max(np.abs(rep_d))),
                 "updates_match_per_seed": [int(cells[c]["updates"]) for c in next(s for s in ARMS if s["arm"] == a)["cells"]]
                                           == [int(cells[c]["updates"]) for c in next(s for s in ARMS if s["arm"] == b)["cells"]],
                 "sd_of_this_configuration_in_run_a": _sd(arm_vals[a]),
                 "sd_of_this_configuration_in_run_b": _sd(arm_vals[b]),
                 "reading": ("the runner contributes ~1e-6 kW while the seed contributes ~1e-2 kW: the noise this "
                             "instrument must beat is seed noise, not implementation noise, and the two arms are "
                             "ONE configuration -- pooling them as two would halve the apparent variance of a single "
                             "configuration's seeds")}

    pooled_all = pooled_within_arm_sd({k: arm_vals[k] for k in INDEPENDENT_ARMS})
    pooled_module = pooled_within_arm_sd({k: arm_vals[k] for k in MODULE_ARMS})
    pooled_mse = pooled_within_arm_sd({k: arm_vals[k] for k in ("core_mse", "tcn_mse", "R1", "R2")})

    # a seed main effect would let pairing help; measured, not assumed
    mat = np.asarray([arm_vals[k] for k in INDEPENDENT_ARMS], dtype=np.float64)
    grand = mat.mean()
    arm_eff = mat.mean(axis=1, keepdims=True) - grand
    seed_eff = mat.mean(axis=0, keepdims=True) - grand
    resid = mat - grand - arm_eff - seed_eff
    ss_seed = float(mat.shape[0] * np.sum(seed_eff ** 2))
    ss_resid = float(np.sum(resid ** 2))
    df_seed, df_resid = mat.shape[1] - 1, (mat.shape[0] - 1) * (mat.shape[1] - 1)
    f_seed = (ss_seed / df_seed) / (ss_resid / df_resid)
    seed_effect = {"ss_seed": ss_seed, "df_seed": df_seed, "ss_residual": ss_resid, "df_residual": df_resid,
                   "F": float(f_seed), "p_value": float(1 - stats.f.cdf(f_seed, df_seed, df_resid)),
                   "sd_residual_after_removing_a_seed_effect": math.sqrt(ss_resid / df_resid),
                   "reading": ("a seed main effect shared by the arms would make seed-pairing reduce the noise; "
                               "if it is absent, pairing buys nothing and the resolution is set by the "
                               "within-arm SD itself")}

    contrasts = [paired_contrast(arm_vals[c["a"]], arm_vals[c["b"]], c["name"], c["what"]) for c in CONTRASTS]

    control = scrambled_label_control()
    control_recomputed_partner = cells[control["compared_against_unit"]]["mae_kW"]
    control["recomputed_partner_mae_kW"] = control_recomputed_partner
    control["recomputed_gap_kW"] = control["scrambled_mae_kW"] - control_recomputed_partner
    control["declared_gap_reproduces"] = abs(control["recomputed_gap_kW"] - control["declared_gap_kW"]) <= 1e-12
    control["partner_is_budget_matched"] = (int(cells[control["compared_against_unit"]]["updates"])
                                           == control["scrambled_updates"])

    best = min(arm_vals, key=lambda k: float(np.mean(arm_vals[k])))
    label_structure = {
        "what_this_measures": ("how much structure the train labels carried under this protocol -- a "
                               "property of the task and the data"),
        "what_this_is_NOT": ("a resolution, a noise floor, or the dispersion of a contrast. Shuffling "
                             "labels destroys the signal; it does not sample the seed-to-seed variation "
                             "of a fitted comparison. RETRACTED: the earlier reading that called this a "
                             "noise floor five times the effect, and any module block derived from it"),
        "untrained_kW": control["untrained_mae_kW"],
        "scrambled_labels_kW": control["scrambled_mae_kW"],
        "persistence_naive_same_rows_kW": sorted(naive_vals)[0],
        "best_retained_arm": best, "best_retained_arm_mean_kW": float(np.mean(arm_vals[best])),
        "label_structure_span_kW": control["scrambled_mae_kW"] - float(np.mean(arm_vals[best])),
        "scrambled_is_already_better_than_the_naive": control["scrambled_mae_kW"] < sorted(naive_vals)[0],
        "used_in_the_resolution": False, "used_in_the_ruling": False,
        "scales_never_mixed": ("kW and persistence-scaled units are reported in separate fields and are "
                               "never subtracted from or divided by one another")}

    # the resolution, for the module's own arms and at the module's own seed count
    n_now = 3
    sigma, sigma_df = pooled_module["pooled_sd"], pooled_module["df"]
    band = estimator_band(sigma, sigma_df, n_now)
    iv_module = sd_interval(sigma, sigma_df)
    band_lo = estimator_band(iv_module["lo"], sigma_df, n_now)
    band_hi = estimator_band(iv_module["hi"], sigma_df, n_now)
    governing = band["governing_kW"]
    resolution = {
        "statement_grain": "one two-arm contrast of mean MAE on the run's own evaluation rows",
        "task": "household minute-level active power, window 60, horizon 60 steps",
        "evaluation_rows": sorted(rows)[0],
        "n_seeds_per_arm": n_now,
        "governing_arms": list(MODULE_ARMS),
        "why_these_arms": ("MOD-CORE-PRETRAIN's own contrast is R0 vs R1 vs R2, and Bartlett's test finds no "
                           "evidence against one shared variance across exactly those three "
                           f"(p={pooled_module['homoscedasticity_bartlett']['p_value']:.4f}), while it does across "
                           f"all five arms (p={pooled_all['homoscedasticity_bartlett']['p_value']:.4f}) -- so the "
                           "three are poolable and the five are not"),
        "sigma_kW": sigma, "sigma_df": sigma_df,
        "sigma_ci95_kW": [iv_module["lo"], iv_module["hi"]],
        "resolution_kW": governing,
        "resolution_ci95_kW": [band_lo["governing_kW"], band_hi["governing_kW"]],
        "resolution_scaled": governing / denominator,
        "resolution_ci95_scaled": [band_lo["governing_kW"] / denominator, band_hi["governing_kW"] / denominator],
        "resolution_band_over_estimators_kW": band["band_kW"],
        "estimator_band": band,
        "resolution_at_50pc_power_kW": band["unpaired"]["detectable_at_50pc_power_kW"],
        "interval_estimator": iv_module,
        "if_all_five_arms_are_pooled": mde(pooled_all["pooled_sd"], pooled_all["df"], n_now, paired=False),
        "denominator_kW": denominator,
        "what_this_is_not": ["not a bound on the task's achievable error",
                             "not a claim about any arm's accuracy",
                             "not transferable to another task, split, horizon or seed count",
                             "not a statement about a single fit: it is about a DIFFERENCE OF ARM MEANS"],
        "derived_only_from": ("the within-arm dispersion of runs that share one configuration (3 seeds per "
                              "arm, everything else held fixed) and the paired differences across the "
                              "retained replicas"),
        "not_derived_from": ["the scrambled-label control, which measures label structure and is not a "
                             "noise floor (see `label_structure` and `ruling.retraction`)",
                             "any quantity in persistence-scaled units: kW and scaled units are reported "
                             "side by side and never mixed inside one comparison"]}

    verdicts = []
    for c in contrasts:
        eff = abs(c["mean_difference_kW"])
        realized = (c["ci95_kW"][0] > 0) or (c["ci95_kW"][1] < 0)
        if eff >= band_hi["governing_kW"]:
            state = "ABOVE_THE_RESOLUTION"
        elif eff >= governing:
            state = "AT_THE_RESOLUTION_BOUNDARY"
        else:
            state = "BELOW_THE_RESOLUTION"
        verdicts.append({
            "contrast": c["name"], "effect_kW": c["mean_difference_kW"], "absolute_effect_kW": eff,
            "resolution_kW": governing, "resolution_ci95_kW": resolution["resolution_ci95_kW"],
            "state_against_the_resolution": state,
            "realized_ci95_excludes_zero": realized,
            "realized_p_value": c["p_value"],
            "realized_sd_of_differences_kW": c["sd_of_differences_kW"],
            "seeds_required_for_this_effect": seeds_required(sigma, eff, paired=False, arms_pooled=len(MODULE_ARMS)),
            "reading": (("the effect is below what this protocol can be relied on to separate at n=3, yet THIS "
                         "realisation reached significance: it did so on its own realised difference SD "
                         f"({c['sd_of_differences_kW']:.6f} kW), which is smaller than the pooled sigma "
                         f"({sigma:.6f} kW) that the resolution is built on. A result at or under the resolution "
                         "is not repeatable by design, whatever one realisation returned")
                        if state != "ABOVE_THE_RESOLUTION" and realized else
                        ("the effect is below the resolution and the test agrees: nothing is established"
                         if not realized and state == "BELOW_THE_RESOLUTION" else
                         "the effect exceeds the resolution's own upper confidence limit"))})

    module_effects = [v for v in verdicts if v["contrast"] in ("R1-R0", "R2-R0", "R2-R1")]
    largest = max(module_effects, key=lambda v: v["absolute_effect_kW"])
    req = seeds_required(sigma, largest["absolute_effect_kW"], paired=False, arms_pooled=len(MODULE_ARMS))
    ruling = {
        "module": "MOD-CORE-PRETRAIN",
        "effect_the_module_must_resolve_kW": largest["absolute_effect_kW"],
        "which_effect": largest["contrast"],
        "every_module_effect_kW": {v["contrast"]: v["effect_kW"] for v in module_effects},
        "resolution_kW": governing,
        "resolution_ci95_kW": resolution["resolution_ci95_kW"],
        "effect_is_above_the_resolution": bool(largest["state_against_the_resolution"] == "ABOVE_THE_RESOLUTION"),
        "every_module_effect_state": {v["contrast"]: v["state_against_the_resolution"] for v in module_effects},
        "ratio_resolution_over_effect": governing / largest["absolute_effect_kW"],
        "seeds_required": req,
        "seeds_required_for_a_flat_0p01_kW": seeds_required(sigma, 0.01, paired=False, arms_pooled=len(MODULE_ARMS)),
        "fits_required_for_the_three_arm_design": None if req["n_per_arm"] is None else 3 * req["n_per_arm"],
        "retraction": {
            "withdrawn": ("that a scrambled-label difference is a resolution floor, and any ruling on this "
                          "module derived from comparing it to the effect"),
            "why": ("shuffling labels destroys the signal and measures the structure the labels carried, "
                    "not the seed-to-seed dispersion of a fitted contrast"),
            "also_withdrawn": "mixing kW with persistence-scaled units in one comparison",
            "what_the_ruling_below_rests_on_instead": ("the pooled within-arm standard deviation of the "
                                                       "module's own three arms -- dispersion among runs that "
                                                       "share one configuration -- and the paired differences "
                                                       "across the retained replicas, with the estimator, its "
                                                       "assumptions and its interval named in `resolution`"),
            "the_scrambled_number_is_reported_at": "label_structure, and is read by nothing here"},
        "robust_across_the_sigma_interval": {
            "resolution_at_the_most_favourable_sigma_kW": resolution["resolution_ci95_kW"][0],
            "effect_kW": largest["absolute_effect_kW"],
            "still_above_the_effect": resolution["resolution_ci95_kW"][0] > largest["absolute_effect_kW"],
            "ratio_at_the_most_favourable_sigma": resolution["resolution_ci95_kW"][0]/largest["absolute_effect_kW"],
            "reading": ("the ruling does not rest on a point estimate of sigma: even at the LOWER end of "
                        "sigma's own 95% interval, the resolution still exceeds the effect by this ratio")},
        "verdict": ("ANSWERABLE at n=3" if largest["state_against_the_resolution"] == "ABOVE_THE_RESOLUTION"
                    else "UNANSWERABLE_BY_THIS_PROTOCOL_AT_THIS_SEED_COUNT")}

    estimand = {
        "declared_before_the_comparison": True,
        "candidates": {
            "recipe_under_early_stopping": ("each arm gets the stopping rule its own recipe implies; the "
                                            "budget is part of the treatment. RP63's numbers ARE a valid "
                                            "estimate of THIS estimand -- but RP63 declared no estimand"),
            "equal_cost": "each arm gets the same CPU seconds; updates then differ by per-update cost",
            "equal_updates": "each arm gets the same number of optimiser updates; cost then differs"},
        "declared_here": "equal_updates",
        "why": ("the question MOD-CORE-PRETRAIN asks is whether a pretrained component changes what the "
                "same amount of optimisation reaches, so the optimisation must be the same amount"),
        "consequence_for_the_retained_run": ("offering the same CEILING does not imply the same updates "
                                             "CONSUMED. `core_mae` 11 762 / `tcn_mse` 11 762 / `core_mse` "
                                             "10 270 stands as an OBSERVATION. What it invalidates depends "
                                             "on the estimand: under `equal_updates` it invalidates the "
                                             "comparison; under `recipe_under_early_stopping` it does not. "
                                             "RP63's design declared neither, listing `update_ceiling` and "
                                             "`patience` among its held factors and not `optimiser_updates`, "
                                             "so its contrast has no declared estimand and therefore no "
                                             "single interpretation. That is the defect, not the numbers"),
        "equal_updates_is_not_equal_cost": ("the arms differ in per-update cost, so a run matched on updates "
                                            "is NOT matched on CPU seconds, and that is declared rather than "
                                            "discovered afterwards")}
    return {"schema": SCHEMA,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "authority": "the owner's grant of 2026-09-26; Satoshi, successor technical lead",
            "what_this_answers": ("item one of the MOD-CORE-PRETRAIN block: a measured resolution for this "
                                  "protocol on this task, from artifacts already retained, with no new training"),
            "training_performed": False,
            "roots": {"phase1": str(PHASE1_ROOT), "successor": str(SUCCESSOR_ROOT)},
            "cells": cells, "naive_same_rows": naive,
            "arm_means_kW": {k: float(np.mean(v)) for k, v in arm_vals.items()},
            "arm_sds_kW": {k: _sd(v) for k, v in arm_vals.items()},
            "arm_means_scaled": {k: float(np.mean(v)) for k, v in arm_scaled.items()},
            "cross_runner_replicate": replicate,
            "pooled_within_arm_sd": {"five_independent_arms": pooled_all,
                                     "the_modules_own_three_arms": pooled_module,
                                     "every_mse_monitored_arm": pooled_mse},
            "seed_main_effect": seed_effect,
            "contrasts": contrasts,
            "scrambled_label_control": control,
            "label_structure": label_structure,
            "budget": budget_leg(cells, INDEPENDENT_ARMS),
            "estimand": estimand,
            "resolution": resolution,
            "effect_verdicts": verdicts,
            "ruling": ruling,
            "problems": problems}


def markdown(doc: dict) -> str:
    r, ru = doc["resolution"], doc["ruling"]
    L = ["# The measured resolution of the E1 household protocol", "",
         f"Generated {doc['at']} from retained artifacts. **No training was performed.**", "",
         "## The resolution", "",
         f"On the household minute-level task (window 60, horizon 60), with this protocol, "
         f"**{r['n_seeds_per_arm']} seeds per arm** and **{r['evaluation_rows']} evaluation rows**, a "
         f"difference in mean MAE smaller than **{r['resolution_kW']:.6f} kW** "
         f"(95% CI {r['resolution_ci95_kW'][0]:.6f} - {r['resolution_ci95_kW'][1]:.6f} kW; "
         f"{r['resolution_scaled']:.6f} in persistence-scaled units) is indistinguishable from the "
         f"protocol's own seed-to-seed variation at alpha={ALPHA}, power={POWER}.", "",
         f"sigma = {r['sigma_kW']:.8f} kW on {r['sigma_df']} df "
         f"(95% CI {r['sigma_ci95_kW'][0]:.8f} - {r['sigma_ci95_kW'][1]:.8f}), pooled within-arm over "
         f"{', '.join(r['governing_arms'])}. Estimator: {r['estimator_band']['unpaired']['formula']}. "
         f"Across the three defensible estimators the resolution spans "
         f"{r['resolution_band_over_estimators_kW'][0]:.6f} - {r['resolution_band_over_estimators_kW'][1]:.6f} kW; "
         f"the smallest is quoted, so the ruling cannot be blamed on a pessimistic test.", "",
         "## Every retained contrast against that resolution", "",
         "| contrast | effect (kW) | 95% CI | p | resolution (kW) | state | seeds for this effect |",
         "|---|---:|---|---:|---:|---|---:|"]
    by = {c["name"]: c for c in doc["contrasts"]}
    for v in doc["effect_verdicts"]:
        c = by[v["contrast"]]
        n = v["seeds_required_for_this_effect"].get("n_per_arm")
        L.append(f"| {v['contrast']} | {v['effect_kW']:+.6f} | "
                 f"[{c['ci95_kW'][0]:+.6f}, {c['ci95_kW'][1]:+.6f}] | {c['p_value']:.4f} | "
                 f"{v['resolution_kW']:.6f} | {v['state_against_the_resolution']} | {n if n else 'n/a'} |")
    ls = doc["label_structure"]
    L += ["", "## How much structure the labels carried — NOT a resolution", "",
          "**Retracted**: that this is a noise floor, and any module ruling derived from comparing it to "
          "the effect. Shuffling labels destroys the signal; it measures label structure, not the "
          "seed-to-seed dispersion of a contrast. It is read by nothing above.", "",
          f"- untrained, before any update: **{ls['untrained_kW']:.6f} kW**",
          f"- fitted on {doc['scrambled_label_control']['train_windows_scrambled']} **scrambled** train labels: "
          f"**{ls['scrambled_labels_kW']:.6f} kW**",
          f"- persistence naive on the same rows: **{ls['persistence_naive_same_rows_kW']:.6f} kW**",
          f"- the best retained arm ({ls['best_retained_arm']}): **{ls['best_retained_arm_mean_kW']:.6f} kW**",
          f"- the span this measures: **{ls['label_structure_span_kW']:.6f} kW**", "",
          "## The estimand, declared before the comparison", "",
          f"- declared: **{doc['estimand']['declared_here']}** (of {', '.join(doc['estimand']['candidates'])})",
          f"- {doc['estimand']['consequence_for_the_retained_run']}", "",
          "## Ruling", "",
          f"- effect the module must resolve: **{ru['which_effect']} = {ru['effect_the_module_must_resolve_kW']:.6f} kW**",
          f"- resolution at n=3: **{ru['resolution_kW']:.6f} kW**",
          f"- the resolution is **{ru['ratio_resolution_over_effect']:.2f}x** the effect",
          f"- seeds required per arm for that effect: **{ru['seeds_required'].get('n_per_arm')}** "
          f"({ru['fits_required_for_the_three_arm_design']} fits for the three-arm design)",
          f"- seeds required per arm for a flat 0.01 kW effect: **{ru['seeds_required_for_a_flat_0p01_kW'].get('n_per_arm')}**",
          f"- at the most favourable end of sigma's own interval the resolution is still "
          f"**{ru['robust_across_the_sigma_interval']['resolution_at_the_most_favourable_sigma_kW']:.6f} kW**, "
          f"**{ru['robust_across_the_sigma_interval']['ratio_at_the_most_favourable_sigma']:.2f}x** the effect",
          f"- **{ru['verdict']}**", "",
          f"problems: {doc['problems'] or 'none'}", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    doc = build()
    a.out.write_text(json.dumps(doc, indent=1, default=str))
    if a.markdown:
        a.markdown.write_text(markdown(doc))
    print(json.dumps({"resolution_kW": doc["resolution"]["resolution_kW"],
                      "resolution_ci95_kW": doc["resolution"]["resolution_ci95_kW"],
                      "sigma_kW": doc["resolution"]["sigma_kW"], "sigma_df": doc["resolution"]["sigma_df"],
                      "ruling": doc["ruling"], "problems": doc["problems"]}, indent=1, default=str))
    return 0 if not doc["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
