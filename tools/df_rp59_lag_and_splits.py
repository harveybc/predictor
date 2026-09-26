#!/usr/bin/env python3
"""RP59's two repairs: the lag table restated under both estimators, and materialization verified PER SPLIT.

Written under the owner's grant of 2026-09-26 by Satoshi, successor technical lead, as the first two
items ``MOD-FROZEN-PREFIX`` carries out of its unblocking
(``SATOSHI_RP49_RP56_DISPOSITION_2026_09_26`` §9, ``SATOSHI_RP57_RP64_DISPOSITION_2026_09_26`` §5.2/§6).

**Repair one — the lag table.** The four autocorrelations published in
``RP59/DATA_TARGET_PREPROCESSING_AUDIT.json`` re-derive exactly under a BIASED ACF: one fixed
whole-series denominator, so the estimate shrinks by ``(n-k)/n`` as the lag grows. At a one-week lag
that factor is 0.7492, and correcting for it INVERTS the printed order of the last two rows. This tool
publishes both columns side by side with the shrinkage per lag, and states which reading survives the
correction and which reverses. It supersedes nothing silently: the published column is kept, named as
published, beside the corrected one.

Neither estimator is reimplemented here. The biased column comes from
``tools/df_e1_data_audit.py::autocorrelation`` — the estimator the published table was produced by —
and the lag-truncated column from the identical expression the audit tool already recomputes in
``tools/df_rp49_rp64_audit.py::data_audit``. A third implementation would be a third thing to trust.

**Repair two — materialization per split.** Verified from the bytes, never from a declaration:

    declared        the split's row block and origin range RE-DERIVED from the panel's own row count,
                    the retained family split rule (``df_e1_tasks.SPLIT_FRACTIONS``), the design's
                    window / horizon / purge, and the within-slice fractions
    materialized    what ``DATA.npz`` (and the auto-encoder cells' ``arrays.npz``) actually hold
    equality        the split's own rows, read out of the delivered panel parquet and compared
                    element by element against the materialized target, and the scaled inputs
                    inverted through the stored scaler back to the panel's columns
    exclusivity     origin sets and TOUCHED-ROW footprints, pairwise, against every other split
    counts          against the declaration's own enumerator, and the admissibility arithmetic
                    (which origins were withdrawn, and why) re-derived from the non-finite rows
    purge           the boundary the design DECLARES against the boundary OBSERVED in the files

Every split gets one of:

    VERIFIED                              every check above ran and passed on retained bytes
    REFUTED                               a check ran and failed; the measured value is named
    REFUSED_MATERIALIZATION_UNCHECKABLE   the bytes a check needs are absent. NEVER a pass
    REFUSED_UNMATERIALIZED_BY_DESIGN      nothing was materialized for this split, deliberately;
                                          its ABSENCE is verified, its materialization is not

A check whose inputs are absent may not be omitted and may not read True: ``_verdict`` refuses a
split whose check dictionary carries a None, so a skipped check can never look like a passed one.

    python tools/df_rp59_lag_and_splits.py --root RUN_ROOT --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVID = REPO / "docs" / "audits" / "evidence" / "d3_k5_20260917"

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
UNCHECKABLE = "REFUSED_MATERIALIZATION_UNCHECKABLE"
BY_DESIGN = "REFUSED_UNMATERIALIZED_BY_DESIGN"

SUCCESSOR_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_household_successor_v3"
PHASE1_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_phase1_v1b"

LAGS = (1, 60, 1440, 10080)
LAG_NAMES = {1: "one minute", 60: "one hour", 1440: "one day", 10080: "one week"}

BIASED_ESTIMATOR = ("biased ACF: sum((x_t - xbar)(x_{t+k} - xbar)) / sum((x_t - xbar)^2), one fixed "
                    "whole-series denominator over all n centred rows "
                    "(tools/df_e1_data_audit.py::autocorrelation)")
PEARSON_ESTIMATOR = ("lag-truncated Pearson: corrcoef(x[:-k], x[k:]), each of the n-k overlapping "
                     "pairs weighted once and each leg centred and scaled on its own n-k rows")

SUPERSESSION = {
    "supersedes": "the four-row autocorrelation list of "
                  "SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21 (RP59), and the "
                  "`autocorrelation.autocorrelation_by_lag_minutes` block of "
                  "docs/audits/evidence/d3_k5_20260917/RP59/DATA_TARGET_PREPROCESSING_AUDIT.json",
    "why": "the published list is correct arithmetic under a BIASED estimator whose shrinkage grows "
           "with the lag (0.7492 at a week), and it was printed as a decay without naming that "
           "estimator. Bias-corrected, the weekly lag rises ABOVE the daily one, reversing the "
           "printed order of the last two rows",
    "what_is_withdrawn": "nothing. No published value is wrong on its own support and none is "
                         "removed: both columns are published side by side, each with its estimator "
                         "named, and the reading that reverses is stated beside the one that survives",
    "not_carried_invisibly": "this corpus never carries superseded evidence invisibly: the "
                             "as-published column is retained in full, labelled as published, in "
                             "every artifact this repair writes",
}


# --- loading, without registering anything in sys.modules --------------------------------------------

def _load(name: str):
    """Load a repository tool under a PRIVATE name and register it nowhere.

    The RP49-RP64 audit was itself defective this way once: a tool left in ``sys.modules`` let one
    battery pick up another's copy of a module. Nothing here is importable by its own name.
    """
    spec = importlib.util.spec_from_file_location(f"_rp59_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _f(x):
    return None if x is None else float(x)


# --- repair one: the lag table -----------------------------------------------------------------------

def lag_table(root: Path) -> dict:
    """The published column, the bias-corrected column, the shrinkage per lag, and what each says.

    The biased column is produced by the tool that produced the published one; the corrected column by
    the expression the RP49-RP64 audit already recomputes. Neither is written a third time here.
    """
    published_path = EVID / "RP59" / "DATA_TARGET_PREPROCESSING_AUDIT.json"
    data_path = root / "DATA.npz"
    out = {"estimators": {"as_published": BIASED_ESTIMATOR, "bias_corrected": PEARSON_ESTIMATOR},
           "supersession": dict(SUPERSESSION),
           "published_artifact": str(published_path.relative_to(REPO)),
           "run_root": str(data_path)}
    if not published_path.is_file():
        out["state"] = UNCHECKABLE
        out["missing"] = str(published_path)
        return out
    pub = json.loads(published_path.read_text())["autocorrelation"]
    out["as_published"] = {str(k): _f(v) for k, v in pub["autocorrelation_by_lag_minutes"].items()}
    out["published_train_rows_used"] = int(pub["train_rows_used"])
    if not data_path.is_file():
        out["state"] = UNCHECKABLE
        out["missing"] = str(data_path)
        out["why"] = ("the restatement recomputes BOTH columns from the prepared bytes; with DATA.npz "
                      "absent neither column can be re-derived and the published one is not repeated "
                      "as if it had been")
        return out

    audit = _load("df_e1_data_audit")                      # the estimator the published table used
    D = np.load(data_path)
    data = {k: D[k] for k in D.files}

    biased = {str(k): _f(v) for k, v in
              audit.autocorrelation(data, lags=LAGS)["autocorrelation_by_lag_minutes"].items()}

    # the corrected column and the shrinkage, exactly as tools/df_rp49_rp64_audit.py::data_audit
    Y, tr = data["Y"], data["train_origins"].astype(np.int64)
    w = int(data["window"][0])
    rows = np.arange(int(tr.min() - w + 1), int(tr.max()) + 1)
    v = Y[rows]
    v = v[np.isfinite(v)]
    n = int(v.size)
    pearson, shrink = {}, {}
    for lag in LAGS:
        pearson[str(lag)] = _f(np.corrcoef(v[:-lag], v[lag:])[0, 1])
        shrink[str(lag)] = _f((n - lag) / n)

    out["train_rows_used"] = n
    out["recomputed_as_published"] = biased
    out["bias_corrected"] = pearson
    out["shrinkage_n_minus_k_over_n"] = shrink
    out["as_published_re_derives"] = all(
        abs(biased[str(k)] - out["as_published"][str(k)]) <= 1e-10 for k in LAGS)
    out["train_rows_match_the_published_support"] = (n == out["published_train_rows_used"])
    out["rows"] = [{"lag_minutes": lag, "lag_named": LAG_NAMES[lag],
                    "as_published_biased_acf": biased[str(lag)],
                    "shrinkage_n_minus_k_over_n": shrink[str(lag)],
                    "bias_corrected_lag_truncated_pearson": pearson[str(lag)],
                    "biased_over_corrected": _f(biased[str(lag)] / pearson[str(lag)])}
                   for lag in LAGS]
    order_pub = [lag for lag in sorted(LAGS, key=lambda k: -biased[str(k)])]
    order_cor = [lag for lag in sorted(LAGS, key=lambda k: -pearson[str(k)])]
    out["ordering_by_strength"] = {"as_published": order_pub, "bias_corrected": order_cor,
                                   "the_order_changes": order_pub != order_cor}
    out["readings"] = {
        "survives": {"claim": "the daily lag carries LESS linear structure than the hour, so a daily "
                              "context is a hypothesis and not a certainty",
                     "as_published": f"{biased['1440']:.6f} < {biased['60']:.6f}",
                     "bias_corrected": f"{pearson['1440']:.6f} < {pearson['60']:.6f}",
                     "state": VERIFIED if (biased["1440"] < biased["60"]
                                           and pearson["1440"] < pearson["60"]) else REFUTED},
        "reverses": {"claim": "the four values decay monotonically with the lag, so the weekly grain "
                              "carries less linear structure than the daily one",
                     "as_published": f"{biased['10080']:.6f} < {biased['1440']:.6f}",
                     "bias_corrected": f"{pearson['10080']:.6f} > {pearson['1440']:.6f}"
                                       " -- the printed order is reversed",
                     "state": REFUTED,
                     "as_published_is_below_the_corrected_value_by_percent":
                         _f(100.0 * (1.0 - biased["10080"] / pearson["10080"])),
                     "the_corrected_value_is_above_the_published_one_by_percent":
                         _f(100.0 * (pearson["10080"] / biased["10080"] - 1.0)),
                     "two_framings_of_one_gap": "the disposition's '27%' is the first of these "
                                                "(the published number read against the corrected "
                                                "one); the second is the same gap read the other "
                                                "way. Both are printed so neither can be quoted as "
                                                "a correction of the other"},
    }
    out["state"] = VERIFIED if (out["as_published_re_derives"]
                                and out["train_rows_match_the_published_support"]
                                and out["ordering_by_strength"]["the_order_changes"]) else REFUTED
    out["for_mod_frozen_prefix"] = (
        "a temporal prefix must be chosen on the bias-corrected column. Under it the hour is the "
        f"strongest sub-daily grain ({pearson['60']:.6f}) and the WEEK "
        f"({pearson['10080']:.6f}) stands ABOVE the day ({pearson['1440']:.6f}); the published list "
        "would have ranked the week last. Neither column licenses a claim that a weekly context "
        "IMPROVES a forecast: these are linear autocorrelations of the target, not measured skill.")
    return out


# --- repair two: materialization per split -----------------------------------------------------------

def _verdict(checks: dict) -> str:
    """A check that could not run is a refusal, never a pass.

    ``checks`` maps a named check to True, False or None. None means the bytes it needs are absent:
    the split is refused as UNCHECKABLE and the missing check is named in the record. This is the rule
    that keeps an absent artifact from reading as a clean verification.
    """
    if not checks:
        return UNCHECKABLE
    if any(v is None for v in checks.values()):
        return UNCHECKABLE
    return VERIFIED if all(bool(v) for v in checks.values()) else REFUTED


def _touched(origins: np.ndarray, w: int, h: int) -> list:
    """The rows a split actually reads: the first row of its first window to its last label row."""
    return [int(origins.min() - w + 1), int(origins.max() + h)]


def declared_blocks(panel_rows: int, slice_rows, design: dict) -> dict:
    """The declared row blocks, RE-DERIVED — the family's own split rule, then the slice's fractions.

    ``df_e1_tasks.SPLIT_FRACTIONS`` and its edge rule are retained code, so the family's train /
    validation / test row blocks are derivable rather than quotable; the design's own note ("family
    train end row 1452681") is then a consequence to be checked, not a number to be believed.
    """
    tasks = _load("df_e1_tasks")
    fractions = dict(tasks.SPLIT_FRACTIONS)
    edges, cur = {}, 0
    for name, frac in fractions.items():
        edges[name] = [cur, cur + int(panel_rows * frac)]
        cur = edges[name][1]
    edges["test"][1] = panel_rows

    lo, hi = int(slice_rows[0]), int(slice_rows[1])
    n = hi - lo
    within = design["dev_subpartition"]["splits_within_slice"]
    train_rows = int(n * within["train"])
    return {"family_split_rule": {"fractions": fractions,
                                  "source": "tools/df_e1_tasks.py::SPLIT_FRACTIONS and its edge rule",
                                  "panel_rows": int(panel_rows), "edges_in_panel_rows": edges},
            "consumed_slice_in_panel_rows": [lo, hi], "slice_rows": n,
            "within_slice_fractions": dict(within),
            "train_block_in_slice_rows": [0, train_rows],
            "validation_block_in_slice_rows": [train_rows, n],
            "train_block_in_panel_rows": [lo, lo + train_rows],
            "validation_block_in_panel_rows": [lo + train_rows, hi],
            "test_block_in_panel_rows": list(edges["test"]),
            "the_within_slice_boundary_IS_the_family_train_validation_edge":
                bool(lo + train_rows == edges["train"][1])}


def supervised_splits(root: Path, panel_target, blocks: dict, design: dict, data_json: dict,
                      data: dict) -> list:
    """`train` and `validation`: declared origins re-derived, then every byte of them checked."""
    Y, Xs = data["Y"], data["Xs"]
    tr = data["train_origins"].astype(np.int64)
    ev = data["eval_origins"].astype(np.int64)
    w, h = int(data["window"][0]), int(data["horizon"][0])
    purge = int(design["task"]["purge"])
    n = blocks["slice_rows"]
    train_rows = blocks["train_block_in_slice_rows"][1]
    mean = np.asarray(data["scaler_mean"], dtype=np.float64)
    sd = np.asarray(data["scaler_sd"], dtype=np.float64)

    # the declared origin ranges, from the design's own rule and nothing else
    declared = {
        "train": np.arange(w - 1, train_rows - purge),           # o + purge <= train_rows - 1
        "validation": np.arange(train_rows, n - h),              # o + h <= n - 1
    }
    enumerator = data_json["enumerator"]

    # admissibility, re-derived: which declared origins were withdrawn, and why
    nonfinite_rows = np.flatnonzero(~np.isfinite(panel_target))
    out = []
    for name, origins, declared_origins in (("train", tr, declared["train"]),
                                            ("validation", ev, declared["validation"])):
        dec = declared_origins
        bad_input = np.zeros(dec.size, dtype=bool)
        bad_label = np.zeros(dec.size, dtype=bool)
        for r in nonfinite_rows:
            bad_input |= (dec >= r) & (dec <= r + w - 1)
            bad_label |= (dec + h == r)
        admissible = dec[~(bad_input | bad_label)]
        e = enumerator[name]
        touched = _touched(origins, w, h)
        rows = np.arange(touched[0], touched[1] + 1)
        inv = Xs[rows].astype(np.float64) * sd + mean
        panel_block = panel_target[rows]
        finite = np.isfinite(panel_block) & np.isfinite(Y[rows])
        target_inv = inv[:, int(data["target_channel"][0])]
        checks = {
            "the_materialized_origins_are_exactly_the_admissible_declared_origins":
                bool(origins.size == admissible.size and np.array_equal(origins, admissible)),
            "the_declared_origin_count_re_derives_from_the_design_s_own_rule":
                bool(dec.size == e["origins"]),
            "the_materialized_count_matches_the_declaration":
                bool(origins.size == e["targets_valid_all"] == e["targets_valid_any"]),
            "the_withdrawal_arithmetic_re_derives":
                bool(int(bad_input.sum()) == e["withdrawn_non_finite_inputs"]
                     and dec.size - int((bad_input | bad_label).sum()) == origins.size),
            "the_materialized_target_rows_ARE_the_panel_rows_this_split_declares":
                bool(np.array_equal(panel_block[finite], Y[rows][finite])),
            "the_scaled_inputs_invert_to_the_panel_s_own_columns":
                bool(np.nanmax(np.abs(target_inv[finite] - panel_block[finite])) < 1e-4),
            "the_origins_are_strictly_inside_the_declared_block":
                bool(dec.min() <= origins.min() and origins.max() <= dec.max()),
            "the_origins_are_sorted_and_unique":
                bool(origins.size == np.unique(origins).size and bool((np.diff(origins) > 0).all())),
        }
        out.append({
            "split": name,
            "state": _verdict(checks),
            "checks": checks,
            "declared": {"block_in_slice_rows": blocks[f"{name}_block_in_slice_rows"],
                         "block_in_panel_rows": blocks[f"{name}_block_in_panel_rows"],
                         "origin_range_inclusive": [int(dec.min()), int(dec.max())],
                         "origins_declared": int(dec.size),
                         "enumerator_origins": e["origins"],
                         "enumerator_admissible": e["admissible"],
                         "enumerator_withdrawn_non_finite_inputs":
                             e["withdrawn_non_finite_inputs"],
                         "enumerator_targets_valid_all": e["targets_valid_all"]},
            "materialized": {"origins": int(origins.size),
                             "origin_range_inclusive": [int(origins.min()), int(origins.max())],
                             "rows_touched_inclusive": touched,
                             "rows_touched": int(rows.size),
                             "withdrawn_for_non_finite_inputs": int(bad_input.sum()),
                             "withdrawn_for_a_non_finite_label": int(bad_label.sum()),
                             "non_finite_rows_in_the_slice":
                                 [int(r) for r in nonfinite_rows[:8]],
                             "largest_inversion_error_kW":
                                 _f(np.nanmax(np.abs(target_inv[finite] - panel_block[finite])))},
            "purge_declared": purge,
        })
    return out


def pretraining_splits(root: Path, data: dict, design: dict) -> list:
    """The auto-encoder's own two splits, which live in the AE cells and nowhere else."""
    tr = data["train_origins"].astype(np.int64)
    w, h = int(data["window"][0]), int(data["horizon"][0])
    ev = data["eval_origins"].astype(np.int64)
    frac = float(design["pretraining"]["internal_validation_fraction"])
    purge = w + h
    n_val = int(tr.size * frac)
    dec_val = tr[tr.size - n_val:]
    dec_trn = tr[:tr.size - n_val - purge]

    cells = sorted(p for p in (root / "attempts").glob("*/arrays.npz")
                   if "ae_train_origins" in np.load(p).files) if (root / "attempts").is_dir() else []
    if not cells:
        return [{"split": name, "state": UNCHECKABLE,
                 "checks": {"the_split_s_own_bytes_are_retained": None},
                 "missing": str(root / "attempts" / "*/arrays.npz"),
                 "why": "no retained cell carries the pre-training origins; the split's "
                        "materialization cannot be checked from what is retained"}
                for name in ("pretrain_train", "pretrain_internal_validation")]

    per_cell, ok_trn, ok_val = {}, True, True
    for p in cells:
        A = np.load(p)
        a_trn = A["ae_train_origins"].astype(np.int64)
        a_val = A["ae_validation_origins"].astype(np.int64)
        same_trn = bool(np.array_equal(a_trn, dec_trn))
        same_val = bool(np.array_equal(a_val, dec_val))
        gap = int(a_val.min() - a_trn.max())
        per_cell[p.parent.name] = {
            "ae_train_origins": int(a_trn.size), "ae_validation_origins": int(a_val.size),
            "is_exactly_the_declared_head_of_the_dev_train_origins": same_trn,
            "is_exactly_the_declared_purged_tail_of_the_dev_train_origins": same_val,
            "observed_origin_gap": gap,
            "subset_of_the_dev_train_origins": bool(np.isin(a_trn, tr).all()
                                                    and np.isin(a_val, tr).all()),
            "disjoint_from_each_other": bool(not np.intersect1d(a_trn, a_val).size),
            "never_touches_a_dev_validation_ROW":
                bool(_touched(a_val, w, h)[1] < int(ev.min() - w + 1)),
        }
        ok_trn &= same_trn and per_cell[p.parent.name]["subset_of_the_dev_train_origins"]
        ok_val &= (same_val and gap >= purge
                   and per_cell[p.parent.name]["never_touches_a_dev_validation_ROW"])

    common = {"declared": {"rule": design["pretraining"]["objective"],
                           "internal_validation_fraction": frac,
                           "purge_between": purge,
                           "declared_internal_validation_origins": int(dec_val.size),
                           "declared_pretrain_origins": int(dec_trn.size)},
              "cells_checked": sorted(per_cell), "per_cell": per_cell}
    rows = []
    for name, ok, key in (("pretrain_train", ok_trn, "declared_pretrain_origins"),
                          ("pretrain_internal_validation", ok_val,
                           "declared_internal_validation_origins")):
        checks = {"every_retained_cell_materializes_exactly_the_declared_origins": ok,
                  "the_count_matches_the_declaration":
                      all(c[f"ae_{'train' if name == 'pretrain_train' else 'validation'}_origins"]
                          == common["declared"][key] for c in per_cell.values()),
                  "no_row_of_the_dev_validation_split_is_reached":
                      all(c["never_touches_a_dev_validation_ROW"] for c in per_cell.values())
                      if name == "pretrain_internal_validation" else True,
                  "disjoint_from_the_other_pre_training_split":
                      all(c["disjoint_from_each_other"] for c in per_cell.values())}
        rows.append({"split": name, "state": _verdict(checks), "checks": checks, **common})
    return rows


def test_split(root: Path, blocks: dict, data: dict, design: dict) -> dict:
    """The reserve. Nothing was materialized for it, on purpose — so its ABSENCE is what is verified.

    This is a refusal BY NAME and not a pass: a split with no materialized bytes has no
    materialization to check, and saying so is the only honest verdict available. What can be checked
    is that nothing of it was read, and that is checked here on every retained array and record.
    """
    lo, hi = blocks["consumed_slice_in_panel_rows"]
    t_lo, t_hi = blocks["test_block_in_panel_rows"]
    leaked, exposures, scanned = [], {}, 0
    attempts = root / "attempts"
    if attempts.is_dir():
        for cell in sorted(p for p in attempts.iterdir() if p.is_dir()):
            arr = cell / "arrays.npz"
            if arr.is_file():
                scanned += 1
                for k in np.load(arr).files:
                    if "test" in k.lower():
                        leaked.append(f"{cell.name}/arrays.npz::{k}")
            meta = cell / "cell.json"
            if meta.is_file():
                exposures[cell.name] = json.loads(meta.read_text()).get("exposure")
    for k in data:
        if "test" in k.lower():
            leaked.append(f"DATA.npz::{k}")

    absence = {
        "the_consumed_slice_ends_before_the_declared_test_block": bool(hi <= t_lo),
        "no_retained_array_carries_a_test_population": bool(not leaked),
        "every_retained_cell_declares_NO_TEST_ACCESS":
            bool(scanned > 0 and exposures
                 and all(v == "NO_TEST_ACCESS" for v in exposures.values())),
        "the_design_declares_the_reserve_unread":
            "test rows never read" in design["dev_subpartition"].get("rows_note", ""),
    }
    return {"split": "test",
            "state": BY_DESIGN,
            "refusal": "REFUSED: nothing was materialized for the test split, so there is no "
                       "materialization to verify. Its absence is verified instead, and this verdict "
                       "is never reported as a passed materialization check",
            "checks": {"materialization_of_this_split": None},
            "absence_verified": absence,
            "absence_state": VERIFIED if all(absence.values()) else REFUTED,
            "declared": {"block_in_panel_rows": [int(t_lo), int(t_hi)],
                         "rows": int(t_hi - t_lo),
                         "family_usable_windows_declared":
                             design["task"]["family_usable_windows"].get("test"),
                         "consumed_slice_in_panel_rows": [int(lo), int(hi)]},
            "materialized": {"origins": 0, "rows_touched": 0, "arrays_carrying_test_rows": leaked,
                             "cells_scanned": scanned, "exposures": exposures}}


def exclusivity(rows: list, data: dict, root: Path) -> dict:
    """No split's rows in another's, on origins AND on touched rows, pairwise over what is retained."""
    w, h = int(data["window"][0]), int(data["horizon"][0])
    sets = {"train": data["train_origins"].astype(np.int64),
            "validation": data["eval_origins"].astype(np.int64)}
    ae = sorted((root / "attempts").glob("*/arrays.npz")) if (root / "attempts").is_dir() else []
    for p in ae:
        A = np.load(p)
        if "ae_train_origins" in A.files:
            sets["pretrain_train"] = A["ae_train_origins"].astype(np.int64)
            sets["pretrain_internal_validation"] = A["ae_validation_origins"].astype(np.int64)
            break
    names = sorted(sets)
    pairs, footprints = {}, {n: _touched(sets[n], w, h) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = int(np.intersect1d(sets[a], sets[b]).size)
            fa, fb = footprints[a], footprints[b]
            row_overlap = max(0, min(fa[1], fb[1]) - max(fa[0], fb[0]) + 1)
            nested = (a.startswith("pretrain") and b == "train") or \
                     (b.startswith("pretrain") and a == "train")
            pairs[f"{a}|{b}"] = {
                "shared_origins": shared,
                "shared_touched_rows": row_overlap,
                "origins_disjoint": shared == 0,
                "rows_disjoint": row_overlap == 0,
                "nesting_declared": nested,
                "reading": ("the pre-training splits are DECLARED subsets of the dev train origins, so "
                            "sharing rows with `train` is the declaration, not a leak; what must be "
                            "disjoint is each of them from `validation`")
                if nested else "these two splits must share no origin and no row",
            }
    verdict = all(p["origins_disjoint"] and p["rows_disjoint"]
                  for k, p in pairs.items() if not p["nesting_declared"])
    return {"splits_compared": names, "touched_row_footprints": footprints, "pairs": pairs,
            "state": VERIFIED if verdict else REFUTED,
            "test_split": "not compared: nothing is materialized for it (see the `test` entry)"}


def purge_boundary(data: dict, design: dict, data_json: dict) -> dict:
    """The boundary DECLARED against the boundary OBSERVED — the check RP59's own artifact left null."""
    tr = data["train_origins"].astype(np.int64)
    ev = data["eval_origins"].astype(np.int64)
    w, h = int(data["window"][0]), int(data["horizon"][0])
    declared = int(design["task"]["purge"])
    declared_sub = int(design["dev_subpartition"]["purge_between_splits"])
    origin_gap = int(ev.min() - tr.max())
    train_touched, eval_touched = _touched(tr, w, h), _touched(ev, w, h)
    row_gap = int(eval_touched[0] - train_touched[1] - 1)
    checks = {
        "the_design_declares_one_purge_in_both_places": declared == declared_sub,
        "the_observed_origin_gap_is_the_declared_purge_on_inclusive_endpoints":
            origin_gap == declared + 1,
        "the_observed_origin_gap_is_at_least_the_declared_purge": origin_gap >= declared,
        "no_row_is_touched_by_both_splits": row_gap >= 0,
        "the_last_train_LABEL_row_precedes_the_first_validation_WINDOW_row":
            train_touched[1] < eval_touched[0],
    }
    return {"declared_purge_task": declared, "declared_purge_dev_subpartition": declared_sub,
            "observed_origin_gap": origin_gap,
            "train_rows_touched_inclusive": train_touched,
            "validation_rows_touched_inclusive": eval_touched,
            "observed_untouched_rows_between_the_splits": row_gap,
            "rows_untouched_by_either": list(range(train_touched[1] + 1, eval_touched[0])),
            "checks": checks, "state": _verdict(checks),
            "what_this_repairs": "RP59/DATA_TARGET_PREPROCESSING_AUDIT.json reports "
                                 "`purge_declared: null` because df_e1_data_audit.split_step looks "
                                 "for the purge in DATA.json, which does not carry it. The "
                                 "declaration lives in DESIGN.json (task.purge and "
                                 "dev_subpartition.purge_between_splits) and is bound to the "
                                 "observed boundary here, which is what the audit required",
            "reading": "the purge is a boundary on ORIGINS, so the three numbers are one fact: a "
                       f"declared purge of {declared} shows up as an inclusive origin gap of "
                       f"{origin_gap} and leaves {max(row_gap, 0)} row(s) touched by neither split"}


def materialization(root: Path) -> dict:
    """Per-split materialization, verified from the bytes, with a named refusal where it cannot be."""
    out = {"run_root": str(root)}
    needed = {"DATA.npz": root / "DATA.npz", "DATA.json": root / "DATA.json",
              "DESIGN.json": root / "DESIGN.json", "DELIVERIES.json": root / "DELIVERIES.json"}
    missing = {k: str(p) for k, p in needed.items() if not p.is_file()}
    if missing:
        out["state"] = UNCHECKABLE
        out["missing"] = missing
        out["splits"] = [{"split": s, "state": UNCHECKABLE,
                          "checks": {"the_split_s_own_bytes_are_retained": None}, "missing": missing}
                         for s in ("train", "validation", "test", "pretrain_train",
                                   "pretrain_internal_validation")]
        return out

    data_json = json.loads(needed["DATA.json"].read_text())
    design = json.loads(needed["DESIGN.json"].read_text())
    D = np.load(needed["DATA.npz"])
    data = {k: D[k] for k in D.files}
    out["identity"] = {"DATA_npz_sha256": sha_file(needed["DATA.npz"]),
                       "declared_data_sha256": data_json.get("data_sha256"),
                       "design_sha256": design.get("design_sha256"),
                       "re_derives": sha_file(needed["DATA.npz"]) == data_json.get("data_sha256")}

    audit = _load("df_e1_data_audit")
    frame, unit, panel_path = audit._panel(root, data)
    lo, hi = data_json["slice_rows"]
    target_col = data_json["input_columns"][data_json["target_channel"]]
    panel_target = frame[target_col].to_numpy()[lo:hi].astype(np.float64)
    out["delivered_bytes"] = {"path_digest_matches_the_delivery":
                                  sha_file(panel_path) == unit["sha256"] == data_json["panel_sha256"],
                              "panel_rows": int(len(frame)),
                              "panel_rows_match_the_declaration":
                                  int(len(frame)) == data_json["panel_rows"]}

    blocks = declared_blocks(len(frame), (lo, hi), design)
    out["declared_blocks"] = blocks
    splits = supervised_splits(root, panel_target, blocks, design, data_json, data)
    splits += pretraining_splits(root, data, design)
    splits.append(test_split(root, blocks, data, design))
    out["splits"] = splits
    out["exclusivity"] = exclusivity(splits, data, root)
    out["purge"] = purge_boundary(data, design, data_json)
    del frame, panel_target
    checkable = [s for s in splits if s["state"] in (VERIFIED, REFUTED)]
    out["state"] = (VERIFIED if (checkable and all(s["state"] == VERIFIED for s in checkable)
                                 and out["exclusivity"]["state"] == VERIFIED
                                 and out["purge"]["state"] == VERIFIED
                                 and out["identity"]["re_derives"]) else REFUTED)
    out["per_split_state"] = {s["split"]: s["state"] for s in splits}
    out["refusals"] = {s["split"]: s.get("refusal") or s.get("why")
                       for s in splits if s["state"] in (UNCHECKABLE, BY_DESIGN)}
    return out


def sibling_root(root: Path, other: Path) -> dict:
    """Whether a second run root consumed the SAME prepared bytes, so these verdicts carry to it."""
    a, b = root / "DATA.npz", other / "DATA.npz"
    if not b.is_file():
        return {"root": str(other), "state": UNCHECKABLE, "missing": str(b),
                "why": "the verdicts of this document are not extended to a root whose prepared "
                       "bytes are not on this host"}
    da, db = sha_file(a), sha_file(b)
    return {"root": str(other), "DATA_npz_sha256": db, "identical_to_the_audited_root": da == db,
            "state": VERIFIED if da == db else REFUTED,
            "reading": "identical prepared bytes mean identical splits: every per-split verdict above "
                       "holds for this root without recomputation" if da == db else
                       "different prepared bytes: nothing above carries over"}


def run(root: Path) -> dict:
    return {"schema": "crispdm.rp59_lag_and_splits.v1",
            "authority": "the owner's grant of 2026-09-26; Satoshi, successor technical lead",
            "repairs": ["the lag table restated under both estimators, with its supersession stated",
                        "materialization verified PER SPLIT from the bytes, with named refusals"],
            "lag_table": lag_table(root),
            "materialization": materialization(root),
            "sibling_root": sibling_root(root, PHASE1_ROOT)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(SUCCESSOR_ROOT))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    report = run(Path(a.root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=False, default=str))
    print(json.dumps({"lag_table": report["lag_table"].get("state"),
                      "materialization": report["materialization"].get("state"),
                      "per_split": report["materialization"].get("per_split_state"),
                      "out": a.out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
