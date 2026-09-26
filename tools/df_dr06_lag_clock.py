#!/usr/bin/env python3
"""DR06 — the lag table recomputed on the retained clock, with the two estimators named apart.

Satoshi, successor technical lead, 2026-09-26. This tool supersedes the lag table TWICE over, and it
carries both superseded versions in every artifact it writes.

**The defect both superseded versions share.** Each of them drops the slice's single non-finite
position and only then applies the lag offsets::

    v = Y[rows]; v = v[np.isfinite(v)]        # tools/df_e1_data_audit.py::autocorrelation:262
                                              # tools/df_rp49_rp64_audit.py::data_audit:690
    ... v[:-k] against v[k:]

After a position is deleted, an index separation of ``k`` is no longer a separation of ``k`` minutes:
every pair that straddles the deletion is really ``k+1`` minutes apart and is counted as if it were
``k``. The repair is not to delete less carefully. It is to leave the clock alone: apply the offsets on
the retained one-minute grid and admit a pair only when **both** of its legs are finite. That is a
**finite-pair mask**, and it is what this tool does.

**The two estimators, named apart.** They are different things and this tool refuses to let one
sentence cover both:

    autocorrelation function with one whole-series denominator
        numerator over the admitted offsets, denominator over every admitted position, one mean
    Pearson correlation over lag-truncated pairs
        each leg of the admitted pairs centred and scaled on its own count

``_assert_names_share_no_word`` fails the run if those two names ever come to share a token. Neither is
called "bias-corrected" here: the second is not the first rescaled, and the arithmetic that proves it is
published beside both (``pearson_is_not_the_rescaled_autocorrelation``). The earlier restatement's label
is withdrawn, not repeated.

**What this is not.** Reanalysis. Nothing is fitted, no model is built, no allocation is taken, no
reserved split is read, and the table is **not a window selector**: ``not_a_window_selector`` is written
into every artifact and the rule set rejects an artifact that loses it. The day-versus-week inversion
persists on the corrected clock and is reported as what it is — an ordering of linear dependence between
a series and its own past — not as a chosen context.

    python tools/df_dr06_lag_clock.py --root RUN_ROOT --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVID = REPO / "docs" / "audits" / "evidence" / "d3_k5_20260917"

SUCCESSOR_ROOT = Path.home() / ".local/state/crispdm-data-foundation/e1_household_successor_v3"

LAGS = (1, 60, 1440, 10080)
LAG_NAMES = {1: "one minute", 60: "one hour", 1440: "one day", 10080: "one week"}

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
UNCHECKABLE = "REFUSED_RECOMPUTATION_UNCHECKABLE"

# --- the two estimators, named so that they cannot be confused for one another -------------------------

ACF = "autocorrelation function with one whole-series denominator"
PEARSON = "Pearson correlation over lag-truncated pairs"

ESTIMATORS = {
    "acf": {
        "name": ACF,
        "formula": "sum over the admitted offsets of (x_i - xbar)(x_{i+k} - xbar), divided by the sum "
                   "over every admitted position of (x_i - xbar)^2, with xbar the mean of the admitted "
                   "positions. The denominator does not shrink with k while the numerator does",
        "retained_implementation": "tools/df_e1_data_audit.py::autocorrelation — the estimator the "
                                   "published four-row list was produced by",
    },
    "pearson": {
        "name": PEARSON,
        "formula": "the two legs of the admitted pairs, each centred and scaled on its own pair count: "
                   "cov(a, b) / (sd(a) sd(b)). Every admitted pair is weighted once and nothing is "
                   "divided by a population it was not measured on",
        "retained_implementation": "np.corrcoef(v[:-k], v[k:]) as tools/df_rp49_rp64_audit.py::"
                                   "data_audit computes it",
    },
}

# --- the two row populations ---------------------------------------------------------------------------

POP_DELETED = "positions_deleted_before_the_offsets"
POP_CLOCK = "offsets_on_the_retained_clock_under_a_finite_pair_mask"

POPULATIONS = {
    POP_DELETED: "the non-finite position is removed from the series and the offsets are applied to "
                 "what remains, so every pair straddling the removal is really k+1 minutes apart and "
                 "is counted as k. This is the population BOTH superseded versions used",
    POP_CLOCK: "the retained one-minute grid is left intact, the offsets are applied on it, and a pair "
               "is admitted only when both of its legs are finite. An index separation of k IS a "
               "separation of k minutes for every admitted pair",
}

# --- what the table is, and what it is not ------------------------------------------------------------

NOT_A_SELECTOR = {
    "this_table_is": "an ordering of the LINEAR dependence between the target and its own past at four "
                     "offsets, measured on the retained training support of one household panel",
    "this_table_is_not": "a context-window selector. It does not rank candidate windows, it does not "
                         "measure the conditional contribution of a lag to a predictor, and it does "
                         "not say that a weekly context improves a forecast. No model was fitted to "
                         "produce any number in it",
    "the_inversion_specifically": "the week standing above the day is a fact about marginal linear "
                                  "dependence at two offsets. A prefix that includes a weekly offset "
                                  "must be justified by measured skill under the programme's own "
                                  "protocol, not by this ordering",
    "a_consumer_that_reads_this_as_a_choice_is_reading_it_wrong": True,
}

SUPERSEDES = [
    {
        "version": "as published",
        "what": "the four-row autocorrelation list of SATOSHI_PROGRAM_RP57_RP64_RETURN_2026_09_21 "
                "(RP59) and the `autocorrelation.autocorrelation_by_lag_minutes` block of "
                "docs/audits/evidence/d3_k5_20260917/RP59/DATA_TARGET_PREPROCESSING_AUDIT.json",
        "estimator": ACF,
        "population": POP_DELETED,
        "defect": "two defects. The estimator was not named, and its denominator does not shrink with "
                  "the offset while its numerator does, so the printed sequence reads as a decay that "
                  "is partly the estimator. And the population deleted a position before applying the "
                  "offsets, so some pairs counted as k minutes apart are k+1",
        "withdrawn": "nothing. The values are correct arithmetic on the population and estimator they "
                     "used, both of which are now named beside them",
    },
    {
        "version": "today's restatement",
        "what": "the `bias_corrected` column of "
                "docs/audits/evidence/RP59_LAG_AND_SPLITS_20260926/"
                "LAG_TABLE_AND_SPLIT_MATERIALIZATION.json and the table of "
                "SATOSHI_RP59_LAG_TABLE_AND_MATERIALIZATION_2026_09_26.md §1.2, at "
                "satoshi/rp59-lag-table-restatement-20260926 tip 6820fcae",
        "estimator": PEARSON,
        "population": POP_DELETED,
        "defect": "two defects. It kept the deleting population it inherited, so it corrected an "
                  "estimator while leaving the clock compressed; and it called its column "
                  "'bias-corrected', which names a Pearson correlation as if it were the published "
                  "autocorrelation function rescaled. It is not: the rescaled autocorrelation function "
                  "and the Pearson correlation differ, and this tool publishes both so the claim can "
                  "be checked rather than believed",
        "withdrawn": "the label 'bias-corrected' is withdrawn. The numbers are retained, labelled by "
                     "the estimator and the population that produced them",
    },
]

SURVIVES = ("the day-versus-week inversion: the weekly offset carries MORE marginal linear dependence "
            "than the daily one, reversing the order the published list printed")


# --- helpers ------------------------------------------------------------------------------------------

def _words(s: str) -> set:
    return {w for w in re.split(r"[^a-z]+", s.lower()) if w}


def _assert_names_share_no_word() -> dict:
    """The order is that the two estimators must not share a word. Machine-checked, not promised."""
    a, b = _words(ACF), _words(PEARSON)
    shared = sorted(a & b)
    if shared:
        raise AssertionError(f"the two estimator names share {shared!r}; name them apart")
    return {"acf_words": sorted(a), "pearson_words": sorted(b), "shared_words": shared,
            "disjoint": True}


def _load(name: str):
    """Load a repository tool under a PRIVATE name and register it nowhere."""
    spec = importlib.util.spec_from_file_location(f"_dr06_{name}", HERE / f"{name}.py")
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


# --- the two estimators, one implementation each, over an explicit pair set ----------------------------

def acf_at(v: np.ndarray, finite: np.ndarray, lag: int) -> dict:
    """The autocorrelation function with one whole-series denominator, over an explicit pair set.

    ``finite`` marks the admitted POSITIONS. A pair at offset ``lag`` is admitted when both legs are
    admitted. The denominator is the centred sum of squares over every admitted position — that is what
    makes this estimator the one the published list used, and what makes it shrink with the offset.
    """
    xbar = float(v[finite].mean())
    c = np.where(finite, v - xbar, 0.0)
    denom = float((c[finite] ** 2).sum())
    pair = finite[:-lag] & finite[lag:]
    num = float((c[:-lag][pair] * c[lag:][pair]).sum())
    return {"value": num / denom, "pairs": int(pair.sum()), "positions": int(finite.sum())}


def pearson_at(v: np.ndarray, finite: np.ndarray, lag: int) -> dict:
    """The Pearson correlation over lag-truncated pairs, over an explicit pair set."""
    pair = finite[:-lag] & finite[lag:]
    a, b = v[:-lag][pair], v[lag:][pair]
    return {"value": float(np.corrcoef(a, b)[0, 1]), "pairs": int(pair.sum())}


def straddling_pairs(n_kept: int, deleted_at: list, lag: int) -> int:
    """Pairs the DELETING population counts as k minutes apart that are really more than k.

    With the positions in ``deleted_at`` (indices into the ORIGINAL clock) removed, compressed index
    ``j`` maps to clock index ``c(j)``; a pair ``(j, j+lag)`` really spans ``c(j+lag) - c(j)``. This
    counts the pairs for which that is not ``lag``.
    """
    keep = np.array(sorted(set(range(n_kept + len(deleted_at))) - set(deleted_at)), dtype=np.int64)
    if lag >= keep.size:
        return 0
    span = keep[lag:] - keep[:-lag]
    return int((span != lag).sum())


# --- the clock itself ---------------------------------------------------------------------------------

def clock_check(panel_path: Path, slice_lo: int, n_clock: int, step_seconds: int = 60) -> dict:
    """The offsets mean minutes only if the retained rows ARE a gapless grid at the declared step.

    Re-derived from the delivered panel's own timestamp column over exactly the rows the lag table is
    measured on, never read from a field that asserts it.
    """
    import datetime as _dt

    import pyarrow.parquet as pq
    if not panel_path.is_file():
        return {"state": UNCHECKABLE, "missing": str(panel_path)}
    col = pq.ParquetFile(panel_path).read(columns=["timestamp_label"]).column("timestamp_label")
    labels = col.slice(slice_lo, n_clock).to_pylist()
    if len(labels) != n_clock:
        return {"state": UNCHECKABLE, "why": f"the panel holds {len(labels)} of {n_clock} clock rows"}
    fmt = "%d/%m/%Y %H:%M:%S"                      # the family contract's declared ts_format
    t = np.array([np.datetime64(_dt.datetime.strptime(d, fmt), "s") for d in labels])
    steps = (t[1:] - t[:-1]).astype(np.int64)
    uniq = sorted({int(s) for s in steps})
    ok = uniq == [step_seconds]
    return {"state": VERIFIED if ok else REFUTED,
            "panel_sha256": sha_file(panel_path),
            "clock_rows": int(n_clock),
            "first_stamp": str(t[0]), "last_stamp": str(t[-1]),
            "declared_step_seconds": step_seconds,
            "distinct_steps_observed_seconds": uniq,
            "strictly_increasing": bool((steps > 0).all()),
            "duplicated_stamps": int(n_clock - len({str(x) for x in t})),
            "an_index_offset_of_k_is_k_minutes": ok,
            "why_this_check_exists": "a finite-pair mask keeps the offsets honest only if the retained "
                                     "rows are a gapless grid at the declared step. If they were not, "
                                     "the repair would have to carry the stamps, not the indices"}


# --- the table ----------------------------------------------------------------------------------------

def code_identity() -> dict:
    """The revision this recomputation ran at, and whether its checkout was byte-clean.

    DR06 requires a clean pinned execution checkout. That is a fact about the run, so it is recorded
    by the run rather than asserted by the document.
    """
    import subprocess

    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return {"revision": _git("rev-parse", "HEAD"),
            "worktree_has_uncommitted_changes": bool(_git("status", "--porcelain")),
            "tool_sha256": sha_file(Path(__file__).resolve())}


def lag_table(root: Path, panel_path: Path | None = None) -> dict:
    out = {"schema": "df_dr06_lag_clock.v1",
           "code_identity": code_identity(),
           "scope": "the retained training support of the household panel. No reserved split is read",
           "estimators": ESTIMATORS,
           "estimator_names_share_no_word": _assert_names_share_no_word(),
           "populations": POPULATIONS,
           "supersedes": SUPERSEDES,
           "not_a_window_selector": dict(NOT_A_SELECTOR),
           "is_this_training": "no. Reanalysis of retained bytes: nothing is fitted, built or scored"}

    published_path = EVID / "RP59" / "DATA_TARGET_PREPROCESSING_AUDIT.json"
    data_path = root / "DATA.npz"
    out["published_artifact"] = str(published_path)
    out["run_root"] = str(root)

    if not published_path.is_file() or not data_path.is_file():
        out["state"] = UNCHECKABLE
        out["missing"] = [str(p) for p in (published_path, data_path) if not p.is_file()]
        out["why"] = ("every column of this table is recomputed from the retained bytes; with a needed "
                      "artifact absent no column is printed and no published value is repeated as if "
                      "it had been re-derived")
        return out

    pub = json.loads(published_path.read_text())["autocorrelation"]
    out["as_published_values"] = {str(k): _f(v)
                                  for k, v in pub["autocorrelation_by_lag_minutes"].items()}
    out["as_published_support_rows"] = int(pub["train_rows_used"])
    out["data_sha256"] = sha_file(data_path)

    D = np.load(data_path)
    data = {k: D[k] for k in D.files}
    Y, tr = data["Y"], data["train_origins"].astype(np.int64)
    w = int(data["window"][0])
    lo, hi = int(tr.min() - w + 1), int(tr.max()) + 1
    clock_rows = np.arange(lo, hi)
    v = Y[clock_rows]
    finite = np.isfinite(v)
    deleted_at = [int(i) for i in np.flatnonzero(~finite)]

    out["support"] = {
        "clock_positions": int(v.size),
        "clock_row_span_in_the_slice": [lo, hi - 1],
        "non_finite_positions": len(deleted_at),
        "non_finite_at_slice_row": deleted_at,
        "positions_the_deleting_population_kept": int(finite.sum()),
        "the_published_support_is_the_deleted_one": int(finite.sum()) == out["as_published_support_rows"],
        "note": "the deleting population's row count is the published support. The clock has one more "
                "position than that, and the correction keeps it",
    }

    # the deleting population, reproduced rather than re-implemented ---------------------------------
    u = v[finite]
    kept_finite = np.ones(u.size, dtype=bool)
    retained_acf = _load("df_e1_data_audit").autocorrelation(data, lags=LAGS)
    out["retained_acf_implementation_reproduces"] = {}

    rows = []
    for lag in LAGS:
        a_del = acf_at(u, kept_finite, lag)
        p_del = pearson_at(u, kept_finite, lag)
        a_clk = acf_at(v, finite, lag)
        p_clk = pearson_at(v, finite, lag)
        n_del = int(u.size)
        rescaled = a_del["value"] * n_del / (n_del - lag)
        ref = _f(retained_acf["autocorrelation_by_lag_minutes"][str(lag)])
        out["retained_acf_implementation_reproduces"][str(lag)] = bool(
            abs(a_del["value"] - ref) <= 1e-12)
        rows.append({
            "lag_minutes": lag,
            "lag_named": LAG_NAMES[lag],
            "acf_positions_deleted": _f(a_del["value"]),
            "acf_positions_deleted_rescaled_n_over_n_minus_k": _f(rescaled),
            "acf_clock_with_finite_pair_mask": _f(a_clk["value"]),
            "pearson_positions_deleted": _f(p_del["value"]),
            "pearson_clock_with_finite_pair_mask": _f(p_clk["value"]),
            "pairs_positions_deleted": a_del["pairs"],
            "pairs_clock_with_finite_pair_mask": a_clk["pairs"],
            "pairs_the_deleting_population_mislabels": straddling_pairs(int(u.size), deleted_at, lag),
            "pearson_minus_rescaled_acf": _f(p_clk["value"] - rescaled),
            "clock_minus_deleted_pearson": _f(p_clk["value"] - p_del["value"]),
        })
    out["rows"] = rows
    by = {r["lag_minutes"]: r for r in rows}

    out["as_published_re_derives"] = all(
        abs(by[k]["acf_positions_deleted"] - out["as_published_values"][str(k)]) <= 1e-10
        for k in LAGS)

    # what the rescaling claim is worth --------------------------------------------------------------
    out["pearson_is_not_the_rescaled_autocorrelation"] = {
        "claim": f"the {PEARSON} is NOT the {ACF} rescaled by n/(n-k). The earlier restatement's "
                 "'bias-corrected' label asserted an equivalence that does not hold",
        "per_lag": {str(k): {"rescaled_acf": by[k]["acf_positions_deleted_rescaled_n_over_n_minus_k"],
                             "pearson_on_the_clock": by[k]["pearson_clock_with_finite_pair_mask"],
                             "difference": by[k]["pearson_minus_rescaled_acf"]} for k in LAGS},
        "largest_absolute_difference": _f(max(abs(by[k]["pearson_minus_rescaled_acf"]) for k in LAGS)),
        "at_lag_minutes": int(max(LAGS, key=lambda k: abs(by[k]["pearson_minus_rescaled_acf"]))),
        "state": VERIFIED if max(abs(by[k]["pearson_minus_rescaled_acf"]) for k in LAGS) > 1e-6
                 else REFUTED,
    }

    # what deleting a position actually did ---------------------------------------------------------
    out["what_the_deleting_population_cost"] = {
        "mislabelled_pairs_per_lag": {str(k): by[k]["pairs_the_deleting_population_mislabels"]
                                      for k in LAGS},
        "shift_in_the_pearson_per_lag": {str(k): by[k]["clock_minus_deleted_pearson"] for k in LAGS},
        "largest_shift": _f(max(abs(by[k]["clock_minus_deleted_pearson"]) for k in LAGS)),
        "reading": "the shift is small in magnitude and it is not the point: a population that counts "
                   "pairs at the wrong offset is wrong whatever the size of the consequence, and its "
                   "size could not be known before it was measured",
    }

    # orderings, and the inversion ------------------------------------------------------------------
    def order(field):
        return [k for k in sorted(LAGS, key=lambda j: -by[j][field])]

    out["ordering_by_strength"] = {
        "acf_positions_deleted": order("acf_positions_deleted"),
        "acf_clock_with_finite_pair_mask": order("acf_clock_with_finite_pair_mask"),
        "pearson_positions_deleted": order("pearson_positions_deleted"),
        "pearson_clock_with_finite_pair_mask": order("pearson_clock_with_finite_pair_mask"),
    }
    week, day = by[10080], by[1440]
    out["day_versus_week_inversion"] = {
        "claim": SURVIVES,
        "under_the_published_acf": f"{week['acf_positions_deleted']:.8f} < "
                                   f"{day['acf_positions_deleted']:.8f} — no inversion",
        "under_the_acf_on_the_clock": f"{week['acf_clock_with_finite_pair_mask']:.8f} vs "
                                      f"{day['acf_clock_with_finite_pair_mask']:.8f}",
        "under_the_pearson_on_the_clock": f"{week['pearson_clock_with_finite_pair_mask']:.8f} > "
                                          f"{day['pearson_clock_with_finite_pair_mask']:.8f}",
        "persists_on_the_corrected_clock": bool(
            week["pearson_clock_with_finite_pair_mask"] > day["pearson_clock_with_finite_pair_mask"]),
        "persists_under_the_published_estimator_too": bool(
            week["acf_clock_with_finite_pair_mask"] > day["acf_clock_with_finite_pair_mask"]),
        "and_still_not_a_selector": NOT_A_SELECTOR["the_inversion_specifically"],
    }
    out["the_reading_that_survives_unchanged"] = {
        "claim": "the daily offset carries less marginal linear dependence than the hourly one, so a "
                 "daily context remains a hypothesis and not a certainty",
        "under_every_column": all(by[1440][f] < by[60][f] for f in
                                  ("acf_positions_deleted", "acf_clock_with_finite_pair_mask",
                                   "pearson_positions_deleted", "pearson_clock_with_finite_pair_mask")),
    }

    if panel_path is not None:
        out["clock"] = clock_check(panel_path, int(json.loads((root / "DATA.json").read_text())
                                                  ["slice_rows"][0]) + lo, int(v.size))

    checks = {
        "as_published_re_derives": out["as_published_re_derives"],
        "retained_acf_implementation_reproduced": all(
            out["retained_acf_implementation_reproduces"].values()),
        "estimator_names_disjoint": out["estimator_names_share_no_word"]["disjoint"],
        "pearson_differs_from_the_rescaled_acf":
            out["pearson_is_not_the_rescaled_autocorrelation"]["state"] == VERIFIED,
        "the_inversion_persists": out["day_versus_week_inversion"]["persists_on_the_corrected_clock"],
        "the_surviving_reading_holds_everywhere": out["the_reading_that_survives_unchanged"][
            "under_every_column"],
        "clock_is_a_gapless_minute_grid": (None if panel_path is None
                                           else out["clock"].get("an_index_offset_of_k_is_k_minutes")),
        "both_superseded_versions_carried": len(out["supersedes"]) == 2,
        "not_a_window_selector_note_present": bool(out["not_a_window_selector"]),
    }
    out["checks"] = checks
    out["state"] = (UNCHECKABLE if any(x is None for x in checks.values())
                    else VERIFIED if all(checks.values()) else REFUTED)
    return out


# --- what an admissible artifact must carry -----------------------------------------------------------

REQUIRED_ROW_FIELDS = ("acf_positions_deleted", "acf_positions_deleted_rescaled_n_over_n_minus_k",
                       "acf_clock_with_finite_pair_mask", "pearson_positions_deleted",
                       "pearson_clock_with_finite_pair_mask", "pairs_clock_with_finite_pair_mask",
                       "pairs_the_deleting_population_mislabels")

SELECTOR_WORDS = ("recommend", "choose the window", "select the window", "optimal window",
                  "the window to use", "therefore use a weekly", "window selector is")


def validate(rec: dict) -> list:
    """Reasons this artifact is not admissible. An empty list is the only pass.

    Its job is to make the three orders of DR06 §2 machine-enforced rather than promised: both
    superseded versions carried, the two estimators named apart, and no column presented as a window
    choice. A record that satisfies it can still be REFUTED on its numbers; a record that fails it may
    not be read at all.
    """
    bad = []
    if rec.get("schema") != "df_dr06_lag_clock.v1":
        bad.append("the schema is not df_dr06_lag_clock.v1")
    sup = rec.get("supersedes") or []
    if len(sup) != 2:
        bad.append(f"both superseded versions must be carried; {len(sup)} present")
    for s in sup:
        for f in ("version", "what", "estimator", "population", "defect", "withdrawn"):
            if not s.get(f):
                bad.append(f"a superseded version is missing {f!r}")
    if {s.get("estimator") for s in sup} != {ACF, PEARSON}:
        bad.append("the two superseded versions must name the two estimators, one each")
    est = rec.get("estimators") or {}
    if {e.get("name") for e in est.values()} != {ACF, PEARSON}:
        bad.append("the estimator block must name exactly the two estimators, each once")
    a, b = _words(ACF), _words(PEARSON)
    if a & b:
        bad.append(f"the two estimator names share {sorted(a & b)!r}")
    if not rec.get("estimator_names_share_no_word", {}).get("disjoint"):
        bad.append("the word-disjointness of the two estimator names is not recorded as checked")
    blob_all = json.dumps(rec)
    if (blob_all.count("bias-corrected") != blob_all.count("'bias-corrected'")
            or blob_all.count("bias_corrected") != blob_all.count("`bias_corrected`")):
        bad.append("no column of THIS record may be labelled bias-corrected; the label is withdrawn "
                   "and may appear only as a quoted or backticked citation of the superseded artifact "
                   "that carried it")
    sel = rec.get("not_a_window_selector") or {}
    for f in ("this_table_is", "this_table_is_not", "the_inversion_specifically"):
        if not sel.get(f):
            bad.append(f"not_a_window_selector is missing {f!r}")
    blob = " ".join(str(v).lower() for v in (rec.get("rows") or []))
    blob += " " + str(rec.get("day_versus_week_inversion", "")).lower()
    for phrase in SELECTOR_WORDS:
        if phrase in blob:
            bad.append(f"the table reads as a window choice: {phrase!r}")
    for r in (rec.get("rows") or []):
        for f in REQUIRED_ROW_FIELDS:
            if f not in r:
                bad.append(f"row at lag {r.get('lag_minutes')} is missing {f!r}")
    if rec.get("state") == VERIFIED and any(v is None for v in (rec.get("checks") or {}).values()):
        bad.append("a check that could not run is recorded inside a VERIFIED record")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=SUCCESSOR_ROOT)
    ap.add_argument("--panel", type=Path, default=None,
                    help="the delivered panel parquet; the clock check is UNCHECKABLE without it")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    panel = a.panel
    if panel is None:
        dj = a.root / "DATA.json"
        if dj.is_file():
            p = Path(json.loads(dj.read_text())["delivery"]["path"])
            panel = p if p.is_file() else None
    rec = lag_table(a.root, panel)
    rec["admissibility"] = {"reasons_it_would_be_inadmissible": validate(rec)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, sort_keys=False) + "\n")
    print(json.dumps({"state": rec["state"], "checks": rec.get("checks"),
                      "inadmissible_because": rec["admissibility"][
                          "reasons_it_would_be_inadmissible"],
                      "out": str(a.out)}, indent=1))
    return 0 if (rec["state"] == VERIFIED and not validate(rec)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
