"""I6-DMO matched short control: the frozen MG1/MG2 common population (plan 20261015 section 2).

NEXT_CONTRASTS_PLAN_20261015 section 2 / SC01-SC04 / SC10, data-lane part. CPU only; nothing
here fits a model, opens TEST or reads a TEST bar (``data.load_input`` refuses any input that
holds one before an array is built).

* Two NEW market-time treatments on the expected open-market-hour grid of the versioned
  calendar: ``mg1`` (control, 25 real observations incl. t, 1 open-market-hour spacing) and
  ``mg2m`` (matched reference, 13 observations, 2 h spacing); both span 24 open-market hours.
  Slots, exactness and C1 per-head business masks come from the audited
  :func:`tools.i6dmo.market_grid.build_market_table`; this module only intersects them.
* For FIT, INNER and the full DEVELOPMENT year (the data lane's VALIDATION split) separately,
  the population is the intersection of the (origin row_id, horizon) pairs valid under BOTH
  treatments; origins with no common valid head are dropped. Every candidate origin keeps a
  typed per-head exclusion reason; unsupported heads are typed, never filled.
* ONE FIT-only scaler: per-feature mean/std (ddof=0) over the UNION of both arms' exact
  input-source bars of the common FIT origins, with the exact source-row digest. Target loss
  scales are the FIT-only naive MAE per horizon on the common FIT mask. INNER/DEVELOPMENT
  never enter either computation.
* Targets are exact ELAPSED ``p(t+h)``; no smoothing, interpolation, ASOF or next-row target.

Public interface (``INTERFACE_COMMON_POPULATION.json``): :class:`SplitPopulation`,
:class:`SharedScaler`, :class:`CommonPopulation`, :func:`build_common_population`,
:func:`write_common_population`, :func:`load_common_population`, :func:`arm_tensors`.
Typed refusals are :class:`CommonPopulationError` with a stable ``code``.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from tools.i6dmo import data, design as D, market_grid, provenance
from tools.i6n_contracts import eligibility as c1

UTC = timezone.utc
HOUR = 3600
SCHEMA = "i6dmo.common_population.v1"
SEAL_SCHEMA = "i6dmo.common_population.seal.v1"
DATA_DESIGN_SCHEMA = "i6dmo.matched_market_design.v1"
HORIZONS = (2, 3, 4, 5, 6, 12)
SPLIT_NAMES = ("FIT", "INNER", "DEVELOPMENT")
#: DEVELOPMENT is the data lane's VALIDATION split (full 2024); the name follows the plan.
SOURCE_SPLIT = {"FIT": "FIT", "INNER": "INNER", "DEVELOPMENT": "VALIDATION"}
ARMS = tuple(D.MATCHED_SCOPE)                      # ("mg1", "mg2m")
#: Typed per-(origin, head) exclusion codes. Precedence: business/label (C1, independent of
#: the input treatment) first, then the treatment-specific input reason.
HEAD_REASONS = ("COMMON_VALID", "ORIGIN_OUTSIDE_CALENDAR_WEEK",
                "OPERATIONALLY_INELIGIBLE", "TARGET_NOT_OBSERVED",
                "TARGET_AVAILABLE_AFTER_LABEL_CUTOFF", "TARGET_BEFORE_SPLIT_START",
                "TARGET_AT_OR_AFTER_SPLIT_END", "TARGET_OUTSIDE_DECLARED_SPLITS",
                "INPUT_INVALID_MG1_ONLY", "INPUT_INVALID_MG2M_ONLY", "INPUT_INVALID_BOTH")
INPUT_REASONS = ("INPUT_OK", "ORIGIN_NOT_EXPECTED_MARKET_HOUR", "INPUT_BEFORE_DATA_START",
                 "INPUT_SLOT_NOT_OBSERVED", "INPUT_VALUE_NOT_FINITE")
_IDENTITY_KEYS = ("experiment_id", "arm", "contrast_id", "asset", "input_sampling",
                  "horizons_hours", "targets", "naive", "splits", "business", "feature_policy",
                  "features", "feature_manifest_sha256", "availability_contract_sha256",
                  "preprocessing", "time_encoder", "matched_population_rules_sha256")


class CommonPopulationError(ValueError):
    """Typed refusal; ``code`` is stable."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


# =========================================================================== data designs

def matched_data_design(short_design: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    """Data-lane DESIGN.<arm>.json: everything that determines the arm's materialized arrays."""
    if arm not in ARMS:
        raise CommonPopulationError("ARM_UNKNOWN", f"{arm!r}; one of {ARMS}")
    base = data.validate_design(short_design)
    if base.get("experiment_id") != D.SHORT_ID:
        raise CommonPopulationError("SHORT_DESIGN_REQUIRED", str(base.get("experiment_id")))
    scope = D.MATCHED_SCOPE[arm]
    doc: Dict[str, Any] = {
        "schema": DATA_DESIGN_SCHEMA,
        "status": "DATA_LANE_SEALED_INPUTS_NOT_A_RUNTIME_CONFIG",
        "experiment_id": scope["experiment_id"], "arm": arm, "role": scope["role"],
        "partner_arm": scope["partner"], "contrast_id": D.MATCHED_CONTRAST_ID,
        "derives_from_design_sha256": provenance.canonical_sha256(base),
        "asset": base["asset"], "seed": base["seed"], "repetitions": base["repetitions"],
        "evaluation_mode": base["evaluation_mode"],
        "retraining_frequency": base["retraining_frequency"],
        "input_sampling": {
            "rule": scope["rule"], "observations": scope["observations"],
            "spacing_open_market_hours": scope["spacing_open_market_hours"],
            "span_open_market_hours": scope["span_open_market_hours"],
            "elapsed_span_hours_fixed": False, "include_origin": True, "issuance_hours": 1,
            "value_mode": "RAW",
            "grid": ("expected whole-hour bar-END instants in (open, close] of the versioned "
                     "calendar; never dataset row counts"),
            "exactness": ("exact finite retained observation required at every expected "
                          "endpoint; unplanned open-session gaps are counted exclusions"),
            "no_asof_no_smoothing_no_interpolation_no_synthetic_prices": True,
        },
        "horizons_hours": list(data.horizons(base)),
        "targets": "exact ELAPSED t+h, unmodified, under the original C1 business masks",
        "naive": "p(t) vs p(t+h) in ELAPSED hours on exactly the common rows, original price",
        "splits": json.loads(json.dumps(base["splits"])),
        "split_names": dict(SOURCE_SPLIT),
        "business": json.loads(json.dumps(base["business"])),
        "feature_policy": json.loads(json.dumps(base["feature_policy"])),
        "features": list(provenance.MANIFEST),
        "feature_manifest_sha256": provenance.EXPECTED_MANIFEST_SHA256,
        "availability_contract_sha256": provenance.EXPECTED_CONTRACT_SHA256,
        "preprocessing": D.MATCHED_POPULATION_RULES["scaler"],
        "loss_scale": D.MATCHED_POPULATION_RULES["loss_scale"],
        "time_encoder": dict(market_grid.TIME_ENCODER),
        "matched_population_rules_sha256": D.MATCHED_POPULATION_RULES["rules_sha256"],
    }
    if list(doc["horizons_hours"]) != list(HORIZONS):
        raise CommonPopulationError("HORIZONS_INVALID", str(doc["horizons_hours"]))
    doc["data_identity_sha256"] = data_identity_sha256(doc)
    return validate_matched_data_design(doc, arm=arm)


def data_identity_sha256(doc: Mapping[str, Any]) -> str:
    return provenance.canonical_sha256({k: doc[k] for k in _IDENTITY_KEYS})


def validate_matched_data_design(doc: Mapping[str, Any], *, arm: Optional[str] = None) -> dict:
    """Typed validation of a matched data design; the closed MG2 data design is refused."""
    if not isinstance(doc, Mapping) or doc.get("schema") != DATA_DESIGN_SCHEMA:
        raise CommonPopulationError("MATCHED_DATA_DESIGN_REQUIRED",
                                    f"schema must be {DATA_DESIGN_SCHEMA}; the closed full-population "
                                    "market design is not a matched identity")
    for key in ("input", "history_span_hours", "spacing_hours"):
        if key in doc:
            raise CommonPopulationError("CONTRADICTORY_TIME_FIELDS", f"{key!r} is an elapsed field")
    got = doc.get("arm")
    if got not in ARMS or (arm is not None and got != arm):
        raise CommonPopulationError("ARM_UNKNOWN", f"design arm {got!r} (expected {arm!r})")
    scope = D.MATCHED_SCOPE[got]
    s = doc.get("input_sampling") or {}
    want = {"rule": scope["rule"], "observations": scope["observations"],
            "spacing_open_market_hours": scope["spacing_open_market_hours"],
            "span_open_market_hours": scope["span_open_market_hours"],
            "elapsed_span_hours_fixed": False, "include_origin": True, "issuance_hours": 1,
            "value_mode": "RAW"}
    for key, value in want.items():
        if s.get(key) != value or type(s.get(key)) is not type(value):
            raise CommonPopulationError("MATCHED_DATA_DESIGN_INVALID",
                                        f"input_sampling.{key} must be {value!r}")
    if doc.get("experiment_id") != scope["experiment_id"] or \
            doc.get("contrast_id") != D.MATCHED_CONTRAST_ID:
        raise CommonPopulationError("MATCHED_DATA_DESIGN_INVALID", "experiment/contrast id")
    if list(doc.get("horizons_hours") or ()) != list(HORIZONS):
        raise CommonPopulationError("MATCHED_DATA_DESIGN_INVALID", "horizons 2,3,4,5,6,12")
    if doc.get("features") != list(provenance.MANIFEST) or \
            doc.get("feature_manifest_sha256") != provenance.EXPECTED_MANIFEST_SHA256:
        raise CommonPopulationError("MATCHED_DATA_DESIGN_INVALID", "19-feature manifest required")
    if doc.get("matched_population_rules_sha256") != D.MATCHED_POPULATION_RULES["rules_sha256"]:
        raise CommonPopulationError("MATCHED_DATA_DESIGN_INVALID", "population rules changed")
    if doc.get("data_identity_sha256") != data_identity_sha256(doc):
        raise CommonPopulationError("MATCHED_DATA_DESIGN_SEAL", "data identity does not verify")
    return json.loads(json.dumps(doc))


def _elapsed_shim(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """The elapsed-schema view the audited loaders consume (its window is never used as time)."""
    s = doc["input_sampling"]
    shim = {
        "schema": "i6dmo.design.v1", "status": "DESIGN_NOT_RUNTIME_CONFIG",
        "experiment_id": doc["experiment_id"], "asset": doc["asset"], "seed": doc["seed"],
        "repetitions": doc["repetitions"], "evaluation_mode": doc["evaluation_mode"],
        "retraining_frequency": doc["retraining_frequency"],
        "input": {"value_mode": "RAW", "history_span_hours": s["span_open_market_hours"],
                  "spacing_hours": s["spacing_open_market_hours"],
                  "observations": s["observations"], "include_origin": True,
                  "issuance_hours": 1, "smoothing": False, "require_exact_timestamps": True},
        "horizons_hours": list(doc["horizons_hours"]), "feature_policy": doc["feature_policy"],
        "splits": doc["splits"], "business": doc["business"],
    }
    return data.validate_design(shim)


# =========================================================================== dataclasses

@dataclass(frozen=True)
class SplitPopulation:
    """Common rows of one split: ordered origin epochs and (n, 6) per-head masks."""

    name: str
    origin_row_ids: np.ndarray            # (n,) int64 origin UTC epoch seconds, increasing
    head_mask: np.ndarray                 # (n, 6) bool common valid (origin, head) pairs
    entry_mask: np.ndarray                # (n, 6) bool C1 ENTRY support (subset of head_mask)
    management_mask: np.ndarray           # (n, 6) bool C1 MANAGEMENT support (== head_mask)
    counts: Dict[int, int]                # per horizon: number of common valid pairs
    digest: str

    def arrays(self) -> Dict[str, np.ndarray]:
        return {"origin_row_ids": self.origin_row_ids, "head_mask": self.head_mask,
                "entry_mask": self.entry_mask, "management_mask": self.management_mask}

    def row_ids(self) -> np.ndarray:
        return np.asarray([data.row_id(int(t)) for t in self.origin_row_ids], dtype=str)


@dataclass(frozen=True)
class SharedScaler:
    """ONE FIT-only per-feature z-score shared by both arms."""

    feature_names: Tuple[str, ...]
    mean: np.ndarray                      # (19,) float64
    std: np.ndarray                       # (19,) float64, ddof=0
    fit_row_digest: str                   # digest of the sorted source-bar UTC epochs
    n_rows: int                           # number of unique source bars

    def transform(self, x: np.ndarray) -> np.ndarray:
        """(x - mean) / std on the last axis; a zero std is a typed refusal, never an epsilon."""
        if np.any(self.std == 0.0):
            zero = [f for f, s in zip(self.feature_names, self.std) if s == 0.0]
            raise CommonPopulationError("ZERO_SCALE", f"features {zero}")
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != len(self.feature_names):
            raise CommonPopulationError("FEATURE_AXIS_MISMATCH", str(x.shape))
        return (x - self.mean) / self.std

    def to_json(self) -> Dict[str, Any]:
        return {"feature_names": list(self.feature_names),
                "mean": [float(v) for v in self.mean], "std": [float(v) for v in self.std],
                "fit_row_digest": self.fit_row_digest, "n_rows": int(self.n_rows),
                "ddof": 0, "basis": D.MATCHED_POPULATION_RULES["scaler"]}


@dataclass(frozen=True)
class CommonPopulation:
    """Frozen common population of the matched MG1/MG2 contrast."""

    splits: Dict[str, SplitPopulation]
    scaler: SharedScaler
    naive_loss_scale: Dict[int, Optional[float]]   # None = typed unsupported head, never filled
    identities: Dict[str, Any]
    digest: str
    schema: str = SCHEMA
    horizons: Tuple[int, ...] = HORIZONS
    report: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    arm_arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = field(
        default_factory=dict, compare=False, repr=False)
    exclusion_arrays: Dict[str, Dict[str, np.ndarray]] = field(
        default_factory=dict, compare=False, repr=False)
    source_dir: Optional[str] = field(default=None, compare=False, repr=False)
    #: arm -> split -> array_sha256 RECOMPUTED by the verified load (never the stored strings);
    #: the source_dir accessor binds to these, so a post-load file+string edit is refused (D3).
    verified_arm_digests: Dict[str, Dict[str, str]] = field(
        default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class ArmTensors:
    """One arm's inputs/time inputs/exact targets on EXACTLY the common rows of one split."""

    arm: str
    split: str
    experiment_id: str
    row_id: np.ndarray                    # (n,) int64, == SplitPopulation.origin_row_ids
    row_ids: np.ndarray                   # (n,) str 'EURUSD|<iso>'
    x: np.ndarray                         # (n, slots, 19) float64 RAW, oldest -> t
    offsets_hours: np.ndarray             # (n, slots) float64 ACTUAL signed offsets; last == 0
    adjacent_delta_hours: np.ndarray      # (n, slots) float64; first column 0
    closure_crossed: np.ndarray           # (n, slots) int64 0/1; first column 0
    slot_instants: np.ndarray             # (n, slots) int64 UTC epochs; last == row_id
    origin_price: np.ndarray              # (n,) float64 p(t)
    target_price: np.ndarray              # (n, 6) float64 exact p(t+h); NaN where ~head_mask
    head_mask: np.ndarray
    entry_mask: np.ndarray
    management_mask: np.ndarray
    week_ids: np.ndarray                  # (n,) str calendar week

    def scaled_x(self, scaler: SharedScaler) -> np.ndarray:
        return scaler.transform(self.x)


# =========================================================================== build

def _split_digest(name: str, arrays: Mapping[str, np.ndarray], counts: Mapping[int, int]) -> str:
    return provenance.canonical_sha256({
        "name": name, "arrays_sha256": provenance.array_sha256(arrays),
        "row_ids_sha256": provenance.array_sha256({"origin_row_ids": arrays["origin_row_ids"]}),
        "counts": {str(h): int(v) for h, v in counts.items()}})


def _scaler_rows_digest(bar_ts: np.ndarray) -> str:
    return provenance.array_sha256({"fit_source_bar_ts": np.asarray(bar_ts, dtype=np.int64)})


def _business_labels(bundle: data.InputBundle, elapsed: Mapping[str, Any],
                     table: market_grid.MarketOriginTable) -> Tuple[np.ndarray, np.ndarray]:
    """C1 MANAGEMENT label eligibility per (origin, head) GIVEN available input + primary reason.

    Independent of the input treatment; ``label_ok(arm) == given & input_ok(arm)`` is asserted.
    """
    splits = data.split_bounds(elapsed)
    _, weeks = data._weeks(bundle)
    opens = [w.market_open for w in weeks]
    observed = c1.ObservedTargets({datetime.fromtimestamp(int(v), UTC):
                                   datetime.fromtimestamp(int(v), UTC) for v in bundle.ts})
    n, nh = len(table.origin_ts), len(table.horizons)
    ok = np.zeros((n, nh), dtype=bool)
    reason = np.zeros((n, nh), dtype=np.int8)
    code = {r: i for i, r in enumerate(HEAD_REASONS)}
    for k, t in enumerate(table.origin_ts.tolist()):
        t_dt = datetime.fromtimestamp(int(t), UTC)
        split = next(s for s in splits if s.start <= t_dt < s.end)
        w = bisect.bisect_right(opens, t_dt) - 1
        if w < 0 or not t_dt < weeks[w].market_close:
            reason[k, :] = code["ORIGIN_OUTSIDE_CALENDAR_WEEK"]
            continue
        decisions = c1.evaluate_heads(t_dt, table.horizons, c1.Role.MANAGEMENT, weeks[w],
                                      input_available=True, observed_targets=observed,
                                      split=split)
        for j, d in enumerate(decisions):
            ok[k, j] = bool(d.label_eligible)
            if not ok[k, j]:
                reason[k, j] = code[d.label_reasons[0].value]
    return ok, reason


def _arm_split_arrays(bundle: data.InputBundle, table: market_grid.MarketOriginTable,
                      idx: np.ndarray, head: np.ndarray, entry: np.ndarray) -> Dict[str, np.ndarray]:
    """Arm tensors for the table rows ``idx`` under the COMMON masks (targets NaN elsewhere)."""
    slots = table.slot_index[idx]
    if slots.size and (slots < 0).any():
        raise CommonPopulationError("INTERNAL_SLOT_MISSING", "a common row lacks an exact slot")
    origins = table.origin_ts[idx].astype(np.int64)
    instants = table.slot_instants[idx].astype(np.int64)
    if len(idx):
        if not np.array_equal(instants[:, -1], origins):
            raise CommonPopulationError("INTERNAL_ORIGIN_NOT_LAST_SLOT", "last slot must be t")
        if (bundle.ts[slots] != instants).any():
            raise CommonPopulationError("INTERNAL_FUTURE_OR_SHIFTED_READ", "slot bar != slot instant")
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    y = np.full(head.shape, np.nan)
    for j, h in enumerate(table.horizons):
        rows = np.flatnonzero(head[:, j])
        if len(rows):
            y[rows, j] = bundle.price[[pos[int(origins[r]) + h * HOUR] for r in rows]]
    off = (instants - origins[:, None]).astype(np.float64) / HOUR
    adj = np.zeros_like(off)
    if off.shape[1] > 1:
        adj[:, 1:] = np.diff(off, axis=1)
    flags = table.closure_crossed[idx].astype(np.int64)
    if flags.size:
        flags[:, 0] = 0
    return {"row_id": origins, "x": bundle.x[slots] if len(idx) else
            np.zeros((0, slots.shape[1], bundle.x.shape[1])),
            "offsets_hours": off, "adjacent_delta_hours": adj, "closure_crossed": flags,
            "slot_instants": instants,
            "origin_price": bundle.price[[pos[int(t)] for t in origins]] if len(idx)
            else np.zeros(0),
            "target_price": y, "head_mask": head.copy(), "entry_mask": entry.copy(),
            "management_mask": head.copy(),
            "week_ids": table.week_id[idx].astype(str)}


def build_common_population(input_dir: str | Path, experiments: Mapping[str, Mapping[str, Any]],
                            data_designs: Mapping[str, Mapping[str, Any]], *,
                            code_commit: Optional[str] = None) -> CommonPopulation:
    """Build the frozen common population from both sealed arms and the retained input.

    ``experiments``/``data_designs`` map ``mg1`` and ``mg2m`` to the sealed EXPERIMENT and
    DESIGN documents. Every identity is cross-checked; a closed full-population design is a
    typed ``MATCHED_IDENTITY_REQUIRED``. Reads only bars before the TEST start.
    """
    if set(experiments) != set(ARMS) or set(data_designs) != set(ARMS):
        raise CommonPopulationError("ARMS_REQUIRED", f"exactly {ARMS}")
    exp_ids, dd = {}, {}
    for arm in ARMS:
        try:
            got = D.matched_arm(dict(experiments[arm]))
        except D.DesignError as exc:
            raise CommonPopulationError(exc.code, str(exc)) from None
        if got != arm:
            raise CommonPopulationError("ARM_MISMATCH", f"{arm} slot holds the {got} design")
        dd[arm] = validate_matched_data_design(data_designs[arm], arm=arm)
        si = experiments[arm]["sealed_identities"]
        if si["data_design_sha256"] != provenance.canonical_sha256(dd[arm]) or \
                si["data_identity_sha256"] != dd[arm]["data_identity_sha256"]:
            raise CommonPopulationError("DESIGN_IDENTITY_MISMATCH",
                                        f"{arm} EXPERIMENT does not bind this data design")
        exp_ids[arm] = D.design_sha256(dict(experiments[arm]))
    for arm in ARMS:
        partner = D.MATCHED_SCOPE[arm]["partner"]
        if experiments[arm]["sealed_identities"]["partner_data_identity_sha256"] != \
                dd[partner]["data_identity_sha256"]:
            raise CommonPopulationError("DESIGN_IDENTITY_MISMATCH",
                                        f"{arm} does not bind the partner {partner} data identity")
    elapsed = {arm: _elapsed_shim(dd[arm]) for arm in ARMS}
    bundles = {arm: data.load_input(input_dir, elapsed[arm]) for arm in ARMS}
    if bundles["mg1"].files != bundles["mg2m"].files:
        raise CommonPopulationError("SOURCE_DIGEST_MISMATCH", "the input changed between loads")
    bundle = bundles["mg1"]
    from tools.i6dmo.market_grid import build_market_table
    tables = {arm: build_market_table(bundles[arm], elapsed[arm]) for arm in ARMS}
    t1, t2 = tables["mg1"], tables["mg2m"]
    if not (np.array_equal(t1.origin_ts, t2.origin_ts) and np.array_equal(t1.split, t2.split)
            and np.array_equal(t1.week_id, t2.week_id)):
        raise CommonPopulationError("INTERNAL_CANDIDATES_DIFFER", "arms disagree on candidates")
    cal_ref = {"calendar_id": t1.identity["calendar_id"],
               "version": t1.identity["calendar_version"]}
    for arm in ARMS:
        sealed = experiments[arm]["sealed_identities"]["calendar"]
        if (sealed["calendar_id"], sealed["version"]) != (cal_ref["calendar_id"], cal_ref["version"]):
            raise CommonPopulationError("CALENDAR_IDENTITY_MISMATCH",
                                        f"{arm} sealed {sealed} but the input supplies {cal_ref}")
    given, given_reason = _business_labels(bundle, elapsed["mg1"], t1)
    for arm, t in tables.items():
        if not np.array_equal(t.label_ok, given & t.input_ok[:, None]):
            raise CommonPopulationError("INTERNAL_MASK_INCONSISTENT",
                                        f"{arm}: label_ok != C1(input available) & input_ok")
    code = {r: i for i, r in enumerate(HEAD_REASONS)}
    in_code = {r: i for i, r in enumerate(INPUT_REASONS)}
    hs = t1.horizons
    splits: Dict[str, SplitPopulation] = {}
    kept: Dict[str, np.ndarray] = {}
    arm_arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {a: {} for a in ARMS}
    exclusion_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    report: Dict[str, Any] = {"before_intersection": {}, "after_intersection": {},
                              "exclusions": {}, "unsupported_heads": {},
                              "mg2m_slots_subset_of_mg1": True}
    for name in SPLIT_NAMES:
        src = SOURCE_SPLIT[name]
        sel = np.flatnonzero(t1.split == src)
        valid = {a: tables[a].label_ok[sel] for a in ARMS}
        common = valid["mg1"] & valid["mg2m"]
        keep = common.any(axis=1)
        entry = tables["mg1"].entry_ok[sel] & tables["mg2m"].entry_ok[sel] & common
        idx = sel[keep]
        head = common[keep]
        ent = entry[keep]
        counts = {int(h): int(head[:, j].sum()) for j, h in enumerate(hs)}
        pop_arrays = {"origin_row_ids": t1.origin_ts[idx].astype(np.int64), "head_mask": head,
                      "entry_mask": ent, "management_mask": head.copy()}
        splits[name] = SplitPopulation(
            name=name, origin_row_ids=pop_arrays["origin_row_ids"], head_mask=head,
            entry_mask=ent, management_mask=pop_arrays["management_mask"], counts=counts,
            digest=_split_digest(name, pop_arrays, counts))
        for arm in ARMS:
            arm_arrays[arm][name] = _arm_split_arrays(bundles[arm], tables[arm], idx, head, ent)
        if len(idx):
            sub = all(set(b) <= set(a) for a, b in zip(
                arm_arrays["mg1"][name]["slot_instants"].tolist(),
                arm_arrays["mg2m"][name]["slot_instants"].tolist()))
            report["mg2m_slots_subset_of_mg1"] &= bool(sub)
        kept[name] = idx
        # typed exclusions for every candidate origin of the split
        in_ok = {a: tables[a].input_ok[sel] for a in ARMS}
        head_reason = given_reason[sel].copy()
        biz = given[sel]
        for j in range(len(hs)):
            col = head_reason[:, j]
            both_bad = biz[:, j] & ~in_ok["mg1"] & ~in_ok["mg2m"]
            only1 = biz[:, j] & ~in_ok["mg1"] & in_ok["mg2m"]
            only2 = biz[:, j] & in_ok["mg1"] & ~in_ok["mg2m"]
            col[both_bad] = code["INPUT_INVALID_BOTH"]
            col[only1] = code["INPUT_INVALID_MG1_ONLY"]
            col[only2] = code["INPUT_INVALID_MG2M_ONLY"]
            col[common[:, j]] = code["COMMON_VALID"]
            head_reason[:, j] = col
        input_reason = {a: np.asarray([in_code[r or "INPUT_OK"] for r in
                                       (tables[a].input_reason[i] for i in sel)], dtype=np.int8)
                        for a in ARMS}
        exclusion_arrays[name] = {"candidate_row_ids": t1.origin_ts[sel].astype(np.int64),
                                  "head_reason": head_reason,
                                  "input_reason_mg1": input_reason["mg1"],
                                  "input_reason_mg2m": input_reason["mg2m"]}
        report["before_intersection"][name] = {
            arm: {"origins_with_any_valid_head": int(valid[arm].any(axis=1).sum()),
                  "heads": {str(h): int(valid[arm][:, j].sum()) for j, h in enumerate(hs)},
                  "entry_heads": {str(h): int(tables[arm].entry_ok[sel][:, j].sum())
                                  for j, h in enumerate(hs)},
                  "origin_input_reasons": dict(sorted(Counter(
                      INPUT_REASONS[v] for v in input_reason[arm].tolist()).items()))}
            for arm in ARMS}
        report["before_intersection"][name]["candidates"] = int(len(sel))
        report["after_intersection"][name] = {
            "origins": int(len(idx)), "heads": {str(h): counts[h] for h in hs},
            "entry_heads": {str(h): int(ent[:, j].sum()) for j, h in enumerate(hs)},
            "first_origin": data.iso(int(t1.origin_ts[idx].min())) if len(idx) else None,
            "last_origin": data.iso(int(t1.origin_ts[idx].max())) if len(idx) else None}
        report["exclusions"][name] = {
            "heads": {str(h): dict(sorted(Counter(HEAD_REASONS[v] for v in
                                                  head_reason[:, j].tolist()).items()))
                      for j, h in enumerate(hs)},
            "origins_dropped_no_common_head": int((~keep).sum())}
        report["unsupported_heads"][name] = [int(h) for h in hs if counts[h] == 0]
    # ---- shared FIT-only scaler over the UNION of both arms' source bars of common FIT origins
    fit_bars = np.unique(np.concatenate([tables[a].slot_index[kept["FIT"]].reshape(-1)
                                         for a in ARMS]))
    if fit_bars.size and (fit_bars < 0).any():
        raise CommonPopulationError("INTERNAL_SLOT_MISSING", "FIT source bar absent")
    if not fit_bars.size:
        raise CommonPopulationError("NO_COMMON_FIT_ROWS", "the common FIT population is empty")
    xs = bundle.x[fit_bars]
    bar_ts = bundle.ts[fit_bars].astype(np.int64)
    fit_end = int(data.split_bounds(elapsed["mg1"])[0].end.timestamp())
    if int(bar_ts.max()) >= fit_end:
        raise CommonPopulationError("INTERNAL_SCALER_LEAK", "a scaler source bar is >= INNER start")
    scaler = SharedScaler(feature_names=tuple(bundle.features), mean=xs.mean(axis=0),
                          std=xs.std(axis=0), fit_row_digest=_scaler_rows_digest(bar_ts),
                          n_rows=int(len(bar_ts)))
    # ---- FIT-only naive loss scale per horizon on the common FIT mask
    pos = {int(v): i for i, v in enumerate(bundle.ts)}
    naive: Dict[int, Optional[float]] = {}
    fit = splits["FIT"]
    report["naive_loss_scale_status"] = {}
    for j, h in enumerate(hs):
        origins = fit.origin_row_ids[fit.head_mask[:, j]]
        if not len(origins):
            naive[int(h)] = None
            report["naive_loss_scale_status"][str(h)] = "UNSUPPORTED_HEAD"
            continue
        p0 = bundle.price[[pos[int(t)] for t in origins]]
        p1 = bundle.price[[pos[int(t) + h * HOUR] for t in origins]]
        value = math.fsum(np.abs(p1 - p0).tolist()) / len(origins)
        naive[int(h)] = value if value > 0.0 else None
        report["naive_loss_scale_status"][str(h)] = "OK" if value > 0.0 else "ZERO_SCALE_HEAD"
    zero = [f for f, s in zip(bundle.features, scaler.std) if s == 0.0]
    report["scaler_status"] = "ZERO_SCALE" if zero else "OK"
    report["zero_scale_features"] = zero
    report["scaler_source_bars"] = {"n": int(len(bar_ts)), "first": data.iso(int(bar_ts.min())),
                                    "last": data.iso(int(bar_ts.max()))}
    report["reference_closed_mg2_rows"] = {
        "descriptive_only_not_a_quota": True,
        "closed_bundle_rows": dict(market_grid.STUDY_REFERENCE["MG2_MARKET_EXACT"]),
        "mg2m_rows_before_intersection": {
            k.lower() if k != "DEVELOPMENT" else "validation":
                report["before_intersection"][k]["mg2m"]["origins_with_any_valid_head"]
            for k in SPLIT_NAMES}}
    report["head_reason_codes"] = list(HEAD_REASONS)
    report["input_reason_codes"] = list(INPUT_REASONS)
    identities = {
        "contrast_id": D.MATCHED_CONTRAST_ID,
        "matched_population_rules_sha256": D.MATCHED_POPULATION_RULES["rules_sha256"],
        "arms": {arm: {"experiment_id": experiments[arm]["experiment_id"],
                       "family": D.MATCHED_SCOPE[arm]["family"],
                       "experiment_design_sha256": exp_ids[arm],
                       "experiment_seal_sha256": experiments[arm]["seal_sha256"],
                       "data_design_sha256": provenance.canonical_sha256(dd[arm]),
                       "data_identity_sha256": dd[arm]["data_identity_sha256"],
                       "observations": D.MATCHED_SCOPE[arm]["observations"],
                       "spacing_open_market_hours": D.MATCHED_SCOPE[arm]["spacing_open_market_hours"]}
                 for arm in ARMS},
        "source": {"inputs": dict(sorted(bundle.files.items())),
                   "input_sha256": bundle.files["INPUT.npz"],
                   "source_clock": {k: bundle.clock.get(k) for k in
                                    ("status", "timestamp_semantics", "availability_policy")},
                   "features": list(bundle.features),
                   "feature_manifest_sha256": provenance.EXPECTED_MANIFEST_SHA256,
                   "availability_contract_sha256": provenance.EXPECTED_CONTRACT_SHA256},
        "calendar": {"calendar_id": cal_ref["calendar_id"], "version": cal_ref["version"],
                     "status": t1.identity["calendar_status"],
                     "sha256": bundle.files["CALENDAR.json"]},
        "splits": {name: {**b.to_json(), "population_name": name}
                   for name, b in zip(SPLIT_NAMES, data.split_bounds(elapsed["mg1"]))},
        "test": {"materialized": False, "start": elapsed["mg1"]["splits"]["test"]["start"]},
        "code": {"commit": code_commit, "files_sha256": _code_digests()},
    }
    pop = CommonPopulation(splits=splits, scaler=scaler, naive_loss_scale=naive,
                           identities=identities, digest="", report=report,
                           arm_arrays=arm_arrays, exclusion_arrays=exclusion_arrays)
    body = _population_body(pop)
    _check_no_test(pop)
    return CommonPopulation(splits=splits, scaler=scaler, naive_loss_scale=naive,
                            identities=identities, digest=_population_digest(body),
                            report=report, arm_arrays=arm_arrays,
                            exclusion_arrays=exclusion_arrays)


def _code_digests() -> Dict[str, str]:
    here = Path(__file__).resolve().parent
    return {n: provenance.file_sha256(here / n)
            for n in ("common_population.py", "market_grid.py", "data.py", "design.py")}


def _check_no_test(pop: CommonPopulation) -> None:
    start = int(data._instant(pop.identities["test"]["start"]).timestamp())
    for name, sp in pop.splits.items():
        if name not in SPLIT_NAMES:
            raise CommonPopulationError("SPLIT_UNKNOWN", f"{name}; TEST is never materialized")
        last = sp.origin_row_ids.max() if len(sp.origin_row_ids) else -1
        if last >= start:
            raise CommonPopulationError("TEST_ROWS_PRESENT", f"{name} holds a TEST origin")
        targets = sp.origin_row_ids[:, None] + np.asarray(pop.horizons, dtype=np.int64) * HOUR
        if int(targets[sp.head_mask].max(initial=-1)) >= start:
            raise CommonPopulationError("TEST_ROWS_PRESENT", f"{name} labels a TEST bar")
        for arm, arrs in pop.arm_arrays.items():
            a = arrs.get(name)
            if a is not None and len(a["row_id"]) and int(a["slot_instants"].max()) >= start:
                raise CommonPopulationError("TEST_ROWS_PRESENT", f"{arm} {name} reads a TEST bar")


# =========================================================================== digests / write

def _array_files(pop: CommonPopulation) -> Dict[str, Dict[str, np.ndarray]]:
    files: Dict[str, Dict[str, np.ndarray]] = {}
    for name, sp in pop.splits.items():
        files[f"POPULATION_{name}.npz"] = sp.arrays()
        files[f"EXCLUSIONS_{name}.npz"] = pop.exclusion_arrays[name]
        for arm in ARMS:
            files[f"ARM_{arm}_{name}.npz"] = pop.arm_arrays[arm][name]
    return files


def _population_body(pop: CommonPopulation, *, array_digests: Optional[Mapping[str, str]] = None,
                     scaler_rows_digest: Optional[str] = None) -> Dict[str, Any]:
    """Everything the population digest covers (file container digests excluded)."""
    digests = dict(array_digests) if array_digests is not None else {
        name: provenance.array_sha256(arrs) for name, arrs in _array_files(pop).items()}
    splits = {}
    for name, sp in pop.splits.items():
        splits[name] = {
            "n_origins": int(len(sp.origin_row_ids)),
            "counts": {str(h): int(v) for h, v in sp.counts.items()},
            "digest": sp.digest,
            "row_ids_sha256": provenance.array_sha256({"origin_row_ids": sp.origin_row_ids}),
            "head_mask_sha256": provenance.array_sha256({"head_mask": sp.head_mask}),
            "entry_mask_sha256": provenance.array_sha256({"entry_mask": sp.entry_mask}),
            "management_mask_sha256": provenance.array_sha256({"management_mask": sp.management_mask}),
            "path": f"POPULATION_{name}.npz", "arrays_sha256": digests[f"POPULATION_{name}.npz"],
            "exclusions_path": f"EXCLUSIONS_{name}.npz",
            "exclusions_sha256": digests[f"EXCLUSIONS_{name}.npz"],
            "arms": {arm: {"path": f"ARM_{arm}_{name}.npz",
                           "arrays_sha256": digests[f"ARM_{arm}_{name}.npz"]} for arm in ARMS},
        }
    ident = pop.identities
    return {
        "schema": SCHEMA, "horizons_hours": list(pop.horizons), "split_names": list(SPLIT_NAMES),
        "splits": splits,
        "scaler": {**pop.scaler.to_json(), "rows_path": "SCALER_ROWS.npz",
                   "rows_sha256": scaler_rows_digest or pop.scaler.fit_row_digest},
        "naive_loss_scale": {str(h): v for h, v in pop.naive_loss_scale.items()},
        "identities": ident,
        "identity_digests": {
            "arms": provenance.canonical_sha256(ident["arms"]),
            "source": provenance.canonical_sha256(ident["source"]),
            "calendar": provenance.canonical_sha256(ident["calendar"]),
            "splits": provenance.canonical_sha256(ident["splits"])},
        "report": pop.report,
        "test_materialized": False,
    }


def _population_digest(body: Mapping[str, Any]) -> str:
    return provenance.canonical_sha256({k: v for k, v in body.items() if k not in ("digest", "files")})


def write_common_population(pop: CommonPopulation, out_dir: str | Path) -> Dict[str, Any]:
    """Write the frozen arrays + POPULATION.json; return the small seal (never overwrites)."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise CommonPopulationError("POPULATION_EXISTS", "a frozen population is never overwritten")
    if not pop.arm_arrays or not pop.exclusion_arrays:
        raise CommonPopulationError("POPULATION_NOT_BUILT", "only a freshly built population is written")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arrays in _array_files(pop).items():
        np.savez(out / name, **arrays)
        files[name] = provenance.file_sha256(out / name)
    bar_ts = _scaler_bar_ts(pop)
    np.savez(out / "SCALER_ROWS.npz", fit_source_bar_ts=bar_ts)
    files["SCALER_ROWS.npz"] = provenance.file_sha256(out / "SCALER_ROWS.npz")
    body = _population_body(pop)
    digest = _population_digest(body)
    if digest != pop.digest:
        raise CommonPopulationError("POPULATION_DIGEST_MISMATCH", "in-memory population changed")
    body["digest"] = digest
    body["files"] = files
    text = json.dumps(body, indent=1, sort_keys=True, allow_nan=False)
    tmp = out / "POPULATION.json.tmp"
    tmp.write_text(text)
    os.replace(tmp, out / "POPULATION.json")
    return seal_summary(body, provenance.file_sha256(out / "POPULATION.json"))


def _scaler_bar_ts(pop: CommonPopulation) -> np.ndarray:
    """Recompute the scaler source bars from the FIT arm arrays (union of both arms' slots)."""
    bars = np.unique(np.concatenate([pop.arm_arrays[a]["FIT"]["slot_instants"].reshape(-1)
                                     for a in ARMS]))
    if _scaler_rows_digest(bars) != pop.scaler.fit_row_digest:
        raise CommonPopulationError("SCALER_ROWS_MISMATCH", "FIT slots disagree with the scaler")
    return bars.astype(np.int64)


def seal_summary(body: Mapping[str, Any], population_json_sha256: str) -> Dict[str, Any]:
    """Small committable seal: hashes, counts, exclusions summary, scaler digest. No arrays."""
    return {
        "schema": SEAL_SCHEMA, "status": "SEALED_BEFORE_ANY_FIT",
        "population_digest": body["digest"], "population_json_sha256": population_json_sha256,
        "population_schema": SCHEMA, "horizons_hours": body["horizons_hours"],
        "splits": {name: {k: info[k] for k in ("n_origins", "counts", "digest", "row_ids_sha256",
                                                "head_mask_sha256", "entry_mask_sha256",
                                                "management_mask_sha256", "arrays_sha256",
                                                "exclusions_sha256")}
                   | {"arms": {a: v["arrays_sha256"] for a, v in info["arms"].items()}}
                   for name, info in body["splits"].items()},
        "counts_before_intersection": body["report"]["before_intersection"],
        "counts_after_intersection": body["report"]["after_intersection"],
        "exclusions": body["report"]["exclusions"],
        "unsupported_heads": body["report"]["unsupported_heads"],
        "scaler": {"fit_row_digest": body["scaler"]["fit_row_digest"],
                   "n_rows": body["scaler"]["n_rows"], "status": body["report"]["scaler_status"],
                   "source_bars": body["report"]["scaler_source_bars"],
                   "values_sha256": provenance.canonical_sha256(
                       {"mean": body["scaler"]["mean"], "std": body["scaler"]["std"]})},
        "naive_loss_scale": body["naive_loss_scale"],
        "naive_loss_scale_status": body["report"]["naive_loss_scale_status"],
        "identities": body["identities"], "identity_digests": body["identity_digests"],
        "mg2m_slots_subset_of_mg1": body["report"]["mg2m_slots_subset_of_mg1"],
        "reference_closed_mg2_rows": body["report"]["reference_closed_mg2_rows"],
        "files": body["files"], "test_materialized": False, "fits_run": 0,
    }


# =========================================================================== load / verify

def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise CommonPopulationError("POPULATION_FILE_MISSING", path.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_common_population(path: str | Path, *, expected_digest: str,
                           input_dir: Optional[str | Path] = None) -> CommonPopulation:
    """Verified read: every row/head/mask/arm/scaler/identity digest is recomputed.

    Typed refusals: ``ROW_DIGEST_MISMATCH`` (changed row ids), ``HEAD_MASK_DIGEST_MISMATCH``,
    ``MASK_DIGEST_MISMATCH``, ``ARM_ARRAYS_MISMATCH``, ``EXCLUSIONS_MISMATCH``,
    ``SCALER_ROWS_MISMATCH``, ``CALENDAR_DIGEST_MISMATCH``, ``SOURCE_DIGEST_MISMATCH``,
    ``DESIGN_IDENTITY_MISMATCH``, ``FILE_DIGEST_MISMATCH``, ``POPULATION_DIGEST_MISMATCH``.
    With ``input_dir`` the retained input/calendar files must still hash to the sealed source.
    """
    root = Path(path)
    if root.is_file():
        root = root.parent
    pj = root / "POPULATION.json"
    if not pj.is_file():
        raise CommonPopulationError("POPULATION_MISSING", str(pj.name))
    body = json.loads(pj.read_text())
    if body.get("schema") != SCHEMA:
        raise CommonPopulationError("POPULATION_SCHEMA", str(body.get("schema")))
    if list(body.get("split_names") or ()) != list(SPLIT_NAMES) or \
            set(body.get("splits") or {}) != set(SPLIT_NAMES):
        raise CommonPopulationError("SPLIT_UNKNOWN", "FIT/INNER/DEVELOPMENT only; TEST never exists")
    ident = body["identities"]
    for key, code in (("calendar", "CALENDAR_DIGEST_MISMATCH"), ("source", "SOURCE_DIGEST_MISMATCH"),
                      ("arms", "DESIGN_IDENTITY_MISMATCH"), ("splits", "SPLIT_BOUNDS_MISMATCH")):
        if provenance.canonical_sha256(ident[key]) != body["identity_digests"][key]:
            raise CommonPopulationError(code, f"identities.{key} changed after sealing")
    if ident["calendar"]["sha256"] != ident["source"]["inputs"].get("CALENDAR.json"):
        raise CommonPopulationError("CALENDAR_DIGEST_MISMATCH", "calendar != source CALENDAR.json")
    if input_dir is not None:
        verify_source(body, input_dir)
    splits: Dict[str, SplitPopulation] = {}
    digests: Dict[str, str] = {}
    verified_arm: Dict[str, Dict[str, str]] = {a: {} for a in ARMS}
    hs = tuple(int(h) for h in body["horizons_hours"])
    if hs != HORIZONS:
        raise CommonPopulationError("HORIZONS_INVALID", str(hs))
    for name in SPLIT_NAMES:
        info = body["splits"][name]
        arrays = _load_npz(root / info["path"])
        if set(arrays) != {"origin_row_ids", "head_mask", "entry_mask", "management_mask"}:
            raise CommonPopulationError("POPULATION_ARRAYS_INVALID", name)
        rid, head = arrays["origin_row_ids"], arrays["head_mask"]
        if provenance.array_sha256({"origin_row_ids": rid}) != info["row_ids_sha256"]:
            raise CommonPopulationError("ROW_DIGEST_MISMATCH", f"{name} origin row ids changed")
        if provenance.array_sha256({"head_mask": head}) != info["head_mask_sha256"]:
            raise CommonPopulationError("HEAD_MASK_DIGEST_MISMATCH", f"{name} head mask changed")
        for key in ("entry_mask", "management_mask"):
            if provenance.array_sha256({key: arrays[key]}) != info[f"{key}_sha256"]:
                raise CommonPopulationError("MASK_DIGEST_MISMATCH", f"{name} {key} changed")
        _check_split_arrays(name, arrays)
        counts = {int(h): int(head[:, j].sum()) for j, h in enumerate(hs)}
        if {str(h): v for h, v in counts.items()} != info["counts"]:
            raise CommonPopulationError("HEAD_MASK_DIGEST_MISMATCH", f"{name} per-head counts")
        sp = SplitPopulation(name=name, origin_row_ids=rid, head_mask=head,
                             entry_mask=arrays["entry_mask"],
                             management_mask=arrays["management_mask"], counts=counts,
                             digest=_split_digest(name, arrays, counts))
        if sp.digest != info["digest"]:
            raise CommonPopulationError("POPULATION_DIGEST_MISMATCH", f"{name} split digest")
        splits[name] = sp
        digests[info["path"]] = provenance.array_sha256(arrays)
        exc = _load_npz(root / info["exclusions_path"])
        digests[info["exclusions_path"]] = provenance.array_sha256(exc)
        if digests[info["exclusions_path"]] != info["exclusions_sha256"]:
            raise CommonPopulationError("EXCLUSIONS_MISMATCH", f"{name} exclusions changed")
        for arm in ARMS:
            arm_info = info["arms"][arm]
            arrs = _load_npz(root / arm_info["path"])
            digests[arm_info["path"]] = provenance.array_sha256(arrs)
            if digests[arm_info["path"]] != arm_info["arrays_sha256"]:
                raise CommonPopulationError("ARM_ARRAYS_MISMATCH", f"{arm} {name} arrays changed")
            _check_arm_against_split(arm, sp, arrs)
            verified_arm[arm][name] = digests[arm_info["path"]]
    scaler_rows = _load_npz(root / body["scaler"]["rows_path"])["fit_source_bar_ts"]
    if _scaler_rows_digest(scaler_rows) != body["scaler"]["fit_row_digest"] or \
            len(scaler_rows) != body["scaler"]["n_rows"]:
        raise CommonPopulationError("SCALER_ROWS_MISMATCH", "scaler source rows changed")
    sc = body["scaler"]
    scaler = SharedScaler(feature_names=tuple(sc["feature_names"]),
                          mean=np.asarray(sc["mean"], dtype=np.float64),
                          std=np.asarray(sc["std"], dtype=np.float64),
                          fit_row_digest=sc["fit_row_digest"], n_rows=int(sc["n_rows"]))
    naive = {int(h): v for h, v in body["naive_loss_scale"].items()}
    pop = CommonPopulation(splits=splits, scaler=scaler, naive_loss_scale=naive,
                           identities=ident, digest="", report=body["report"],
                           source_dir=str(root))
    recomputed = _population_digest(_population_body(pop, array_digests=digests,
                                                      scaler_rows_digest=sc["rows_sha256"]))
    if recomputed != body.get("digest"):
        raise CommonPopulationError("POPULATION_DIGEST_MISMATCH", "POPULATION.json body changed")
    if recomputed != expected_digest:
        raise CommonPopulationError("POPULATION_DIGEST_MISMATCH",
                                    f"sealed {recomputed[:12]} != expected {str(expected_digest)[:12]}")
    for name, file_sha in body["files"].items():
        if provenance.file_sha256(root / name) != file_sha:
            raise CommonPopulationError("FILE_DIGEST_MISMATCH", name)
    _check_no_test(pop)
    return CommonPopulation(splits=splits, scaler=scaler, naive_loss_scale=naive,
                            identities=ident, digest=recomputed, report=body["report"],
                            source_dir=str(root), verified_arm_digests=verified_arm)


def verify_source(body_or_pop: Any, input_dir: str | Path) -> Dict[str, str]:
    """The retained input/calendar files must hash to the sealed source identities."""
    ident = body_or_pop.identities if isinstance(body_or_pop, CommonPopulation) \
        else body_or_pop["identities"]
    root = Path(input_dir)
    got = {}
    for name, sha in ident["source"]["inputs"].items():
        if not (root / name).is_file():
            raise CommonPopulationError("SOURCE_DIGEST_MISMATCH", f"{name} missing")
        got[name] = provenance.file_sha256(root / name)
    if got.get("CALENDAR.json") != ident["calendar"]["sha256"]:
        raise CommonPopulationError("CALENDAR_DIGEST_MISMATCH", "CALENDAR.json changed")
    for name, sha in ident["source"]["inputs"].items():
        if got[name] != sha:
            raise CommonPopulationError("SOURCE_DIGEST_MISMATCH", f"{name} changed")
    return got


def _check_split_arrays(name: str, arrays: Mapping[str, np.ndarray]) -> None:
    rid, head = arrays["origin_row_ids"], arrays["head_mask"]
    n = len(rid)
    if rid.dtype != np.int64 or (n > 1 and np.any(np.diff(rid) <= 0)):
        raise CommonPopulationError("ROW_DIGEST_MISMATCH", f"{name} row ids not increasing int64")
    for key in ("head_mask", "entry_mask", "management_mask"):
        if arrays[key].dtype != bool or arrays[key].shape != (n, len(HORIZONS)):
            raise CommonPopulationError("MASK_DIGEST_MISMATCH", f"{name} {key} shape/dtype")
    if n and not head.any(axis=1).all():
        raise CommonPopulationError("HEAD_MASK_DIGEST_MISMATCH", f"{name} row without a common head")
    if (arrays["entry_mask"] & ~head).any() or not np.array_equal(arrays["management_mask"], head):
        raise CommonPopulationError("MASK_DIGEST_MISMATCH", f"{name} entry/management inconsistent")


def _check_arm_against_split(arm: str, sp: SplitPopulation, arrs: Mapping[str, np.ndarray]) -> None:
    if not np.array_equal(arrs["row_id"], sp.origin_row_ids):
        raise CommonPopulationError("POPULATION_ARM_MISMATCH", f"{arm} {sp.name} row order")
    for key in ("head_mask", "entry_mask", "management_mask"):
        if not np.array_equal(arrs[key], getattr(sp, key)):
            raise CommonPopulationError("POPULATION_ARM_MISMATCH", f"{arm} {sp.name} {key}")
    slots = D.MATCHED_SCOPE[arm]["observations"]
    if arrs["x"].shape[1:] != (slots, len(provenance.MANIFEST)):
        raise CommonPopulationError("POPULATION_ARM_MISMATCH", f"{arm} x shape {arrs['x'].shape}")
    if len(sp.origin_row_ids):
        if not np.array_equal(arrs["slot_instants"][:, -1], sp.origin_row_ids) or \
                not np.all(arrs["offsets_hours"][:, -1] == 0):
            raise CommonPopulationError("POPULATION_ARM_MISMATCH", f"{arm} last slot is not t")
        if np.isnan(arrs["target_price"][sp.head_mask]).any() or \
                not np.isnan(arrs["target_price"][~sp.head_mask]).all():
            raise CommonPopulationError("POPULATION_ARM_MISMATCH", f"{arm} targets vs head mask")


# =========================================================================== per-arm accessor

def arm_tensors(pop: CommonPopulation, arm_design: Mapping[str, Any], split: str) -> ArmTensors:
    """The arm's tensors for EXACTLY the frozen common rows/heads of ``split``.

    ``arm_design`` is the sealed EXPERIMENT document of the arm. The closed full-population MG2,
    the elapsed hourly control or any design whose digest the population does not bind is a
    typed refusal (``MATCHED_IDENTITY_REQUIRED`` / ``DESIGN_IDENTITY_MISMATCH``).
    """
    if split not in SPLIT_NAMES:
        raise CommonPopulationError("SPLIT_UNKNOWN", f"{split!r}; TEST is never materialized")
    try:
        arm = D.matched_arm(dict(arm_design))
    except D.DesignError as exc:
        raise CommonPopulationError(exc.code, str(exc)) from None
    sealed = pop.identities["arms"][arm]
    if D.design_sha256(dict(arm_design)) != sealed["experiment_design_sha256"]:
        raise CommonPopulationError("DESIGN_IDENTITY_MISMATCH",
                                    f"{arm} design is not the one this population sealed")
    sp = pop.splits[split]
    if pop.arm_arrays:
        arrs = pop.arm_arrays[arm][split]
    elif pop.source_dir:
        # D3: bind to the digest the verified load RECOMPUTED, never to a re-read stored string
        expected = (pop.verified_arm_digests.get(arm) or {}).get(split)
        if not expected:
            raise CommonPopulationError("POPULATION_NOT_VERIFIED",
                                        "source_dir access requires load_common_population")
        arrs = _load_npz(Path(pop.source_dir) / f"ARM_{arm}_{split}.npz")
        if provenance.array_sha256(arrs) != expected:
            raise CommonPopulationError("ARM_ARRAYS_MISMATCH",
                                        f"{arm} {split} arrays changed after the verified load")
    else:
        raise CommonPopulationError("POPULATION_NOT_BUILT", "no arm arrays and no source dir")
    _check_arm_against_split(arm, sp, arrs)
    return ArmTensors(
        arm=arm, split=split, experiment_id=sealed["experiment_id"], row_id=arrs["row_id"],
        row_ids=np.asarray([data.row_id(int(t)) for t in arrs["row_id"]], dtype=str),
        x=arrs["x"], offsets_hours=arrs["offsets_hours"],
        adjacent_delta_hours=arrs["adjacent_delta_hours"],
        closure_crossed=arrs["closure_crossed"], slot_instants=arrs["slot_instants"],
        origin_price=arrs["origin_price"], target_price=arrs["target_price"],
        head_mask=arrs["head_mask"], entry_mask=arrs["entry_mask"],
        management_mask=arrs["management_mask"], week_ids=arrs["week_ids"])


def replay_against_input(tensors: ArmTensors, input_dir: str | Path) -> Dict[str, Any]:
    """Independent replay: inputs at the exact slot bars, p(t) and exact p(t+h) from disk."""
    with np.load(Path(input_dir) / "INPUT.npz", allow_pickle=False) as z:
        ts, x, price = z["ts"], z["x"], z["price"]
    pos = {int(v): i for i, v in enumerate(ts)}
    for r, t in enumerate(tensors.row_id.tolist()):
        idx = [pos.get(int(s)) for s in tensors.slot_instants[r]]
        if any(i is None for i in idx) or not np.array_equal(x[idx], tensors.x[r]):
            raise CommonPopulationError("REPLAY_MISMATCH", f"{data.row_id(t)} inputs")
        if price[pos[int(t)]] != tensors.origin_price[r]:
            raise CommonPopulationError("REPLAY_MISMATCH", f"{data.row_id(t)} origin price")
        for j, h in enumerate(HORIZONS):
            if tensors.head_mask[r, j]:
                q = pos.get(int(t) + h * HOUR)
                if q is None or price[q] != tensors.target_price[r, j]:
                    raise CommonPopulationError("REPLAY_MISMATCH", f"{data.row_id(t)} h{h}")
    return {"status": "REPLAYED", "rows": int(len(tensors.row_id)), "arm": tensors.arm,
            "split": tensors.split}


# =========================================================================== CLI

def _load_matched_dir(matched_dir: Path) -> Tuple[dict, dict]:
    exps, dds = {}, {}
    for arm in ARMS:
        exps[arm] = D.load_design(matched_dir / f"EXPERIMENT.{arm}.json")
        dds[arm] = json.loads((matched_dir / f"DESIGN.{arm}.json").read_text())
    return exps, dds


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.i6dmo.common_population")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build + write the frozen common population (no fit)")
    b.add_argument("--matched-dir", required=True, help="dir with EXPERIMENT/DESIGN.{mg1,mg2m}.json")
    b.add_argument("--input-dir", required=True, help="private no-STLFSI4 input directory")
    b.add_argument("--out", required=True, help="private output directory (must be empty/new)")
    b.add_argument("--seal-out", required=True, help="small committable seal JSON path")
    b.add_argument("--code-commit", default=None)
    v = sub.add_parser("verify", help="verify a written population against an expected digest")
    v.add_argument("--population-dir", required=True)
    v.add_argument("--expected-digest", required=True)
    v.add_argument("--input-dir", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            exps, dds = _load_matched_dir(Path(args.matched_dir))
            pop = build_common_population(args.input_dir, exps, dds, code_commit=args.code_commit)
            seal = write_common_population(pop, args.out)
            seal["private_location"] = "$EXPERIMENT_STORAGE_ROOT/i6dmo/matched-short-20261015/population"
            Path(args.seal_out).write_text(json.dumps(seal, indent=1, sort_keys=True) + "\n")
            print(json.dumps({"status": seal["status"], "population_digest": seal["population_digest"],
                              "population_json_sha256": seal["population_json_sha256"],
                              "after": {k: v["heads"] for k, v in
                                        seal["counts_after_intersection"].items()}}, sort_keys=True))
        else:
            pop = load_common_population(args.population_dir, expected_digest=args.expected_digest,
                                         input_dir=args.input_dir)
            print(json.dumps({"status": "VERIFIED", "population_digest": pop.digest,
                              "counts": {k: v.counts for k, v in pop.splits.items()}},
                             sort_keys=True))
    except (CommonPopulationError, data.DataContractError, D.DesignError) as exc:
        print(json.dumps({"status": "REFUSED", "code": getattr(exc, "code", type(exc).__name__),
                          "detail": str(exc)[:300]}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
