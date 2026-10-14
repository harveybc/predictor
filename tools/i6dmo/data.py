"""I6-DMO data lane: exact DS2 origin windows, business masks and a frozen bundle.

Implements order section B.2/C and PLAN sections 2 and 4 for the static short
experiment (and, unchanged, any later design with the same schema):

* An origin ``t`` is an OBSERVED hourly bar on the UTC bar-END clock of the
  retained input (its own value included). Every observed bar inside a split
  is a candidate; origins are never thinned to the input spacing.
* The input window has ``observations`` slot INSTANTS ``s = t - span .. t`` at
  ``spacing`` hours (Satoshi ruling 2026-10-14, rule ``DS2_ASOF_LAST_AVAILABLE``):
  each slot takes, per feature, the last retained observation whose
  ``available_time`` (its UTC bar end) is ``<= s`` and whose value is finite.
  Never the next row, never interpolation. Known past observations across a
  weekend or a vendor gap are therefore retained (PLAN section 4). Per slot
  and feature the staleness ``s - observation_time`` is recorded (aux arrays,
  not a model input); origins whose staleness exceeds the longest calendar
  closure are a declared, counted category, not a refusal. An origin is
  refused only when some feature has NO finite observation ``<= s`` in the
  whole retained history (``INPUT_NO_OBSERVATION_AVAILABLE``) or by C1.
* Each head ``h`` needs the exact bar ``t + h``; masks come from the audited
  C1 component (``tools.i6n_contracts.eligibility``): MANAGEMENT ``M`` is the
  forecasting support S_h, ENTRY ``M`` is the narrower opening support.
  The origin's split must also contain the target (cross-boundary labels are
  purged, ``TARGET_AT_OR_AFTER_SPLIT_END``); FIT labels must be available by
  the inner boundary.
* TRAIN ``[2020-01-01, 2024-01-01)`` = FIT + INNER (last eight weeks);
  VALIDATION ``[2024-01-01, 2025-01-01)``; TEST is never materialized and an
  input holding a TEST-year bar is refused before any array is built.
* No sampling, row caps or shortened periods exist in this module.

The calendar is the one supplied with the input (``CALENDAR.json``); the I6N
calendar is ``FX_NY_17_ASSUMED_NOT_BROKER`` and is labelled ASSUMED.
"""

from __future__ import annotations

import bisect
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tools.i6dmo import provenance
from tools.i6n_contracts import eligibility as c1

SCHEMA = "i6dmo.data_bundle.v1"
UTC = timezone.utc
HOUR = 3600
ASSET = "EURUSD"
SPLITS = ("FIT", "INNER", "VALIDATION")
CONTRACT_SPLITS = {
    "train": ("2020-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "validation": ("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    "test": ("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
}
INNER_TAIL_WEEKS = 8
DESIGN_KEYS = frozenset({"schema", "status", "experiment_id", "asset", "seed", "repetitions",
                         "evaluation_mode", "retraining_frequency", "input", "horizons_hours",
                         "feature_policy", "splits", "business", "model", "training", "metrics",
                         "execution"})
#: Present only in a long design sealed by the model lane; they never enter the data identity.
SEAL_KEYS = frozenset({"derived_from", "seal_sha256"})
DATA_CONTRACT_KEYS = ("asset", "input", "horizons_hours", "feature_policy", "splits", "business")
INPUT_KEYS = frozenset({"value_mode", "history_span_hours", "spacing_hours", "observations",
                        "include_origin", "issuance_hours", "smoothing", "require_exact_timestamps"})
SPLIT_KEYS = frozenset({"train", "inner_tail_weeks", "validation", "test", "end_exclusive",
                        "purge_by_label_availability"})
ORIGIN_REASONS = ("ORIGIN_OUTSIDE_CALENDAR_WEEK", "INPUT_NO_OBSERVATION_AVAILABLE",
                  "NO_VALID_HEAD")
SAMPLING_RULE = "DS2_ASOF_LAST_AVAILABLE"      # short family; long family is DS12_...
SAMPLING_DESCRIPTION = ("each slot instant s takes, per feature, the last retained finite "
                        "observation with available_time (UTC bar end) <= s; never the next row, "
                        "never interpolation; staleness s - observation_time recorded per slot "
                        "and feature; refusal only when no observation <= s exists")
INPUT_FILES = ("INPUT.npz", "CLOCK.json", "CALENDAR.json", "DERIVATION.json",
               "PREPARATION.json", "CLOCK_SCOPE.json")


class DataContractError(ValueError):
    """Typed refusal of the data contract; ``code`` is stable."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _instant(text: str) -> datetime:
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise DataContractError("TIMESTAMP_INVALID", f"{text} has no offset")
    return value.astimezone(UTC)


def _epoch(value: datetime) -> int:
    return int(value.timestamp())


def iso(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(int(epoch_seconds), UTC).isoformat()


def row_id(epoch_seconds: int, asset: str = ASSET) -> str:
    """Stable row identity: asset and UTC origin instant."""
    return f"{asset}|{iso(epoch_seconds)}"


# --------------------------------------------------------------------------- design

@dataclass(frozen=True)
class WindowSpec:
    """Exact sparse input window: ``observations`` bars at ``spacing`` hours ending at t."""

    span_hours: int
    spacing_hours: int
    observations: int
    issuance_hours: int = 1

    @classmethod
    def from_design(cls, design: Mapping[str, Any]) -> "WindowSpec":
        raw = design.get("input")
        if not isinstance(raw, Mapping) or set(raw) != INPUT_KEYS:
            raise DataContractError("WINDOW_NOT_CONTRACT", f"input keys must be {sorted(INPUT_KEYS)}")
        ints = [raw[k] for k in ("history_span_hours", "spacing_hours", "observations",
                                 "issuance_hours")]
        if any(type(v) is not int or v <= 0 for v in ints):
            raise DataContractError("WINDOW_NOT_CONTRACT", "window fields must be positive ints")
        span, spacing, obs, issuance = ints
        if (raw["value_mode"] != "RAW" or raw["smoothing"] is not False
                or raw["require_exact_timestamps"] is not True or raw["include_origin"] is not True
                or issuance != 1 or span != spacing * (obs - 1)):
            raise DataContractError(
                "WINDOW_NOT_CONTRACT",
                "need RAW, no smoothing, exact timestamps, origin included, hourly issuance and "
                "span == spacing * (observations - 1)")
        return cls(span, spacing, obs, issuance)

    def offsets_hours(self) -> Tuple[int, ...]:
        return tuple(range(-self.span_hours, 1, self.spacing_hours))

    def to_json(self) -> dict:
        return {"span_hours": self.span_hours, "spacing_hours": self.spacing_hours,
                "observations": self.observations, "issuance_hours": self.issuance_hours,
                "offsets_hours": list(self.offsets_hours())}


def horizons(design: Mapping[str, Any]) -> Tuple[int, ...]:
    hs = design.get("horizons_hours")
    if (not isinstance(hs, list) or not hs or any(type(h) is not int or h <= 0 for h in hs)
            or len(set(hs)) != len(hs)):
        raise DataContractError("HORIZONS_INVALID", f"{hs!r}")
    return tuple(hs)


def validate_design(design: Mapping[str, Any], *, _allow_noncontract_splits: bool = False) -> dict:
    """Check the design fields this lane consumes; unknown top-level fields are refused."""
    if not isinstance(design, Mapping):
        raise DataContractError("DESIGN_INVALID", "design must be an object")
    unknown = set(design) - DESIGN_KEYS - SEAL_KEYS
    if unknown:
        raise DataContractError("UNKNOWN_DESIGN_FIELD", f"{sorted(unknown)}")
    if design.get("schema") != "i6dmo.design.v1" or design.get("asset") != ASSET:
        raise DataContractError("DESIGN_INVALID", "schema i6dmo.design.v1 and asset EURUSD required")
    WindowSpec.from_design(design)
    horizons(design)
    split_bounds(design, _allow_noncontract_splits=_allow_noncontract_splits)
    policy = design.get("feature_policy") or {}
    if ("fred.stress.stlfsi4.level" not in policy.get("exclude", ())
            or policy.get("reject_stlfsi4_aliases") is not True):
        raise DataContractError("FEATURE_POLICY_INVALID", "STLFSI4 exclusion must be explicit")
    business = design.get("business") or {}
    if (business.get("per_output_masks") is not True
            or business.get("keep_row_if_any_head_valid") is not True
            or business.get("preserve_observed_past_input_history") is not True):
        raise DataContractError("BUSINESS_INVALID", "per-output masks / keep-row rules required")
    return dict(design)


def split_bounds(design: Mapping[str, Any], *,
                 _allow_noncontract_splits: bool = False) -> Tuple[c1.SplitBounds, ...]:
    """FIT, INNER and VALIDATION as C1 ``SplitBounds`` (target-instant membership).

    ``_allow_noncontract_splits`` exists only for falsification tests that place
    a boundary mid-week; production callers never pass it.
    """
    raw = design.get("splits")
    if not isinstance(raw, Mapping) or set(raw) != SPLIT_KEYS:
        raise DataContractError("SPLIT_NOT_CONTRACT", f"split keys must be {sorted(SPLIT_KEYS)}")
    if (raw["end_exclusive"] is not True or raw["purge_by_label_availability"] is not True
            or raw["inner_tail_weeks"] != INNER_TAIL_WEEKS or raw["test"].get("open") is not False):
        raise DataContractError("SPLIT_NOT_CONTRACT", "exclusive ends, purge, 8 inner weeks, TEST closed")
    got = {k: (raw[k]["start"], raw[k]["end"]) for k in ("train", "validation", "test")}
    if not _allow_noncontract_splits:
        for key, (start, end) in CONTRACT_SPLITS.items():
            if (_instant(got[key][0]), _instant(got[key][1])) != (_instant(start), _instant(end)):
                raise DataContractError("SPLIT_NOT_CONTRACT",
                                        f"{key} {got[key]} != contract {(start, end)}")
    train0, train1 = (_instant(v) for v in got["train"])
    val0, val1 = (_instant(v) for v in got["validation"])
    test0 = _instant(got["test"][0])
    if not (train0 < train1 <= val0 < val1 <= test0):
        raise DataContractError("SPLIT_NOT_CONTRACT", "splits must be ordered and disjoint")
    inner0 = train1 - timedelta(weeks=INNER_TAIL_WEEKS)
    return (c1.SplitBounds("FIT", train0, inner0, label_available_by=inner0),
            c1.SplitBounds("INNER", inner0, train1, label_available_by=train1),
            c1.SplitBounds("VALIDATION", val0, val1, label_available_by=val1))


def sampling_rule_name(window: "WindowSpec") -> str:
    """Rule name parametrized by input spacing (DS2 short, DS12 long)."""
    return f"DS{window.spacing_hours}_ASOF_LAST_AVAILABLE"


def data_contract_sha256(design: Mapping[str, Any]) -> str:
    """Identity of the data-relevant design fields (seal/training fields excluded)."""
    return provenance.canonical_sha256({k: design[k] for k in DATA_CONTRACT_KEYS})


#: PLAN.md section 5 long-family data fields (mirrors the model lane's LONG_FROM_PLAN).
LONG_DATA_FIELDS = {"experiment_id": "I6-DMO-LONG-DS12-STATIC-R0",
                    "horizons_hours": [18, 24, 36, 48, 60, 72],
                    "input.history_span_hours": 144, "input.spacing_hours": 12}


def long_data_view(short_design: Mapping[str, Any]) -> dict:
    """Unsealed long data view: the short design with the PLAN section 5 long fields.

    Its ``data_contract_sha256`` equals that of the model lane's sealed long design
    (seal fields never enter the data identity), so a bundle prepared before the
    short closure is reusable once the long design is sealed.
    """
    if short_design.get("experiment_id") != "I6-DMO-SHORT-DS2-STATIC-R0":
        raise DataContractError("SHORT_DESIGN_REQUIRED", str(short_design.get("experiment_id")))
    doc = json.loads(json.dumps(short_design))
    for path, value in LONG_DATA_FIELDS.items():
        node, keys = doc, path.split(".")
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    return doc


def test_start(design: Mapping[str, Any]) -> datetime:
    return _instant(design["splits"]["test"]["start"])


# --------------------------------------------------------------------------- input

@dataclass
class InputBundle:
    """Validated retained input (no TEST bar) with its file digests."""

    ts: np.ndarray
    x: np.ndarray
    price: np.ndarray
    features: Tuple[str, ...]
    calendar_config: dict
    files: Dict[str, str]
    clock: dict
    design: dict
    window: WindowSpec
    horizons: Tuple[int, ...]
    input_dir: Path = field(repr=False)
    _allow_noncontract_splits: bool = False
    sampling_rule: str = SAMPLING_RULE


def load_input(input_dir: str | Path, design: Mapping[str, Any], *,
               _allow_noncontract_splits: bool = False) -> InputBundle:
    """Validate design + retained input; refuse TEST bars, manifest drift and short coverage."""
    design = validate_design(design, _allow_noncontract_splits=_allow_noncontract_splits)
    window = WindowSpec.from_design(design)
    hs = horizons(design)
    root = Path(input_dir)
    for name in ("INPUT.npz", "CLOCK.json", "CALENDAR.json"):
        if not (root / name).is_file():
            raise DataContractError("INPUT_FILE_MISSING", name)
    files = {name: provenance.file_sha256(root / name)
             for name in INPUT_FILES if (root / name).is_file()}
    clock = json.loads((root / "CLOCK.json").read_text())
    if clock.get("input_sha256") != files["INPUT.npz"]:
        raise DataContractError("INPUT_DIGEST_MISMATCH", "CLOCK.json input_sha256 != INPUT.npz")
    if clock.get("timestamp_semantics") != "UTC_BAR_END" or clock.get("status") != "RESOLVED":
        raise DataContractError("CLOCK_INVALID", "UTC_BAR_END resolved clock required")
    features = tuple(clock.get("features") or ())
    provenance.check_manifest(features)
    excluded = set(design["feature_policy"]["exclude"])
    if excluded & set(features):
        raise provenance.ProvenanceError("EXCLUDED_FEATURE_ALIAS", str(excluded & set(features)))
    t_test = _epoch(test_start(design))
    with np.load(root / "INPUT.npz", allow_pickle=False) as z:
        if set(z.files) != {"ts", "x", "price"}:
            raise DataContractError("INPUT_INVALID", f"keys {z.files}")
        ts = np.asarray(z["ts"])
        if ts.dtype != np.int64 or ts.ndim != 1 or not len(ts) or np.any(np.diff(ts) <= 0):
            raise DataContractError("INPUT_INVALID", "ts must be strictly increasing int64 seconds")
        if int(ts.max()) >= t_test:
            raise DataContractError("TEST_ROWS_PRESENT",
                                    f"input holds bars at or after the TEST start {iso(t_test)}")
        x = np.asarray(z["x"], dtype=np.float64)
        price = np.asarray(z["price"], dtype=np.float64)
    if x.shape != (len(ts), len(features)) or price.shape != (len(ts),):
        raise DataContractError("INPUT_INVALID", "x/price shapes disagree with ts/features")
    if not np.isfinite(price).all() or (price <= 0).any():
        raise DataContractError("INPUT_INVALID", "price must be finite and positive")
    calendar = json.loads((root / "CALENDAR.json").read_text())
    business = design["business"]
    if (calendar.get("entry_delay_after_open_hours") != business["entry_delay_after_open_hours"]
            or calendar.get("force_flat_before_close_hours")
            != business["force_flat_before_close_hours"]):
        raise DataContractError("CALENDAR_OFFSET_MISMATCH", "calendar offsets differ from design")
    bundle = InputBundle(ts=ts, x=x, price=price, features=features, calendar_config=calendar,
                         files=files, clock=clock, design=design, window=window, horizons=hs,
                         input_dir=root, _allow_noncontract_splits=_allow_noncontract_splits,
                         sampling_rule=sampling_rule_name(window))
    _check_coverage(bundle)
    return bundle


def _weeks(bundle: InputBundle):
    try:
        return c1.load_calendar(bundle.calendar_config)
    except c1.EligibilityError as exc:
        raise DataContractError("CALENDAR_INVALID", str(exc)) from None


def _check_coverage(bundle: InputBundle) -> None:
    """Span coverage: history before the first TRAIN origin and bars in the last VALIDATION week."""
    splits = split_bounds(bundle.design, _allow_noncontract_splits=bundle._allow_noncontract_splits)
    start, end = _epoch(splits[0].start), _epoch(splits[-1].end)
    _, weeks = _weeks(bundle)
    if _epoch(weeks[0].market_open) > start or _epoch(weeks[-1].market_close) < end:
        raise DataContractError("CALENDAR_INCOMPLETE", "calendar must cover TRAIN..VALIDATION")
    if int(bundle.ts[0]) > start - bundle.window.span_hours * HOUR:
        raise DataContractError("COVERAGE_INCOMPLETE",
                                f"first bar {iso(bundle.ts[0])} leaves no input history before "
                                f"{iso(start)}; the declared TRAIN span is not shortened")
    last_open = max(_epoch(w.market_open) for w in weeks if _epoch(w.market_open) < end)
    if int(bundle.ts[-1]) < last_open:
        raise DataContractError("COVERAGE_INCOMPLETE",
                                f"last bar {iso(bundle.ts[-1])} precedes the final VALIDATION week")


def coverage_report(bundle: InputBundle) -> dict:
    """Expected vs observed hourly bars per calendar week inside TRAIN..VALIDATION."""
    splits = split_bounds(bundle.design, _allow_noncontract_splits=bundle._allow_noncontract_splits)
    start, end = _epoch(splits[0].start), _epoch(splits[-1].end)
    _, weeks = _weeks(bundle)
    have = set(int(v) for v in bundle.ts)
    deficits, per_year = [], {}
    for w in weeks:
        o, c = _epoch(w.market_open), _epoch(w.market_close)
        expected = [t for t in range(o + HOUR, c + 1, HOUR) if start <= t < end]
        if not expected:
            continue
        observed = sum(1 for t in expected if t in have)
        year = datetime.fromtimestamp(expected[0], UTC).year
        slot = per_year.setdefault(str(year), {"expected_bars": 0, "observed_bars": 0})
        slot["expected_bars"] += len(expected)
        slot["observed_bars"] += observed
        if observed != len(expected):
            deficits.append({"week_id": w.week_id, "expected_bars": len(expected),
                             "observed_bars": observed})
    return {"status": "SPAN_COMPLETE_BARS_MISSING" if deficits else "COMPLETE",
            "basis": "whole-hour bar ends in (open, close] of each supplied calendar week",
            "per_year": per_year, "weeks_with_missing_bars": deficits,
            "missing_bars_are": "counted origin/label exclusions; never filled or bridged"}


# --------------------------------------------------------------------------- origin table

@dataclass(frozen=True)
class OriginRecord:
    origin: int
    split: str
    week_id: Optional[str]
    input_ok: bool
    input_reason: Optional[str]
    label_ok: Tuple[bool, ...]
    entry_ok: Tuple[bool, ...]
    label_reasons: Tuple[Tuple[str, ...], ...]


@dataclass
class OriginTable:
    """Every observed hourly bar of FIT/INNER/VALIDATION with masks and reasons."""

    horizons: Tuple[int, ...]
    origin_ts: np.ndarray            # (n,) int64
    split: np.ndarray                # (n,) str
    week_id: np.ndarray              # (n,) str ('' outside the calendar)
    input_ok: np.ndarray             # (n,) bool
    input_reason: List[Optional[str]]
    label_ok: np.ndarray             # (n, H) bool, C1 MANAGEMENT M
    entry_ok: np.ndarray             # (n, H) bool, C1 ENTRY M
    label_reasons: List[Tuple[Tuple[str, ...], ...]]
    slot_index: np.ndarray           # (n, obs, F) int64 as-of source row per feature, -1 none
    origin_index: np.ndarray         # (n,) int64 row of the observed origin bar
    slot_instants: np.ndarray        # (n, obs) int64 slot instants s
    target_index: np.ndarray         # (n, H) int64, -1 absent
    counts: Dict[str, Any]
    split_weeks: Dict[str, List[str]]
    calendar: Dict[str, str]
    index: Dict[int, int] = field(default_factory=dict)

    def record(self, t: int) -> OriginRecord:
        i = self.index[int(t)]
        return OriginRecord(
            origin=int(self.origin_ts[i]), split=str(self.split[i]),
            week_id=str(self.week_id[i]) or None, input_ok=bool(self.input_ok[i]),
            input_reason=self.input_reason[i],
            label_ok=tuple(bool(v) for v in self.label_ok[i]),
            entry_ok=tuple(bool(v) for v in self.entry_ok[i]),
            label_reasons=self.label_reasons[i])

    def origins_in_week(self, week_id: str) -> List[int]:
        return [int(t) for t, w in zip(self.origin_ts, self.week_id) if w == week_id]

    def weeks_in_split(self, split: str) -> List[str]:
        return list(self.split_weeks[split])

    def materialized(self, split: str) -> np.ndarray:
        return (self.split == split) & self.input_ok & self.label_ok.any(axis=1)


def asof_slot_index(ts: np.ndarray, x: np.ndarray, origins: np.ndarray,
                    offsets_seconds: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Per (origin, slot, feature) row of the last finite observation with ts <= s.

    Returns ``(index, instants)``: ``index`` (n, obs, F) int64 with -1 when no
    such observation exists, and the slot instants (n, obs). Only rows with
    ``ts <= s`` are ever selected, so no slot can read the future.
    """
    rows = np.arange(len(ts), dtype=np.int64)
    finite = np.isfinite(x)
    last_valid = np.maximum.accumulate(np.where(finite, rows[:, None], -1), axis=0)
    instants = np.asarray(origins, dtype=np.int64)[:, None] + np.asarray(offsets_seconds,
                                                                           dtype=np.int64)
    k = np.searchsorted(ts, instants, side="right") - 1
    index = np.where((k >= 0)[:, :, None], last_valid[np.maximum(k, 0)], -1)
    return index.astype(np.int64), instants


def build_origin_table(bundle: InputBundle, design: Mapping[str, Any], *,
                       _allow_noncontract_splits: bool = False) -> OriginTable:
    """Evaluate every candidate origin with as-of slots and C1 per-head masks."""
    allow = _allow_noncontract_splits or bundle._allow_noncontract_splits
    design = validate_design(design, _allow_noncontract_splits=allow)
    splits = split_bounds(design, _allow_noncontract_splits=allow)
    ref, weeks = _weeks(bundle)
    hs = horizons(design)
    window = WindowSpec.from_design(design)
    ts = bundle.ts
    pos = {int(v): i for i, v in enumerate(ts)}
    observed = c1.ObservedTargets({datetime.fromtimestamp(int(v), UTC):
                                   datetime.fromtimestamp(int(v), UTC) for v in ts})
    opens = [w.market_open for w in weeks]
    offsets = [o * HOUR for o in window.offsets_hours()]
    lo, hi = _epoch(splits[0].start), _epoch(splits[-1].end)
    cand = np.flatnonzero((ts >= lo) & (ts < hi))
    n, nh = len(cand), len(hs)
    out_split = np.empty(n, dtype=object)
    out_week = np.empty(n, dtype=object)
    input_ok = np.zeros(n, dtype=bool)
    label_ok = np.zeros((n, nh), dtype=bool)
    entry_ok = np.zeros((n, nh), dtype=bool)
    slot_index, slot_instants = asof_slot_index(ts, bundle.x, ts[cand], offsets)
    no_obs = (slot_index < 0).any(axis=(1, 2))
    target_index = np.full((n, nh), -1, dtype=np.int64)
    input_reason: List[Optional[str]] = []
    reasons_out: List[Tuple[Tuple[str, ...], ...]] = []
    counts = {s: {"origins": {"candidates": 0, "input_ok": 0, **{r: 0 for r in ORIGIN_REASONS}},
                  "heads": {str(h): {"label_valid": 0, "entry_valid": 0, "reasons": Counter()}
                            for h in hs},
                  "rows_materialized": 0} for s in SPLITS}
    for k, i in enumerate(cand):
        t = int(ts[i])
        t_dt = datetime.fromtimestamp(t, UTC)
        split = next(s for s in splits if s.start <= t_dt < s.end)
        out_split[k] = split.split_id
        c = counts[split.split_id]
        c["origins"]["candidates"] += 1
        reason = "INPUT_NO_OBSERVATION_AVAILABLE" if no_obs[k] else None
        input_ok[k] = reason is None
        input_reason.append(reason)
        if reason is None:
            c["origins"]["input_ok"] += 1
        else:
            c["origins"][reason] += 1
        for j, h in enumerate(hs):
            target_index[k, j] = pos.get(t + h * HOUR, -1)
        w = bisect.bisect_right(opens, t_dt) - 1
        if w < 0 or not t_dt < weeks[w].market_close:
            out_week[k] = ""
            c["origins"]["ORIGIN_OUTSIDE_CALENDAR_WEEK"] += 1
            reasons_out.append(tuple(("ORIGIN_OUTSIDE_CALENDAR_WEEK",) for _ in hs))
            for h in hs:
                c["heads"][str(h)]["reasons"]["ORIGIN_OUTSIDE_CALENDAR_WEEK"] += 1
            continue
        week = weeks[w]
        out_week[k] = week.week_id
        mgmt = c1.evaluate_heads(t_dt, hs, c1.Role.MANAGEMENT, week, input_available=bool(input_ok[k]),
                                 observed_targets=observed, split=split)
        entry = c1.evaluate_heads(t_dt, hs, c1.Role.ENTRY, week, input_available=bool(input_ok[k]),
                                  observed_targets=observed, split=split)
        reasons_out.append(tuple(tuple(r.value for r in d.label_reasons) for d in mgmt))
        for j, h in enumerate(hs):
            label_ok[k, j] = bool(mgmt[j].label_eligible)
            entry_ok[k, j] = bool(entry[j].label_eligible)
            head = c["heads"][str(h)]
            if label_ok[k, j]:
                head["label_valid"] += 1
            else:
                head["reasons"][mgmt[j].label_reasons[0].value] += 1
            head["entry_valid"] += int(entry_ok[k, j])
        if input_ok[k] and not label_ok[k].any():
            c["origins"]["NO_VALID_HEAD"] += 1
    split_arr = out_split.astype(str)
    closure = max(((_epoch(b.market_open) - _epoch(a.market_close)) // HOUR
                   for a, b in zip(weeks, weeks[1:])), default=0)
    stale = np.where(slot_index >= 0, slot_instants[:, :, None] - ts[np.maximum(slot_index, 0)], 0)
    exact_pos = np.searchsorted(ts, slot_instants)
    exact_exists = (exact_pos < len(ts)) & (ts[np.minimum(exact_pos, len(ts) - 1)] == slot_instants)
    nonfinite_exact = exact_exists[:, :, None] & (slot_index != exact_pos[:, :, None]) \
        & (slot_index >= 0)
    for s in SPLITS:
        sel = (split_arr == s) & input_ok & label_ok.any(axis=1)
        counts[s]["rows_materialized"] = int(sel.sum())
        st = stale[sel]
        counts[s]["asof"] = {
            "rule": bundle.sampling_rule, "scope": "materialized rows",
            "closure_threshold_hours": int(closure),
            "origins_all_slots_exact": int((st.reshape(len(st), -1) == 0).all(axis=1).sum()),
            "origins_with_asof_slot": int((st.reshape(len(st), -1) > 0).any(axis=1).sum()),
            "origins_asof_due_to_nonfinite_bar": int(nonfinite_exact[sel].any(axis=(1, 2)).sum()),
            "origins_stale_beyond_closure": int((st.reshape(len(st), -1)
                                                 > closure * HOUR).any(axis=1).sum()),
            "slots_asof_by_feature": {f: int((st[:, :, i] > 0).sum())
                                      for i, f in enumerate(bundle.features)},
            "nonfinite_exact_slots_by_feature": {f: int(nonfinite_exact[sel][:, :, i].sum())
                                                 for i, f in enumerate(bundle.features)},
            "max_staleness_hours": (float(st.max()) / HOUR) if st.size else None}
        for h in hs:
            counts[s]["heads"][str(h)]["reasons"] = dict(sorted(counts[s]["heads"][str(h)]["reasons"].items()))
    split_weeks = {}
    for s in splits:
        split_weeks[s.split_id] = [w.week_id for w in weeks
                                   if w.operational_start < s.end and w.force_flat > s.start]
        grid_missing = 0
        have = set(int(v) for v in ts)
        for w in weeks:
            a, b = max(w.operational_start, s.start), min(w.force_flat, s.end)
            t = _epoch(a) + (-_epoch(a)) % HOUR
            while t < _epoch(b):
                grid_missing += t not in have
                t += HOUR
        counts[s.split_id]["operational_grid_hours_without_observed_bar"] = grid_missing
    table = OriginTable(horizons=hs, origin_ts=ts[cand].astype(np.int64), split=split_arr,
                        week_id=out_week.astype(str), input_ok=input_ok, input_reason=input_reason,
                        label_ok=label_ok, entry_ok=entry_ok, label_reasons=reasons_out,
                        slot_index=slot_index, origin_index=cand.astype(np.int64),
                        slot_instants=slot_instants, target_index=target_index, counts=counts,
                        split_weeks=split_weeks,
                        calendar={"calendar_id": ref.calendar_id, "version": ref.version})
    table.index = {int(t): k for k, t in enumerate(table.origin_ts)}
    return table


# --------------------------------------------------------------------------- arrays

@dataclass
class SplitArrays:
    """Materialized rows of one split (input ok and >= 1 valid head)."""

    split: str
    horizons: Tuple[int, ...]
    features: Tuple[str, ...]
    X: np.ndarray                 # (n, obs, F) float64 RAW, as-of sampled
    staleness_s: np.ndarray       # (n, obs, F) int64 seconds s - observation_time (aux only)
    y_price: np.ndarray           # (n, H) float64 original price p(t+h); NaN where mask False
    mask: np.ndarray              # (n, H) bool  MANAGEMENT support S_h
    entry_mask: np.ndarray        # (n, H) bool  ENTRY support
    price_origin: np.ndarray      # (n,) float64 p(t)
    origin_ts: np.ndarray         # (n,) int64
    row_ids: np.ndarray           # (n,) str
    week_ids: np.ndarray          # (n,) str
    expected_weeks: Tuple[str, ...]
    calendar_id: str

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        """Model-lane format (INTERFACE_MODEL consumed_data_bundle_contract): exactly five arrays."""
        return {"row_id": self.origin_ts.astype(np.int64), "x": self.X,
                "origin_price": self.price_origin, "target_price": self.y_price,
                "mask": self.mask}

    def to_aux_dict(self) -> Dict[str, np.ndarray]:
        """Business/strategy side arrays aligned row by row with ``to_npz_dict``."""
        return {"row_id": self.origin_ts.astype(np.int64), "row_ids": self.row_ids,
                "entry_mask": self.entry_mask, "week_ids": self.week_ids,
                "slot_staleness_s": self.staleness_s}


def materialize(bundle: InputBundle, table: OriginTable, split: str) -> SplitArrays:
    if split not in SPLITS:
        raise DataContractError("SPLIT_UNKNOWN", f"{split!r}; TEST is never materialized")
    sel = np.flatnonzero(table.materialized(split))
    slots = table.slot_index[sel]
    if (slots < 0).any():
        raise DataContractError("INTERNAL_SLOT_MISSING", split)
    origin_idx = table.origin_index[sel]
    if not np.array_equal(bundle.ts[origin_idx], table.origin_ts[sel]):
        raise DataContractError("INTERNAL_ORIGIN_MISMATCH", split)
    source_ts = bundle.ts[slots]
    staleness = table.slot_instants[sel][:, :, None] - source_ts
    if (staleness < 0).any():
        raise DataContractError("INTERNAL_FUTURE_READ", split)
    nf = bundle.x.shape[1]
    X = bundle.x[slots, np.arange(nf)]
    mask = table.label_ok[sel].copy()
    targets = table.target_index[sel]
    y = np.full(mask.shape, np.nan)
    y[mask] = bundle.price[targets[mask]]
    return SplitArrays(
        split=split, horizons=table.horizons, features=bundle.features,
        X=X, staleness_s=staleness.astype(np.int64), y_price=y, mask=mask, entry_mask=table.entry_ok[sel] & mask,
        price_origin=bundle.price[origin_idx], origin_ts=table.origin_ts[sel].copy(),
        row_ids=np.asarray([row_id(t) for t in table.origin_ts[sel]], dtype=str),
        week_ids=table.week_id[sel].copy(), expected_weeks=tuple(table.weeks_in_split(split)),
        calendar_id=table.calendar["calendar_id"])


def fit_statistics(bundle: InputBundle, table: OriginTable) -> dict:
    """Feature mean/std over FIT window bars and fit-only naive MAE per head.

    Computed from FIT rows only. A zero std or a zero/absent naive scale is a
    typed status (``ZERO_SCALE`` / ``UNSUPPORTED_HEAD``), never an epsilon.
    """
    sel = np.flatnonzero(table.materialized("FIT"))
    if not len(sel):
        return {"status": "NO_FIT_ROWS"}
    per_feature = [np.unique(table.slot_index[sel][:, :, f].reshape(-1))
                   for f in range(bundle.x.shape[1])]
    bars = np.unique(np.concatenate(per_feature))
    mean = np.asarray([bundle.x[idx, f].mean() for f, idx in enumerate(per_feature)])
    std = np.asarray([bundle.x[idx, f].std() for f, idx in enumerate(per_feature)])
    naive_mae, unsupported = {}, []
    for j, h in enumerate(table.horizons):
        rows = sel[table.label_ok[sel, j]]
        if not len(rows):
            naive_mae[str(h)] = None
            unsupported.append(h)
            continue
        p0 = bundle.price[table.origin_index[rows]]
        p1 = bundle.price[table.target_index[rows, j]]
        value = math.fsum(np.abs(p1 - p0).tolist()) / len(rows)
        naive_mae[str(h)] = value
        if value == 0.0:
            unsupported.append(h)
    zero = [f for f, s in zip(bundle.features, std) if s == 0.0]
    status = "UNSUPPORTED_HEAD" if unsupported else ("ZERO_SCALE" if zero else "OK")
    return {"status": status, "basis": "FIT rows only; unique as-of source observations per feature; std ddof=0",
            "fit_rows": int(len(sel)), "fit_window_bars": int(len(bars)),
            "feature_mean": {f: float(v) for f, v in zip(bundle.features, mean)},
            "feature_std": {f: float(v) for f, v in zip(bundle.features, std)},
            "zero_scale_features": zero, "naive_mae_fit": naive_mae,
            "unsupported_heads": unsupported}


# --------------------------------------------------------------------------- frozen bundle

def history_provenance(bundle: InputBundle, table: OriginTable) -> dict:
    """Earliest retained bar any materialized slot consumes, and the clock scope covering it."""
    sel = np.zeros(len(table.origin_ts), dtype=bool)
    for s in SPLITS:
        sel |= table.materialized(s)
    idx = table.slot_index[sel]
    earliest = int(bundle.ts[idx[idx >= 0].min()]) if idx.size else None
    train_start = _epoch(split_bounds(bundle.design,
                                      _allow_noncontract_splits=bundle._allow_noncontract_splits)[0].start)
    out = {"earliest_consumed_bar": iso(earliest) if earliest is not None else None,
           "train_start": iso(train_start),
           "consumed_before_train_start": earliest is not None and earliest < train_start,
           "clock_policy": bundle.clock.get("clock_policy"),
           "declared_assumptions": [], "base_scope_warmup": None,
           "consumed_before_base_scope_start": None}
    scope_path = bundle.input_dir / "CLOCK_SCOPE.json"
    if scope_path.is_file():
        scope = json.loads(scope_path.read_text())
        out["clock_scope_sha256"] = provenance.file_sha256(scope_path)
        declared = scope.get("declared_assumption") or {}
        if declared:
            span = declared.get("span") or {}
            end = _epoch(_instant(span["end"])) if span.get("end") else None
            out["declared_assumptions"].append({
                "assumption_id": declared.get("assumption_id"),
                "assumption_sha256": scope.get("assumption_sha256"),
                "status": declared.get("status"), "span": span,
                "overlaps_consumed_history": (earliest is not None and end is not None
                                              and earliest < end)})
        base = scope.get("base_scope") or {}
        out["base_scope_warmup"] = base.get("warmup_2019_december")
        if base.get("data_start") and earliest is not None:
            out["consumed_before_base_scope_start"] = earliest < _epoch(_instant(base["data_start"]))
    return out


def _code_digests() -> dict:
    here = Path(__file__).resolve().parent
    names = ("data.py", "naive.py", "provenance.py", "producer_recovery.v1.json",
             "feature_availability.successor.v2.json")
    return {n: provenance.file_sha256(here / n) for n in names if (here / n).is_file()}


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Content identity: excludes zip-container file digests (npz members carry write times)."""
    body = {k: v for k, v in manifest.items() if k not in ("bundle_sha256", "files", "splits")}
    body["splits"] = {s: {k: v for k, v in info.items() if k != "sha256"}
                      for s, info in manifest["splits"].items()}
    return provenance.canonical_sha256(body)


def build_frozen_bundle(input_dir: str | Path, design: Mapping[str, Any], out_dir: str | Path,
                        *, code_commit: Optional[str] = None) -> dict:
    """Build FIT/INNER/VALIDATION arrays and an immutable BUNDLE.json with digests."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise DataContractError("BUNDLE_EXISTS", "a frozen bundle is never overwritten")
    bundle = load_input(input_dir, design)
    table = build_origin_table(bundle, design)
    gate = provenance.availability_gate(list(bundle.features))
    out.mkdir(parents=True, exist_ok=True)
    splits_json, files = {}, {}
    for s, bounds in zip(SPLITS, split_bounds(bundle.design)):
        arrays = materialize(bundle, table, s)
        name, aux = f"{s.lower()}.npz", f"{s.lower()}_aux.npz"
        np.savez(out / name, **arrays.to_npz_dict())
        np.savez(out / aux, **arrays.to_aux_dict())
        files[name] = provenance.file_sha256(out / name)
        files[aux] = provenance.file_sha256(out / aux)
        splits_json[s.lower()] = {
            **bounds.to_json(), "path": name, "aux_path": aux, "n_rows": int(len(arrays.row_ids)),
            "row_ids_sha256": provenance.canonical_sha256(arrays.row_ids.tolist()),
            "arrays_sha256": provenance.array_sha256(arrays.to_npz_dict()),
            "aux_sha256": provenance.array_sha256(arrays.to_aux_dict()),
            "head_valid": {str(h): int(arrays.mask[:, j].sum()) for j, h in enumerate(arrays.horizons)},
            "head_entry_valid": {str(h): int(arrays.entry_mask[:, j].sum())
                                 for j, h in enumerate(arrays.horizons)},
            "counts": table.counts[s], "expected_weeks": len(arrays.expected_weeks),
            "first_origin": iso(arrays.origin_ts.min()) if len(arrays.origin_ts) else None,
            "last_origin": iso(arrays.origin_ts.max()) if len(arrays.origin_ts) else None,
        }
    calendar_id = table.calendar["calendar_id"]
    manifest = {
        "schema": SCHEMA,
        "experiment_id": bundle.design["experiment_id"],
        "design_sha256": provenance.canonical_sha256(bundle.design),
        "data_contract_sha256": data_contract_sha256(bundle.design),
        "history_provenance": history_provenance(bundle, table),
        "asset": ASSET,
        "window": bundle.window.to_json(),
        "horizons_hours": list(bundle.horizons),
        "offsets_hours": list(bundle.window.offsets_hours()),
        "input_sampling": {"rule": bundle.sampling_rule, "description": SAMPLING_DESCRIPTION,
                           "staleness_array": (f"<split>_aux.npz slot_staleness_s (n,{bundle.window.observations},"
                                               f"{len(bundle.features)}) seconds; not a model input"),
                           "ruling": "Satoshi technical ruling 2026-10-14 under PLAN section 4"},
        "features": list(bundle.features),
        "row_id_format": "npz row_id = origin UTC epoch seconds (int64); aux row_ids = 'EURUSD|<UTC ISO origin>'",
        "price_clock": "UTC_BAR_END; original price USD per EUR",
        "inputs": {name: {"name": name, "sha256": digest} for name, digest in bundle.files.items()},
        "calendar": {**table.calendar, "sha256": bundle.files["CALENDAR.json"],
                     "label": ("ASSUMED_NOT_BROKER" if "ASSUMED" in calendar_id
                               else "SUPPLIED_UNVERIFIED")},
        "feature_manifest": list(bundle.features),
        "feature_manifest_sha256": gate["manifest_sha256"],
        "availability_contract_sha256": gate["contract_sha256"],
        "availability_gate": gate,
        "producer_status": provenance.producer_status(),
        "confirmatory_test_blocked": gate["confirmatory_test_blocked"],
        "splits": splits_json,
        "coverage": coverage_report(bundle),
        "fit_statistics": fit_statistics(bundle, table),
        "test_materialized": False,
        "development_selection_used_2024": True,
        "code": {"files_sha256": _code_digests(), "commit": code_commit},
    }
    manifest["bundle_sha256"] = _manifest_digest(manifest)
    for info in splits_json.values():
        info["sha256"] = files[info["path"]]
    manifest["files"] = files
    (out / "BUNDLE.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def verify_bundle(out_dir: str | Path) -> dict:
    """Re-hash every array file, row-id list and the manifest; typed refusal on any change."""
    out = Path(out_dir)
    manifest = json.loads((out / "BUNDLE.json").read_text())
    if _manifest_digest(manifest) != manifest.get("bundle_sha256"):
        raise DataContractError("BUNDLE_DIGEST_MISMATCH", "BUNDLE.json body changed")
    if set(manifest["splits"]) != {s.lower() for s in SPLITS}:
        raise DataContractError("BUNDLE_DIGEST_MISMATCH", "split set changed")
    for s, info in manifest["splits"].items():
        for key, digest_key in (("path", "arrays_sha256"), ("aux_path", "aux_sha256")):
            path = out / info[key]
            if provenance.file_sha256(path) != manifest["files"].get(info[key]):
                raise DataContractError("BUNDLE_DIGEST_MISMATCH", f"{info[key]} file digest")
            with np.load(path, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            if provenance.array_sha256(arrays) != info[digest_key]:
                raise DataContractError("BUNDLE_DIGEST_MISMATCH", f"{s} {key} content")
            if key == "aux_path" and (provenance.canonical_sha256(arrays["row_ids"].tolist())
                                      != info["row_ids_sha256"]):
                raise DataContractError("BUNDLE_DIGEST_MISMATCH", f"{s} row ids")
        if info["sha256"] != manifest["files"][info["path"]]:
            raise DataContractError("BUNDLE_DIGEST_MISMATCH", f"{s} split sha256")
    return {"status": "VERIFIED", "bundle_sha256": manifest["bundle_sha256"]}


def load_split(out_dir: str | Path, split: str) -> Dict[str, np.ndarray]:
    """Read one verified split (model arrays + aux arrays) of a frozen bundle."""
    verify_bundle(out_dir)
    root = Path(out_dir)
    info = json.loads((root / "BUNDLE.json").read_text())["splits"][split.lower()]
    merged: Dict[str, np.ndarray] = {}
    for key in ("path", "aux_path"):
        with np.load(root / info[key], allow_pickle=False) as z:
            merged.update({k: z[k] for k in z.files})
    return merged
