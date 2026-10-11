"""Strictly paired model/naive error metrics for I6-N (candidate component C2).

Candidate implementation of section 5 of the I6-N bounded engineering order.
It is not deployed, it is not a scientific measurement, and it writes nothing
to any warehouse. It receives an ALREADY DECLARED eligible population and never
recomputes a market calendar: eligibility belongs to component C1.

Input document (``INPUT_SCHEMA_VERSION``), all timestamps zone-aware ISO-8601::

    contract                 unit, scale, roles, horizons_hours, protocol_id,
                             calendar_id, config_id
    forecasts                records: asset, origin, horizon_hours,
                             target_timestamp, role, price_origin (price(t)),
                             price_target (price(t+h), or null when the label
                             is not observable), prediction, checkpoint, unit,
                             scale
    forecast_failures        typed failures per forecast identity
                             (asset, origin, horizon_hours, reason)
    opportunity_population   SEALED S support: population_id, sealed_at, keys
                             (asset, origin, horizon_hours, role), upstream
                             exclusion counts, sha256 over the rest
    usage_receipts           query receipts recorded before the outcome; the
                             ONLY source of the real-use population U

Rules implemented (see the tests for the executable form):

* Everything is validated before any error is computed: zone-aware instants
  inside the representable calendar, ``target_timestamp == origin + h`` as UTC
  instants, non-empty identities, finite numbers (booleans are not numbers,
  integers too large for a float are refused), declared unit/scale/role/horizon,
  duplicates and conflicting values for one forecast, and ONE observed price
  per (asset, instant): ``price_origin`` at the origin and ``price_target`` at
  the target instant of every record form a single price series, and two rows
  that disagree about the price (or about whether it was observed) at the same
  instant are refused with ``PRICE_CONFLICT``.
* For each population kind (S, U), horizon and role, plus a role-deduplicated
  ``ALL_ROLES`` group: n, sums of absolute and squared errors, model and naive
  MAE/MSE, both over exactly the same rows. The naive error of a row is read
  from the validated price series, ``price(asset, origin) - price(asset,
  target)``, on the same declared scale as target and prediction.
  ``skill = (model - naive) / naive``; negative improves. A zero naive, or a
  skill that is not a finite float, leaves the skill ``None`` with a typed
  reason; JSON never carries NaN or Infinity.
* Pairing is structural: ONE sorted row list feeds both the model and the naive
  sums, so there is no in-process model/naive pairing check (it could never
  fail). ``row_keys_sha256`` lets an external reader check that two published
  records cover exactly the same rows; ``model_rows_sha256`` and
  ``naive_rows_sha256`` digest the inputs of each side on those rows.
* An empty population has no score but keeps exclusion and failure counts.
* Two kinds of exclusion are kept apart. ``exclusions`` holds ONLY exclusions
  made here (``LABEL_NOT_OBSERVABLE``), each a declared member of C2's
  population, so ``n_excluded == sum(exclusions)``. The sealed upstream counts
  (made before C2, e.g. by eligibility) are NOT members of C2's declared
  population: a per-role group carries them in ``upstream_exclusions``; they
  carry no row identities and cannot be de-duplicated across roles, so an
  ``ALL_ROLES`` group reports them unpooled in ``upstream_exclusions_by_role``
  and its OLAP rows never repeat them (``exclusions_scope`` says which).
* A declared member without a verifiable forecast is a failure: the group is
  ``INCOMPLETE`` with explicit coverage. It is never silently dropped.
  Every group satisfies ``n_declared == n + n_excluded + sum(failures)``;
  ``coverage = n / (n_declared - n_excluded)``; failures stay in the
  denominator, exclusions made here do not. State, coverage and the
  unavailable-score reason are DERIVED from those counts by one function, used
  both when scoring and when a sealed group is read back.
* Every group carries its identity (config, protocol, calendar, contract
  digest) and its own ``sha256``.
* ``compare_configs`` requires the identical previously sealed S support and
  refuses (``NOT_COMPARABLE_INCOMPLETE``) instead of building an intersection.
* ``verify_result`` also checks the result against its contract: the
  ``contract`` field is required and re-validated, ``contract_sha256`` must be
  the digest of that body, and the result must hold EXACTLY one group per
  (population kind in {S, U}) x declared horizon x (declared roles +
  ``ALL_ROLES``), see ``expected_group_keys``; a missing, duplicate or
  undeclared group is a typed refusal, and so are group identities or embedded
  OLAP rows that disagree. ``compare_configs`` and ``olap_records`` verify first.
* ``aggregate_groups``, ``verify_result`` and ``compare_configs`` never trust a
  sealed field: they re-verify every group's types (non-bool non-negative
  integer counts, typed reason->count maps, scalar row keys of the group's own
  h/role), the count identity above, finite non-negative error sums that are
  zero for zero rows and mutually possible (sqrt(sum e^2) <= sum |e| <=
  sqrt(n * sum e^2), with a rounding/underflow tolerance), and every derived
  field (state, coverage, means, skills and their reasons). Every refusal is a
  typed ``PairedMetricsError``, never a ``TypeError``. Pooling uses sums and
  counts, never means of weeks, only across groups with the same identity.

Limit: digests are unkeyed SHA-256. They detect corruption and unilateral
edits, not an adversary who recomputes a digest after editing. Arithmetic
contradictions are refused without any key; a forgery that is plausible AND
self-consistent (false but possible counts and sums) is not detectable without
the rows themselves and external custody.

The canonical digest is byte-identical to ``tools.i6m_queue.digest``; it is
re-implemented here only to avoid importing a module that loads ``sqlite3``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

INPUT_SCHEMA_VERSION = "i6n.paired_metrics.input.v1"
RESULT_SCHEMA_VERSION = "i6n.paired_metrics.result.v2"
COMPARISON_SCHEMA_VERSION = "i6n.paired_metrics.comparison.v1"
COMPONENT = "I6N_C2_PAIRED_METRICS"

OPPORTUNITY = "S"
USAGE = "U"
ALL_ROLES = "ALL_ROLES"

MISSING_FORECAST = "MISSING_FORECAST"
LABEL_NOT_OBSERVABLE = "LABEL_NOT_OBSERVABLE"
EMPTY_POPULATION = "EMPTY_POPULATION"
NO_SCORED_ROWS = "NO_SCORED_ROWS"
EXCLUSIONS_C2_ONLY_UPSTREAM_SEPARATE = "C2_ONLY_UPSTREAM_SEPARATE"
EXCLUSIONS_C2_ONLY = "C2_ONLY_UPSTREAM_REPORTED_PER_ROLE"
EMPTY_STATE = {OPPORTUNITY: "NO_ELIGIBLE_ROWS", USAGE: "NO_RECEIPTS"}

CLAIMS = {
    "candidate_implementation": True,
    "deployed": False,
    "scientific_measurement": False,
    "warehouse_written": False,
    "olap_schema_changed": False,
    "end_to_end_or_trading_acceptance": False,
}

# Design basis read once (uncommitted design object over base 269c2d13), verified
# against section 2 of the I6-N audit request. Carried in every result so a
# reader can tell which contract version this candidate implements.
DESIGN_BASIS = {
    "base_commit": "269c2d132b21a55e59cd6faa8242b1ac74f0f7ae",
    "order": {"docs/handoffs/SATOSHI_I6N_BOUNDED_ENGINEERING_2026_10_11.md":
              "243fc9e18bc82145962e67e6c2adb361344ba0aeedee978c12320476e1ad2025"},
    "contracts": {
        "docs/tres_temas_entrevista/program_v3/I6N_JOINT_CONTEXT_CAPACITY_PROPOSAL.md":
            "d4341e72f4fc05531cfcffd5036c1968d522a105fa534d2add0ae2d10eb56487",
        "docs/tres_temas_entrevista/program_v3/BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md":
            "604f94809c41613e7346ed630d698591fadd7ecc9961b9e3cb04fcad00c01c76",
        "docs/tres_temas_entrevista/program_v3/BUSINESS_WEEKLY_WALK_FORWARD_CONTRACT_2026_10_03.md":
            "d64206b875462c913087ccba387b98422ae51c0ab88c6f490362532867b5a4c1",
    },
    "audit_request": {"docs/handoffs/I6N_STRICT_INDEPENDENT_PLAN_AUDIT_REQUEST_2026_10_11.md":
                      "2e060f7d27b4ee74ead00e3f7987269892f0fa57490dcd8b55530c9cde5c605f"},
    "implemented_section": "order section 5 (C2) and market-week contract section 5",
}

_CONTRACT_FIELDS = ("unit", "scale", "roles", "horizons_hours", "protocol_id", "calendar_id", "config_id")
_RECORD_FIELDS = ("asset", "origin", "horizon_hours", "target_timestamp", "role", "price_origin",
                  "price_target", "prediction", "checkpoint", "unit", "scale")
_FAILURE_FIELDS = ("asset", "origin", "horizon_hours", "reason")
_KEY_FIELDS = ("asset", "origin", "horizon_hours", "role")
_EXCLUSION_FIELDS = ("horizon_hours", "role", "reason", "count")
_POPULATION_FIELDS = ("population_id", "sealed_at", "keys", "exclusions", "sha256")
_RECEIPT_FIELDS = ("receipt_id", "asset", "origin", "horizon_hours", "role", "target_timestamp",
                   "checkpoint", "calendar_id", "protocol_id", "recorded_at", "action",
                   "forecast_sha256")
_DOCUMENT_FIELDS = ("schema_version", "contract", "forecasts", "forecast_failures",
                    "opportunity_population", "usage_receipts")
_COMPARABLE_CONTRACT = ("unit", "scale", "roles", "horizons_hours", "protocol_id", "calendar_id")


class PairedMetricsError(ValueError):
    """Typed refusal raised before any metric is emitted."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# --- canonical identity -----------------------------------------------------------

def canonical_json(value: Any) -> bytes:
    """Canonical bytes: sorted keys, compact separators, NaN/Infinity refused."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except ValueError as error:
        raise PairedMetricsError("NONFINITE_VALUE", f"not canonically serialisable: {error}") from error
    except (TypeError, RecursionError) as error:
        raise PairedMetricsError("SCHEMA_INVALID", f"not a JSON value: {type(error).__name__}: "
                                 f"{str(error)[:200]}") from error


def canonical_sha256(value: Any) -> str:
    """SHA-256 of ``canonical_json``; equal to ``tools.i6m_queue.digest``."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict:
    body = {k: v for k, v in value.items() if k != "sha256"}
    return dict(body, sha256=canonical_sha256(body))


def _seal_holds(value: Mapping[str, Any]) -> bool:
    return value.get("sha256") == canonical_sha256({k: v for k, v in value.items() if k != "sha256"})


def seal_population(population: Mapping[str, Any]) -> dict:
    """Seal an S support before any error is seen; the digest covers every other field."""
    return _sealed(population)


# --- primitive validation ---------------------------------------------------------

def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PairedMetricsError("IDENTITY_INVALID", f"{name} must be a non-empty trimmed string")
    return value


def _horizon(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairedMetricsError("INVALID_HORIZON", f"horizon_hours must be a positive integer, got {value!r}")
    return value


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedMetricsError("NONFINITE_VALUE", f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise PairedMetricsError("NONFINITE_VALUE", f"{name} must be a finite float, got {str(value)[:40]}")
    return number


def _fsum(values: list) -> float:
    try:
        return math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise PairedMetricsError("NONFINITE_RESULT", f"an error sum overflowed: {error}") from error


def parse_utc(name: str, value: Any) -> datetime:
    """Parse a zone-aware ISO-8601 instant and return it in UTC; naive times are refused."""
    if not isinstance(value, str):
        raise PairedMetricsError("TIMESTAMP_INVALID", f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise PairedMetricsError("TIMESTAMP_INVALID", f"{name}: {error}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PairedMetricsError("TIMESTAMP_NOT_AWARE", f"{name} has no zone: {value!r}")
    try:
        return parsed.astimezone(timezone.utc)
    except OverflowError as error:
        raise PairedMetricsError("TIMESTAMP_OUT_OF_RANGE", f"{name} {value!r}: {error}") from error


def iso_utc(value: datetime) -> str:
    """Canonical UTC text used in every key and digest."""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fields(name: str, value: Any, required: Sequence[str], optional: Sequence[str] = ()) -> Mapping:
    if not isinstance(value, Mapping):
        raise PairedMetricsError("SCHEMA_INVALID", f"{name} must be an object")
    missing = [f for f in required if f not in value]
    extra = sorted(set(value) - set(required) - set(optional))
    if missing or extra:
        raise PairedMetricsError("SCHEMA_INVALID", f"{name}: missing {missing}, unexpected {extra}")
    return value


def _list(name: str, value: Any) -> list:
    if not isinstance(value, list):
        raise PairedMetricsError("SCHEMA_INVALID", f"{name} must be a list")
    return value


def _contract(raw: Any) -> dict:
    raw = _fields("contract", raw, _CONTRACT_FIELDS, ("description",))
    result = {f: _text(f"contract.{f}", raw[f])
              for f in ("unit", "scale", "protocol_id", "calendar_id", "config_id")}
    roles = [_text("contract.roles[]", r) for r in _list("contract.roles", raw["roles"])]
    horizons = [_horizon(h) for h in _list("contract.horizons_hours", raw["horizons_hours"])]
    if not roles or len(set(roles)) != len(roles) or ALL_ROLES in roles:
        raise PairedMetricsError("CONTRACT_INVALID", "roles must be unique, non-empty and exclude ALL_ROLES")
    if not horizons or len(set(horizons)) != len(horizons):
        raise PairedMetricsError("CONTRACT_INVALID", "horizons_hours must be unique and non-empty")
    result.update(roles=roles, horizons_hours=horizons)
    if "description" in raw:
        # Kept verbatim: the normalised contract must hash to the declared contract
        # body, so ``contract_sha256`` can be recomputed from a result's contract.
        result["description"] = raw["description"]
    return result


def _identity(contract: dict, raw: Mapping, where: str) -> tuple:
    asset = _text(f"{where}.asset", raw["asset"])
    origin = parse_utc(f"{where}.origin", raw["origin"])
    h = _horizon(raw["horizon_hours"])
    if h not in contract["horizons_hours"]:
        raise PairedMetricsError("HORIZON_UNDECLARED", f"{where}: horizon {h} is not in the contract")
    return asset, origin, h


def _role(contract: dict, raw: Mapping, where: str) -> str:
    role = _text(f"{where}.role", raw["role"])
    if role not in contract["roles"]:
        raise PairedMetricsError("ROLE_UNDECLARED", f"{where}: role {role!r} is not in the contract")
    return role


def _target(raw: Mapping, origin: datetime, h: int, where: str) -> datetime:
    target = parse_utc(f"{where}.target_timestamp", raw["target_timestamp"])
    try:
        expected = origin + timedelta(hours=h)
    except OverflowError as error:
        raise PairedMetricsError("TIMESTAMP_OUT_OF_RANGE",
                                 f"{where}: origin + {h}h is outside the calendar: {error}") from error
    if target != expected:
        raise PairedMetricsError("TARGET_TIMESTAMP_MISMATCH",
                                 f"{where}: target {iso_utc(target)} != origin + {h}h")
    return target


def forecast_digest(record: Mapping[str, Any]) -> str:
    """Digest of a forecast as it was known at its origin (role and outcome excluded).

    A usage receipt carries this digest so that the scored prediction is proven
    to be the one the strategy consulted.
    """
    origin = parse_utc("origin", record["origin"])
    return canonical_sha256({
        "asset": record["asset"],
        "origin": iso_utc(origin),
        "horizon_hours": record["horizon_hours"],
        "target_timestamp": iso_utc(parse_utc("target_timestamp", record["target_timestamp"])),
        "checkpoint": record["checkpoint"],
        "prediction": _finite("prediction", record["prediction"]),
        "price_origin": _finite("price_origin", record["price_origin"]),
        "unit": record["unit"],
        "scale": record["scale"],
    })


def _observe(prices: dict, asset: str, instant: str, price: Optional[float], where: str) -> None:
    """Record one observation of the price series; ``None`` means not observable."""
    previous = prices.setdefault((asset, instant), price)
    if previous != price:
        raise PairedMetricsError("PRICE_CONFLICT", f"{where}: price of {asset} at {instant} is "
                                 f"{price!r} here but {previous!r} in another row")


def _forecasts(contract: dict, raws: list) -> tuple:
    """Validate forecast records; returns (records by key, one price per (asset, instant))."""
    by_key: dict = {}
    by_identity: dict = {}
    prices: dict = {}
    for index, raw in enumerate(raws):
        where = f"forecasts[{index}]"
        raw = _fields(where, raw, _RECORD_FIELDS)
        asset, origin, h = _identity(contract, raw, where)
        role = _role(contract, raw, where)
        target = _target(raw, origin, h, where)
        checkpoint = _text(f"{where}.checkpoint", raw["checkpoint"])
        price_origin = _finite(f"{where}.price_origin", raw["price_origin"])
        prediction = _finite(f"{where}.prediction", raw["prediction"])
        price_target = (None if raw["price_target"] is None
                        else _finite(f"{where}.price_target", raw["price_target"]))
        if raw["unit"] != contract["unit"] or raw["scale"] != contract["scale"]:
            raise PairedMetricsError("SCALE_MISMATCH", f"{where}: {raw['unit']!r}/{raw['scale']!r} differs "
                                     f"from the declared {contract['unit']!r}/{contract['scale']!r}")
        identity = (asset, iso_utc(origin), h)
        value = {"asset": asset, "origin": identity[1], "horizon_hours": h, "role": role,
                 "target_timestamp": iso_utc(target), "checkpoint": checkpoint,
                 "price_origin": price_origin, "price_target": price_target,
                 "prediction": prediction, "unit": raw["unit"], "scale": raw["scale"]}
        value["forecast_sha256"] = forecast_digest(value)
        if identity + (role,) in by_key:
            raise PairedMetricsError("DUPLICATE_RECORD", f"{where}: duplicate forecast {identity + (role,)}")
        shared = {k: v for k, v in value.items() if k != "role"}
        previous = by_identity.setdefault(identity, shared)
        if previous != shared:
            raise PairedMetricsError("FORECAST_CONFLICT", f"{where}: conflicting values for forecast {identity}")
        _observe(prices, asset, value["origin"], price_origin, f"{where}.price_origin")
        _observe(prices, asset, value["target_timestamp"], price_target, f"{where}.price_target")
        by_key[identity + (role,)] = value
    return by_key, prices


def _failures(contract: dict, raws: list, forecasts: dict) -> dict:
    present = {key[:3] for key in forecasts}
    failures: dict = {}
    for index, raw in enumerate(raws):
        where = f"forecast_failures[{index}]"
        raw = _fields(where, raw, _FAILURE_FIELDS)
        asset, origin, h = _identity(contract, raw, where)
        identity = (asset, iso_utc(origin), h)
        reason = _text(f"{where}.reason", raw["reason"])
        if identity in failures:
            raise PairedMetricsError("DUPLICATE_FAILURE", f"{where}: duplicate failure {identity}")
        if identity in present:
            raise PairedMetricsError("FORECAST_CONFLICT", f"{where}: {identity} is both failed and forecast")
        failures[identity] = reason
    return failures


def _population(contract: dict, raw: Any) -> dict:
    raw = _fields("opportunity_population", raw, _POPULATION_FIELDS)
    if not _seal_holds(raw):
        raise PairedMetricsError("POPULATION_SEAL_INVALID", "the sealed S support digest does not verify")
    population_id = _text("opportunity_population.population_id", raw["population_id"])
    sealed_at = parse_utc("opportunity_population.sealed_at", raw["sealed_at"])
    keys = []
    seen = set()
    for index, item in enumerate(_list("opportunity_population.keys", raw["keys"])):
        where = f"opportunity_population.keys[{index}]"
        item = _fields(where, item, _KEY_FIELDS)
        asset, origin, h = _identity(contract, item, where)
        key = (asset, iso_utc(origin), h, _role(contract, item, where))
        if key in seen:
            raise PairedMetricsError("DUPLICATE_POPULATION_KEY", f"{where}: duplicate key {key}")
        seen.add(key)
        keys.append(key)
    exclusions: dict = {}
    for index, item in enumerate(_list("opportunity_population.exclusions", raw["exclusions"])):
        where = f"opportunity_population.exclusions[{index}]"
        item = _fields(where, item, _EXCLUSION_FIELDS)
        h = _horizon(item["horizon_hours"])
        if h not in contract["horizons_hours"]:
            raise PairedMetricsError("HORIZON_UNDECLARED", f"{where}: horizon {h} is not in the contract")
        role = _role(contract, item, where)
        reason = _text(f"{where}.reason", item["reason"])
        count = item["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PairedMetricsError("SCHEMA_INVALID", f"{where}.count must be a non-negative integer")
        slot = exclusions.setdefault((h, role), {})
        if reason in slot:
            raise PairedMetricsError("DUPLICATE_EXCLUSION", f"{where}: duplicate reason {reason!r}")
        slot[reason] = count
    return {"population_id": population_id, "sealed_at": iso_utc(sealed_at), "keys": keys,
            "exclusions": exclusions, "sha256": raw["sha256"]}


def _receipts(contract: dict, raws: list, forecasts: dict, failures: dict) -> list:
    receipts = []
    ids, keys = set(), set()
    for index, raw in enumerate(raws):
        where = f"usage_receipts[{index}]"
        raw = _fields(where, raw, _RECEIPT_FIELDS)
        receipt_id = _text(f"{where}.receipt_id", raw["receipt_id"])
        asset, origin, h = _identity(contract, raw, where)
        role = _role(contract, raw, where)
        target = _target(raw, origin, h, where)
        checkpoint = _text(f"{where}.checkpoint", raw["checkpoint"])
        _text(f"{where}.action", raw["action"])
        digest = _text(f"{where}.forecast_sha256", raw["forecast_sha256"])
        if (raw["calendar_id"] != contract["calendar_id"]
                or raw["protocol_id"] != contract["protocol_id"]):
            raise PairedMetricsError("RECEIPT_PROTOCOL_MISMATCH", f"{where}: calendar/protocol differ")
        recorded = parse_utc(f"{where}.recorded_at", raw["recorded_at"])
        if recorded < origin:
            raise PairedMetricsError("RECEIPT_BEFORE_ORIGIN", f"{where}: recorded before its origin")
        if recorded >= target:
            raise PairedMetricsError("RECEIPT_AFTER_OUTCOME", f"{where}: recorded at or after the outcome")
        identity = (asset, iso_utc(origin), h)
        key = identity + (role,)
        if receipt_id in ids or key in keys:
            raise PairedMetricsError("DUPLICATE_RECEIPT", f"{where}: duplicate receipt {receipt_id!r} / {key}")
        ids.add(receipt_id)
        keys.add(key)
        if identity in failures:
            raise PairedMetricsError("RECEIPT_CONFLICT", f"{where}: consulted a forecast declared failed")
        record = forecasts.get(key)
        if record is not None and (record["checkpoint"] != checkpoint
                                   or record["forecast_sha256"] != digest):
            raise PairedMetricsError("RECEIPT_CONFLICT", f"{where}: consulted forecast differs from the scored one")
        receipts.append({"receipt_id": receipt_id, "key": key, "recorded_at": iso_utc(recorded),
                         "action": raw["action"], "forecast_sha256": digest, "checkpoint": checkpoint})
    return receipts


# --- scoring ----------------------------------------------------------------------

def _skill(model: Optional[float], naive: Optional[float], label: str) -> tuple:
    if model is None or naive is None:
        return None, None
    if naive == 0.0:
        return None, f"NAIVE_{label}_ZERO"
    try:
        skill = (model - naive) / naive
    except (OverflowError, ZeroDivisionError):
        skill = math.inf
    if not math.isfinite(skill):
        return None, f"SKILL_{label}_NOT_FINITE"
    return skill, None


def _metrics(n: int, sums: dict) -> dict:
    if any(not math.isfinite(v) for v in sums.values()):
        raise PairedMetricsError("NONFINITE_RESULT", "an error sum overflowed")
    mean = (lambda total: total / n) if n else (lambda total: None)
    values = {"mae": mean(sums["sum_abs_error"]), "mse": mean(sums["sum_squared_error"]),
              "naive_mae": mean(sums["naive_sum_abs_error"]),
              "naive_mse": mean(sums["naive_sum_squared_error"])}
    values["skill_mae"], values["skill_mae_undefined_reason"] = _skill(values["mae"], values["naive_mae"], "MAE")
    values["skill_mse"], values["skill_mse_undefined_reason"] = _skill(values["mse"], values["naive_mse"], "MSE")
    return values


def _row_key(key: tuple) -> list:
    return list(key)


def _derived(kind: str, n_declared: int, n: int, n_excluded: int, n_failed: int, where: str) -> dict:
    """State, coverage and unavailable-score reason derived from the counts alone.

    Raises ``GROUP_INCONSISTENT`` unless ``n_declared == n + n_excluded + n_failed``.
    """
    if n_declared != n + n_excluded + n_failed:
        raise PairedMetricsError("GROUP_INCONSISTENT", f"{where}: n_declared {n_declared} != n {n} + "
                                 f"n_excluded {n_excluded} + failures {n_failed}")
    if n_declared == 0:
        state, unavailable = EMPTY_STATE[kind], EMPTY_POPULATION
    elif n_failed:
        state, unavailable = "INCOMPLETE", (None if n else NO_SCORED_ROWS)
    elif n:
        state, unavailable = "COMPLETE", None
    else:
        state, unavailable = "NO_OBSERVABLE_LABELS", NO_SCORED_ROWS
    denominator = n_declared - n_excluded
    return {"state": state, "score_unavailable_reason": unavailable, "coverage_numerator": n,
            "coverage_denominator": denominator, "coverage": (n / denominator) if denominator else None}


def _group(kind: str, population: dict, h: int, role: str, entries: list, upstream: Optional[dict],
           contract: dict, prices: dict, identity: dict,
           upstream_by_role: Optional[dict] = None) -> dict:
    """Score one population slice; ``entries`` are (row_key, status, record_or_reason).

    One sorted row list feeds both sides. The model error uses the record's
    prediction; the naive error uses only the validated price series at the
    row's origin and target instants.
    """
    rows = sorted((k, r) for k, status, r in entries if status == "ROW")
    failures = Counter(r for _, status, r in entries if status == "FAILURE")
    excluded = Counter(r for _, status, r in entries if status == "EXCLUDED")
    row_keys, model_abs, model_sq, naive_abs, naive_sq = [], [], [], [], []
    for row_key, record in rows:
        asset = record["asset"]
        observed_target = prices[(asset, record["target_timestamp"])]
        model_error = record["prediction"] - observed_target
        naive_error = prices[(asset, record["origin"])] - observed_target
        row_keys.append(_row_key(row_key))
        model_abs.append(abs(model_error))
        model_sq.append(model_error * model_error)
        naive_abs.append(abs(naive_error))
        naive_sq.append(naive_error * naive_error)
    n = len(rows)
    sums = {"sum_abs_error": _fsum(model_abs), "sum_squared_error": _fsum(model_sq),
            "naive_sum_abs_error": _fsum(naive_abs), "naive_sum_squared_error": _fsum(naive_sq)}
    n_declared = len(entries)
    n_excluded = sum(excluded.values())
    derived = _derived(kind, n_declared, n, n_excluded, sum(failures.values()), f"{kind}/h{h}/{role}")
    group = {
        "population_kind": kind,
        "population_id": population["population_id"],
        "population_sha256": population["sha256"],
        **identity,
        "horizon_hours": h,
        "role": role,
        "unit": contract["unit"],
        "scale": contract["scale"],
        "n_declared": n_declared,
        "n_excluded": n_excluded,
        "n": n,
        **derived,
        "failures": dict(sorted(failures.items())),
        "exclusions": dict(sorted(excluded.items())),
        "exclusions_scope": EXCLUSIONS_C2_ONLY_UPSTREAM_SEPARATE if upstream_by_role is None else EXCLUSIONS_C2_ONLY,
        **sums,
        **_metrics(n, sums),
        "row_keys": row_keys,
        "row_keys_sha256": canonical_sha256(row_keys),
        "naive_rows_sha256": canonical_sha256([
            [_row_key(k), r["target_timestamp"], prices[(r["asset"], r["origin"])],
             prices[(r["asset"], r["target_timestamp"])]] for k, r in rows]),
        "model_rows_sha256": canonical_sha256([
            [_row_key(k), r["target_timestamp"], prices[(r["asset"], r["target_timestamp"])],
             r["prediction"], r["checkpoint"]] for k, r in rows]),
    }
    if upstream_by_role is None:
        group["upstream_exclusions"] = dict(upstream or {})
    else:
        group["upstream_exclusions_by_role"] = upstream_by_role
    return _sealed(group)


def _entry(key: tuple, forecasts: dict, failures: dict, kind: str) -> tuple:
    identity = key[:3]
    if identity in failures:
        return key, "FAILURE", failures[identity]
    record = forecasts.get(key)
    if record is None:
        return key, "FAILURE", MISSING_FORECAST
    if record["price_target"] is None:
        if kind == OPPORTUNITY:
            raise PairedMetricsError("DECLARED_LABEL_ABSENT",
                                     f"S member {key} has no observed label although M declared it")
        return key, "EXCLUDED", LABEL_NOT_OBSERVABLE
    return key, "ROW", record


def _all_roles(entries: list) -> list:
    """One entry per forecast identity: a failure in any role marks the identity failed."""
    by_identity: dict = {}
    for key, status, value in entries:
        by_identity.setdefault(key[:3], []).append((status, value))
    merged = []
    for identity, items in by_identity.items():
        failed = sorted(v for s, v in items if s == "FAILURE")
        rows = [v for s, v in items if s == "ROW"]
        if failed:
            merged.append((identity, "FAILURE", failed[0]))
        elif rows:
            merged.append((identity, "ROW", rows[0]))
        else:
            merged.append((identity, "EXCLUDED", sorted(v for _, v in items)[0]))
    return merged


def validate(document: Mapping[str, Any]) -> dict:
    """Validate the whole input before computing anything; returns normalised parts."""
    document = _fields("document", document, _DOCUMENT_FIELDS)
    if document["schema_version"] != INPUT_SCHEMA_VERSION:
        raise PairedMetricsError("SCHEMA_INVALID", f"schema_version must be {INPUT_SCHEMA_VERSION}")
    contract = _contract(document["contract"])
    forecasts, prices = _forecasts(contract, _list("forecasts", document["forecasts"]))
    failures = _failures(contract, _list("forecast_failures", document["forecast_failures"]), forecasts)
    population = _population(contract, document["opportunity_population"])
    receipts = _receipts(contract, _list("usage_receipts", document["usage_receipts"]), forecasts, failures)
    for key in population["keys"]:
        _entry(key, forecasts, failures, OPPORTUNITY)
    return {"contract": contract, "forecasts": forecasts, "failures": failures,
            "population": population, "receipts": receipts, "prices": prices}


def input_digests(document: Mapping[str, Any]) -> dict:
    """``config_sha256`` covers the contract; ``input_sha256`` covers the data blocks."""
    return {
        "config_sha256": canonical_sha256(document.get("contract")),
        "input_sha256": canonical_sha256({k: v for k, v in document.items() if k != "contract"}),
    }


def score(document: Mapping[str, Any],
          progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Score S_h and U_h per horizon and role; returns a sealed, JSON-safe result.

    ``progress(done, total)`` is called after each group, so a CLI can publish
    an atomic status without polling.
    """
    parts = validate(document)
    contract, forecasts = parts["contract"], parts["forecasts"]
    failures, population = parts["failures"], parts["population"]
    usage = {"population_id": "usage_receipts",
             "sha256": canonical_sha256(sorted(
                 [[*r["key"]], r["receipt_id"], r["recorded_at"], r["action"], r["forecast_sha256"]]
                 for r in parts["receipts"]))}
    digests = input_digests(document)
    if canonical_sha256(contract) != digests["config_sha256"]:
        raise PairedMetricsError("CONTRACT_INVALID", "the normalised contract differs from the declared one")
    identity = {"config_id": contract["config_id"], "protocol_id": contract["protocol_id"],
                "calendar_id": contract["calendar_id"], "contract_sha256": digests["config_sha256"]}
    s_entries = [_entry(k, forecasts, failures, OPPORTUNITY) for k in population["keys"]]
    u_entries = [_entry(r["key"], forecasts, failures, USAGE) for r in parts["receipts"]]
    plan = expected_group_keys(contract)
    groups = []
    for done, (kind, h, role) in enumerate(plan, start=1):
        entries = s_entries if kind == OPPORTUNITY else u_entries
        meta = population if kind == OPPORTUNITY else usage
        same_h = [e for e in entries if e[0][2] == h]

        def upstream(r):
            return dict(sorted(population["exclusions"].get((h, r), {}).items())) if kind == OPPORTUNITY else {}

        if role == ALL_ROLES:
            selected, upstream_counts = _all_roles(same_h), None
            by_role = {r: upstream(r) for r in contract["roles"]}
        else:
            selected, upstream_counts, by_role = [e for e in same_h if e[0][3] == role], upstream(role), None
        groups.append(_group(kind, meta, h, role, selected, upstream_counts, contract,
                             parts["prices"], identity, by_role))
        if progress is not None:
            progress(done, len(plan))
    used = {e[0] for e in s_entries + u_entries if e[1] != "FAILURE"}
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "component": COMPONENT,
        "state": "INCOMPLETE" if any(g["state"] == "INCOMPLETE" for g in groups) else "COMPLETE",
        "contract": contract,
        "contract_sha256": digests["config_sha256"],
        "input_sha256": digests["input_sha256"],
        "opportunity_support_sha256": population["sha256"],
        "opportunity_population_id": population["population_id"],
        "opportunity_sealed_at": population["sealed_at"],
        "usage_receipts_sha256": usage["sha256"],
        "counts": {
            "forecasts": len(forecasts),
            "forecast_failures": len(failures),
            "usage_receipts": len(parts["receipts"]),
            "forecasts_outside_populations": sum(1 for k in forecasts if k not in used),
        },
        "groups": groups,
        "claims": dict(CLAIMS),
        "design_basis": DESIGN_BASIS,
    }
    result["olap_records"] = _olap_rows(result)
    return _sealed(result)


_OLAP_METRIC_FIELDS = (
    "population_kind", "population_id", "population_sha256", "horizon_hours", "role", "state",
    "unit", "scale", "n_declared", "n_excluded", "n", "coverage_numerator", "coverage_denominator",
    "coverage", "sum_abs_error", "sum_squared_error", "naive_sum_abs_error", "naive_sum_squared_error",
    "mae", "mse", "naive_mae", "naive_mse", "skill_mae", "skill_mae_undefined_reason", "skill_mse",
    "skill_mse_undefined_reason", "score_unavailable_reason", "exclusions_scope", "row_keys_sha256",
    "naive_rows_sha256", "model_rows_sha256")


def olap_records(result: Mapping[str, Any]) -> list:
    """Flat, scalar-only rows for a FUTURE publication; nothing is written anywhere.

    The result is verified first (``verify_result``): a result whose group set
    differs from its contract, or that lacks its contract, is a typed refusal,
    so no group is ever published twice and no ``KeyError`` escapes.
    """
    return _olap_rows(verify_result(result))


def _olap_rows(result: Mapping[str, Any]) -> list:
    """Unverified row builder, used while scoring and inside ``verify_result``."""
    contract = result["contract"]
    common = {"schema_version": RESULT_SCHEMA_VERSION, "component": COMPONENT,
              "config_id": contract["config_id"], "protocol_id": contract["protocol_id"],
              "calendar_id": contract["calendar_id"], "contract_sha256": result["contract_sha256"],
              "input_sha256": result["input_sha256"]}
    rows = []
    for group in result["groups"]:
        rows.append({"record_type": "I6N_PAIRED_METRIC", **common,
                     **{f: group[f] for f in _OLAP_METRIC_FIELDS}, "group_sha256": group["sha256"]})
        # Upstream counts are published once, from the per-role groups; an ALL_ROLES
        # group never repeats them (roles can overlap, the counts carry no row identity).
        categories = [("FAILURE", group["failures"]), ("EXCLUSION", group["exclusions"])]
        if "upstream_exclusions" in group:
            categories.append(("UPSTREAM_EXCLUSION", group["upstream_exclusions"]))
        for category, counts in categories:
            for reason, count in counts.items():
                rows.append({"record_type": "I6N_POPULATION_COUNT", **common,
                             "population_kind": group["population_kind"],
                             "population_sha256": group["population_sha256"],
                             "horizon_hours": group["horizon_hours"], "role": group["role"],
                             "category": category, "reason": reason, "count": count})
    return rows


# --- aggregation and comparison ---------------------------------------------------

_SUM_FIELDS = ("sum_abs_error", "sum_squared_error", "naive_sum_abs_error", "naive_sum_squared_error")
_MEAN_FIELDS = ("mae", "mse", "naive_mae", "naive_mse", "skill_mae", "skill_mse",
                "skill_mae_undefined_reason", "skill_mse_undefined_reason")
_DERIVED_FIELDS = ("state", "score_unavailable_reason", "coverage_numerator", "coverage_denominator", "coverage")
_COUNT_FIELDS = ("n", "n_declared", "n_excluded")
_DIGEST_FIELDS = ("population_sha256", "row_keys_sha256", "naive_rows_sha256", "model_rows_sha256",
                  "contract_sha256")

_IDENTITY_FIELDS = ("population_kind", "horizon_hours", "role", "unit", "scale", "config_id",
                    "protocol_id", "calendar_id", "contract_sha256")
_GROUP_REQUIRED = tuple(dict.fromkeys(
    _IDENTITY_FIELDS + _SUM_FIELDS + _MEAN_FIELDS + _DERIVED_FIELDS + _COUNT_FIELDS + _DIGEST_FIELDS
    + ("sha256", "failures", "exclusions", "exclusions_scope", "row_keys")))

# Relative and absolute slack for the necessary sum relations; squares of errors
# below ~1.5e-154 underflow, so each row may lose up to that much of sqrt(sum e^2).
_REL_TOLERANCE = 1e-9
_UNDERFLOW_SLACK = 1.5e-154


def _schema(where: str, detail: str) -> PairedMetricsError:
    return PairedMetricsError("SCHEMA_INVALID", f"{where}: {detail}")


def _inconsistent(where: str, detail: str) -> PairedMetricsError:
    return PairedMetricsError("GROUP_INCONSISTENT", f"{where}: {detail}")


def _count(where: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _schema(where, f"must be a non-negative integer (not a boolean), got {value!r:.60}")
    return value


def _reason(where: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _schema(where, f"reason must be a non-empty trimmed string, got {value!r:.60}")
    return value


def _count_map(where: str, value: Any) -> dict:
    """A typed reason -> non-negative integer count map."""
    if not isinstance(value, Mapping):
        raise _schema(where, "must be a reason -> count object")
    return {_reason(where, reason): _count(f"{where}[{reason!r}]", count) for reason, count in value.items()}


def _check_sums(where: str, n: int, sums: dict) -> None:
    """Necessary conditions on error sums: finite, non-negative, zero for zero rows, possible."""
    for field, value in sums.items():
        if value < 0.0:
            raise _inconsistent(where, f"{field} is negative ({value!r})")
        if n == 0 and value != 0.0:
            raise _inconsistent(where, f"{field} is {value!r} with zero rows")
    for abs_field, sq_field in (("sum_abs_error", "sum_squared_error"),
                                ("naive_sum_abs_error", "naive_sum_squared_error")):
        total, squares = sums[abs_field], sums[sq_field]
        # sqrt(sum e^2) <= sum |e| <= sqrt(n * sum e^2) holds for every real row set.
        low = math.sqrt(squares) * (1.0 - _REL_TOLERANCE)
        high = math.sqrt(n * squares) * (1.0 + _REL_TOLERANCE) + n * _UNDERFLOW_SLACK
        if not low <= total <= high:
            raise _inconsistent(where, f"{abs_field}={total!r} is impossible with {sq_field}={squares!r} "
                                f"over n={n}")


def _verified_group(g: Any, index: Any) -> Mapping[str, Any]:
    """Re-derive and check everything a scored group asserts; never trust a sealed field.

    Types first (typed ``SCHEMA_INVALID``, never ``TypeError``), then the seal
    (``GROUP_SEAL_INVALID``) and the row digest (``ROW_DIGEST_INVALID``), then
    arithmetic consistency (``GROUP_INCONSISTENT``).
    """
    where = f"groups[{index}]"
    if not isinstance(g, Mapping) or any(f not in g for f in _GROUP_REQUIRED):
        raise _schema(where, "is not a scored group")
    if not _seal_holds(g):
        raise PairedMetricsError("GROUP_SEAL_INVALID", f"{where} digest does not verify")
    kind, role = g["population_kind"], g["role"]
    if kind not in EMPTY_STATE:
        raise _schema(where, f"population_kind must be {sorted(EMPTY_STATE)}")
    h = g["horizon_hours"]
    if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
        raise _schema(where, "horizon_hours must be a positive integer")
    for field in ("role", "unit", "scale", "config_id", "protocol_id", "calendar_id") + _DIGEST_FIELDS:
        if not isinstance(g[field], str) or not g[field]:
            raise _schema(where, f"{field} must be a non-empty string")
    all_roles = role == ALL_ROLES
    if all_roles:
        by_role = g.get("upstream_exclusions_by_role")
        if not isinstance(by_role, Mapping) or "upstream_exclusions" in g:
            raise _schema(where, "an ALL_ROLES group carries only upstream_exclusions_by_role")
        upstream = {_reason(where, r): _count_map(f"{where}.upstream_exclusions_by_role", c)
                    for r, c in by_role.items()}
    else:
        if "upstream_exclusions_by_role" in g or "upstream_exclusions" not in g:
            raise _schema(where, "a per-role group carries only upstream_exclusions")
        upstream = _count_map(f"{where}.upstream_exclusions", g["upstream_exclusions"])
    rows = g["row_keys"]
    if not isinstance(rows, list) or any(
            not isinstance(row, list) or any(isinstance(v, bool) or not isinstance(v, (str, int)) for v in row)
            for row in rows):
        raise _schema(where, "row_keys must be a list of scalar keys")
    if canonical_sha256(rows) != g["row_keys_sha256"]:
        raise PairedMetricsError("ROW_DIGEST_INVALID", f"{where} row_keys do not match their digest")
    for row in rows:
        expected_len = 3 if all_roles else 4
        if (len(row) != expected_len or not isinstance(row[0], str) or not isinstance(row[1], str)
                or row[2] != h or (not all_roles and row[3] != role)):
            raise _inconsistent(where, f"row {row} does not belong to h={h} role={role}")
    if len({tuple(row) for row in rows}) != len(rows):
        raise PairedMetricsError("DOUBLE_COUNT", f"{where} lists a row twice")
    n, n_declared, n_excluded = (_count(f"{where}.{f}", g[f]) for f in _COUNT_FIELDS)
    failures = _count_map(f"{where}.failures", g["failures"])
    exclusions = _count_map(f"{where}.exclusions", g["exclusions"])
    if n != len(rows):
        raise _inconsistent(where, f"n={n} but {len(rows)} row keys")
    if n_excluded != sum(exclusions.values()):
        raise _inconsistent(where, f"n_excluded={n_excluded} but exclusions sum to {sum(exclusions.values())}")
    derived = _derived(kind, n_declared, n, n_excluded, sum(failures.values()), where)
    derived["exclusions_scope"] = EXCLUSIONS_C2_ONLY if all_roles else EXCLUSIONS_C2_ONLY_UPSTREAM_SEPARATE
    for field, value in derived.items():
        if g[field] != value or type(g[field]) is not type(value):
            raise _inconsistent(where, f"{field}={g[field]!r:.60} but the counts give {value!r}")
    sums = {f: _finite(f"{where}.{f}", g[f]) for f in _SUM_FIELDS}
    _check_sums(where, n, sums)
    metrics = _metrics(n, sums)
    for field in _MEAN_FIELDS:
        if g[field] != metrics[field] or type(g[field]) is not type(metrics[field]):
            raise _inconsistent(where, f"{field}={g[field]!r:.60} disagrees with its sums and n")
    return {"group": g, "failures": failures, "exclusions": exclusions, "upstream": upstream}


_RESULT_DIGESTS = ("contract_sha256", "input_sha256", "opportunity_support_sha256", "usage_receipts_sha256")
_RESULT_REQUIRED = ("contract", "state", "olap_records") + _RESULT_DIGESTS
_GROUP_CONTRACT_FIELDS = ("unit", "scale", "config_id", "protocol_id", "calendar_id")


def expected_group_keys(contract: Mapping[str, Any]) -> list:
    """The exact group set a contract declares: (population_kind, horizon_hours, role).

    One group per population kind in {S, U}, per declared horizon and per
    declared role plus ``ALL_ROLES``. Both kinds are always present: an empty S
    support or a run without receipts still yields its groups, in state
    ``NO_ELIGIBLE_ROWS`` / ``NO_RECEIPTS``. A contract cannot omit a kind, and
    its roles and horizons are non-empty, so the set is never empty. This is
    the order in which ``score`` emits the groups; a reader must not depend on
    that order.
    """
    return [(kind, h, role)
            for kind in (OPPORTUNITY, USAGE)
            for h in sorted(contract["horizons_hours"])
            for role in [*contract["roles"], ALL_ROLES]]


def verify_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify a scored result's seal, its contract AND every group; returns the result.

    Typed refusals: ``RESULT_SEAL_INVALID`` when the digest does not verify;
    ``SCHEMA_INVALID`` / ``CONTRACT_INVALID`` / ``IDENTITY_INVALID`` /
    ``INVALID_HORIZON`` when a required field (including ``contract``) is
    missing or malformed; ``CONTRACT_DIGEST_INVALID`` when ``contract_sha256``
    is not the digest of the contract body; the typed group codes of
    ``aggregate_groups`` for an impossible group; ``GROUP_UNDECLARED``,
    ``DUPLICATE_GROUP`` and ``GROUP_MISSING`` unless the result holds exactly
    one group per key of ``expected_group_keys(contract)``; and
    ``RESULT_INCONSISTENT`` when a group's identity, the result state or the
    embedded ``olap_records`` disagree with the groups and the contract.
    """
    if not isinstance(result, Mapping) or not _seal_holds(result):
        raise PairedMetricsError("RESULT_SEAL_INVALID", "result digest does not verify")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION or not isinstance(result.get("groups"), list):
        raise _schema("result", f"schema_version must be {RESULT_SCHEMA_VERSION} with a groups list")
    missing = [f for f in _RESULT_REQUIRED if f not in result]
    if missing:
        raise _schema("result", f"missing {missing}")
    for field in _RESULT_DIGESTS:
        if not isinstance(result[field], str) or not result[field]:
            raise _schema("result", f"{field} must be a non-empty string")
    contract = _contract(result["contract"])
    if contract != result["contract"]:
        raise _schema("result", "contract is not in its normalised form")
    if canonical_sha256(contract) != result["contract_sha256"]:
        raise PairedMetricsError("CONTRACT_DIGEST_INVALID", "contract_sha256 does not match the contract body")
    expected = set(expected_group_keys(contract))
    populations = {OPPORTUNITY: result["opportunity_support_sha256"], USAGE: result["usage_receipts_sha256"]}
    seen: set = set()
    for index, g in enumerate(result["groups"]):
        g = _verified_group(g, index)["group"]
        group_key = (g["population_kind"], g["horizon_hours"], g["role"])
        if group_key not in expected:
            raise PairedMetricsError("GROUP_UNDECLARED", f"groups[{index}] {group_key} is not declared by the contract")
        if group_key in seen:
            raise PairedMetricsError("DUPLICATE_GROUP", f"groups[{index}] {group_key} is listed twice")
        seen.add(group_key)
        if (g["contract_sha256"] != result["contract_sha256"]
                or g["population_sha256"] != populations[g["population_kind"]]
                or any(g[f] != contract[f] for f in _GROUP_CONTRACT_FIELDS)):
            raise PairedMetricsError("RESULT_INCONSISTENT", f"groups[{index}] identity differs from the result")
    if seen != expected:
        raise PairedMetricsError("GROUP_MISSING", f"the contract declares {sorted(expected - seen)} "
                                 f"but the result has no such group")
    state = "INCOMPLETE" if any(g["state"] == "INCOMPLETE" for g in result["groups"]) else "COMPLETE"
    if result["state"] != state:
        raise PairedMetricsError("RESULT_INCONSISTENT", f"result state {result['state']!r} but groups give {state}")
    if result["olap_records"] != _olap_rows(result):
        raise PairedMetricsError("RESULT_INCONSISTENT", "olap_records differ from the rows its groups give")
    return result


def aggregate_groups(groups: Sequence[Mapping[str, Any]]) -> dict:
    """Pool scored groups (e.g. weeks) by sums and counts; returns a sealed aggregate.

    Every group is re-verified by ``_verified_group`` (types, seal, row digest,
    count identity, possible sums, derived state/coverage/means); all groups
    must share one identity (population kind, h, role, unit, scale, config,
    protocol, calendar and contract digest) or ``AGGREGATE_MISMATCH`` is raised,
    so S never pools with U and different configs, protocols or calendars never
    pool silently. A row in two groups raises ``DOUBLE_COUNT``. Upstream
    exclusions are pooled apart from C2's exclusions (per role for
    ``ALL_ROLES``). The aggregate is an end product: it is sealed but is not
    itself accepted as an input of another aggregation.
    """
    if not isinstance(groups, Sequence) or isinstance(groups, (str, bytes)) or not groups:
        raise PairedMetricsError("EMPTY_AGGREGATE", "nothing to aggregate")
    checked = [_verified_group(g, i) for i, g in enumerate(groups)]
    groups = [c["group"] for c in checked]
    shape = {tuple(g[f] for f in _IDENTITY_FIELDS) for g in groups}
    if len(shape) != 1:
        raise PairedMetricsError("AGGREGATE_MISMATCH", f"groups differ in identity {_IDENTITY_FIELDS}: {sorted(shape)}")
    identity = dict(zip(_IDENTITY_FIELDS, shape.pop()))
    all_roles = identity["role"] == ALL_ROLES
    seen = set()
    for g in groups:
        for row in g["row_keys"]:
            marker = tuple(row)
            if marker in seen:
                raise PairedMetricsError("DOUBLE_COUNT", f"row {row} appears in more than one group")
            seen.add(marker)
    n = sum(g["n"] for g in groups)
    sums = {f: _fsum([g[f] for g in groups]) for f in _SUM_FIELDS}
    failures, exclusions, upstream = Counter(), Counter(), Counter()
    upstream_by_role: dict = {}
    for c in checked:
        failures.update(c["failures"])
        exclusions.update(c["exclusions"])
        if all_roles:
            for r, counts in c["upstream"].items():
                upstream_by_role.setdefault(r, Counter()).update(counts)
        else:
            upstream.update(c["upstream"])
    n_declared = sum(g["n_declared"] for g in groups)
    n_excluded = sum(g["n_excluded"] for g in groups)
    derived = _derived(identity["population_kind"], n_declared, n, n_excluded, sum(failures.values()), "aggregate")
    row_keys = [list(r) for r in sorted(seen)]
    aggregate = {
        **identity, "population_id": "AGGREGATE",
        "population_sha256": canonical_sha256(sorted(g["population_sha256"] for g in groups)),
        "n_declared": n_declared, "n_excluded": n_excluded, "n": n,
        **derived,
        "failures": dict(sorted(failures.items())), "exclusions": dict(sorted(exclusions.items())),
        "exclusions_scope": EXCLUSIONS_C2_ONLY if all_roles else EXCLUSIONS_C2_ONLY_UPSTREAM_SEPARATE,
        **sums, **_metrics(n, sums),
        "row_keys": row_keys, "row_keys_sha256": canonical_sha256(row_keys),
        "constituent_group_sha256": sorted(g["sha256"] for g in groups),
        "constituent_naive_rows_sha256": canonical_sha256(sorted(g["naive_rows_sha256"] for g in groups)),
        "constituent_model_rows_sha256": canonical_sha256(sorted(g["model_rows_sha256"] for g in groups)),
        "constituents": len(groups),
    }
    if all_roles:
        aggregate["upstream_exclusions_by_role"] = {r: dict(sorted(c.items()))
                                                    for r, c in sorted(upstream_by_role.items())}
    else:
        aggregate["upstream_exclusions"] = dict(sorted(upstream.items()))
    return _sealed(aggregate)


def compare_configs(result_a: Mapping[str, Any], result_b: Mapping[str, Any]) -> dict:
    """Compare two configurations on the S population of one previously sealed support.

    No intersection is ever built: a different support raises
    ``SUPPORT_MISMATCH``; any incomplete group returns
    ``NOT_COMPARABLE_INCOMPLETE`` with coverage and no metric differences.
    U populations depend on each strategy's own decisions and are not compared.
    Both results go through ``verify_result`` first, so a result without its
    contract, with a group set that differs from its contract or with a
    contract that does not match its digest is a typed refusal.
    """
    verify_result(result_a)
    verify_result(result_b)
    for field in _COMPARABLE_CONTRACT:
        if result_a["contract"][field] != result_b["contract"][field]:
            raise PairedMetricsError("CONTRACT_MISMATCH", f"contract field {field!r} differs")
    if result_a["opportunity_support_sha256"] != result_b["opportunity_support_sha256"]:
        raise PairedMetricsError("SUPPORT_MISMATCH", "results were not scored on the same sealed S support")

    def opportunity(result):
        return {(g["horizon_hours"], g["role"]): g for g in result["groups"]
                if g["population_kind"] == OPPORTUNITY}

    a, b = opportunity(result_a), opportunity(result_b)
    if set(a) != set(b):
        raise PairedMetricsError("CONTRACT_MISMATCH", "group sets differ")
    order = sorted(a, key=lambda k: (k[0], k[1]))
    coverage = [{"horizon_hours": h, "role": role,
                 "a": {f: a[(h, role)][f] for f in ("state", "n", "n_declared")},
                 "b": {f: b[(h, role)][f] for f in ("state", "n", "n_declared")}}
                for h, role in order]
    comparison = {"schema_version": COMPARISON_SCHEMA_VERSION, "component": COMPONENT,
                  "support_sha256": result_a["opportunity_support_sha256"],
                  "config_a": result_a["contract"]["config_id"],
                  "config_b": result_b["contract"]["config_id"],
                  "result_a_sha256": result_a["sha256"], "result_b_sha256": result_b["sha256"],
                  "coverage": coverage, "claims": dict(CLAIMS)}
    if any(a[k]["state"] == "INCOMPLETE" or b[k]["state"] == "INCOMPLETE" for k in order):
        return _sealed(dict(comparison, state="NOT_COMPARABLE_INCOMPLETE", comparisons=None))
    comparisons = []
    for h, role in order:
        ga, gb = a[(h, role)], b[(h, role)]
        if ga["naive_rows_sha256"] != gb["naive_rows_sha256"] or ga["row_keys_sha256"] != gb["row_keys_sha256"]:
            raise PairedMetricsError("NAIVE_MISMATCH", f"h={h} role={role}: rows or naive inputs differ")
        comparisons.append({
            "horizon_hours": h, "role": role, "state": ga["state"], "n": ga["n"],
            "naive_mae": ga["naive_mae"], "mae_a": ga["mae"], "mae_b": gb["mae"],
            "delta_mae": None if ga["mae"] is None else gb["mae"] - ga["mae"],
            "skill_mae_a": ga["skill_mae"], "skill_mae_b": gb["skill_mae"],
            "naive_mse": ga["naive_mse"], "mse_a": ga["mse"], "mse_b": gb["mse"],
        })
    return _sealed(dict(comparison, state="COMPARABLE", comparisons=comparisons))


__all__ = [
    "ALL_ROLES", "CLAIMS", "COMPONENT", "DESIGN_BASIS", "EMPTY_STATE", "EXCLUSIONS_C2_ONLY",
    "EXCLUSIONS_C2_ONLY_UPSTREAM_SEPARATE", "INPUT_SCHEMA_VERSION", "OPPORTUNITY", "RESULT_SCHEMA_VERSION",
    "USAGE", "PairedMetricsError", "aggregate_groups", "canonical_json", "canonical_sha256",
    "compare_configs", "expected_group_keys", "forecast_digest", "input_digests", "iso_utc", "olap_records", "parse_utc",
    "score", "seal_population", "validate", "verify_result",
]
