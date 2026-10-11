"""Bounded, pure-stdlib C1 -> C2 adapter; not an automated campaign runner.

The caller supplies a validated C1 SupportPlan, one asset, one explicit split
and the actual artifact seal timestamp. No calendar, labels, predictions, receipts or
timestamps are invented. S is exactly C1's per-head M, never its prediction
availability mask. C1 exclusions remain upstream; C2 owns paired arithmetic,
forecast failures and usage-receipt timing/identity validation. The adapter also
checks each U receipt against C1 E and the declared input scope. Audited
components are unchanged.

Limits: the supplied plan, observed-label evidence and seal time require
external custody. Unkeyed digests cannot prove a historical S artifact was
sealed before candidate comparison or authenticate prices. S sealing is not
prospective availability. This adapter cannot turn historical label availability
into prospective knowledge. It does not validate causal feature construction,
train models, authorize TEST, publish metrics, or establish campaign readiness.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from . import eligibility as c1
from . import paired_metrics as c2


_ROLES = {c1.Role.ENTRY: "entry_origin", c1.Role.MANAGEMENT: "management_origin"}


class BusinessMetricsError(ValueError):
    """Adapter refusal with stable ``code`` and ``detail`` attributes.

Native C1 EligibilityError and C2 PairedMetricsError propagate unchanged.
"""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _identity(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BusinessMetricsError("IDENTITY_INVALID", f"{name} must be non-empty trimmed text")
    return value


def calendar_identity(plan: c1.SupportPlan) -> str:
    """Return C2's calendar_id: ``i6n.calendar:`` + canonical JSON of ID/version.

This collision-free encoding preserves both mandatory C1 identity fields in
C2's single text field. It is not a calendar lookup or a calendar default.
"""
    if not isinstance(plan, c1.SupportPlan):
        raise BusinessMetricsError("PLAN_REQUIRED", "supply a C1 prepare_support_plan result")
    return "i6n.calendar:" + c2.canonical_json({
        "calendar_id": plan.calendar.calendar_id, "version": plan.calendar.version,
    }).decode("utf-8")


def build_opportunity_population(
    plan: c1.SupportPlan, *, asset: str, split_id: str, sealed_at: str,
) -> dict:
    """Construct and seal C2 S from EXACT per-head C1 M, with typed exclusions.

``plan`` must come from C1 ``prepare_support_plan`` with observed evidence
(an empty evidence set is allowed), exactly one split, and no outside-calendar
origins. ``split_id`` must name that split; membership is by target instant.
ENTRY maps to entry_origin and MANAGEMENT to management_origin. Invalid heads
never remove valid siblings. No prediction availability enters this function.

``sealed_at`` is required zone-aware ISO text for the ACTUAL artifact seal time,
not prospective label availability. Historical M may be sealed after its
outcomes (e.g. 2024 history sealed in 2026); never backdate it to precede those
outcomes. It is normalized, not replaced by the wall clock. The caller must
retain independent evidence that the artifact was sealed before candidate
comparison; this pure adapter cannot attest that ordering. U receipt times
remain subject to C2's separate origin <= recorded_at < target constraint.
The population ID binds the asset, calendar/version, split bounds, C1 support
table and exact keys/counts. Primary exclusion counts use C1's own precedence
and are summed across weeks per role/head, not deduplicated across roles.
"""
    calendar = calendar_identity(plan)
    asset = _identity("asset", asset)
    split_id = _identity("split_id", split_id)
    if plan.observed is None:
        raise BusinessMetricsError("LABEL_EVIDENCE_REQUIRED", "C1 observed label evidence is mandatory")
    if len(plan.splits) != 1:
        raise BusinessMetricsError("SINGLE_SPLIT_REQUIRED", "supply exactly one explicit C1 split")
    split = plan.splits[0]
    if split.split_id != split_id:
        raise BusinessMetricsError("SPLIT_MISMATCH", "requested split differs from the C1 split")
    if plan.origins_outside_calendar_weeks:
        raise BusinessMetricsError("OUTSIDE_CALENDAR_ORIGIN", "C1 reports outside-calendar origins")
    sealed = c2.parse_utc("sealed_at", sealed_at)
    keys = []
    for week in plan.weeks:
        for origin in plan.origins_by_week[week.week_id]:
            if not week.market_open <= origin < week.market_close:
                raise BusinessMetricsError("OUTSIDE_CALENDAR_ORIGIN", "origin is outside its C1 week")
            for role in plan.roles:
                decisions = c1.evaluate_heads(
                    origin, plan.horizons, role, week,
                    input_available=origin not in plan.input_unavailable,
                    observed_targets=plan.observed, split=split,
                )
                for decision in decisions:
                    if decision.label_eligible:
                        keys.append({"asset": asset, "origin": c2.iso_utc(origin),
                                     "horizon_hours": decision.horizon_hours, "role": _ROLES[role]})
    keys.sort(key=lambda k: (k["asset"], k["origin"], k["horizon_hours"], k["role"]))

    # C1's support table owns primary-reason precedence, notably split exclusions.
    rows = list(c1.iter_support_rows(plan))
    counts = Counter()
    for row in rows:
        for reasons in (row["operational_exclusions"], row["label"]["exclusions_among_operational"]):
            for reason, count in reasons.items():
                counts[(row["horizon_hours"], _ROLES[c1.Role(row["role"])], reason)] += count
    exclusions = [{"horizon_hours": h, "role": role, "reason": reason, "count": count}
                  for (h, role, reason), count in sorted(counts.items())]
    provenance = {
        "adapter_schema": "i6n.business_metrics.support.v1", "asset": asset,
        "calendar_id": calendar, "split": split.to_json(),
        "c1_support": c1.assemble_support_table(plan, rows),
        "keys": keys, "exclusions": exclusions,
    }
    return c2.seal_population({
        "population_id": "i6n.c1-m:" + c2.canonical_sha256(provenance),
        "sealed_at": c2.iso_utc(sealed), "keys": keys, "exclusions": exclusions,
    })


def score_business_metrics(
    plan: c1.SupportPlan, document: Mapping[str, Any], *, asset: str,
    split_id: str, sealed_at: str, stored_result: Mapping[str, Any] | None = None,
) -> dict:
    """Verify C1/C2 binding and freshly score the supplied C2 forecast document.

Contract horizons/roles must match C1 in declaration order; calendar_id must
equal ``calendar_identity(plan)``. The whole supplied population must equal
``build_opportunity_population`` canonically, including exclusions, provenance
ID and seal. A resealed subset cannot drop difficult rows. Forecasts, failures
and receipts must use the explicit asset and enumerated C1 origins. Extra
ineligible forecast heads are not filtered: C2 validates them and reports
unused forecasts; they cannot shrink S. Every U receipt must also pass C1 E
for its own origin, head, role and supplied week, even without a forecast. This
validates calendar E, not full strategy eligibility. U is NOT intersected with
historical M: absent future labels are handled by C2, independently of E.
C2's pre-outcome receipt constraints are unchanged by historical S seal times.

Returns a native C2 result computed from the INPUT forecasts. Optional
``stored_result`` is verification-only: it must exactly match the fresh result,
even if its seals and arithmetic are internally consistent. C2 missing forecast
failures retain their denominator. Null S labels contradicting observed C1 M
raise C2 DECLARED_LABEL_ABSENT; missing C1 label evidence is counted upstream.
Neither inputs nor stored metrics are mutated. No persistence or I/O occurs.
"""
    expected = build_opportunity_population(plan, asset=asset, split_id=split_id, sealed_at=sealed_at)
    if not isinstance(document, Mapping) or not isinstance(document.get("contract"), Mapping):
        raise BusinessMetricsError("CONTRACT_MISMATCH", "a C2 document and contract are required")
    contract = document["contract"]
    required = {"horizons_hours": list(plan.horizons),
                "roles": [_ROLES[r] for r in plan.roles], "calendar_id": calendar_identity(plan)}
    for name, value in required.items():
        if c2.canonical_json(contract.get(name)) != c2.canonical_json(value):
            raise BusinessMetricsError("CONTRACT_MISMATCH", f"C2 {name} differs from C1")
    if c2.canonical_json(document.get("opportunity_population")) != c2.canonical_json(expected):
        raise BusinessMetricsError("SUPPORT_MISMATCH", "supplied population differs from fresh C1 M support")

    parts = c2.validate(document)
    origins = {c2.iso_utc(t): (t, week) for week in plan.weeks
               for t in plan.origins_by_week[week.week_id]}
    identities = [*parts["forecasts"], *parts["failures"], *(r["key"] for r in parts["receipts"])]
    for identity in identities:
        if identity[0] != asset or identity[1] not in origins:
            raise BusinessMetricsError("INPUT_SCOPE_MISMATCH", "input asset/origin is outside the supplied C1 plan")
    c1_roles = {name: role for role, name in _ROLES.items()}
    for receipt in parts["receipts"]:
        _, origin_text, h, role = receipt["key"]
        origin, week = origins[origin_text]
        eligible, reasons = c1.operational_eligibility(origin, h, c1_roles[role], week)
        if not eligible:
            raise BusinessMetricsError(
                "USAGE_OPERATIONALLY_INELIGIBLE",
                f"receipt {receipt['receipt_id']}: C1 E failed: "
                + ", ".join(reason.value for reason in reasons),
            )
    fresh = c2.score(document)
    if stored_result is not None and c2.canonical_json(stored_result) != c2.canonical_json(fresh):
        raise BusinessMetricsError("RESULT_MISMATCH", "stored result differs from freshly scored input forecasts")
    return fresh
