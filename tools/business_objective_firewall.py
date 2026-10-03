"""Dependency-free firewall between model selection and external test scoring.

The firewall is a pure, immutable state machine.  It exposes opaque test-week
identifiers during validation, seals the selected procedure before test data
may be observed, and releases aggregate test metrics only after every expected
week has a terminal disposition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping, Sequence


class FirewallError(ValueError):
    """Raised when an operation would cross the external-test boundary."""


class Phase(str, Enum):
    """Lifecycle phases for one validation-and-test protocol."""

    VALIDATION_SELECTION = "VALIDATION_SELECTION"
    PROCEDURE_SEALED = "PROCEDURE_SEALED"
    TEST_TRAVERSAL = "TEST_TRAVERSAL"
    CLOSED = "CLOSED"


class Actor(str, Enum):
    """Selection-side consumers that must never receive external test data."""

    CANDIDATE_WORKER = "CANDIDATE_WORKER"
    SELECTOR = "SELECTOR"
    OPTIMIZER_CALLBACK = "OPTIMIZER_CALLBACK"
    DOIN_PAYLOAD = "DOIN_PAYLOAD"


class WeekDisposition(str, Enum):
    """Terminal outcomes for an expected external test week."""

    SCORED = "SCORED"
    FAILED = "FAILED"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True, order=True)
class MetricValue:
    """One finite metric value stored in deterministic key order."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise FirewallError("metric name must be non-empty")
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise FirewallError(f"metric {self.name!r} must be finite")


@dataclass(frozen=True)
class ProcedureIdentity:
    """Digest-bound identity of the validation-selected procedure."""

    selected_candidate_id: str
    digest: str

    def __post_init__(self) -> None:
        if not self.selected_candidate_id:
            raise FirewallError("selected candidate id must be non-empty")
        if len(self.digest) != 64 or any(c not in "0123456789abcdef" for c in self.digest):
            raise FirewallError("procedure digest must be lowercase SHA-256")


@dataclass(frozen=True)
class SelectionReceipt:
    """Digest-only record proving that a selection payload crossed the guard."""

    actor: Actor
    payload_digest: str


@dataclass(frozen=True)
class WeekObservation:
    """A terminal disposition and optional metrics for one expected week."""

    week_id: str
    disposition: WeekDisposition
    metrics: tuple[MetricValue, ...]
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.week_id:
            raise FirewallError("week id must be non-empty")
        if tuple(sorted(self.metrics)) != self.metrics:
            raise FirewallError("week metrics must be sorted by name")
        if len({metric.name for metric in self.metrics}) != len(self.metrics):
            raise FirewallError("week metric names must be unique")
        if self.disposition is WeekDisposition.SCORED:
            if not self.metrics:
                raise FirewallError("a scored week requires metrics")
            if self.reason:
                raise FirewallError("a scored week cannot carry a failure reason")
        else:
            if self.metrics:
                raise FirewallError("an unscored terminal week cannot carry metrics")
            if not self.reason:
                raise FirewallError("an unscored terminal week requires a reason")

    @classmethod
    def scored(cls, week_id: str, metrics: Mapping[str, float]) -> "WeekObservation":
        """Construct a scored observation from a metric mapping."""

        values = tuple(
            sorted(MetricValue(str(name), float(value)) for name, value in metrics.items())
        )
        return cls(week_id, WeekDisposition.SCORED, values)


def _canonical_bytes(value: Any, *, enforce_selection_boundary: bool = True) -> bytes:
    """Return strict canonical JSON bytes after validating supported values."""

    _validate_json_value(
        value,
        path="$",
        allow_test_ids=True if enforce_selection_boundary else None,
    )
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise FirewallError(f"payload is not canonical JSON: {exc}") from exc
    return encoded.encode("ascii")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


_OPAQUE_TEST_ID_KEYS = {"test_week_id", "test_week_ids", "expected_test_week_ids"}


def _is_opaque_identifier_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    return (
        isinstance(value, (list, tuple))
        and all(isinstance(item, str) and item for item in value)
    )


def _validate_json_value(value: Any, path: str, allow_test_ids: bool | None) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FirewallError(f"non-finite number at {path}")
        return
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise FirewallError(f"mapping keys must be non-empty strings at {path}")
            key = raw_key.lower()
            child_path = f"{path}.{raw_key}"
            if allow_test_ids is not None and "test" in key:
                if not (
                    allow_test_ids
                    and key in _OPAQUE_TEST_ID_KEYS
                    and _is_opaque_identifier_value(child)
                ):
                    raise FirewallError(
                        f"external test material is forbidden during selection at {child_path}"
                    )
                continue
            _validate_json_value(child, child_path, allow_test_ids)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]", allow_test_ids)
        return
    raise FirewallError(f"unsupported payload value at {path}: {type(value).__name__}")


@dataclass(frozen=True)
class BusinessObjectiveFirewall:
    """Immutable validation/test firewall with crash-safe replay semantics."""

    expected_test_week_ids: tuple[str, ...]
    phase: Phase = Phase.VALIDATION_SELECTION
    selection_receipts: tuple[SelectionReceipt, ...] = ()
    procedure_identity: ProcedureIdentity | None = None
    test_observations: tuple[WeekObservation, ...] = ()
    aggregate_metrics: tuple[MetricValue, ...] = ()

    def __post_init__(self) -> None:
        self._validate_state()

    @classmethod
    def create(cls, expected_test_week_ids: Sequence[str]) -> "BusinessObjectiveFirewall":
        """Create a firewall with an ordered, non-empty opaque week denominator."""

        return cls(tuple(expected_test_week_ids))

    def opaque_test_week_ids(self) -> tuple[str, ...]:
        """Return identifiers only; no arrays, labels, or metrics are exposed."""

        return self.expected_test_week_ids

    def record_selection_payload(
        self, actor: Actor, payload: Mapping[str, Any]
    ) -> "BusinessObjectiveFirewall":
        """Validate and receipt one selection-side payload without retaining it."""

        if self.phase is not Phase.VALIDATION_SELECTION:
            raise FirewallError("selection is closed after the procedure is sealed")
        receipt = SelectionReceipt(actor=actor, payload_digest=_digest(payload))
        return self._replace(selection_receipts=self.selection_receipts + (receipt,))

    def seal_procedure(
        self,
        selected_candidate_id: str,
        procedure: Mapping[str, Any],
        validation_summary: Mapping[str, Any],
    ) -> "BusinessObjectiveFirewall":
        """Freeze a validation-selected procedure before any test observation."""

        if self.phase is not Phase.VALIDATION_SELECTION:
            raise FirewallError("procedure is already sealed; reselection is forbidden")
        if not selected_candidate_id:
            raise FirewallError("selected candidate id must be non-empty")
        identity_payload = {
            "selected_candidate_id": selected_candidate_id,
            "procedure": procedure,
            "validation_summary": validation_summary,
            "selection_receipts": [
                {"actor": item.actor.value, "payload_digest": item.payload_digest}
                for item in self.selection_receipts
            ],
        }
        identity = ProcedureIdentity(selected_candidate_id, _digest(identity_payload))
        return self._replace(phase=Phase.PROCEDURE_SEALED, procedure_identity=identity)

    def open_test_traversal(self) -> "BusinessObjectiveFirewall":
        """Open the single external traversal after procedure sealing."""

        if self.phase is Phase.VALIDATION_SELECTION:
            raise FirewallError("a sealed procedure is required before opening test")
        if self.phase is not Phase.PROCEDURE_SEALED:
            raise FirewallError("test traversal can be opened exactly once")
        return self._replace(phase=Phase.TEST_TRAVERSAL)

    def observe_week(self, observation: WeekObservation) -> "BusinessObjectiveFirewall":
        """Append one ordered terminal week, or idempotently replay the same receipt."""

        for retained in self.test_observations:
            if retained.week_id == observation.week_id:
                if retained == observation:
                    return self
                raise FirewallError(
                    f"conflicting duplicate observation for {observation.week_id}"
                )
        if self.phase is not Phase.TEST_TRAVERSAL:
            raise FirewallError("week observations require an open test traversal")
        index = len(self.test_observations)
        if index >= len(self.expected_test_week_ids):
            raise FirewallError("all expected test weeks already have dispositions")
        expected = self.expected_test_week_ids[index]
        if observation.week_id != expected:
            raise FirewallError(
                f"ordered traversal expected {expected}, got {observation.week_id}"
            )
        return self._replace(test_observations=self.test_observations + (observation,))

    def close_test_traversal(self) -> "BusinessObjectiveFirewall":
        """Close a complete traversal and compute week-macro aggregate metrics."""

        if self.phase is not Phase.TEST_TRAVERSAL:
            raise FirewallError("only an open test traversal can be closed")
        if len(self.test_observations) != len(self.expected_test_week_ids):
            missing = self.expected_test_week_ids[len(self.test_observations) :]
            raise FirewallError(f"missing terminal dispositions for weeks: {missing}")
        aggregate = self._compute_aggregate()
        return self._replace(phase=Phase.CLOSED, aggregate_metrics=aggregate)

    def aggregated_test_metrics(self) -> dict[str, float]:
        """Release aggregate external metrics only after complete closure."""

        if self.phase is not Phase.CLOSED:
            raise FirewallError("aggregated test metrics are sealed until traversal closes")
        return {metric.name: metric.value for metric in self.aggregate_metrics}

    def to_dict(self) -> dict[str, Any]:
        """Serialize state using JSON-compatible values for crash recovery."""

        return {
            "schema": "business_objective_firewall.v1",
            "expected_test_week_ids": list(self.expected_test_week_ids),
            "phase": self.phase.value,
            "selection_receipts": [
                {"actor": item.actor.value, "payload_digest": item.payload_digest}
                for item in self.selection_receipts
            ],
            "procedure_identity": None
            if self.procedure_identity is None
            else {
                "selected_candidate_id": self.procedure_identity.selected_candidate_id,
                "digest": self.procedure_identity.digest,
            },
            "test_observations": [self._observation_to_dict(item) for item in self.test_observations],
            "aggregate_metrics": [
                {"name": item.name, "value": item.value} for item in self.aggregate_metrics
            ],
        }

    def to_json(self) -> str:
        """Serialize state to canonical ASCII JSON."""

        return _canonical_bytes(
            self.to_dict(), enforce_selection_boundary=False
        ).decode("ascii")

    @classmethod
    def from_json(cls, encoded: str) -> "BusinessObjectiveFirewall":
        """Restore and validate state from serialized JSON."""

        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise FirewallError(f"invalid firewall JSON: {exc}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BusinessObjectiveFirewall":
        """Restore state while rejecting malformed or inconsistent records."""

        try:
            if payload["schema"] != "business_objective_firewall.v1":
                raise FirewallError("unsupported firewall schema")
            receipts = tuple(
                SelectionReceipt(Actor(item["actor"]), str(item["payload_digest"]))
                for item in payload["selection_receipts"]
            )
            raw_identity = payload["procedure_identity"]
            identity = None
            if raw_identity is not None:
                identity = ProcedureIdentity(
                    str(raw_identity["selected_candidate_id"]),
                    str(raw_identity["digest"]),
                )
            observations = tuple(
                cls._observation_from_dict(item) for item in payload["test_observations"]
            )
            aggregate = tuple(
                MetricValue(str(item["name"]), float(item["value"]))
                for item in payload["aggregate_metrics"]
            )
            return cls(
                expected_test_week_ids=tuple(payload["expected_test_week_ids"]),
                phase=Phase(payload["phase"]),
                selection_receipts=receipts,
                procedure_identity=identity,
                test_observations=observations,
                aggregate_metrics=aggregate,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, FirewallError):
                raise
            raise FirewallError(f"malformed firewall state: {exc}") from exc

    def _validate_state(self) -> None:
        weeks = self.expected_test_week_ids
        if not weeks or any(not isinstance(item, str) or not item for item in weeks):
            raise FirewallError("expected test week ids must be non-empty strings")
        if len(set(weeks)) != len(weeks):
            raise FirewallError("expected test week ids must be unique")
        observed_ids = tuple(item.week_id for item in self.test_observations)
        if observed_ids != weeks[: len(observed_ids)]:
            raise FirewallError("retained observations are not an ordered week prefix")
        if self.phase is Phase.VALIDATION_SELECTION:
            if self.procedure_identity or self.test_observations or self.aggregate_metrics:
                raise FirewallError("validation state cannot contain test or sealed state")
        else:
            if self.procedure_identity is None:
                raise FirewallError("post-validation state requires a procedure identity")
        if self.phase is Phase.PROCEDURE_SEALED and self.test_observations:
            raise FirewallError("sealed state cannot contain test observations")
        if self.phase is not Phase.CLOSED and self.aggregate_metrics:
            raise FirewallError("aggregate metrics may exist only in closed state")
        if self.phase is Phase.CLOSED:
            if len(self.test_observations) != len(weeks):
                raise FirewallError("closed state requires every terminal disposition")
            if self.aggregate_metrics != self._compute_aggregate():
                raise FirewallError("closed aggregate does not match week observations")

    def _compute_aggregate(self) -> tuple[MetricValue, ...]:
        scored = [
            {metric.name: metric.value for metric in item.metrics}
            for item in self.test_observations
            if item.disposition is WeekDisposition.SCORED
        ]
        if not scored:
            return ()
        names = set(scored[0])
        if any(set(item) != names for item in scored[1:]):
            raise FirewallError("all scored weeks must expose the same metric names")
        return tuple(
            MetricValue(name, math.fsum(item[name] for item in scored) / len(scored))
            for name in sorted(names)
        )

    def _replace(self, **changes: Any) -> "BusinessObjectiveFirewall":
        values = {
            "expected_test_week_ids": self.expected_test_week_ids,
            "phase": self.phase,
            "selection_receipts": self.selection_receipts,
            "procedure_identity": self.procedure_identity,
            "test_observations": self.test_observations,
            "aggregate_metrics": self.aggregate_metrics,
        }
        values.update(changes)
        return BusinessObjectiveFirewall(**values)

    @staticmethod
    def _observation_to_dict(observation: WeekObservation) -> dict[str, Any]:
        return {
            "week_id": observation.week_id,
            "disposition": observation.disposition.value,
            "metrics": [
                {"name": metric.name, "value": metric.value}
                for metric in observation.metrics
            ],
            "reason": observation.reason,
        }

    @staticmethod
    def _observation_from_dict(payload: Mapping[str, Any]) -> WeekObservation:
        return WeekObservation(
            week_id=str(payload["week_id"]),
            disposition=WeekDisposition(payload["disposition"]),
            metrics=tuple(
                MetricValue(str(item["name"]), float(item["value"]))
                for item in payload["metrics"]
            ),
            reason=str(payload["reason"]),
        )
