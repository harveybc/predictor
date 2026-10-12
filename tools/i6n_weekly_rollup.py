"""I6-N lane C: annual roll-up of weekly-refit receipts (sums/counts, identical naive).

Consumes ``i6n.candidate_receipt.v1`` rows (lane A, INTERFACE_A) for ONE arm,
horizon, head, role, population and split, one row per weekly unit (``fold``),
plus ``support_sha256`` = the unit's frozen ``origins_sha256`` from the weekly
campaign plan. The annual metric is sum(error) / sum(n) with the naive pooled
over exactly the same rows; the mean of weekly MAEs is reported only as a
labelled diagnostic. A year with an unresolved missing week is INCOMPLETE and
has no annual score. Weeks the plan froze with zero support carry no score.

Scope guard: an INNER_FOLD (TRAIN/inner search fitness, e.g. 70 overlapping
inner rows of h72) is never comparable with an ANNUAL_WEEKLY_REFIT result, and
different horizons are never compared.

Pure standard library; no I/O, no warehouse writes, no fits.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional

ANNUAL_SCOPE = "ANNUAL_WEEKLY_REFIT"
INNER_SCOPE = "INNER_FOLD"
STATE_SCHEMA = "i6n.weekly_rollup.state.v1"
SUMMARY_SCHEMA = "i6n.weekly_rollup.summary.v1"
RECEIPT_SCHEMA = "i6n.candidate_receipt.v1"
SUMS = ("sum_abs_error", "sum_sq_error", "naive_sum_abs_error", "naive_sum_sq_error")
IDENTITY = ("schema_version", "experiment", "candidate_identity_sha256", "horizon_hours",
            "head_offset_hours", "role", "population", "split", "metric", "unit", "scale")


class RollupError(ValueError):
    """Typed refusal with a stable ``code``."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _nonneg(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise RollupError("RECEIPT_INVALID", f"{name} must be a finite non-negative number")
    return float(value)


def _skill(model: Optional[float], naive: Optional[float]):
    if model is None or naive is None:
        return None, "NO_SCORE"
    if naive == 0:
        return None, "NAIVE_ZERO"
    return (model - naive) / naive, None


class AnnualAccumulator:
    """Durable, order-independent, idempotent accumulator of weekly rows."""

    def __init__(self, *, experiment: str, candidate_identity_sha256: str, horizon_hours: int,
                 role: str, expected: Mapping[str, Mapping[str, Any]], population: str = "S",
                 split: str = "VALIDATION", head_offset_hours: Optional[int] = None,
                 metric: str = "ABS_AND_SQUARED_ERROR_SUMS", unit: str = "PRICE",
                 scale: str = "ORIGINAL_PRICE"):
        if not expected:
            raise RollupError("EXPECTED_EMPTY", "the plan's weekly support is required")
        for fold, e in expected.items():
            if type(e.get("n")) is not int or e["n"] < 0 or not isinstance(e.get("origins_sha256"), str):
                raise RollupError("EXPECTED_INVALID", fold)
        self.candidate_identity_sha256 = candidate_identity_sha256
        self.identity = {"schema_version": RECEIPT_SCHEMA, "experiment": experiment,
                         "candidate_identity_sha256": candidate_identity_sha256,
                         "horizon_hours": int(horizon_hours),
                         "head_offset_hours": int(horizon_hours if head_offset_hours is None else head_offset_hours),
                         "role": role, "population": population, "split": split, "metric": metric,
                         "unit": unit, "scale": scale}
        self.expected = {k: {"n": v["n"], "origins_sha256": v["origins_sha256"]}
                         for k, v in sorted(expected.items())}
        self.rows: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------- input
    def add(self, row: Mapping[str, Any]) -> str:
        """Add one weekly row: ``ADDED`` or ``DUPLICATE``; conflicts are refused."""
        for k in IDENTITY:
            if row.get(k) != self.identity[k]:
                raise RollupError("IDENTITY_MISMATCH", f"{k}={row.get(k)!r}")
        fold = row.get("fold")
        if fold not in self.expected:
            raise RollupError("FOLD_NOT_IN_PLAN", str(fold))
        exp = self.expected[fold]
        if type(row.get("n")) is not int or row["n"] != exp["n"]:
            raise RollupError("SUPPORT_COUNT_MISMATCH", f"{fold}: {row.get('n')} != {exp['n']}")
        if row.get("support_sha256") != exp["origins_sha256"]:
            raise RollupError("SUPPORT_DIGEST_MISMATCH", fold)
        if exp["n"] == 0:
            if row.get("status") != "NO_SPLIT_SUPPORT" or any(row.get(k) is not None for k in SUMS):
                raise RollupError("RECEIPT_INVALID", f"{fold}: zero support must be NO_SPLIT_SUPPORT")
            clean = {"n": 0, **{k: None for k in SUMS}, "status": "NO_SPLIT_SUPPORT"}
        else:
            if row.get("status") != "SCORED":
                raise RollupError("RECEIPT_NOT_SCORED", f"{fold}: {row.get('status')}")
            clean = {"n": row["n"], **{k: _nonneg(k, row[k]) for k in SUMS}, "status": "SCORED"}
        clean["c2_receipt_sha256"] = row.get("c2_receipt_sha256")
        if fold in self.rows:
            if self.rows[fold] != clean:
                raise RollupError("CONFLICTING_DUPLICATE", fold)
            return "DUPLICATE"
        self.rows[fold] = clean
        return "ADDED"

    # ---------------------------------------------------------------- state
    def to_json(self) -> Dict[str, Any]:
        body = {"schema": STATE_SCHEMA, "identity": self.identity, "expected": self.expected,
                "rows": dict(sorted(self.rows.items()))}
        return dict(body, state_sha256=_digest(body))

    @classmethod
    def from_json(cls, state: Mapping[str, Any]) -> "AnnualAccumulator":
        body = {k: v for k, v in state.items() if k != "state_sha256"}
        if state.get("schema") != STATE_SCHEMA or _digest(body) != state.get("state_sha256"):
            raise RollupError("STATE_DAMAGED")
        i = state["identity"]
        acc = cls(experiment=i["experiment"], candidate_identity_sha256=i["candidate_identity_sha256"],
                  horizon_hours=i["horizon_hours"], role=i["role"], expected=state["expected"],
                  population=i["population"], split=i["split"], head_offset_hours=i["head_offset_hours"],
                  metric=i["metric"], unit=i["unit"], scale=i["scale"])
        if acc.identity != i:
            raise RollupError("STATE_DAMAGED", "identity")
        for fold, r in state["rows"].items():
            acc.add(dict(i, fold=fold, support_sha256=acc.expected.get(fold, {}).get("origins_sha256"), **r))
        return acc

    # ---------------------------------------------------------------- output
    def summary(self) -> Dict[str, Any]:
        supported = [f for f, e in self.expected.items() if e["n"] > 0]
        missing = [f for f in supported if f not in self.rows]
        scored = [self.rows[f] for f in supported if f in self.rows]
        n = sum(r["n"] for r in scored)
        sums = {k: math.fsum(r[k] for r in scored) for k in SUMS}
        complete = not missing and n > 0
        mae = sums["sum_abs_error"] / n if complete else None
        naive = sums["naive_sum_abs_error"] / n if complete else None
        mse = sums["sum_sq_error"] / n if complete else None
        naive_mse = sums["naive_sum_sq_error"] / n if complete else None
        skill, reason = _skill(mae, naive)
        skill_mse, reason_mse = _skill(mse, naive_mse)
        weekly = [r["sum_abs_error"] / r["n"] for r in scored]
        body = {"schema": SUMMARY_SCHEMA, "scope": ANNUAL_SCOPE, **self.identity,
                "fold": "ANNUAL", "state": "ANNUAL_COMPLETE" if complete else "INCOMPLETE",
                "weeks_expected": len(supported), "weeks_present": len(scored),
                "weeks_no_support": [f for f, e in self.expected.items() if e["n"] == 0],
                "missing_folds": missing, "n": n, **sums,
                "mae": mae, "naive_mae": naive, "skill_mae": skill, "skill_mae_reason": reason,
                "mse": mse, "naive_mse": naive_mse, "skill_mse": skill_mse, "skill_mse_reason": reason_mse,
                "weekly_mean_mae_diagnostic": (math.fsum(weekly) / len(weekly)) if complete and weekly else None,
                "support_sha256": _digest(self.expected),
                "aggregation": "sum(error)/sum(n) over frozen weekly support; naive on identical rows"}
        return dict(body, sha256=_digest(body))


def compare_annual(a: AnnualAccumulator, b: AnnualAccumulator) -> Dict[str, Any]:
    """Pair two arms on the identical frozen weekly support and identical naive."""
    ia = {k: v for k, v in a.identity.items() if k != "candidate_identity_sha256"}
    ib = {k: v for k, v in b.identity.items() if k != "candidate_identity_sha256"}
    if ia != ib or a.expected != b.expected:
        raise RollupError("NOT_COMPARABLE_IDENTITY", "different head/split/support")
    sa, sb = a.summary(), b.summary()
    base = {"schema": "i6n.weekly_rollup.comparison.v1", "scope": ANNUAL_SCOPE,
            "horizon_hours": a.identity["horizon_hours"], "arm_a": a.candidate_identity_sha256,
            "arm_b": b.candidate_identity_sha256, "support_sha256": sa["support_sha256"]}
    if sa["state"] != "ANNUAL_COMPLETE" or sb["state"] != "ANNUAL_COMPLETE":
        return dict(base, state="NOT_COMPARABLE_INCOMPLETE",
                    missing_a=sa["missing_folds"], missing_b=sb["missing_folds"])
    for fold, ra in a.rows.items():
        rb = b.rows[fold]
        if (ra["n"], ra["naive_sum_abs_error"], ra["naive_sum_sq_error"]) != \
                (rb["n"], rb["naive_sum_abs_error"], rb["naive_sum_sq_error"]):
            raise RollupError("NAIVE_MISMATCH", fold)
    return dict(base, state="COMPARABLE", n=sa["n"], naive_mae=sa["naive_mae"],
                mae_a=sa["mae"], mae_b=sb["mae"], delta_mae=sb["mae"] - sa["mae"],
                skill_mae_a=sa["skill_mae"], skill_mae_b=sb["skill_mae"])


def fold_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Wrap one inner-fold (search) receipt row with its non-annual scope."""
    n = row.get("n")
    mae = row["sum_abs_error"] / n if n else None
    naive = row["naive_sum_abs_error"] / n if n else None
    return {"scope": INNER_SCOPE, **{k: row.get(k) for k in IDENTITY}, "fold": row.get("fold"),
            "n": n, "mae": mae, "naive_mae": naive, "support_sha256": row.get("support_sha256")}


def assert_comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Refuse cross-scope or cross-horizon comparisons (e.g. inner fold vs annual h12)."""
    if a.get("scope") != b.get("scope"):
        raise RollupError("NOT_COMPARABLE_SCOPE", f"{a.get('scope')} vs {b.get('scope')}")
    if a.get("horizon_hours") != b.get("horizon_hours") or a.get("head_offset_hours") != b.get("head_offset_hours"):
        raise RollupError("NOT_COMPARABLE_HORIZON", f"{a.get('horizon_hours')} vs {b.get('horizon_hours')}")
    for k in ("split", "role", "population", "fold", "support_sha256"):
        if a.get(k) != b.get(k):
            raise RollupError("NOT_COMPARABLE_SUPPORT", k)
