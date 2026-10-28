"""I6-N lane C: annual roll-up of weekly-refit receipts (sums/counts, identical naive).

Consumes ``i6n.candidate_receipt.v1`` rows (lane A, INTERFACE_A) for ONE arm
(a search finalist refit every week, keyed by the receipt header's
``genotype_canonical_sha256`` or ``effective_config_sha256`` plus the arm role),
horizon, head, role, population and split, one canonical row per weekly unit
(``fold``), plus ``support_sha256`` = the unit's frozen ``origins_sha256`` from
the weekly campaign plan. Each week's A ``candidate_identity_sha256`` differs
(the cell names the week) and is kept per row (review E-C1). The annual metric is sum(error) / sum(n) with the naive pooled
over exactly the same rows; the mean of weekly MAEs is reported only as a
labelled diagnostic. A year with an unresolved missing week is INCOMPLETE and
has no annual score. Weeks the plan froze with zero support carry no score.

Scope guard: an INNER_FOLD (TRAIN/inner search fitness, e.g. 70 overlapping
inner rows of h72) is never comparable with an ANNUAL_WEEKLY_REFIT result, and
different horizons are never compared.

Pure standard library; no I/O (besides reading the sealed contamination label), no warehouse writes, no fits.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional

from tools import i6n_contamination as contamination

ANNUAL_SCOPE = "ANNUAL_WEEKLY_REFIT"
INNER_SCOPE = "INNER_FOLD"
STATE_SCHEMA = "i6n.weekly_rollup.state.v2"
SUMMARY_SCHEMA = "i6n.weekly_rollup.summary.v2"
RECEIPT_SCHEMA = "i6n.candidate_receipt.v1"
SUMS = ("sum_abs_error", "sum_sq_error", "naive_sum_abs_error", "naive_sum_sq_error")
#: A's row identity keys (INTERFACE_A); candidate_identity_sha256 differs every week and is kept per row.
IDENTITY = ("schema_version", "experiment", "candidate_identity_sha256", "horizon_hours",
            "head_offset_hours", "role", "population", "split", "metric", "unit", "scale")
ARM_IDENTITY = tuple(k for k in IDENTITY if k != "candidate_identity_sha256")
ARM_KEYS = ("genotype_canonical_sha256", "effective_config_sha256")
MAX_ATTEMPTS = 3


class RollupError(ValueError):
    """Typed refusal with a stable ``code``."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        super().__init__(f"{code}: {detail}" if detail else code)


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


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
    """Durable, order-independent, idempotent accumulator of one ARM's weekly rows.

    The arm is the finalist recipe refit every week: ``arm_sha256`` is its
    ``genotype_canonical_sha256`` (default) or ``effective_config_sha256``
    (``arm_key``) taken from each receipt HEADER, plus ``arm_role``. Each week's
    A ``candidate_identity_sha256`` differs (the cell names the week) and is
    stored per row. The canonical measurement of a week is its lowest attempt
    index; higher physical attempts are kept as history, never pooled.
    """

    def __init__(self, *, experiment: str, horizon_hours: int, role: str,
                 expected: Mapping[str, Mapping[str, Any]], arm_sha256: Optional[str] = None,
                 arm_role: Optional[str] = None, arm_key: str = ARM_KEYS[0],
                 candidate_identity_sha256: Optional[str] = None,
                 population: str = "S", split: str = "VALIDATION", head_offset_hours: Optional[int] = None,
                 metric: str = "ABS_AND_SQUARED_ERROR_SUMS", unit: str = "PRICE",
                 scale: str = "ORIGINAL_PRICE"):
        if candidate_identity_sha256 is not None:
            raise RollupError("LEGACY_IDENTITY_KEY", "C.3 keys the year by arm: pass arm_sha256 (receipt "
                              "genotype_canonical_sha256) and arm_role; weekly candidate identities differ")
        if not expected:
            raise RollupError("EXPECTED_EMPTY", "the plan's weekly support is required")
        for fold, e in expected.items():
            if type(e.get("n")) is not int or e["n"] < 0 or not isinstance(e.get("origins_sha256"), str):
                raise RollupError("EXPECTED_INVALID", fold)
        if not _hex64(arm_sha256) or arm_key not in ARM_KEYS or not isinstance(arm_role, str) or not arm_role:
            raise RollupError("ARM_INVALID", "arm_sha256 (sha256), arm_key and arm_role are required")
        self.arm = {"arm_key": arm_key, "arm_sha256": arm_sha256, "arm_role": arm_role}
        self.identity = {"schema_version": RECEIPT_SCHEMA, "experiment": experiment,
                         "horizon_hours": int(horizon_hours),
                         "head_offset_hours": int(horizon_hours if head_offset_hours is None else head_offset_hours),
                         "role": role, "population": population, "split": split, "metric": metric,
                         "unit": unit, "scale": scale}
        self.expected = {k: {"n": v["n"], "origins_sha256": v["origins_sha256"]}
                         for k, v in sorted(expected.items())}
        self.rows: Dict[str, Dict[int, Dict[str, Any]]] = {}

    @property
    def arm_sha256(self) -> str:
        return self.arm["arm_sha256"]

    # ---------------------------------------------------------------- input
    def add(self, row: Mapping[str, Any], *, arm_sha256: Optional[str] = None, attempt_index: int = 1) -> str:
        """Add one weekly row of this arm: ``ADDED`` or ``DUPLICATE``; conflicts are refused."""
        if arm_sha256 != self.arm["arm_sha256"]:
            raise RollupError("ARM_MISMATCH", f"{str(arm_sha256)[:12]} is not this accumulator's arm")
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int) or attempt_index < 1:
            raise RollupError("RECEIPT_INVALID", "attempt_index must be an integer >= 1")
        if attempt_index > MAX_ATTEMPTS:
            raise RollupError("ATTEMPTS_EXCEEDED", f"attempt {attempt_index} > {MAX_ATTEMPTS}")
        for k in ARM_IDENTITY:
            if row.get(k) != self.identity[k]:
                raise RollupError("IDENTITY_MISMATCH", f"{k}={row.get(k)!r}")
        if not _hex64(row.get("candidate_identity_sha256")):
            raise RollupError("RECEIPT_INVALID", "candidate_identity_sha256 must be a sha256")
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
        clean["candidate_identity_sha256"] = row["candidate_identity_sha256"]
        attempts = self.rows.setdefault(fold, {})
        if attempt_index in attempts:
            if attempts[attempt_index] != clean:
                raise RollupError("CONFLICTING_DUPLICATE", f"{fold} attempt {attempt_index}")
            return "DUPLICATE"
        attempts[attempt_index] = clean
        return "ADDED"

    def add_receipt(self, receipt: Mapping[str, Any]) -> list:
        """Add the matching row(s) of one A receipt; arm and attempt come from its header."""
        attempt = (receipt.get("attempt") or {}).get("index", 1)
        arm = receipt.get(self.arm["arm_key"])
        results = []
        for row in receipt.get("rows", []):
            if all(row.get(k) == self.identity[k] for k in ARM_IDENTITY):
                results.append(self.add(dict(row, fold=row.get("fold", receipt.get("fold"))),
                                        arm_sha256=arm, attempt_index=attempt))
        return results

    def _canonical(self) -> Dict[str, tuple]:
        return {fold: (min(a), a[min(a)]) for fold, a in self.rows.items() if a}

    # ---------------------------------------------------------------- state
    def to_json(self) -> Dict[str, Any]:
        body = {"schema": STATE_SCHEMA, "arm": self.arm, "identity": self.identity, "expected": self.expected,
                "rows": {f: {str(i): r for i, r in sorted(a.items())} for f, a in sorted(self.rows.items())}}
        return dict(body, state_sha256=_digest(body))

    @classmethod
    def from_json(cls, state: Mapping[str, Any]) -> "AnnualAccumulator":
        body = {k: v for k, v in state.items() if k != "state_sha256"}
        if state.get("schema") != STATE_SCHEMA or _digest(body) != state.get("state_sha256"):
            raise RollupError("STATE_DAMAGED")
        i, arm = state["identity"], state["arm"]
        acc = cls(experiment=i["experiment"], arm_sha256=arm["arm_sha256"], arm_role=arm["arm_role"],
                  arm_key=arm["arm_key"], horizon_hours=i["horizon_hours"], role=i["role"],
                  expected=state["expected"], population=i["population"], split=i["split"],
                  head_offset_hours=i["head_offset_hours"], metric=i["metric"], unit=i["unit"],
                  scale=i["scale"])
        if acc.identity != i:
            raise RollupError("STATE_DAMAGED", "identity")
        for fold, attempts in state["rows"].items():
            for index, r in attempts.items():
                acc.add(dict(i, fold=fold, support_sha256=acc.expected.get(fold, {}).get("origins_sha256"), **r),
                        arm_sha256=arm["arm_sha256"], attempt_index=int(index))
        return acc

    # ---------------------------------------------------------------- output
    def summary(self) -> Dict[str, Any]:
        canonical = self._canonical()
        supported = [f for f, e in self.expected.items() if e["n"] > 0]
        missing = [f for f in supported if f not in canonical]
        scored = [canonical[f][1] for f in supported if f in canonical]
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
        body = {"schema": SUMMARY_SCHEMA, "scope": ANNUAL_SCOPE, **self.arm, **self.identity,
                "fold": "ANNUAL", "state": "ANNUAL_COMPLETE" if complete else "INCOMPLETE",
                "weeks_expected": len(supported), "weeks_present": len(scored),
                "weeks_no_support": [f for f, e in self.expected.items() if e["n"] == 0],
                "missing_folds": missing,
                "canonical_measurements": {f: {"attempt_index": i, "candidate_identity_sha256":
                                               r["candidate_identity_sha256"]}
                                           for f, (i, r) in sorted(canonical.items())},
                "non_canonical_attempts": {f: sorted(a)[1:] for f, a in sorted(self.rows.items()) if len(a) > 1},
                "n": n, **sums,
                "mae": mae, "naive_mae": naive, "skill_mae": skill, "skill_mae_reason": reason,
                "mse": mse, "naive_mse": naive_mse, "skill_mse": skill_mse, "skill_mse_reason": reason_mse,
                "weekly_mean_mae_diagnostic": (math.fsum(weekly) / len(weekly)) if complete and weekly else None,
                "support_sha256": _digest(self.expected),
                "aggregation": "sum(error)/sum(n) over frozen weekly support; naive on identical rows; "
                               "lowest attempt per week is canonical"}
        return dict(body, sha256=_digest(body))


def compare_annual(a: AnnualAccumulator, b: AnnualAccumulator) -> Dict[str, Any]:
    """Pair two arms on the identical frozen weekly support and identical naive."""
    # Owner order 2026-10-12: point-in-time-contaminated identities are never valid skill.
    contamination.compare_skill(dict(a.identity), dict(b.identity))
    if a.identity != b.identity or a.expected != b.expected:
        raise RollupError("NOT_COMPARABLE_IDENTITY", "different head/split/support")
    sa, sb = a.summary(), b.summary()
    base = {"schema": "i6n.weekly_rollup.comparison.v1", "scope": ANNUAL_SCOPE,
            "horizon_hours": a.identity["horizon_hours"], "arm_a": dict(a.arm), "arm_b": dict(b.arm),
            "support_sha256": sa["support_sha256"]}
    if sa["state"] != "ANNUAL_COMPLETE" or sb["state"] != "ANNUAL_COMPLETE":
        return dict(base, state="NOT_COMPARABLE_INCOMPLETE",
                    missing_a=sa["missing_folds"], missing_b=sb["missing_folds"])
    ca, cb = a._canonical(), b._canonical()
    for fold, (_, ra) in ca.items():
        rb = cb[fold][1]
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
    contamination.compare_skill(a, b)
    if a.get("scope") != b.get("scope"):
        raise RollupError("NOT_COMPARABLE_SCOPE", f"{a.get('scope')} vs {b.get('scope')}")
    if a.get("horizon_hours") != b.get("horizon_hours") or a.get("head_offset_hours") != b.get("head_offset_hours"):
        raise RollupError("NOT_COMPARABLE_HORIZON", f"{a.get('horizon_hours')} vs {b.get('horizon_hours')}")
    for k in ("split", "role", "population", "fold", "support_sha256"):
        if a.get(k) != b.get(k):
            raise RollupError("NOT_COMPARABLE_SUPPORT", k)
