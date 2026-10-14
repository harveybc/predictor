"""I6-DMO paired metrics: C2 scoring, independent naive replay and weekly diagnostics.

* ``naive_replay`` re-reads the retained ``INPUT.npz`` from disk and recomputes
  ``|p(t+h) - p(t)|`` per head from the ROW IDS (not from the materialized
  arrays), looking up both prices by exact UTC bar-end timestamp. A missing
  bar is a ``NaiveReplayMismatch``, never the next row.
* ``paired_metrics`` scores the model and the naive on exactly the same rows
  and physical horizon with the audited C2 component
  (``tools.i6n_contracts.paired_metrics``), then cross-checks C2 against a
  manual recomputation and the independent replay. Disagreement raises.
* ``weekly_diagnostics`` aggregates by sums and counts. The annual pooled MAE is
  ``sum |e| / n``; the unweighted mean of weekly relative errors is labelled a
  diagnostic. Absent weeks and heads are explicit.

Relative error is ``(MAE_h - naive_h) / naive_h``; negative is better.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from tools.i6dmo import provenance
from tools.i6n_contracts import paired_metrics as c2

UTC = timezone.utc
HOUR = 3600
ROLE = "MANAGEMENT"
UNIT = "USD_per_EUR"
SCALE = "original_price"
REL_TOL = 1e-12


class NaiveReplayMismatch(ValueError):
    """The independent replay disagrees with the scored rows."""


def _origin_from_row_id(rid: str) -> int:
    asset, text = rid.split("|", 1)
    if asset != "EURUSD":
        raise NaiveReplayMismatch(f"unexpected asset in row id {rid!r}")
    return int(datetime.fromisoformat(text).timestamp())


def naive_replay(input_dir: str | Path, arrays: Any, horizons: Sequence[int],
                 mask: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """Recompute the naive per head from disk on the rows selected by ``mask``."""
    with np.load(Path(input_dir) / "INPUT.npz", allow_pickle=False) as z:
        ts, price = z["ts"], z["price"]
    lookup = dict(zip(ts.tolist(), price.tolist()))
    mask = arrays.mask if mask is None else mask
    out = {}
    for j, h in enumerate(horizons):
        rows = [str(r) for r in np.asarray(arrays.row_ids)[mask[:, j]]]
        errors = []
        for rid in rows:
            t = _origin_from_row_id(rid)
            if t not in lookup or t + h * HOUR not in lookup:
                raise NaiveReplayMismatch(f"{rid} h{h}: origin or exact target bar absent")
            errors.append(abs(lookup[t + h * HOUR] - lookup[t]))
        total = math.fsum(errors)
        out[str(h)] = {"n": len(rows), "sum_abs": total,
                       "mae": total / len(rows) if rows else None,
                       "row_ids_sha256": provenance.canonical_sha256(rows)}
    return out


def _close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= REL_TOL * max(1.0, abs(a), abs(b))


def build_c2_document(arrays: Any, predictions: np.ndarray, design: Mapping[str, Any], *,
                      checkpoint: str, sealed_at: str,
                      upstream_exclusions: Sequence[Mapping[str, Any]] = ()) -> dict:
    """C2 input for one split: S support = every (row, head) with a valid label."""
    hs = list(arrays.horizons)
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.shape != arrays.mask.shape:
        raise ValueError(f"predictions shape {pred.shape} != mask {arrays.mask.shape}")
    contract = {"unit": UNIT, "scale": SCALE, "roles": [ROLE], "horizons_hours": hs,
                "protocol_id": design["experiment_id"], "calendar_id": arrays.calendar_id,
                "config_id": provenance.canonical_sha256(dict(design))}
    forecasts, failures, keys = [], [], []
    for r in range(len(arrays.row_ids)):
        t = int(arrays.origin_ts[r])
        origin = datetime.fromtimestamp(t, UTC).isoformat()
        for j, h in enumerate(hs):
            if not arrays.mask[r, j]:
                continue
            keys.append({"asset": "EURUSD", "origin": origin, "horizon_hours": h, "role": ROLE})
            if not math.isfinite(pred[r, j]):
                failures.append({"asset": "EURUSD", "origin": origin, "horizon_hours": h,
                                 "reason": "PREDICTION_NOT_FINITE"})
                continue
            forecasts.append({
                "asset": "EURUSD", "origin": origin, "horizon_hours": h,
                "target_timestamp": datetime.fromtimestamp(t + h * HOUR, UTC).isoformat(),
                "role": ROLE, "price_origin": float(arrays.price_origin[r]),
                "price_target": float(arrays.y_price[r, j]), "prediction": float(pred[r, j]),
                "checkpoint": checkpoint, "unit": UNIT, "scale": SCALE})
    population = c2.seal_population({
        "population_id": f"{design['experiment_id']}|{arrays.split}|S", "sealed_at": sealed_at,
        "keys": keys, "exclusions": [dict(e) for e in upstream_exclusions]})
    return {"schema_version": c2.INPUT_SCHEMA_VERSION, "contract": contract,
            "forecasts": forecasts, "forecast_failures": failures,
            "opportunity_population": population, "usage_receipts": []}


def paired_metrics(arrays: Any, predictions: np.ndarray, design: Mapping[str, Any], *,
                   input_dir: str | Path, checkpoint: str, sealed_at: str,
                   upstream_exclusions: Sequence[Mapping[str, Any]] = ()) -> dict:
    """Score with C2, cross-check manually and against the disk replay; per-head report."""
    pred = np.asarray(predictions, dtype=np.float64)
    result = c2.score(build_c2_document(arrays, pred, design, checkpoint=checkpoint,
                                        sealed_at=sealed_at,
                                        upstream_exclusions=upstream_exclusions))
    scored = arrays.mask & np.isfinite(pred)
    replay = naive_replay(input_dir, arrays, arrays.horizons, mask=scored)
    groups = {g["horizon_hours"]: g for g in result["groups"]
              if g["population_kind"] == c2.OPPORTUNITY and g["role"] == ROLE}
    per_h, problems = {}, []
    model_abs = np.where(scored, np.abs(pred - arrays.y_price), np.nan)
    naive_abs = np.where(scored, np.abs(arrays.y_price - arrays.price_origin[:, None]), np.nan)
    for j, h in enumerate(arrays.horizons):
        g = groups[h]
        m = scored[:, j]
        n = int(m.sum())
        manual = math.fsum(model_abs[m, j].tolist()) / n if n else None
        manual_naive = math.fsum(naive_abs[m, j].tolist()) / n if n else None
        rows = [str(r) for r in arrays.row_ids[m]]
        if g["n"] != n or replay[str(h)]["n"] != n:
            problems.append(f"h{h}: n C2={g['n']} manual={n} replay={replay[str(h)]['n']}")
        if not _close(g["mae"], manual):
            problems.append(f"h{h}: C2 mae {g['mae']} != manual {manual}")
        if not (_close(g["naive_mae"], manual_naive) and _close(g["naive_mae"], replay[str(h)]["mae"])):
            problems.append(f"h{h}: naive C2 {g['naive_mae']} manual {manual_naive} "
                            f"replay {replay[str(h)]['mae']}")
        if provenance.canonical_sha256(rows) != replay[str(h)]["row_ids_sha256"]:
            problems.append(f"h{h}: row ids differ between model and replay")
        per_h[str(h)] = {
            "n": g["n"], "n_declared": g["n_declared"], "n_failed": sum(g["failures"].values()),
            "state": g["state"], "mae": g["mae"], "naive_mae": g["naive_mae"],
            "relative_error": g["skill_mae"],
            "relative_error_undefined_reason": g["skill_mae_undefined_reason"],
            "sum_abs_error": g["sum_abs_error"], "naive_sum_abs_error": g["naive_sum_abs_error"],
            "row_ids_sha256": provenance.canonical_sha256(rows),
            "c2_row_keys_sha256": g["row_keys_sha256"],
            "eligible_mae_below_naive": (None if g["mae"] is None or g["naive_mae"] is None
                                         else g["mae"] < g["naive_mae"]),
        }
    if problems:
        raise NaiveReplayMismatch("; ".join(problems))
    weekly = weekly_diagnostics(arrays.week_ids, model_abs, naive_abs, scored, arrays.horizons,
                                expected_weeks=list(arrays.expected_weeks))
    return {"schema": "i6dmo.paired_metrics.v1", "split": arrays.split, "unit": UNIT,
            "scale": SCALE, "relative_error_definition": "(MAE_h-naive_h)/naive_h; negative is better",
            "per_horizon": per_h, "weekly": weekly,
            "agreement": {"status": "AGREE", "checks": ["C2_vs_manual_model_mae",
                                                         "C2_vs_manual_naive", "C2_vs_disk_replay",
                                                         "row_ids_model_vs_replay"]},
            "c2": {"state": result["state"], "sha256": result["sha256"],
                   "contract_sha256": result["contract_sha256"],
                   "input_sha256": result["input_sha256"]}}


def weekly_diagnostics(week_ids: Sequence[str], abs_err: np.ndarray, naive_abs_err: np.ndarray,
                       mask: np.ndarray, horizons: Sequence[int], *,
                       expected_weeks: Sequence[str]) -> dict:
    """Per week x head sums/counts with explicit ABSENT weeks; annual = pooled sums/counts."""
    weeks = [str(w) for w in week_ids]
    unknown = sorted(set(weeks) - set(expected_weeks))
    if unknown:
        raise ValueError(f"rows in undeclared weeks {unknown[:3]}")
    week_arr = np.asarray(weeks, dtype=object)
    out = {}
    for j, h in enumerate(horizons):
        per_week, rels = {}, []
        for w in expected_weeks:
            m = (week_arr == w) & mask[:, j]
            n = int(m.sum())
            if not n:
                per_week[w] = {"status": "ABSENT", "n": 0}
                continue
            s = math.fsum(abs_err[m, j].tolist())
            ns = math.fsum(naive_abs_err[m, j].tolist())
            rel = (s - ns) / ns if ns else None
            per_week[w] = {"status": "OBSERVED", "n": n, "sum_abs_error": s,
                           "naive_sum_abs_error": ns, "mae": s / n, "naive_mae": ns / n,
                           "relative_error": rel}
            if rel is not None:
                rels.append(rel)
        m = mask[:, j]
        n = int(m.sum())
        if not n:
            annual = {"status": "NO_ROWS", "n": 0}
        else:
            s = math.fsum(abs_err[m, j].tolist())
            ns = math.fsum(naive_abs_err[m, j].tolist())
            annual = {"status": "OBSERVED", "n": n, "sum_abs_error": s, "naive_sum_abs_error": ns,
                      "mae": s / n, "naive_mae": ns / n,
                      "relative_error": (s - ns) / ns if ns else None}
        out[str(h)] = {
            "annual": annual, "weeks": per_week,
            "weeks_expected": len(expected_weeks),
            "weeks_absent": sum(1 for v in per_week.values() if v["status"] == "ABSENT"),
            "unweighted_weekly_mean_relative_error": {
                "label": "DIAGNOSTIC_UNWEIGHTED",
                "value": (math.fsum(rels) / len(rels)) if rels else None,
                "weeks_counted": len(rels)}}
    return {"aggregation": "pooled sums and counts; weekly MAEs are never averaged into the annual",
            "per_horizon": out}
