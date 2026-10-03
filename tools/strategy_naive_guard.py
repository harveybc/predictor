#!/usr/bin/env python3
"""Same-row-naive guard in front of every heuristic-strategy invocation.

Master plan v3, rule 2 and section 11: a forecast that does not STRICTLY beat
the naive of the same rows, scale, metric and horizon never enters the
heuristic strategy and never produces trading artifacts.

``admit_horizons`` decides per horizon. A horizon is admitted only when:

- model and naive errors are finite and non-negative, and the naive is > 0
  (skill is undefined on a zero naive -> refused, not admitted);
- the model and the naive were scored on the same rows (``rows_sha256`` equal
  to ``naive_rows_sha256``), the same scale and the same metric;
- skill = 1 - model_error / naive_error is strictly > 0.

``guarded_strategy_call`` performs ZERO invocations of the strategy callable
when no horizon is admitted, and otherwise passes only the admitted rows.

Wiring status (recorded 2026-10-03): predictor has no live strategy export at
this base; the only in-repo invocation is the dead module
``app/data_processor copy.py`` (``strategy_plugin.evaluate_candidate``). The
consumer that must call this guard before ``evaluate_candidate`` is
heuristic-strategy ``app/data_processor.py`` where predictions are loaded
(``hourly_predictions_file`` / ``daily_predictions_file``), and any predictor
exporter that writes prediction files for it. See
docs/audits/evidence/selected_manifest_gate_20261003/WIRING.json.
"""
from __future__ import annotations

import math

GUARD = "strategy_naive_guard.v1"


def skill(model_error, naive_error):
    """1 - model/naive, or None when undefined (zero / non-finite naive)."""
    try:
        m, n = float(model_error), float(naive_error)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(m) and math.isfinite(n)) or m < 0 or n <= 0:
        return None
    return 1.0 - m / n


def admit_horizons(rows) -> dict:
    admitted, refused = [], {}
    for r in rows or []:
        h = r.get("horizon")
        key = f"{r.get('family', '')}:{h}"
        if r.get("rows_sha256") is None or r.get("rows_sha256") != r.get("naive_rows_sha256"):
            refused[key] = "NOT_SAME_ROWS"
            continue
        if r.get("scale") is None or r.get("scale") != r.get("naive_scale"):
            refused[key] = "NOT_SAME_SCALE"
            continue
        if r.get("metric") is None or r.get("metric") != r.get("naive_metric", r.get("metric")):
            refused[key] = "NOT_SAME_METRIC"
            continue
        s = skill(r.get("model_error"), r.get("naive_error"))
        if s is None:
            refused[key] = "SKILL_UNDEFINED"
        elif s <= 0:
            refused[key] = f"SKILL_NOT_POSITIVE({s:.6g})"
        else:
            admitted.append({**r, "skill": s})
    return {"guard": GUARD, "admitted": admitted, "refused": refused,
            "strategy_invocations_allowed": bool(admitted)}


def guarded_strategy_call(rows, strategy_callable, *args, **kwargs):
    """Call ``strategy_callable(admitted_rows, *args, **kwargs)`` at most once, and never
    when no horizon beats its same-row naive. Returns (result_or_None, report)."""
    report = admit_horizons(rows)
    report["strategy_invocations"] = 0
    if not report["strategy_invocations_allowed"]:
        return None, report
    result = strategy_callable(report["admitted"], *args, **kwargs)
    report["strategy_invocations"] = 1
    return result, report
