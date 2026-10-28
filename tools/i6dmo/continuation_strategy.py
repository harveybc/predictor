"""Retrospective 2024 strategy close hook. No fitting, TEST access or new policy.

``selection`` maps short/long to unit_ids, pooled annual metrics, cadence and
optional horizons (column order). Prediction values must be original prices.
``spec.strategy_inputs`` supplies bars/calendar pins with absolute path/sha256
or an explicit root plus relpath/sha256. Inputs are prepared elsewhere.
"""
import math
from pathlib import Path

import numpy as np

from tools.i6dmo import strategy_adapter as SA


HEADS = {"short": (2, 3, 4, 5, 6, 12), "long": (18, 24, 36, 48, 60, 72)}
START, END = 1704067200, 1735689600
LABEL = "RETROSPECTIVE_DEVELOPMENT_NOT_OOS"


class _Stop(ValueError):
    def __init__(self, code, missing=None):
        self.code, self.missing = code, missing


def _require(condition, code):
    if not condition:
        raise _Stop(code)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _eligibility(selection):
    eligibility = {}
    for group, heads in HEADS.items():
        chosen = selection.get(group) or {}
        rows = chosen.get("metrics", [])
        _require(isinstance(rows, list), "ANNUAL_METRICS_INVALID")
        by_head = {}
        for row in rows:
            _require(isinstance(row, dict), "ANNUAL_METRICS_INVALID")
            h = row.get("horizon")
            _require(type(h) is int and h in heads and h not in by_head, "ANNUAL_HEAD_INVALID")
            by_head[h] = row
        result = {}
        for h in heads:
            row = by_head.get(h, {})
            n, mae, naive, relative = (row.get(k) for k in ("n", "mae", "naive_mae", "relative_error"))
            complete = (chosen.get("state", "COMPLETE") == "COMPLETE" and
                row.get("state", "COMPLETE") == "COMPLETE" and type(n) is int and n > 0 and
                row.get("n_declared", n) == n and row.get("n_failed", 0) == 0)
            finite = all(_number(v) for v in (mae, naive, relative))
            if complete and finite and mae >= 0 and naive > 0:
                _require(_close(relative, (mae - naive) / naive), "ANNUAL_RELATIVE_ERROR_MISMATCH")
                for key, value in (("sum_abs_error", mae * n), ("sum_abs_naive_error", naive * n),
                                   ("naive_sum_abs_error", naive * n)):
                    if key in row:
                        _require(_number(row[key]) and _close(row[key], value), "ANNUAL_SUM_MISMATCH")
                digest = row.get("pairs_sha256")
                _require(isinstance(digest, str) and len(digest) == 64 and
                         all(c in "0123456789abcdef" for c in digest), "PAIR_IDENTITY_MISSING")
                reason = "BEATS_OWN_NAIVE" if mae < naive else "NOT_BELOW_OWN_NAIVE"
            else:
                reason = "INCOMPLETE_ANNUAL_HEAD"
            result[str(h)] = dict(eligible=reason == "BEATS_OWN_NAIVE", reason=reason,
                                  n=n if type(n) is int else None,
                                  mae=mae if _number(mae) else None,
                                  naive_mae=naive if _number(naive) else None,
                                  relative_error=relative if _number(relative) else None,
                                  pairs_sha256=row.get("pairs_sha256"))
        eligibility[group] = result
    return eligibility


def _validation_arrays(root, group, chosen):
    heads = HEADS[group]
    columns = chosen.get("horizons", list(heads))
    _require(isinstance(columns, (list, tuple)) and len(columns) == len(heads) and
             all(type(h) is int for h in columns) and set(columns) == set(heads), "HORIZON_SUPPORT_MISMATCH")
    order = [columns.index(h) for h in heads]
    units = chosen.get("unit_ids")
    _require(isinstance(units, list) and units and len(set(units)) == len(units), "SELECTED_UNITS_INVALID")
    pieces, identities = [], {}
    for unit in units:
        _require(isinstance(unit, str) and unit not in (".", "..") and
                 unit and all(c.isalnum() or c in "._-" for c in unit), "SELECTED_UNITS_INVALID")
        rel = f"results/{unit}/VALIDATION.npz"
        path = root / rel
        _require(path.resolve().is_relative_to(root.resolve()), "ARTIFACT_PATH_INVALID")
        if not path.is_file():
            raise _Stop("VALIDATION_ARTIFACT_MISSING", [rel])
        identities[unit] = SA._file_sha(path)
        with np.load(path, allow_pickle=False) as z:
            _require(set(("prediction", "origin_ts", "origin_price", "target_price", "mask")) <= set(z.files),
                     "VALIDATION_FIELDS_MISSING")
            ts, origin, pred, target, mask = (z[k] for k in
                ("origin_ts", "origin_price", "prediction", "target_price", "mask"))
            _require(ts.ndim == 1 and len(ts) > 0 and np.issubdtype(ts.dtype, np.integer), "ORIGIN_INVALID")
            _require(np.all((ts >= START) & (ts < END)), "TEST_FORBIDDEN")
            _require(np.all(np.diff(ts) > 0), "DUPLICATE_OR_UNORDERED_ORIGIN")
            shape = (len(ts), len(heads))
            _require(origin.shape == (len(ts),) and pred.shape == target.shape == mask.shape == shape and
                     mask.dtype == np.bool_, "VALIDATION_SHAPE_INVALID")
            if "horizons" in z.files or "horizons_hours" in z.files:
                key = "horizons" if "horizons" in z.files else "horizons_hours"
                _require(z[key].tolist() == list(columns), "HORIZON_SUPPORT_MISMATCH")
            _require(np.all(np.isfinite(origin) & (origin > 0)) and
                     np.all(np.isfinite(pred[mask]) & (pred[mask] > 0)) and
                     np.all(np.isfinite(target[mask]) & (target[mask] > 0)), "VALIDATION_PRICE_INVALID")
            _require(np.all((ts[:, None] + np.asarray(columns)[None, :] * 3600)[mask] < END), "TEST_FORBIDDEN")
            pieces.append((ts, origin, pred[:, order], target[:, order], mask[:, order]))
    merged = tuple(np.concatenate([piece[j] for piece in pieces]) for j in range(5))
    ts = merged[0]
    _require(len(np.unique(ts)) == len(ts), "DUPLICATE_ORIGIN")
    order = np.argsort(ts)
    return tuple(values[order] for values in merged), identities


def _replay(group, arrays, eligibility):
    ts, origin, pred, target, mask = arrays
    for j, h in enumerate(HEADS[group]):
        valid = mask[:, j]
        n = int(valid.sum())
        row = eligibility[group][str(h)]
        _require(n == row["n"], "METRICS_REPLAY_MISMATCH")
        _require(SA._canonical_sha(ts[valid].tolist()) == row["pairs_sha256"], "PAIRS_REPLAY_MISMATCH")
        mae = math.fsum(np.abs(pred[valid, j] - target[valid, j]).tolist()) / n
        naive = math.fsum(np.abs(origin[valid] - target[valid, j]).tolist()) / n
        _require(_close(mae, row["mae"]) and _close(naive, row["naive_mae"]) and naive > 0 and
                 _close((mae - naive) / naive, row["relative_error"]), "METRICS_REPLAY_MISMATCH")


def _coherent_prices(arrays):
    short, long = arrays["short"], arrays["long"]
    _require(np.array_equal(short[0], long[0]), "PAIR_ORIGIN_MISMATCH")
    _require(np.allclose(short[1], long[1], rtol=0, atol=1e-12), "ORIGIN_PRICE_MISMATCH")
    prices = dict(zip(short[0].tolist(), short[1].tolist()))
    for group, (ts, _, _, target, mask) in arrays.items():
        for j, h in enumerate(HEADS[group]):
            for t, price in zip(ts[mask[:, j]] + h * 3600, target[mask[:, j], j]):
                key = int(t)
                _require(key not in prices or math.isclose(prices[key], float(price), rel_tol=0, abs_tol=1e-12),
                         "TARGET_PRICE_MISMATCH")
                prices[key] = float(price)


def _input(spec, key):
    inputs = spec.get("strategy_inputs") or {}
    pin, root = inputs.get(key), inputs.get("root")
    # Also accept the adapter's established explicitly rooted pin layout.
    if not inputs:
        pin = ((spec.get("effective_config") or {}).get("data_pins") or {}).get("strategy_" + key)
        root = spec.get("data_root")
    if not isinstance(pin, dict) or not (pin.get("path") or (root and pin.get("relpath"))):
        raise _Stop("STRATEGY_INPUT_MISSING", ["strategy_inputs." + key])
    if pin.get("path"):
        path = Path(pin["path"])
        _require(path.is_absolute(), "INPUT_PATH_INVALID")
        if root:
            _require(path.resolve().is_relative_to(Path(root).resolve()), "INPUT_PATH_INVALID")
        name = path.name
    else:
        relative = Path(pin["relpath"])
        path, root = Path(root) / relative, Path(root).resolve()
        _require(not relative.is_absolute() and path.resolve().is_relative_to(root), "INPUT_PATH_INVALID")
        name = relative.as_posix()
    _require(not any(part.casefold().split(".")[0] == "test" for part in path.parts), "INPUT_PATH_INVALID")
    if not path.is_file():
        raise _Stop("STRATEGY_INPUT_MISSING", ["strategy_inputs." + key + ":" + name])
    _require(SA._file_sha(path) == pin.get("sha256"), "STRATEGY_INPUT_DIGEST_MISMATCH")
    return path


def _numeric_metrics(values, prefix=""):
    result = {}
    for key, value in values.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(_numeric_metrics(value, name + "_"))
        elif _number(value):
            result[name] = value
    return result


def finalize(spec, selection, artifact_root):
    """Return annual eligibility, then (only if eligible) the unchanged adapter result.

    Missing real bytes/pins return WAIT_ARTIFACT, corrupt or incoherent evidence
    returns FAILED. No result here is held-out/OOS evidence. No input is written.
    """
    result = dict(schema="i6dmo.continuation_strategy.v1", state="COMPLETE", label=LABEL, test_read=False,
                  window=dict(start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z", split="validation"),
                  initial_cash=10000., per_head_eligibility={}, metrics={})
    try:
        _require(isinstance(spec, dict) and isinstance(selection, dict), "SELECTION_INVALID")
        eligibility = _eligibility(selection)
        result["per_head_eligibility"] = eligibility
        eligible = all(row["eligible"] for rows in eligibility.values() for row in rows.values())
        result["metrics"] = dict(pair_eligible=int(eligible), strategy_evaluated=0,
            **{group + "_eligible_heads": sum(row["eligible"] for row in rows.values())
               for group, rows in eligibility.items()})
        if not eligible:
            return dict(result, status="COMPLETE", disposition="PAIR_NOT_ELIGIBLE")
        arrays, identities = {}, {}
        for group in HEADS:
            arrays[group], identities[group] = _validation_arrays(Path(artifact_root), group, selection[group])
        _coherent_prices(arrays)
        for group in HEADS:
            _replay(group, arrays[group], eligibility)
        preds = {group: SA.family_predictions_from_arrays(row_id=values[0], horizons_hours=HEADS[group],
            predicted_price=values[2], family=group) for group, values in arrays.items()}
        bars_path, calendar_path = _input(spec, "bars"), _input(spec, "calendar")
        import json
        calendar_document = json.loads(calendar_path.read_text())
        _, weeks = SA.c1.load_calendar(calendar_document)
        _require(all(any(week.market_open < t <= week.market_close for week in weeks)
                     for t in preds["short"]), "CALENDAR_SUPPORT_MISMATCH")
        # Reject mixed-year files instead of silently selecting TEST out of them.
        with np.load(bars_path, allow_pickle=False) as z:
            _require(np.all((z["close_time"] >= START) & (z["close_time"] < END)), "TEST_FORBIDDEN")
        window = (SA.parse_time(result["window"]["start"]), SA.parse_time(result["window"]["end"]))
        bars = SA._bars(bars_path, window, allow_test=False)
        by_time = {int(bar.close_time.timestamp()): bar.close for bar in bars}
        for t, price in zip(arrays["short"][0], arrays["short"][1]):
            _require(int(t) in by_time and _close(by_time[int(t)], float(price)), "BAR_ORIGIN_PRICE_MISMATCH")
        for group, (ts, _, _, target, mask) in arrays.items():
            for j, h in enumerate(HEADS[group]):
                for t, price in zip(ts[mask[:, j]] + h * 3600, target[mask[:, j], j]):
                    _require(int(t) not in by_time or _close(by_time[int(t)], float(price)), "BAR_TARGET_PRICE_MISMATCH")
        policy = SA.load_pinned_policy()
        receipt = SA.derive_execution_policy_from_file(policy=policy)
        execution = SA.execution_from_policy(receipt)
        _require(execution["execution"].initial_equity == 10000., "INITIAL_CASH_MISMATCH")
        backtest = SA.run_backtest(bars=bars, weeks=weeks, short_predictions=preds["short"],
            long_predictions=preds["long"], policy=policy, entry_params=execution["entry"],
            execution=execution["execution"], validation_eligible=HEADS, allow_test=False)
        return dict(result, status="COMPLETE", disposition="PAIR_ELIGIBLE", backtest=backtest,
            strategy_status="EXECUTION_FAILURES" if backtest["failures"] else "COMPLETE",
            metrics={**result["metrics"], "strategy_evaluated": 1, **_numeric_metrics(backtest["metrics"]),
                     **_numeric_metrics(backtest["counters"], "count_")},
            cost_verification=execution["cost_verification"],
            costs_label="ASSUMED_NOT_BROKER_VERIFIED", costs_sha256=execution["costs_sha256"],
            calendar_verification="ASSUMED_NOT_BROKER_VERIFIED", calendar=calendar_document,
            inputs_sha256=dict(bars=SA._file_sha(bars_path), calendar=SA._file_sha(calendar_path)),
            prediction_artifacts_sha256=identities,
            selected={g: dict(unit_ids=selection[g]["unit_ids"], cadence=selection[g].get("cadence", "static"))
                      for g in HEADS})
    except _Stop as exc:
        if exc.missing is not None:
            return dict(result, state="WAIT_ARTIFACT", status="WAIT_ARTIFACT", disposition="WAIT_ARTIFACT",
                        code=exc.code, missing=exc.missing)
        return dict(result, status="FAILED", disposition="TECHNICAL_ERROR", code=exc.code)
    except (SA.StrategyAdapterError, SA._Refusal, SA.c1.EligibilityError) as exc:
        return dict(result, status="FAILED", disposition="TECHNICAL_ERROR", code=str(exc.code))
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return dict(result, status="FAILED", disposition="TECHNICAL_ERROR", code="STRATEGY_ARTIFACT_INVALID")
