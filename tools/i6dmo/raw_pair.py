"""Finite post-long RAW pair evaluation. No fitting, services, polling or TEST.

The short pin is a retained report, NOT a fit under the current campaign.
Negative eligibility authenticates prediction *digest declarations* without
opening prediction files; eligible execution verifies their bytes and replay.
Independent actual-path review is required before deployment.
"""
import time
from pathlib import Path

from tools.i6dmo import campaign as C, matched_runner as MR
from tools.i6dmo import reporting as R, strategy_adapter as SA


class Refusal(Exception):
    def __init__(self, code, required=()):
        super().__init__(code)
        self.required = list(required)


def _require(ok, code):
    if not ok:
        raise Refusal(code)


def _json(path):
    import json
    return json.loads(Path(path).read_text())


def _artifact(folder, child, name):
    art = (child.get("artifacts") or {}).get(name)
    _require(isinstance(art, dict) and isinstance(art.get("path"), str),
             f"UPSTREAM_ARTIFACT_MISSING:{child.get('unit_id')}:{name}")
    base = (folder / "out").resolve()
    path = (base / art["path"]).resolve()
    _require(path.is_relative_to(base), "ARTIFACT_PATH_ESCAPE")
    return path, art


def _stored(unit, pin, expected_code=None):
    _require(isinstance(pin, dict) and Path(pin.get("path", "")).is_absolute(),
             f"ABSOLUTE_STORED_TERMINAL_REQUIRED:{unit}")
    _require(Path(pin["path"]).name == "TERMINAL.json", f"STORED_TERMINAL_REQUIRED:{unit}")
    folder, stored, child = SA._authenticated_stored(unit, pin)
    identity = stored.get("identity") or {}
    _require(C.unit_identity(identity) == folder.name and identity.get("unit") == unit,
             f"UPSTREAM_IDENTITY_COMPONENTS:{unit}")
    _require(child.get("status") == "COMPLETE", f"UPSTREAM_NOT_COMPLETE:{unit}")
    # Older reporting children omit the standard envelope; do not relabel them.
    for key, expected in (("unit_id", unit), ("identity_sha256", folder.name)):
        _require(key not in child or child[key] == expected, f"UPSTREAM_CHILD_IDENTITY:{unit}")
    _require(bool(identity.get("code_commit")) and
             (expected_code is None or identity["code_commit"] == expected_code),
             f"UPSTREAM_CODE_MISMATCH:{unit}")
    return folder, stored, child


def _report(unit, pin, arm, population=None, expected_code=None):
    folder, stored, child = _stored(unit, pin, expected_code)
    _artifact(folder, child, "metrics")  # constrain paths before the existing verifier reads
    metrics, record = MR._authenticated_report(unit, pin, population)
    block = metrics.get("matched") or {}
    heads = MR.HORIZONS if arm == "mg2m" else MR.LONG_HORIZONS
    per_h, eligibility = metrics.get("per_horizon") or {}, child.get("eligibility") or {}
    _require(stored["identity"].get("family") == ("short" if arm == "mg2m" else "long")
             and stored["identity"].get("kind") == "report", f"REPORT_IDENTITY_KIND:{unit}")
    _require(block.get("schema") == R.MATCHED_REPORT_SCHEMA and block.get("arm") == arm
             and block.get("population") == "COMMON_DEVELOPMENT", f"MATCHED_REPORT_SCHEMA:{unit}")
    lineage = child.get("lineage") or {}
    _require(lineage.get("arm") == arm and block.get("common_population_sha256")
             and lineage.get("common_population_sha256") == block["common_population_sha256"],
             f"REPORT_LINEAGE_MISMATCH:{unit}")
    _require(block.get("scientific_fit") is True and block.get("fit_device") == "gpu",
             f"NON_SCIENTIFIC_REPORT:{unit}")
    try:
        R.validate_matched_report(metrics, horizons=heads, arm=arm, child=child)
    except R.ReportFailure as exc:
        raise Refusal(f"{exc.code}:{unit}:{exc}") from None
    normalized = {h: dict(eligibility[h]) for h in per_h}
    name = next((n for n in SA.PREDICTION_ARTIFACT_NAMES if n in child["artifacts"]), None)
    _require(name is not None, f"PREDICTION_PIN_MISSING:{unit}")
    _, art = _artifact(folder, child, name)
    digest = art.get("sha256")
    _require(isinstance(digest, str) and len(digest) == 64 and
             all(c in "0123456789abcdef" for c in digest) and
             digest == metrics.get("predictions_sha256") == lineage.get("predictions_sha256"),
             f"PREDICTION_BINDING_MISMATCH:{unit}")
    record.update(child_terminal_sha256=stored["child_terminal_sha256"],
                  code_commit=stored["identity"]["code_commit"], identity=stored["identity"],
                  lineage=lineage, matched=block, predictions_sha256=digest,
                  prediction_bytes_verified=False)
    return dict(unit=unit, folder=folder, child=child, metrics=metrics, record=record, rows=normalized)


def _comparison(spec, reports, population):
    unit, pin = "comparison", spec["upstream"]["comparison"]
    folder, stored, child = _stored(unit, pin, spec["effective_config"]["code_commit"])
    _require(stored["identity"].get("kind") == "comparison" and
             sorted(stored["identity"].get("upstream") or []) ==
             sorted(r["record"]["identity_sha256"] for r in reports.values()),
             "COMPARISON_UPSTREAM_IDENTITY")
    path, art = _artifact(folder, child, "comparison")
    _require(path.is_file() and MR._sha(path) == art.get("sha256"), "COMPARISON_DIGEST")
    doc = _json(path)
    v = doc.get("verdict") or {}
    _require(doc.get("schema") == "i6dmo.matched_comparison.v1" and
             v.get("schema") == "i6dmo.matched_decision.v1" and
             doc.get("rule") == "reporting.decide_matched" and doc.get("split") == "DEVELOPMENT"
             and doc.get("test_read") is False and not doc.get("missing") and not v.get("missing")
             and doc.get("common_population_sha256") == population, "COMPARISON_SCHEMA_OR_INCOMPLETE")
    _require(doc.get("status", "COMPLETE") == "COMPLETE" and
             set(v.get("scores") or {}) == set(MR.LONG_ARMS) and
             set(v.get("per_head") or {}) == {str(h) for h in MR.LONG_HORIZONS},
             "COMPARISON_HEAD_SCHEMA")
    _require(v.get("selected") in MR.LONG_ARMS and doc.get("disposition") == MR.disposition_of(v)
             == child.get("disposition"), "COMPARISON_SELECTION_CONTRADICTION")
    blocks = [reports[a]["metrics"]["matched"] for a in MR.LONG_ARMS]
    for key in ("common_population_sha256", "development_row_ids_sha256",
                "development_head_mask_sha256", "development_counts"):
        _require(blocks[0].get(key) == blocks[1].get(key), f"COMPARISON_SUPPORT:{key}")
    scores = {}
    for arm, report in reports.items():
        recorded = (doc.get("inputs") or {}).get(arm + ".report") or {}
        _require({"identity_sha256", "stored_terminal_sha256", "metrics_sha256",
                  "common_population_sha256"} <= set(recorded) and
                 all(report["record"].get(k) == value for k, value in recorded.items()),
                 f"COMPARISON_INPUT_CHANGED:{arm}")
        selection = R.selection_score(report["metrics"]["per_horizon"], MR.LONG_HORIZONS)
        declared = report["metrics"]["matched"]["selection"]
        _require(selection["status"] == declared.get("status") == "COMPLETE" and
                 not declared.get("failed_heads") and R._close(selection["score"], declared.get("score"))
                 and R._close(selection["score"], (v.get("scores") or {}).get(arm)),
                 f"COMPARISON_SCORE:{arm}")
        scores[arm] = selection["score"]
        for h, row in report["metrics"]["per_horizon"].items():
            _require(R._close((v.get("per_head") or {}).get(h, {}).get(arm), row["relative_error"]),
                     f"COMPARISON_HEAD:{arm}:h{h}")
    diff = scores["mg1l"] - scores["mg12m"]
    expected = "mg12m" if abs(diff) <= MR.TOLERANCE or diff > 0 else "mg1l"
    _require(v["selected"] == expected and v.get("decision") ==
             ("TIE_RETAIN_CHEAPER" if abs(diff) <= MR.TOLERANCE else "LOWER_SCORE"),
             "COMPARISON_SCORE_SELECTION_CONTRADICTION")
    return v["selected"], dict(stored_terminal_sha256=pin["sha256"], comparison_sha256=art["sha256"],
                                document=doc)


def _input(pin, spec, key):
    _require(isinstance(pin, dict) and pin.get("sha256"), f"INPUT_PIN_INVALID:{key}")
    if pin.get("path"):
        path = Path(pin["path"])
        _require(path.is_absolute(), f"INPUT_ABSOLUTE_PATH_REQUIRED:{key}")
    else:
        _require(pin.get("relpath") and spec.get("data_root"), f"INPUT_ROOT_MISSING:{key}")
        root = Path(spec["data_root"]).resolve()
        path = (root / pin["relpath"]).resolve()
        _require(path.is_relative_to(root), f"INPUT_PATH_ESCAPE:{key}")
    if not path.is_file():
        raise Refusal("WAIT_ARTIFACT", [f"effective_config.raw_pair.{key}"])
    _require(MR._sha(path) == pin["sha256"], f"INPUT_DIGEST_MISMATCH:{key}")
    return path


def _predictions(report, family, window):
    import numpy as np
    from tools.i6dmo import data, provenance
    name = next(n for n in SA.PREDICTION_ARTIFACT_NAMES if n in report["child"]["artifacts"])
    path, _ = _artifact(report["folder"], report["child"], name)
    _require(path.is_file() and MR._sha(path) == report["record"]["predictions_sha256"],
             f"PREDICTION_DIGEST:{report['unit']}")
    with np.load(path, allow_pickle=False) as z:
        hs = z["horizons_hours"].tolist()
        _require(hs == list(SA.FAMILIES[family]), f"PREDICTION_HORIZONS:{family}")
        ids, mask = z["row_id"], z["mask"]
        _require(ids.ndim == 1 and ids.dtype.kind in "iu" and mask.dtype.kind == "b" and
                 mask.shape == z["target_price"].shape == z["predicted_price"].shape == (len(ids), len(hs))
                 and z["origin_price"].shape == ids.shape and np.all(np.diff(ids) > 0),
                 f"PREDICTION_SUPPORT_SHAPE:{family}")
        # Time bounds are checked before touching target/prediction values for replay.
        _require(all(window[0].timestamp() <= int(t) < window[1].timestamp() for t in ids),
                 f"PREDICTION_OUTSIDE_VALIDATION:{family}")
        for j, h in enumerate(hs):
            rows = [data.row_id(int(t)) for t in ids[mask[:, j]]]
            _require(provenance.canonical_sha256(rows) ==
                     report["metrics"]["per_horizon"][str(h)]["row_ids_sha256"],
                     f"PREDICTION_HEAD_POPULATION:{family}:h{h}")
        replay = R.manual_pooled_replay(z, hs)
        for h, row in replay.items():
            for key in ("n", "mae", "naive_mae", "relative_error"):
                _require(R._close(row[key], report["metrics"]["per_horizon"][h].get(key)),
                         f"PREDICTION_REPLAY:{family}:h{h}:{key}")
    report["record"]["prediction_bytes_verified"] = True
    return SA._family_predictions(report["unit"], report["folder"], report["child"], family, window)


def run_unit(spec, out_dir, progress_path):
    """Evaluation hook; returns standard terminal plus retained decision/OLAP reports.

    raw_pair: short_report {absolute path, sha256, unit_id: mg2m.report},
    policy {min_valid_long_heads, min_valid_short_heads}, optional pinned
    strategy_bars/calendar {path|relpath, sha256} and strategy_execution.
    """
    started = time.time()
    out = Path(out_dir)
    artifacts, gates, reports = {}, dict(test_read=False), []

    def put(name, filename, value):
        MR._atomic(out / filename, value)
        artifacts[name] = dict(path=filename, sha256=MR._sha(out / filename))

    def terminal(status, disposition):
        return dict(schema="i6dmo.terminal.v1", unit_id=spec.get("unit_id"),
            identity_sha256=spec.get("identity_sha256"), status=status, disposition=disposition,
            artifacts=artifacts, eligibility={}, gates=gates, reports=reports,
            costs=dict(wall_seconds=time.time() - started, fit_seconds=0.,
                       score_seconds=time.time() - started, cgroup_peak_bytes=None, gpu_uuid=None))

    try:
        _require(not spec.get("test_open"), "TEST_OPEN_REFUSED")
        effective = spec.get("effective_config") or {}
        _require(spec.get("kind") == "evaluation" and effective.get("mode") == "MATCHED_LONG_CONTRAST",
                 "KIND_OR_MODE_INVALID")
        raw = effective.get("raw_pair") or {}
        required = ["upstream." + u for u in ("comparison", "mg1l.report", "mg12m.report")
                    if not (spec.get("upstream") or {}).get(u)]
        required += ["effective_config.raw_pair." + k for k in ("short_report", "policy") if not raw.get(k)]
        if required:
            raise Refusal("WAIT_ARTIFACT", required)
        policy = raw["policy"]
        _require(set(policy) == {"min_valid_long_heads", "min_valid_short_heads"} and
                 all(type(v) is int and 1 <= v <= 6 for v in policy.values()), "FROZEN_PAIR_POLICY_INVALID")
        short_pin = raw["short_report"]
        _require(short_pin.get("unit_id") == "mg2m.report", "SHORT_REPORT_UNIT_MISMATCH")
        population = ((effective.get("data_pins") or {}).get("common_population") or {}).get("population_sha256")
        _require(bool(population) and effective.get("code_commit"), "LONG_POPULATION_OR_CODE_PIN_MISSING")
        longs = {a: _report(a + ".report", spec["upstream"][a + ".report"], a, population,
                             effective["code_commit"]) for a in MR.LONG_ARMS}
        selected, comparison = _comparison(spec, longs, population)
        short = _report("mg2m.report", short_pin, "mg2m")
        pair = dict(short=short, long=longs[selected])
        decision = dict(SA.pair_eligibility(short["rows"], pair["long"]["rows"], **policy),
            schema=SA.PAIR_DECISION_SCHEMA, selected_long=selected, comparison=comparison,
            inputs={r["unit"]: r["record"] for r in [short, *longs.values()]},
            frozen_policy=policy, strategy_policy_pin=SA.STRATEGY_PIN,
            test_read=False, predictions_read=False,
            prediction_authentication="digest declarations only until eligible execution")
        put("eligibility", "ELIGIBILITY.json", decision)
        report = SA._decision_report(spec, decision, artifacts["eligibility"]["sha256"])
        for family, item in pair.items():
            for h, row in item["rows"].items():
                for key in ("mae", "naive", "eligible"):
                    if row[key] is not None:
                        report["metrics"].append(dict(metric=f"{family.upper()}_{key}",
                            value=int(row[key]) if key == "eligible" else row[key],
                            split="validation", horizon=int(h), unit="bool" if key == "eligible" else R.UNIT))
        report["tags"].update(selected_long=selected, retained_short="mg2m.report")
        reports.append(report)
        gates.update(pair_status=decision["status"], selected_long=selected)
        if decision["status"] != "ELIGIBLE":
            SA._progress(progress_path, 4, "publish")
            return terminal("COMPLETE", "PAIR_NOT_ELIGIBLE")
        required = ["effective_config.raw_pair." + k for k in
                    ("strategy_bars", "strategy_calendar", "strategy_execution") if not raw.get(k)]
        if not (effective.get("design") or {}).get("splits", {}).get("validation"):
            required.append("effective_config.design.splits.validation")
        if required:
            raise Refusal("WAIT_ARTIFACT", required)
        window = SA._validation_window(spec)
        _require(window[0] < window[1], "VALIDATION_WINDOW_INVALID")
        bars_path = _input(raw["strategy_bars"], spec, "strategy_bars")
        calendar_path = _input(raw["strategy_calendar"], spec, "strategy_calendar")
        try:
            pinned = SA.load_pinned_policy()
        except SA.StrategyAdapterError as exc:
            if exc.code == "STRATEGY_PIN_UNRESOLVED":
                raise Refusal("WAIT_ARTIFACT", ["strategy_policy:" + SA.STRATEGY_PIN["commit"] +
                                               ":" + SA.STRATEGY_PIN["path"]]) from None
            raise
        execution = raw["strategy_execution"]
        if execution.get("instrument") is not None:
            sealed = SA.derive_execution_policy(execution["instrument"], policy=pinned)
        elif execution.get("instrument_file"):
            if not Path(execution["instrument_file"]).is_file():
                raise Refusal("WAIT_ARTIFACT", ["effective_config.raw_pair.strategy_execution.instrument_file"])
            sealed = SA.derive_execution_policy_from_file(execution["instrument_file"], policy=pinned)
        else:
            raise Refusal("WAIT_ARTIFACT", ["effective_config.raw_pair.strategy_execution.instrument"])
        ruled = SA.execution_from_policy(sealed)
        _require((ruled["min_long"], ruled["min_short"]) ==
                 (policy["min_valid_long_heads"], policy["min_valid_short_heads"]), "EXECUTION_PAIR_POLICY_MISMATCH")
        preds = {f: _predictions(r, f, window) for f, r in pair.items()}
        bars = SA._bars(bars_path, window)
        _, weeks = SA.c1.load_calendar(_json(calendar_path))
        result = SA.run_backtest(bars=bars, weeks=weeks, short_predictions=preds["short"],
            long_predictions=preds["long"], policy=pinned, entry_params=ruled["entry"],
            execution=ruled["execution"], validation_eligible=decision["eligible"], allow_test=False)
        result.update(execution_policy_sha256=sealed["receipt_sha256"],
                      cost_verification=ruled["cost_verification"], inputs=decision["inputs"],
                      source_pins={k: raw[k] for k in ("strategy_bars", "strategy_calendar")},
                      validation_window=effective["design"]["splits"]["validation"],
                      selected_long=selected, test_read=False)
        put("execution_policy", "EXECUTION_POLICY.json", sealed)
        put("backtest", "BACKTEST.json", result)
        status = "PASS" if not result["failures"] else "EXECUTION_FAILURES"
        reports.append(SA._business_report(spec, result, status))
        gates.update(pair_status=status, strategy_status=status, costs_sha256=ruled["costs_sha256"],
                     execution_policy_sha256=sealed["receipt_sha256"])
        SA._progress(progress_path, 4, "publish")
        return terminal("COMPLETE", status)
    except Refusal as exc:
        if exc.required:
            gates["required_objects"] = exc.required
            return terminal("FAILED", "RAW_PAIR_MISSING_ARTIFACT")
        return terminal("FAILED", str(exc))
    except (MR._Refusal, SA._Refusal, SA.StrategyAdapterError, SA.c1.EligibilityError) as exc:
        return terminal("FAILED", str(getattr(exc, "code", None) or getattr(exc, "disposition", None) or exc))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        return terminal("FAILED", "RAW_PAIR_INVALID:" + type(exc).__name__)
