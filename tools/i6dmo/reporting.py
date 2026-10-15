"""I6-DMO report unit: per-horizon original-price MAE vs naive on identical rows, weekly diagnostics.

Hook for ``python -m tools.i6dmo.unit_runner --kind report run-unit ...`` (``run_unit``) and the
worker's pre-claim check (``preconditions``). It never fits and never reads TEST.

Inputs, all digest-checked: the data lane's frozen bundle (BUNDLE.json: identity, sampling rule,
validation row ids, support counts) and its retained input directory (INPUT.npz), plus the model
lane's ``predictions/validation.npz`` from the upstream fit terminal. The validation rows are
re-materialized through the data lane's public API (load_input -> build_origin_table ->
materialize) and must reproduce the bundle's row-id digest; the predictions must carry exactly
those row ids, horizons and masks. Scoring is the data lane's ``naive.paired_metrics`` (audited C2
plus manual and disk-replay cross-checks) with its ``weekly_diagnostics``. Any disagreement is a
typed FAILED terminal, never a silent repair.

Outputs: METRICS.json (paired metrics + weekly), SUPPORT.json (S_h counts and exclusion reasons),
eligibility per head (MAE_h < naive_h on the same support), and governed OLAP reports: one annual
report and one per expected week (absent weeks explicit, Week_N = 0).
"""
import hashlib
import json
import math
import re
import time

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

SAMPLING_RULE = "DS2_EXACT"
SAMPLING_BY_FAMILY = {"short": "DS2_EXACT", "long": "DS12_EXACT"}
MARKET_SAMPLING_BY_FAMILY = {"short": "MG2_MARKET_EXACT", "long": "MG12_MARKET_EXACT"}


def _expected_rule(spec, family):
    """The sampling rule this unit's bundle must declare, from the sealed design's own schema."""
    effective = spec.get("effective_config") or {}
    design = (effective.get("long") or {}).get("design") if family == "long" else effective.get("design")
    block = (design or {}).get("market_time")
    if isinstance(block, dict) and block.get("rule"):
        return block["rule"]
    return SAMPLING_BY_FAMILY.get(family, SAMPLING_RULE)
SUPERSEDED_BUNDLE_PREFIXES = ("5221615d",)
UNIT = "USD_per_EUR"


class ReportFailure(ValueError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic(path, value):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False, default=_jsonable))
    tmp.replace(path)


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def preconditions(context):
    """None when the unit can start; otherwise a typed WAIT_ARTIFACT cause naming the owner."""
    family = context["unit"]["family"]
    pins = context["effective_config"].get("data_pins") or {}
    if f"{family}_bundle" not in pins:
        return f"BUNDLE_PIN_MISSING:{family}_bundle owner=data lane (relayed by Satoshi)"
    if f"{family}_input" not in pins:
        return f"INPUT_PIN_MISSING:{family}_input owner=data lane (retained INPUT.npz directory)"
    return None


def _design(spec):
    effective = spec["effective_config"]
    return effective["long"]["design"] if spec["family"] == "long" else effective["design"]


def _check_bundle(pin, family="short", expected_rule=None):
    manifest_path = Path(pin["path"]) / "BUNDLE.json"
    if not manifest_path.exists() or _sha(manifest_path) != pin["sha256"]:
        raise ReportFailure("BUNDLE_PIN_MISMATCH", "BUNDLE.json digest differs from the pin")
    manifest = json.loads(manifest_path.read_text())
    identity = str(manifest.get("bundle_sha256", ""))
    if identity.startswith(SUPERSEDED_BUNDLE_PREFIXES):
        raise ReportFailure("SUPERSEDED_BUNDLE", identity[:8])
    if pin.get("bundle_sha256") and pin["bundle_sha256"] != identity:
        raise ReportFailure("BUNDLE_PIN_MISMATCH", "bundle_sha256 differs from the pin")
    want = expected_rule or SAMPLING_BY_FAMILY.get(family, SAMPLING_RULE)
    if (manifest.get("input_sampling") or {}).get("rule") != want:
        raise ReportFailure("BUNDLE_SAMPLING", str((manifest.get("input_sampling") or {}).get("rule")))
    return manifest


def materialize_validation(spec):
    """Default loader: re-materialize VALIDATION through the data lane's public API."""
    from tools.i6dmo import data
    design = _design(spec)
    if design.get("schema") == "i6dmo.market_time_design.v1":
        from tools.i6dmo import market_grid, train
        elapsed = train._elapsed_view(design)
        expected = market_grid.market_design(elapsed)
        path = spec["pins"][f"{spec['family']}_bundle"]["path"]
        market_grid.authenticate_market_bundle(market_grid.load_market_manifest(path), expected)
        bundle = data.load_input(spec["pins"][f"{spec['family']}_input"]["path"], elapsed)
        table = market_grid.build_market_table(bundle, elapsed)
        arrays = market_grid.materialize(bundle, table, "VALIDATION")
        arrays.expected_weeks = tuple(table.split_weeks["VALIDATION"])
        arrays.calendar_id = table.identity["calendar_id"]
        return arrays
    bundle = data.load_input(spec["pins"][f"{spec['family']}_input"]["path"], design)
    table = data.build_origin_table(bundle, design)
    return data.materialize(bundle, table, "VALIDATION")


def _fit_key(spec):
    """The upstream fit unit: ``<family>.fit`` or the matched spec's declared fit unit."""
    return (spec.get("matched") or {}).get("fit_unit") or f"{spec['family']}.fit"


def pred_doc_path(spec):
    up = spec["upstream"][_fit_key(spec)]
    stored = json.loads(Path(up["path"]).read_text())
    art = stored["terminal"]["artifacts"]["predictions/validation.npz"]
    return Path(up["path"]).parent / "out" / art["path"]


def _predictions(spec):
    up = spec["upstream"].get(_fit_key(spec)) or {}
    path = Path(up.get("path", ""))
    if not path.exists() or _sha(path) != up.get("sha256"):
        raise ReportFailure("UPSTREAM_TERMINAL_DIGEST", _fit_key(spec))
    stored = json.loads(path.read_text())
    terminal = stored["terminal"]
    art = (terminal.get("artifacts") or {}).get("predictions/validation.npz")
    if art is None:
        raise ReportFailure("PREDICTIONS_MISSING", "predictions/validation.npz not in the fit terminal")
    file = path.parent / "out" / art["path"]
    if not file.exists() or _sha(file) != art["sha256"]:
        raise ReportFailure("PREDICTIONS_DIGEST", art["path"])
    import numpy as np
    with np.load(file, allow_pickle=False) as z:
        doc = {k: z[k] for k in z.files}
    weights = (terminal.get("trainer_identity") or {}).get("weights_sha256") or stored.get("identity_sha256")
    sealed = datetime.fromtimestamp(float(stored.get("stored_at", 0)), timezone.utc).isoformat()
    return doc, weights, sealed, _sha(file)


def _align(arrays, pred):
    import numpy as np
    if tuple(int(h) for h in pred["horizons_hours"]) != tuple(int(h) for h in arrays.horizons):
        raise ReportFailure("HORIZON_MISMATCH", "prediction head order differs from the bundle")
    if pred["row_id"].shape != arrays.origin_ts.shape or not np.array_equal(pred["row_id"].astype(np.int64),
                                                                            arrays.origin_ts.astype(np.int64)):
        raise ReportFailure("ROW_ID_MISMATCH", "prediction rows differ from the validation rows")
    if not np.array_equal(pred["mask"].astype(bool), arrays.mask.astype(bool)):
        raise ReportFailure("MASK_MISMATCH", "prediction support differs from S_h")
    values = np.asarray(pred["predicted_price"], dtype=np.float64)
    if values.shape != arrays.mask.shape:
        raise ReportFailure("PREDICTION_SHAPE", str(values.shape))
    return values


def _key(text):
    return re.sub(r"[^A-Za-z0-9._:-]", "_", str(text))[:40]


def _report(spec, suffix, tags, metrics, manifest):
    base_tags = {"unit": spec["unit_id"], "family": spec["family"], "split": "validation", "test_read": "false",
                 "sampling": _expected_rule(spec, spec["family"]), "bundle_sha256": str(manifest.get("bundle_sha256")),
                 "cadence": str(spec.get("cadence")), "relative_error": "(MAE_h-naive_h)/naive_h; negative is better",
                 "scale": "original_price"}
    return dict(experiment_key=f"i6dmo:{spec['campaign_id']}:{spec['unit_id']}:{suffix}"[:128],
                experiment_set_key=f"i6dmo:{spec['campaign_id']}", actor="predictor", lake="LOCAL_RETAINED",
                lineage="UNVERIFIED", config_sha256=spec["identity_sha256"],
                code_commit=spec["effective_config"]["code_commit"], project="predictor", phase="I6DMO",
                tags={**base_tags, **tags}, metrics=metrics)


def _metric(name, value, horizon, unit):
    return dict(metric=name, value=value, split="validation", horizon=horizon, unit=unit)


def _annual_report(spec, result, support, manifest):
    metrics, tags = [], {}
    for h, row in result["per_horizon"].items():
        h_int = int(h)
        for name, key, unit in (("Price_MAE", "mae", UNIT), ("Price_Naive_MAE", "naive_mae", UNIT),
                                ("Relative_Error", "relative_error", "ratio"),
                                ("Sum_Abs_Error", "sum_abs_error", UNIT),
                                ("Naive_Sum_Abs_Error", "naive_sum_abs_error", UNIT)):
            if row.get(key) is not None:
                metrics.append(_metric(name, float(row[key]), h_int, unit))
        if row.get("relative_error") is None:
            tags[f"h{h}_relative_error_undefined"] = str(row.get("relative_error_undefined_reason"))
        metrics += [_metric("Scored_Rows", row["n"], h_int, "count"),
                    _metric("Declared_Rows", row["n_declared"], h_int, "count"),
                    _metric("Failed_Rows", row["n_failed"], h_int, "count")]
        if row.get("eligible_mae_below_naive") is not None:
            metrics.append(_metric("Eligible_MAE_Below_Naive", 1.0 if row["eligible_mae_below_naive"] else 0.0,
                                   h_int, "bool"))
        head = (support.get("heads") or {}).get(h, {})
        if "head_valid" in support:
            metrics.append(_metric("Support_Valid_Labels", support["head_valid"].get(h), h_int, "count"))
        if "head_entry_valid" in support:
            metrics.append(_metric("Support_Entry_Valid", support["head_entry_valid"].get(h), h_int, "count"))
        for reason, count in sorted((head.get("reasons") or {}).items()):
            metrics.append(_metric(f"Excluded_{_key(reason)}", count, h_int, "count"))
    for key, count in sorted((support.get("origins") or {}).items()):
        if isinstance(count, (int, float)):
            metrics.append(_metric(f"Origins_{_key(key)}", count, None, "count"))
    metrics = [m for m in metrics if m["value"] is not None]
    return _report(spec, "annual", tags, metrics, manifest)


def _weekly_reports(spec, weekly, manifest):
    per_h = weekly["per_horizon"]
    weeks = list(next(iter(per_h.values()))["weeks"]) if per_h else []
    reports = []
    for week in weeks:
        metrics, tags = [], {"week_id": str(week)}
        for h, block in per_h.items():
            row = block["weeks"][week]
            tags[f"h{h}"] = row["status"]
            metrics.append(_metric("Week_N", row["n"], int(h), "count"))
            if row["status"] == "OBSERVED":
                for name, key, unit in (("Week_MAE", "mae", UNIT), ("Week_Naive_MAE", "naive_mae", UNIT),
                                        ("Week_Sum_Abs_Error", "sum_abs_error", UNIT),
                                        ("Week_Naive_Sum_Abs_Error", "naive_sum_abs_error", UNIT),
                                        ("Week_Relative_Error", "relative_error", "ratio")):
                    if row.get(key) is not None:
                        metrics.append(_metric(name, float(row[key]), int(h), unit))
        reports.append(_report(spec, f"week:{_key(week)}", tags, metrics, manifest))
    return reports


def _progress(path, done, phase):
    _atomic(path, dict(done=done, total=5, unit="step", phase=phase, updated_at=time.time()))


def _failed(code, detail, started):
    return dict(status="FAILED", disposition=code, detail=str(detail)[:300], artifacts={}, eligibility={}, gates={},
                reports=[], costs=dict(fit_seconds=0.0, score_seconds=time.time() - started))


def run_unit(spec, out_dir, progress_path, *, arrays_loader=None):
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if spec.get("test_open") or spec.get("kind") != "report":
        return _failed("NOT_A_VALIDATION_REPORT", "report units never read TEST", started)
    if spec.get("common_population") is not None:
        if arrays_loader is not None:
            return _failed("MATCHED_LOADER_INJECTION", "the matched report reads the frozen "
                           "population and the arm source itself; no arrays loader", started)
        return run_matched_unit(spec, out, progress_path, started=started)
    from tools.i6dmo import naive
    try:
        family = spec["family"]
        pins = spec.get("pins") or {}
        if f"{family}_bundle" not in pins or f"{family}_input" not in pins:
            raise ReportFailure("PIN_MISSING", "bundle/input pins absent")
        manifest = _check_bundle(pins[f"{family}_bundle"], family, _expected_rule(spec, family))
        input_pin = pins[f"{family}_input"]
        if _sha(Path(input_pin["path"]) / "INPUT.npz") != input_pin["sha256"]:
            raise ReportFailure("INPUT_PIN_MISMATCH", "INPUT.npz digest differs from the pin")
        _progress(progress_path, 1, "verify")
        arrays = (arrays_loader or materialize_validation)(spec)
        from tools.i6dmo import provenance
        info = manifest["splits"]["validation"]
        if str(arrays.split).upper() != "VALIDATION":
            raise ReportFailure("SPLIT_NOT_VALIDATION", str(arrays.split))
        if provenance.canonical_sha256([str(r) for r in arrays.row_ids.tolist()]) != info["row_ids_sha256"]:
            raise ReportFailure("ROW_ID_MISMATCH", "re-materialized validation rows differ from the bundle")
        _progress(progress_path, 2, "load")
        pred_doc, checkpoint, sealed_at, pred_sha = _predictions(spec)
        values = _align(arrays, pred_doc)
        _progress(progress_path, 3, "score")
        try:
            result = naive.paired_metrics(arrays, values, _design(spec), input_dir=input_pin["path"],
                                          checkpoint=checkpoint, sealed_at=sealed_at)
        except naive.NaiveReplayMismatch as exc:
            raise ReportFailure("NAIVE_REPLAY_DISAGREES", str(exc)) from None
    except ReportFailure as exc:
        return _failed(exc.code, exc, started)
    except Exception as exc:  # data-lane typed errors keep their code
        return _failed(getattr(exc, "code", type(exc).__name__), exc, started)
    support = dict(counts=info.get("counts"), head_valid=info.get("head_valid"),
                   head_entry_valid=info.get("head_entry_valid"), n_rows=info.get("n_rows"),
                   heads=(info.get("counts") or {}).get("heads"), origins=(info.get("counts") or {}).get("origins"),
                   scored={h: row["n"] for h, row in result["per_horizon"].items()})
    _atomic(out / "METRICS.json", dict(result, predictions_sha256=pred_sha, bundle_sha256=manifest["bundle_sha256"]))
    import shutil
    shutil.copyfile(pred_doc_path(spec), out / "validation_predictions.npz")
    if _sha(out / "validation_predictions.npz") != pred_sha:
        return _failed("PREDICTIONS_COPY_DIGEST", "copy differs", started)
    _atomic(out / "SUPPORT.json", support)
    _progress(progress_path, 4, "publish")
    eligibility = {h: dict(n=row["n"], mae=row["mae"], naive=row["naive_mae"], relative_error=row["relative_error"],
                           eligible=row["eligible_mae_below_naive"], n_declared=row["n_declared"],
                           n_failed=row["n_failed"], state=row["state"], row_ids_sha256=row["row_ids_sha256"])
                   for h, row in result["per_horizon"].items()}
    reports = [_annual_report(spec, result, support, manifest)] + _weekly_reports(spec, result["weekly"], manifest)
    _progress(progress_path, 5, "done")
    return dict(status="COMPLETE", disposition="PAIRED_METRICS_AGREE",
                artifacts={"metrics": dict(path="METRICS.json", sha256=_sha(out / "METRICS.json")),
                           "support": dict(path="SUPPORT.json", sha256=_sha(out / "SUPPORT.json")),
                           "validation_predictions": dict(path="validation_predictions.npz", sha256=pred_sha)},
                eligibility=eligibility, gates={}, reports=reports,
                costs=dict(fit_seconds=0.0, score_seconds=time.time() - started),
                lineage=dict(bundle_sha256=manifest["bundle_sha256"], predictions_sha256=pred_sha,
                             checkpoint=checkpoint, c2=result["c2"]))


# =========================================================================== matched report (SC05/SC06)
#
# NEXT_CONTRASTS_PLAN section 2: each matched arm is scored on the FROZEN common DEVELOPMENT rows
# and per-head masks of the data lane's common population. Model and naive per horizon use the
# same rows, the same mask and original-price units; C2 scoring, the manual pooled replay from
# the SAVED predictions and the independent disk naive replay must agree. Per-head numbers are
# primary; the equal-head selection score never hides a failed head.

MATCHED_REPORT_SCHEMA = "i6dmo.matched_report.v1"
SELECTION_TIE = 1e-8
_REPLAY_REL_TOL = 1e-12


def _close(a, b):
    if a is None or b is None:
        return a is b
    return abs(a - b) <= _REPLAY_REL_TOL * max(1.0, abs(a), abs(b))


def _upstream_receipt(spec):
    """The matched fit's RECEIPT.json through the upstream terminal's digest chain."""
    up = spec["upstream"].get(_fit_key(spec)) or {}
    path = Path(up.get("path", ""))
    if not path.exists() or _sha(path) != up.get("sha256"):
        raise ReportFailure("UPSTREAM_TERMINAL_DIGEST", _fit_key(spec))
    stored = json.loads(path.read_text())
    terminal = stored.get("terminal") or {}
    art = (terminal.get("artifacts") or {}).get("RECEIPT")
    if art is None:
        raise ReportFailure("NOT_A_MATCHED_FIT", "the upstream fit terminal carries no RECEIPT")
    file = path.parent / "out" / art["path"]
    if not file.exists() or _sha(file) != art["sha256"]:
        raise ReportFailure("RECEIPT_DIGEST", art["path"])
    receipt = json.loads(file.read_text())
    if terminal.get("trainer_identity") is not None and terminal["trainer_identity"] != receipt.get("identity"):
        raise ReportFailure("NOT_A_MATCHED_FIT", "terminal trainer identity differs from the receipt")
    return receipt


def _input_pin(spec):
    matched = spec.get("matched") or {}
    pin = matched.get("input") or (spec.get("pins") or {}).get(f"{spec['family']}_input")
    if not pin:
        raise ReportFailure("PIN_MISSING", "matched report needs the retained INPUT pin")
    if _sha(Path(pin["path"]) / "INPUT.npz") != pin["sha256"]:
        raise ReportFailure("INPUT_PIN_MISMATCH", "INPUT.npz digest differs from the pin")
    return pin


def _arm_development(population, design, input_dir):
    """DEVELOPMENT prices/weeks of exactly the common rows through the data lane's accessor,
    independently replayed by timestamp against the retained input (REPLAY_MISMATCH)."""
    from tools.i6dmo import train as T
    tensors = T.arm_split(population, design, "DEVELOPMENT")
    cp = T._common_population_module()
    try:
        replay = cp.replay_against_input(tensors, input_dir)
    except Exception as exc:
        raise ReportFailure(getattr(exc, "code", None) or "REPLAY_MISMATCH", str(exc)[:300]) from None
    dev = population.splits["DEVELOPMENT"]
    if not np.array_equal(np.asarray(tensors.row_id, np.int64), dev.origin_row_ids) or \
            not np.array_equal(np.asarray(tensors.head_mask, bool), np.asarray(dev.head_mask, bool)):
        raise ReportFailure("COMMON_POPULATION_ARM_MISMATCH", "DEVELOPMENT")
    return {"origin_price": np.asarray(tensors.origin_price, np.float64),
            "target_price": np.asarray(tensors.target_price, np.float64),
            "week_ids": np.asarray(tensors.week_ids), "replay": replay}


def manual_pooled_replay(pred_doc, horizons):
    """Per-head pooled MAE/naive/relative error recomputed ONLY from the saved predictions file."""
    out = {}
    mask = np.asarray(pred_doc["mask"], bool)
    pred = np.asarray(pred_doc["predicted_price"], np.float64)
    target = np.asarray(pred_doc["target_price"], np.float64)
    origin = np.asarray(pred_doc["origin_price"], np.float64)
    for j, h in enumerate(horizons):
        sel = mask[:, j]
        n = int(sel.sum())
        if not n:
            out[str(h)] = {"n": 0, "mae": None, "naive_mae": None, "relative_error": None}
            continue
        model_sum = math.fsum(np.abs(pred[sel, j] - target[sel, j]).tolist())
        naive_sum = math.fsum(np.abs(target[sel, j] - origin[sel]).tolist())
        mae, naive_mae = model_sum / n, naive_sum / n
        out[str(h)] = {"n": n, "sum_abs_error": model_sum, "naive_sum_abs_error": naive_sum,
                       "mae": mae, "naive_mae": naive_mae,
                       "relative_error": (mae - naive_mae) / naive_mae if naive_mae > 0 else None}
    return out


def selection_score(per_horizon, horizons):
    """Equal-head mean of annual pooled relative MAE; INCOMPLETE (score None) on any failed head."""
    failed, rels = [], []
    for h in horizons:
        row = per_horizon.get(str(h)) or {}
        if not row.get("n") or row.get("relative_error") is None or row.get("n_failed"):
            failed.append({"horizon": int(h), "n": row.get("n"), "n_failed": row.get("n_failed"),
                           "relative_error": row.get("relative_error"),
                           "reason": row.get("relative_error_undefined_reason") or "missing_or_failed"})
        else:
            rels.append(float(row["relative_error"]))
    doc = {"definition": "equal-head mean over the six heads of annual pooled relative MAE "
                         "(MAE_h-naive_h)/naive_h on the frozen common DEVELOPMENT rows; "
                         "negative is better; per-head numbers stay primary",
           "heads": [int(h) for h in horizons]}
    if failed:
        return dict(doc, status="INCOMPLETE", score=None, failed_heads=failed)
    return dict(doc, status="COMPLETE", score=math.fsum(rels) / len(rels), failed_heads=[])


def _matched_expected_weeks(design, input_dir, observed_week_ids):
    """Expected DEVELOPMENT weeks from the sealed calendar through the market_grid table, exactly as
    the closed MG2 report derived them (the population identities carry no empty weeks). The input
    directory's CALENDAR.json is the sealed calendar: load_common_population(input_dir=...) already
    verified it against the population's calendar digest. Every observed common week must be one of
    them (EXPECTED_WEEKS_MISMATCH otherwise); weeks without common rows stay explicit (Week_N = 0)."""
    from tools.i6dmo import data as data_mod, market_grid, train as T
    elapsed = T._elapsed_view(design)
    table = market_grid.build_market_table(data_mod.load_input(input_dir, elapsed), elapsed)
    expected = tuple(str(w) for w in table.split_weeks["VALIDATION"])
    stray = sorted({str(w) for w in observed_week_ids} - set(expected))
    if stray:
        raise ReportFailure("EXPECTED_WEEKS_MISMATCH", f"observed weeks outside the calendar: {stray[:5]}")
    return expected, "sealed_calendar_market_grid_validation_weeks"


def run_matched_unit(spec, out, progress_path, *, started=None):
    started = time.time() if started is None else started
    out = Path(out)
    from tools.i6dmo import data as data_mod, naive, train as T
    from tools.i6n_contracts.feature_availability import MANIFEST_NO_STLFSI4
    try:
        design = _design(spec)
        horizons = [int(h) for h in design["horizons_hours"]]
        matched = spec.get("matched") or {}
        require_scientific_fit = matched.get("require_scientific_fit") is not False
        arm = matched.get("arm")
        design_arm = T.matched_arm(design)
        if arm != design_arm:
            raise ReportFailure("ARM_DESIGN_MISMATCH", f"spec arm {arm!r} vs design arm {design_arm!r}")
        pin = spec["common_population"]
        input_pin = _input_pin(spec)
        population = T.load_matched_population(pin["path"], pin.get("sha256"), horizons,
                                               list(MANIFEST_NO_STLFSI4), input_dir=input_pin["path"])
        _progress(progress_path, 1, "verify")
        receipt = _upstream_receipt(spec)
        if receipt.get("status") != "COMPLETED":
            raise ReportFailure("UPSTREAM_FIT_NOT_COMPLETED", str(receipt.get("code")))
        T.verify_matched_identity(receipt.get("identity"), population.digest, arm)
        if require_scientific_fit and (receipt.get("scientific") is not True
                                       or receipt.get("device") != "gpu"):
            raise ReportFailure("NON_SCIENTIFIC_FIT", f"a scientific matched report needs a scientific GPU fit; "
                                f"receipt scientific={receipt.get('scientific')!r} device={receipt.get('device')!r}")
        block = receipt.get("matched") or {}
        if block.get("common_population_sha256") != population.digest or block.get("arm") != arm:
            raise ReportFailure("NOT_A_MATCHED_FIT", "receipt matched block differs")
        from tools.i6dmo import model as M
        capacity = block.get("capacity") or {}
        literal = M.DECLARED_CAPACITY[M.MATCHED_ARMS[arm]]
        if capacity.get("arm") != arm or any(capacity.get("counts", {}).get(k) != v
                                             for k, v in literal.items()):
            raise ReportFailure("CAPACITY_UNREPORTED_CHANGE", f"{arm} capacity record")
        pred_doc, checkpoint, sealed_at, pred_sha = _predictions(spec)
        if (receipt.get("artifacts") or {}).get("predictions/validation.npz") != pred_sha:
            raise ReportFailure("PREDICTIONS_DIGEST", "terminal predictions differ from the receipt")
        _progress(progress_path, 2, "load")
        dev = population.splits["DEVELOPMENT"]
        if [int(h) for h in pred_doc["horizons_hours"]] != horizons:
            raise ReportFailure("HORIZON_MISMATCH", "prediction head order differs from the design")
        if not np.array_equal(np.asarray(pred_doc["row_id"], np.int64), dev.origin_row_ids):
            raise ReportFailure("ROW_ID_MISMATCH", "predictions are not the common DEVELOPMENT rows")
        mask = np.asarray(dev.head_mask, bool)
        if not np.array_equal(np.asarray(pred_doc["mask"], bool), mask):
            raise ReportFailure("MASK_MISMATCH", "prediction support differs from the common mask")
        values = np.asarray(pred_doc["predicted_price"], np.float64)
        if values.shape != mask.shape:
            raise ReportFailure("PREDICTION_SHAPE", str(values.shape))
        arm_dev = _arm_development(population, design, input_pin["path"])
        saved_target = np.asarray(pred_doc["target_price"], np.float64)
        if not np.array_equal(np.asarray(pred_doc["origin_price"], np.float64), arm_dev["origin_price"]) \
                or not np.array_equal(saved_target[mask], arm_dev["target_price"][mask]):
            raise ReportFailure("PRICE_REPLAY_MISMATCH",
                                "saved origin/target prices differ from the arm source")
        identities = getattr(population, "identities", None) or {}
        calendar = (identities.get("calendar") or {}).get("calendar_id")
        if not calendar:
            raise ReportFailure("CALENDAR_ID_MISSING", "population identities carry no calendar_id")
        expected, weeks_source = _matched_expected_weeks(design, input_pin["path"], arm_dev["week_ids"])

        class _View:
            pass
        view = _View()
        view.split, view.horizons = "DEVELOPMENT", tuple(horizons)
        view.mask, view.entry_mask = mask, np.asarray(dev.entry_mask, bool)
        view.y_price = np.where(mask, arm_dev["target_price"], np.nan)
        view.price_origin, view.origin_ts = arm_dev["origin_price"], np.asarray(dev.origin_row_ids)
        view.row_ids = np.asarray([data_mod.row_id(int(t)) for t in dev.origin_row_ids], dtype=str)
        view.week_ids = np.asarray([str(w) for w in arm_dev["week_ids"]], dtype=object)
        view.expected_weeks, view.calendar_id = tuple(str(w) for w in expected), calendar
        c2_design = {"experiment_id": design["experiment_id"], "horizons_hours": horizons,
                     "arm": arm, "common_population_sha256": population.digest}
        _progress(progress_path, 3, "score")
        try:
            result = naive.paired_metrics(view, values, c2_design, input_dir=input_pin["path"],
                                          checkpoint=checkpoint, sealed_at=sealed_at)
        except naive.NaiveReplayMismatch as exc:
            raise ReportFailure("NAIVE_REPLAY_DISAGREES", str(exc)) from None
        manual = manual_pooled_replay(pred_doc, horizons)
        problems = []
        for h in horizons:
            got, ref = result["per_horizon"][str(h)], manual[str(h)]
            if got["n"] != ref["n"] or not all(_close(got[a], ref[b]) for a, b in (
                    ("mae", "mae"), ("naive_mae", "naive_mae"), ("relative_error", "relative_error"))):
                problems.append(f"h{h}: reported {got['mae']}/{got['naive_mae']}/{got['relative_error']} "
                                f"vs saved-prediction replay {ref['mae']}/{ref['naive_mae']}/{ref['relative_error']}")
        if problems:
            raise ReportFailure("MANUAL_REPLAY_DISAGREES", "; ".join(problems))
        selection = selection_score(result["per_horizon"], horizons)
    except ReportFailure as exc:
        return _failed(exc.code, exc, started)
    except Exception as exc:  # typed errors from other lanes keep their code
        return _failed(getattr(exc, "code", type(exc).__name__), exc, started)
    counts = {str(h): int(c) for h, c in sorted(dict(dev.counts).items())}
    matched_doc = {"schema": MATCHED_REPORT_SCHEMA, "arm": arm, "rule": design["market_time"]["rule"],
                   "population": "COMMON_DEVELOPMENT",
                   "common_population_sha256": population.digest,
                   "development_row_ids_sha256": T._ids_digest(dev.origin_row_ids),
                   "development_head_mask_sha256": T._array_digest(mask),
                   "development_counts": counts, "expected_weeks_source": weeks_source,
                   "input_replay": arm_dev["replay"],
                   "capacity": capacity, "fit_identity": receipt.get("identity"),
                   "scientific_fit": receipt.get("scientific"), "fit_device": receipt.get("device"),
                   "fit_runtime": (receipt.get("resources") or {}).get("tensorflow_version"),
                   "fit_gpu_class": ((receipt.get("resources") or {}).get("gpu_device_names") or [None])[0],
                   "require_scientific_fit": require_scientific_fit,
                   "manual_pooled_replay": manual, "selection": selection,
                   "same_rows": "model and naive per horizon on identical common rows, common "
                                "mask and original-price units; C2, saved-prediction replay "
                                "and disk naive replay agree"}
    _atomic(out / "METRICS.json", dict(result, matched=matched_doc, predictions_sha256=pred_sha))
    import shutil
    shutil.copyfile(pred_doc_path(spec), out / "validation_predictions.npz")
    if _sha(out / "validation_predictions.npz") != pred_sha:
        return _failed("PREDICTIONS_COPY_DIGEST", "copy differs", started)
    support = dict(population="COMMON_DEVELOPMENT", common_population_sha256=population.digest,
                   head_valid=counts, n_rows=int(len(dev.origin_row_ids)),
                   head_entry_valid={str(h): int(np.asarray(dev.entry_mask)[:, j].sum())
                                     for j, h in enumerate(horizons)},
                   origins={"common_development": int(len(dev.origin_row_ids))},
                   scored={h: row["n"] for h, row in result["per_horizon"].items()})
    _atomic(out / "SUPPORT.json", support)
    _progress(progress_path, 4, "publish")
    eligibility = {h: dict(n=row["n"], mae=row["mae"], naive=row["naive_mae"],
                           relative_error=row["relative_error"],
                           eligible=row["eligible_mae_below_naive"], n_declared=row["n_declared"],
                           n_failed=row["n_failed"], state=row["state"],
                           row_ids_sha256=row["row_ids_sha256"])
                   for h, row in result["per_horizon"].items()}
    manifest = {"bundle_sha256": (receipt.get("identity") or {}).get("bundle_manifest_sha256")}
    reports = [_annual_report(spec, result, support, manifest)] + \
        _weekly_reports(spec, result["weekly"], manifest)
    reports[0]["metrics"].append(dict(metric="Selection_Score_Equal_Head_Relative_MAE",
                                      value=selection["score"], split="validation", horizon=None,
                                      unit="ratio"))
    reports[0]["metrics"] = [m for m in reports[0]["metrics"] if m["value"] is not None]
    for report in reports:
        report["tags"].update(population="COMMON_DEVELOPMENT", arm=arm,
                              common_population_sha256=population.digest,
                              selection_status=selection["status"])
    _progress(progress_path, 5, "done")
    return dict(status="COMPLETE", disposition="MATCHED_PAIRED_METRICS_AGREE",
                artifacts={"metrics": dict(path="METRICS.json", sha256=_sha(out / "METRICS.json")),
                           "support": dict(path="SUPPORT.json", sha256=_sha(out / "SUPPORT.json")),
                           "validation_predictions": dict(path="validation_predictions.npz",
                                                          sha256=pred_sha)},
                eligibility=eligibility, gates={}, reports=reports,
                costs=dict(fit_seconds=0.0, score_seconds=time.time() - started),
                lineage=dict(common_population_sha256=population.digest, predictions_sha256=pred_sha,
                             checkpoint=checkpoint, c2=result["c2"], arm=arm),
                selection=selection)


def decide_matched(mg1_metrics, mg2_metrics, *, require_scientific=True):
    """Deterministic recipe decision from the two matched METRICS.json documents.

    Lower selection score wins; within SELECTION_TIE the cheaper MG2 is retained; any failed
    head or arm closes INCOMPLETE_COMPARISON with the missing evidence (never a selection from
    incomplete heads). Both arms must carry the same common population, the same DEVELOPMENT
    rows/mask/counts and the declared capacity difference only. Scientific GPU fits are required
    unless engineering callers explicitly pass require_scientific=False.
    """
    from tools.i6dmo import model as M
    a, b = (mg1_metrics or {}).get("matched") or {}, (mg2_metrics or {}).get("matched") or {}
    if a.get("arm") != "mg1" or b.get("arm") != "mg2m":
        raise ReportFailure("ARM_DESIGN_MISMATCH", f"arms {a.get('arm')!r}/{b.get('arm')!r}")
    if require_scientific is not False:
        for block in (a, b):
            if block.get("scientific_fit") is not True or block.get("fit_device") != "gpu":
                raise ReportFailure("NON_SCIENTIFIC_FIT", f"{block.get('arm')}: scientific_fit="
                                    f"{block.get('scientific_fit')!r} fit_device={block.get('fit_device')!r}")
    for key in ("common_population_sha256", "development_row_ids_sha256",
                "development_head_mask_sha256", "development_counts"):
        if a.get(key) != b.get(key):
            raise ReportFailure("COMMON_POPULATION_MISMATCH", key)
    try:
        delta = M.matched_capacity_delta(a.get("capacity"), b.get("capacity"))
    except M.CapacityError as exc:
        raise ReportFailure(exc.code, str(exc)) from None
    per_head = {str(h): {"mg1": (mg1_metrics["per_horizon"].get(str(h)) or {}).get("relative_error"),
                         "mg2m": (mg2_metrics["per_horizon"].get(str(h)) or {}).get("relative_error")}
                for h in a["selection"]["heads"]}
    doc = {"schema": "i6dmo.matched_decision.v1", "capacity_delta": delta, "per_head": per_head,
           "scores": {"mg1": a["selection"]["score"], "mg2m": b["selection"]["score"]},
           "tie_tolerance": SELECTION_TIE,
           "note": "one seed establishes a recipe decision, not statistical superiority; an "
                   "aggregate never makes a failed head trading-eligible"}
    missing = [dict(f, arm=arm) for arm, sel in (("mg1", a["selection"]), ("mg2m", b["selection"]))
               for f in sel.get("failed_heads", [])]
    if missing or a["selection"]["status"] != "COMPLETE" or b["selection"]["status"] != "COMPLETE":
        return dict(doc, decision="INCOMPLETE_COMPARISON", selected=None, missing=missing)
    diff = a["selection"]["score"] - b["selection"]["score"]
    if abs(diff) <= SELECTION_TIE:
        return dict(doc, decision="TIE_RETAIN_CHEAPER", selected="mg2m", difference=diff)
    return dict(doc, decision="LOWER_SCORE", selected="mg1" if diff < 0 else "mg2m", difference=diff)
