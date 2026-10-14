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
from datetime import datetime, timezone
from pathlib import Path

SAMPLING_RULE = "DS2_EXACT"
SAMPLING_BY_FAMILY = {"short": "DS2_EXACT", "long": "DS12_EXACT"}
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


def _check_bundle(pin, family="short"):
    manifest_path = Path(pin["path"]) / "BUNDLE.json"
    if not manifest_path.exists() or _sha(manifest_path) != pin["sha256"]:
        raise ReportFailure("BUNDLE_PIN_MISMATCH", "BUNDLE.json digest differs from the pin")
    manifest = json.loads(manifest_path.read_text())
    identity = str(manifest.get("bundle_sha256", ""))
    if identity.startswith(SUPERSEDED_BUNDLE_PREFIXES):
        raise ReportFailure("SUPERSEDED_BUNDLE", identity[:8])
    if pin.get("bundle_sha256") and pin["bundle_sha256"] != identity:
        raise ReportFailure("BUNDLE_PIN_MISMATCH", "bundle_sha256 differs from the pin")
    if (manifest.get("input_sampling") or {}).get("rule") != SAMPLING_BY_FAMILY.get(family, SAMPLING_RULE):
        raise ReportFailure("BUNDLE_SAMPLING", str((manifest.get("input_sampling") or {}).get("rule")))
    return manifest


def materialize_validation(spec):
    """Default loader: re-materialize VALIDATION through the data lane's public API."""
    from tools.i6dmo import data
    design = _design(spec)
    bundle = data.load_input(spec["pins"][f"{spec['family']}_input"]["path"], design)
    table = data.build_origin_table(bundle, design)
    return data.materialize(bundle, table, "VALIDATION")


def pred_doc_path(spec):
    up = spec["upstream"][f"{spec['family']}.fit"]
    stored = json.loads(Path(up["path"]).read_text())
    art = stored["terminal"]["artifacts"]["predictions/validation.npz"]
    return Path(up["path"]).parent / "out" / art["path"]


def _predictions(spec):
    up = spec["upstream"].get(f"{spec['family']}.fit") or {}
    path = Path(up.get("path", ""))
    if not path.exists() or _sha(path) != up.get("sha256"):
        raise ReportFailure("UPSTREAM_TERMINAL_DIGEST", f"{spec['family']}.fit")
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
                 "sampling": SAMPLING_BY_FAMILY.get(spec["family"], SAMPLING_RULE), "bundle_sha256": str(manifest.get("bundle_sha256")),
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
    from tools.i6dmo import naive
    try:
        family = spec["family"]
        pins = spec.get("pins") or {}
        if f"{family}_bundle" not in pins or f"{family}_input" not in pins:
            raise ReportFailure("PIN_MISSING", "bundle/input pins absent")
        manifest = _check_bundle(pins[f"{family}_bundle"], family)
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
