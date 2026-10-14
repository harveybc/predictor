"""DMO08/DMO09 at integration level: the report unit (tools.i6dmo.reporting) on the data lane's
paired_metrics/weekly_diagnostics and the model lane's predictions/validation.npz.

Synthetic, CPU only: a tiny INPUT.npz, a hand-built data.SplitArrays and a fake stored fit
terminal. Independent recomputation here never calls the code under test.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from tests.i6dmo.ops_fakes import load_design

H = (2, 3, 4, 5, 6, 12)
T0 = int(datetime(2024, 1, 8, 12, tzinfo=timezone.utc).timestamp())  # a Monday noon, VALIDATION year
HOUR = 3600


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _price(t):
    k = (t - T0) // HOUR
    return 1.1 + 0.0007 * math.sin(k / 3.0) + 0.00001 * k


def _rows():
    """W1: one origin; W2: three origins; W3: expected but absent. Some heads invalid."""
    origins = [(T0, "W1"), (T0 + 7 * 24 * HOUR, "W2"), (T0 + 7 * 24 * HOUR + 5 * HOUR, "W2"),
               (T0 + 7 * 24 * HOUR + 30 * HOUR, "W2")]
    mask = np.ones((len(origins), len(H)), dtype=bool)
    mask[1, 5] = False  # h12 not supported for that origin
    mask[3, 0] = False
    return origins, mask


def _arrays(split="VALIDATION"):
    from tools.i6dmo import data
    origins, mask = _rows()
    ts = np.asarray([t for t, _ in origins], dtype=np.int64)
    y = np.full(mask.shape, np.nan)
    for i, t in enumerate(ts):
        for j, h in enumerate(H):
            if mask[i, j]:
                y[i, j] = _price(int(t) + h * HOUR)
    n = len(ts)
    return data.SplitArrays(
        split=split, horizons=H, features=tuple(f"f{i}" for i in range(19)), X=np.zeros((n, 13, 19)),
        staleness_s=np.zeros((n, 13, 19), dtype=np.int64), y_price=y, mask=mask, entry_mask=mask.copy(),
        price_origin=np.asarray([_price(int(t)) for t in ts]), origin_ts=ts,
        row_ids=np.asarray([data.row_id(int(t)) for t in ts], dtype=str),
        week_ids=np.asarray([w for _, w in origins], dtype=str), expected_weeks=("W1", "W2", "W3"),
        calendar_id="FAKE_ASSUMED_CALENDAR")


def _predictions(arrays, *, shuffle=False, horizons=H):
    rng = np.random.default_rng(6102026)
    pred = arrays.price_origin[:, None] + rng.normal(0, 0.0004, size=arrays.mask.shape)
    order = np.arange(len(arrays.origin_ts))[::-1] if shuffle else np.arange(len(arrays.origin_ts))
    return dict(row_id=arrays.origin_ts[order], horizons_hours=np.asarray(horizons),
                origin_price=arrays.price_origin[order], predicted_price=pred[order],
                target_price=arrays.y_price[order], mask=arrays.mask[order]), pred


def _setup(tmp_path, *, shuffle=False, horizons=H, bundle_sha="3e42c3f5" + "0" * 56):
    from tools.i6dmo import provenance
    arrays = _arrays()
    inp = tmp_path / "input"
    inp.mkdir()
    grid = np.arange(T0 - 48 * HOUR, T0 + 30 * 24 * HOUR, HOUR, dtype=np.int64)
    np.savez(inp / "INPUT.npz", ts=grid, price=np.asarray([_price(int(t)) for t in grid]), x=np.zeros((len(grid), 19)))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    counts = {"origins": {"candidates": 9, "input_ok": 4, "INPUT_NO_OBSERVATION_AVAILABLE": 5},
              "heads": {str(h): {"label_valid": int(arrays.mask[:, j].sum()), "entry_valid": int(arrays.mask[:, j].sum()),
                                 "reasons": {"TARGET_AFTER_FORCE_FLAT": 1} if h in (2, 12) else {}}
                        for j, h in enumerate(H)}}
    manifest = dict(bundle_sha256=bundle_sha, input_sampling=dict(rule="DS2_ASOF_LAST_AVAILABLE"),
                    horizons_hours=list(H), splits=dict(validation=dict(
                        row_ids_sha256=provenance.canonical_sha256(arrays.row_ids.tolist()),
                        n_rows=len(arrays.row_ids), counts=counts,
                        head_valid={str(h): int(arrays.mask[:, j].sum()) for j, h in enumerate(H)},
                        head_entry_valid={str(h): int(arrays.mask[:, j].sum()) for j, h in enumerate(H)})))
    (bundle / "BUNDLE.json").write_text(json.dumps(manifest))
    fit = tmp_path / "terminals" / ("f" * 64)
    (fit / "out" / "train" / "predictions").mkdir(parents=True)
    pred_doc, pred = _predictions(arrays, shuffle=shuffle, horizons=horizons)
    np.savez(fit / "out" / "train" / "predictions" / "validation.npz", **pred_doc)
    psha = _sha(fit / "out" / "train" / "predictions" / "validation.npz")
    stored = dict(schema="i6dmo.stored_terminal.v1", unit_id="short.fit", identity_sha256="f" * 64,
                  terminal=dict(status="COMPLETE", trainer_identity=dict(weights_sha256="w" * 64),
                                artifacts={"predictions/validation.npz": dict(path="train/predictions/validation.npz",
                                                                            sha256=psha)}),
                  stored_at=1760000000.0)
    (fit / "TERMINAL.json").write_text(json.dumps(stored))
    spec = dict(unit_id="short.report", kind="report", family="short", identity_sha256="r" * 64,
                campaign_id="i6dmo-test", cadence="NONE", test_open=False,
                effective_config=dict(design=load_design(), code_commit="c" * 40),
                upstream={"short.fit": dict(path=str(fit / "TERMINAL.json"), sha256=_sha(fit / "TERMINAL.json"))},
                pins={"short_bundle": dict(path=str(bundle), sha256=_sha(bundle / "BUNDLE.json"),
                                           bundle_sha256=bundle_sha),
                      "short_input": dict(path=str(inp), sha256=_sha(inp / "INPUT.npz"))})
    out = tmp_path / "out"
    return spec, out, arrays, pred


def _run(spec, out, arrays, tmp_path):
    from tools.i6dmo import reporting
    return reporting.run_unit(spec, out, tmp_path / "progress.json", arrays_loader=lambda s: arrays)


def test_dmo08_mae_and_naive_on_identical_rows_agree_with_independent_recomputation(tmp_path):
    from tools.i6dmo import outbox
    spec, out, arrays, pred = _setup(tmp_path)
    doc = _run(spec, out, arrays, tmp_path)
    assert doc["status"] == "COMPLETE", doc.get("disposition")
    for j, h in enumerate(H):
        rows = [i for i in range(len(arrays.origin_ts)) if arrays.mask[i, j]]
        mae = math.fsum(abs(pred[i, j] - _price(int(arrays.origin_ts[i]) + h * HOUR)) for i in rows) / len(rows)
        naive = math.fsum(abs(_price(int(arrays.origin_ts[i]) + h * HOUR) - _price(int(arrays.origin_ts[i])))
                          for i in rows) / len(rows)
        e = doc["eligibility"][str(h)]
        assert e["n"] == len(rows)
        assert math.isclose(e["mae"], mae, rel_tol=1e-12) and math.isclose(e["naive"], naive, rel_tol=1e-12)
        assert math.isclose(e["relative_error"], (mae - naive) / naive, rel_tol=1e-9)
        assert e["eligible"] is (mae < naive)
    annual = next(r for r in doc["reports"] if r["experiment_key"].endswith(":annual"))
    by = {(m["metric"], m["horizon"]): m["value"] for m in annual["metrics"]}
    assert math.isclose(by[("Price_MAE", 2)], doc["eligibility"]["2"]["mae"], rel_tol=1e-12)
    assert by[("Support_Valid_Labels", 12)] == int(arrays.mask[:, 5].sum())
    assert by[("Excluded_TARGET_AFTER_FORCE_FLAT", 2)] == 1
    assert annual["tags"]["test_read"] == "false" and annual["tags"]["sampling"] == "DS2_ASOF_LAST_AVAILABLE"
    box = outbox.Outbox(tmp_path / "ob.sqlite3")
    for report in doc["reports"]:
        box.enqueue(report, source_unit="short.report")  # every fact is a valid governed report
    metrics = json.loads((out / "METRICS.json").read_text())
    assert metrics["agreement"]["status"] == "AGREE"
    assert doc["artifacts"]["metrics"]["sha256"] == _sha(out / "METRICS.json")


@pytest.mark.parametrize("kind,code", [("shuffle", "ROW_ID_MISMATCH"), ("horizons", "HORIZON_MISMATCH")])
def test_dmo08_misaligned_predictions_are_a_typed_failure(tmp_path, kind, code):
    if kind == "shuffle":
        spec, out, arrays, _ = _setup(tmp_path, shuffle=True)
    else:
        spec, out, arrays, _ = _setup(tmp_path, horizons=(2, 3, 4, 5, 12, 6))
    doc = _run(spec, out, arrays, tmp_path)
    assert doc["status"] == "FAILED" and doc["disposition"] == code and not doc["reports"]


def test_dmo08_superseded_or_unpinned_bundle_is_refused(tmp_path):
    spec, out, arrays, _ = _setup(tmp_path, bundle_sha="5221615d" + "0" * 56)
    doc = _run(spec, out, arrays, tmp_path)
    assert doc["status"] == "FAILED" and doc["disposition"] == "SUPERSEDED_BUNDLE"
    (tmp_path / "b").mkdir()
    spec, out, arrays, _ = _setup(tmp_path / "b")
    spec["pins"]["short_bundle"]["sha256"] = "0" * 64
    assert _run(spec, out, arrays, tmp_path / "b")["disposition"] == "BUNDLE_PIN_MISMATCH"


def test_dmo08_init_refuses_superseded_or_exact_slot_bundle_pins(tmp_path):
    from tests.i6dmo.ops_fakes import make_campaign
    from tools.i6dmo import campaign
    good = {"relpath": "b", "sha256": "1" * 64, "bundle_sha256": "3e42c3f5" + "0" * 56,
            "input_sampling": "DS2_ASOF_LAST_AVAILABLE"}
    make_campaign(tmp_path, name="ok", data_pins={"short_bundle": good})
    with pytest.raises(campaign.Refused, match="SUPERSEDED_BUNDLE"):
        make_campaign(tmp_path, name="old", data_pins={"short_bundle": dict(good, bundle_sha256="5221615d" + "0" * 56)})
    with pytest.raises(campaign.Refused, match="BUNDLE_SAMPLING"):
        make_campaign(tmp_path, name="slot", data_pins={"short_bundle": dict(good, input_sampling="EXACT_SLOT")})


def test_dmo09_unequal_weeks_pool_by_sums_and_absent_weeks_are_explicit(tmp_path):
    spec, out, arrays, pred = _setup(tmp_path)
    doc = _run(spec, out, arrays, tmp_path)
    weekly = json.loads((out / "METRICS.json").read_text())["weekly"]["per_horizon"]["3"]
    j = H.index(3)
    errs = {w: [abs(pred[i, j] - arrays.y_price[i, j]) for i in range(len(arrays.origin_ts))
                if arrays.week_ids[i] == w and arrays.mask[i, j]] for w in ("W1", "W2")}
    pooled = math.fsum(errs["W1"] + errs["W2"]) / (len(errs["W1"]) + len(errs["W2"]))
    mean_of_weeks = (math.fsum(errs["W1"]) / len(errs["W1"]) + math.fsum(errs["W2"]) / len(errs["W2"])) / 2
    assert not math.isclose(pooled, mean_of_weeks, rel_tol=1e-6)  # the fixture really has unequal weeks
    assert math.isclose(weekly["annual"]["mae"], pooled, rel_tol=1e-12)
    assert math.isclose(doc["eligibility"]["3"]["mae"], pooled, rel_tol=1e-12)
    assert weekly["weeks"]["W3"] == {"status": "ABSENT", "n": 0} and weekly["weeks_absent"] == 1
    week_reports = {r["tags"]["week_id"]: r for r in doc["reports"] if ":week:" in r["experiment_key"]}
    assert set(week_reports) == {"W1", "W2", "W3"}
    w3 = {(m["metric"], m["horizon"]): m["value"] for m in week_reports["W3"]["metrics"]}
    assert w3[("Week_N", 3)] == 0 and ("Week_MAE", 3) not in w3
    assert week_reports["W3"]["tags"]["h3"] == "ABSENT"
    w2 = {(m["metric"], m["horizon"]): m["value"] for m in week_reports["W2"]["metrics"]}
    assert w2[("Week_N", 3)] == 3 and math.isclose(w2[("Week_MAE", 3)], math.fsum(errs["W2"]) / 3, rel_tol=1e-12)


def test_report_preconditions_wait_for_pins(tmp_path):
    from tools.i6dmo import reporting
    ctx = dict(unit=dict(family="short", kind="report"), effective_config=dict(data_pins={}), data_root=str(tmp_path))
    assert reporting.preconditions(ctx).startswith("BUNDLE_PIN_MISSING:short_bundle")
    ctx["effective_config"]["data_pins"] = {"short_bundle": {"relpath": "b", "sha256": "1" * 64}}
    assert reporting.preconditions(ctx).startswith("INPUT_PIN_MISSING:short_input")
