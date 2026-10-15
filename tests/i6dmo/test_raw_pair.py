"""RAW pair contract: authenticated lineage, frozen head gate, no TEST or fits.

Run with --noconftest: the parent conftest imports TensorFlow at collection.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from tools.i6dmo import campaign as C, matched_runner as MR
from tools.i6dmo import raw_pair as RP


def write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, sort_keys=True))
    return dict(path=str(path), sha256=MR._sha(path))


def store(root, unit, family, artifacts, identity_upstream=(), **extra):
    identity = dict(unit=unit, family=family, kind="comparison" if unit == "comparison" else "report",
                    cadence="NONE", effective_config_sha256="e" * 64, data_sha256="d" * 64,
                    code_commit="c" * 40, seed=6102026, upstream=sorted(identity_upstream))
    digest = C.unit_identity(identity)
    folder = root / "terminals" / digest
    arts = {}
    for name, (filename, doc) in artifacts.items():
        if isinstance(doc, bytes):
            path = folder / "out" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(doc)
            pin = dict(sha256=MR._sha(path))
        else:
            pin = write(folder / "out" / filename, doc)
        arts[name] = dict(path=filename, sha256=pin["sha256"])
    child = dict(schema="i6dmo.terminal.v1", unit_id=unit, identity_sha256=digest,
                 status="COMPLETE", artifacts=arts, **extra)
    cp = write(folder / "CHILD_TERMINAL.json", child)
    return write(folder / "TERMINAL.json", dict(schema="i6dmo.stored_terminal.v1", unit_id=unit,
        identity=identity, identity_sha256=digest, child_terminal_sha256=cp["sha256"], terminal=child))


def report(root, arm, better, prediction_arrays=None):
    hs = MR.HORIZONS if arm == "mg2m" else MR.LONG_HORIZONS
    rows = {str(h): dict(n=2, n_declared=2, n_failed=0, state="COMPLETE", mae=.5 if h in better else 1.5,
        naive_mae=1., relative_error=-.5 if h in better else .5,
        eligible_mae_below_naive=h in better, row_ids_sha256="a" * 64) for h in hs}
    prediction_bytes = b"synthetic prediction bytes"
    if prediction_arrays is not None:
        from tools.i6dmo import data, provenance, reporting
        buffer = io.BytesIO()
        np.savez(buffer, **prediction_arrays)
        prediction_bytes = buffer.getvalue()
        replay = reporting.manual_pooled_replay(prediction_arrays, hs)
        for j, (h, row) in enumerate(rows.items()):
            row.update(replay[h], n_declared=replay[h]["n"])
            row["eligible_mae_below_naive"] = row["mae"] < row["naive_mae"]
            row["row_ids_sha256"] = provenance.canonical_sha256([
                data.row_id(int(t)) for t in prediction_arrays["row_id"][prediction_arrays["mask"][:, j]]])
    from tools.i6dmo import reporting
    for row in rows.values():
        row.update(sum_abs_error=row["mae"] * row["n"], naive_sum_abs_error=row["naive_mae"] * row["n"],
                   relative_error_undefined_reason=None, c2_row_keys_sha256="4" * 64)
    selection = reporting.selection_score(rows, hs)
    matched = dict(schema="i6dmo.matched_report.v1", arm=arm, population="COMMON_DEVELOPMENT",
        common_population_sha256=("s" if arm == "mg2m" else "l") * 64,
        development_row_ids_sha256="r" * 64, development_head_mask_sha256="m" * 64,
        development_counts={h: r["n_declared"] for h, r in rows.items()}, scientific_fit=True, fit_device="gpu",
        fit_identity={"retained": arm}, selection=selection,
        manual_pooled_replay={h: {k: r[k] for k in ("n", "mae", "naive_mae", "relative_error")}
                             for h, r in rows.items()})
    pred_hash = hashlib.sha256(prediction_bytes).hexdigest()
    c2 = dict(state="COMPLETE", sha256="1" * 64, contract_sha256="2" * 64, input_sha256="3" * 64)
    metrics = dict(schema="i6dmo.paired_metrics.v1", split="DEVELOPMENT", unit=reporting.UNIT,
                   scale="original_price", agreement=dict(status="AGREE"), c2=c2,
                   per_horizon=rows, matched=matched, predictions_sha256=pred_hash)
    eligibility = {h: dict(n=r["n"], mae=r["mae"], naive=r["naive_mae"],
                          eligible=r["eligible_mae_below_naive"], row_ids_sha256=r["row_ids_sha256"],
                          n_declared=r["n_declared"], n_failed=r["n_failed"], state=r["state"],
                          relative_error=r["relative_error"])
                   for h, r in rows.items()}
    return store(root, arm + ".report", "short" if arm == "mg2m" else "long",
        dict(metrics=("METRICS.json", metrics), validation_predictions=("validation_predictions.npz",
        prediction_bytes)), disposition="MATCHED_PAIRED_METRICS_AGREE", eligibility=eligibility,
        lineage=dict(common_population_sha256=matched["common_population_sha256"],
                     predictions_sha256=pred_hash, arm=arm, checkpoint="original-fit", c2=c2),
        selection=selection)


def replace_artifact(pin, name, change):
    path = Path(pin["path"])
    stored = json.loads(path.read_text())
    art = stored["terminal"]["artifacts"][name]
    ap = path.parent / "out" / art["path"]
    doc = json.loads(ap.read_text())
    change(doc)
    art["sha256"] = write(ap, doc)["sha256"]
    stored["child_terminal_sha256"] = write(path.parent / "CHILD_TERMINAL.json", stored["terminal"])["sha256"]
    pin.update(write(path, stored))


def replace_child(pin, change):
    path = Path(pin["path"])
    stored = json.loads(path.read_text())
    change(stored["terminal"])
    stored["child_terminal_sha256"] = write(path.parent / "CHILD_TERMINAL.json", stored["terminal"])["sha256"]
    pin.update(write(path, stored))


@pytest.fixture
def env(tmp_path):
    short = dict(report(tmp_path / "closed", "mg2m", set()), unit_id="mg2m.report")
    ups = {a + ".report": report(tmp_path / "long", a, {18, 24} if a == "mg12m" else {18})
           for a in MR.LONG_ARMS}
    metrics, records = {}, {}
    for a in MR.LONG_ARMS:
        metrics[a], records[a + ".report"] = MR._authenticated_report(a + ".report", ups[a + ".report"], "l" * 64)
    per_head = {str(h): {a: metrics[a]["per_horizon"][str(h)]["relative_error"] for a in MR.LONG_ARMS}
                for h in MR.LONG_HORIZONS}
    verdict = dict(schema="i6dmo.matched_decision.v1", decision="LOWER_SCORE", selected="mg12m",
                   missing=[], scores={a: metrics[a]["matched"]["selection"]["score"] for a in MR.LONG_ARMS},
                   per_head=per_head)
    doc = dict(schema="i6dmo.matched_comparison.v1", disposition="MG12_SELECTED", missing=[],
               rule="reporting.decide_matched", verdict=verdict, inputs=records,
               common_population_sha256="l" * 64, split="DEVELOPMENT", test_read=False)
    ups["comparison"] = store(tmp_path / "long", "comparison", "comparison",
        dict(comparison=("COMPARISON.json", doc)), disposition="MG12_SELECTED",
        identity_upstream=[r["identity_sha256"] for r in records.values()])
    spec = dict(kind="evaluation", unit_id="raw.pair", campaign_id="synthetic", identity_sha256="f" * 64,
        test_open=False, upstream=ups, effective_config=dict(mode="MATCHED_LONG_CONTRAST", code_commit="c" * 40,
        data_pins={"common_population": {"population_sha256": "l" * 64}}, raw_pair=dict(short_report=short,
        policy=dict(min_valid_long_heads=2, min_valid_short_heads=1))))
    return spec, tmp_path / "out", tmp_path / "progress.json"


def run(env):
    return RP.run_unit(*env)


def test_negative_pair_is_complete_reported_and_reads_no_data(env, monkeypatch):
    original = Path.open
    def guard(path, *args, **kw):
        assert path.suffix != ".npz", "negative gate must not read predictions or bars"
        return original(path, *args, **kw)
    monkeypatch.setattr(Path, "open", guard)
    for name in ("_bars", "_family_predictions", "run_backtest", "load_pinned_policy"):
        monkeypatch.setattr(RP.SA, name, lambda *a, **kw: pytest.fail("negative gate read data or policy"))
    monkeypatch.setattr(RP.SA.c1, "load_calendar", lambda *a, **kw: pytest.fail("negative calendar read"))
    terminal = run(env)
    assert (terminal["status"], terminal["disposition"]) == ("COMPLETE", "PAIR_NOT_ELIGIBLE")
    decision = json.loads((env[1] / "ELIGIBILITY.json").read_text())
    assert decision["eligible"] == {"short": [], "long": [18, 24]}
    assert len(decision["head_status"]["short"]) == 6
    assert decision["inputs"]["mg2m.report"]["lineage"]["checkpoint"] == "original-fit"
    assert terminal["reports"] and terminal["gates"]["test_read"] is False


@pytest.mark.parametrize("target", ["short", "mg1l.report", "comparison"])
def test_forged_stored_terminal_is_refused(env, target):
    spec = env[0]
    pin = spec["effective_config"]["raw_pair"]["short_report"] if target == "short" else spec["upstream"][target]
    path = Path(pin["path"])
    path.write_text(path.read_text() + " ")
    assert run(env)["status"] == "FAILED"


def test_absent_comparison_does_not_infer_a_winner(env):
    del env[0]["upstream"]["comparison"]
    result = run(env)
    assert (result["status"], result["disposition"]) == ("FAILED", "RAW_PAIR_MISSING_ARTIFACT")
    assert "upstream.comparison" in result["gates"]["required_objects"]


@pytest.mark.parametrize("change", [
    lambda d: d.update(disposition="MG1_LONG_SELECTED"),
    lambda d: d["verdict"].update(selected="mg1l"),
    lambda d: d.update(schema="unofficial"),
    lambda d: d.update(test_read=True),
    lambda d: d["inputs"]["mg12m.report"].update(metrics_sha256="0" * 64),
])
def test_contradictory_comparison_is_refused(env, change):
    replace_artifact(env[0]["upstream"]["comparison"], "comparison", change)
    assert run(env)["status"] == "FAILED"


@pytest.mark.parametrize("change", [
    lambda d: d["per_horizon"].update({"72": d["per_horizon"].pop("2")}),
    lambda d: d["per_horizon"]["2"].update(naive_mae=99.),
    lambda d: d["matched"]["manual_pooled_replay"]["2"].update(naive_mae=99.),
    lambda d: d.update(predictions_sha256="0" * 64),
])
def test_wrong_short_heads_naive_or_prediction_binding_refused(env, change):
    replace_artifact(env[0]["effective_config"]["raw_pair"]["short_report"], "metrics", change)
    assert run(env)["status"] == "FAILED"


def test_repinning_forged_identity_does_not_authenticate_code(env):
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    path = Path(pin["path"])
    stored = json.loads(path.read_text())
    stored["identity"]["code_commit"] = "0" * 40
    pin.update(write(path, stored))
    assert run(env)["status"] == "FAILED"


def test_eligible_minimum_not_all_six_and_exact_missing_pins(env, tmp_path):
    env[0]["effective_config"]["raw_pair"]["short_report"] = dict(
        report(tmp_path / "eligible-short", "mg2m", {2}), unit_id="mg2m.report")
    result = run(env)
    assert (result["status"], result["disposition"]) == ("FAILED", "RAW_PAIR_MISSING_ARTIFACT"), result
    assert result["gates"]["pair_status"] == "ELIGIBLE"
    assert set(result["gates"]["required_objects"]) == {
        "effective_config.raw_pair.strategy_bars", "effective_config.raw_pair.strategy_calendar",
        "effective_config.raw_pair.strategy_execution", "effective_config.design.splits.validation"}
    assert result["reports"]


def test_test_open_refused_before_upstream_access(env):
    env[0].update(test_open=True, upstream={})
    assert run(env)["disposition"] == "TEST_OPEN_REFUSED"


def _execution_fixture(env, tmp_path):
    """Actual arrays, calendar and existing versioned instrument; no fit or model stub."""
    from datetime import datetime, timedelta, timezone
    from tools.i6dmo import strategy_adapter as sa
    spec = env[0]
    start = datetime(2024, 1, 7, 22, tzinfo=timezone.utc)
    end = start + timedelta(hours=120)
    times = np.array([int((start + timedelta(hours=i)).timestamp()) for i in range(1, 121)])
    for arm, better in (("mg2m", {2}), ("mg1l", {18}), ("mg12m", {18, 24})):
        hs = MR.HORIZONS if arm == "mg2m" else MR.LONG_HORIZONS
        origin = np.full(len(times), 1.1)
        target = np.full((len(times), 6), 1.101)
        predicted = target + np.array([.0005 if h in better else .0015 for h in hs])
        arrays = dict(row_id=times, origin_price=origin, target_price=target,
                      predicted_price=predicted, mask=np.ones_like(target, dtype=bool),
                      horizons_hours=np.array(hs))
        pin = report(tmp_path / "array-reports", arm, better, arrays)
        if arm == "mg2m":
            spec["effective_config"]["raw_pair"]["short_report"] = dict(pin, unit_id="mg2m.report")
        else:
            spec["upstream"][arm + ".report"] = pin
    # Build the persisted comparison contract without decide_matched's TensorFlow import.
    def update(doc):
        scores = {}
        for arm in MR.LONG_ARMS:
            metrics, record = MR._authenticated_report(arm + ".report", spec["upstream"][arm + ".report"], "l" * 64)
            doc["inputs"][arm + ".report"] = record
            scores[arm] = metrics["matched"]["selection"]["score"]
            for h, row in metrics["per_horizon"].items():
                doc["verdict"]["per_head"][h][arm] = row["relative_error"]
        doc["verdict"]["scores"] = scores
    replace_artifact(spec["upstream"]["comparison"], "comparison", update)
    bars = tmp_path / "bars.npz"
    np.savez(bars, close_time=times, open=np.full(len(times), 1.1), close=np.full(len(times), 1.1),
             high=np.full(len(times), 1.10005), low=np.full(len(times), 1.09995))
    calendar = write(tmp_path / "calendar.json", dict(market_calendar=dict(
        calendar_id="synthetic.assumed_fx.test", version="v1", weeks=[dict(week_id="2024-W02",
        market_open=start.isoformat(), market_close=end.isoformat(),
        entry_cutoff=(end - timedelta(hours=41)).isoformat())]),
        entry_delay_after_open_hours=12, force_flat_before_close_hours=12))
    raw = spec["effective_config"]["raw_pair"]
    raw.update(strategy_bars=dict(path=str(bars), sha256=MR._sha(bars)), strategy_calendar=calendar,
               strategy_execution=dict(instrument_file=str(sa.DEFAULT_INSTRUMENT_FILE)))
    spec["effective_config"]["design"] = dict(splits=dict(validation=dict(
        start="2024-01-01T00:00:00Z", end="2025-01-01T00:00:00Z")))


def test_accepted_arrays_run_existing_pinned_backtest(env, tmp_path):
    _execution_fixture(env, tmp_path)
    result = run(env)
    assert result["status"] == "COMPLETE", result
    assert result["disposition"] == "PASS", result
    assert result["gates"]["pair_status"] == "PASS"
    backtest = json.loads((env[1] / "BACKTEST.json").read_text())
    assert backtest["test_read"] is False
    assert backtest["selected_long"] == "mg12m"
    assert backtest["inputs"]["mg2m.report"]["prediction_bytes_verified"] is True
    assert backtest["metrics"]["closed_trades"] > 0
    assert len(result["reports"]) == 2


def test_eligible_tampered_prediction_bytes_never_backtest(env, tmp_path, monkeypatch):
    _execution_fixture(env, tmp_path)
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    path = Path(pin["path"]).parent / "out" / "validation_predictions.npz"
    path.write_bytes(b"tampered")
    monkeypatch.setattr(RP.SA, "run_backtest", lambda **kw: pytest.fail("must not backtest"))
    result = run(env)
    assert result["status"] == "FAILED"
    assert "PREDICTION_DIGEST" in result["disposition"]


@pytest.mark.parametrize("key", ["strategy_bars", "strategy_calendar"])
def test_eligible_changed_source_pin_refused(env, tmp_path, key):
    _execution_fixture(env, tmp_path)
    env[0]["effective_config"]["raw_pair"][key]["sha256"] = "0" * 64
    assert run(env)["disposition"] == "INPUT_DIGEST_MISMATCH:" + key


def test_child_terminal_tampering_is_refused(env):
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    (Path(pin["path"]).parent / "CHILD_TERMINAL.json").write_text("{}")
    assert "CHILD_TERMINAL" in run(env)["disposition"]


def test_long_support_contradiction_is_refused(env):
    replace_artifact(env[0]["upstream"]["mg1l.report"], "metrics",
                     lambda d: d["matched"].update(development_head_mask_sha256="x" * 64))
    assert run(env)["status"] == "FAILED"


def test_positive_test_window_refused_before_prediction_read(env, tmp_path, monkeypatch):
    _execution_fixture(env, tmp_path)
    env[0]["effective_config"]["design"]["splits"]["validation"]["end"] = "2026-01-01T00:00:00Z"
    monkeypatch.setattr(RP, "_predictions", lambda *a: pytest.fail("TEST window must be refused first"))
    assert run(env)["disposition"] == "TEST_FORBIDDEN"


def test_positive_head_population_digest_replayed(env, tmp_path):
    _execution_fixture(env, tmp_path)
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    path = Path(pin["path"])
    stored = json.loads(path.read_text())
    stored["terminal"]["eligibility"]["2"]["row_ids_sha256"] = "0" * 64
    stored["child_terminal_sha256"] = write(path.parent / "CHILD_TERMINAL.json", stored["terminal"])["sha256"]
    pin.update(write(path, stored))
    replace_artifact(pin, "metrics", lambda d: d["per_horizon"]["2"].update(row_ids_sha256="0" * 64))
    assert "PREDICTION_HEAD_POPULATION" in run(env)["disposition"]


@pytest.mark.parametrize("positive", [False, True])
def test_bad_short_c2_cannot_admit_or_execute_pair(env, tmp_path, positive):
    if positive:
        _execution_fixture(env, tmp_path)
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    def corrupt(metrics):
        metrics["c2"]["state"] = "INCOMPLETE"
        metrics["agreement"]["status"] = "DISAGREE"
    replace_artifact(pin, "metrics", corrupt)
    result = run(env)
    assert result["status"] == "FAILED", result
    assert not (env[1] / "BACKTEST.json").exists()
    assert "pair_status" not in result["gates"]


@pytest.mark.parametrize("positive", [False, True])
def test_incomplete_short_score_is_technical_failure_before_eligibility(env, tmp_path, positive):
    if positive:
        _execution_fixture(env, tmp_path)
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    def corrupt(metrics):
        row = metrics["per_horizon"]["2"]
        row.update(n_declared=row["n"] + 1, n_failed=1, state="INCOMPLETE")
        metrics["matched"]["development_counts"]["2"] = row["n_declared"]
        from tools.i6dmo import reporting
        metrics["matched"]["selection"] = reporting.selection_score(metrics["per_horizon"], MR.HORIZONS)
    replace_artifact(pin, "metrics", corrupt)
    result = run(env)
    assert result["status"] == "FAILED", result
    assert "pair_status" not in result["gates"]
    assert not (env[1] / "BACKTEST.json").exists()


@pytest.mark.parametrize("change", [
    lambda m: m.update(schema="unofficial"),
    lambda m: m.update(split="TEST"),
    lambda m: m.update(scale="normalized"),
    lambda m: m.update(unit="ratio"),
    lambda m: m["agreement"].update(status="DISAGREE"),
    lambda m: m["c2"].update(state="INCOMPLETE"),
    lambda m: m["c2"].update(sha256="0" * 64),
    lambda m: m["matched"]["selection"].update(status="INCOMPLETE", score=None),
    lambda m: m["matched"]["selection"].update(score=-99.),
    lambda m: m["per_horizon"]["2"].update(state="INCOMPLETE"),
])
def test_short_original_metadata_and_cached_selection_are_validated(env, change):
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    replace_artifact(pin, "metrics", change)
    assert run(env)["status"] == "FAILED"


@pytest.mark.parametrize("change", [
    lambda child: child["lineage"].pop("c2"),
    lambda child: child["lineage"]["c2"].update(input_sha256="0" * 64),
    lambda child: child["eligibility"]["2"].update(n_declared=3),
    lambda child: child["eligibility"]["2"].update(n_failed=1),
    lambda child: child["eligibility"]["2"].update(state="INCOMPLETE"),
    lambda child: child["selection"].update(score=-99.),
])
def test_short_child_support_c2_and_cached_selection_are_validated(env, change):
    pin = env[0]["effective_config"]["raw_pair"]["short_report"]
    replace_child(pin, change)
    assert run(env)["status"] == "FAILED"


@pytest.mark.parametrize("change,code", [
    (lambda metrics: metrics["per_horizon"].pop("72"), "MATCHED_HORIZONS"),
    (lambda metrics: metrics["matched"]["selection"].update(score=-99.), "MATCHED_SELECTION_MISMATCH"),
])
def test_raw_pair_reuses_strict_long_report_validation(env, change, code):
    pin = env[0]["upstream"]["mg1l.report"]
    replace_artifact(pin, "metrics", change)
    result = run(env)
    assert result["status"] == "FAILED"
    assert result["disposition"].startswith(code + ":mg1l.report:")
    assert "pair_status" not in result["gates"]


@pytest.mark.parametrize("positive", [False, True])
def test_all_pair_reports_enqueue_without_numeric_metric_rejections(env, tmp_path, positive):
    from tools.i6dmo import worker
    from tools.i6dmo.outbox import Outbox
    if positive:
        _execution_fixture(env, tmp_path)
    terminal = run(env)
    assert terminal["status"] == "COMPLETE", terminal
    root = tmp_path / "governed-outbox"
    queue = C.open_queue(root)
    try:
        rejected = worker._enqueue_reports(root, "raw.pair", terminal, queue)
        assert rejected == 0, queue.events(1000)
        assert not [event for event in queue.events(1000) if event["event"] == "REPORT_REJECTED"]
    finally:
        queue.close()
    box = Outbox(root / "state/outbox.sqlite3")
    try:
        assert len(box.items()) == len(terminal["reports"])
    finally:
        box.close()
    eligible = [metric for report in terminal["reports"] for metric in report["metrics"]
                if metric["metric"] in ("SHORT_eligible", "LONG_eligible")]
    assert len(eligible) == 12
    assert all(type(metric["value"]) is int and metric["value"] in (0, 1) and
               metric["unit"] == "bool" for metric in eligible)
