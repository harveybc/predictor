"""Continuation close hook: synthetic 2024 arrays only, no fitting or services."""
import hashlib
import json

import numpy as np
import pytest

from tools.i6dmo import continuation_strategy as CS, strategy_adapter as SA


HEADS = {"short": [2, 3, 4, 5, 6, 12], "long": [18, 24, 36, 48, 60, 72]}
T0 = 1704708000  # Monday 2024-01-08 10:00 UTC


def selection():
    return {g: dict(unit_ids=[g], cadence="static", metrics=[
        dict(horizon=h, n=2, mae=.0005, naive_mae=.001, relative_error=-.5,
             pairs_sha256=SA._canonical_sha([T0, T0 + 3600])) for h in hs]) for g, hs in HEADS.items()}


def arrays(root, unit, horizons, *, ts=None, mask=None, prediction=None, origin=None):
    ts = np.array([T0, T0 + 3600], dtype=np.int64) if ts is None else np.asarray(ts)
    folder = root / "results" / unit
    folder.mkdir(parents=True, exist_ok=True)
    target = np.full((len(ts), len(horizons)), 1.101)
    np.savez(folder / "VALIDATION.npz", origin_ts=ts,
             origin_price=np.full(len(ts), 1.1) if origin is None else origin,
             target_price=target, mask=np.ones(target.shape, bool) if mask is None else mask,
             prediction=target - .0005 if prediction is None else prediction)


def pinned_inputs(root):
    folder = root / "inputs"
    folder.mkdir()
    times = np.arange(T0 - 3600, T0 + 2 * 3600, 3600, dtype=np.int64)
    np.savez(folder / "BARS.npz", close_time=times, open=np.full(len(times), 1.1),
             high=np.full(len(times), 1.1001), low=np.full(len(times), 1.0999), close=np.full(len(times), 1.1))
    calendar = dict(entry_delay_after_open_hours=12, force_flat_before_close_hours=12,
        market_calendar=dict(calendar_id="synthetic.assumed_fx", version="v1", weeks=[dict(
            week_id="2024-W02", market_open="2024-01-07T22:00:00Z", market_close="2024-01-12T22:00:00Z",
            entry_cutoff="2024-01-11T05:00:00Z")]))
    (folder / "CALENDAR.json").write_text(json.dumps(calendar))
    pins = {name: dict(relpath=file, sha256=hashlib.sha256((folder / file).read_bytes()).hexdigest())
            for name, file in (("bars", "BARS.npz"), ("calendar", "CALENDAR.json"))}
    return dict(strategy_inputs=dict(root=str(folder), **pins))


def test_negative_pair_never_opens_artifacts_or_prices(tmp_path, monkeypatch):
    chosen = selection()
    chosen["long"]["metrics"][-1].update(mae=.001, relative_error=0.)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("negative pair read arrays"))
    monkeypatch.setattr(SA, "run_backtest", lambda **k: pytest.fail("negative pair backtested"))
    result = CS.finalize({}, chosen, tmp_path)
    assert result["state"] == "COMPLETE"
    assert result["disposition"] == "PAIR_NOT_ELIGIBLE"
    assert result["per_head_eligibility"]["long"]["72"]["eligible"] is False
    assert result["metrics"] == dict(pair_eligible=0, strategy_evaluated=0,
                                    short_eligible_heads=6, long_eligible_heads=5)
    assert result["test_read"] is False and list(tmp_path.iterdir()) == []


def test_positive_missing_predictions_names_real_object_without_private_path(tmp_path):
    result = CS.finalize({}, selection(), tmp_path)
    assert result["status"] == "WAIT_ARTIFACT"
    assert result["state"] == "WAIT_ARTIFACT"
    assert result["missing"] == ["results/short/VALIDATION.npz"]
    assert str(tmp_path) not in json.dumps(result)


@pytest.mark.parametrize("defect,code", [("duplicate", "DUPLICATE_ORIGIN"),
    ("alignment", "PAIR_ORIGIN_MISMATCH"), ("price", "ORIGIN_PRICE_MISMATCH"),
    ("cache", "METRICS_REPLAY_MISMATCH"), ("test", "TEST_FORBIDDEN")])
def test_raw_array_alignment_and_replay_refuse_before_prices(tmp_path, defect, code):
    chosen = selection()
    arrays(tmp_path, "short", HEADS["short"])
    kwargs = {}
    if defect == "alignment":
        kwargs["ts"] = [T0 + 3600, T0 + 7200]
    if defect == "price":
        kwargs["origin"] = [1.1001, 1.1001]
    if defect == "test":
        kwargs["ts"] = [1735689600, 1735693200]
    arrays(tmp_path, "long", HEADS["long"], **kwargs)
    if defect == "duplicate":
        arrays(tmp_path, "long2", HEADS["long"])
        chosen["long"]["unit_ids"].append("long2")
    if defect == "cache":
        chosen["short"]["metrics"][0].update(mae=.0001, relative_error=-.9)
    result = CS.finalize({}, chosen, tmp_path)
    assert result["status"] == "FAILED" and result["code"] == code


def test_adapter_route_ignores_label_mask_for_actors_and_keeps_costs_10k(tmp_path, monkeypatch):
    chosen = selection()
    for group, hs in HEADS.items():
        mask = np.ones((2, 6), bool)
        mask[0, 0] = False
        chosen[group]["metrics"][0]["n"] = 1
        chosen[group]["metrics"][0]["pairs_sha256"] = SA._canonical_sha([T0 + 3600])
        arrays(tmp_path, group, hs, mask=mask)
    spec = pinned_inputs(tmp_path)
    calls = []
    real = SA.run_backtest
    def backtest(**kwargs):
        calls.append(kwargs)
        assert kwargs["allow_test"] is False
        assert kwargs["execution"].initial_equity == 10000.
        assert kwargs["execution"].spread_price == .0002
        assert kwargs["short_predictions"][SA.parse_time("2024-01-08T10:00:00Z")][2] == pytest.approx(1.1005)
        return real(**kwargs)
    monkeypatch.setattr(SA, "run_backtest", backtest)
    result = CS.finalize(spec, chosen, tmp_path)
    assert len(calls) == 1 and result["status"] == "COMPLETE"
    assert result["label"] == "RETROSPECTIVE_DEVELOPMENT_NOT_OOS"
    assert result["disposition"] == "PAIR_ELIGIBLE" and result["state"] == "COMPLETE"
    assert all(isinstance(v, (int, float)) and np.isfinite(v) for v in result["metrics"].values())
    assert result["cost_verification"] == "NOT_BROKER_VERIFIED"
    assert result["calendar_verification"] == "ASSUMED_NOT_BROKER_VERIFIED"
    assert result["backtest"]["references"]["cash"]["final_equity"] == 10000.


def test_all_selected_periods_are_pooled_not_last_period_only(tmp_path, monkeypatch):
    chosen = selection()
    for group, hs in HEADS.items():
        chosen[group].update(unit_ids=[group + "1", group + "2"], cadence="quarterly")
        arrays(tmp_path, group + "1", hs, ts=[T0])
        arrays(tmp_path, group + "2", hs, ts=[T0 + 3600])
    result = CS.finalize({}, chosen, tmp_path)
    assert result["status"] == "WAIT_ARTIFACT"
    assert result["code"] == "STRATEGY_INPUT_MISSING"


def test_incomplete_annual_head_is_ineligible_without_arrays(tmp_path):
    chosen = selection()
    chosen["short"]["metrics"].pop()
    assert CS.finalize({}, chosen, tmp_path)["disposition"] == "PAIR_NOT_ELIGIBLE"


def test_input_pins_cannot_escape_explicit_root(tmp_path):
    for group, hs in HEADS.items():
        arrays(tmp_path, group, hs)
    spec = pinned_inputs(tmp_path)
    spec["strategy_inputs"]["bars"]["relpath"] = "../TEST.npz"
    result = CS.finalize(spec, selection(), tmp_path)
    assert result["status"] == "FAILED" and result["code"] == "INPUT_PATH_INVALID"


def test_main_absolute_pins_and_flat_numeric_metrics(tmp_path):
    for group, hs in HEADS.items():
        arrays(tmp_path, group, hs)
    spec = pinned_inputs(tmp_path)
    inputs = spec["strategy_inputs"]
    spec["strategy_inputs"] = {k: dict(path=str(tmp_path / "inputs" / inputs[k]["relpath"]),
        sha256=inputs[k]["sha256"]) for k in ("bars", "calendar")}
    result = CS.finalize(spec, selection(), tmp_path)
    assert result["state"] == "COMPLETE" and result["disposition"] == "PAIR_ELIGIBLE"
    assert result["metrics"]["initial_equity"] == 10000.
    assert "costs_spread" in result["metrics"]
    assert "sharpe" not in result["metrics"]  # undefined is not invented as zero
    assert result["backtest"]["metrics"]["sharpe_reason"] == "INSUFFICIENT_RETURNS"


@pytest.mark.parametrize("group", ["short", "long"])
def test_main_incomplete_selection_returns_without_any_artifact_read(tmp_path, monkeypatch, group):
    chosen = selection()
    chosen[group]["state"] = "INCOMPLETE"
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("incomplete pair read artifacts"))
    result = CS.finalize({}, chosen, tmp_path)
    assert result["state"] == "COMPLETE" and result["disposition"] == "PAIR_NOT_ELIGIBLE"


def test_actual_pair_support_hash_must_replay(tmp_path):
    chosen = selection()
    chosen["long"]["metrics"][-1]["pairs_sha256"] = "a" * 64
    for group, hs in HEADS.items():
        arrays(tmp_path, group, hs)
    result = CS.finalize({}, chosen, tmp_path)
    assert result["code"] == "PAIRS_REPLAY_MISMATCH"


@pytest.mark.parametrize("key", ["bars", "calendar"])
def test_present_input_bytes_require_exact_digest(tmp_path, key):
    for group, hs in HEADS.items():
        arrays(tmp_path, group, hs)
    spec = pinned_inputs(tmp_path)
    spec["strategy_inputs"][key]["sha256"] = "a" * 64
    result = CS.finalize(spec, selection(), tmp_path)
    assert result["status"] == "FAILED" and result["code"] == "STRATEGY_INPUT_DIGEST_MISMATCH"


def test_masked_target_never_crosses_into_test(tmp_path):
    ts = [1735686000]
    chosen = selection()
    for group, hs in HEADS.items():
        for row in chosen[group]["metrics"]:
            row.update(n=1, pairs_sha256=SA._canonical_sha(ts))
        arrays(tmp_path, group, hs, ts=ts)
    result = CS.finalize({}, chosen, tmp_path)
    assert result["code"] == "TEST_FORBIDDEN"


def test_calendar_must_cover_selected_origins_before_adapter(tmp_path, monkeypatch):
    for group, hs in HEADS.items():
        arrays(tmp_path, group, hs)
    spec = pinned_inputs(tmp_path)
    path = tmp_path / "inputs" / "CALENDAR.json"
    document = json.loads(path.read_text())
    week = document["market_calendar"]["weeks"][0]
    week.update(week_id="2024-W03", market_open="2024-01-14T22:00:00Z",
                market_close="2024-01-19T22:00:00Z", entry_cutoff="2024-01-18T05:00:00Z")
    path.write_text(json.dumps(document))
    spec["strategy_inputs"]["calendar"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(SA, "run_backtest", lambda **k: pytest.fail("calendar mismatch backtested"))
    result = CS.finalize(spec, selection(), tmp_path)
    assert result["code"] == "CALENDAR_SUPPORT_MISMATCH"


def test_nonfinite_annual_head_is_negative_and_serializable_without_reads(tmp_path, monkeypatch):
    chosen = selection()
    chosen["short"]["metrics"][0]["mae"] = float("nan")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("invalid annual head read artifacts"))
    result = CS.finalize({}, chosen, tmp_path)
    assert result["disposition"] == "PAIR_NOT_ELIGIBLE"
    json.dumps(result, allow_nan=False)
