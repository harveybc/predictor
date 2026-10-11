"""Acceptance tests for the I6-N C1 pure horizon eligibility contract.

Every calendar here is a SYNTHETIC fixture built inside the test. None of
them represents a broker, an exchange or New York hours; they only exercise
the rules of BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md section 2-3 and the
owner order section 4. Nothing is trained and Keras masking is not exercised.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.i6n_contracts import eligibility_cli
from tools.i6n_contracts.eligibility import (
    CalendarRef,
    EligibilityError,
    ErrorCode,
    LabelReason,
    OperationalReason,
    OperationalWeek,
    Role,
    SplitBounds,
    build_support_table,
    evaluate_heads,
    historical_label_eligibility,
    input_admissibility,
    load_calendar,
    operational_eligibility,
)

UTC = timezone.utc
H = timedelta(hours=1)
CAL = CalendarRef("SYNTHETIC_FIXTURE_CALENDAR", "fixture-1")
SHORT = (2, 3, 4, 5, 6, 12)
LONG = (18, 24, 36, 48, 60, 72)
ALL = SHORT + LONG

# Synthetic 120-hour week: O Sunday 22:00Z, C Friday 22:00Z.
O1 = datetime(2026, 1, 4, 22, tzinfo=UTC)
C1 = datetime(2026, 1, 9, 22, tzinfo=UTC)
A1 = O1 + 12 * H                                # 2026-01-05 10:00Z
B1 = C1 - 12 * H                                # 2026-01-09 10:00Z
CUT1 = datetime(2026, 1, 8, 0, tzinfo=UTC)      # fixture "Thursday 00:00"
# Next synthetic week opens one hour earlier so that A2 = Friday 09:00Z + 72h.
O2 = datetime(2026, 1, 11, 21, tzinfo=UTC)
C2 = datetime(2026, 1, 16, 22, tzinfo=UTC)
CUT2 = datetime(2026, 1, 15, 0, tzinfo=UTC)


def _week(o=O1, c=C1, cut=CUT1, wid="W1", **kw):
    return OperationalWeek.derive(
        week_id=wid, calendar=CAL, market_open=o, market_close=c,
        entry_cutoff=cut, entry_delay_after_open_hours=12,
        force_flat_before_close_hours=12, **kw)


W1 = _week()
W2 = _week(O2, C2, CUT2, "W2")


def _hourly(start, end):
    out, t = [], start
    while t < end:
        out.append(t)
        t += H
    return out


def _observed(start=O1 - 48 * H, end=C2 + 48 * H, drop=()):
    """Exact target bars, each available at its own timestamp."""
    drop = set(drop)
    return {t: t for t in _hourly(start, end) if t not in drop}


TRAIN = SplitBounds("TRAIN", datetime(2025, 1, 1, tzinfo=UTC),
                    datetime(2027, 1, 1, tzinfo=UTC), label_available_by=None)


def _iso(t):
    return t.isoformat()


# ---------------------------------------------------------------- inputs


@pytest.mark.parametrize("bad", [True, False, 0, -1, -72, 2.0, "2", None])
def test_horizon_must_be_positive_plain_int(bad):
    with pytest.raises(EligibilityError) as err:
        operational_eligibility(A1, bad, Role.MANAGEMENT, W1)
    assert err.value.code is ErrorCode.INVALID_HORIZON


def test_naive_datetimes_are_rejected_everywhere():
    naive = datetime(2026, 1, 6, 10)
    with pytest.raises(EligibilityError) as err:
        operational_eligibility(naive, 2, Role.MANAGEMENT, W1)
    assert err.value.code is ErrorCode.NAIVE_DATETIME
    with pytest.raises(EligibilityError) as err:
        _week(o=datetime(2026, 1, 4, 22))
    assert err.value.code is ErrorCode.NAIVE_DATETIME
    with pytest.raises(EligibilityError) as err:
        SplitBounds("TRAIN", datetime(2025, 1, 1), datetime(2027, 1, 1, tzinfo=UTC),
                    label_available_by=None)
    assert err.value.code is ErrorCode.NAIVE_DATETIME


def test_role_must_be_typed():
    with pytest.raises(EligibilityError) as err:
        operational_eligibility(A1, 2, "ENTRY", W1)
    assert err.value.code is ErrorCode.INVALID_ROLE


def test_duplicate_heads_are_rejected():
    with pytest.raises(EligibilityError) as err:
        evaluate_heads(A1, (2, 2), Role.MANAGEMENT, W1)
    assert err.value.code is ErrorCode.DUPLICATE_HORIZON


def test_week_offsets_and_order_are_validated():
    assert (W1.operational_start, W1.force_flat) == (A1, B1)
    with pytest.raises(EligibilityError) as err:
        _week(declared_operational_start=A1 + H)
    assert err.value.code is ErrorCode.CALENDAR_OFFSET_MISMATCH
    with pytest.raises(EligibilityError) as err:
        _week(o=C1, c=O1)
    assert err.value.code is ErrorCode.CALENDAR_INVALID
    with pytest.raises(EligibilityError) as err:
        OperationalWeek.derive(
            week_id="W", calendar=CAL, market_open=O1, market_close=C1,
            entry_cutoff=CUT1, entry_delay_after_open_hours=True,
            force_flat_before_close_hours=12)
    assert err.value.code is ErrorCode.OFFSET_INVALID


# ------------------------------------------------- E: operational, known at t


@pytest.mark.parametrize("remaining,expected", [(73, True), (72, False), (48, False)])
def test_h72_against_remaining_hours(remaining, expected):
    origin = B1 - remaining * H
    ok, reasons = operational_eligibility(origin, 72, Role.MANAGEMENT, W1)
    assert ok is expected
    if not expected:
        assert OperationalReason.TARGET_AT_OR_AFTER_FORCE_FLAT in reasons


def test_h72_ineligible_head_never_drops_valid_short_heads():
    origin = B1 - 48 * H
    heads = evaluate_heads(origin, ALL, Role.MANAGEMENT, W1)
    assert [d.horizon_hours for d in heads] == list(ALL)
    eligible = {d.horizon_hours for d in heads if d.operational_eligible}
    assert eligible == set(SHORT) | {18, 24, 36}
    h72 = heads[-1]
    assert h72.target_timestamp == origin + 72 * H      # endpoint never moved
    assert h72.remaining_seconds_to_force_flat == 48 * 3600
    assert h72.eligible_until == B1 - 72 * H
    assert h72.label_evaluated is False and h72.label_eligible is None


def test_origin_equal_to_A_is_allowed_and_equal_to_B_is_not():
    for role in Role:
        assert operational_eligibility(A1, 2, role, W1)[0] is True
        ok, reasons = operational_eligibility(A1 - H, 2, role, W1)
        assert not ok and OperationalReason.ORIGIN_BEFORE_OPERATIONAL_START in reasons
    ok, reasons = operational_eligibility(B1, 2, Role.MANAGEMENT, W1)
    assert not ok and OperationalReason.ORIGIN_AT_OR_AFTER_FORCE_FLAT in reasons
    assert operational_eligibility(B1 - 3 * H, 2, Role.MANAGEMENT, W1)[0] is True


def test_thursday_cutoff_closes_entry_but_not_management():
    t = CUT1 + 2 * H
    ok, reasons = operational_eligibility(t, 2, Role.ENTRY, W1)
    assert not ok and reasons == (OperationalReason.ORIGIN_AT_OR_AFTER_ENTRY_CUTOFF,)
    assert operational_eligibility(t, 2, Role.MANAGEMENT, W1)[0] is True
    assert operational_eligibility(CUT1, 2, Role.ENTRY, W1)[0] is False
    assert operational_eligibility(CUT1 - H, 2, Role.ENTRY, W1)[0] is True


def test_dst_change_uses_elapsed_hours_and_utc_instants():
    # Synthetic fixture: offset changes from -04:00 to -05:00 inside the week.
    edt, est = timezone(timedelta(hours=-4)), timezone(timedelta(hours=-5))
    week = _week(datetime(2026, 10, 25, 17, tzinfo=edt),
                 datetime(2026, 10, 30, 17, tzinfo=est),
                 datetime(2026, 10, 29, 0, tzinfo=edt), "WDST")
    assert week.operational_start == datetime(2026, 10, 26, 9, tzinfo=UTC)
    assert week.force_flat == datetime(2026, 10, 30, 10, tzinfo=UTC)
    origin = datetime(2026, 10, 29, 9, tzinfo=edt)       # 13:00Z
    # Wall clock says 20 h to B (09:00 -> 05:00 local); elapsed time is 21 h.
    (h20,) = evaluate_heads(origin, (20,), Role.MANAGEMENT, week)
    assert h20.operational_eligible is True
    assert h20.target_timestamp == datetime(2026, 10, 30, 9, tzinfo=UTC)
    assert h20.remaining_seconds_to_force_flat == 21 * 3600
    assert operational_eligibility(origin, 21, Role.MANAGEMENT, week)[0] is False
    same_instant = origin.astimezone(timezone(timedelta(hours=2)))
    assert evaluate_heads(same_instant, (20, 21), Role.MANAGEMENT, week) == \
        evaluate_heads(origin, (20, 21), Role.MANAGEMENT, week)


def test_next_week_target_is_rejected_even_inside_next_operational_window():
    origin = datetime(2026, 1, 9, 9, tzinfo=UTC)
    target = origin + 72 * H
    assert target == W2.operational_start                # inside next [A, B)
    ok, reasons = operational_eligibility(origin, 72, Role.MANAGEMENT, W1)
    assert not ok
    assert OperationalReason.TARGET_AT_OR_AFTER_FORCE_FLAT in reasons
    assert OperationalReason.TARGET_AT_OR_AFTER_MARKET_CLOSE in reasons
    d = historical_label_eligibility(origin, 72, Role.MANAGEMENT, W1, input_available=True,
                                     observed_targets=_observed(), split=TRAIN)
    assert d.label_eligible is False
    assert d.label_reasons[0] is LabelReason.OPERATIONALLY_INELIGIBLE


# ------------------------------------------------- M: historical label


def test_removing_a_future_bar_changes_M_and_never_E():
    origin, h = A1 + 24 * H, 24
    full = historical_label_eligibility(origin, h, Role.ENTRY, W1, input_available=True,
                                        observed_targets=_observed(), split=TRAIN)
    assert full.operational_eligible and full.label_eligible
    gap = historical_label_eligibility(
        origin, h, Role.ENTRY, W1, input_available=True,
        observed_targets=_observed(drop={origin + h * H}), split=TRAIN)
    assert gap.operational_eligible is True
    assert gap.operational_reasons == full.operational_reasons == ()
    assert gap.label_eligible is False
    assert gap.label_reasons == (LabelReason.TARGET_NOT_OBSERVED,)
    other = historical_label_eligibility(
        origin, h, Role.ENTRY, W1, input_available=True,
        observed_targets=_observed(drop={origin + (h + 1) * H}), split=TRAIN)
    assert other == full
    assert operational_eligibility(origin, h, Role.ENTRY, W1) == (True, ())


def test_past_gaps_do_not_move_A_or_B():
    origin = A1 + 24 * H
    past_gap = set(_hourly(O1, A1 + 12 * H))             # Monday bars missing
    full = historical_label_eligibility(origin, 24, Role.MANAGEMENT, W1,
                                        input_available=True,
                                        observed_targets=_observed(), split=TRAIN)
    gapped = historical_label_eligibility(origin, 24, Role.MANAGEMENT, W1,
                                          input_available=True,
                                          observed_targets=_observed(drop=past_gap),
                                          split=TRAIN)
    assert gapped == full
    assert (W1.operational_start, W1.force_flat) == (A1, B1)
    base = _table_config(label=True)
    holed = _table_config(label=True, drop=past_gap)
    rows = {(r["week_id"], r["role"], r["horizon_hours"]): r
            for r in build_support_table(base)["rows"]}
    rows_g = {(r["week_id"], r["role"], r["horizon_hours"]): r
              for r in build_support_table(holed)["rows"]}
    for key, row in rows.items():
        g = rows_g[key]
        for field in ("operational_start", "force_flat", "operational_eligible",
                      "eligible_origins_sha256"):
            assert g[field] == row[field]


def test_target_on_exclusive_split_end_is_rejected_and_start_is_inclusive():
    origin, h = A1 + 24 * H, 24
    q = origin + h * H
    end_on_q = SplitBounds("TRAIN", q - 1000 * H, q, label_available_by=None)
    d = historical_label_eligibility(origin, h, Role.MANAGEMENT, W1, input_available=True,
                                     observed_targets=_observed(), split=end_on_q)
    assert d.operational_eligible and not d.label_eligible
    assert d.label_reasons == (LabelReason.TARGET_AT_OR_AFTER_SPLIT_END,)
    just_inside = SplitBounds("TRAIN", q - 1000 * H, q + H, label_available_by=None)
    assert historical_label_eligibility(
        origin, h, Role.MANAGEMENT, W1, input_available=True,
        observed_targets=_observed(), split=just_inside).label_eligible
    start_on_q = SplitBounds("VALIDATION", q, q + 1000 * H, label_available_by=None)
    assert historical_label_eligibility(
        origin, h, Role.MANAGEMENT, W1, input_available=True,
        observed_targets=_observed(), split=start_on_q).label_eligible


def test_label_cutoff_and_input_availability_are_label_conditions_only():
    origin, h = A1 + 24 * H, 24
    q = origin + h * H
    late = SplitBounds("TRAIN", TRAIN.start, TRAIN.end, label_available_by=q - H)
    d = historical_label_eligibility(origin, h, Role.MANAGEMENT, W1, input_available=True,
                                     observed_targets=_observed(), split=late)
    assert d.label_reasons == (LabelReason.TARGET_AVAILABLE_AFTER_LABEL_CUTOFF,)
    exact = SplitBounds("TRAIN", TRAIN.start, TRAIN.end, label_available_by=q)
    assert historical_label_eligibility(
        origin, h, Role.MANAGEMENT, W1, input_available=True,
        observed_targets=_observed(), split=exact).label_eligible
    no_input = historical_label_eligibility(origin, h, Role.MANAGEMENT, W1,
                                            input_available=False,
                                            observed_targets=_observed(), split=TRAIN)
    assert no_input.operational_eligible is True
    assert no_input.label_reasons == (LabelReason.INPUT_UNAVAILABLE,)
    with pytest.raises(EligibilityError) as err:
        historical_label_eligibility(origin, h, Role.MANAGEMENT, W1, input_available=1,
                                     observed_targets=_observed(), split=TRAIN)
    assert err.value.code is ErrorCode.INPUT_AVAILABILITY_INVALID


def test_label_arguments_are_all_or_nothing():
    with pytest.raises(EligibilityError) as err:
        evaluate_heads(A1, (2,), Role.MANAGEMENT, W1, input_available=True)
    assert err.value.code is ErrorCode.LABEL_EVIDENCE_INVALID


def test_known_non_trading_inputs_remain_admissible_and_future_ones_do_not():
    origin = A1 + 2 * H
    for observed_at in (O1 - 30 * H,          # weekend before the open
                        O1 + H,               # [O, A) no-trade band of this week
                        O1 - 60 * H,          # synthetic prior force-flat band
                        origin):              # t itself is included
        ok, reason = input_admissibility(origin, observed_at)
        assert ok is True and reason is None
    ok, reason = input_admissibility(origin, origin + H)
    assert ok is False and reason == "INPUT_NOT_YET_AVAILABLE"


# ------------------------------------------------- calendar + support table


def _calendar_json(weeks=((O1, C1, CUT1, "W1"),)):
    return {"calendar_id": CAL.calendar_id, "version": CAL.version,
            "weeks": [{"week_id": w, "market_open": _iso(o), "market_close": _iso(c),
                       "entry_cutoff": _iso(cut)} for o, c, cut, w in weeks]}


def _table_config(label=False, drop=(), grid=None, horizons=(2, 72, 96, 120),
                  weeks=((O1, C1, CUT1, "W1"),)):
    cfg = {
        "schema_version": "i6n.eligibility.config.v1",
        "market_calendar": _calendar_json(weeks),
        "entry_delay_after_open_hours": 12,
        "force_flat_before_close_hours": 12,
        "roles": ["ENTRY", "MANAGEMENT"],
        "horizons_hours": list(horizons),
        "origin_grid": grid or {"kind": "UTC_WHOLE_HOURS"},
        "input_unavailable_origins": [],
        "splits": [{"split_id": "TRAIN", "start": "2025-01-01T00:00:00+00:00",
                    "end": "2027-01-01T00:00:00+00:00", "label_available_by": None}],
        "label_evidence": None,
    }
    if label:
        cfg["label_evidence"] = {"observed_targets": [
            {"timestamp": _iso(t), "available_at": _iso(a)}
            for t, a in _observed(drop=drop).items()]}
    return cfg


def test_missing_calendar_is_a_typed_error_with_no_fallback():
    for cfg in ({}, {"market_calendar": None},
                {"market_calendar": {"calendar_id": "X", "weeks": []}},
                {"market_calendar": {"version": "1", "weeks": []}}):
        with pytest.raises(EligibilityError) as err:
            load_calendar(cfg)
        assert err.value.code is ErrorCode.CALENDAR_REQUIRED
    cfg = _table_config(label=True)
    del cfg["market_calendar"]
    with pytest.raises(EligibilityError) as err:      # prices never imply a calendar
        build_support_table(cfg)
    assert err.value.code is ErrorCode.CALENDAR_REQUIRED


def test_calendar_rejects_overlapping_weeks_and_mismatched_declared_bounds():
    with pytest.raises(EligibilityError) as err:
        load_calendar({"market_calendar": _calendar_json(
            ((O1, C1, CUT1, "W1"), (C1 - H, C2, CUT2, "W2"))),
            "entry_delay_after_open_hours": 12, "force_flat_before_close_hours": 12})
    assert err.value.code is ErrorCode.CALENDAR_INVALID
    cal = _calendar_json()
    cal["weeks"][0]["force_flat"] = _iso(B1 + H)
    with pytest.raises(EligibilityError) as err:
        load_calendar({"market_calendar": cal, "entry_delay_after_open_hours": 12,
                       "force_flat_before_close_hours": 12})
    assert err.value.code is ErrorCode.CALENDAR_OFFSET_MISMATCH


def test_support_table_enumerates_every_week_role_head():
    table = build_support_table(_table_config())
    rows = {(r["role"], r["horizon_hours"]): r for r in table["rows"]}
    assert len(table["rows"]) == 1 * 2 * 4
    assert all(r["origins_enumerated"] == 120 for r in table["rows"])
    assert rows[("MANAGEMENT", 2)]["operational_eligible"] == 94
    assert rows[("MANAGEMENT", 72)]["operational_eligible"] == 24
    assert rows[("ENTRY", 2)]["operational_eligible"] == 62
    assert rows[("ENTRY", 72)]["operational_eligible"] == 24
    for role in ("ENTRY", "MANAGEMENT"):
        for h in (96, 120):
            row = rows[(role, h)]
            assert row["operational_eligible"] == 0
            assert row["operational_status"] == "NO_ELIGIBLE_ROWS"
            assert row["first_eligible_origin"] is None
    m72 = rows[("MANAGEMENT", 72)]
    assert m72["first_eligible_origin"] == _iso(A1)
    assert m72["last_eligible_origin"] == _iso(B1 - 73 * H)
    assert sum(m72["operational_exclusions"].values()) == 120 - 24
    assert m72["label"] is None and m72["label_status"] == "LABEL_EVIDENCE_NOT_SUPPLIED"
    assert table["candidate_implementation"] is True
    assert table["scientific_measurement"] is False
    assert table["origin_grid"] == {"kind": "UTC_WHOLE_HOURS"}


def test_support_depends_on_enumerated_timestamps_not_a_closed_form():
    explicit = [t for t in _hourly(O1, C1) if t.hour % 2 == 0] + \
        [datetime(2026, 1, 10, 3, tzinfo=UTC)]                 # weekend: no week
    cfg = _table_config(grid={"kind": "EXPLICIT",
                              "timestamps": [_iso(t) for t in explicit]},
                        horizons=(72,))
    table = build_support_table(cfg)
    row = [r for r in table["rows"] if r["role"] == "MANAGEMENT"][0]
    assert row["origins_enumerated"] == 60
    assert row["operational_eligible"] == 12               # not (B-A)-h = 24
    assert table["origins_outside_calendar_weeks"] == 1


def test_table_label_counts_by_split_and_input_availability():
    cfg = _table_config(label=True, horizons=(24,))
    q_missing = A1 + 48 * H
    cfg["label_evidence"]["observed_targets"] = [
        o for o in cfg["label_evidence"]["observed_targets"]
        if o["timestamp"] != _iso(q_missing)]
    cfg["input_unavailable_origins"] = [_iso(A1)]
    split_at = A1 + 60 * H
    cfg["splits"] = [
        {"split_id": "TRAIN", "start": _iso(O1 - 1000 * H), "end": _iso(split_at),
         "label_available_by": None},
        {"split_id": "VALIDATION", "start": _iso(split_at), "end": _iso(C2),
         "label_available_by": None}]
    row = [r for r in build_support_table(cfg)["rows"] if r["role"] == "MANAGEMENT"][0]
    assert row["operational_eligible"] == 72
    label = row["label"]
    assert label["label_eligible"] == 70
    assert label["by_split"] == {"TRAIN": 34, "VALIDATION": 36}
    assert label["exclusions_among_operational"] == {
        "INPUT_UNAVAILABLE": 1, "TARGET_NOT_OBSERVED": 1}


def test_splits_must_not_overlap():
    cfg = _table_config(label=True)
    cfg["splits"].append(dict(cfg["splits"][0], split_id="OTHER"))
    with pytest.raises(EligibilityError) as err:
        build_support_table(cfg)
    assert err.value.code is ErrorCode.SPLIT_INVALID


def test_example_config_is_synthetic_and_runs():
    path = Path(eligibility_cli.__file__).parent / "examples" / "eligibility_synthetic.json"
    cfg = json.loads(path.read_text())
    assert "SYNTHETIC" in cfg["market_calendar"]["calendar_id"]
    table = build_support_table(cfg)
    assert len(table["rows"]) == (len(cfg["market_calendar"]["weeks"])
                                  * len(cfg["roles"]) * len(cfg["horizons_hours"]))


# ------------------------------------------------- CLI and STATUS.json


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


STATUS_KEYS = {"schema_version", "state", "units_done", "units_total", "started_at",
               "last_progress_at", "duration_seconds", "eta_seconds",
               "eta_unavailable_reason", "input_sha256", "config_sha256",
               "code_sha256", "result", "error"}


def test_cli_runs_reuses_and_refuses_a_different_input(tmp_path, capsys):
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 0
    status = json.loads((out / "STATUS.json").read_text())
    assert STATUS_KEYS <= set(status)
    assert status["state"] == "COMPLETED"
    assert status["units_done"] == status["units_total"] == 8
    assert status["error"] is None and status["eta_seconds"] == 0
    result = out / status["result"]["path"]
    first = result.read_bytes()
    stamp = os.stat(result).st_mtime_ns

    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 0
    assert "REUSED" in capsys.readouterr().out
    assert os.stat(result).st_mtime_ns == stamp

    other = _write(tmp_path / "b.json", _table_config(horizons=(2,)))
    assert eligibility_cli.main(["--config", str(other), "--output", str(out)]) == 2
    assert "OUTPUT_HOLDS_DIFFERENT_INPUT" in capsys.readouterr().out
    assert result.read_bytes() == first
    assert json.loads((out / "STATUS.json").read_text()) == status


def test_cli_status_is_read_only_and_returns_once(tmp_path, capsys):
    out = tmp_path / "never"
    assert eligibility_cli.main(["--status", "--output", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["state"] == "NOT_STARTED"
    assert printed["error"]["code"] == "STATUS_ABSENT"
    assert not out.exists()
    cfg = _write(tmp_path / "a.json", _table_config())
    done = tmp_path / "done"
    eligibility_cli.main(["--config", str(cfg), "--output", str(done)])
    capsys.readouterr()
    before = {p.name: p.read_bytes() for p in done.iterdir()}
    assert eligibility_cli.main(["--status", "--output", str(done)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["state"] == "COMPLETED" and printed["result_intact"] is True
    assert {p.name: p.read_bytes() for p in done.iterdir()} == before


def test_cli_missing_calendar_fails_typed_without_result(tmp_path, capsys):
    cfg = _table_config()
    del cfg["market_calendar"]
    path = _write(tmp_path / "bad.json", cfg)
    out = tmp_path / "out"
    assert eligibility_cli.main(["--config", str(path), "--output", str(out)]) == 1
    status = json.loads((out / "STATUS.json").read_text())
    assert status["state"] == "FAILED"
    assert status["error"]["code"] == "CALENDAR_REQUIRED"
    assert status["result"] is None
    assert status["eta_seconds"] is None and status["eta_unavailable_reason"]
