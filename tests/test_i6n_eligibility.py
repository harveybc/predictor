"""Acceptance tests for the I6-N C1 pure horizon eligibility contract.

Every calendar here is a SYNTHETIC fixture built inside the test. None of
them represents a broker, an exchange or New York hours; they only exercise
the rules of BUSINESS_MARKET_WEEK_CONTRACT_2026_10_11.md section 2-3 and the
owner order section 4. Nothing is trained and Keras masking is not exercised.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tools.i6n_contracts import eligibility as elig_mod
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
    # CR03: an absent output is a distinguishable NON-success (10), never 0.
    assert eligibility_cli.main(["--status", "--output", str(out)]) == 10
    printed = json.loads(capsys.readouterr().out)
    assert printed["disposition"] == "ABSENT" and printed["result_intact"] is False
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


# ------------------------------------------------- review round 1 (FIX_REQUIRED)


def test_label_path_never_drops_valid_heads():
    """Never-drop rule on the M path: invalid long heads stay reported."""
    origin = B1 - 48 * H
    heads = evaluate_heads(origin, ALL, Role.MANAGEMENT, W1, input_available=True,
                           observed_targets=_observed(), split=TRAIN)
    assert [d.horizon_hours for d in heads] == list(ALL)
    assert all(d.label_evaluated for d in heads)
    assert {d.horizon_hours for d in heads if d.label_eligible} == set(SHORT) | {18, 24, 36}
    for d in heads:
        assert d.target_timestamp == origin + d.horizon_hours * H
        if d.horizon_hours in (48, 60, 72):
            assert d.operational_eligible is False and d.label_eligible is False
            assert d.label_reasons[0] is LabelReason.OPERATIONALLY_INELIGIBLE


def test_dst_end_with_real_zoneinfo_uses_elapsed_hours():
    # Timezone RULE only (US DST ends 2026-11-01 02:00 local); the week itself
    # is a synthetic fixture and is not a broker or exchange calendar.
    ny = ZoneInfo("America/New_York")
    o = datetime(2026, 10, 31, 12, tzinfo=ny)            # EDT, 16:00Z
    c = datetime(2026, 11, 5, 12, tzinfo=ny)             # EST, 17:00Z
    assert c.replace(tzinfo=None) - o.replace(tzinfo=None) == 120 * H   # wall clock
    assert c.astimezone(UTC) - o.astimezone(UTC) == 121 * H             # elapsed
    week = _week(o, c, datetime(2026, 11, 4, 0, tzinfo=ny), "WNY")
    assert week.operational_start == datetime(2026, 11, 1, 4, tzinfo=UTC)
    assert week.force_flat == datetime(2026, 11, 5, 5, tzinfo=UTC)
    origin = datetime(2026, 11, 1, 0, tzinfo=ny)         # = A, before the 02:00 change
    # Wall clock: 96 h to B (00:00 -> 00:00). Elapsed: 97 h, so h=96 is valid.
    (h96,) = evaluate_heads(origin, (96,), Role.MANAGEMENT, week)
    assert h96.operational_eligible is True and h96.operational_reasons == ()
    assert h96.remaining_seconds_to_force_flat == 97 * 3600
    assert h96.target_timestamp == datetime(2026, 11, 5, 4, tzinfo=UTC)
    assert h96.to_json()["target_timestamp"] == "2026-11-05T04:00:00+00:00"
    ok, reasons = operational_eligibility(origin, 97, Role.MANAGEMENT, week)
    assert not ok and reasons == (OperationalReason.TARGET_AT_OR_AFTER_FORCE_FLAT,)


def test_per_head_available_is_explicit_input_driven_and_separate_from_E_and_M():
    reason = elig_mod.AvailabilityReason
    origin = B1 - 48 * H
    for d in evaluate_heads(origin, (2, 72), Role.MANAGEMENT, W1):
        assert d.availability_evaluated is False and d.available is None
        assert d.to_json()["available"] is None            # never a default True/False
    h2, h72 = evaluate_heads(origin, (2, 72), Role.MANAGEMENT, W1, prediction_available=True)
    assert (h2.available, h2.availability_reasons) == (True, ())
    assert (h72.available, h72.availability_reasons) == (
        False, (reason.OPERATIONALLY_INELIGIBLE,))
    h2, h72 = evaluate_heads(origin, (2, 72), Role.MANAGEMENT, W1,
                             input_available=True, observed_targets=_observed(),
                             split=TRAIN, prediction_available={2: False, 72: True})
    assert h2.operational_eligible is True and h2.label_eligible is True   # E, M unchanged
    assert (h2.available, h2.availability_reasons) == (False, (reason.PREDICTION_UNAVAILABLE,))
    assert h2.to_json()["available"] is False
    assert h2.to_json()["availability_reasons"] == ["PREDICTION_UNAVAILABLE"]
    assert (h72.available, h72.availability_reasons) == (
        False, (reason.OPERATIONALLY_INELIGIBLE,))
    for bad in (1, "yes", {2: True}, {2: True, 72: True, 3: True}, {2: True, 72: 1},
                {True: True, 72: True}):
        with pytest.raises(EligibilityError) as err:
            evaluate_heads(origin, (2, 72), Role.MANAGEMENT, W1, prediction_available=bad)
        assert err.value.code is ErrorCode.PREDICTION_AVAILABILITY_INVALID


def test_input_unavailable_origins_must_match_an_enumerated_origin():
    cfg = _table_config(label=True, horizons=(24,))
    for bad in (["2030-01-01T00:00:00+00:00"],                        # matches nothing
                [_iso(A1), _iso(A1)],                                 # duplicate
                [_iso(datetime(2026, 1, 10, 3, tzinfo=UTC))]):        # weekend, no week
        cfg["input_unavailable_origins"] = bad
        with pytest.raises(EligibilityError) as err:
            build_support_table(cfg)
        assert err.value.code is ErrorCode.INPUT_AVAILABILITY_INVALID
    cfg["input_unavailable_origins"] = [_iso(A1)]
    table = build_support_table(cfg)
    assert table["input_unavailable_origins_applied"] == 1
    no_label = _table_config(horizons=(24,))
    no_label["input_unavailable_origins"] = [_iso(A1)]       # would have no effect at all
    with pytest.raises(EligibilityError) as err:
        build_support_table(no_label)
    assert err.value.code is ErrorCode.INPUT_AVAILABILITY_INVALID


@pytest.mark.parametrize("span", [24, 20])
def test_week_with_empty_operational_window_is_a_typed_error(span):
    # 12 h + 12 h offsets: span 24 gives A == B, span 20 gives A > B.
    with pytest.raises(EligibilityError) as err:
        _week(o=O1, c=O1 + span * H, cut=O1 + H, wid="WSHORT")
    assert err.value.code is ErrorCode.OPERATIONAL_WINDOW_EMPTY
    with pytest.raises(EligibilityError) as err:
        build_support_table(_table_config(weeks=((O1, O1 + span * H, O1 + H, "WSHORT"),)))
    assert err.value.code is ErrorCode.OPERATIONAL_WINDOW_EMPTY


def test_cli_imports_no_database_dataframe_or_tensor_library():
    root = Path(__file__).resolve().parents[1]
    probe = ("import sys, tools.i6n_contracts.eligibility_cli; "
             "bad = sorted({'sqlite3', '_sqlite3', 'pandas', 'tensorflow', 'numpy'} "
             "& set(sys.modules)); print(bad); sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", probe], cwd=root, capture_output=True,
                          text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    names = {name for name, _ in eligibility_cli._CODE_FILES}
    assert "tools/i6m_queue.py" not in names


def test_cli_refuses_a_damaged_completed_result(tmp_path, capsys):
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 0
    capsys.readouterr()
    result = out / "SUPPORT_TABLE.json"
    damaged = result.read_bytes() + b" "
    result.write_bytes(damaged)
    status_bytes = (out / "STATUS.json").read_bytes()
    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 2
    assert "TERMINAL_OUTPUT_DAMAGED" in capsys.readouterr().out
    assert result.read_bytes() == damaged                      # never silently rewritten
    assert (out / "STATUS.json").read_bytes() == status_bytes
    # CR03: a damaged terminal result is never reported with exit 0.
    assert eligibility_cli.main(["--status", "--output", str(out)]) == 16
    printed = json.loads(capsys.readouterr().out)
    assert printed["disposition"] == "DAMAGED" and printed["result_intact"] is False


def test_code_digest_tracks_code_bytes_and_refuses_different_code(tmp_path, monkeypatch,
                                                                  capsys):
    copies = []
    for name, path in eligibility_cli._CODE_FILES:
        dst = tmp_path / "code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
        copies.append((name, dst))
    monkeypatch.setattr(eligibility_cli, "_CODE_FILES", tuple(copies))
    before = eligibility_cli.code_sha256()
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 0
    capsys.readouterr()
    status_bytes = (out / "STATUS.json").read_bytes()
    assert json.loads(status_bytes)["code_sha256"] == before
    with copies[0][1].open("ab") as stream:
        stream.write(b"\n# one changed byte sequence\n")
    assert eligibility_cli.code_sha256() != before
    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 2
    assert "OUTPUT_HOLDS_DIFFERENT_CODE" in capsys.readouterr().out
    assert (out / "STATUS.json").read_bytes() == status_bytes


def test_status_is_replaced_atomically_and_result_is_created_never_replaced(tmp_path,
                                                                           monkeypatch):
    # CR01 (structural): STATUS.json is swapped atomically; SUPPORT_TABLE.json
    # is created from a complete temp file with os.link, which cannot clobber.
    real_replace, real_link = os.replace, os.link
    swapped, linked = [], []

    def link_spy(src, dst):
        src, dst = Path(src), Path(dst)
        assert src.parent == dst.parent and not dst.exists()
        json.loads(src.read_text())
        linked.append(dst.name)
        return real_link(src, dst)

    def spy(src, dst):
        src, dst = Path(src), Path(dst)
        assert src.parent == dst.parent and src.name != dst.name
        json.loads(src.read_text())                      # staged copy complete first
        swapped.append(dst.name)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    monkeypatch.setattr(os, "link", link_spy)
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    assert eligibility_cli.main(["--config", str(cfg), "--output", str(out)]) == 0
    assert swapped.count("STATUS.json") >= 3
    assert "SUPPORT_TABLE.json" not in swapped
    assert linked == ["SUPPORT_TABLE.json"]
    assert sorted(p.name for p in out.iterdir()) == ["STATUS.json", "SUPPORT_TABLE.json"]


# ------------------------------------- review round 2 (Musashi CR01-CR04)
# Every test below demands REJECTION, NO OVERWRITE or NON-SUCCESS. None of
# them freezes the behaviour the review probe observed on the audited commit
# a1e64e4c; they were run there first (PRE) and failed.

STATUS_EXIT = {"COMPLETE_INTACT": 0, "ABSENT": 10, "RUNNING": 11, "INCOMPLETE": 12,
               "RUNNING_UNVERIFIED": 13, "FAILED": 14, "INTERRUPTED": 15,
               "DAMAGED": 16, "ORPHAN_RESULT": 17}


def _cli(*args):
    try:
        return eligibility_cli.main([str(a) for a in args])
    except SystemExit as exc:                # argparse usage error (unknown flag)
        return exc.code


def _lines(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def _snapshot(directory):
    if not directory.exists():
        return None
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completed(tmp_path, capsys, name="out", **cfg_kw):
    out = tmp_path / name
    cfg = _write(tmp_path / f"{name}.json", _table_config(**cfg_kw))
    assert _cli("--config", cfg, "--output", out) == 0
    capsys.readouterr()
    return cfg, out


def _status_of(out, capsys):
    code = _cli("--status", "--output", out)
    (printed,) = _lines(capsys)
    return code, printed


def _raiser(exc):
    def raise_it(*_args, **_kwargs):
        raise exc
    return raise_it


def _copy_code(tmp_path, monkeypatch):
    copies = []
    for name, path in eligibility_cli._CODE_FILES:
        dst = tmp_path / "code" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path.read_bytes())
        copies.append((name, dst))
    monkeypatch.setattr(eligibility_cli, "_CODE_FILES", tuple(copies))
    return copies


def _no_temp_files(out):
    return not [p.name for p in out.iterdir() if p.name.endswith(".tmp")]


# ---- CR01: result identity is verified independently of STATUS.json


def test_cr01_orphan_result_of_another_input_is_refused_without_writing(tmp_path, capsys):
    _, out = _completed(tmp_path, capsys)
    (out / "STATUS.json").unlink()
    before = _snapshot(out)
    other = _write(tmp_path / "other.json", _table_config(horizons=(2,)))
    for extra in ((), ("--recover",)):
        assert _cli("--config", other, "--output", out, *extra) == 2
        assert "OUTPUT_HOLDS_DIFFERENT_INPUT" in capsys.readouterr().out
        assert _snapshot(out) == before          # result kept, no STATUS written


def test_cr01_orphan_result_of_other_code_is_refused_without_writing(tmp_path, monkeypatch,
                                                                    capsys):
    copies = _copy_code(tmp_path, monkeypatch)
    cfg, out = _completed(tmp_path, capsys)
    (out / "STATUS.json").unlink()
    with copies[0][1].open("ab") as stream:
        stream.write(b"\n# one changed byte sequence\n")
    before = _snapshot(out)
    assert _cli("--config", cfg, "--output", out) == 2
    assert "OUTPUT_HOLDS_DIFFERENT_CODE" in capsys.readouterr().out
    assert _snapshot(out) == before


@pytest.mark.parametrize("damage", ["garbage", "truncated", "edited_content"])
def test_cr01_damaged_orphan_result_is_refused_without_writing(tmp_path, capsys, damage):
    cfg, out = _completed(tmp_path, capsys)
    (out / "STATUS.json").unlink()
    result = out / "SUPPORT_TABLE.json"
    data = result.read_bytes()
    if damage == "garbage":
        result.write_bytes(b"corrupt")
    elif damage == "truncated":
        result.write_bytes(data[: len(data) // 2])
    else:
        table = json.loads(data)
        table["rows"][0]["operational_eligible"] += 1
        result.write_text(json.dumps(table, indent=2) + "\n")
    before = _snapshot(out)
    for extra in ((), ("--recover",)):
        assert _cli("--config", cfg, "--output", out, *extra) == 2
        assert "RESULT_DAMAGED" in capsys.readouterr().out
        assert _snapshot(out) == before


def test_cr01_identical_orphan_result_needs_explicit_recovery_never_adoption(tmp_path,
                                                                           capsys):
    cfg, out = _completed(tmp_path, capsys)
    original = json.loads((out / "STATUS.json").read_text())
    (out / "STATUS.json").unlink()
    result = out / "SUPPORT_TABLE.json"
    before = _snapshot(out)
    stat = os.stat(result)
    assert _cli("--config", cfg, "--output", out) == 2      # existence is not adoption
    (event,) = _lines(capsys)
    assert event["event"] == "REFUSED"
    assert event["error"]["code"] == "RECOVERY_REQUIRED"
    assert _snapshot(out) == before
    assert _cli("--config", cfg, "--output", out, "--recover") == 0
    (event,) = _lines(capsys)
    assert event["event"] == "RECOVERED"
    after = os.stat(result)
    assert (after.st_ino, after.st_mtime_ns) == (stat.st_ino, stat.st_mtime_ns)
    assert result.read_bytes() == before["SUPPORT_TABLE.json"]
    status = json.loads((out / "STATUS.json").read_text())
    assert status["state"] == "COMPLETED" and status["recovered"] is True
    assert status["identity_sha256"] == original["identity_sha256"]
    assert status["result"]["sha256"] == original["result"]["sha256"]
    assert _status_of(out, capsys)[0] == 0


def test_cr01_recovery_refuses_bytes_that_differ_from_the_recomputation(tmp_path, capsys):
    cfg, out = _completed(tmp_path, capsys)
    (out / "STATUS.json").unlink()
    result = out / "SUPPORT_TABLE.json"
    result.write_bytes(result.read_bytes() + b" ")          # same JSON, other bytes
    before = _snapshot(out)
    assert _cli("--config", cfg, "--output", out, "--recover") == 2
    assert "RECOVERY_MISMATCH" in capsys.readouterr().out
    assert _snapshot(out) == before


def test_cr01_failed_status_beside_an_intact_result_is_not_adopted(tmp_path, capsys):
    cfg, out = _completed(tmp_path, capsys)
    status_path = out / "STATUS.json"
    record = json.loads(status_path.read_text())
    record.update(state="FAILED", result=None,
                  error={"code": "UNEXPECTED_ERROR", "message": "fixture"})
    status_path.write_text(json.dumps(record))
    before = _snapshot(out)
    assert _cli("--config", cfg, "--output", out) == 2
    assert "RECOVERY_REQUIRED" in capsys.readouterr().out
    assert _snapshot(out) == before


# ---- CR03: --status is read-only and exits 0 only for an intact terminal


def _scenario(kind, tmp_path, capsys):
    out = tmp_path / kind
    if kind == "absent":
        return out
    cfg = _write(tmp_path / f"{kind}.json", _table_config())
    if kind == "failed":
        bad = _table_config()
        del bad["market_calendar"]
        _write(cfg, bad)
        assert _cli("--config", cfg, "--output", out) == 1
        capsys.readouterr()
        return out
    assert _cli("--config", cfg, "--output", out) == 0
    capsys.readouterr()
    status_path, result = out / "STATUS.json", out / "SUPPORT_TABLE.json"
    if kind == "result_garbage":
        result.write_text("corrupt")
    elif kind == "result_extra_byte":
        result.write_bytes(result.read_bytes() + b" ")
    elif kind == "result_missing":
        result.unlink()
    elif kind == "status_missing":
        status_path.unlink()
    elif kind == "status_garbage":
        status_path.write_text("{")
    elif kind in ("result_edited_status_resealed", "result_of_other_identity_resealed"):
        if kind == "result_edited_status_resealed":
            table = json.loads(result.read_text())
            del table["rows"][-1]
            table["units_total"] -= 1
            result.write_text(json.dumps(table, indent=2) + "\n")
        else:
            other_cfg = _write(tmp_path / "other.json", _table_config(horizons=(2,)))
            other = tmp_path / "other_out"
            assert _cli("--config", other_cfg, "--output", other) == 0
            capsys.readouterr()
            result.write_bytes((other / "SUPPORT_TABLE.json").read_bytes())
        record = json.loads(status_path.read_text())
        record["result"]["sha256"] = _sha(result)                  # status resealed
        status_path.write_text(json.dumps(record))
    return out


@pytest.mark.parametrize("kind, disposition", [
    ("complete", "COMPLETE_INTACT"),
    ("absent", "ABSENT"),
    ("failed", "FAILED"),
    ("result_garbage", "DAMAGED"),
    ("result_extra_byte", "DAMAGED"),
    ("result_missing", "DAMAGED"),
    ("status_garbage", "DAMAGED"),
    ("result_edited_status_resealed", "DAMAGED"),
    ("result_of_other_identity_resealed", "DAMAGED"),
    ("status_missing", "ORPHAN_RESULT"),
])
def test_cr03_status_exit_zero_only_for_an_intact_complete_terminal(tmp_path, capsys, kind,
                                                                   disposition):
    out = _scenario(kind, tmp_path, capsys)
    before = _snapshot(out)
    code, printed = _status_of(out, capsys)
    assert printed["disposition"] == disposition
    assert code == STATUS_EXIT[disposition]
    assert printed["read_only"] is True
    assert printed["result_intact"] is (disposition == "COMPLETE_INTACT")
    assert _snapshot(out) == before                       # nothing created or rewritten


def test_cr03_status_exit_codes_are_distinct_and_only_success_is_zero():
    assert eligibility_cli.STATUS_EXIT_CODES == STATUS_EXIT
    assert len(set(STATUS_EXIT.values())) == len(STATUS_EXIT)
    assert [k for k, v in STATUS_EXIT.items() if v == 0] == ["COMPLETE_INTACT"]


# ---- CR04: ordinary failures become typed FAILED; interrupts are recorded


def test_cr04_ordinary_exception_is_recorded_as_typed_failed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(elig_mod, "assemble_support_table",
                        _raiser(RuntimeError("injected")))
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    assert _cli("--config", cfg, "--output", out) == 1
    assert [e["event"] for e in _lines(capsys)] == ["FAILED"]
    status = json.loads((out / "STATUS.json").read_text())
    assert status["state"] == "FAILED" and status["result"] is None
    assert status["error"]["code"] == "UNEXPECTED_ERROR"
    assert status["error"]["exception_type"] == "RuntimeError"
    assert sorted(p.name for p in out.iterdir()) == ["STATUS.json"]
    code, printed = _status_of(out, capsys)
    assert (code, printed["disposition"]) == (14, "FAILED")
    direct = tmp_path / "direct"           # the library call re-raises after recording
    with pytest.raises(RuntimeError):
        eligibility_cli.run_command(cfg, direct)
    assert json.loads((direct / "STATUS.json").read_text())["state"] == "FAILED"
    capsys.readouterr()
    monkeypatch.undo()                                  # control: FAILED is retried
    assert _cli("--config", cfg, "--output", out) == 0
    capsys.readouterr()
    assert _status_of(out, capsys)[0] == 0


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="needs /proc fd links")
def test_cr04_failure_while_writing_the_result_is_failed_without_result_or_temp(
        tmp_path, monkeypatch, capsys):
    real_fsync = os.fsync

    def fsync(fd):
        if "SUPPORT_TABLE" in os.readlink(f"/proc/self/fd/{fd}"):
            raise OSError(errno.ENOSPC, "injected: no space left on device")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    assert _cli("--config", cfg, "--output", out) == 1
    status = json.loads((out / "STATUS.json").read_text())
    assert status["state"] == "FAILED"
    assert status["error"]["exception_type"] == "OSError"
    assert sorted(p.name for p in out.iterdir()) == ["STATUS.json"]


def test_cr04_interrupt_is_recorded_and_propagated(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(elig_mod, "assemble_support_table", _raiser(KeyboardInterrupt()))
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    with pytest.raises(KeyboardInterrupt):
        eligibility_cli.main(["--config", str(cfg), "--output", str(out)])
    status = json.loads((out / "STATUS.json").read_text())
    assert status["state"] == "INTERRUPTED" and status["error"]["code"] == "INTERRUPTED"
    capsys.readouterr()
    code, printed = _status_of(out, capsys)
    assert (code, printed["disposition"]) == (15, "INTERRUPTED")
    monkeypatch.undo()                                   # control: recomputed once
    assert _cli("--config", cfg, "--output", out) == 0


def _persisted_running(tmp_path, monkeypatch, capsys):
    """A genuine RUNNING record: the run fails and its terminal write fails too."""
    monkeypatch.setattr(elig_mod, "assemble_support_table",
                        _raiser(RuntimeError("injected")))
    real_replace = os.replace

    def replace(src, dst):
        if (Path(dst).name == "STATUS.json"
                and json.loads(Path(src).read_text())["state"] != "RUNNING"):
            raise OSError(errno.EIO, "injected: status cannot be recorded")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    out = tmp_path / "out"
    cfg = _write(tmp_path / "a.json", _table_config())
    code = _cli("--config", cfg, "--output", out)
    events = _lines(capsys)
    monkeypatch.undo()
    return cfg, out, code, events


def test_cr04_unrecordable_terminal_status_exits_nonzero_without_claiming_failed(
        tmp_path, monkeypatch, capsys):
    _, out, code, events = _persisted_running(tmp_path, monkeypatch, capsys)
    assert code == 3
    assert not [e for e in events if e["event"] in ("FAILED", "COMPLETED")]
    (event,) = [e for e in events if e["event"] == "STATUS_NOT_RECORDED"]
    assert event["durable"] is False and event["attempted_state"] == "FAILED"
    assert json.loads((out / "STATUS.json").read_text())["state"] == "RUNNING"
    assert sorted(p.name for p in out.iterdir()) == ["STATUS.json"] and _no_temp_files(out)


def test_cr04_unwritable_output_exits_nonzero_without_status(tmp_path, capsys):
    out = tmp_path / "is_a_file"
    out.write_text("x")
    cfg = _write(tmp_path / "a.json", _table_config())
    assert _cli("--config", cfg, "--output", out) == 3
    assert "STATUS_NOT_RECORDED" in capsys.readouterr().out
    assert out.read_text() == "x"


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="needs /proc")
def test_cr04_persisted_running_is_distinguished_from_a_live_writer(tmp_path, monkeypatch,
                                                                   capsys):
    cfg, out, _, _ = _persisted_running(tmp_path, monkeypatch, capsys)
    status_path = out / "STATUS.json"
    record = json.loads(status_path.read_text())
    assert record["state"] == "RUNNING" and record["writer"]["pid"] == os.getpid()

    # Writer alive (this very process): RUNNING, and a second writer is refused.
    code, printed = _status_of(out, capsys)
    assert (code, printed["disposition"], printed["writer_liveness"]) == (
        11, "RUNNING", "ALIVE")
    before = _snapshot(out)
    assert _cli("--config", cfg, "--output", out) == 2
    assert "RUN_IN_PROGRESS" in capsys.readouterr().out
    assert _snapshot(out) == before

    # Same pid number, another process start (pid reuse): not the writer.
    reused = dict(record, writer=dict(record["writer"],
                                      process_start_ticks=record["writer"]
                                      ["process_start_ticks"] + 1))
    status_path.write_text(json.dumps(reused))
    code, printed = _status_of(out, capsys)
    assert (code, printed["disposition"], printed["writer_liveness"]) == (
        12, "INCOMPLETE", "NOT_FOUND")

    # Liveness that cannot be checked is not success and is not taken over.
    status_path.write_text(json.dumps(dict(record, writer={"pid": None})))
    code, printed = _status_of(out, capsys)
    assert (code, printed["disposition"]) == (13, "RUNNING_UNVERIFIED")
    assert _cli("--config", cfg, "--output", out) == 2
    capsys.readouterr()

    # Writer gone (SIGKILL or power loss): INCOMPLETE, then recomputed once.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    status_path.write_text(json.dumps(dict(record, writer=dict(record["writer"],
                                                                pid=child.pid))))
    code, printed = _status_of(out, capsys)
    assert (code, printed["disposition"], printed["writer_liveness"]) == (
        12, "INCOMPLETE", "NOT_FOUND")
    assert _cli("--config", cfg, "--output", out) == 0
    capsys.readouterr()
    final = json.loads(status_path.read_text())
    assert final["state"] == "COMPLETED" and final["previous_disposition"] == "INCOMPLETE"


# ---- atomic writer: cleanup on failure (re-review test gap, mutant N10)


def test_atomic_writer_removes_its_temp_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "STATUS.json"
    target.write_text('{"previous": true}\n')
    with pytest.raises(ValueError):
        eligibility_cli.atomic_json(target, {"x": float("nan")})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["STATUS.json"]
    monkeypatch.setattr(os, "replace", _raiser(OSError(errno.EIO, "injected")))
    with pytest.raises(OSError):
        eligibility_cli.atomic_json(target, {"ok": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["STATUS.json"]
    assert target.read_text() == '{"previous": true}\n'
