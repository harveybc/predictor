"""Whether a feature could have been known when its target was decided.

R1 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "Distinguish historical target-as-input from a future label. Tests use row IDs,
     timestamps, decision time and target horizon, including repeated equal values."

The rule the previous round got wrong: it compared the *values* of a window against the value
of its target and called equality a leak. Repeated values are ordinary in a price series — a
flat market produces them honestly — so equality proves nothing and inequality proves nothing
either. What decides the question is identity: WHICH observation is in the window, WHEN it
became available, and WHEN the target is decided.

Vocabulary, all declared per row rather than inferred:

* `row_id`   — the identity of an observation; two rows may carry equal values and still be
               different observations;
* `event_time` — when the thing happened;
* `available_time` — when we could first have read it (never earlier than the event);
* decision time of a target at index t: the availability of the last row of its window;
* horizon: how far ahead the target looks.

A feature is admissible for a target when its `available_time` is not after the decision time
of that target. That is the whole test: no numeric comparison anywhere.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

LOOKBACK = 3
HORIZON = 2
#: Deliberately flat: every value repeats, so any rule that reads values instead of identities
#: has nothing to grip and shows itself.
FLAT_VALUE = 1.0


def series(rows=12, lag_minutes=0):
    """A trajectory with identities, an event clock and an availability clock."""
    event = pd.date_range("2024-01-01 00:00:00", periods=rows, freq="h")
    return pd.DataFrame({
        "row_id": [f"r{index:03d}" for index in range(rows)],
        "event_time": event,
        "available_time": event + pd.Timedelta(minutes=lag_minutes),
        "CLOSE": [FLAT_VALUE] * rows,
    })


def windows(frame, lookback=LOOKBACK, horizon=HORIZON):
    """(window row_ids, target row_id) pairs, the way a trailing window is built."""
    out = []
    for end in range(lookback - 1, len(frame) - horizon):
        window = frame.iloc[end - lookback + 1:end + 1]
        target = frame.iloc[end + horizon]
        out.append((list(window["row_id"]), target["row_id"]))
    return out


def decision_time(frame, window_ids):
    """When the last observation of the window became available: the moment we may decide."""
    rows = frame[frame["row_id"].isin(window_ids)]
    return rows["available_time"].max()


def admissible(frame, window_ids, target_id):
    """Every row of the window known by the decision time, and the target strictly after it.

    Three ways to fail, all by identity or clock and none by value: the target sits inside its
    own window; a row of the window is not readable yet when the decision is taken; the target
    has already happened by then.
    """
    if target_id in window_ids:
        return False
    decided = decision_time(frame, window_ids)
    rows = frame[frame["row_id"].isin(window_ids)]
    if (rows["available_time"] > decided).any():
        return False
    target = frame[frame["row_id"] == target_id].iloc[0]
    return target["event_time"] > decided


def test_a_window_never_contains_the_observation_it_predicts():
    """By identity, not by value: every value in this frame is identical on purpose."""
    frame = series()
    assert frame["CLOSE"].nunique() == 1, "the fixture must be flat or it proves nothing"
    for window_ids, target_id in windows(frame):
        assert target_id not in window_ids, (
            f"{target_id} is both an input and the thing predicted")


def test_equal_values_are_not_a_leak_and_the_rule_does_not_say_they_are():
    """The previous round's mistake, written down as a rule so it cannot come back."""
    frame = series()
    for window_ids, target_id in windows(frame):
        target_value = frame.loc[frame["row_id"] == target_id, "CLOSE"].iloc[0]
        window_values = frame.loc[frame["row_id"].isin(window_ids), "CLOSE"]
        assert (window_values == target_value).all(), "fixture check: the values do repeat"
        assert admissible(frame, window_ids, target_id), (
            "equal values must not make an otherwise sound window inadmissible")


def test_a_target_of_the_same_variable_is_admissible_when_it_is_the_past_of_that_variable():
    """`CLOSE` predicting future `CLOSE` is legitimate; what matters is which rows."""
    frame = series()
    for window_ids, target_id in windows(frame):
        target_index = frame.index[frame["row_id"] == target_id][0]
        last_index = frame.index[frame["row_id"] == window_ids[-1]][0]
        assert target_index - last_index == HORIZON
        assert admissible(frame, window_ids, target_id)


def test_a_window_that_reaches_forward_is_refused_by_identity():
    """Shift the window one row into the future: the values do not change at all."""
    frame = series()
    for window_ids, target_id in windows(frame):
        target_index = frame.index[frame["row_id"] == target_id][0]
        forward = list(frame.iloc[target_index - LOOKBACK + 1:target_index + 1]["row_id"])
        assert target_id in forward, "fixture check: the shifted window does reach the target"
        assert not admissible(frame, forward, target_id), (
            "a window containing the observation it predicts must never be admissible, even "
            "though every value in this frame is identical")


def test_availability_later_than_the_event_moves_the_decision_time():
    """A row known an hour after it happened cannot be used as if it were known at once."""
    immediate = series(lag_minutes=0)
    # 150 minutes: the window's last row only becomes readable at 04:30, after the 04:00 bar
    # it would have predicted. 90 minutes would still have been in time, which is the point —
    # the boundary is arithmetic on the clocks, not a feeling about "some delay".
    delayed = series(lag_minutes=150)
    window_ids, target_id = windows(immediate)[0]
    assert decision_time(delayed, window_ids) > decision_time(immediate, window_ids)
    assert admissible(immediate, window_ids, target_id)
    assert not admissible(delayed, window_ids, target_id), (
        "with a 150-minute publication lag the target's own hour has already passed when "
        "window becomes readable: that decision could not have been made")


def test_a_longer_horizon_restores_admissibility_under_the_same_lag():
    """The delay does not forbid prediction; it forbids predicting THAT close."""
    delayed = series(lag_minutes=150)
    for window_ids, target_id in windows(delayed, horizon=4):
        assert admissible(delayed, window_ids, target_id)


def test_the_last_rows_produce_no_target_instead_of_an_invented_one():
    frame = series()
    produced = windows(frame)
    assert produced[-1][1] == frame["row_id"].iloc[-1]
    assert len(produced) == len(frame) - LOOKBACK + 1 - HORIZON


def test_extending_the_future_does_not_change_an_earlier_window():
    """Append-only: rows arriving later cannot alter a decision already taken."""
    short = series(rows=12)
    longer = series(rows=20)
    assert windows(short) == windows(longer)[:len(windows(short))]


def test_an_availability_column_declared_as_metadata_never_becomes_a_feature():
    """The link back to the contract: this clock is read, and is not model input."""
    from app.column_roles import resolve, select_features

    frame = series()
    contract = {"time": "event_time", "features": ["CLOSE"],
                "metadata": ["row_id", "available_time"]}
    plan = resolve({"column_roles": contract}, list(frame.columns))
    assert list(select_features(frame, plan).columns) == ["CLOSE"]
    assert "available_time" in plan.metadata
