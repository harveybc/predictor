"""The production transformations, run against perturbations of the future.

R2 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "Retain toy controls as controls... Enumerate the plugins actually selected by the four
     smoke configurations... Import and execute those entry points."

The previous battery defined its own `zscore_fit`, `windows` and a three-tap convolution, so
the real scaler and the real window builder could regress without a single rule turning red.
These rules import what the smoke configurations actually select and run it:

| configuration | entry point exercised here |
|---|---|
| predictor `phase_1_ann_1575_1d` | `preprocessor_plugins.sliding_windows.create_sliding_windows` |
| the same | `preprocessor_plugins.stl_preprocessor.PreprocessorPlugin._add_window_stats_features` |
| the same | `preprocessor_plugins.stl_preprocessor.PreprocessorPlugin._align_sliding_windows_with_targets` |

The method is one deterministic chronological trajectory and a perturbation of its FUTURE:
append rows, change future values, punch a hole in the future. Whatever the transformation
computes for the admissible prefix must not move. Values are compared exactly, not by shape.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from preprocessor_plugins.sliding_windows import create_sliding_windows
from preprocessor_plugins.stl_preprocessor import PreprocessorPlugin

WINDOW = 8
PREFIX = 40


def trajectory(rows=60, seed=20260915, tail_shift=0.0, hole=None):
    """One chronological trajectory; `tail_shift` and `hole` only ever touch the future.

    Deterministic without a global seed: an explicit xorshift, so the prefix of a longer
    trajectory is byte-identical to the shorter one.
    """
    state = seed
    values = []
    level = 100.0
    for index in range(rows):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        level += ((state % 1000) / 1000.0 - 0.5)
        value = level
        if index >= PREFIX:
            value += tail_shift
        if hole is not None and index >= hole:
            value = float("nan")
        values.append(value)
    frame = pd.DataFrame({
        "typical_price": values,
        "OPEN": [value if value == value else float("nan") for value in values],
    })
    dates = pd.date_range("2024-01-01", periods=rows, freq="h")
    return frame, dates


def windows_of(frame, dates, rows=None):
    """The REAL window builder, on the real dictionary shape the pipeline passes it."""
    data = {"x_train_df": frame if rows is None else frame.iloc[:rows]}
    stamps = dates if rows is None else dates[:rows]
    config = {"window_size": WINDOW, "use_returns": False}
    return create_sliding_windows(data, config, stamps)


def x_train(result):
    for key in ("X_train", "x_train", "X_train_windows"):
        if isinstance(result, dict) and key in result and result[key] is not None:
            return np.asarray(result[key])
    raise AssertionError(f"the real builder returned no train windows: {list(result)}")


def test_the_real_window_builder_is_invariant_to_rows_that_arrive_later():
    """Append-only: the windows of the first 40 rows must not move when 20 more arrive."""
    frame, dates = trajectory()
    short = x_train(windows_of(frame, dates, rows=PREFIX))
    long = x_train(windows_of(frame, dates))
    assert short.shape[0] <= long.shape[0]
    np.testing.assert_array_equal(short, long[:short.shape[0]])


def test_the_real_window_builder_ignores_a_changed_future():
    """Same prefix, different future values: every admissible window is bit-identical."""
    base, dates = trajectory()
    moved, _ = trajectory(tail_shift=25.0)
    pd.testing.assert_frame_equal(base.iloc[:PREFIX], moved.iloc[:PREFIX])
    admissible = PREFIX - WINDOW + 1
    np.testing.assert_array_equal(x_train(windows_of(base, dates))[:admissible],
                                  x_train(windows_of(moved, dates))[:admissible])


def test_the_real_window_builder_ignores_a_hole_punched_in_the_future():
    base, dates = trajectory()
    gapped, _ = trajectory(hole=PREFIX + 5)
    admissible = PREFIX - WINDOW + 1
    np.testing.assert_array_equal(x_train(windows_of(base, dates))[:admissible],
                                  x_train(windows_of(gapped, dates))[:admissible])


def stats_of(windows_result, feature_names):
    """The REAL rolling std / EMA / price-minus-EMA step, executed on real windows."""
    plugin = PreprocessorPlugin()
    payload = dict(windows_result)
    payload["feature_names"] = list(feature_names)
    plugin._add_window_stats_features(payload, {"add_window_stats": True,
                                                "window_stats_periods": [4],
                                                "target_column": "typical_price"})
    return np.asarray(payload["X_train"]), payload["feature_names"]


def check_future_does_not_move_the_filters():
    """The rule itself, as a callable, so the mutation test runs THIS check and not a copy."""
    base, dates = trajectory()
    moved, _ = trajectory(tail_shift=25.0)
    names = ["typical_price", "OPEN"]
    base_stats, produced = stats_of(windows_of(base, dates), names)
    moved_stats, _ = stats_of(windows_of(moved, dates), names)
    assert produced[len(names):] == ["rolling_std_4", "rolling_ema_4", "price_minus_ema_4"]
    admissible = PREFIX - WINDOW + 1
    np.testing.assert_array_equal(base_stats[:admissible], moved_stats[:admissible])


def test_the_real_rolling_filters_do_not_read_the_future():
    """std, EMA and price-minus-EMA recomputed after the future changes: same numbers.

    These are the deployed filters, not a three-tap substitute. If one of them ever reached
    outside its own window, moving the tail by 25 would move these values.
    """
    check_future_does_not_move_the_filters()


def test_the_real_rolling_filters_are_constant_inside_a_window_as_documented():
    """The docstring says the statistics are broadcast to every timestep: check it."""
    frame, dates = trajectory()
    stats, names = stats_of(windows_of(frame, dates), ["typical_price", "OPEN"])
    index = names.index("rolling_std_4")
    column = stats[:, :, index]
    assert np.allclose(column, column[:, :1]), (
        "a statistic that varies inside its own window is reading different rows per timestep")


def test_the_real_filters_have_a_warm_up_and_it_is_measured_not_assumed():
    """Period 4 inside a window of 8: the first rows of a window have less history.

    What is recorded here is the measurement, not a promise: the statistic is computed per
    window from the window's own rows, so its warm-up is bounded by the window, and a window
    is never padded into existence.
    """
    frame, dates = trajectory()
    result = windows_of(frame, dates)
    stats, names = stats_of(result, ["typical_price", "OPEN"])
    assert stats.shape[0] == x_train(result).shape[0], "no window was invented or dropped"
    assert stats.shape[1] == WINDOW
    assert np.isfinite(stats[:, :, names.index("rolling_ema_4")]).all(), (
        "the deployed EMA emits a number for every window it is given; a NaN here would be a "
        "warm-up the pipeline does not declare")


def test_a_mutated_filter_makes_this_very_rule_fail(monkeypatch):
    """Musashi's check on the check, applied to the PRODUCTIVE path.

    The deployed `_add_window_stats_features` is replaced by a version that differs in one
    way: the standard deviation is taken over the whole series instead of over the window it
    belongs to. That is the classic leak. `check_future_does_not_move_the_filters` — the same
    callable the rule above runs — must turn red, or it was never testing anything.
    """
    original = PreprocessorPlugin._add_window_stats_features

    def leaking(self, sliding_windows, config):
        original(self, sliding_windows, config)
        names = sliding_windows.get("feature_names", [])
        if "rolling_std_4" not in names:
            return
        array = np.asarray(sliding_windows["X_train"])
        index = names.index("rolling_std_4")
        array[:, :, index] = float(np.nanstd(array[:, :, names.index("typical_price")]))
        sliding_windows["X_train"] = array

    monkeypatch.setattr(PreprocessorPlugin, "_add_window_stats_features", leaking)
    with pytest.raises(AssertionError):
        check_future_does_not_move_the_filters()


def test_the_real_alignment_trims_from_the_end_and_never_invents_a_window():
    """`_align_sliding_windows_with_targets`, the deployed step, on real windows.

    Targets run out before windows do: the last windows have no future to be scored against.
    The real step must drop those from the END — keeping the oldest windows, which are the
    ones whose targets exist — and must never pad the count back up.
    """
    frame, dates = trajectory()
    result = windows_of(frame, dates)
    windows = x_train(result)
    available = windows.shape[0] - 3
    targets = {"y_train": {"output_horizon_1": np.arange(available)}}

    aligned = PreprocessorPlugin()._align_sliding_windows_with_targets(
        dict(result), targets, {"predicted_horizons": [1]})
    kept = np.asarray(aligned["X_train"])
    assert kept.shape[0] == available, "alignment must not invent or lose a window"
    np.testing.assert_array_equal(kept, windows[:available])


def test_the_real_alignment_leaves_a_shorter_window_set_alone():
    """Fewer windows than targets is not fixed by padding: the windows stay as they are."""
    frame, dates = trajectory()
    result = windows_of(frame, dates)
    windows = x_train(result)
    targets = {"y_train": {"output_horizon_1": np.arange(windows.shape[0] + 10)}}
    aligned = PreprocessorPlugin()._align_sliding_windows_with_targets(
        dict(result), targets, {"predicted_horizons": [1]})
    np.testing.assert_array_equal(np.asarray(aligned["X_train"]), windows)
