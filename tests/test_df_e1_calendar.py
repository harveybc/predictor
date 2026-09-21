"""Phase 2 (calendar): a calendar input reads the row's own label and nothing after it.

Acceptance before any training: the production feature path is prefix-invariant to future values
and to future timestamps; a numeric timestamp, a missing label, an unparseable label and an absent
column are refused, never guessed; the grid and clock are reported for the slice, never fixed; and
the parameters the added channels bring are measured so the design can hold capacity constant.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("df_e1_calendar")
SPEC = C.CalendarSpec(timestamp_column="timestamp_label", ts_format="%d/%m/%Y %H:%M:%S", step_seconds=60)


def _frame(n=3000, start="2009-08-23 12:45:00"):
    ts = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({"timestamp_label": ts.strftime("%d/%m/%Y %H:%M:%S"),
                         "Global_active_power": np.random.default_rng(0).normal(1.0, 0.3, n)})


def test_features_are_a_function_of_the_label_only_and_bounded():
    out = C.build(_frame(), SPEC)
    f = out["features"]
    assert f.shape == (3000, 4) and out["names"] == list(C.FEATURES)
    assert np.all(np.abs(f) <= 1.0+1e-12)
    # sin^2 + cos^2 = 1 for each pair: the encoding is on the circle
    assert np.allclose(f[:, 0]**2+f[:, 1]**2, 1.0) and np.allclose(f[:, 2]**2+f[:, 3]**2, 1.0)


def test_prefix_invariance_future_values_and_future_timestamps_change_nothing_before_them():
    base = _frame()
    ref = C.build(base, SPEC)["features"]
    # 1. every value after row t changed: no feature moves anywhere
    shocked = base.copy()
    shocked.loc[1500:, "Global_active_power"] += 1000.0
    assert np.array_equal(C.build(shocked, SPEC)["features"], ref)
    # 2. every timestamp after row t shifted by a day: rows <= t keep their features exactly
    later = base.copy()
    ts = pd.to_datetime(later["timestamp_label"], format=SPEC.ts_format)
    ts[1500:] = ts[1500:] + pd.Timedelta(days=1)
    later["timestamp_label"] = ts.dt.strftime(SPEC.ts_format)
    moved = C.build(later, SPEC)["features"]
    assert np.array_equal(moved[:1500], ref[:1500])
    assert not np.array_equal(moved[1500:], ref[1500:])       # and the rows that DID change, changed


def test_a_deliberately_leaky_calendar_control_is_caught_by_the_same_prefix_test():
    """A feature that looks at the NEXT row's label (a shift-minus-one, the classic leak) fails the
    same invariance the honest path passes."""
    base = _frame()
    ts = pd.to_datetime(base["timestamp_label"], format=SPEC.ts_format)

    def leaky(ts):
        nxt = ts.shift(-1).fillna(ts.iloc[-1])
        return C.features(nxt, SPEC)
    ref = leaky(ts)
    later = ts.copy()
    later[1500:] = later[1500:] + pd.Timedelta(days=1)
    moved = leaky(later)
    assert not np.array_equal(moved[:1500], ref[:1500])        # row 1499 saw row 1500's shifted label


@pytest.mark.parametrize("case", ["numeric", "missing", "unparseable", "absent"])
def test_refusals_never_guess(case):
    f = _frame(200)
    if case == "numeric":
        f["timestamp_label"] = np.arange(200)
    elif case == "missing":
        f.loc[10, "timestamp_label"] = None
    elif case == "unparseable":
        f.loc[10, "timestamp_label"] = "2009-08-23 12:55:00"    # a different format, silently wrong if parsed
    else:
        f = f.drop(columns=["timestamp_label"])
    with pytest.raises(C.CalendarRefusal):
        C.build(f, SPEC)


def test_the_grid_and_clock_are_reported_not_fixed():
    f = _frame(500)
    f = pd.concat([f.iloc[:200], f.iloc[260:]], ignore_index=True)    # a one-hour gap, as a DST spring would leave
    g = C.build(f, SPEC)["grid"]
    assert g["rows_off_grid"] == 1 and g["duplicated_labels"] == 0
    assert g["clock"] == "NAIVE_WALL_CLOCK" and "UNKNOWN" in g["timezone"]
    assert "not closed" in g["dst_note"]


def test_the_parameter_delta_of_added_channels_is_measured():
    P = _load("df_e1_pilot")
    design_graph = {"assignment": [0, 1, 0, 2, 2, 2, 0], "core_kind": "tcn_w"}

    def builder(window, channels, **kw):
        assignment = design_graph["assignment"] + [3]*(channels-7)       # calendar channels as one extra group
        return P._model_for_target(assignment, window, channels, 6, 1, core=design_graph["core_kind"])
    d = C.parameter_delta(builder, 60, 7, 4)
    assert d["parameters_before"] == 8127 and d["delta"] > 0
    assert "control arm" in d["reading"]
