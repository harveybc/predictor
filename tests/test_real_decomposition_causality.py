"""The real STL decomposition: what it is, and what it must not be admitted as.

R2 of `docs/handoffs/MUSASHI_TO_SATOSHI_REAL_PLUGIN_CAUSALITY_AND_REPLAY_2026_09_14.md`:

    "For each actual wavelet/STL/filter path, test boundary extension, padding and
     reconstruction on full series versus admissible prefixes. A three-tap substitute does not
     test that implementation. A noncausal operator can remain an offline oracle, but cannot
     be admitted to a causal configuration under a renamed label."

Inventory first, because it decides what this file can honestly claim. The names
`stl_pipeline` and `stl_preprocessor` decompose nothing: `STLPreprocessorZScore` neither
imports `statsmodels` nor computes a seasonal component. Two real operators exist:

* `target_plugins/stl_target.py` — the genuine `statsmodels.tsa.seasonal.STL`, applied to the
  TARGET series;
* `tools/df_snr.py::_offline_wavelet_mad_kernel` — a db4 transform with periodization, already
  declared `OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL` and guarded so it cannot produce a
  per-timestamp value.

Both are exercised here.

The rules below measure it rather than assume it, and the measurement is the point: STL fits
over the whole series it is handed, so the value it reports at row *t* depends on rows after
*t*. It is a legitimate offline oracle — for describing history, for diagnosing a regime —
and it is not admissible as a feature or a label in a causal configuration unless it is
recomputed from the admissible prefix at each decision point.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("statsmodels")

from target_plugins.stl_target import TargetPlugin

PERIOD = 12
PREFIX = 96
ROWS = 144


def signal(rows=ROWS, tail_shift=0.0, prefix=PREFIX):
    """Trend plus a clean seasonal cycle; `tail_shift` only ever touches the future."""
    index = np.arange(rows, dtype=float)
    values = 0.05 * index + np.sin(2 * np.pi * index / PERIOD)
    values[prefix:] += tail_shift
    return values


def decompose(series):
    plugin = TargetPlugin()
    plugin.set_params(stl_period=PERIOD, stl_seasonal=13, stl_trend=None)
    return plugin._decompose_signal(np.asarray(series, dtype=float))


def test_the_decomposition_reconstructs_the_series_it_was_given():
    """trend + seasonal + residual == input, to floating point. The identity STL guarantees."""
    values = signal()
    trend, seasonal, resid = decompose(values)
    np.testing.assert_allclose(np.asarray(trend) + np.asarray(seasonal) + np.asarray(resid),
                               values, rtol=0, atol=1e-8)


def test_the_decomposition_emits_a_value_for_every_row_including_the_edges():
    """No warm-up hole and no padding: the edges are estimated, not left empty."""
    values = signal()
    for component in decompose(values):
        component = np.asarray(component)
        assert component.shape == values.shape
        assert np.isfinite(component).all()


def test_the_decomposition_reads_the_future_and_this_is_measured_not_assumed():
    """The finding: change only rows after 96 and the value AT row 0 moves.

    This is not a defect in statsmodels — it is what a two-sided decomposition does. It is
    recorded here so that no configuration can later present this component as something that
    was knowable at its own timestamp.
    """
    base_trend, base_seasonal, _ = decompose(signal())
    moved_trend, moved_seasonal, _ = decompose(signal(tail_shift=5.0))
    assert not np.allclose(np.asarray(base_trend)[:PREFIX], np.asarray(moved_trend)[:PREFIX]), (
        "if this ever passes, STL has become causal and this rule must be rewritten, not "
        "deleted")
    first_row_drift = abs(float(np.asarray(base_trend)[0]) - float(np.asarray(moved_trend)[0]))
    assert first_row_drift > 1e-6, (
        f"the future moved the very first row by {first_row_drift:.6f}: the operator is "
        "two-sided and cannot be a causal feature")
    # the seasonal part moves too, so neither component escapes the finding
    assert not np.allclose(np.asarray(base_seasonal)[:PREFIX],
                           np.asarray(moved_seasonal)[:PREFIX])


def test_a_consistent_future_barely_changes_the_past_estimate():
    """Measured, against my own expectation.

    I expected decomposing the prefix alone to differ from decomposing everything. It does
    not, when the future continues the same pattern: the gap on this signal is ~3.6e-14, i.e.
    floating-point noise. The operator's dependence on the future is real but conditional —
    it bites when the future DEVIATES, which is the next rule. Recording the measurement
    rather than the expectation is the point.
    """
    values = signal()
    full_trend, _, _ = decompose(values)
    prefix_trend, _, _ = decompose(values[:PREFIX])
    gap = float(np.max(np.abs(np.asarray(full_trend)[:PREFIX] - np.asarray(prefix_trend))))
    assert gap < 1e-9, f"expected agreement on a consistent continuation, measured {gap:g}"


def test_a_deviating_future_moves_the_past_estimate_and_worst_at_the_edge():
    """The size of the mistake a run makes by taking a full-series component as knowable."""
    values = signal(tail_shift=5.0)
    full_trend, _, _ = decompose(values)
    prefix_trend, _, _ = decompose(values[:PREFIX])
    full_on_prefix = np.asarray(full_trend)[:PREFIX]
    prefix_only = np.asarray(prefix_trend)
    gap = np.abs(full_on_prefix - prefix_only)
    assert float(np.max(gap)) > 1e-3, "a jump in the future must reach back into the estimate"
    edge = float(np.max(gap[-PERIOD:]))
    interior = float(np.max(gap[:PREFIX // 2]))
    assert edge > interior, (
        "boundary behaviour: the estimates nearest the end of the available data are the ones "
        "the missing future changes most")


def test_a_broken_series_comes_back_as_NaN_and_the_run_continues():
    """What the failure paths actually do, measured against two wrong expectations of mine.

    I expected a four-point series to fail: it does not, STL returns a constant trend. I then
    expected a NaN to trip the `except` and return zeros: it does not either — statsmodels
    propagates, so every component comes back all-NaN. Both are recorded as they are.

    The finding that matters for a run: a decomposition can fail to mean anything without
    failing loudly. `_decompose_signal` catches every exception and substitutes zeros, and a
    NaN never even reaches that branch; downstream, an all-zero or all-NaN component is
    indistinguishable from a series with no trend, and no artifact records which happened.
    """
    short = decompose(np.arange(4, dtype=float))
    assert not np.all(np.asarray(short[0]) == 0), "a short series is decomposed, not refused"

    broken = signal().copy()
    broken[10] = np.nan
    trend, seasonal, resid = decompose(broken)
    assert np.isnan(np.asarray(trend)).all(), "one NaN contaminates the whole component"
    assert np.isnan(np.asarray(seasonal)).all()
    assert np.isnan(np.asarray(resid)).all()


def test_the_zero_substitution_does_not_actually_return_zeros(capsys):
    """The `except` path, driven directly — and it does not do what it says.

    `_decompose_signal` catches everything and runs `np.zeros_like(series)`. For the only
    input I could make reach that branch, a non-numeric one, `zeros_like` returns an array of
    the INPUT's dtype: `array('', dtype='<U2')`, not a numeric zero, and not even the right
    length. The printed promise is "Returning 0s for components"; what a caller receives is a
    one-element string array that will break or silently corrupt whatever consumes it.

    Recorded, not fixed: this is a target plugin no smoke configuration selects, so changing
    its behaviour belongs to a separate, declared change rather than to R2's measurement.
    """
    plugin = TargetPlugin()
    plugin.set_params(stl_period=PERIOD, stl_seasonal=13, stl_trend=None)
    trend, seasonal, resid = plugin._decompose_signal("no")
    assert "Returning 0s" in capsys.readouterr().out, "the branch under test did run"
    assert np.asarray(trend).dtype.kind == "U", (
        "zeros_like of a string returns a string: the advertised zeros never appear")
    assert np.asarray(trend).size == 1


@pytest.mark.parametrize("values", [np.array([], dtype=float), np.array([1.0, 2.0]),
                                    np.where(np.arange(ROWS) == 5, np.inf,
                                             np.arange(ROWS, dtype=float))])
def test_no_numeric_input_i_tried_reaches_the_failure_branch(values):
    """Declared scope of the finding above: empty, two-point and infinite series all decompose.

    So the silent-substitution risk is narrower than "any failure": it needs an input that is
    not numeric at all. Saying which inputs were tried is part of the measurement.
    """
    trend, _, _ = decompose(values)
    assert np.asarray(trend).shape == values.shape


# --- the real wavelet operator ------------------------------------------------------------

def test_the_wavelet_estimator_refuses_to_produce_a_per_timestamp_value():
    """The guard is the contract: db4/periodization may only run as an offline aggregate."""
    from tools import df_snr

    with pytest.raises(df_snr.SnrRefusal, match="OFFLINE_TRAIN_DIAGNOSTIC_NON_CAUSAL"):
        df_snr._offline_wavelet_mad_kernel(
            signal(), {"wavelet": "db4", "mode": "periodization", "level": 1})


def test_the_declared_wavelet_transform_reconstructs_what_it_was_given():
    """Reconstruction, on the exact parameters the estimator declares."""
    pywt = pytest.importorskip("pywt")

    values = signal()
    approx, detail = pywt.dwt(values, "db4", mode="periodization")
    restored = pywt.idwt(approx, detail, "db4", mode="periodization")
    np.testing.assert_allclose(restored[:len(values)], values, rtol=0, atol=1e-8)


def test_the_declared_wavelet_transform_lets_the_last_sample_move_the_first_coefficient():
    """Periodization wraps the series onto itself, which is why the kernel is guarded.

    The docstring of `_offline_wavelet_mad_kernel` says "the last sample moves the first
    coefficients". This measures it instead of trusting it.
    """
    pywt = pytest.importorskip("pywt")

    values = signal()
    changed = values.copy()
    changed[-1] += 50.0
    _, base_detail = pywt.dwt(values, "db4", mode="periodization")
    _, moved_detail = pywt.dwt(changed, "db4", mode="periodization")
    drift = float(abs(base_detail[0] - moved_detail[0]))
    assert drift > 1e-6, (
        f"the first detail coefficient moved by {drift:.6f} when only the LAST sample changed: "
        "this operator cannot be admitted to a causal configuration under any label")
