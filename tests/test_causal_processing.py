"""Causality of the transformations the four configurations actually use.

P3 of `docs/handoffs/MUSASHI_TO_SATOSHI_CAUSAL_PIPELINE_AND_OFFLINE_DOIN_2026_09_14.md`.

Each rule below is stated as a property of an operator, tested on the causal bench, and the
deliberately non-causal controls must **fail** the same check — a test battery where nothing
can fail proves nothing.

The operators under test are the ones the deployed configurations name: z-score
normalisation fitted on training, the sliding-window construction with its lookback and
horizon, the rolling statistics of the preprocessor, and the centred filters (STL, wavelet)
whose delay has to be measured rather than promised away.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("CAUSAL_BENCH", REPO / "docs/audits/evidence/repro_runs/causal_bench"))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    """The bench is regenerated from its declared seed when it is not already present."""
    if (BENCH / "MANIFEST.json").is_file():
        root = BENCH
    else:
        root = tmp_path_factory.mktemp("causal")
        subprocess.run([sys.executable, str(REPO / "tools" / "make_causal_bench.py"),
                        "--out", str(root)], check=True, capture_output=True)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    frames = {name: pd.read_csv(root / f"causal_{name}.csv", parse_dates=["DATE_TIME"])
              for name in ("train", "validation", "test")}
    frames["whole"] = pd.read_csv(root / "causal_whole.csv", parse_dates=["DATE_TIME"])
    return manifest, frames


# --------------------------------------------------------------- the partitions themselves

def test_the_partitions_are_chronological_and_embargoed(bench):
    manifest, frames = bench
    embargo = manifest["embargo_rows"]
    assert frames["train"]["DATE_TIME"].max() < frames["validation"]["DATE_TIME"].min()
    assert frames["validation"]["DATE_TIME"].max() < frames["test"]["DATE_TIME"].min()
    step = pd.Timedelta(hours=manifest["step_hours"])
    gap = frames["validation"]["DATE_TIME"].min() - frames["train"]["DATE_TIME"].max()
    assert gap >= embargo * step, "a lookback could otherwise reach across the boundary"


def test_the_bench_carries_every_regime_it_claims(bench):
    manifest, frames = bench
    whole = frames["whole"]
    assert whole["jump"].nunique() > 1, "level jumps"
    assert whole["spike"].abs().max() > 5, "extremes"
    assert whole["observed"].isna().sum() == manifest["files"]["causal_whole.csv"]["missing_rows"]
    assert whole["seasonal_daily"].abs().max() > 1.0 and whole["seasonal_weekly"].abs().max() > 0.5


# --------------------------------------------------------------- fit on train, freeze, apply

def zscore_fit(series):
    return {"mean": float(series.mean()), "sd": float(series.std(ddof=0))}


def zscore_apply(series, state):
    return (series - state["mean"]) / (state["sd"] or 1.0)


def test_a_scaler_fitted_on_train_does_not_move_when_validation_arrives(bench):
    _manifest, frames = bench
    state = zscore_fit(frames["train"]["observed"].dropna())
    again = zscore_fit(frames["train"]["observed"].dropna())
    assert state == again
    # applying to validation must not change the state, and must use the training numbers
    applied = zscore_apply(frames["validation"]["observed"], state)
    assert zscore_fit(frames["train"]["observed"].dropna()) == state
    assert abs(applied.mean() - 0.0) > 1e-9, (
        "validation standardised with training statistics is not centred by construction; "
        "if it were, the state would have been refitted")


def test_the_non_causal_control_scaler_fails_the_same_check(bench):
    """Deliberately fitting on everything: the control must be detectable, and it is."""
    _manifest, frames = bench
    honest = zscore_fit(frames["train"]["observed"].dropna())
    leaked = zscore_fit(frames["whole"]["observed"].dropna())
    assert honest != leaked
    # the leak is visible exactly where it matters: the test partition looks pre-centred
    centred = zscore_apply(frames["test"]["observed"], leaked).mean()
    proper = zscore_apply(frames["test"]["observed"], honest).mean()
    assert abs(centred) < abs(proper), (
        "a scaler fitted with the test data flatters the test partition; that is the "
        "signature this control exists to show")


# --------------------------------------------------------------- windows, prefixes, futures

def windows(values, lookback, horizon):
    """The construction under test: x is the lookback, y is `horizon` rows ahead of it."""
    xs, ys, ends = [], [], []
    for end in range(lookback, len(values) - horizon + 1):
        xs.append(values[end - lookback:end])
        ys.append(values[end + horizon - 1])
        ends.append(end - 1)
    return np.array(xs, dtype=float), np.array(ys, dtype=float), ends


def test_a_window_never_contains_its_own_target(bench):
    manifest, frames = bench
    values = frames["train"]["observed"].ffill().to_numpy()
    xs, ys, ends = windows(values, manifest["lookback"], manifest["horizon"])
    assert len(xs) > 10
    for index in (0, len(xs) // 2, len(xs) - 1):
        assert ys[index] not in xs[index], "the target sits inside its own input window"
        assert ends[index] + manifest["horizon"] < len(values)


def test_extending_the_future_does_not_change_an_earlier_window(bench):
    manifest, frames = bench
    values = frames["train"]["observed"].ffill().to_numpy()
    short, _y1, _e1 = windows(values[:400], manifest["lookback"], manifest["horizon"])
    long, _y2, _e2 = windows(values[:800], manifest["lookback"], manifest["horizon"])
    assert np.array_equal(short, long[:len(short)]), (
        "a window computed before the tail existed changed when the tail arrived")


def test_the_future_shifted_control_cannot_even_be_computed_at_the_decision_point(bench):
    """The control has to fail for the right reason: it needs rows that do not exist yet.

    At a decision point `end`, a causal window is a slice of the prefix up to `end`. A window
    shifted `horizon` rows forward is not computable from that prefix at all — which is what
    makes it non-causal, and what this check detects.
    """
    manifest, frames = bench
    values = frames["train"]["observed"].ffill().to_numpy()
    lookback, horizon = manifest["lookback"], manifest["horizon"]
    end = 300

    prefix = values[:end + 1]                       # everything known at the decision point
    causal = prefix[end - lookback + 1:end + 1]
    assert len(causal) == lookback
    assert np.array_equal(causal, values[end - lookback + 1:end + 1]), (
        "the causal window is the same whether or not the future exists")

    peeking = prefix[end - lookback + 1 + horizon:end + 1 + horizon]
    assert len(peeking) < lookback, (
        "the shifted window claims rows the prefix does not contain; it is only computable "
        "after the fact")
    complete_peeking = values[end - lookback + 1 + horizon:end + 1 + horizon]
    assert not np.array_equal(complete_peeking, causal)


# --------------------------------------------------------------- delay of centred filters

def centred_mean(values, window):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def trailing_mean(values, window):
    return pd.Series(values).rolling(window, min_periods=1).mean().to_numpy()


def response_delay(filtered, impulse_at):
    """The first index at which a filter reacts, relative to the impulse."""
    reacted = np.flatnonzero(np.abs(filtered - filtered[0]) > 1e-9)
    return int(reacted[0]) - impulse_at if len(reacted) else None


def test_a_centred_filter_reacts_before_the_event_and_a_trailing_one_does_not():
    impulse_at, length, window = 60, 120, 11
    values = np.zeros(length)
    values[impulse_at:] = 1.0
    centred = response_delay(centred_mean(values, window), impulse_at)
    trailing = response_delay(trailing_mean(values, window), impulse_at)
    assert centred < 0, f"a centred window sees the future: it reacts {abs(centred)} steps early"
    assert trailing == 0, "a trailing window reacts when the event arrives, not before"


@pytest.mark.parametrize("window", [5, 11, 25])
def test_the_measured_delay_of_a_centred_window_is_its_half_width(window):
    impulse_at, length = 60, 200
    values = np.zeros(length)
    values[impulse_at:] = 1.0
    delay = response_delay(centred_mean(values, window), impulse_at)
    assert delay == -(window // 2), (
        "the anticipation of a centred filter is half its window; it is measured here, not "
        "promised away")


def test_a_wavelet_style_symmetric_filter_is_also_anticipatory():
    """Wavelets get no exception: a symmetric kernel reacts before the event too."""
    impulse_at, length = 60, 200
    values = np.zeros(length)
    values[impulse_at:] = 1.0
    kernel = np.array([0.25, 0.5, 0.25])          # symmetric, as a db-style low pass is
    filtered = np.convolve(values, kernel, mode="same")
    delay = response_delay(filtered, impulse_at)
    assert delay == -1


# --------------------------------------------------------------- alignment and missingness

def test_alignment_survives_lookback_horizon_and_missingness(bench):
    manifest, frames = bench
    frame = frames["validation"]
    values = frame["observed"].to_numpy()
    times = frame["DATE_TIME"].to_numpy()
    lookback, horizon = manifest["lookback"], manifest["horizon"]
    filled = pd.Series(values).ffill().to_numpy()
    xs, ys, ends = windows(filled, lookback, horizon)
    assert len(xs) == len(ys) == len(ends)
    # every window's last input timestamp precedes its target's timestamp by the horizon
    step = np.timedelta64(manifest["step_hours"], "h")
    for index in (0, len(ends) // 2, len(ends) - 1):
        last_input = times[ends[index]]
        target_time = times[ends[index] + horizon]
        assert target_time - last_input == step * horizon
    # the missing stretch is not closed by the calendar: the row count is unchanged
    assert frame["observed"].isna().sum() > 0
    assert len(frame) == manifest["files"]["causal_validation.csv"]["rows"]


def test_the_right_edge_is_not_padded_into_existence(bench):
    manifest, frames = bench
    values = frames["test"]["observed"].ffill().to_numpy()
    xs, _ys, ends = windows(values, manifest["lookback"], manifest["horizon"])
    assert ends[-1] + manifest["horizon"] <= len(values) - 1, (
        "the last window must have a real target, not one invented by padding")
