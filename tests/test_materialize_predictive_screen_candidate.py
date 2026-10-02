"""Tests for the exploratory TRAIN-only predictive screen."""

import numpy as np

from tools.materialize_predictive_screen_candidate import screen


def test_screen_finds_leading_signal_and_is_deterministic():
    rng = np.random.default_rng(11)
    rows = 200
    lead = rng.normal(size=rows)
    windows = rng.normal(size=(rows, 4, 4))
    windows[:, -1, 2] = lead
    targets = rng.normal(0, 0.05, size=(rows, 2, 1))
    targets[:, :, 0] += lead[:, None]

    selected, scores = screen(windows, targets, ["a", "b", "lead", "d"],
                              top_k=1, fit_stop=150)
    repeated, _ = screen(windows, targets, ["a", "b", "lead", "d"],
                         top_k=1, fit_stop=150)

    assert selected == repeated == ["lead"]
    assert max(row["score"] for row in scores if row["feature"] != "lead") < 0.3
