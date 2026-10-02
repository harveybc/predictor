"""Tests for the exploratory TRAIN-only predictive screen."""

import numpy as np
import pytest

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


def test_contiguous_grouping_is_balanced_and_unknown_is_rejected():
    from tools.materialize_predictive_screen_candidate import group_selected

    selected = [f"f{i}" for i in range(8)]
    windows = np.zeros((3, 2, 8))
    assert [len(group) for group in group_selected(windows, selected, 3, "contiguous")] == [3, 3, 2]
    with pytest.raises(ValueError, match="correlation or contiguous"):
        group_selected(windows, selected, 3, "unknown")
