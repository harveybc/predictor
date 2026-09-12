"""C72: the independent recomputation is independent, declared in
advance, and correct on inputs whose answers are known by hand."""
from __future__ import annotations

import ast
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import lake_descriptor_recompute as R  # noqa: E402


def test_the_module_imports_no_producer_code():
    tree = ast.parse((REPO / "tools/lake_descriptor_recompute.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    assert mods <= {"__future__", "math", "zlib", "numpy"}, mods


@pytest.mark.parametrize("rows,total,used,capped", [
    (193465, 193465, 135425, False),
    (682378, 200000, 140000, True),
    (595, 595, 416, False),
    (56612, 56612, 39628, False),
])
def test_window_matches_published_window_contracts(rows, total, used, capped):
    w = R.window_rows(rows)
    assert (w["rows_total"], w["rows_used"], w["capped"]) == (total, used, capped)


def test_counts_and_missingness_on_a_hand_computed_series():
    w = np.array([1.0, np.nan, 2.0, np.inf, 2.0])
    out = R.recompute(w)
    assert out["n_observations"][0] == 5
    assert out["missing_count"][0] == 1
    assert out["non_finite_count"][0] == 2
    assert out["missingness_fraction"][0] == pytest.approx(1 - 3 / 5)


def test_distribution_descriptors_on_known_values():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    out = R.recompute(w)
    assert out["mean"][0] == 2.5
    assert out["std"][0] == pytest.approx(np.sqrt(5 / 3))
    assert (out["min"][0], out["max"][0], out["median"][0]) == (1.0, 4.0, 2.5)
    assert out["constant"][0] == 0.0
    assert out["duplicate_count"][0] == 0


def test_autocorrelation_skips_missing_pairs_without_closing_up():
    w = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
    v, ok = R.recompute(w)["autocorrelation_lag1"]
    a = np.array([1.0, 4.0, 5.0, 6.0])
    b = np.array([2.0, 5.0, 6.0, 7.0])
    assert ok and v == pytest.approx(np.corrcoef(a, b)[0, 1])


def test_compression_ratio_and_entropy():
    f = np.array([0.0, 1.0, 0.0, 1.0])
    out = R.recompute(f)
    raw = f.astype("<f8").tobytes()
    assert out["compressed_length_ratio"][0] == len(zlib.compress(raw, 9)) / len(raw)
    assert out["discrete_entropy_bits"][0] == pytest.approx(1.0)


def test_fewer_than_three_finite_values_is_not_identifiable():
    out = R.recompute(np.array([1.0, np.nan, 2.0]))
    assert out["insufficient_finite_observations"] == (None, False)
    assert "mean" not in out


def test_conditional_descriptors_become_underspecified_with_missing_values():
    assert R.specificity("mean", 0) == (R.FULLY, None)
    level, why = R.specificity("mean", 3)
    assert level == R.UNDERSPECIFIED and "non-finite" in why


def test_underspecified_descriptors_are_never_upgraded():
    for d in ("p01", "p99", "iqr", "mad", "spectral_centroid",
              "window_mean_dispersion", "difference_to_level_dispersion"):
        assert R.specificity(d, 0)[0] == R.UNDERSPECIFIED


def test_descriptors_without_a_declared_reading_are_not_computed():
    out = R.recompute(np.arange(50, dtype=float))
    assert "spectral_centroid" not in out
    assert "window_mean_dispersion" not in out


def test_tolerance_is_exact_for_counts_and_relative_for_reals():
    assert not R.agree("missing_count", 3.0, 3.0000000001)
    assert R.agree("mean", 1e6, 1e6 * (1 + 5e-10))
    assert not R.agree("mean", 1.0, 1.0 + 1e-6)
    assert R.agree("mean", None, None) and not R.agree("mean", None, 0.0)
    assert R.agree("std", math.nan, math.nan)
