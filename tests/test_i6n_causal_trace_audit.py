"""Focused tests for tools/i6n_causal_trace_audit.py (synthetic data only).

No campaign file, model, GPU or TEST row is touched here. Every check that the
audit runner applies to the retained evidence is exercised on tiny generated
arrays so that it demonstrably fails closed: a future-row perturbation that
changes a window, a mask identity mismatch with equal counts, a digest
mismatch, a scaler fitted across its boundary, a feature value available
before its source date, and a selection made after the fold cutoff.
"""

import hashlib
import json
import math

import numpy as np
import pytest

from tools import i6n_causal_trace_audit as audit


def _spec(tmp_path, **override):
    spec = dict(audit.MINIMAL_SPEC)
    spec.update(override)
    path = tmp_path / "SPEC.json"
    path.write_text(json.dumps(spec))
    return path


def test_spec_requires_declared_tolerances_before_running(tmp_path):
    spec = audit.load_spec(_spec(tmp_path))
    assert spec["tolerances"]["mutation"] == "exact"
    assert spec["tolerances"]["digest"] == "exact"
    bad = dict(audit.MINIMAL_SPEC)
    bad["tolerances"] = dict(bad["tolerances"])
    del bad["tolerances"]["metric_replay"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="tolerance"):
        audit.load_spec(path)


def test_status_is_written_atomically_with_counts_per_state(tmp_path):
    findings = audit.Findings()
    findings.add("A", "s", "x", audit.PROVEN_SAFE, "e", [])
    findings.add("B", "s", "y", audit.RISK_UNRESOLVED, "e", [])
    findings.add("C", "s", "z", audit.PROVEN_LEAK, "e", [])
    status = audit.Status(tmp_path, findings)
    status.write("RUNNING", phase="p1")
    assert not list(tmp_path.glob("*.tmp"))
    loaded = json.loads((tmp_path / "STATUS.json").read_text())
    assert loaded["state"] == "RUNNING" and loaded["phase"] == "p1"
    assert loaded["counts"] == {audit.PROVEN_SAFE: 1, audit.PROVEN_LEAK: 1,
                                audit.RISK_UNRESOLVED: 1, audit.NOT_APPLICABLE: 0}
    with pytest.raises(ValueError):
        findings.add("D", "s", "w", "SAFE", "e", [])


def test_states_are_counted_never_averaged():
    findings = audit.Findings()
    for state in (audit.PROVEN_SAFE, audit.PROVEN_SAFE, audit.RISK_UNRESOLVED):
        findings.add(f"id{len(findings.rows)}", "s", "a", state, "e", [])
    counts = findings.counts()
    assert sum(counts.values()) == 3
    assert "mean" not in json.dumps(counts) and findings.worst() == audit.RISK_UNRESOLVED
    findings.add("leak", "s", "a", audit.PROVEN_LEAK, "e", [])
    assert findings.worst() == audit.PROVEN_LEAK


def test_window_timestamps_rightmost_is_origin_and_grid_is_right_anchored():
    stamps = audit.window_timestamps(1_000_000 * 3600, samples=7, spacing_hours=4)
    assert stamps[-1] == 1_000_000 * 3600
    assert all(b - a == 4 * 3600 for a, b in zip(stamps, stamps[1:]))
    assert all(s < stamps[-1] for s in stamps[:-1])


def _grid(n=48, width=3):
    ts = np.arange(n, dtype=np.int64) * 3600 + 1_600_000_000
    rng = np.random.default_rng(0)
    return ts, rng.normal(size=(n, width))


def test_future_row_perturbation_leaves_window_unchanged_but_origin_row_changes():
    ts, x = _grid()
    t = int(ts[30])
    out = audit.window_mutation_checks(ts, x, t, samples=5, spacing_hours=2, sentinel=1e6)
    assert out["future_rows_unchanged"] is True
    assert out["origin_row_changes_rightmost"] is True
    assert out["rightmost_is_origin"] is True
    assert out["state"] == audit.PROVEN_SAFE


def test_centered_transform_is_reported_as_leak():
    ts, x = _grid()
    t = int(ts[30])

    def centered(values):  # a two-sided moving average: uses the next row
        result = values.copy()
        result[1:-1] = (values[:-2] + values[1:-1] + values[2:]) / 3
        return result

    out = audit.window_mutation_checks(ts, x, t, samples=5, spacing_hours=2, sentinel=1e6,
                                       transform=centered)
    assert out["future_rows_unchanged"] is False
    assert out["state"] == audit.PROVEN_LEAK


def test_trailing_transform_is_safe_and_impulse_after_t_does_not_move_past():
    ts, x = _grid()
    t = int(ts[30])

    def trailing(values):
        result = values.copy()
        result[1:] = (values[:-1] + values[1:]) / 2
        return result

    out = audit.window_mutation_checks(ts, x, t, samples=5, spacing_hours=2, sentinel=1e6,
                                       transform=trailing)
    assert out["future_rows_unchanged"] is True and out["state"] == audit.PROVEN_SAFE
    impulse = audit.impulse_after_t(ts, x, t, trailing)
    assert impulse["rows_at_or_before_t_unchanged"] is True


def test_target_perturbation_changes_label_not_input():
    ts, x = _grid()
    price = np.linspace(1.0, 1.1, len(ts))
    t = int(ts[30])
    out = audit.target_mutation_check(ts, x, price, t, samples=5, spacing_hours=2, horizon_hours=4)
    assert out["target_is_t_plus_4_elapsed_hours"] is True
    assert out["label_changes"] is True and out["input_unchanged"] is True


def test_mask_parity_requires_identical_ordered_identities_not_equal_counts():
    model = [(1, 5), (2, 6), (3, 7)]
    naive = [(1, 5), (2, 6), (4, 8)]
    parity = audit.mask_parity(model, naive)
    assert parity["counts_equal"] is True and parity["identical"] is False
    assert parity["only_model"] == [[3, 7]] and parity["only_naive"] == [[4, 8]]
    assert audit.mask_parity(model, list(model))["identical"] is True


def test_metric_replay_is_exact_and_detects_a_forged_sum():
    current = np.array([1.0, 1.1, 1.2])
    future = np.array([1.05, 1.0, 1.25])
    prediction = np.array([1.02, 1.08, 1.21])
    sums = audit.replay_rows(current, future, prediction)
    assert sums["n"] == 3
    assert sums["sum_abs_error"] == math.fsum(np.abs(prediction - future).tolist())
    assert sums["naive_sum_abs_error"] == math.fsum(np.abs(current - future).tolist())
    fold = audit.fold_fitness({"train": 2.0, "inner": 1.0}, {"train": 4.0, "inner": 2.0})
    assert fold == 0.5 * (2.0 / 4.0 + 1.0 / 2.0)
    assert audit.aggregate([1.0, 0.9, 1.1, 1.0]) == math.fsum([1.0, 0.9, 1.1, 1.0]) / 4
    assert audit.inner_only_fitness({"inner": 1.0}, {"inner": 2.0}) == 0.5
    check = audit.compare_exact(sums["sum_abs_error"], sums["sum_abs_error"] + 1e-15)
    assert check is False


def test_scaler_boundary_detects_rows_fitted_across_the_boundary():
    _, x = _grid(60, 4)
    fit = np.arange(0, 40)
    base = audit.scaler_digest(x, fit)
    outside = x.copy()
    outside[45:] += 1e6  # INNER / later-fold rows
    inside = x.copy()
    inside[10:12] += 1e6  # rows inside the fit interval
    assert audit.scaler_digest(outside, fit) == base
    assert audit.scaler_digest(inside, fit) != base
    assert audit.fitted_state_check(x, fit, perturb=np.arange(45, 60))["byte_identical"] is True
    assert audit.fitted_state_check(x, fit, perturb=np.arange(10, 12))["byte_identical"] is False


def test_selection_after_fold_cutoff_is_a_future_selection_leak():
    winners = {"Y_a": {"RAW": {"members": ["f1", "f2"]}}, "Y_b": {"RAW": {"members": ["f2", "f3"]}}}
    union = audit.fs4_union(winners)
    assert union == ["f1", "f2", "f3"]
    verdict = audit.selection_chronology(selection_max_time="2024-12-30T00:00:00+00:00",
                                         fold_cutoff="2023-03-31T21:00:00+00:00")
    assert verdict == audit.PROVEN_LEAK
    verdict = audit.selection_chronology(selection_max_time="2019-01-01T00:00:00+00:00",
                                         fold_cutoff="2023-03-31T21:00:00+00:00")
    assert verdict == audit.PROVEN_SAFE


def test_availability_lag_flags_values_used_before_their_source_date():
    hours = 24 * 6
    ts = np.arange(hours, dtype=np.int64) * 3600 + 1_577_836_800  # 2020-01-01T00:00Z
    column = np.zeros(hours)
    column[:48] = 1.0
    column[48:] = 2.0  # value 2.0 first appears at 2020-01-03T00:00Z
    raw = {"2019-12-31": 1.0, "2020-01-03": 2.0}  # 1.0 is a day old at the first hour (lag 24 h)
    lags = audit.availability_lags(ts, column, raw)
    assert lags["matched"] == 2 and lags["lag_hours"] == [24.0, 0.0]
    # a bound at 20:00 UTC of the source date is violated by a 0 h lag
    assert audit.availability_state(lags, bound_hours=20) == audit.PROVEN_LEAK
    assert audit.availability_state(lags, bound_hours=0) == audit.PROVEN_SAFE
    late = column.copy()
    late[48:69] = 1.0  # value 2.0 appears at 21:00 UTC of its own date
    lags = audit.availability_lags(ts, late, raw)
    assert audit.availability_state(lags, bound_hours=20) == audit.PROVEN_SAFE


def test_digest_mismatch_fails_closed(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"retained")
    good = hashlib.sha256(b"retained").hexdigest()
    assert audit.verify_digest(path, good) is True
    with pytest.raises(audit.DigestMismatch):
        audit.verify_digest(path, "0" * 64)


def test_redaction_removes_private_identifiers():
    text = "root /var/tmp/x island host-a-0000 ip 203.0.113.7 home /home/someone/y"
    out = audit.redact(text, {"host-a-0000": "ISLAND_0", "/var/tmp/x": "<campaign-root>"})
    assert "host-a" not in out and "203.0.113.7" not in out and "/home/" not in out
    assert "ISLAND_0" in out and "<campaign-root>" in out


def test_asof_alignment_identifies_the_lag_despite_repeated_values():
    hours = 24 * 10
    ts = np.arange(hours, dtype=np.int64) * 3600 + 1_577_836_800  # 2020-01-01T00:00Z
    raw = {"2019-12-30": 1.0, "2020-01-02": 2.0, "2020-01-04": 1.0, "2020-01-07": 2.0}
    days = np.array([audit.epoch(d + "T00:00:00+00:00") for d in sorted(raw)])
    vals = np.array([raw[d] for d in sorted(raw)])
    column = vals[np.searchsorted(days + 21 * 3600, ts, side="right") - 1]  # available at 21:00 UTC of its date
    out = audit.asof_alignment(ts, column, raw)
    assert out["best_lag_hours"] == 21 and out["best_match_rate"] == 1.0
    assert audit.asof_state(out, bound_hours=20) == audit.PROVEN_SAFE
    early = vals[np.searchsorted(days + 2 * 3600, ts, side="right") - 1]  # used from 02:00 UTC of its own date
    out = audit.asof_alignment(ts, early, raw)
    assert out["best_lag_hours"] == 2 and audit.asof_state(out, bound_hours=20) == audit.PROVEN_LEAK
    assert audit.asof_state(dict(best_lag_hours=None, best_match_rate=0.0), 20) == audit.RISK_UNRESOLVED
