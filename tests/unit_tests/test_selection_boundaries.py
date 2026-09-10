"""P4: the selection boundaries, tested by execution.

The order's test shape is specific and it is the right one:
*change the future tail and prove the past stays identical*. A
boundary is clean exactly when data that arrives later cannot
reach back and alter what was fitted or selected earlier.

Each test below either demonstrates a corrected boundary or
pins a defect that is now labelled `LEGACY_NON_AUTHORITATIVE`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def plugin():
    mod = pytest.importorskip(
        "preprocessor_plugins.phase2_6_preprocessor")
    cls = getattr(mod, "PreprocessorPlugin", None) or \
        getattr(mod, "Plugin", None)
    if cls is None:
        pytest.skip("phase 2.6 plugin class not found")
    return cls()


def _series(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(0, 1, n)) * scale).astype(
        np.float32)


# ==============================================================
# predictor phase 2.6: the scaler must be fitted on train only
# ==============================================================

def test_future_tail_cannot_change_the_training_features(plugin):
    """Append a wildly different future and prove the TRAIN
    features are bit-identical."""
    train = _series(400, seed=1)
    test_a = _series(200, seed=2)
    test_b = _series(200, seed=3, scale=50.0)   # a different tail

    tr1, scaler1 = plugin._apply_causal_mtm_decomposition(
        train, 32, 3, "train")
    tr2, scaler2 = plugin._apply_causal_mtm_decomposition(
        train, 32, 3, "train")
    assert np.array_equal(tr1, tr2), (
        "the training features are not even reproducible")

    # the corrected path applies the TRAIN scaler to the tail
    te_a, _ = plugin._apply_causal_mtm_decomposition(
        test_a, 32, 3, "test", scaler=scaler1, fit_scaler=False)
    te_b, _ = plugin._apply_causal_mtm_decomposition(
        test_b, 32, 3, "test", scaler=scaler1, fit_scaler=False)

    tr3, _ = plugin._apply_causal_mtm_decomposition(
        train, 32, 3, "train")
    assert np.array_equal(tr1, tr3), (
        "the past changed when the future changed")
    assert not np.array_equal(te_a, te_b), (
        "a different future produced identical features — the "
        "test is not exercising anything")


def test_train_fitted_scaler_is_the_one_applied_to_the_tail(
        plugin):
    """The scaling PARAMETERS used on the tail must be the
    training parameters, unchanged by the tail's own moments."""
    train = _series(400, seed=11)
    tail = _series(200, seed=12, scale=25.0)

    _, train_scaler = plugin._apply_causal_mtm_decomposition(
        train, 32, 3, "train")
    mean_before = np.array(train_scaler.mean_, copy=True)
    scale_before = np.array(train_scaler.scale_, copy=True)

    plugin._apply_causal_mtm_decomposition(
        tail, 32, 3, "test", scaler=train_scaler,
        fit_scaler=False)

    assert np.array_equal(train_scaler.mean_, mean_before)
    assert np.array_equal(train_scaler.scale_, scale_before), (
        "applying the scaler to the tail refitted it")


def test_the_legacy_per_split_fit_is_the_defect_it_is_labelled(
        plugin):
    """Pin the old behaviour so the correction is visible: fitting
    on the split makes the tail's own moments decide its
    features."""
    tail = _series(200, seed=21, scale=25.0)
    train = _series(400, seed=22)
    _, train_scaler = plugin._apply_causal_mtm_decomposition(
        train, 32, 3, "train")

    corrected, _ = plugin._apply_causal_mtm_decomposition(
        tail, 32, 3, "test", scaler=train_scaler,
        fit_scaler=False)
    legacy, _ = plugin._apply_causal_mtm_decomposition(
        tail, 32, 3, "test", fit_scaler=True)

    assert not np.allclose(corrected, legacy), (
        "the corrected and legacy paths agree — then there was "
        "no boundary defect to correct")
    # the legacy path standardises the tail BY THE TAIL: its own
    # columns come out with ~zero mean, which is exactly the
    # tell-tale of a fit performed outside training
    assert abs(float(legacy.mean())) < 1e-5
    assert abs(float(corrected.mean())) > abs(float(legacy.mean()))


def test_legacy_escape_is_opt_in_and_labelled():
    src = (REPO / "preprocessor_plugins/"
                  "phase2_6_preprocessor.py").read_text()
    assert "mtm_legacy_per_split_scaler" in src
    assert "LEGACY_NON_AUTHORITATIVE" in src
    assert "not boundary-clean" in src
    # default is the CORRECTED path
    assert 'config.get("mtm_legacy_per_split_scaler", False)' \
        in src


# ==============================================================
# agent-multi: a group filter must not fail open
# ==============================================================

def test_group_filter_cannot_fail_open():
    """An empty eligible universe must refuse, not widen."""
    src = (REPO.parent / "agent-multi/optimizer_plugins/"
                         "project3_full_genome_optimizer.py")
    if not src.is_file():
        pytest.skip("agent-multi not present")
    text = src.read_text()
    assert "refusing rather than widening" in text
    # the pre-existing required-group fallback still exists, but
    # it now runs BEFORE the gate, so the gate has the last word
    assert text.index("mixed genome disabled every feature "
                      "group") < text.index("filter_to_eligible(")


def test_gate_filter_can_only_remove():
    from eligibility import gate
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "t", "entries": []}
    doc["manifest_sha256"] = gate._self_sha(doc)
    assert gate.filter_to_eligible(doc, ["a", "b", "c"],
                                   scope="forecasting") == []


# ==============================================================
# a genome may not admit every numeric column by default
# ==============================================================

def test_numeric_columns_do_not_enter_without_the_gate():
    """With a manifest configured, only reviewed columns survive
    — a column is never admitted for being numeric."""
    from eligibility import gate
    entries = [{
        "subject_kind": "variable", "subject_id": "col.reviewed",
        "version": "1",
        "io_schema": {"input": ["float"], "output": ["float"]},
        "unit": "u",
        "temporal_availability": {"event_time": "t",
                                  "available_time": "t"},
        "fit_scope": "TRAIN_ONLY",
        "incremental_state_policy": "none", "parameters": {},
        "digests": {"data": "d" * 64, "code": "c" * 64,
                    "partitions": "p" * 64, "evidence": "e" * 64},
        "measured_cost": {}, "decision": "PUBLICLY_ELIGIBLE",
        "decision_scope": "representation",
        "decision_reason": "r", "reviewer": "x",
        "reviewed_at": "2026-09-09T00:00:00Z"}]
    doc = {"schema": gate.MANIFEST_SCHEMA,
           "issued_at": "2026-09-10T00:00:00Z",
           "issuer": "x", "scope": "t", "entries": entries}
    doc["manifest_sha256"] = gate._self_sha(doc)
    numeric_columns = ["col.reviewed", "col.numeric_1",
                       "col.numeric_2", "col.numeric_3"]
    kept = gate.filter_to_eligible(doc, numeric_columns,
                                   scope="representation")
    assert kept == ["col.reviewed"]
