"""Scientific and recovery contracts for the unattended multihorizon runner."""

import json
import sqlite3

import numpy as np
import pytest

from tools.i6m_data import partitions, targets, epoch, price_prediction
from tools.i6m_queue import Queue, digest, seal, verify
from tools.i6e_horizon_plan import experiment_grid, weekly_schedule


def test_exact_future_endpoints_and_no_fill():
    ts = np.array([0, 3600, 7200, 14400], dtype=np.int64)
    price = np.array([1., 2., 3., 5.])
    y, end = targets(ts, price, 2)
    np.testing.assert_allclose(y[[0, 2]], np.log([3., 5./3.]))
    assert np.isnan(y[[1, 3]]).all()
    assert end.tolist() == [2, -1, 3, -1]


def test_purge_and_full_context():
    ts = np.arange(epoch("2020-01-01T00:00:00Z"), epoch("2024-02-01T00:00:00Z"), 3600)
    week = weekly_schedule(2024)[0]
    y, _ = targets(ts, np.ones(len(ts)), 120)
    fit, inner, score = partitions(ts, y, week, 120)
    cut = epoch(week["fit_cutoff"])
    assert (ts[fit]+120*3600 < cut-28*86400).all()
    assert (ts[inner]+120*3600 < cut).all()
    assert ts[fit].min()-240*3600 >= epoch(week["fit_start"])
    assert ts[score].min() == epoch(week["score_start"])


def test_test_rows_rejected_and_partial_week_not_faked():
    ts = np.arange(epoch("2024-12-29T00:00:00Z"), epoch("2025-01-01T00:00:00Z"), 3600)
    y, _ = targets(ts, np.ones(len(ts)), 120)
    # Final 2-day segment has no exact 120h endpoint before TEST.
    _, _, score = partitions(ts, y, weekly_schedule(2024)[-1], 120, require_fit=False)
    assert len(score) == 0


def test_unscale_float64_before_price_metrics():
    got = price_prediction(np.array([1.], np.float64), np.array([1.], np.float32), .01, .02)
    assert got.dtype == np.float64
    assert got[0] == np.exp(.03)
    with pytest.raises(ValueError):
        price_prediction(np.array([1.]), np.array([1e30]), 0., 1.)


def test_user_skill_formula_and_horizon_naive():
    from tools.i6m_data import metrics
    m = metrics([1., 1.], [1.1, 1.2], [1.1, 1.2], 6)
    assert m["skill_mae"] == -1 and m["skill_mae_conventional"] == 1
    assert m["naive_label"] == "naive para el horizonte de 6 horas"
    assert m["skill_formula"] == "(MAE_h - naive_h) / naive_h"


def test_cost_fit_claim_requires_first_week_without_duplicate(tmp_path):
    q = Queue(tmp_path/"q.db", design())
    a = q.claim("preferred", "NEURAL", measured_samples=[])
    b = q.claim("secondary", "NEURAL", measured_samples=[])
    assert a["week"]["ordinal"] == b["week"]["ordinal"] == 0
    assert a["arm"]["arm_id"] != b["arm"]["arm_id"]


def test_statistical_prediction_never_conditions_on_future():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    values = np.linspace(0., 1., 100)
    a = SARIMAX(values, order=(1,1,0), trend="n").filter([.5, .1])
    values[51:] = 99
    b = SARIMAX(values, order=(1,1,0), trend="n").filter([.5, .1])
    pa = a.get_prediction(start=51, end=54, dynamic=True).predicted_mean
    pb = b.get_prediction(start=51, end=54, dynamic=True).predicted_mean
    np.testing.assert_array_equal(pa, pb)


def design():
    arms, _ = experiment_grid()
    return seal(dict(arms=arms, weeks=weekly_schedule(2024), seed=6102026, input_sha256="a"*64))


def test_queue_exclusive_claims_stage_barrier_and_order(tmp_path):
    q = Queue(tmp_path / "q.db", design())
    a = q.claim("preferred", "NEURAL")
    b = q.claim("secondary", "NEURAL")
    assert a["id"] != b["id"]
    assert a["arm"]["samples"] == 3 and a["arm"]["horizon_hours"] == 2
    assert a["arm"]["stage"] == b["arm"]["stage"] == "DOWNSAMPLING"
    assert q.claim("preferred", "NEURAL") is None
    # A waiting/running downsampling cell prevents any hourly claim.
    with sqlite3.connect(q.path) as conn:
        conn.execute("UPDATE jobs SET state='FAILED' WHERE stage='DOWNSAMPLING' AND id NOT IN (?,?)", (a["id"], b["id"]))
    assert q.claim("third", "NEURAL") is None
    q.complete(a, seal(dict(job_id=a["id"], lease=a["lease"], status="FAILED", design_sha256=q.identity)))
    q.complete(b, seal(dict(job_id=b["id"], lease=b["lease"], status="FAILED", design_sha256=q.identity)))
    h = q.claim("preferred", "NEURAL")
    assert h["arm"]["stage"] == "HOURLY" and h["arm"]["samples"] == 5


def test_terminal_lease_and_content_integrity(tmp_path):
    q = Queue(tmp_path / "q.db", design())
    a = q.claim("preferred", "NEURAL")
    bad = seal(dict(job_id=a["id"], lease="wrong", status="COMPLETED", design_sha256=q.identity))
    with pytest.raises(ValueError, match="lease"):
        q.complete(a, bad)
    good = seal(dict(job_id=a["id"], lease=a["lease"], status="FAILED", design_sha256=q.identity))
    q.complete(a, good)
    q.complete(a, good)
    altered = dict(good, status="COMPLETED")
    with pytest.raises(ValueError):
        verify(altered)
    with pytest.raises(ValueError):
        Queue(q.path, seal(dict(design(), seed=5)))


def test_restart_retains_claim_without_training_again(tmp_path):
    d = design()
    q = Queue(tmp_path / "q.db", d)
    a = q.claim("preferred", "NEURAL")
    q2 = Queue(q.path, d)
    assert q2.running()[0]["lease"] == a["lease"]
    assert q2.claim("preferred", "NEURAL") is None
    assert q2.status()["counts"]["RUNNING"] == 1


def test_sha_includes_order_and_scientific_contract():
    assert digest(["a", "b"]) != digest(["b", "a"])
    assert digest(dict(h=2)) != digest(dict(h=3))


@pytest.mark.parametrize("samples", [3, 5, 7, 9, 13, 49, 241])
def test_real_model_shapes_masks_reload_and_causal_prefix(samples, tmp_path):
    tf = pytest.importorskip("tensorflow")
    from tools.i6m_model import build_model
    tf.keras.utils.set_random_seed(6102026)
    model = build_model(samples, 20, 2)
    assert model.count_params() == 34921
    x = np.zeros((2, samples, 41), np.float32)
    x[:, :, -1] = 1
    a = model(x, training=False).numpy()
    # A placeholder's arbitrary content is invisible.
    x[:, 0, -1] = 0
    b = model(x, training=False).numpy()
    x[:, 0, :-1] = 999
    np.testing.assert_array_equal(b, model(x, training=False).numpy())
    artifact = tmp_path / "model.keras"
    model.save(artifact)
    np.testing.assert_array_equal(model(x, training=False).numpy(), tf.keras.models.load_model(artifact)(x, training=False).numpy())
    core = tf.keras.Model(model.input, model.get_layer("core_mask_2").output)
    before = core(x, training=False).numpy()
    x[:, -1, :-1] = 123
    after = core(x, training=False).numpy()
    np.testing.assert_array_equal(before[:, :-1], after[:, :-1])
    assert np.isfinite(a).all()


@pytest.mark.parametrize("samples", [3, 5, 9])
def test_general_factory_matches_sealed_factory(samples):
    tf = pytest.importorskip("tensorflow")
    from tools.i6m_model import build_model
    from tools.i6e_masked_model import build_model as retained
    tf.keras.utils.set_random_seed(6102026)
    a = build_model(samples, 20, 2)
    tf.keras.utils.set_random_seed(6102026)
    b = retained(samples, 20, 2)
    for x, y in zip(a.get_weights(), b.get_weights()):
        np.testing.assert_array_equal(x, y)
    inputs = np.zeros((2, samples, 41), np.float32)
    inputs[:, :, -1] = 1
    np.testing.assert_array_equal(a(inputs, training=False).numpy(), b(inputs, training=False).numpy())
