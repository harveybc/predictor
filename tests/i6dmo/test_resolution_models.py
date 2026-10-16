"""Resolution model contracts: synthetic CPU forwards/rebuilds, no scientific fit."""

import copy
import itertools

import numpy as np
import pytest

from tools.i6dmo import model as M, provenance, resolution_models as R

FEATURES = list(provenance.MANIFEST)
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
KINDS = ("conv", "linear", "dlinear")


@pytest.fixture(autouse=True)
def clear_graph():
    yield
    M.keras.backend.clear_session()


def inputs(slots=13, spacing=2, batch=2):
    x = np.random.default_rng(51).normal(size=(batch, slots, 19)).astype(np.float32)
    offsets = np.tile(np.arange(-(slots - 1) * spacing, 1, spacing, dtype=np.float32), (batch, 1))
    return [x, offsets, np.zeros_like(offsets)]


def matrix(model, values):
    out = model(values, training=False)
    assert isinstance(out, list)
    assert all(tuple(v.shape) == (len(values[0]), 1) for v in out)
    return np.concatenate([v.numpy() for v in out], axis=1)


def test_build_config_all_approved_grids_heads_and_kinds():
    for slots, spacing, heads, kind in itertools.product((6, 9, 13, 18, 24), (2, 4), (SHORT, LONG, [72]), KINDS):
        config = R.build_config(FEATURES, slots, spacing, heads, kind=kind)
        assert config["schema"] == "i6dmo.resolution_model.v1"
        assert config["slots"] == slots and config["spacing_open_market_hours"] == spacing
        assert config["horizons_hours"] == heads and config["seed"] == 6102026
        assert config["kind"] == kind and config["features"] == FEATURES
    heads, features = SHORT.copy(), FEATURES.copy()
    config = R.build_config(features, 13, 2, heads)
    heads.clear()
    features.clear()
    assert config["horizons_hours"] == SHORT and config["features"] == FEATURES


@pytest.mark.parametrize("patch", [{"slots": 7}, {"slots": True}, {"spacing": 1}, {"spacing": 2.0},
                                  {"horizons": [12]}, {"horizons": [72, 72]}, {"horizons": [True]},
                                  {"kind": "autoencoder"}, {"seed": True}, {"features": ["x", "x"]},
                                  {"features": []}, {"features": [""]}])
def test_invalid_config_refused(patch):
    args = dict(features=FEATURES, slots=13, spacing=2, horizons=SHORT)
    with pytest.raises(ValueError):
        R.build_config(**dict(args, **patch))


@pytest.mark.parametrize("patch", [{"unknown": True}, {"schema": M.MODEL_SCHEMA}, {"slots": 25},
                                  {"kind": "R2"}, {"spacing_open_market_hours": 12}])
def test_build_revalidates_serialized_config(patch):
    config = R.build_config(FEATURES, 13, 2, SHORT)
    with pytest.raises(ValueError):
        R.build(dict(config, **patch))


@pytest.mark.parametrize("kind,slots", itertools.product(KINDS, (6, 9, 13, 18, 24)))
def test_real_graph_inputs_outputs_and_known_capacity(kind, slots):
    heads = [72] if slots in (6, 24) else LONG
    config = R.build_config(FEATURES, slots, 4, heads, kind=kind)
    bundle = R.build(config)
    assert bundle.config == config
    assert [tuple(t.shape[1:]) for t in bundle.forecast_model.inputs] == [(slots, 19), (slots,), (slots,)]
    result = matrix(bundle.forecast_model, inputs(slots, 4))
    assert result.shape == (2, len(heads)) and np.isfinite(result).all()
    h = len(heads)
    expected = {"conv": 2284 + h * (32 * slots + 17),
                "linear": h * (21 * slots + 2), "dlinear": h * (40 * slots + 3)}[kind]
    assert bundle.forecast_model.count_params() == expected
    for variable in bundle.forecast_model.weights:
        if variable.name == "bias":
            np.testing.assert_array_equal(variable.numpy(), np.zeros(variable.shape))


def test_conv_13_is_exact_legacy_recipe_weights_and_predictions():
    config = R.build_config(FEATURES, 13, 2, SHORT)
    resolution = R.build(config).forecast_model
    legacy = M.build_dmo(M.model_config(FEATURES, family="short_mg2")).forecast_model
    assert resolution.count_params() == legacy.count_params() == 4882
    assert len(resolution.weights) == len(legacy.weights)
    for a, b in zip(resolution.get_weights(), legacy.get_weights()):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(matrix(resolution, inputs()), matrix(legacy, inputs()))
    encoder = resolution.get_layer("physical_time_encoder")
    assert isinstance(encoder, M.DynamicPhysicalTimeEncoder)
    branches = [resolution.get_layer(f"branch_{i:02d}") for i in range(19)]
    for branch in branches:
        conv = branch.layers[-1]
        assert (conv.filters, conv.kernel_size, conv.strides, conv.padding) == (4, (3,), (1,), "causal")
    assert tuple(resolution.get_layer("temporal_core").output_shape[1:]) == (13, 4)


@pytest.mark.parametrize("kind,heads", itertools.product(KINDS, (SHORT, [72])))
def test_seed_dispatch_and_actual_monitor_rebuild_weights_roundtrip(kind, heads, tmp_path):
    from tools.i6dmo import train as T
    config = R.build_config(FEATURES, 6, 4, heads, kind=kind)
    bundle = R.build(config)
    initial = M.weight_tensor_digests(bundle.forecast_model)
    rebuilt = M.build_dmo(copy.deepcopy(config))
    assert initial == M.weight_tensor_digests(rebuilt.forecast_model)
    # Change weights so rebuilding the seed alone cannot accidentally pass the load check.
    modified = [w + np.float32(0.01) for w in bundle.forecast_model.get_weights()]
    bundle.forecast_model.set_weights(modified)
    loaded = T._monitor_model(bundle, tmp_path)
    assert M.weight_tensor_digests(loaded) == M.weight_tensor_digests(bundle.forecast_model)
    np.testing.assert_array_equal(matrix(loaded, inputs(6, 4)), matrix(bundle.forecast_model, inputs(6, 4)))
    assert rebuilt.config == config
    assert M.DETERMINISM["op_determinism"] is True


@pytest.mark.parametrize("kind", KINDS)
def test_two_epoch_real_fit_single_head_monitor_stopper_and_reload(kind, tmp_path):
    from tools.i6dmo import train as T

    config = R.build_config(FEATURES, 6, 4, [72], kind=kind)
    bundle = R.build(config)
    splits = []
    for rows in (8, 4):
        values = inputs(6, 4, batch=rows)
        values[1][:, :3] -= 48
        values[2][:, 3] = 1
        labels = (0.2 * values[0][:, -1, :1] + 0.05 * values[0][:, 0, 1:2]).astype(np.float32)
        mask = np.ones((rows, 1), dtype=bool)
        mask[1, 0] = False
        labels[1, 0] = np.nan
        splits.append((values, labels, mask))
    fit, inner = splits
    ids = (np.arange(8, dtype=np.int64) * 3600 + 1640995200,
           np.arange(4, dtype=np.int64) * 3600 + 1672531200)
    # A deliberately large threshold makes the second epoch stale, exercising real patience.
    training = dict(learning_rate=0.001, weight_decay=0.0001, batch_size=4,
                    max_epochs=2, patience=1, min_delta=1e6)
    progress = []
    result = T.fit_dmo(bundle, fit, inner, training, config["seed"],
                       progress=progress.append, ids=ids, record_device="cpu")
    assert result["epochs_completed"] == len(result["history"]) == len(progress) == 2
    assert result["updates"] == 4 and result["stop_reason"] == "patience"
    assert result["initial_weights_sha256"] != result["last_weights_sha256"]
    best = min(result["history"], key=lambda row: (row["monitored"], row["epoch"]))
    assert result["best_epoch"] == best["epoch"]
    assert result["best_monitored"] == best["monitored"]
    assert result["restored_weights_sha256"] == best["weights_sha256"] == M.weights_hash(bundle.forecast_model)
    for row in result["history"]:
        assert np.isfinite(row["monitored"])
        assert row["monitored"] == 0.5 * (row["fit_loss"] + row["inner_loss"])
    record = result["monitor_evaluation"]
    assert record["self_check"] == "PASS" and record["monitor_instance"] == "canonical_loader_rebuild"
    assert record["loader_overhead_s"]["epochs"] == 2 and record["witness"]["ids"] == "row_id"
    for name, row_ids in zip(("fit", "inner"), ids):
        assert record["witness"][name]["row_ids_sha256"] == T._ids_sha(row_ids)
    loaded = T._monitor_model(bundle, tmp_path)
    np.testing.assert_array_equal(matrix(loaded, fit[0]), matrix(bundle.forecast_model, fit[0]))
    fm, im, monitored, replay = T.replay_monitored(loaded, fit, inner, record, ids=ids)
    assert replay["status"] == "EXACT" and monitored == best["monitored"]
    assert fm["counts"] == [7] and im["counts"] == [3]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fault", ("future", "missing_origin", "compressed", "closure_first", "nonfinite", "closure_fraction"))
def test_no_future_and_invalid_time_inputs_refused(kind, fault):
    bundle = R.build(R.build_config(FEATURES, 6, 4, [72], kind=kind))
    values = inputs(6, 4)
    if fault == "future":
        values[1][:, -2] = 1
    elif fault == "missing_origin":
        values[1] -= 4
    elif fault == "compressed":
        values[1] /= 2
    elif fault == "closure_first":
        values[2][:, 0] = 1
    elif fault == "nonfinite":
        values[1][:, 0] = np.nan
    else:
        values[2][:, 1] = 0.5
    with pytest.raises((ValueError, M.tf.errors.InvalidArgumentError)):
        bundle.forecast_model(values, training=False)


@pytest.mark.parametrize("kind", KINDS)
def test_origin_and_every_observation_participate_without_cross_sample_leakage(kind):
    model = R.build(R.build_config(FEATURES, 6, 2, [72], kind=kind)).forecast_model
    # Positive controlled kernels remove dead-ReLU ambiguity from the Conv participation check.
    model.set_weights([np.full_like(w, 0.1) if len(w.shape) > 1 else np.zeros_like(w)
                       for w in model.get_weights()])
    values = inputs(6, 2)
    values[0][:] = 1
    baseline = matrix(model, values)
    for slot in range(6):
        changed = [a.copy() for a in values]
        changed[0][0, slot, :] += 1
        observed = matrix(model, changed)
        assert observed[0, 0] != baseline[0, 0]
        np.testing.assert_array_equal(observed[1], baseline[1])


@pytest.mark.parametrize("kind", ("linear", "dlinear"))
def test_controls_are_affine_in_values_and_physical_time_context(kind):
    model = R.build(R.build_config(FEATURES, 9, 2, SHORT, kind=kind)).forecast_model
    a, b = inputs(9, 2), inputs(9, 4)
    b[0] *= -2
    b[2][:, 4] = 1
    # Closure flags remain valid: fix them across the affine segment, vary offsets and values.
    a[2][:] = b[2]
    alpha = np.float32(0.375)
    mixed = [alpha * x + (1 - alpha) * y for x, y in zip(a, b)]
    np.testing.assert_allclose(matrix(model, mixed), alpha * matrix(model, a) + (1 - alpha) * matrix(model, b),
                               rtol=2e-5, atol=2e-6)
    time_only = [a[0], b[1], a[2]]
    assert not np.array_equal(matrix(model, a), matrix(model, time_only))
    closure_only = [a[0], a[1], np.zeros_like(a[2])]
    assert not np.array_equal(matrix(model, a), matrix(model, closure_only))


def test_dlinear_kernel3_endpoint_replication_and_decomposition_identity():
    layer = R.WindowDecomposition()
    x = np.asarray([[[0.], [3.], [9.], [12.], [18.], [30.]]], dtype=np.float32)
    residual, trend = layer(x)
    expected = np.asarray([[[1.], [4.], [8.], [13.], [20.], [26.]]], dtype=np.float32)
    np.testing.assert_array_equal(trend.numpy(), expected)
    np.testing.assert_array_equal((residual + trend).numpy(), x)
    assert not layer.weights
    assert layer.get_config()["kernel_size"] == 3
    with pytest.raises(ValueError):
        R.WindowDecomposition(kernel_size=5)
    # The last window value is reused, not any point beyond the issue origin.
    changed = x.copy()
    changed[:, -1] += 6
    _, updated = layer(changed)
    np.testing.assert_array_equal((updated - trend).numpy()[0, :, 0], [0, 0, 0, 0, 2, 4])


def test_dlinear_independent_trend_residual_kernels_are_both_used():
    model = R.build(R.build_config(FEATURES, 6, 2, [72], kind="dlinear")).forecast_model
    trend = model.get_layer("h72_trend")
    residual = model.get_layer("h72_residual")
    assert trend.kernel is not residual.kernel
    model.set_weights([np.zeros_like(w) for w in model.get_weights()])
    values = inputs(6, 2)
    seasonal, slow = R.WindowDecomposition()(values[0])
    for layer, component in ((trend, slow), (residual, seasonal)):
        weights = np.zeros_like(layer.kernel.numpy())
        weights[0, 0] = 1
        layer.set_weights([weights, np.zeros(1, dtype=np.float32)])
        np.testing.assert_allclose(matrix(model, values)[:, 0], component.numpy()[:, 0, 0], atol=1e-6)
        layer.set_weights([np.zeros_like(weights), np.zeros(1, dtype=np.float32)])


def test_legacy_config_fingerprints_remain_unchanged():
    from tests.i6dmo.test_context_bridge_design_model import (
        test_cb01_legacy_schema_effective_and_model_config_fingerprints_unchanged,
    )
    test_cb01_legacy_schema_effective_and_model_config_fingerprints_unchanged()
