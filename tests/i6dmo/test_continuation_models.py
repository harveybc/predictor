"""Continuation model contracts: no PE, private heads, full TRAIN/VALIDATION stopping.

All arrays here are tiny synthetic fixtures, never a scientific dataset. TRAIN
and VALIDATION are the two complete fixture splits. There is no third/INNER
dataset; fit_dmo's historical ``inner`` parameter receives VALIDATION unchanged.
"""

import copy
import itertools

import numpy as np
import pytest

from tools.i6dmo import continuation_models as C, model as M, provenance, resolution_models as R

FEATURES = list(provenance.MANIFEST)
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
HEADS = ("dense", "conv", "lstm")
CORES = ((2,), (4,), (4, 2), (8, 4))


@pytest.fixture(autouse=True)
def clear_graph():
    yield
    M.keras.backend.clear_session()


def inputs(slots=6, spacing=4, rows=2):
    values = np.random.default_rng(29).normal(size=(rows, slots, 19)).astype(np.float32)
    offsets = np.tile(np.arange(-(slots - 1) * spacing, 1, spacing, dtype=np.float32), (rows, 1))
    closure = np.zeros_like(offsets)
    offsets[:, :3] -= 48
    closure[:, 3] = 1
    return [values, offsets, closure]


def matrix(model, arrays):
    outputs = model(arrays, training=False)
    assert isinstance(outputs, list)
    assert all(tuple(y.shape) == (len(arrays[0]), 1) for y in outputs)
    return np.concatenate([y.numpy() for y in outputs], axis=1)


def test_all_config_choices_and_unused_affine_choices_canonicalize():
    for slots, spacing, horizons, head, core in itertools.product(
        (6, 9, 13, 18, 24), (2, 4), (SHORT, LONG, [4], [72]), HEADS, CORES,
    ):
        config = C.build_config(FEATURES, slots, spacing, horizons, head=head, core=core)
        assert config["schema"] == "i6dmo.continuation_model.v1"
        assert config["head"] == head and config["core"] == list(core)
        assert config["slots"] == slots and config["spacing_open_market_hours"] == spacing
        assert config["horizons_hours"] == horizons and config["seed"] == 6102026
    for kind in ("linear", "dlinear"):
        canonical = C.build_config(FEATURES, 13, 2, [4], kind=kind)
        assert canonical["head"] == "none" and canonical["core"] == []
        for head, core in itertools.product(HEADS, CORES):
            assert C.build_config(FEATURES, 13, 2, [4], kind=kind, head=head, core=core) == canonical
        assert C.build_config(FEATURES, 13, 2, [4], kind=kind, head="none", core=[]) == canonical


@pytest.mark.parametrize("patch", [{"slots": 25}, {"slots": True}, {"spacing": 12},
                                  {"spacing": 2.0}, {"horizons": [2]}, {"horizons": [True]},
                                  {"head": "bidirectional"}, {"head": "none"}, {"core": ()},
                                  {"core": (8,)}, {"core": (True,)}, {"seed": True},
                                  {"features": []}, {"features": ["x", "x"]}, {"kind": "R2"}])
def test_undeclared_config_refused(patch):
    args = dict(features=FEATURES, slots=6, spacing=4, horizons=[72])
    with pytest.raises(ValueError):
        C.build_config(**dict(args, **patch))


@pytest.mark.parametrize("patch", [{"schema": R.SCHEMA}, {"extra": True}, {"head": "transformer"},
                                  {"core": [16, 4]}, {"horizons_hours": [24]}])
def test_build_revalidates_stored_config(patch):
    with pytest.raises(ValueError):
        C.build(dict(C.build_config(FEATURES, 6, 4, [72]), **patch))


def test_exact_context_appends_intervals_without_modifying_any_fused_value():
    arrays = inputs()
    values = np.zeros((2, 6, 76), dtype=np.float32)
    values[:, 2, 5] = 1.25
    layer = C.ElapsedContext()
    encoded = layer([values, arrays[1], arrays[2]]).numpy()
    np.testing.assert_array_equal(encoded[:, :, :76], values)
    adjacent = np.concatenate([np.zeros((2, 1), dtype=np.float32), np.diff(arrays[1], axis=1)], axis=1)
    np.testing.assert_array_equal(encoded[:, :, 76], adjacent / 24)
    np.testing.assert_array_equal(encoded[:, :, 77], arrays[2])
    assert encoded.shape == (2, 6, 78) and not layer.weights


def core_count(core):
    channels, count = 78, 0
    for filters in core:
        count += (3 * channels + 1) * filters
        channels = filters
    return count


@pytest.mark.parametrize("head,core", itertools.product(HEADS, CORES))
def test_real_graph_private_heads_known_capacity_no_pe_and_zero_bias(head, core):
    config = C.build_config(FEATURES, 6, 4, SHORT, head=head, core=core)
    model = C.build(config).forecast_model
    assert [tuple(x.shape[1:]) for x in model.inputs] == [(6, 19), (6,), (6,)]
    assert np.isfinite(matrix(model, inputs())).all()
    assert isinstance(model.get_layer("time_input_guard"), R.TimeInputGuard)
    assert isinstance(model.get_layer("physical_context"), C.ElapsedContext)
    assert not any(isinstance(x, (M.DynamicPhysicalTimeEncoder, M.PhysicalOffsetEncoding)) for x in model.layers)
    latent = model.get_layer("temporal_core")
    convolutions = [x for x in latent.layers if isinstance(x, M.keras.layers.Conv1D)]
    assert [x.filters for x in convolutions] == list(core)
    assert all(x.padding == "causal" and x.kernel_size == (3,) and x.strides == (1,) for x in convolutions)
    private = [model.get_layer(f"h{h}") for h in SHORT]
    assert len({id(v) for part in private for v in part.weights}) == sum(len(p.weights) for p in private)
    channels = core[-1]
    per_head = {"dense": 8 * 6 * channels + 17,
                "conv": 32 * 6 + 12 * channels + 21,
                "lstm": 16 * (channels + 5) + 49}[head]
    assert model.count_params() == 304 + core_count(core) + 6 * per_head
    for part in private:
        dense = [layer for layer in part.layers if isinstance(layer, M.keras.layers.Dense)]
        assert [layer.units for layer in dense] == [8, 1]
        if head == "conv":
            conv = part.get_layer(part.name + "_conv")
            assert (conv.filters, conv.kernel_size, conv.strides, conv.padding) == (4, (3,), (1,), "causal")
        if head == "lstm":
            lstm = part.get_layer(part.name + "_lstm")
            assert lstm.units == 4 and lstm.go_backwards is False and lstm.return_sequences is False
    for variable in model.weights:
        if variable.name == "bias":
            np.testing.assert_array_equal(variable.numpy(), np.zeros(variable.shape))


@pytest.mark.parametrize("slots,spacing,horizons", [(6, 2, [4]), (9, 4, [72]), (13, 2, SHORT),
                                                   (18, 4, LONG), (24, 2, [72])])
def test_window_and_head_shapes(slots, spacing, horizons):
    bundle = C.build(C.build_config(FEATURES, slots, spacing, horizons, head="dense", core=(2,)))
    assert matrix(bundle.forecast_model, inputs(slots, spacing)).shape == (2, len(horizons))


@pytest.mark.parametrize("kind,head", [("conv", head) for head in HEADS] + [("linear", "none"), ("dlinear", "none")])
def test_seed_and_fresh_rebuild_after_weights_serialization(kind, head, tmp_path):
    from tools.i6dmo import train as T
    config = C.build_config(FEATURES, 6, 4, [4], kind=kind, head=head, core=(4, 2))
    original = C.build(config)
    rebuilt = M.build_dmo(copy.deepcopy(config))
    assert M.weight_tensor_digests(original.forecast_model) == M.weight_tensor_digests(rebuilt.forecast_model)
    original.forecast_model.set_weights([w + np.float32(0.001) for w in original.forecast_model.get_weights()])
    loaded = T._monitor_model(original, tmp_path)
    assert M.weight_tensor_digests(original.forecast_model) == M.weight_tensor_digests(loaded)
    np.testing.assert_array_equal(matrix(loaded, inputs()), matrix(original.forecast_model, inputs()))


@pytest.mark.parametrize("head", HEADS)
def test_tiny_positive_kernels_every_slot_origin_and_private_head_participate(head):
    model = C.build(C.build_config(FEATURES, 6, 4, SHORT, head=head, core=(4, 2))).forecast_model
    model.set_weights([np.full_like(w, 0.03) if len(w.shape) > 1 else np.zeros_like(w) for w in model.get_weights()])
    arrays = inputs()
    arrays[0][:] = 1
    baseline = matrix(model, arrays)
    for slot in range(6):
        changed = [x.copy() for x in arrays]
        changed[0][0, slot] += 0.25
        observed = matrix(model, changed)
        assert np.all(observed[0] != baseline[0])
        np.testing.assert_array_equal(observed[1], baseline[1])
    head0 = model.get_layer("h2")
    weights = head0.get_weights()
    weights[-1] += 0.1
    head0.set_weights(weights)
    shifted = matrix(model, arrays)
    assert np.all(shifted[:, 0] != baseline[:, 0])
    np.testing.assert_array_equal(shifted[:, 1:], baseline[:, 1:])


@pytest.mark.parametrize("kind", ("conv", "linear", "dlinear"))
@pytest.mark.parametrize("fault", ("future", "missing_origin", "compressed", "closure"))
def test_future_and_invalid_context_refused(kind, fault):
    model = C.build(C.build_config(FEATURES, 6, 4, [72], kind=kind, core=(2,))).forecast_model
    arrays = inputs()
    if fault == "future":
        arrays[1][:, -2] = 1
    elif fault == "missing_origin":
        arrays[1] -= 4
    elif fault == "compressed":
        arrays[1][:, -2] = -1
    else:
        arrays[2][:, 0] = 1
    with pytest.raises((ValueError, M.tf.errors.InvalidArgumentError)):
        model(arrays, training=False)


@pytest.mark.parametrize("kind", ("linear", "dlinear"))
def test_affine_control_math_and_dlinear_kernel_are_reused_without_pe(kind):
    config = C.build_config(FEATURES, 6, 4, [72], kind=kind)
    model = C.build(config).forecast_model
    reference = R.build(R.build_config(FEATURES, 6, 4, [72], kind=kind)).forecast_model
    np.testing.assert_array_equal(matrix(model, inputs()), matrix(reference, inputs()))
    assert M.weight_tensor_digests(model) == M.weight_tensor_digests(reference)
    if kind == "dlinear":
        layer = model.get_layer("window_decomposition")
        assert isinstance(layer, R.WindowDecomposition) and layer.kernel_size == 3
    assert not any(isinstance(layer, M.DynamicPhysicalTimeEncoder) for layer in model.layers)


@pytest.mark.parametrize("kind,head", [("conv", head) for head in HEADS] + [("linear", "none"), ("dlinear", "none")])
def test_real_two_epoch_train_validation_fit_and_exact_replay(kind, head, tmp_path):
    from tools.i6dmo import train as T

    config = C.build_config(FEATURES, 6, 4, [72], kind=kind, head=head, core=(2,))
    bundle = C.build(config)
    splits = []
    for rows in (8, 4):
        x = inputs(rows=rows)
        y = (0.2 * x[0][:, -1, :1]).astype(np.float32)
        mask = np.ones((rows, 1), dtype=bool)
        mask[1, 0] = False
        y[1, 0] = np.nan
        splits.append((x, y, mask))
    train, validation = splits
    ids = (1640995200 + np.arange(8, dtype=np.int64) * 3600,
           1704067200 + np.arange(4, dtype=np.int64) * 3600)
    training = dict(learning_rate=0.001, weight_decay=0.0001, batch_size=4,
                    max_epochs=2, patience=1, min_delta=1e6)
    result = T.fit_dmo(bundle, train, validation, training, config["seed"], ids=ids, record_device="cpu")
    assert result["epochs_completed"] == 2 and result["updates"] == 4
    assert result["stop_reason"] == "patience"
    assert result["initial_weights_sha256"] != result["last_weights_sha256"]
    best = min(result["history"], key=lambda row: (row["monitored"], row["epoch"]))
    assert result["restored_weights_sha256"] == best["weights_sha256"] == M.weights_hash(bundle.forecast_model)
    assert result["best_epoch"] == best["epoch"]
    for row in result["history"]:
        assert row["monitored"] == 0.5 * (row["fit_loss"] + row["inner_loss"])
    record = result["monitor_evaluation"]
    assert record["self_check"] == "PASS" and record["loader_overhead_s"]["epochs"] == 2
    assert record["witness"]["fit"]["row_ids_sha256"] == T._ids_sha(ids[0])
    assert record["witness"]["inner"]["row_ids_sha256"] == T._ids_sha(ids[1])
    loaded = T._monitor_model(bundle, tmp_path)
    np.testing.assert_array_equal(matrix(loaded, validation[0]), matrix(bundle.forecast_model, validation[0]))
    fm, vm, monitored, replay = T.replay_monitored(loaded, train, validation, record, ids=ids)
    assert replay["status"] == "EXACT" and monitored == best["monitored"]
    assert fm["counts"] == [7] and vm["counts"] == [3]
    perturbed = (validation[0], validation[1] + 1, validation[2])
    _, _, changed, _ = T.evaluate_monitored(loaded, train, perturbed, 4, ids=ids)
    assert changed != monitored


def test_legacy_and_resolution_config_fingerprints_unchanged():
    from tests.i6dmo.test_context_bridge_design_model import (
        test_cb01_legacy_schema_effective_and_model_config_fingerprints_unchanged,
    )
    test_cb01_legacy_schema_effective_and_model_config_fingerprints_unchanged()
    assert R.build_config(FEATURES, 13, 2, SHORT)["schema"] == R.SCHEMA
