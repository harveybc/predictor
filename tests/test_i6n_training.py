"""Synthetic candidate acceptance; no historical data or campaign claims.

Traceability: mask/gradient/skip -> first three tests; exact counts/owner mean
and evaluation invariance -> evaluation tests; checkpoint/patience -> restore
test; strict boundary -> invalid-input/config tests; audit -> JSON/events test.
"""

import inspect
import json

import numpy as np
import pytest
import tensorflow as tf

from tools.i6n_training import _evaluate_masked, fit_masked
from tools.modular_candidate_evaluator import _weight_digest


def config(**changes):
    cfg = dict(max_epochs=1, patience=2, min_delta=0., batch_size=2,
               learning_rate=.05, weight_decay=0., huber_delta=1.,
               seed=17, head_weights=[1., 3.])
    return {**cfg, **changes}


def model(heads=2, initial=0.):
    return tf.keras.Sequential([
        tf.keras.Input((1,)),
        tf.keras.layers.Dense(heads, use_bias=False,
                              kernel_initializer=tf.keras.initializers.Constant(initial)),
    ])


def arrays():
    x = np.ones((3, 1), np.float32)
    y = np.array([[1., 2.], [1., np.nan], [1., np.nan]], np.float32)
    return x, y, np.isfinite(y)


def fit(m, data=None, inner=None, cfg=None, progress=None):
    data = arrays() if data is None else data
    inner = data if inner is None else inner
    return fit_masked(m, *data, *inner, config() if cfg is None else cfg, progress)


def test_partial_head_gradient_masks_nan_before_residual():
    m = model()
    x = np.ones((2, 1), np.float32)
    y = np.array([[1., np.nan], [1., np.nan]], np.float32)
    result = fit(m, (x, y, np.isfinite(y)))
    weights = m.get_weights()[0][0]
    assert weights[0] > 0
    assert weights[1] == 0
    assert result['observed_updates'] == 1
    assert result['history'][0]['train']['counts'] == [2, 0]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('split', ['TRAIN', 'INNER'])
def test_empty_split_rejected_before_optimizer_mutation(split, monkeypatch):
    m = model(initial=2.)
    x = np.ones((3, 1), np.float32)
    y = np.full((3, 2), np.nan, np.float32)
    empty = (x, y, np.zeros_like(y, bool))
    before = _weight_digest(m.get_weights())
    created, events = [], []
    original = tf.keras.optimizers.AdamW

    def optimizer(*args, **kwargs):
        created.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(tf.keras.optimizers, 'AdamW', optimizer)
    with pytest.raises(ValueError, match=f'^NO_ELIGIBLE_{split}$'):
        fit(m, empty if split == 'TRAIN' else arrays(),
            empty if split == 'INNER' else arrays(),
            cfg=config(weight_decay=.9), progress=events.append)
    assert _weight_digest(m.get_weights()) == before
    assert created == events == []


def test_empty_evaluation_utility_remains_zero():
    x = np.ones((3, 1), np.float32)
    y = np.full((3, 2), np.nan, np.float32)
    metrics = _evaluate_masked(model(), x, y, np.zeros_like(y, bool), [1., 3.], 1., 2)
    assert metrics == dict(sums=[0., 0.], counts=[0, 0], per_head=[0., 0.], loss=0.)


def test_mixed_empty_and_valid_batches_have_one_observed_update(monkeypatch):
    x = np.ones((4, 1), np.float32)
    y = np.array([[np.nan, np.nan], [np.nan, np.nan], [1., 2.], [1., 2.]])
    m = model(initial=.5)
    before = _weight_digest(m.get_weights())
    events, calls = [], []
    original = tf.keras.optimizers.AdamW.apply_gradients

    def apply(optimizer, *args, **kwargs):
        calls.append((int(optimizer.iterations.numpy()), _weight_digest(m.get_weights())))
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(tf.keras.optimizers.AdamW, 'apply_gradients', apply)
    result = fit(m, (x, y, np.isfinite(y)),
                 cfg=config(weight_decay=.2), progress=events.append)
    assert calls == [(0, before)]
    assert result['observed_updates'] == result['optimizer_iterations'] == 1
    assert result['skipped_batches'] == 1
    assert [e['event'] for e in events] == ['update', 'epoch', 'restored']


@pytest.mark.parametrize('batch_size', [1, 2, 3, 7])
def test_fixed_checkpoint_evaluation_is_batch_size_invariant(batch_size):
    x, y, mask = arrays()
    metrics = _evaluate_masked(model(), x, y, mask, [1., 3.], 1., batch_size)
    assert metrics['counts'] == [3, 1]
    assert metrics['sums'] == pytest.approx([1.5, 1.5])
    assert metrics['per_head'] == pytest.approx([.5, 1.5])
    assert metrics['loss'] == pytest.approx(1.25)


def test_epoch_exact_counts_and_owner_mean_at_same_checkpoint():
    m = model()
    data = arrays()
    inner = (np.ones((1, 1), np.float32), np.array([[3., 4.]], np.float32),
             np.ones((1, 2), bool))
    result = fit(m, data, inner)
    row = result['history'][0]
    for name, split in [('train', data), ('inner', inner)]:
        x, y, mask = split
        p = m(x, training=False).numpy().astype(np.float64)
        expected = []
        for head in range(2):
            residual = np.abs(p[mask[:, head], head] - y[mask[:, head], head])
            expected.append(float(np.sum(np.where(residual <= 1, .5 * residual**2,
                                                  residual - .5))))
        assert row[name]['sums'] == pytest.approx(expected)
        assert row[name]['counts'] == mask.sum(axis=0).tolist()
    assert row['monitored_loss'] == .5 * (row['train']['loss'] + row['inner']['loss'])
    assert result['best_monitored_loss'] == row['monitored_loss']
    assert row['train']['counts'] == [3, 1]
    assert row['inner']['counts'] == [1, 1]


def test_best_finite_epoch_restored_even_below_min_delta():
    m = model(heads=1)
    x = np.ones((1, 1), np.float32)
    y = np.full((1, 1), .1, np.float32)
    snapshots = {}

    def progress(event):
        if event['event'] == 'epoch':
            snapshots[event['epoch']] = _weight_digest(m.get_weights())

    result = fit(m, (x, y, np.ones_like(y, bool)),
                 cfg=config(max_epochs=10, patience=2, min_delta=10.,
                            learning_rate=1., head_weights=[1.]), progress=progress)
    history = result['history']
    best = min(history, key=lambda row: row['monitored_loss'])
    assert result['stop_reason'] == 'patience'
    assert result['epochs_completed'] == 3
    assert best['epoch'] == 2
    assert result['selected_epoch'] == best['epoch']
    assert result['selected_weights_sha256'] == snapshots[2]
    assert result['restored_weights_sha256'] == _weight_digest(m.get_weights())
    assert result['restored_weights_sha256'] != result['last_weights_sha256']
    assert result['selected_updates'] == 2
    assert result['observed_updates'] == 3


@pytest.mark.parametrize('split', ['train', 'inner'])
@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_invalid_eligible_labels_rejected_before_any_update(split, bad):
    data = arrays()
    invalid_y = data[1].copy()
    invalid_y[0, 0] = bad
    invalid = (data[0], invalid_y, data[2])
    m = model()
    before = _weight_digest(m.get_weights())
    with pytest.raises(ValueError, match='eligible'):
        fit(m, invalid if split == 'train' else data,
            invalid if split == 'inner' else data)
    assert _weight_digest(m.get_weights()) == before


def test_eligible_prediction_rejected_but_masked_nan_prediction_excluded():
    inp = tf.keras.Input((1,))
    m = tf.keras.Model(inp, tf.keras.layers.Lambda(
        lambda x: tf.concat([x * 0, x * float('nan')], axis=1))(inp))
    x, y, mask = arrays()
    with pytest.raises(ValueError, match='eligible prediction'):
        _evaluate_masked(m, x, y, mask, [1., 3.], 1., 2)
    mask[:, 1] = False
    metrics = _evaluate_masked(m, x, y, mask, [1., 3.], 1., 2)
    assert metrics['loss'] == .5


@pytest.mark.parametrize('key', list(config()))
def test_config_boolean_rejected(key):
    with pytest.raises(ValueError):
        fit(model(), cfg=config(**{key: True}))


@pytest.mark.parametrize('key,value', [
    ('max_epochs', 0), ('max_epochs', 101), ('patience', 0),
    ('batch_size', 1.5), ('batch_size', 0), ('seed', -1), ('seed', [17, 18]),
    ('learning_rate', 0.), ('learning_rate', np.inf), ('weight_decay', -.1),
    ('min_delta', np.nan), ('huber_delta', 0.),
])
def test_invalid_numeric_settings_rejected(key, value):
    with pytest.raises(ValueError):
        fit(model(), cfg=config(**{key: value}))


@pytest.mark.parametrize('weights', [[True, 1.], [0., 1.], [-1., 1.],
                                     [np.nan, 1.], [np.inf, 1.], [1.], [[1., 1.]]])
def test_head_weights_strict_positive(weights):
    with pytest.raises(ValueError):
        fit(model(), cfg=config(head_weights=weights))


@pytest.mark.parametrize('key', list(config()))
def test_every_config_key_explicit(key):
    cfg = config()
    del cfg[key]
    with pytest.raises(ValueError, match='config'):
        fit(model(), cfg=cfg)


@pytest.mark.parametrize('key', ['x_validation', 'y_test', 'outer', 'monitor', 'seeds', 'reruns'])
def test_no_outer_split_or_rerun_config(key):
    with pytest.raises(ValueError, match='config'):
        fit(model(), cfg=config(**{key: 'forbidden'}))
    assert list(inspect.signature(fit_masked).parameters) == [
        'model', 'x_train', 'y_train', 'mask_train', 'x_inner', 'y_inner',
        'mask_inner', 'config', 'progress']


def test_shape_mask_and_callback_contracts():
    x, y, mask = arrays()
    for invalid in [(x, y, mask.astype(int)), (x, y, mask[:, :1]),
                    (x[:2], y, mask), (x, y[:, 0], mask[:, 0])]:
        with pytest.raises(ValueError):
            fit(model(), invalid)
    with pytest.raises(ValueError, match='progress'):
        fit(model(), progress=True)
    with pytest.raises(ValueError, match='prediction'):
        fit(model(heads=1))


def test_failure_after_checkpoint_restores_without_retry():
    m = model()
    checkpoint = []
    updates = []

    def progress(event):
        if event['event'] == 'epoch':
            checkpoint.append(_weight_digest(m.get_weights()))
        if event['event'] == 'update':
            updates.append(event['updates'])
            if event['epoch'] == 2:
                raise RuntimeError('callback failure')

    with pytest.raises(RuntimeError, match='callback failure'):
        fit(m, cfg=config(max_epochs=3), progress=progress)
    assert len(checkpoint) == 1
    assert _weight_digest(m.get_weights()) == checkpoint[0]
    assert updates == [1, 2, 3]


def test_finite_json_audit_events_fixed_seed_and_single_attempt(monkeypatch):
    seeds = []
    original = tf.keras.utils.set_random_seed

    def record(seed):
        seeds.append(seed)
        original(seed)

    monkeypatch.setattr(tf.keras.utils, 'set_random_seed', record)
    events = []
    result = fit(model(), progress=events.append)
    assert seeds == [17]
    assert result['attempts'] == 1
    assert result['settings'] == config()
    assert result['stop_reason'] == 'max_epochs'
    assert result['early_stop']['rule']['monitor'] == 'train_inner_mean'
    assert result['early_stop']['rule']['monitor_every'] == 1
    assert len(result['early_stop']['sha256']) == 64
    assert [e['event'] for e in events] == ['update', 'update', 'epoch', 'restored']
    assert [e['updates'] for e in events] == [1, 2, 2, 2]
    json.dumps({'result': result, 'events': events}, allow_nan=False)
