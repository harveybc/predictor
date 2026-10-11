"""Bounded masked I6N candidate training on caller-supplied arrays only.

``fit_masked`` accepts TRAIN and a TRAIN-derived inner stopping split, never
outer VALIDATION/TEST. The caller owns split provenance, eligibility masks,
and target scales: fit scales on eligible TRAIN labels only, outside this
helper, then apply the frozen scales to both arrays. No scaling or data reads
occur here. Array names alone cannot establish provenance.

Config is a flat, complete mapping: max_epochs (1..100), patience (1..50),
min_delta (>=0), batch_size (1..65536), learning_rate (>0), weight_decay (>=0),
huber_delta (>0), seed (0..2**31-1), head_weights (strictly positive list).
Booleans, missing and unknown settings are errors. The monitor is fixed to
mean(TRAIN, inner), evaluated every epoch at the SAME frozen checkpoint.
Each split uses global per-head Huber sums/counts, then a weighted mean over
nonempty heads; empty heads are excluded. The evaluation utility returns zero
for an entirely empty split, but fitting rejects TRAIN or inner without any
eligible labels as NO_ELIGIBLE_TRAIN/NO_ELIGIBLE_INNER before optimizer setup.

One fixed seed is installed once, with no shuffle, retries or model resets.
The caller must also seed model construction when reproducible initialization
is required: this API receives an existing model. AdamW is fresh per call.
An entirely masked batch skips the forward pass AND optimizer/weight decay.
Regularization/model-added losses are not part of the masked objective.

Patience counts epochs without improvement > min_delta against its last
significant reference. Independently, the lowest finite epoch is restored
(first wins ties), including improvements smaller than min_delta. Only model
weights are restored, not optimizer state; this is selection, not resumption.
Invalid labels/predictions, nonfinite arithmetic or callback errors fail fast,
without retry; a previously selected checkpoint is restored before raising.
Progress receives JSON-safe update/epoch/restored dicts and must not mutate
the model or inputs, raise, or block. Resource admission belongs to the caller.
"""

from collections.abc import Mapping
import hashlib
import json
import math

import numpy as np

from tools.modular_candidate_evaluator import _finite, _integer, _number, _weight_digest


_KEYS = frozenset(('max_epochs', 'patience', 'min_delta', 'batch_size',
                   'learning_rate', 'weight_decay', 'huber_delta', 'seed',
                   'head_weights'))


def _settings(config):
    if not isinstance(config, Mapping) or set(config) != _KEYS:
        raise ValueError('config requires exactly the documented training keys')
    cfg = dict(config)
    for name, high in (('max_epochs', 100), ('patience', 50),
                       ('batch_size', 65536), ('seed', 2**31 - 1)):
        _integer(cfg[name], name, 0 if name == 'seed' else 1, high)
    for name in ('learning_rate', 'huber_delta', 'weight_decay', 'min_delta'):
        _number(cfg[name], name, 0, inclusive=name in ('weight_decay', 'min_delta'))
        cfg[name] = float(cfg[name])
    weights = cfg['head_weights']
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError('head_weights must be a nonempty positive list')
    for weight in weights:
        _number(weight, 'head_weights', 0, inclusive=False)
    cfg['head_weights'] = [float(weight) for weight in weights]
    return cfg


def _split(x, y, mask, name):
    x, y, mask = np.asarray(x), np.asarray(y), np.asarray(mask)
    if x.ndim < 2 or min(x.shape) < 1 or x.dtype.kind not in 'fi':
        raise ValueError(f'{name} inputs must be nonempty numeric arrays')
    if y.ndim != 2 or min(y.shape) < 1 or y.dtype.kind not in 'fi':
        raise ValueError(f'{name} labels must have shape (examples, heads)')
    if len(x) != len(y) or mask.shape != y.shape or mask.dtype != np.bool_:
        raise ValueError(f'{name} mask must be boolean and match labels and inputs')
    _finite(x, f'{name} inputs')
    _finite(y[mask], f'{name} eligible labels')
    return x, y, mask


def _evaluate_masked(model, x, y, mask, head_weights, delta, batch_size):
    """Frozen-checkpoint metrics, never a mean of batch means or SGD losses."""
    sums = [[] for _ in head_weights]
    counts = np.zeros(len(head_weights), dtype=np.int64)
    for start in range(0, len(x), batch_size):
        stop = start + batch_size
        valid = mask[start:stop]
        if not valid.any():
            continue
        prediction = np.asarray(model(x[start:stop], training=False))
        target = y[start:stop]
        if prediction.shape != target.shape or prediction.dtype.kind not in 'fi':
            raise ValueError('prediction must match (examples, heads) labels')
        _finite(prediction[valid], 'eligible prediction')
        for head in range(len(head_weights)):
            eligible = valid[:, head]
            # Selection precedes subtraction; placeholders never form residuals.
            with np.errstate(over='ignore', invalid='ignore'):
                residual = np.abs(prediction[eligible, head].astype(np.float64)
                                  - target[eligible, head].astype(np.float64))
                quadratic = np.minimum(residual, delta)
                loss = (.5 * quadratic) * quadratic + delta * (residual - quadratic)
            _finite(loss, 'eligible Huber loss')
            sums[head].append(math.fsum(loss.tolist()))
            counts[head] += int(eligible.sum())
    total = [math.fsum(parts) for parts in sums]
    _finite(total, 'per-head sums')
    per_head = [value / int(n) if n else 0. for value, n in zip(total, counts)]
    # Normalize weights first to avoid overflow in otherwise finite means.
    active = np.asarray(head_weights, np.float64)[counts > 0]
    if len(active):
        active = active / active.max()
        active = active / math.fsum(active.tolist())
        loss = math.fsum(value * float(weight) for value, weight in
                         zip(np.asarray(per_head)[counts > 0], active))
    else:
        loss = 0.
    _finite(loss, 'split loss')
    return dict(sums=total, counts=counts.tolist(), per_head=per_head, loss=loss)


def _early_stop_identity(cfg):
    rule = dict(implementation='i6n.masked_early_stop.v1',
                monitor='train_inner_mean', monitor_every=1,
                aggregation='global_per_head_sums_counts',
                restore='lowest_finite_epoch_first_tie',
                min_delta_role='patience_only',
                patience=cfg['patience'], min_delta=cfg['min_delta'],
                max_epochs=cfg['max_epochs'])
    canonical = json.dumps(rule, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(rule=rule, sha256=hashlib.sha256(canonical.encode()).hexdigest())


def fit_masked(model, x_train, y_train, mask_train, x_inner, y_inner, mask_inner,
               config, progress=None):
    """Fit once, restore best model weights, and return a finite JSON audit dict.

    ``history`` includes per-split sums/counts/per_head/loss, owner mean, updates,
    skipped batches, patience state and checkpoint hash for each complete epoch.
    No file, outer-split, device-selection or optimizer override is accepted.
    """
    cfg = _settings(config)
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')
    train = _split(x_train, y_train, mask_train, 'TRAIN')
    inner = _split(x_inner, y_inner, mask_inner, 'inner')
    if train[0].shape[1:] != inner[0].shape[1:] or train[1].shape[1:] != inner[1].shape[1:]:
        raise ValueError('TRAIN and inner dimensions disagree')
    if len(cfg['head_weights']) != train[1].shape[1]:
        raise ValueError('head_weights must match label heads')
    for name, split in (('TRAIN', train), ('INNER', inner)):
        if not split[2].any():
            raise ValueError(f'NO_ELIGIBLE_{name}')

    import tensorflow as tf
    from tools.i6n_contracts.temporal_ops import masked_huber

    tf.keras.utils.set_random_seed(cfg['seed'])
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg['learning_rate'],
                                         weight_decay=cfg['weight_decay'])
    # Preflight both splits before the first optimizer mutation, including lazy
    # model construction, output shapes and eligible predictions.
    for split in (train, inner):
        prediction = np.asarray(model(split[0][:1], training=False))
        if prediction.shape != split[1][:1].shape:
            raise ValueError('prediction must match (examples, heads) labels')
        _evaluate_masked(model, *split, cfg['head_weights'], cfg['huber_delta'], cfg['batch_size'])
    initial_digest = _weight_digest(model.get_weights())
    history, best_weights, best_digest = [], None, None
    best_loss, patience_reference = math.inf, math.inf
    best_epoch = selected_updates = stale = updates = skipped = 0
    stop_reason = 'max_epochs'

    def report(**fields):
        if progress is not None:
            event = dict(updates=updates, best_epoch=best_epoch,
                         best_monitored_loss=None if best_weights is None else best_loss,
                         **fields)
            json.dumps(event, allow_nan=False)
            progress(event)

    try:
        for epoch in range(1, cfg['max_epochs'] + 1):
            epoch_skipped = 0
            for start in range(0, len(train[0]), cfg['batch_size']):
                stop = start + cfg['batch_size']
                xb, yb, mb = (part[start:stop] for part in train)
                if not mb.any():
                    skipped += 1
                    epoch_skipped += 1
                    continue
                with tf.GradientTape() as tape:
                    prediction = model(xb, training=True)
                    loss = masked_huber(yb, prediction, mb, cfg['head_weights'], cfg['huber_delta'])
                _finite(loss.numpy(), 'training loss')
                gradients = tape.gradient(loss, model.trainable_variables)
                pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
                if not pairs:
                    raise ValueError('eligible batch has no trainable gradients')
                for gradient, _ in pairs:
                    value = gradient.values if isinstance(gradient, tf.IndexedSlices) else gradient
                    _finite(value.numpy(), 'training gradient')
                before = int(optimizer.iterations.numpy())
                optimizer.apply_gradients(pairs)
                if int(optimizer.iterations.numpy()) != before + 1:
                    raise ValueError('expected one optimizer update per eligible batch')
                updates += 1
                report(event='update', epoch=epoch, batch_start=start)
            train_metrics, inner_metrics = (
                _evaluate_masked(model, *split, cfg['head_weights'], cfg['huber_delta'],
                                 cfg['batch_size']) for split in (train, inner))
            monitored_loss = .5 * train_metrics['loss'] + .5 * inner_metrics['loss']
            _finite(monitored_loss, 'monitored loss')
            epoch_digest = _weight_digest(model.get_weights())
            if monitored_loss < best_loss:
                best_loss, best_epoch, selected_updates = monitored_loss, epoch, updates
                best_weights = [value.copy() for value in model.get_weights()]
                best_digest = epoch_digest
            if monitored_loss < patience_reference - cfg['min_delta']:
                patience_reference, stale = monitored_loss, 0
            else:
                stale += 1
            history.append(dict(epoch=epoch, train=train_metrics, inner=inner_metrics,
                                monitored_loss=monitored_loss, updates=updates,
                                skipped_batches=epoch_skipped, stale_epochs=stale,
                                weights_sha256=epoch_digest))
            report(event='epoch', epoch=epoch, train_loss=train_metrics['loss'],
                   inner_loss=inner_metrics['loss'], monitored_loss=monitored_loss,
                   stale_epochs=stale, weights_sha256=epoch_digest)
            if stale >= cfg['patience']:
                stop_reason = 'patience'
                break
    except Exception:
        if best_weights is not None:
            model.set_weights(best_weights)
        raise

    last_digest = _weight_digest(model.get_weights())
    model.set_weights(best_weights)
    restored_digest = _weight_digest(model.get_weights())
    if restored_digest != best_digest:
        raise ValueError('restored weights differ from selected checkpoint')
    iterations = int(optimizer.iterations.numpy())
    if iterations != updates:
        raise ValueError('optimizer iterations disagree with observed updates')
    report(event='restored', epoch=best_epoch, weights_sha256=restored_digest)
    result = dict(settings=cfg, attempts=1, history=history,
                  counts=dict(train=train[2].sum(axis=0).tolist(),
                              inner=inner[2].sum(axis=0).tolist()),
                  epochs_completed=len(history), monitor_evaluations=len(history),
                  best_epoch=best_epoch, selected_epoch=best_epoch,
                  selected_updates=selected_updates, best_monitored_loss=best_loss,
                  observed_updates=updates, optimizer_iterations=iterations,
                  initial_optimizer_iterations=0, skipped_batches=skipped,
                  stop_reason=stop_reason,
                  stop_class='no_improvement' if stop_reason == 'patience' else 'budget',
                  restored_best_weights=True, initial_weights_sha256=initial_digest,
                  selected_weights_sha256=best_digest, restored_weights_sha256=restored_digest,
                  last_weights_sha256=last_digest, early_stop=_early_stop_identity(cfg))
    json.dumps(result, allow_nan=False)
    return result
