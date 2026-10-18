"""Declared epoch monitors; gradient populations and objectives remain unchanged."""
from datetime import datetime
import math

import numpy as np


ARMS = ('legacy4', 'relative4', 'relative1', 'legacy4_bn', 'relative4_bn', 'relative1_bn')


def monitor_indices(origins, cutoff, years):
    end = datetime.fromisoformat(cutoff)
    try:
        start = end.replace(year=end.year-years)
    except ValueError:
        start = end.replace(year=end.year-years, day=28)
    origins = np.asarray(origins)
    if np.any(origins >= end.timestamp()):
        raise ValueError('MONITOR_ORIGIN_AFTER_CUTOFF')
    ids = np.flatnonzero(origins >= start.timestamp())
    if not len(ids):
        raise ValueError('EMPTY_FIT_MONITOR')
    return ids


def relative_metrics(metrics, naive):
    values = []
    for error, baseline in zip(metrics['per_head'], naive, strict=True):
        if error is None:
            values.append(None)
        elif not math.isfinite(baseline) or baseline <= 0:
            raise ValueError('ZERO_MONITOR_NAIVE')
        else:
            values.append(error / baseline - 1)
    active = [v for v in values if v is not None]
    if not active:
        raise ValueError('NO_SUPPORTED_MONITOR_HEAD')
    return dict(metrics, original_loss=metrics.get('loss'), per_head=values,
                own_naive=list(naive), loss=math.fsum(active)/len(active))


def normalized_naive(y, mask):
    return [math.fsum(np.abs(y[mask[:, h], h]).tolist())/int(mask[:, h].sum())
            if mask[:, h].any() else 0. for h in range(mask.shape[1])]


class MonitorAdapter:
    def __init__(self, indices, fit_naive, inner_naive, arm):
        if arm not in ARMS:
            raise ValueError('UNKNOWN_MONITOR_ARM')
        self.indices = np.asarray(indices, dtype=np.int64)
        self.fit_naive = list(fit_naive)
        self.inner_naive = list(inner_naive)
        self.arm = arm

    def identity(self):
        from tools.i6dmo import weekly18 as W
        return dict(arm=self.arm, fit_rows=len(self.indices),
                    indices_sha256=W.B.digest(self.indices.tolist()),
                    fit_naive=self.fit_naive, calibration_naive=self.inner_naive,
                    monitor='mean(half_FIT_relative+half_known_calibration_relative)' if
                    self.arm.startswith('relative') else 'legacy_TRAIN_naive_normalization')

    def evaluate(self, model, fit, inner, batch_size, ids=None):
        from tools.i6dmo import train as T
        x, y, mask = fit
        selected = (T._take(x, self.indices), y[self.indices], mask[self.indices])
        selected_ids = None if ids is None else (np.asarray(ids[0])[self.indices], ids[1])
        fm, im, _, witness = T.evaluate_monitored(model, selected, inner, batch_size, ids=selected_ids)
        if self.arm.startswith('relative'):
            fm, im = relative_metrics(fm, self.fit_naive), relative_metrics(im, self.inner_naive)
        witness['monitor'] = self.identity()
        return fm, im, .5*(fm['loss']+im['loss']), witness


def adapter(arm, fit_arrays, fit, inner, cutoff):
    years = 1 if arm.startswith('relative1') else 4
    indices = monitor_indices(fit_arrays['origin_ts'], cutoff, years)
    return MonitorAdapter(indices, normalized_naive(fit[1][indices], fit[2][indices]),
                          normalized_naive(inner[1], inner[2]), arm)


def build_model(config):
    from tools.i6dmo import continuation_models as M
    if set(config) != {'schema', 'base', 'batch_normalization'} or config['schema'] != 'i6dmo.monitor_model.v1':
        raise ValueError('MONITOR_MODEL_CONFIG')
    c = M._validated(config['base'])
    if config['batch_normalization'] is not True or c['kind'] != 'conv':
        raise ValueError('BN_CONV_REQUIRED')
    M.M.enable_determinism()
    M.keras.utils.set_random_seed(c['seed'])
    observations = M.keras.Input((c['slots'], len(c['features'])), name='observations')
    offsets = M.keras.Input((c['slots'],), name='offsets_hours')
    closure = M.keras.Input((c['slots'],), name='closure_crossed')
    checked = M.R.TimeInputGuard(c['spacing_open_market_hours'], name='time_input_guard')([offsets, closure])
    outputs = M._conv_outputs(c, observations, *checked, batch_norm=True)
    model = M.R.HeadListModel([observations, offsets, closure], outputs, name='i6dmo_monitor_forecast')
    return M.ContinuationBundle(config, model)


def insert_bn(original, config):
    from predictor_plugins.modular_temporal import weights_hash
    c = dict(schema='i6dmo.monitor_model.v1', base=config, batch_normalization=True)
    bundle = build_model(c)
    for layer in original.layers:
        target = bundle.forecast_model.get_layer(layer.name)
        target.set_weights(layer.get_weights())
        if any(not np.array_equal(a, b) for a, b in zip(layer.get_weights(), target.get_weights(), strict=True)):
            raise ValueError('BN_CHANGED_EXISTING_LAYER')
    return bundle, dict(original_weights_sha256=weights_hash(original),
                        copied_layers=[l.name for l in original.layers],
                        new_layer='core_batch_normalization', momentum=.99, epsilon=.001)
