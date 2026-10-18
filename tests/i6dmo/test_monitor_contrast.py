from datetime import datetime, timezone

import numpy as np
import pytest

from tools.i6dmo import monitor_contrast as MC


def test_own_naive_and_equal_head_then_equal_set_weight():
    # First head improves 50%, second loses 100%; different absolute scales.
    metrics = dict(per_head=[.5, 6.], loss=3.25)
    got = MC.relative_metrics(metrics, [1., 3.])
    assert got['per_head'] == [-.5, 1.]
    assert got['loss'] == .25
    assert metrics['loss'] == 3.25


def test_unsupported_and_zero_naive_are_not_silent_zeroes():
    assert MC.relative_metrics(dict(per_head=[None, 2.]), [0., 4.])['loss'] == -.5
    with pytest.raises(ValueError, match='ZERO_MONITOR_NAIVE'):
        MC.relative_metrics(dict(per_head=[1.]), [0.])


def test_calendar_monitor_selection_leaves_fit_intact():
    def ts(s):
        return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp())
    origins = np.array([ts('2022-02-28'), ts('2023-02-27'), ts('2023-02-28'), ts('2024-02-28')])
    before = origins.copy()
    idx = MC.monitor_indices(origins, '2024-02-29T00:00:00+00:00', 1)
    assert idx.tolist() == [2, 3]
    assert np.array_equal(before, origins)
    with pytest.raises(ValueError, match='MONITOR_ORIGIN_AFTER_CUTOFF'):
        MC.monitor_indices(origins, '2023-01-01T00:00:00+00:00', 1)


def test_current_formula_preserved_when_adapter_absent():
    from tools.i6dmo.train import MeanMonitorStopper
    stop = MeanMonitorStopper(10, 1e-8, 200)
    stop.update(.4, .8, [np.array([1.])])
    assert stop.best_monitored == pytest.approx(.6)


def test_selector_and_normalizer_apply_only_to_monitor(monkeypatch):
    from tools.i6dmo import train as T
    x = np.arange(6).reshape(3, 2)
    fit = (x, np.array([[1.], [2.], [4.]]), np.ones((3, 1), bool))
    inner = (x[:1], np.array([[8.]]), np.ones((1, 1), bool))
    seen = []
    def evaluate(model, a, b, batch, ids=None):
        seen.append((a[0].copy(), ids))
        return ({'per_head': [1.], 'loss': 1.}, {'per_head': [2.], 'loss': 2.}, 1.5, {})
    monkeypatch.setattr(T, 'evaluate_monitored', evaluate)
    adapter = MC.MonitorAdapter(np.array([1, 2]), [3.], [8.], 'relative1')
    fm, vm, m, witness = adapter.evaluate(None, fit, inner, 256, (np.array([10, 20, 30]), np.array([40])))
    assert seen[0][0].tolist() == x[1:].tolist()
    assert seen[0][1][0].tolist() == [20, 30]
    assert len(fit[0]) == 3
    assert m == pytest.approx((1/3-1 + 2/8-1)/2)
    assert witness['monitor']['arm'] == 'relative1'


def test_packet_and_result_changes_are_rejected(tmp_path):
    from tools.i6dmo import monitor_campaign as C
    packet = tmp_path/'PREPARED.json'
    C.seal(packet, dict(files={}, units=[]))
    assert C.checked_doc(packet)['units'] == []
    with pytest.raises(ValueError, match='IMMUTABLE_DOCUMENT_CHANGED'):
        C.seal(packet, dict(files={}, units=[1]))
    C.W.B.put(packet, dict(files={}, units=[1], sha256='wrong'))
    with pytest.raises(ValueError, match='DOCUMENT_IDENTITY'):
        C.checked_doc(packet)


def test_fit_metric_is_not_reported_as_train_replay(tmp_path):
    from tools.i6dmo import monitor_campaign as C
    doc = dict(units=[dict(id='short__week', family='short')], initial={'short':{'model':{'horizons_hours':[2]}}}, coverage={'short':{}})
    report, _ = C.summarize(doc, tmp_path, 'identity', 'short')
    assert report['arms']['relative1']['TRAIN_REPLAY'] is None
    assert report['arms']['relative1']['train_replay_status'] == 'PENDING_NOT_FIT'
