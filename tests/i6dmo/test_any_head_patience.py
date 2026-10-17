import math

import numpy as np
import pytest

from tools.i6dmo import train as T


def update(s, vector):
    return s.update(float(np.mean(vector)), float(np.mean(vector)),
                    [np.array([len(s.history)], dtype=float)], per_head=vector)


def test_one_head_resets_despite_worse_mean_and_preserves_best_checkpoint():
    s = T.AnyHorizonMonitorStopper(2, 1e-8, 20)
    assert not update(s, [1., 1.])
    assert not update(s, [.9, 3.])
    assert s.stale == 0 and s.history[-1]['reset_heads'] == [0]
    assert s.best_epoch == 1
    assert not update(s, [1., 3.])
    assert update(s, [1., 3.])
    assert s.stop_reason == 'patience'


def test_oscillating_above_reference_does_not_reset():
    s = T.AnyHorizonMonitorStopper(2, 1e-8, 20)
    update(s, [1., 1.])
    update(s, [2., 2.])
    assert update(s, [1.5, 1.5])
    assert s.stale == 2


def test_threshold_is_absolute_and_cumulative_against_reference():
    s = T.AnyHorizonMonitorStopper(10, 1e-8, 20)
    update(s, [1., 1.])
    update(s, [1.-5e-9, 1.])
    assert s.stale == 1
    update(s, [1.-2e-8, 1.])
    assert s.stale == 0


def test_ceiling_still_applies_when_improvement_continues():
    s = T.AnyHorizonMonitorStopper(10, 1e-8, 3)
    update(s, [3., 3.])
    update(s, [2., 3.])
    assert update(s, [1., 3.])
    assert s.stop_reason == 'max_epochs'


@pytest.mark.parametrize('vector', [[], [None], [math.nan], [math.inf]])
def test_invalid_vectors_fail_without_updating_history(vector):
    s = T.AnyHorizonMonitorStopper(10, 1e-8, 200)
    with pytest.raises(T.TrainDisposition):
        s.update(1., 1., [np.zeros(1)], per_head=vector)
    assert not s.history


def test_head_order_and_count_cannot_change():
    s = T.AnyHorizonMonitorStopper(10, 1e-8, 200)
    update(s, [1., 1.])
    with pytest.raises(T.TrainDisposition):
        update(s, [1.])


def test_default_policy_still_uses_mean():
    assert type(T.make_stopper(dict(patience=10,min_delta=1e-8,max_epochs=200))) is T.MeanMonitorStopper
    assert isinstance(T.make_stopper(dict(patience=10,min_delta=1e-8,max_epochs=200,
                                         patience_reset='ANY_HORIZON')),T.AnyHorizonMonitorStopper)
    with pytest.raises(T.TrainDisposition):
        T.make_stopper(dict(patience=10,min_delta=1e-8,max_epochs=200,patience_reset='TYPO'))


def test_actual_fit_loop_uses_vector_and_retains_monitor_self_check():
    from tools.i6dmo import model as M
    M.keras.backend.clear_session()
    model = M.build_dmo(M.model_config(['f'+str(i) for i in range(19)],
                                      family='short_mg2_all_horizons'))
    rng = np.random.default_rng(44)
    x = [rng.normal(size=(4,13,19)).astype('float32'),
         np.tile(np.arange(-24,1,2,dtype='float32'),(4,1)),np.zeros((4,13),'float32')]
    split = (x,rng.normal(size=(4,12)),np.ones((4,12),bool))
    settings = dict(patience=10,min_delta=1e-8,max_epochs=2,batch_size=4,
                    learning_rate=.001,weight_decay=.0001,patience_reset='ANY_HORIZON')
    result = T.fit_dmo(model,split,split,settings,6102026,
                       ids=(np.arange(4),np.arange(4)),record_device='cpu')
    assert result['epochs_completed']==2
    assert result['stop_reason']=='max_epochs'
    assert result['early_stop']['patience_reset']=='ANY_HORIZON'
    assert result['monitor_evaluation']['self_check']=='PASS'
    for row in result['history']:
        assert row['per_horizon_monitored']==[.5*(a+b) for a,b in
                                              zip(row['fit']['per_head'],row['inner']['per_head'])]
