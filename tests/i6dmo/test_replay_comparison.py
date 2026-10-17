import copy

import pytest

from tools.i6dmo.replay_comparison import compare_training


def reference():
    return dict(settings={'seed': 12}, initial_weights_sha256='initial', epochs_completed=2,
                best_epoch=1, stop_reason='patience', restored_weights_sha256='best',
                history=[dict(epoch=i, fit_loss=.2, inner_loss=.3, monitored=.25,
                              stale=i-1, best_epoch=1, updates=i*5, weights_sha256=str(i))
                         for i in (1, 2)])


def test_identical_trajectory_is_reproduced():
    a = reference()
    assert compare_training(a, copy.deepcopy(a))['exact_training_reproduction']


@pytest.mark.parametrize('field', ['weights_sha256', 'monitored', 'updates', 'best_epoch'])
def test_cannot_hide_changed_training_even_when_scores_are_close(field):
    a = reference()
    b = copy.deepcopy(a)
    b['history'][0][field] = 'changed'
    result = compare_training(a, b)
    assert not result['exact_training_reproduction']
    assert result['first_different_epoch'] == 1


def test_longer_trajectory_is_not_exact_reproduction():
    a = reference()
    b = copy.deepcopy(a)
    b['history'].append(dict(b['history'][-1], epoch=3))
    assert not compare_training(a, b)['exact_training_reproduction']
