import copy
import inspect

import pytest

from tools.i6dmo import all_horizons as A


def configs():
    c = dict(model=dict(horizons_hours=A.SHORT+A.LONG,core=[8,4]),seed=6102026,
             source_receipt_sha256='source',training=dict(patience_reset='ANY_HORIZON',patience=10),
             scales=dict(input_mean=[1],input_std=[2],rows=10,
                         naive_mae=list(range(1,13)),counts=[5]*12,scales_sha256='long'),
             support={'fit':dict(rows=10,counts=[5]*12)})
    s = copy.deepcopy(c)
    s['model']['horizons_hours']=A.SHORT
    s['scales']['naive_mae']=s['scales']['naive_mae'][:6]
    s['scales']['counts']=s['scales']['counts'][:6]
    s['scales']['scales_sha256']='short'
    s['support']['fit']['counts']=s['support']['fit']['counts'][:6]
    return s,c


def test_exact_projection_only_removes_long_heads():
    short,control=configs()
    assert A.HORIZONS==A.SHORT
    A.verify_short_projection(short,control)


@pytest.mark.parametrize('change',['model','seed','training','scales','support','source'])
def test_other_changes_rejected(change):
    short,control=configs()
    if change=='model':short['model']['core']=[4]
    elif change=='seed':short['seed']=4
    elif change=='training':short['training']['patience']=2
    elif change=='scales':short['scales']['input_mean']=[7]
    elif change=='support':short['support']['fit']['rows']=9
    else:short['source_receipt_sha256']='other'
    with pytest.raises(ValueError):A.verify_short_projection(short,control)


def test_replay_calls_original_entrypoint_without_extra_fit_or_forward():
    source=inspect.getsource(A.run)
    assert 'T.train_unit(' in source
    assert 'T.fit_dmo(' not in source
    assert 'build_dmo(' not in source
    assert 'predict_matrix(' not in source
