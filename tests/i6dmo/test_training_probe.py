import pytest

from tools.i6dmo import training_probe as P


def result(key, epoch, date):
    return dict(key=key,arm='legacy4',epochs=epoch,state='COMPLETE',
                unit={'period':{'train_end':date}})


def test_case_selection_is_deterministic_and_has_no_score_input():
    rows=[result('later',11,'2024-02-01'),result('early',11,'2024-01-01'),result('longest',97,'2024-03-01')]
    assert P.select_cases(rows)==[('legacy4_first11','early'),('legacy4_longest','longest')]
    assert P.select_cases(list(reversed(rows)))==P.select_cases(rows)


def test_epoch_prefix_rejects_changed_monitor_and_missing_epoch():
    row=dict(epoch=1,fit_loss=.4,inner_loss=.8,monitored=.6,updates=65,weights_sha256='weights')
    assert P.prefix([row],[dict(row),dict(row,epoch=2)])['exact']
    assert not P.prefix([row],[dict(row,monitored=.7)])['exact']
    assert not P.prefix([row],[dict(row,weights_sha256='different')])['exact']
    assert not P.prefix([row],[])['exact']


def test_budget_changes_only_two_explicit_stopping_parameters():
    old=dict(batch_size=256,learning_rate=.001,weight_decay=.0001,max_epochs=200,patience=10,min_delta=1e-8)
    new=P.budget(old)
    assert new['max_epochs']==2000 and new['patience']==50
    assert {k for k in old if old[k]!=new[k]}=={'max_epochs','patience'}
    assert old['max_epochs']==200


def test_worker_waits_for_parent_before_any_gpu_operation(tmp_path,monkeypatch):
    packet=tmp_path/'packet';parent=tmp_path/'parent';root=tmp_path/'output'
    P.C.seal(packet/'PREPARED.json',dict(units=[dict(id='case',family='short')],files={}))
    P.W.B.put(parent/'STATUS.json',dict(complete=299,total=300))
    def forbidden(*a,**k):raise AssertionError('GPU operation while parent still running')
    monkeypatch.setattr(P.subprocess,'check_output',forbidden)
    result=P.worker(dict(packet=str(packet),output=str(root),parent_output=str(parent),code={}), 'short')
    assert result['state']=='WAIT_PARENT' and result['parent_complete']==299
