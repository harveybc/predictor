import ast
import copy
import subprocess

import numpy as np
import pytest

from tools.i6dmo import weekly_window_core as S, weekly18 as W


def test_finite_plan_is_control_first_then_conditional_joint_arms():
    arms=S.arms()
    assert [a['id'] for a in arms[:4]]==['window13','window7','window25','window49']
    assert len(arms)==12
    assert all(a['core']==[2] for a in arms[:4])
    assert {(a['slots'],tuple(a['core'])) for a in arms[4:]}=={(n,c) for n in (7,13,25,49) for c in ((4,),(4,2))}


def test_market_window_includes_t_and_skips_closure_not_missing_observation():
    grid=np.array([0,1,2,3,4,8,9,10,11,12,13,14,15])*3600
    source=dict(ts=grid.copy(),x=np.arange(len(grid)*2).reshape(-1,2).astype(float))
    old=dict(row_ids=np.array([grid[-1]]),mask=np.ones((1,7),bool))
    a,valid=S.window_arrays(old,source,grid,7)
    assert valid.all() and a['offsets_hours'][0,-1]==0
    assert a['x'][0,-1].tolist()==source['x'][-1].tolist()
    assert a['closure_crossed'].any() and np.all(a['offsets_hours']<=0)
    missing=source['ts']!=grid[-3];source={k:v[missing] for k,v in source.items()}
    _,valid=S.window_arrays(old,source,grid,7)
    assert not valid.any()


def test_common_rows_preserve_head_masks_not_all_head_intersection():
    a=dict(row_ids=np.arange(3),mask=np.array([[1,0],[1,1],[0,1]],bool),x=np.ones((3,2,1)))
    b=S.subset(a,np.array([1,1,0],bool))
    assert len(b['row_ids'])==2 and b['mask'].tolist()==[[True,False],[True,True]]


def test_annual_rank_never_uses_partial_arm_and_tie_prefers_small_window():
    rows=[dict(horizon=h,n=2,mae=.9,naive_mae=1.,relative_error=-.1,
               sum_abs_error=1.8,sum_abs_naive_error=2.) for h in S.HEADS]
    summaries={a['id']:dict(state='COMPLETE',metrics=rows,slots=a['slots']) for a in S.arms()[:4]}
    assert S.best_windows(summaries)==[7,13]
    summaries['window49']['state']='INCOMPLETE'
    assert S.best_windows(summaries) is None


def test_changed_cohort_cannot_be_ranked():
    rows=[dict(horizon=h,n=2,naive_mae=1.,relative_error=-.1) for h in S.HEADS]
    s={a['id']:dict(state='COMPLETE',metrics=copy.deepcopy(rows),slots=a['slots']) for a in S.arms()[:4]}
    s['window49']['metrics'][0]['naive_mae']=2.
    with pytest.raises(ValueError,match='UNPAIRED_ANNUAL'):
        S.best_windows(s)


def test_frozen_trainer_and_layer_builders_only_slot_catalogue_changed():
    for rel in W.FROZEN:
        assert (W.REPO/rel).read_bytes()==subprocess.check_output(['git','show',S.CODE_BASE+':'+rel],cwd=W.REPO)
    rel='tools/i6dmo/resolution_models.py'
    old=subprocess.check_output(['git','show',S.CODE_BASE+':'+rel],cwd=W.REPO).decode()
    assert (W.REPO/rel).read_text()==old.replace('SLOTS = (6, 9, 13, 18, 24)',S.SLOT_CATALOGUE)


def test_core_selection_reuses_completed_baseline_and_skips_unselected_windows():
    assert S.arm_disposition(dict(stage='windows',slots=49),None)=='RUN'
    assert S.arm_disposition(dict(stage='cores',slots=7),[7,13])=='RUN'
    assert S.arm_disposition(dict(stage='cores',slots=49),[7,13])=='SKIPPED_NOT_SELECTED'
    assert S.arm_disposition(dict(stage='cores',slots=7),None)=='BLOCKED_DEPENDENCY'


def test_window_is_invariant_to_future_values():
    grid=np.arange(40)*3600
    source=dict(ts=grid,x=np.arange(80,dtype=float).reshape(40,2))
    old=dict(row_ids=np.array([grid[25]]))
    a,ok=S.window_arrays(old,source,grid,7)
    source['x'][26:]=9999
    b,ok2=S.window_arrays(old,source,grid,7)
    assert ok.all() and ok2.all() and np.array_equal(a['x'],b['x'])


def test_partial_or_failed_coarse_year_cannot_authorize_core_fits():
    assert S.best_windows({}) is None
    assert S.arm_disposition(dict(stage='cores',slots=7),S.best_windows({}))=='BLOCKED_DEPENDENCY'


def test_corrupted_terminal_rejected_before_reuse(tmp_path):
    c=dict(id='one');spec=dict(batch_id='sweep')
    W.B.put(tmp_path/'results/one/RESULT.json',dict(cell=c,batch_id='sweep',state='COMPLETE',files={},sha256='bad'))
    with pytest.raises(ValueError,match='TERMINAL_IDENTITY'):W.terminal(tmp_path,c,spec)
