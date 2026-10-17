import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    path=Path(__file__).parents[2]/'tools/i6dmo/best_registry.py'
    spec=importlib.util.spec_from_file_location('best_registry',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def arrays():
    return dict(origin=np.array([1,2,3],dtype='int64'), current=np.array([1.,1.,1.]),
                target=np.array([2.,3.,4.]), prediction=np.array([1.5,2.,3.]))


def test_own_naive_and_original_scale():
    m=module();a=arrays();r=m.score(a,4)
    assert r['naive_mae']==2
    assert r['mae']==pytest.approx(2.5/3)
    assert r['relative_error']==pytest.approx((2.5/3-2)/2)


def test_changed_horizon_is_not_a_shared_naive():
    m=module();a=arrays();b=arrays();b['target']=b['target']+5
    assert m.score(a,2)['naive_mae']!=m.score(b,12)['naive_mae']


def test_common_support_excludes_nonshared_rows_and_keeps_native_score():
    m=module();a=arrays();b={k:v[1:] for k,v in arrays().items()}
    p=[dict(id='a',scope='ANNUAL',scores={4:a}),dict(id='b',scope='ANNUAL',scores={4:b})]
    board=m.common_board(p)
    assert board['horizons']['4']['n']==2
    assert m.score(a,4)['n']==3


def test_conflicting_truth_fails_closed():
    m=module();a=arrays();b=arrays();b['target'][1]+=1
    with pytest.raises(ValueError,match='COMMON_TRUTH_MISMATCH'):
        m.common_board([dict(id='a',scope='ANNUAL',scores={4:a}),dict(id='b',scope='ANNUAL',scores={4:b})])


def test_partial_period_never_becomes_annual_champion():
    m=module();a=arrays();p=[dict(id='partial',scope='PERIOD',scores={4:a})]
    assert m.common_board(p)['horizons']=={}


def test_complete_recipe_not_a_best_head_mosaic():
    m=module();a=arrays()
    one=dict(id='only-h4',scope='ANNUAL',scores={4:a})
    all_heads=dict(id='whole',scope='ANNUAL',scores={h:a for h in (2,3,4,5,6,12)})
    assert m.common_board([one,all_heads])['families']['short']['winner']=='whole'


def test_cohort_changes_when_comparison_population_changes():
    m=module();a=arrays()
    first=m.common_board([dict(id='a',scope='ANNUAL',scores={4:a})])
    b={k:v[1:] for k,v in a.items()}
    second=m.common_board([dict(id='a',scope='ANNUAL',scores={4:a}),dict(id='b',scope='ANNUAL',scores={4:b})])
    assert first['horizons']['4']['cohort']!=second['horizons']['4']['cohort']


def test_duplicate_origins_rejected():
    a=arrays();a['origin'][1]=a['origin'][0]
    with pytest.raises(ValueError,match='DUPLICATE_ORIGIN'):module().score(a,4)


def test_improvement_event_is_idempotent_and_records_full_identity(tmp_path):
    m=module();entry=dict(winner='a',relative_error=-.1,cohort='c',horizon=4)
    assert len(m.record_events(tmp_path,{'h4':entry}))==1
    assert len(m.record_events(tmp_path,{'h4':entry}))==0
    entry.update(winner='b',relative_error=-.2)
    event=m.record_events(tmp_path,{'h4':entry})[0]
    assert event['previous']['winner']=='a' and event['current']['winner']=='b'
    assert event['kind']=='IMPROVEMENT'


def test_new_cohort_is_not_called_an_improvement(tmp_path):
    m=module();x=dict(winner='a',relative_error=-.1,cohort='old',horizon=4)
    m.record_events(tmp_path,{'h4':x});x.update(cohort='new',relative_error=-.5)
    assert m.record_events(tmp_path,{'h4':x})[0]['kind']=='NEW_COHORT_BASELINE'


def test_nonfinite_or_empty_scores_rejected():
    m=module();a=arrays();a['prediction'][0]=float('nan')
    with pytest.raises(ValueError,match='INVALID_PAIRED_SCORE'):m.score(a,4)


def test_cadence_overlap_rejected():
    with pytest.raises(ValueError,match='DUPLICATE_ORIGIN'):
        a=arrays();module().score({k:np.concatenate([v,v]) for k,v in a.items()},4)


def test_digest_tamper_and_traversal_rejected(tmp_path):
    m=module();(tmp_path/'artifact').write_text('original')
    digest=m.file_sha256(tmp_path/'artifact')
    m.authenticated(tmp_path,{'artifact':digest})
    (tmp_path/'artifact').write_text('tampered')
    for relative in ('artifact','../artifact','/artifact'):
        with pytest.raises(ValueError,match='ARTIFACT_DIGEST_MISMATCH'):
            m.authenticated(tmp_path,{relative:digest})


def test_report_carries_full_winning_recipe_and_original_scores():
    m=module();profile=dict(id='p',code_commit='c',model={'seed':7},
        training={'early_stop':{'patience':10}},data={'input_sha256':'d'},metrics=[{'horizon':4,'mae':2}])
    event=dict(id='e',key='h4',kind='IMPROVEMENT',previous=None,
               current=dict(winner='p',cohort='cohort',relative_error=-.1,horizon=4,n=9,mae=2.,naive_mae=3.))
    report=m.event_report(event,profile)
    assert m.read_json_tag(report,'profile_json')==profile
    assert m.read_json_tag(report,'event_json')==event
    assert {r['metric'] for r in report['metrics']} >= {'MAE','Naive_MAE','N','Best_Relative_Error_Vs_Naive'}


def test_publication_cannot_silently_report_ready():
    m=module()
    assert m.readiness([],{'pending':1,'mismatch':0},{'state':'VERIFIED'})=='WAIT_PUBLICATION'
    assert m.readiness([],{'pending':0,'mismatch':1},{'state':'VERIFIED'})=='PUBLICATION_MISMATCH'
    assert m.readiness([],None,None)=='COMPUTED_NOT_PUBLISHED'
    assert m.readiness([],{'pending':0,'mismatch':0},{'state':'MISMATCH'})=='WAIT_MIRRORS'
    assert m.readiness([],{'pending':0,'mismatch':0},{'state':'VERIFIED'})=='READY'


def test_registry_copy_excludes_live_database_secrets_and_locks(tmp_path):
    m=module();root=tmp_path/'root';root.mkdir()
    m.put(root/'profiles/p.json',{'seed':7});m.put(root/'BEST.json',{'winner':'p'})
    for name in ('outbox.sqlite3','registry.lock','PRIVATE.secret.env'):
        (root/name).write_text('not mirrored')
    spec={'mirrors':{'local':str(tmp_path/'local'),'remote':str(tmp_path/'remote')}}
    proof=m.sync_registry(root,spec)
    assert proof['state']=='VERIFIED' and proof['files']==2
    for folder in ('local','remote'):
        assert (tmp_path/folder/'profiles/p.json').exists()
        assert not (tmp_path/folder/'outbox.sqlite3').exists()
        assert not (tmp_path/folder/'PRIVATE.secret.env').exists()


def test_publication_retry_never_fits_and_does_not_duplicate(tmp_path):
    from tools.i6dmo.outbox import Outbox, expected_rows
    m=module();profile=dict(id='a'*64,code_commit='b'*40,model={'seed':7},training={},data={},metrics=[])
    event=dict(id='c'*64,key='h4',kind='IMPROVEMENT',current=dict(
        winner=profile['id'],cohort='d'*64,relative_error=-.1,horizon=4))
    report=m.event_report(event,profile);box=Outbox(tmp_path/'outbox.sqlite3')
    class Client:
        failing=True
        def post_report(self,r):
            if self.failing:raise RuntimeError('offline')
            return {'report_sha256':r['report_sha256']}
        def query_metrics(self,d):return [dict(metric=r[0],value=r[1],unit=r[2],horizon=r[3],split=r[4])
            for r in expected_rows(report)]
    client=Client()
    try:
        box.enqueue(report,source_unit='test');assert box.pump(client)['pending']==1
        client.failing=False;box.enqueue(report,source_unit='test')
        assert box.pump(client)['delivered']==1 and len(box.items())==1
    finally:box.close()


def test_timer_is_bounded_cpu_only_and_inherits_credentials_without_reading_them(tmp_path):
    m=module();spec=tmp_path/'spec.json'
    m.put(spec,{'root':str(tmp_path/'results'),'warehouse':{'environment_file':'/private/credentials.env'}})
    calls=[]
    m.install_timer(spec,unit_dir=tmp_path/'units',run=lambda command,**kwargs:calls.append(command))
    unit=(tmp_path/'units/i6dmo-best-registry.service').read_text()
    assert 'MemoryMax=768M' in unit and 'CPUQuota=50%' in unit
    assert 'Environment=CUDA_VISIBLE_DEVICES=' in unit
    assert 'EnvironmentFile=/private/credentials.env\n' in unit
    assert 'WorkingDirectory=/' in unit
    assert 'tick --spec' in unit and '120s' in (tmp_path/'units/i6dmo-best-registry.timer').read_text()
    assert calls[-1]==['systemctl','--user','enable','--now','i6dmo-best-registry.timer']


def test_reference_is_pinned_by_identity_not_ambiguous_experiment_name(tmp_path):
    m=module();p=dict(id='exact',label='same-name',scores={4:arrays()})
    m.pin_reference(tmp_path,[p,dict(p,id='other')],'exact')
    assert m.read(tmp_path/'REFERENCE.json')=={'id':'exact','label':'same-name'}
    with pytest.raises(ValueError,match='REFERENCE_PROFILE_NOT_RECOVERED'):
        m.pin_reference(tmp_path,[dict(p,id='other')],'exact')


def test_olap_provenance_readback_checks_full_tags_and_caches_only_verified(tmp_path):
    m=module();tags={'profile_json':'{"seed":7}'}
    class Box:
        def items(self):return [dict(state='DELIVERED',report_sha256='a'*64,body='{"tags":'+__import__('json').dumps(tags)+'}')]
    class Client:
        calls=0;bad=True
        def _call(self,*args):
            self.calls+=1
            return {'rows':[{'tags_json':__import__('json').dumps({} if self.bad else tags)}]}
    client=Client()
    assert m.verify_provenance(tmp_path,Box(),client)['mismatch']==1
    client.bad=False;assert m.verify_provenance(tmp_path,Box(),client)['verified']==1
    m.verify_provenance(tmp_path,Box(),client);assert client.calls==2
    assert m.readiness([],{'pending':0,'mismatch':0,'provenance':{'mismatch':1,'pending':0}},
                       {'state':'VERIFIED'})=='PUBLICATION_MISMATCH'
