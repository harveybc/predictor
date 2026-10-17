"""Read-only model score recovery and durable best-result ledger. Never trains."""
import argparse
import fcntl
import hashlib
import json
import math
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np

from tools.i6n_swarm.common import atomic_json, file_sha256, sha256_json

FAMILIES = {'short': (2,3,4,5,6,12), 'long': (18,24,36,48,60,72)}


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    atomic_json(Path(path), value, mode=0o600)


def score(a, horizon):
    if not len(a['origin']) or len(set(a['origin'].tolist())) != len(a['origin']):
        raise ValueError('DUPLICATE_ORIGIN_OR_EMPTY_SCORE')
    if any(len(v)!=len(a['origin']) for v in a.values()) or not all(np.isfinite(v).all() for v in a.values()):
        raise ValueError('INVALID_PAIRED_SCORE')
    ix=np.argsort(a['origin']); a={k:v[ix] for k,v in a.items()}
    se=math.fsum(np.abs(a['prediction']-a['target']).tolist())
    sn=math.fsum(np.abs(a['current']-a['target']).tolist())
    if sn<=0:
        raise ValueError('INVALID_PAIRED_SCORE_ZERO_NAIVE')
    digest=hashlib.sha256()
    digest.update(str(horizon).encode())
    for key,dtype in (('origin','<i8'),('current','<f8'),('target','<f8')):
        digest.update(np.asarray(a[key],dtype=dtype).tobytes())
    return dict(horizon=horizon,n=len(ix),mae=se/len(ix),naive_mae=sn/len(ix),
                relative_error=(se-sn)/sn,cohort=digest.hexdigest())


def common_board(profiles):
    annual=[p for p in profiles if p['scope']=='ANNUAL']
    hs=sorted({h for p in annual for h in p['scores']})
    by_h={}; projected={p['id']:{} for p in annual}
    for h in hs:
        candidates=[p for p in annual if h in p['scores']]
        shared=set(candidates[0]['scores'][h]['origin'].tolist())
        for p in candidates[1:]:
            shared.intersection_update(p['scores'][h]['origin'].tolist())
        if not shared:
            raise ValueError('COMMON_SUPPORT_EMPTY:'+str(h))
        origins=np.array(sorted(shared),dtype='int64'); reference=None; ranks=[]
        for p in candidates:
            a=p['scores'][h]; ix=np.argsort(a['origin']); a={k:v[ix] for k,v in a.items()}
            idx=np.searchsorted(a['origin'],origins); a={k:v[idx] for k,v in a.items()}
            if reference is not None and (not np.array_equal(reference['current'],a['current'])
                                         or not np.array_equal(reference['target'],a['target'])):
                raise ValueError('COMMON_TRUTH_MISMATCH:'+str(h))
            reference=a
            r=score(a,h); projected[p['id']][h]=r
            ranks.append(dict(profile=p['id'],**r))
        ranks.sort(key=lambda r:(r['relative_error'],r['profile']))
        by_h[str(h)]=dict(horizon=h,cohort=ranks[0]['cohort'],n=len(origins),
                          winner=ranks[0]['profile'],relative_error=ranks[0]['relative_error'],ranking=ranks)
    families={}
    for name, required in FAMILIES.items():
        ranks=[]
        for p in annual:
            if set(required)<=set(projected[p['id']]):
                rows=[projected[p['id']][h] for h in required]
                ranks.append(dict(profile=p['id'],relative_error=math.fsum(r['relative_error'] for r in rows)/len(rows),
                                  wins=sum(r['relative_error']<0 for r in rows),metrics=rows))
        if ranks:
            ranks.sort(key=lambda r:(r['relative_error'],r['profile']))
            families[name]=dict(winner=ranks[0]['profile'],relative_error=ranks[0]['relative_error'],
                cohort=sha256_json([by_h[str(h)]['cohort'] for h in required]),ranking=ranks)
    return dict(horizons=by_h,families=families)


def record_events(root, entries):
    root=Path(root); state_path=root/'BEST.json'
    previous=read(state_path) if state_path.exists() else {}
    changed=[]
    for key,current in entries.items():
        archive_key=key+':'+current['cohort']
        old=previous.get(archive_key)
        if old is not None and current['relative_error']>=old['relative_error']:
            continue
        event=dict(key=key,kind='NEW_COHORT_BASELINE' if old is None else 'IMPROVEMENT',
                   previous=old,current=current)
        event['id']=sha256_json(event)
        path=root/'events'/(event['id']+'.json')
        if not path.exists():
            put(path,dict(event,observed_at=time.time()))
        previous[archive_key]=current
        changed.append(event)
    put(state_path,previous)
    return changed


def authenticated(root, files):
    for rel, expected in files.items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or file_sha256(root/rel)!=expected:
            raise ValueError('ARTIFACT_DIGEST_MISMATCH:'+rel)


def load_predictions(path, horizons):
    with np.load(path,allow_pickle=False) as z:
        names=z.files
        actual=z['horizons_hours'].tolist() if 'horizons_hours' in names else horizons
        if list(actual)!=list(horizons):
            raise ValueError('PREDICTION_HORIZONS_MISMATCH')
        origin=z['row_id'] if 'row_id' in names else z['origin_ts']
        prediction=z['predicted_price'] if 'predicted_price' in names else z['prediction']
        result={}
        for j,h in enumerate(horizons):
            mask=z['mask'][:,j]
            if mask.any():
                result[int(h)]=dict(origin=origin[mask].copy(),current=z['origin_price'][mask].copy(),
                    target=z['target_price'][mask,j].copy(),prediction=prediction[mask,j].copy())
        return result


def training_summary(training, settings):
    return dict(settings=settings,early_stop=training['early_stop'],
                split_roles=training.get('split_roles',{'fit':'FIT','inner':'INNER'}),
                epochs=training['epochs_completed'],best_epoch=training['best_epoch'],
                best_monitored=training['best_monitored'],stop_reason=training['stop_reason'],
                initial_weights_sha256=training['initial_weights_sha256'],
                restored_weights_sha256=training['restored_weights_sha256'],
                monitor_identity={k:v for k,v in training.get('monitor_evaluation',{}).items()
                    if k in ('procedure','partition','graph','batch_size','monitor_forward','determinism_flags')})


def load_profile(path, source):
    root=path.parent; doc=read(path)
    if source['layout']=='legacy':
        if doc.get('status')!='COMPLETED' or doc.get('scientific') is not True:
            return None
        model_path=root/'model/MODEL.json'
        if not model_path.exists() or not (root/'predictions/validation.npz').exists():
            return None
        manifest=read(model_path); model=manifest['config']; hs=model['horizons_hours']
        authenticated(root,doc['artifacts'])
        training=read(root/'TRAINING.json'); settings=training.get('settings',{})
        data=dict(doc['identity']); params=manifest['components']['parameters']['total']
        scores=load_predictions(root/'predictions/validation.npz',hs)
        terminal=root.parent.parent/'TERMINAL.json'
        terminal_doc=read(terminal) if terminal.exists() else {}
        unit=terminal_doc.get('unit_id','')
        device=terminal_doc.get('attempt',{}).get('device_class')
        config=Path(source['root'])/'EFFECTIVE_CONFIG.json'
        if config.exists():
            effective=read(config)
            design=effective.get('adapter_effective',{}).get('design',{})
            data['design_contract']={k:design[k] for k in
                ('splits','business','feature_policy','evaluation_mode','retraining_frequency','sealed_identities')
                if k in design}
            data['effective_config_file_sha256']=file_sha256(config)
        label=source['label']+':'+doc['experiment_id']+(':'+unit if unit else '')
        scope='ANNUAL'; runtime=doc.get('resources',{}).get('tensorflow_version')
        weights=doc['identity']['weights_sha256']; components=manifest['components']
    else:
        if doc.get('state')!='COMPLETE':
            return None
        if doc.get('sha256')!=sha256_json({k:v for k,v in doc.items() if k!='sha256'}):
            raise ValueError('RESULT_IDENTITY_MISMATCH')
        authenticated(root,doc['files'])
        cfg=read(root/'CONFIG.json'); model=cfg['model']; hs=model['horizons_hours']
        training=read(root/'TRAINING.json'); settings=cfg.get('training',training.get('settings',{}))
        data=cfg.get('population',dict(prepared_sha256=doc.get('prepared_sha256')))
        if data.get('test_read'):
            raise ValueError('TEST_READ_RESULT_NOT_ALLOWED')
        pred=root/('VALIDATION.npz' if (root/'VALIDATION.npz').exists() else 'validation.npz')
        scores=load_predictions(pred,hs)
        cell=doc.get('cell',{}); label=source['label']+':'+cell.get('id',doc.get('batch_id','unknown'))
        scope='PERIOD' if cell.get('period') else 'ANNUAL'
        params=doc['parameters']; weights=training['restored_weights_sha256']
        runtime=doc.get('tensorflow',doc.get('resources',{}).get('tensorflow_version'))
        components={};device=doc.get('device_class')
    metrics=[score(scores[h],h) for h in sorted(scores)]
    payload=dict(label=label,source_label=source['label'],scope=scope,asset=source['asset'],
                 code_commit=source['code_commit'],result_file_sha256=file_sha256(path),
                 artifact_ref=str(path.relative_to(Path(source['root']))),
                 model=model,model_components=components,parameters=params,seed=model['seed'],
                 training=training_summary(training,settings),data=data,metrics=metrics,
                 weights_sha256=weights,tensorflow=runtime,device_class=device,
                 qualifications=source.get('qualifications',[]),test_read=False)
    payload['id']=sha256_json(dict(code_commit=source['code_commit'],result_file_sha256=file_sha256(path)))
    return dict(payload,scores=scores)


def collect(spec):
    found={};errors=[]
    for source in spec['sources']:
        root=Path(source['root']); pattern='RECEIPT.json' if source['layout']=='legacy' else 'RESULT.json'
        if not root.is_dir():
            errors.append(dict(source=source['label'],code='SOURCE_ROOT_ABSENT'))
            continue
        for path in sorted(root.rglob(pattern)):
            try:
                p=load_profile(path,source)
                if p is not None:
                    found[p['id']]=p
                    if source['layout']=='batch':
                        checkpoint=checkpoint_profile(p,path)
                        if checkpoint is not None:
                            found[checkpoint['id']]=checkpoint
            except (ValueError,KeyError,OSError,TypeError) as exc:
                errors.append(dict(source=source['label'],artifact_ref=str(path.relative_to(root)),
                                   code=type(exc).__name__+':'+str(exc)[:150]))
    return list(found.values()),errors


def checkpoint_profile(parent, result_path):
    """Retain a predeclared epoch2 snapshot without promoting it as selected."""
    root=Path(result_path).parent
    prediction=root/'EPOCH2.VALIDATION.npz';weights=root/'epoch2.weights.h5'
    if not prediction.exists() or not weights.exists():
        return None
    history=read(root/'TRAINING.json')
    rows=[r for r in history['history'] if r['epoch']==2]
    if len(rows)!=1:
        raise ValueError('DIAGNOSTIC_CHECKPOINT_HISTORY_MISMATCH')
    checkpoint=dict(epoch=2,selected_best_epoch=history['best_epoch'],parent_profile=parent['id'],
        weights_file_sha256=file_sha256(weights),prediction_file_sha256=file_sha256(prediction),
        artifact_ref='epoch2.weights.h5',selection='FIXED_EPOCH_DIAGNOSTIC_NOT_PROMOTED',
        monitored=rows[0]['monitored'])
    p={k:v for k,v in parent.items() if k not in ('id','scores')}
    p.update(scope='DIAGNOSTIC_ANNUAL',label=parent['label']+':epoch2_diagnostic',checkpoint=checkpoint,
             weights_sha256=rows[0]['weights_sha256'])
    scores=load_predictions(prediction,parent['model']['horizons_hours'])
    p['metrics']=[score(scores[h],h) for h in sorted(scores)]
    p['id']=sha256_json(dict(parent=parent['id'],checkpoint=checkpoint))
    return dict(p,scores=scores)


def diagnostic_report(profile):
    profile={k:v for k,v in profile.items() if k!='scores'}
    metrics=[]
    for row in profile['metrics']:
        for key,name,unit in (('mae','MAE','USD_per_EUR'),('naive_mae','Naive_MAE','USD_per_EUR'),
                             ('relative_error','Diagnostic_Relative_Error_Vs_Naive','ratio'),('n','N','count')):
            metrics.append(dict(metric=name,value=row[key],unit=unit,horizon=row['horizon'],split='validation'))
    return dict(experiment_key='i6dmo:checkpoint:'+profile['id'],experiment_set_key='i6dmo:checkpoint-registry',
        actor='predictor',lake='LOCAL_RETAINED',lineage='UNVERIFIED',config_sha256=profile['id'],
        code_commit=profile['code_commit'],project='predictor',phase='I6DMO_DIAGNOSTIC_CHECKPOINT',
        metrics=metrics,tags=dict(profile_identity_sha256=profile['id'],
            profile_content_sha256=sha256_json(profile),metric_scope='DIAGNOSTIC_NATIVE_SUPPORT',
            checkpoint_not_promoted='true',test_read='false',
            profile_json=json.dumps(profile,sort_keys=True,allow_nan=False)))


def cadence_profiles(spec, profiles):
    """Reuse the existing annual completeness/support contract, never pool partial years."""
    from tools.i6dmo import continuation as C
    generated=[]
    for source in spec['sources']:
        if source.get('group_by_arm') and not source.get('annual_arm'):
            path=source.get('weekly_annual')
            if not path or not Path(path).is_file():continue
            plan=read(path)
            names=sorted({c['arm'] for c in plan['cells']})
            generated.extend(cadence_profiles(dict(sources=[dict(source,annual_arm=a,
                annual_label='ANNUAL_'+a) for a in names]),profiles))
            continue
        if source.get('weekly_annual'):
            if not Path(source['weekly_annual']).is_file():
                continue
            plan=read(source['weekly_annual']); root=Path(source['root'])
            expected=[c['id'] for c in plan['cells'] if not source.get('annual_arm') or c.get('arm')==source['annual_arm']]
            if not expected:raise ValueError('WEEKLY_ANNUAL_EMPTY_ARM')
            results={p.parent.name:read(p) for p in (root/'results').glob('*/RESULT.json')}
            if any(i not in results or results[i]['state'] not in ('COMPLETE','SKIPPED_NO_EVAL_ROWS') for i in expected):
                continue
            measured={i for i in expected if results[i]['state']=='COMPLETE'}
            members=[p for p in profiles if p['source_label']==source['label'] and p['scope']=='PERIOD'
                     and p['label'].removeprefix(source['label']+':') in measured]
            if len(members)!=len(measured) or not members:
                raise ValueError('WEEKLY_ANNUAL_MEMBERS_MISSING')
            horizons=members[0]['model']['horizons_hours']
            if any(p['model']!=members[0]['model'] or set(p['scores'])-set(horizons) for p in members):
                raise ValueError('WEEKLY_ANNUAL_RECIPE_MISMATCH')
            for p in members:
                counts=p['data'].get('counts',{}).get('VALIDATION',[])
                missing=set(horizons)-set(p['scores'])
                if missing and (len(counts)!=len(horizons) or any(counts[horizons.index(h)]!=0 for h in missing)):
                    raise ValueError('WEEKLY_ANNUAL_UNDECLARED_MISSING_HEAD')
            scores={}
            for h in horizons:
                pieces=[p['scores'][h] for p in members if h in p['scores']]
                if not pieces:raise ValueError('WEEKLY_ANNUAL_HEAD_UNSUPPORTED')
                scores[h]={k:np.concatenate([a[k] for a in pieces]) for k in pieces[0]}
            payload=dict(label=source['label']+':'+source.get('annual_label','ANNUAL_short_weekly_plus18'),source_label=source['label'],
                scope='ANNUAL',asset=source['asset'],code_commit=source['code_commit'],
                model=members[0]['model'],parameters=members[0]['parameters'],seed=members[0]['seed'],
                training=dict(cadence='weekly',member_profiles=[p['id'] for p in members],individual_rule=members[0]['training']),
                data=dict(annual_period_ids=[p['data'].get('period') for p in members]),
                metrics=[score(scores[h],h) for h in sorted(scores)],weights_sha256=None,
                member_profiles=[p['id'] for p in members],qualifications=[
                    'ANNUAL_ROLLING_ENSEMBLE_OF_PERIOD_CHECKPOINTS_NOT_ONE_MODEL',
                    'DEVELOPMENT_2024_USED_FOR_CHECKPOINT_SELECTION_NOT_INDEPENDENT_OOS'],test_read=False)
            payload['id']=sha256_json(payload);generated.append(dict(payload,scores=scores))
            continue
        if not source.get('batch_spec'):
            continue
        batch=read(source['batch_spec']); batch['aggregation_root']=source['root']
        root=Path(source['root']); results={}
        for path in (root/'results').glob('*/RESULT.json'):
            d=read(path);results[d['cell']['id']]=d
        for group in FAMILIES:
            for cadence in C.CADENCES:
                rows=C.aggregate(batch,results,group,cadence)
                if rows is None:
                    continue
                members=[p for p in profiles if p['source_label']==source['label'] and p['scope']=='PERIOD'
                    and any(p['label'].endswith(c['id']) for c in batch['cells'] if c['stage']=='cadence'
                            and c['group']==group and c['cadence']==cadence)]
                if not members:
                    raise ValueError('CADENCE_MEMBERS_MISSING')
                scores={}
                for h in FAMILIES[group]:
                    pieces=[p['scores'][h] for p in members if h in p['scores']]
                    scores[h]={k:np.concatenate([a[k] for a in pieces]) for k in pieces[0]}
                    actual=score(scores[h],h); expected=next(r for r in rows if r['horizon']==h)
                    if actual['n']!=expected['n'] or abs(actual['mae']-expected['mae'])>1e-14:
                        raise ValueError('CADENCE_REPLAY_MISMATCH')
                payload=dict(label=source['label']+':ANNUAL_'+group+'_'+cadence,source_label=source['label'],
                    scope='ANNUAL',asset=source['asset'],code_commit=source['code_commit'],
                    model=members[0]['model'],parameters=members[0]['parameters'],seed=members[0]['seed'],
                    training=dict(cadence=cadence,member_profiles=[p['id'] for p in members],
                                  individual_rule=members[0]['training']),
                    data=dict(annual_period_ids=[p['data'].get('period') for p in members]),
                    metrics=[score(scores[h],h) for h in sorted(scores)],
                    weights_sha256=None,member_profiles=[p['id'] for p in members],
                    qualifications=['ANNUAL_ROLLING_ENSEMBLE_OF_PERIOD_CHECKPOINTS_NOT_ONE_MODEL'],test_read=False)
                payload['id']=sha256_json(payload);generated.append(dict(payload,scores=scores))
    return generated


def entries(board):
    out={}
    for h,r in board['horizons'].items():
        out['h'+h]={k:r[k] for k in ('winner','cohort','relative_error','horizon','n')}
        out['h'+h].update({k:r['ranking'][0][k] for k in ('mae','naive_mae')})
    for group,r in board['families'].items():
        out['family:'+group]=dict(winner=r['winner'],cohort=r['cohort'],relative_error=r['relative_error'],
                                 metrics=r['ranking'][0]['metrics'])
    return out


def read_json_tag(report, key):
    return json.loads(report['tags'][key])


def pin_reference(root, profiles, identity):
    if identity is None:
        return
    found=[p for p in profiles if p['id']==identity]
    if len(found)!=1:
        raise ValueError('REFERENCE_PROFILE_NOT_RECOVERED')
    put(Path(root)/'REFERENCE.json',{k:v for k,v in found[0].items() if k!='scores'})


def event_report(event, profile):
    current=event['current'];h=current.get('horizon')
    metrics=[dict(metric='Best_Relative_Error_Vs_Naive',value=current['relative_error'],
                  unit='ratio',horizon=h,split='validation')]
    rows=current.get('metrics', [current] if 'mae' in current else [])
    for row in rows:
        for key,name,unit in (('mae','MAE','USD_per_EUR'),('naive_mae','Naive_MAE','USD_per_EUR'),('n','N','count')):
            metrics.append(dict(metric=name,value=row[key],unit=unit,horizon=row['horizon'],split='validation'))
    return dict(experiment_key='i6dmo:best:'+event['id'],experiment_set_key='i6dmo:best-registry',
        actor='predictor',lake='LOCAL_RETAINED',lineage='UNVERIFIED',
        config_sha256=profile['id'],code_commit=profile['code_commit'],project='predictor',phase='I6DMO_BEST_REGISTRY',
        metrics=metrics,tags=dict(profile_identity_sha256=profile['id'],
            profile_content_sha256=sha256_json(profile),cohort_sha256=current['cohort'],event=event['kind'],
            record=event['id'],test_read='false',metric_scope='ANNUAL_COMMON_SUPPORT',
            profile_metrics_scope='ORIGINAL_NATIVE_POPULATION',
            profile_json=json.dumps(profile,sort_keys=True,allow_nan=False),
            event_json=json.dumps(event,sort_keys=True,allow_nan=False)))


def readiness(errors, publication, mirrors):
    if publication is None:
        return 'COMPUTED_NOT_PUBLISHED'
    if publication['mismatch'] or publication.get('provenance',{}).get('mismatch'):
        return 'PUBLICATION_MISMATCH'
    if publication['pending'] or publication.get('provenance',{}).get('pending'):
        return 'WAIT_PUBLICATION'
    if mirrors is None or mirrors['state']!='VERIFIED':
        return 'WAIT_MIRRORS'
    return 'READY_WITH_QUARANTINED_RECORDS' if errors else 'READY'


def verify_provenance(root, box, client):
    path=Path(root)/'PROVENANCE_READBACK.json';cache=read(path) if path.exists() else {}
    verified=0;mismatch=0;pending=0
    for item in box.items():
        if item['state']!='DELIVERED':
            pending+=1;continue
        digest=item['report_sha256'];tags=json.loads(item['body'])['tags'];expected=sha256_json(tags)
        if cache.get(digest)==expected:
            verified+=1;continue
        try:
            sql="SELECT tags_json FROM gov_report WHERE report_sha256='"+digest+"' LIMIT 2"
            rows=client._call('GET','/api/v1/query?'+urllib.parse.urlencode({'sql':sql}))['rows']
            if len(rows)!=1 or json.loads(rows[0]['tags_json'])!=tags:
                mismatch+=1;continue
            cache[digest]=expected;verified+=1
        except Exception:
            pending+=1
    put(path,cache)
    return dict(verified=verified,mismatch=mismatch,pending=pending)


def sync_registry(root, spec):
    from tools.i6dmo.mirror import _sync_local, _sync_remote
    root=Path(root)
    paths=[root/name for name in ('BEST.json','LEADERBOARD.json','LEADERBOARD.md','REFERENCE.json','PROVENANCE_READBACK.json','DIAGNOSTICS.json')
           if (root/name).exists()]
    for directory in ('profiles','events'):
        paths.extend(sorted((root/directory).glob('*.json')))
    files={str(p.relative_to(root)):file_sha256(p) for p in paths}
    targets={};mirrors=spec['mirrors']
    for role in ('local','remote'):
        try:
            if role=='remote' and mirrors.get('ssh_alias'):
                (root/'backup').mkdir(exist_ok=True)
                bad=_sync_remote(root,mirrors['ssh_alias'],mirrors[role],files,subprocess.run)
            else:
                bad=_sync_local(root,mirrors[role],files)
            targets[role]=dict(state='VERIFIED' if not bad else 'MISMATCH',bad=bad[:20])
        except (OSError,subprocess.SubprocessError):
            targets[role]=dict(state='UNAVAILABLE')
    proof=dict(state='VERIFIED' if all(t['state']=='VERIFIED' for t in targets.values()) else 'MISMATCH',
               manifest_sha256=sha256_json(files),files=len(files),targets=targets,at=time.time())
    put(root/'MIRRORS.json',proof)
    return proof


def render(profiles, board):
    by_id={p['id']:p for p in profiles}
    lines=['# Best Forecast Registry','','DEVELOPMENT only; TEST closed. Negative relative error is better.',
           'Annual common-support ranking, not a mosaic of separate models.','',
           '| Horizon | Best profile on common rows | MAE | Naive for horizon | Relative error | N |',
           '|---|---|---:|---:|---:|---:|']
    for h,r in board['horizons'].items():
        top=r['ranking'][0]
        lines.append(f"| {h}h | {by_id[r['winner']]['label']} | {top['mae']:.12f} | {top['naive_mae']:.12f} | {100*top['relative_error']:+.6f}% | {top['n']} |")
    lines+=['','## Complete Recipes','', '| Family | Profile | Mean relative error | Naive-winning heads |',
            '|---|---|---:|---:|']
    for group,r in board['families'].items():
        lines.append(f"| {group} | {by_id[r['winner']]['label']} | {100*r['relative_error']:+.6f}% | {r['ranking'][0]['wins']}/6 |")
    lines+=['','## Native Annual Short Results','',
            'Own original populations: descriptive records, not a controlled cross-population ranking.','',
            '| Profile | Mean relative error | Naive-winning heads | Profile identity |','|---|---:|---:|---|']
    native=[]
    for p in profiles:
        rows=[r for r in p['metrics'] if r['horizon'] in FAMILIES['short']]
        if p['scope']=='ANNUAL' and len(rows)==6:
            native.append((math.fsum(r['relative_error'] for r in rows)/6,p,rows))
    for value,p,rows in sorted(native,key=lambda x:(x[0],x[1]['id'])):
        lines.append(f"| {p['label']} | {100*value:+.6f}% | {sum(r['relative_error']<0 for r in rows)}/6 | {p['id']} |")
    lines+=['','Full model, data, monitor, seed, epochs and checkpoint hashes: profiles/<identity>.json.',
            'Partial-period results are retained but cannot become annual champions.']
    diagnostic=[p for p in profiles if p['scope']=='DIAGNOSTIC_ANNUAL']
    if diagnostic:
        lines+=['','## Fixed-Epoch Diagnostics (Not Selected)','',
                'Own native populations; retained observations, not promoted champions or TEST results.','',
                '| Profile | Horizon | Relative error | N | Checkpoint profile |','|---|---:|---:|---:|---|']
        for p in diagnostic:
            for r in p['metrics']:
                lines.append(f"| {p['label']} | {r['horizon']}h | {100*r['relative_error']:+.6f}% | {r['n']} | {p['id']} |")
    return '\n'.join(lines)+'\n'


def tick(spec):
    root=Path(spec['root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'registry.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        if spec.get('registry_code_sha256') and file_sha256(Path(__file__))!=spec['registry_code_sha256']:
            raise ValueError('REGISTRY_CODE_IDENTITY_MISMATCH')
        profiles,errors=collect(spec);profiles+=cadence_profiles(spec,profiles)
        pin_reference(root,profiles,spec.get('reference_profile_id'))
        if len({p['asset'] for p in profiles})!=1:
            raise ValueError('REGISTRY_REQUIRES_ONE_ASSET')
        board=common_board(profiles)
        for p in profiles:
            put(root/'profiles'/(p['id']+'.json'),{k:v for k,v in p.items() if k!='scores'})
        diagnostics=[{k:v for k,v in p.items() if k!='scores'} for p in profiles if p['scope']=='DIAGNOSTIC_ANNUAL']
        put(root/'DIAGNOSTICS.json',diagnostics)
        changes=record_events(root,entries(board))
        put(root/'LEADERBOARD.json',board)
        tmp=root/'LEADERBOARD.md.tmp';tmp.write_text(render(profiles,board));tmp.replace(root/'LEADERBOARD.md')
        publication=None
        if spec.get('warehouse'):
            from tools.i6dmo.outbox import Outbox,HttpWarehouse
            box=Outbox(root/'outbox.sqlite3')
            try:
                for path in sorted((root/'events').glob('*.json')):
                    event=read(path);p=read(root/'profiles'/(event['current']['winner']+'.json'))
                    box.enqueue(event_report(event,p),source_unit='best-registry:'+event['id'])
                for p in diagnostics:
                    box.enqueue(diagnostic_report(p),source_unit='diagnostic-checkpoint:'+p['id'])
                client=HttpWarehouse(spec['warehouse']['url'],spec['warehouse']['token_env'])
                publication=box.pump(client)
                publication['provenance']=verify_provenance(root,box,client)
            finally:
                box.close()
        mirrors=sync_registry(root,spec) if spec.get('mirrors') else None
        state=dict(state=readiness(errors,publication,mirrors),profiles=len(profiles),
                   changed=len(changes),events=len(list((root/'events').glob('*.json'))),errors=errors,
                   diagnostic_checkpoints=len(diagnostics),
                   publication=publication,mirrors=mirrors,at=time.time(),elapsed_s=time.monotonic()-started,
                   fit_executed=False,test_read=False)
        put(root/'STATUS.json',state)
        return state


def install_timer(spec_path, unit_dir=None, run=None):
    """Install only the CPU registry sidecar. Never alters campaign/GPU units."""
    run=run or subprocess.run
    spec_path=Path(spec_path).resolve();spec=read(spec_path)
    folder=Path(unit_dir) if unit_dir else Path.home()/'.config/systemd/user'
    folder.mkdir(parents=True,exist_ok=True)
    def quote(value):
        value=str(value)
        if '\n' in value or '\r' in value:
            raise ValueError('INVALID_SYSTEMD_PATH')
        return '"'+value.replace('\\','\\\\').replace('"','\\"').replace('%','%%')+'"'
    wd=Path(__file__).resolve().parents[2]
    env=spec.get('warehouse',{}).get('environment_file')
    def directive_path(value):
        value=str(value)
        if not Path(value).is_absolute() or any(c.isspace() for c in value):
            raise ValueError('INVALID_SYSTEMD_DIRECTIVE_PATH')
        return value.replace('%','%%')
    service=('''[Unit]
Description=CPU-only authenticated forecast best-result registry
[Service]
Type=oneshot
Environment=CUDA_VISIBLE_DEVICES=
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=PYTHONNOUSERSITE=1
MemoryMax=768M
MemorySwapMax=0
CPUQuota=50%
TasksMax=32
TimeoutStartSec=600
Nice=10
'''+('EnvironmentFile='+directive_path(env)+'\n' if env else '')+
        'WorkingDirectory='+directive_path(wd)+'\n'+
        'ExecStart='+quote(sys.executable)+' -B -m tools.i6dmo.best_registry tick --spec '+quote(spec_path)+'\n')
    timer='''[Unit]
Description=Refresh forecast best results without agent polling
[Timer]
OnBootSec=60s
OnUnitInactiveSec=120s
AccuracySec=5s
Persistent=true
[Install]
WantedBy=timers.target
'''
    for suffix,body in (('service',service),('timer',timer)):
        path=folder/('i6dmo-best-registry.'+suffix)
        tmp=path.with_suffix('.tmp');tmp.write_text(body);tmp.replace(path)
    run(['systemctl','--user','daemon-reload'],check=True,timeout=30)
    run(['systemctl','--user','enable','--now','i6dmo-best-registry.timer'],check=True,timeout=30)
    return dict(state='TIMER_INSTALLED',timer='i6dmo-best-registry.timer',fit_executed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tick','status','install']);p.add_argument('--spec',required=True)
    a=p.parse_args();s=read(a.spec)
    if a.action=='status':
        state=read(Path(s['root'])/'STATUS.json');state['age_s']=time.time()-state['at']
        print(json.dumps(state,sort_keys=True));return
    if a.action=='install':
        print(json.dumps(install_timer(a.spec),sort_keys=True));return
    try:
        result=tick(s)
    except Exception as exc:
        put(Path(s['root'])/'STATUS.json',dict(state='FAILED_KEEP_PREVIOUS_CHAMPIONS',
            error=type(exc).__name__+':'+str(exc)[:300],at=time.time(),fit_executed=False,test_read=False))
        raise
    if result.get('publication'):result['publication'].pop('delivery_seconds',None)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
