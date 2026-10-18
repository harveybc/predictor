"""Finite weekly VALIDATION contrasts, authenticated resumptions and OLAP publication."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from tools.i6dmo import weekly18 as W, continuation_data as CD, prequential_test as S
from tools.i6dmo import monitor_contrast as MC, validation_replay as R

NAME = 'i6dmo-monitor-contrast-20261018'
QUALIFICATION = 'REUSED_DEVELOPMENT_2024_FORWARD_MONITOR_CONTRAST'


def checked_doc(path):
    doc = W.read(path)
    if doc.get('sha256') != W.B.digest({k:v for k,v in doc.items() if k != 'sha256'}):
        raise ValueError('DOCUMENT_IDENTITY')
    return doc


def seal(path, body):
    doc = dict(body, sha256=W.B.digest(body))
    if Path(path).exists():
        if W.read(path) != doc:
            raise ValueError('IMMUTABLE_DOCUMENT_CHANGED')
    else:
        W.B.put(path, doc)
    return doc


def prepare(replay_packet, sources, packet):
    packet, replay_packet = Path(packet), Path(replay_packet)
    replay = checked_doc(replay_packet/'PREPARED.json')
    recipe = dict(replay=replay['sha256'], sources={k:W.file_sha256(v) for k,v in sources.items()})
    if (packet/'PREPARED.json').exists():
        doc = checked_doc(packet/'PREPARED.json')
        if doc['recipe'] != recipe:
            raise ValueError('PREPARED_RECIPE_CHANGED')
        W.authenticate(packet, doc['files'])
        return doc
    W.authenticate(replay_packet, replay['files'])
    files, initial, units, coverage = {}, {}, [], {}
    for family in ('short', 'long'):
        src = W.read(sources[family]); prepared = W.prepared_document(src, src['data_root'])
        members = [u for u in replay['units'] if u['family'] == family]
        first = members[0]
        target = packet/'initial'/family; target.mkdir(parents=True, exist_ok=True)
        for name in ('forecast_model.keras','CONFIG.json','SCALES.json'):
            shutil.copy2(replay_packet/first['id']/name, target/name)
            files[str((target/name).relative_to(packet))] = W.file_sha256(target/name)
        model = W.read(target/'CONFIG.json')['model']
        body = dict(root='data', archive='initial/'+family+'/forecast_model.keras', model=model,
                    scales=W.read(target/'SCALES.json'), weights_sha256=first['weights_sha256'],
                    available_by=first['period']['validation_end'],
                    files={k:v for k,v in files.items() if k.startswith('initial/'+family+'/')})
        initial[family] = dict(body, id=W.B.digest(body))
        coverage[family] = dict(calibration_only=[first['period']],
            available_source_weeks=len(members), forward_weeks=len(members)-1,
            excludes_source_boundary_weeks=True, train_replay='PENDING_NOT_FIT')
        for member in members[1:]:
            cid = member['id']; out = packet/cid; out.mkdir(parents=True, exist_ok=True)
            cfg = W.read(replay_packet/cid/'CONFIG.json'); prefix = cfg['cell']['id']+'/'
            names = [prefix+name for name in ('PERIOD.json','FIT.npz','VALIDATION.npz')]
            W.authenticate(src['data_root'], {k:prepared['files'][k] for k in names})
            old = Path(src['data_root'])/cfg['cell']['id']; record = W.read(old/'PERIOD.json')
            if record['model'] != model:
                raise ValueError('FAMILY_MODEL_NOT_CONSTANT')
            record['cell'] = dict(record['cell'], id=cid, family=family)
            W.B.put(out/'PERIOD.json', record)
            shutil.copy2(old/'FIT.npz', out/'FIT.npz')
            shutil.copy2(old/'VALIDATION.npz', out/'OBSERVED.npz')
            a = CD._read(out/'OBSERVED.npz')
            S.validate_update_span(member['period'], CD._read(out/'FIT.npz'), a, model['horizons_hours'])
            np.savez_compressed(out/'PREDICT.npz', **{k:a[k] for k in S.PREDICT_KEYS})
            for name in ('PERIOD.json','FIT.npz','OBSERVED.npz','PREDICT.npz'):
                files[cid+'/'+name] = W.file_sha256(out/name)
            units.append(dict(id=cid, family=family, period=member['period']))
    body = dict(schema='i6dmo.monitor_campaign.v1', recipe=recipe, initial=initial, files=files,
                units=units, arms=list(MC.ARMS), coverage=coverage, training=W.B.TRAINING,
                qualification=QUALIFICATION, test_read=False)
    return seal(packet/'PREPARED.json', body)


class Backend(S.GpuBackend):
    def __init__(self, packet, root, arm):
        super().__init__(dict(gpu_data=str(packet)), root, evaluation_split='VALIDATION', qualification=QUALIFICATION)
        self.arm = arm

    def monitor_adapter(self, fit, observed, prepared, record):
        return MC.adapter(self.arm, fit, prepared['FIT'], prepared['OBSERVED'], record['cell']['period']['train_end'])


def identity(doc, code):
    return W.B.digest(dict(packet=doc['sha256'], code=code, arms=list(MC.ARMS)))


def terminal(root, key, digest):
    folder = Path(root)/'results'/key
    if not (folder/'RESULT.json').exists():
        return None
    r = checked_doc(folder/'RESULT.json')
    if r['identity'] != digest or r['key'] != key:
        raise ValueError('TERMINAL_IDENTITY')
    W.authenticate(folder, r['files'])
    return r


def summarize(doc, root, digest, family):
    units = [u for u in doc['units'] if u['family'] == family]
    results = {arm+'__'+u['id']:r for arm in MC.ARMS for u in units
               if (r := terminal(root, arm+'__'+u['id'], digest)) is not None}
    arms = {}
    for arm in MC.ARMS:
        done = [results[arm+'__'+u['id']] for u in units if arm+'__'+u['id'] in results]
        measured = [r for r in done if r['state'] == 'COMPLETE']
        arms[arm] = dict(complete=len(measured), total=len(units),
            failed=[r['key'] for r in done if r['state']=='FAILED'],
            state='COMPLETE' if len(measured)==len(units) else 'INCOMPLETE',
            VALIDATION_FORWARD=R.pool([m for r in measured for m in r['metrics']['VALIDATION']],
                                     doc['initial'][family]['model']['horizons_hours']),
            TRAIN_REPLAY=None, train_replay_status='PENDING_NOT_FIT')
    fits = [r for r in results.values() if r['state']=='COMPLETE']
    failure = any(a['failed'] for a in arms.values())
    complete = len(fits)==len(units)*len(MC.ARMS)
    seconds = sum(r['training_seconds'] for r in fits)/len(fits) if fits else None
    return dict(state='COMPLETE' if complete else 'PARTIAL_FAILURE' if failure else 'RUNNING',
        complete=len(fits), total=len(units)*len(MC.ARMS), arms=arms, family=family,
        eta_seconds=None if failure or seconds is None else seconds*(len(units)*len(MC.ARMS)-len(fits)),
        eta_kind='MEASURED_FIT_MEAN_EXCLUDES_QUEUES', identity=digest, coverage=doc['coverage'][family],
        qualification=QUALIFICATION, test_read=False, at=time.time()), results


def initial_donor(doc, packet, root, family, arm):
    donor = doc['initial'][family]
    if not arm.endswith('_bn'):
        return donor
    folder = Path(root)/'initial'/arm
    if (folder/'DONOR.json').exists():
        retained = checked_doc(folder/'DONOR.json')
        if retained['source_id'] != donor['id']:
            raise ValueError('BN_SOURCE_CHANGED')
        W.authenticate(root, retained['donor']['files'])
        return retained['donor']
    original = Backend(packet, root, arm).load(donor)
    bundle, proof = MC.insert_bn(original.forecast_model, donor['model'])
    folder.mkdir(parents=True, exist_ok=True)
    bundle.forecast_model.save(folder/'forecast_model.keras')
    W.B.put(folder/'INSERTION.json', proof)
    from predictor_plugins.modular_temporal import weights_hash
    result = S.checkpoint_files(Path(root), folder, bundle.config, donor['scales'],
                                weights_hash(bundle.forecast_model), donor['available_by'])
    seal(folder/'DONOR.json', dict(source_id=donor['id'], donor=result))
    return result


def worker(spec, family):
    packet, root = Path(spec['packet']), Path(spec['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        W.authenticate(W.REPO, spec['code'])
        doc = checked_doc(packet/'PREPARED.json'); digest = identity(doc, spec['code'])
        state, results = summarize(doc, root, digest, family)
        units = [u for u in doc['units'] if u['family']==family]
        selected = None
        for arm in MC.ARMS:
            if arm.endswith('_bn') and any(results.get(arm[:-3]+'__'+u['id'], {}).get('state')!='COMPLETE' for u in units):
                continue
            for i, unit in enumerate(units):
                key = arm+'__'+unit['id']
                if key in results:
                    if results[key]['state']=='FAILED':
                        break
                    continue
                previous = results.get(arm+'__'+units[i-1]['id']) if i else None
                selected = (arm, unit, key, previous)
                break
            if selected:
                break
        if selected is None:
            state['state'] = 'COMPLETE' if state['complete']==state['total'] else 'BLOCKED_FAILED_ARMS'
            W.B.put(root/'STATUS.json', state)
            return state
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
        mem = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
        if apps.strip() or mem < 7*1024**3:
            state.update(state='WAIT_RESOURCES', eta_seconds=None, available_bytes=mem)
            W.B.put(root/'STATUS.json', state)
            return state
        arm, unit, key, previous = selected
        folder = root/'results'/key; folder.mkdir(parents=True, exist_ok=True)
        attempt_path = folder/'ATTEMPTS.json'
        attempts = W.read(attempt_path) if attempt_path.exists() else []
        exhausted = len(attempts)>=3
        if attempts and not exhausted and time.time()-attempts[-1]['at'] < 60:
            state.update(state='RETRY_PENDING', current=key, eta_seconds=None)
            W.B.put(root/'STATUS.json', state)
            return state
        try:
            if exhausted and not (folder/'update'/'UPDATE.json').exists():
                raise RuntimeError('INTERRUPTED_PROCESS_RETRY_EXHAUSTED')
            W.authenticate(packet, {k:v for k,v in doc['files'].items() if k.startswith(unit['id']+'/')})
            donor = previous['next_checkpoint'] if previous else initial_donor(doc, packet, root, family, arm)
            record = W.read(packet/unit['id']/'PERIOD.json')
            base = donor['model'].get('base', donor['model'])
            if record['model'] != base:
                raise ValueError('SOURCE_MODEL_CHANGED')
            record['model'] = donor['model']; record['cell'] = dict(record['cell'], id=key)
            attempts.append(dict(at=time.time(), donor_id=donor['id'], number=len(attempts)+1))
            W.B.put(attempt_path, attempts)
            state.update(current=key, state='RUNNING'); W.B.put(root/'STATUS.json', state)
            reader = lambda name: CD._read(packet/unit['id']/(name+'.npz'))
            result = S.execute_week(record, donor, folder, reader, Backend(packet, root, arm),
                input_sha256=doc['files'][unit['id']+'/PREDICT.npz'], evaluation_split='VALIDATION', qualification=QUALIFICATION)
            result.update(state='COMPLETE')
        except Exception as exc:
            result = dict(state='FAILED', reason=type(exc).__name__+':'+str(exc)[:600], test_read=False)
        files = {str(p.relative_to(folder)):W.file_sha256(p) for p in folder.rglob('*') if p.is_file() and p.name!='RESULT.json'}
        result.update(key=key, identity=digest, arm=arm, family=family, unit=unit, files=files)
        seal(folder/'RESULT.json', result)
        state, _ = summarize(doc, root, digest, family)
        W.B.put(root/'STATUS.json', state)
        return {k:v for k,v in state.items() if k!='arms'}


def publish(doc, root, report, results, outbox, url, token_env):
    from tools.i6dmo.outbox import Outbox, HttpWarehouse
    box = Outbox(outbox)
    try:
        for key,r in results.items():
            if r['state']!='COMPLETE':
                continue
            metrics = []
            for split,rows in r['metrics'].items():
                for row in rows:
                    for name in ('mae','naive_mae','relative_error','n'):
                        if row[name] is not None:
                            metrics.append(dict(metric=split+'_'+name, value=row[name], horizon=row['horizon'],
                                split='train' if split=='FIT' else 'validation',
                                unit='ratio' if name=='relative_error' else 'count' if name=='n' else 'USD_per_EUR'))
            for name in ('epochs','best_epoch','training_seconds','parameters','cgroup_peak_bytes'):
                if r.get(name) is not None:
                    metrics.append(dict(metric=name, value=r[name], split='train', unit='bytes' if name=='cgroup_peak_bytes' else 'seconds' if name=='training_seconds' else 'count'))
            body = dict(experiment_key=NAME+':publication-v2:'+key, experiment_set_key=NAME, actor='predictor', lake='LOCAL_RETAINED',
                lineage='UNVERIFIED', project='predictor', phase='MONITOR_CONTRAST_VALIDATION',
                config_sha256=report['identity'], code_commit=doc['code_commit'],
                tags=dict(arm=r['arm'],family=r['family'],qualification=QUALIFICATION,test_read='false',
                    train_reporting='FIT_ONLY_TRAIN_REPLAY_PENDING',result_sha256=r['sha256'],
                    publication_schema='2',supersedes_report_key=NAME+':'+key), metrics=metrics)
            box.enqueue(body, source_unit='publication-v2:'+key)
        publication = box.pump(HttpWarehouse(url, token_env))
        W.B.put(Path(root)/'PUBLICATION.json', publication)
        return {k:publication[k] for k in ('expected','delivered','pending','mismatch')}
    finally:
        box.close()


def collect(spec):
    root = Path(spec['root']); root.mkdir(parents=True, exist_ok=True)
    with (root/'collect.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        doc = checked_doc(Path(spec['packet'])/'PREPARED.json'); digest = identity(doc, spec['code'])
        status = {}
        for family, role in spec['roles'].items():
            local = Path(spec['artifacts'])/family; local.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(['rsync','-a','--checksum','--exclude=*.lock',role['host']+':'+role['output']+'/',str(local)+'/'],check=True,timeout=120)
                report, results = summarize(doc, local, digest, family)
                remote_status = W.read(local/'STATUS.json') if (local/'STATUS.json').exists() else {}
                if remote_status.get('state') in ('WAIT_RESOURCES','RETRY_PENDING','BLOCKED_FAILED_ARMS'):
                    report.update(state=remote_status['state'],eta_seconds=None)
                publication = publish(dict(doc,code_commit=spec['code_commit']),local,report,results,
                                      spec['outbox'],spec['warehouse']['url'],spec['warehouse']['token_env'])
                report['publication'] = publication
                W.B.put(local/'REPORT.json', report)
                status[family] = {k:v for k,v in report.items() if k!='arms'}
                if report['state']=='COMPLETE' and publication['pending']==publication['mismatch']==0:
                    subprocess.run(['ssh',role['host'],'systemctl','--user','disable','--now',NAME+'-'+family+'.timer'],check=True,timeout=30)
            except (subprocess.SubprocessError, OSError) as exc:
                status[family] = dict(state='UNAVAILABLE',eta_seconds=None,error=str(exc)[:300])
        closed = all(s.get('state')=='COMPLETE' and s['publication']['pending']==s['publication']['mismatch']==0 for s in status.values())
        result = dict(state='CLOSED' if closed else 'RUNNING',families=status,at=time.time(),train_replay='PENDING',test_read=False)
        W.B.put(root/'STATUS.json', result)
        if closed:
            subprocess.run(['systemctl','--user','disable','--now',NAME+'-collect.timer'],check=True)
        return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('prepare','worker','collect','status'))
    parser.add_argument('--spec'); parser.add_argument('--family',choices=('short','long'))
    parser.add_argument('--replay-packet'); parser.add_argument('--packet')
    parser.add_argument('--short-spec'); parser.add_argument('--long-spec')
    args = parser.parse_args()
    try:
        if args.action=='prepare':
            r = prepare(args.replay_packet,dict(short=args.short_spec,long=args.long_spec),args.packet)
            result = dict(state='PREPARED',units=len(r['units']),arms=r['arms'],sha256=r['sha256'])
        else:
            spec = W.read(args.spec)
            if args.action=='worker': result = worker(spec,args.family)
            elif args.action=='collect': result = collect(spec)
            else: result = W.read(Path(spec['root'])/'STATUS.json')
        print(json.dumps(result,sort_keys=True))
    except BlockingIOError:
        print(json.dumps(dict(state='BUSY')))


if __name__=='__main__':
    main()
