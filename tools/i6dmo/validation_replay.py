"""Reload retained weekly VALIDATION models; no fit and no TEST access."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import time

import numpy as np

from tools.i6dmo import weekly18 as W, continuation as C, continuation_data as CD

QUALIFICATION = 'VALIDATION_RETAINED_CHECKPOINT_REPLAY_NOT_INDEPENDENT_OOS'


def check_available(available_by, origins):
    if len(origins) and datetime.fromisoformat(available_by).timestamp() > int(np.min(origins)):
        raise ValueError('DONOR_NOT_YET_AVAILABLE')


def compare(a, prediction, retained, heads):
    for key in ('origin_ts', 'origin_price', 'target_price', 'mask'):
        if not np.array_equal(a[key], retained[key], equal_nan=True):
            raise ValueError('RETAINED_POPULATION_CHANGED:' + key)
    if prediction.shape != retained['prediction'].shape:
        raise ValueError('REPLAY_SHAPE')
    rows = C.measured_metrics(a, prediction, heads)
    expected = C.measured_metrics(a, retained['prediction'], heads)
    return dict(prediction_exact=np.array_equal(prediction, retained['prediction']),
                max_abs_prediction_delta=float(np.max(np.abs(prediction-retained['prediction']))),
                metrics_exact=rows == expected, metrics=rows, expected=expected)


def pool(rows, heads):
    return [W.pool_metrics(rows, h) for h in heads]


def adopt(folder, unit, digest):
    path = Path(folder) / 'RESULT.json'
    if not path.exists():
        return None
    r = W.read(path)
    if r['unit'] != unit or r['input_digest'] != digest:
        raise ValueError('REPLAY_TERMINAL_IDENTITY')
    return r


def prepare(pair_root, sources, packet):
    pair_root, packet = Path(pair_root), Path(packet)
    manifest = W.read(pair_root/'PAIR.json')
    units, files = [], {}
    for family, info in manifest['families'].items():
        source = W.read(sources[family])
        prepared = W.prepared_document(source, source['data_root'])
        for member in sorted(info['members'], key=lambda m: m['period']['train_end']):
            original = pair_root/member['folder']
            W.authenticate(original, member['files'])
            archive = W.read(original/'ARCHIVE.json')
            if W.file_sha256(original/'forecast_model.keras') != archive['archive_sha256']:
                raise ValueError('ARCHIVE_CHANGED')
            cfg = W.read(original/'CONFIG.json')
            if cfg['model'] != info['model'] or cfg['training'] != W.B.TRAINING:
                raise ValueError('RETAINED_CONFIG_CHANGED')
            source_dir = Path(source['data_root'])/cfg['cell']['id']
            W.authenticate(source['data_root'], {name:prepared['files'][name] for name in
                (cfg['cell']['id']+'/PERIOD.json', cfg['cell']['id']+'/VALIDATION.npz')})
            record = W.read(source_dir/'PERIOD.json')
            if record['scales'] != W.read(original/'SCALES.json'):
                raise ValueError('RETAINED_SCALES_CHANGED')
            cid = family+'__'+member['period']['id']
            dest = packet/cid; dest.mkdir(parents=True, exist_ok=True)
            for name, src in [('forecast_model.keras', original/'forecast_model.keras'),
                              ('CONFIG.json', original/'CONFIG.json'), ('SCALES.json', original/'SCALES.json'),
                              ('REFERENCE.npz', original/'VALIDATION.npz'),
                              ('INPUT.npz', source_dir/'VALIDATION.npz')]:
                shutil.copy2(src, dest/name)
                files[cid+'/'+name] = W.file_sha256(dest/name)
            training = W.read(original/'TRAINING.json')
            units.append(dict(id=cid, family=family, period=member['period'], heads=info['model']['horizons_hours'],
                              weights_sha256=member['weights_sha256'], source_member=member['id'],
                              original_initial_weights_sha256=training['initial_weights_sha256'],
                              early_stop=training['early_stop']))
    doc = dict(schema='i6dmo.validation_replay.v1', pair_id=manifest['id'],
               pair_sha256=W.file_sha256(pair_root/'PAIR.json'), units=units, files=files,
               reference_metrics={f:v['metrics'] for f,v in manifest['families'].items()},
               fits_executed=0, test_read=False, qualification=QUALIFICATION)
    doc['sha256'] = W.B.digest(doc)
    W.B.put(packet/'PREPARED.json', doc)
    return dict(state='PREPARED', units=len(units), input_digest=doc['sha256'])


def summarize(doc, results):
    families = {}
    for family in doc['reference_metrics']:
        units = [u for u in doc['units'] if u['family'] == family]
        done = [results[u['id']] for u in units if u['id'] in results and results[u['id']]['state']=='COMPLETE']
        rows = [m for r in done for m in r['replay']['metrics']]
        expected = [m for r in done for m in r['replay']['expected']]
        lagged = [m for r in done for m in r.get('next_week_metrics', [])]
        lag_reference = [m for r in done if r.get('next_week_id') in results
                         for m in results[r['next_week_id']].get('replay', {}).get('expected', [])]
        families[family] = dict(complete=len(done), total=len(units),
            metrics=pool(rows, units[0]['heads']), reference=pool(expected, units[0]['heads']),
            next_week_diagnostic=pool(lagged, units[0]['heads']),
            next_week_reference=pool(lag_reference, units[0]['heads']),
            exact_predictions=all(r['replay']['prediction_exact'] for r in done),
            exact_metrics=all(r['replay']['metrics_exact'] for r in done),
            max_abs_prediction_delta=max((r['replay']['max_abs_prediction_delta'] for r in done), default=None))
    return families


def analyse(packet, output):
    packet, output = Path(packet), Path(output)
    doc = W.read(packet/'PREPARED.json'); original = W.read(output/'REPORT.json')
    results = {u['id']:adopt(output/'results'/u['id'], u['id'], original['input_digest']) for u in doc['units']}
    if any(r is None or r['state']!='COMPLETE' for r in results.values()):
        raise ValueError('INCOMPLETE_REPLAY')
    report = dict(original, families=summarize(doc, results))
    for family, info in report['families'].items():
        info['original_initial_weight_hashes'] = sorted({u['original_initial_weights_sha256']
            for u in doc['units'] if u['family']==family})
        for later, reference in zip(info['next_week_diagnostic'], info['next_week_reference'], strict=True):
            if later['n'] != reference['n'] or later['naive_mae'] != reference['naive_mae']:
                raise ValueError('NEXT_WEEK_PAIRED_NAIVE_CHANGED')
    W.B.put(output/'COMPARISON.json', report)
    return report


def publish(packet, output, outbox, url, token_env, is_pilot=False):
    from tools.i6dmo.outbox import Outbox, HttpWarehouse
    report = W.read(Path(output)/'REPORT.json') if is_pilot else analyse(packet, output)
    metrics = []
    for family, info in report['families'].items():
        for arm in (('metrics',) if is_pilot else ('metrics', 'reference', 'next_week_diagnostic', 'next_week_reference')):
            for row in info[arm]:
                for key in ('mae', 'naive_mae', 'relative_error', 'n'):
                    if row[key] is not None:
                        metrics.append(dict(metric=family+'_'+arm+'_'+key, value=row[key], horizon=row['horizon'],
                            split='validation', unit='ratio' if key=='relative_error' else 'count' if key=='n' else 'USD_per_EUR'))
        for key in ('exact_predictions', 'exact_metrics', 'max_abs_prediction_delta'):
            if key in info:
                metrics.append(dict(metric=family+'_'+key, value=float(info[key]), split='validation',
                                    unit='USD_per_EUR' if key=='max_abs_prediction_delta' else 'boolean'))
    mode = report.get('inference_construction','warm_start_two_weeks')
    body = dict(experiment_key='i6dmo:validation-replay-20261018:'+mode,
        experiment_set_key='i6dmo:validation-replay-20261018', actor='predictor', lake='LOCAL_RETAINED',
        lineage='UNVERIFIED', project='predictor', phase='VALIDATION_REPLAY',
        config_sha256=report['input_digest'], code_commit=W.run_cmd(['git','rev-parse','HEAD']),
        tags=dict(qualification=report['qualification'], test_read='false',
                  fits_executed=str(report['fits_executed']), construction=mode), metrics=metrics)
    identity_path=Path(output)/'PUBLISH_IDENTITY.json'
    if identity_path.exists():body['code_commit']=W.read(identity_path)['code_commit']
    else:W.B.put(identity_path,dict(code_commit=body['code_commit']))
    box = Outbox(outbox)
    try:
        box.enqueue(body, source_unit=mode)
        for path in sorted((Path(output)/'results').glob('*/RESULT.json')):
            row=W.read(path); facts=[]
            arms=row['metrics'] if is_pilot else dict(replay=row['replay']['metrics'],reference=row['replay']['expected'],
                next_week=row.get('next_week_metrics',[]))
            for arm,values in arms.items():
                for value in values:
                    for key in ('mae','naive_mae','relative_error','n'):
                        if value[key] is not None:facts.append(dict(metric=arm+'_'+key,value=value[key],
                            horizon=value['horizon'],split='validation',unit='ratio' if key=='relative_error' else 'count' if key=='n' else 'USD_per_EUR'))
            detail=dict(body,experiment_key=body['experiment_key']+':'+row['unit'],metrics=facts,
                        tags=dict(body['tags'],family=row['family'],unit=row['unit']))
            box.enqueue(detail,source_unit=mode+':'+row['unit'])
        result = box.pump(HttpWarehouse(url, token_env))
        W.B.put(Path(output)/'PUBLICATION.json', result)
        return result
    finally:
        box.close()


def collect(worker, remote, local, outbox, url, token_env):
    local=Path(local);local.mkdir(parents=True,exist_ok=True)
    with (Path(outbox).parent/'collect.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        states={}
        for source,dest,is_pilot in [('output','archive-reload',False),('builder','builder-reload',False),('pilot','pilot',True)]:
            folder=local/dest;folder.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-a','--checksum','--exclude=*.lock',worker+':'+remote+'/'+source+'/',str(folder)+'/'],check=True)
            if not (folder/'REPORT.json').exists():
                states[dest]=dict(state='WAIT_REPORT');continue
            result=publish(local/('pilot-packet' if is_pilot else 'packet'),folder,outbox,url,token_env,is_pilot)
            subprocess.run(['rsync','-a',str(folder/'PUBLICATION.json'),worker+':'+remote+'/'+source+'/'],check=True)
            states[dest]=result
        closed=len(states)==3 and all(s.get('pending')==0 and s.get('mismatch')==0 for s in states.values())
        W.B.put(local/'COLLECTION.json',dict(state='CLOSED' if closed else 'RUNNING',publication=states,at=time.time()))
        if closed:
            subprocess.run(['systemctl','--user','stop','i6dmo-validation-replay-20261018-collect.timer'],check=False)
        return dict(state='CLOSED' if closed else 'RUNNING',publication=states)


def prepare_pilot(replay_packet, sources, packet):
    from tools.i6dmo import prequential_test as S
    replay_packet, packet = Path(replay_packet), Path(packet)
    replay = W.read(replay_packet/'PREPARED.json'); W.authenticate(replay_packet, replay['files'])
    units, files, donors = [], {}, {}
    for family in ('short','long'):
        source = W.read(sources[family]); prepared = W.prepared_document(source, source['data_root'])
        members = [u for u in replay['units'] if u['family']==family][:3]
        first = members[0]; initial = packet/'initial'/family; initial.mkdir(parents=True, exist_ok=True)
        for name in ('forecast_model.keras','CONFIG.json','SCALES.json'):
            dst=initial/name; shutil.copy2(replay_packet/first['id']/name,dst)
            files[str(dst.relative_to(packet))]=W.file_sha256(dst)
        cfg=W.read(initial/'CONFIG.json'); scales=W.read(initial/'SCALES.json')
        donor=dict(root='data',archive=str((initial/'forecast_model.keras').relative_to(packet)),
            model=cfg['model'],scales=scales,weights_sha256=first['weights_sha256'],
            files={k:v for k,v in files.items() if k.startswith('initial/'+family+'/')},
            available_by=first['period']['validation_end'])
        donor['id']=W.B.digest(donor); donors[family]=donor
        for member in members[1:]:
            cid=member['id']; out=packet/cid; out.mkdir(parents=True,exist_ok=True)
            oldcfg=W.read(replay_packet/cid/'CONFIG.json'); prefix=oldcfg['cell']['id']+'/'
            names=[prefix+name for name in ('PERIOD.json','FIT.npz','VALIDATION.npz')]
            W.authenticate(source['data_root'],{k:prepared['files'][k] for k in names})
            original=Path(source['data_root'])/oldcfg['cell']['id']
            record=W.read(original/'PERIOD.json'); record['cell']=dict(record['cell'],id=cid,family=family)
            W.B.put(out/'PERIOD.json',record)
            shutil.copy2(original/'FIT.npz',out/'FIT.npz')
            shutil.copy2(original/'VALIDATION.npz',out/'OBSERVED.npz')
            a=CD._read(out/'OBSERVED.npz')
            np.savez_compressed(out/'PREDICT.npz',**{k:a[k] for k in S.PREDICT_KEYS})
            for path in out.iterdir():files[str(path.relative_to(packet))]=W.file_sha256(path)
            units.append(dict(id=cid,family=family,period=member['period']))
    doc=dict(units=sorted(units,key=lambda u:(u['period']['train_end'],u['family'])),files=files,
             initial=donors,training=W.B.TRAINING,qualification='VALIDATION_2024_WARM_START_TWO_WEEKS',test_read=False)
    doc['sha256']=W.B.digest(doc); W.B.put(packet/'PREPARED.json',doc)
    return dict(state='PREPARED',planned_fits=len(units),input_digest=doc['sha256'])


def pilot(packet, output):
    from tools.i6dmo import prequential_test as S
    packet,output=Path(packet),Path(output); output.mkdir(parents=True,exist_ok=True)
    doc=W.read(packet/'PREPARED.json'); W.authenticate(packet,doc['files'])
    donors=dict(doc['initial']); results={}
    with (output/'worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for unit in doc['units']:
            cid=unit['id']; folder=output/'results'/cid
            old=adopt(folder,cid,doc['sha256'])
            if old is not None:
                if old['state']!='COMPLETE':raise ValueError('PILOT_TERMINAL_FAILURE')
                results[cid]=old;donors[unit['family']]=old['next_checkpoint'];continue
            W.B.put(output/'STATUS.json',dict(state='RUNNING',current=cid,complete=len(results),total=len(doc['units']),test_read=False))
            record=W.read(packet/cid/'PERIOD.json')
            backend=S.GpuBackend(dict(gpu_data=str(packet)),output,evaluation_split='VALIDATION',qualification=doc['qualification'])
            reader=lambda name:CD._read(packet/cid/(name+'.npz'))
            result=S.execute_week(record,donors[unit['family']],folder,reader,backend,
                input_sha256=doc['files'][cid+'/PREDICT.npz'],evaluation_split='VALIDATION',qualification=doc['qualification'])
            result.update(state='COMPLETE',unit=cid,input_digest=doc['sha256'],family=unit['family'])
            W.B.put(folder/'RESULT.json',result);results[cid]=result;donors[unit['family']]=result['next_checkpoint']
        report=dict(state='COMPLETE',fits_executed=len(results),input_digest=doc['sha256'],test_read=False,
            qualification=doc['qualification'],families={f:dict(metrics=pool([m for r in results.values() if r['family']==f
                for m in r['metrics']['VALIDATION']],doc['initial'][f]['model']['horizons_hours'])) for f in ('short','long')})
        W.B.put(output/'REPORT.json',report);W.B.put(output/'STATUS.json',dict(state='COMPLETE',complete=len(results),
            total=len(doc['units']),fits_executed=len(results),eta_seconds=0,test_read=False))
        return report


def run(packet, output, rebuild=False):
    from tools.i6dmo import train as T, continuation_models as M
    from predictor_plugins.modular_temporal import weights_hash
    import importlib.metadata
    packet, output = Path(packet), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if M.tf.__version__ != '2.21.0' or importlib.metadata.version('keras') != '3.13.2':
        raise ValueError('RETAINED_RUNTIME_REQUIRED')
    if not M.tf.config.list_physical_devices('GPU'):
        raise ValueError('GPU_REQUIRED')
    doc = W.read(packet/'PREPARED.json')
    if doc['sha256'] != W.B.digest({k:v for k,v in doc.items() if k!='sha256'}):
        raise ValueError('PACKET_IDENTITY')
    W.authenticate(packet, doc['files'])
    identity = W.B.digest(dict(prepared=doc['sha256'], mode='original_builder_with_retained_tensors')) if rebuild else doc['sha256']
    results = {}
    with (output/'worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index, unit in enumerate(doc['units']):
            cid = unit['id']; folder = output/'results'/cid
            old = adopt(folder, cid, identity)
            if old is not None:
                results[cid] = old
                continue
            started = time.monotonic()
            W.B.put(output/'STATUS.json', dict(state='RUNNING', complete=len(results), total=len(doc['units']),
                    current=cid, at=time.time(), fits_executed=0, test_read=False))
            try:
                cfg = W.read(packet/cid/'CONFIG.json')
                scales = W.read(packet/cid/'SCALES.json')
                M.keras.backend.clear_session(); M.tf.keras.utils.set_random_seed(cfg['model']['seed'])
                model = M.keras.models.load_model(packet/cid/'forecast_model.keras', compile=False, safe_mode=True)
                if rebuild:
                    retained_weights = model.get_weights()
                    model = M.build(cfg['model']).forecast_model
                    model.set_weights(retained_weights)
                if weights_hash(model) != unit['weights_sha256']:
                    raise ValueError('REPLAY_WEIGHTS_CHANGED')
                def predict(a):
                    x, _, _ = T.transform(a, scales)
                    inputs = [x, a['offsets_hours'].astype('float32'), a['closure_crossed'].astype('float32')]
                    return T.to_price(T.monitor_forward(model, inputs, cfg['training']['batch_size']), a, scales)
                a = CD._read(packet/cid/'INPUT.npz')
                reference = CD._read(packet/cid/'REFERENCE.npz')
                replay = compare(a, predict(a), reference, unit['heads'])
                result = dict(state='COMPLETE', replay=replay)
                following = next((u for u in doc['units'][index+1:] if u['family']==unit['family']), None)
                if following:
                    b = CD._read(packet/following['id']/'INPUT.npz')
                    check_available(unit['period']['validation_end'], b['origin_ts'])
                    result['next_week_metrics'] = C.measured_metrics(b, predict(b), unit['heads'])
                    result['next_week_id'] = following['id']
                if weights_hash(model) != unit['weights_sha256']:
                    raise ValueError('INFERENCE_CHANGED_WEIGHTS')
                del model
            except Exception as exc:
                result = dict(state='FAILED', reason=type(exc).__name__+':'+str(exc)[:600])
            result.update(unit=cid, input_digest=identity, family=unit['family'],
                          seconds=time.monotonic()-started, fits_executed=0, test_read=False)
            W.B.put(folder/'RESULT.json', result); results[cid] = result
            elapsed = [r['seconds'] for r in results.values()]
            W.B.put(output/'STATUS.json', dict(state='RUNNING', complete=len(results), total=len(doc['units']),
                     current=cid, eta_seconds=(len(doc['units'])-len(results))*sum(elapsed)/len(elapsed),
                     failed=[k for k,r in results.items() if r['state']=='FAILED'], at=time.time(),
                     fits_executed=0, test_read=False))
        report = dict(state='COMPLETE' if all(r['state']=='COMPLETE' for r in results.values()) else 'COMPLETED_WITH_FAILURES',
                      input_digest=identity, families=summarize(doc, results),
                      inference_construction='original_builder_with_retained_tensors' if rebuild else 'keras_archive',
                      qualification=QUALIFICATION, next_week_qualification='SAVED_WEEK_N_MODEL_ON_WEEK_N_PLUS_1_DIAGNOSTIC',
                      units=len(results), fits_executed=0, test_read=False)
        W.B.put(output/'REPORT.json', report)
        W.B.put(output/'STATUS.json', dict(state=report['state'], complete=len(results), total=len(doc['units']),
                  eta_seconds=0, failed=[k for k,r in results.items() if r['state']=='FAILED'], at=time.time(),
                  fits_executed=0, test_read=False))
        return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'status', 'deploy', 'analyse', 'publish', 'prepare-pilot', 'pilot', 'collect'])
    parser.add_argument('--packet', required=True)
    parser.add_argument('--output')
    parser.add_argument('--pair-root')
    parser.add_argument('--short-spec'); parser.add_argument('--long-spec')
    parser.add_argument('--worker'); parser.add_argument('--source-unit')
    parser.add_argument('--runtime'); parser.add_argument('--python'); parser.add_argument('--launcher')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--outbox'); parser.add_argument('--warehouse-url')
    parser.add_argument('--token-env', default='WAREHOUSE_TOKEN')
    parser.add_argument('--pilot',action='store_true');parser.add_argument('--replay-packet')
    parser.add_argument('--remote-root')
    args = parser.parse_args()
    if args.action == 'prepare':
        result = prepare(args.pair_root, dict(short=args.short_spec, long=args.long_spec), args.packet)
    elif args.action == 'prepare-pilot':
        result=prepare_pilot(args.replay_packet,dict(short=args.short_spec,long=args.long_spec),args.packet)
    elif args.action == 'pilot':
        result=pilot(args.packet,args.output);result={k:v for k,v in result.items() if k!='families'}
    elif args.action == 'deploy':
        raw = subprocess.check_output(['ssh', args.worker, 'systemctl', '--user', 'show',
                                      args.source_unit, '--property=Environment', '--value'], text=True)
        allowed = ('LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'XLA_FLAGS', 'TF_NUM_INTEROP_THREADS',
                   'TF_NUM_INTRAOP_THREADS', 'PYTHONNOUSERSITE', 'PYTHONSAFEPATH')
        env = dict(item.split('=', 1) for item in shlex.split(raw) if item.split('=', 1)[0] in allowed)
        if not env.get('CUDA_VISIBLE_DEVICES') or not env.get('LD_LIBRARY_PATH'):
            raise ValueError('RETAINED_WORKER_ENV_REQUIRED')
        env.update(PYTHONPATH=args.runtime, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        suffix='-pilot' if args.pilot else '-builder' if args.rebuild else ''
        command = ['systemd-run', '--user', '--unit=i6dmo-validation-replay-20261018'+suffix,
                   '--property=WorkingDirectory='+args.runtime, '--property=MemoryMax=6G',
                   '--property=MemorySwapMax=0', '--property=RuntimeMaxSec=1900']
        command += ['--setenv='+k+'='+v for k,v in env.items()]
        command += [args.launcher, '-m', '4352M', '-t', '1800s', '-n', 'validation-replay', '--',
                    args.python, '-B', '-m', 'tools.i6dmo.validation_replay', 'pilot' if args.pilot else 'run',
                    '--packet', args.packet, '--output', args.output]
        if args.rebuild:
            command.append('--rebuild')
        subprocess.run(['ssh', args.worker, shlex.join(command)], check=True)
        result = dict(state='LAUNCH_REQUESTED', fits_executed=0, test_read=False)
    elif args.action == 'analyse':
        result = analyse(args.packet, args.output)
        result = {k:v for k,v in result.items() if k!='families'}
    elif args.action == 'publish':
        result = publish(args.packet, args.output, args.outbox, args.warehouse_url, args.token_env,args.pilot)
    elif args.action == 'collect':
        result=collect(args.worker,args.remote_root,args.output,args.outbox,args.warehouse_url,args.token_env)
    elif args.action == 'run':
        result = run(args.packet, args.output, args.rebuild)
        result = {k:v for k,v in result.items() if k!='families'}
    else:
        result = W.read(Path(args.output)/'STATUS.json')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
