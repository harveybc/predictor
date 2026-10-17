"""One exact historical short-recipe fit with six appended long heads."""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np

from tools.i6n_swarm.common import atomic_json, file_sha256, sha256_json

REPO = Path(__file__).resolve().parents[2]
BASE = '24f5a6ccc262c60c70425fc1af8ddd11dd476331'
SHORT = [2, 3, 4, 5, 6, 12]
LONG = [18, 24, 36, 48, 60, 72]
HORIZONS = SHORT + LONG
NAME = 'i6dmo-short-all-horizons-20261016'


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    atomic_json(Path(path), value, mode=0o600)


def command(argv, timeout=600):
    r = subprocess.run([str(x) for x in argv], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError('COMMAND_FAILED:' + str(argv[0]) + ':' + r.stderr[-800:])
    return r.stdout


def legacy_code_proof():
    original = command(['git', 'show', BASE+':tools/i6dmo/model.py'])
    current = (REPO/'tools/i6dmo/model.py').read_text()
    old, new = ast.parse(original), ast.parse(current)
    for node in new.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MARKET_FAMILIES'
                                               for t in node.targets):
            i = next(i for i, k in enumerate(node.value.keys)
                     if isinstance(k, ast.Constant) and k.value == 'short_mg2_all_horizons')
            node.value.keys.pop(i)
            node.value.values.pop(i)
        if isinstance(node, ast.FunctionDef) and node.name == '_normalize_market_config':
            for item in ast.walk(node):
                if isinstance(item, ast.comprehension) and item.ifs and isinstance(item.ifs[0], ast.BoolOp):
                    item.ifs[0] = item.ifs[0].values[0]
    if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
        raise ValueError('ORIGINAL_MODEL_CHANGED_BEYOND_HORIZON_ADMISSION')
    expected = hashlib.sha256(command(['git', 'show', BASE+':tools/i6dmo/train.py']).encode()).hexdigest()
    if file_sha256(REPO/'tools/i6dmo/train.py') != expected:
        raise ValueError('ORIGINAL_TRAINER_CHANGED')
    return dict(trainer_file_unchanged=True, builder_unchanged=True,
                admission_only='one additional horizon tuple; normalizer matches that tuple explicitly',
                base=BASE)


def extended_config(base):
    if base.get('horizons_hours') != SHORT or base.get('slots') != 13:
        raise ValueError('NOT_ORIGINAL_SHORT_MODEL')
    result = copy.deepcopy(base)
    result['horizons_hours'] = HORIZONS
    return result


def extend_arrays(old, table, bundle, split):
    origins = old['row_id']
    positions = np.searchsorted(table.origin_ts, origins)
    if np.any(positions >= len(table.origin_ts)) or not np.array_equal(table.origin_ts[positions], origins):
        raise ValueError('ORIGINAL_ORIGIN_MISSING')
    if not table.input_ok[positions].all() or np.any(table.split[positions] != split):
        raise ValueError('ORIGINAL_INPUT_OR_SPLIT_CHANGED')
    mask = table.label_ok[positions].copy()
    if not np.array_equal(mask[:, :6], old['mask']):
        raise ValueError('ORIGINAL_SHORT_MASK_CHANGED')
    target = np.full(mask.shape, np.nan)
    for j, h in enumerate(HORIZONS):
        stamps = origins[mask[:, j]] + h*3600
        ix = np.searchsorted(bundle.ts, stamps)
        if np.any(ix >= len(bundle.ts)) or not np.array_equal(bundle.ts[ix], stamps):
            raise ValueError('EXACT_TARGET_MISSING')
        target[mask[:, j], j] = bundle.price[ix]
    if not np.array_equal(target[:, :6], old['target_price'], equal_nan=True):
        raise ValueError('ORIGINAL_SHORT_TARGET_CHANGED')
    return {**old, 'mask': mask, 'target_price': target}


def verify_model(original, extended):
    a, b = original.forecast_model, extended.forecast_model
    if a.count_params() != 4882 or b.count_params() != 7480:
        raise ValueError('UNEXPECTED_PARAMETER_CHANGE')
    for layer in a.layers:
        other = b.get_layer(layer.name)
        if layer.get_config() != other.get_config():
            raise ValueError('ORIGINAL_LAYER_CONFIG_CHANGED:'+layer.name)
        for x, y in zip(layer.get_weights(), other.get_weights(), strict=True):
            if not np.array_equal(x, y):
                raise ValueError('ORIGINAL_INITIAL_WEIGHT_CHANGED:'+layer.name)
    x = np.random.default_rng(17).normal(size=(3, 13, 19)).astype('float32')
    offsets = np.tile(np.arange(-24, 1, 2, dtype='float32'), (3, 1))
    closure = np.zeros_like(offsets)
    p, q = a([x, offsets, closure], training=False), b([x, offsets, closure], training=False)
    if not all(np.array_equal(x, y) for x, y in zip(p, q[:6], strict=True)):
        raise ValueError('ORIGINAL_INITIAL_SHORT_FORWARD_CHANGED')
    return dict(parameters=int(b.count_params()), added_parameters=2598,
                original_layers='IDENTICAL_CONFIG_AND_INITIAL_WEIGHTS',
                short_initial_forward='BITWISE_IDENTICAL')


def prepare(spec):
    from tools.i6dmo import train as T, data as D, market_grid as G
    legacy_code_proof()
    root = Path(spec['data_root'])
    if (root/'PREPARED.json').exists():
        return read(root/'PREPARED.json')
    root.mkdir(parents=True, exist_ok=True)
    source = Path(spec['source_fit'])
    receipt = read(source/'RECEIPT.json')
    if receipt['status'] != 'COMPLETED' or not receipt['scientific']:
        raise ValueError('SOURCE_NOT_SCIENTIFIC_COMPLETE')
    for rel, sha in receipt['artifacts'].items():
        if file_sha256(source/rel) != sha:
            raise ValueError('SOURCE_ARTIFACT_CHANGED:'+rel)
    effective = T.effective_from_design(spec['source_design'])
    original = T.load_data_bundle(spec['source_bundle'], effective)
    if original['manifest_sha256'] != receipt['identity']['bundle_manifest_sha256']:
        raise ValueError('SOURCE_BUNDLE_CHANGED')
    manifest = original['manifest']
    for rel, pin in manifest['inputs'].items():
        if file_sha256(Path(spec['input_dir'])/rel) != pin['sha256']:
            raise ValueError('SOURCE_INPUT_CHANGED:'+rel)
    view = T._elapsed_view(effective['design'])
    view['horizons_hours'] = HORIZONS
    for key in D.SEAL_KEYS:
        view.pop(key, None)
    bundle = D.load_input(spec['input_dir'], view)
    table = G.build_market_table(bundle, view)
    originals = {}
    arrays = {}
    for split in T.SPLITS:
        originals[split], _ = T.trainable_rows(original[split])
        arrays[split] = extend_arrays(originals[split], table, bundle, split.upper())
        ix = np.searchsorted(table.origin_ts, originals[split]['row_id'])
        if not np.array_equal(bundle.x[table.slot_index[ix]], originals[split]['x']) \
                or not np.array_equal((table.slot_instants[ix]-originals[split]['row_id'][:,None])/3600,
                                      originals[split]['offsets_hours']) \
                or not np.array_equal(table.closure_crossed[ix], originals[split]['closure_crossed']):
            raise ValueError('ORIGINAL_INPUT_WINDOW_CHANGED')
    old_scales = read(source/'SCALES.json')
    if T.fit_scales(originals['fit']) != old_scales:
        raise ValueError('ORIGINAL_SCALES_NOT_REPRODUCED')
    scales = T.fit_scales(arrays['fit'])
    for key in ('input_mean', 'input_std', 'rows'):
        if scales[key] != old_scales[key]:
            raise ValueError('ORIGINAL_INPUT_SCALER_CHANGED')
    if scales['naive_mae'][:6] != old_scales['naive_mae']:
        raise ValueError('ORIGINAL_SHORT_SCALES_CHANGED')
    base_model = read(source/'model/MODEL.json')['config']
    training = read(source/'TRAINING.json')['settings']
    if training != effective['training']:
        raise ValueError('SOURCE_TRAINING_CHANGED')
    files = {}
    support = {}
    for split, a in arrays.items():
        path = root/(split+'.npz')
        np.savez(path, **a)
        os.chmod(path, 0o600)
        files[path.name] = file_sha256(path)
        with np.load(source/'predictions'/f'{split}.npz', allow_pickle=False) as prior:
            if not np.array_equal(prior['row_id'], a['row_id']):
                raise ValueError('ORIGINAL_REPORTED_ROWS_CHANGED')
        support[split] = dict(rows=len(a['row_id']), counts=a['mask'].sum(axis=0).tolist())
    config = dict(model=extended_config(base_model), training=training, seed=effective['seed'],
                  scales=scales, source_receipt_sha256=file_sha256(source/'RECEIPT.json'),
                  source_model_sha256=file_sha256(source/'model/MODEL.json'),
                  source_scales_sha256=file_sha256(source/'SCALES.json'),
                  source_validation=receipt['model_side_summary']['validation'],
                  original_model=base_model, support=support, legacy_code=legacy_code_proof(),
                  test_read=False, evaluation='DEVELOPMENT_TUNING',
                  historical_runtime_seal='DECLARED_2_18_OBSERVED_2_21_RETAINED_NOT_RESEALED')
    put(root/'CONFIG.json', config)
    files['CONFIG.json'] = file_sha256(root/'CONFIG.json')
    doc = dict(files=files, support=support, source_receipt=config['source_receipt_sha256'],
               test_read=False, original_short_arrays='UNCHANGED', original_scalers='UNCHANGED')
    doc['sha256'] = sha256_json(doc)
    put(root/'PREPARED.json', doc)
    return doc


def verified_data(spec):
    root = Path(spec['gpu_data'])
    prepared = read(root/'PREPARED.json')
    if prepared['sha256'] != spec['prepared_sha256']:
        raise ValueError('POPULATION_CHANGED')
    for rel, sha in prepared['files'].items():
        if file_sha256(root/rel) != sha:
            raise ValueError('PREPARED_FILE_CHANGED:'+rel)
    return read(root/'CONFIG.json')


def metric_rows(a, prediction):
    rows = []
    for j, h in enumerate(HORIZONS):
        m = a['mask'][:, j]
        y, p, origin = a['target_price'][m, j], prediction[m, j], a['origin_price'][m]
        if not len(y) or not np.isfinite(p).all():
            raise ValueError('INVALID_METRIC_SUPPORT')
        mae = math.fsum(np.abs(p-y).tolist())/len(y)
        naive = math.fsum(np.abs(origin-y).tolist())/len(y)
        if naive <= 0:
            raise ValueError('ZERO_NAIVE')
        rows.append(dict(horizon=h, n=len(y), mae=mae, naive_mae=naive,
                         relative_error=(mae-naive)/naive,
                         origins_sha256=sha256_json(a['row_id'][m].tolist())))
    return rows


def terminal(root, spec):
    path = Path(root)/'RESULT.json'
    if not path.exists():
        return None
    d = read(path)
    if d['batch_id'] != NAME or d['prepared_sha256'] != spec['prepared_sha256'] \
            or d['sha256'] != sha256_json({k:v for k,v in d.items() if k != 'sha256'}):
        raise ValueError('TERMINAL_IDENTITY_CHANGED')
    for rel, sha in d.get('files', {}).items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts or file_sha256(path.parent/rel) != sha:
            raise ValueError('TERMINAL_FILE_CHANGED')
    return d


def run(spec):
    root = Path(spec['gpu_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if terminal(root, spec):
            return
        if command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip():
            put(root/'PROGRESS.json', dict(state='WAIT_GPU',at=time.time()))
            return
        for rel, sha in spec['code_files'].items():
            if file_sha256(REPO/rel) != sha:
                raise ValueError('CODE_PIN_CHANGED:'+rel)
        from tools.i6dmo import train as T, model as M
        import tensorflow as tf
        import importlib.metadata
        if not tf.config.list_physical_devices('GPU'):
            raise ValueError('GPU_REQUIRED_NO_CPU_FALLBACK')
        if tf.__version__ != '2.21.0' or importlib.metadata.version('keras') != '3.13.2':
            raise ValueError('RUNTIME_PIN_MISMATCH')
        config = verified_data(spec)
        legacy_code_proof()
        if (root/'STARTED.json').exists():
            result = dict(state='FAILED_INTERRUPTED', reason='no automatic fit repetition', files={})
        else:
            put(root/'STARTED.json', dict(at=time.time(), prepared_sha256=spec['prepared_sha256']))
            try:
                arrays = {}
                for split in T.SPLITS:
                    with np.load(Path(spec['gpu_data'])/(split+'.npz'), allow_pickle=False) as z:
                        arrays[split] = {k:z[k] for k in z.files}
                tf.keras.backend.clear_session()
                original = M.build_dmo(config['original_model'])
                tf.keras.backend.clear_session()
                added = M.build_dmo(config['model'])
                proof = verify_model(original, added)
                prepared = {s:T.materialize_monitor_inputs(a, config['scales']) for s,a in arrays.items()}
                def progress(event):
                    put(root/'PROGRESS.json', dict(state='RUNNING', at=time.time(), **event))
                progress(dict(epoch=0))
                training = T.fit_dmo(added, prepared['fit'], prepared['inner'], config['training'],
                                     config['seed'], progress,
                                     ids=(arrays['fit']['row_id'], arrays['inner']['row_id']), record_device='gpu')
                added.forecast_model.save_weights(root/'model.weights.h5')
                fresh = M.build_dmo(config['model'])
                fresh.forecast_model.load_weights(root/'model.weights.h5')
                reports = {}
                for split, a in arrays.items():
                    pred = M.predict_matrix(added.forecast_model, prepared[split][0])
                    reload = M.predict_matrix(fresh.forecast_model, prepared[split][0])
                    if not np.array_equal(pred, reload):
                        raise ValueError('RELOAD_PREDICTIONS_DIFFER')
                    price = T.to_price(pred, a, config['scales'])
                    reports[split] = metric_rows(a, price)
                    np.savez_compressed(root/(split+'.npz'), prediction=price,
                                        row_id=a['row_id'], origin_price=a['origin_price'],
                                        target_price=a['target_price'], mask=a['mask'])
                    if split == 'validation':
                        common = a['mask'][:, 5] & a['mask'][:, 6]
                        shared = []
                        for j in (5, 6):
                            target = a['target_price'][common, j]
                            mae = math.fsum(np.abs(price[common, j]-target).tolist())/int(common.sum())
                            naive = math.fsum(np.abs(a['origin_price'][common]-target).tolist())/int(common.sum())
                            shared.append(dict(horizon=HORIZONS[j], n=int(common.sum()), mae=mae,
                                               naive_mae=naive, relative_error=(mae-naive)/naive))
                put(root/'TRAINING.json', training)
                put(root/'MODEL_PROOF.json', proof)
                put(root/'CONFIG.json', config)
                result = dict(state='COMPLETE', metrics=reports, common_12_18=shared,
                              parameters=proof['parameters'], epochs=training['epochs_completed'],
                              best_epoch=training['best_epoch'], fit_seconds=training['fit_seconds'],
                              seed=config['seed'], resources=T._resources('gpu'),
                              files={n:file_sha256(root/n) for n in
                                     ['TRAINING.json','MODEL_PROOF.json','CONFIG.json','model.weights.h5',
                                      'fit.npz','inner.npz','validation.npz']}, test_read=False)
            except Exception as exc:
                result = dict(state='FAILED', reason=type(exc).__name__+':'+str(exc)[:1500], files={})
        result.update(batch_id=NAME, prepared_sha256=spec['prepared_sha256'])
        result['sha256'] = sha256_json(result)
        put(root/'RESULT.json', result)
        put(root/'PROGRESS.json', dict(state=result['state'], at=time.time()))


def report(spec, result):
    rows = []
    for split, values in result.get('metrics', {}).items():
        for r in values:
            for metric, key, unit in [('MAE','mae','USD_per_EUR'),('Naive_MAE','naive_mae','USD_per_EUR'),
                                      ('Relative_Error_Vs_Naive','relative_error','ratio'),('N_Paired','n','count')]:
                rows.append(dict(metric=metric, value=r[key], unit=unit, horizon=r['horizon'], split=split))
    rows.append(dict(metric='Fit_Complete', value=int(result['state']=='COMPLETE'), unit='bool',
                     horizon=None, split='validation'))
    return dict(experiment_key='i6dmo:'+NAME, experiment_set_key='i6dmo:'+NAME,
                actor='predictor', lake='LOCAL_RETAINED', lineage='UNVERIFIED',
                config_sha256=spec['prepared_sha256'], code_commit=spec['code_commit'],
                project='predictor', phase='I6DMO_SHORT_EXACT_ALL_HORIZONS', metrics=rows,
                tags=dict(evaluation='DEVELOPMENT_TUNING', test_read='false',
                          evidence_sha256=result['sha256'], recipe='ORIGINAL_SHORT_PLUS_LONG_HEADS',
                          seed='6102026', disposition=result['state']))


def tick(spec):
    from tools.i6dmo.outbox import Outbox, HttpWarehouse
    root = Path(spec['root'])
    artifacts = Path(spec['artifacts'])
    artifacts.mkdir(parents=True, exist_ok=True)
    with (root/'tick.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/'CLOSE.json').exists():
            return read(root/'CLOSE.json')
        command(['rsync','-a','--checksum','--exclude=*.lock', spec['ssh_alias']+':'+spec['gpu_root']+'/',
                 str(artifacts)+'/'])
        result = terminal(artifacts, spec)
        state = dict(state='RUNNING', batch_id=NAME, test_read=False)
        if (artifacts/'PROGRESS.json').exists():
            state['progress'] = read(artifacts/'PROGRESS.json')
        if result:
            box = Outbox(root/'outbox.sqlite3')
            try:
                box.enqueue(report(spec, result), source_unit='all_horizons.fit')
                pub = box.pump(HttpWarehouse(spec['warehouse']['url'],spec['warehouse']['token_env']))
                state.update(state='WAIT_PUBLICATION', result_state=result['state'], publication=pub)
                if not pub['pending'] and not pub['mismatch'] and pub['delivered']==1:
                    put(artifacts/'PUBLICATION.json', pub)
                    put(artifacts/'READBACK.json', [dict(report_sha256=x['report_sha256'],state=x['state'],
                        readback_sha256=x['readback_sha256']) for x in box.items()])
                    lines = ['# Exact short recipe, all horizons', '', 'DEVELOPMENT 2024; TEST closed.', '',
                             '| Horizon | N | MAE | Naive for horizon | Relative error |',
                             '|---|---:|---:|---:|---:|']
                    for r in result.get('metrics',{}).get('validation',[]):
                        lines.append(f"| {r['horizon']} h | {r['n']} | {r['mae']:.12f} | {r['naive_mae']:.12f} | {100*r['relative_error']:+.6f}% |")
                    (artifacts/'RESULTS.md').write_text('\n'.join(lines)+'\n')
                    files = {str(p.relative_to(artifacts)):file_sha256(p) for p in artifacts.rglob('*') if p.is_file()}
                    mirror = Path(spec['local_mirror'])
                    mirror.mkdir(parents=True, exist_ok=True)
                    command(['rsync','-a','--checksum',str(artifacts)+'/',str(mirror)+'/'])
                    if any(file_sha256(mirror/n)!=h for n,h in files.items()):
                        raise ValueError('LOCAL_MIRROR_MISMATCH')
                    command(['ssh',spec['ssh_alias'],'mkdir','-p',spec['remote_mirror']])
                    command(['rsync','-a','--checksum',str(artifacts)+'/',spec['ssh_alias']+':'+spec['remote_mirror']+'/'])
                    code = ('import pathlib,hashlib,json; r=pathlib.Path('+repr(spec['remote_mirror'])+'); d='+repr(files)+'; '
                            'print(json.dumps([n for n,h in d.items() if hashlib.sha256((r/n).read_bytes()).hexdigest()!=h]))')
                    if json.loads(command(['ssh',spec['ssh_alias'],'python3','-c', __import__('shlex').quote(code)])):
                        raise ValueError('REMOTE_MIRROR_MISMATCH')
                    state.update(state='CLOSED' if result['state']=='COMPLETE' else 'CLOSED_WITH_FAILURE',
                                 result=result, mirrors=dict(local='VERIFIED',second_host='VERIFIED',files=len(files)))
                    put(root/'CLOSE.json', state)
                    command(['ssh',spec['ssh_alias'],'systemctl','--user','disable','--now',NAME+'-gpu.timer'])
                    command(['systemctl','--user','disable','--now',NAME+'-tick.timer'])
            finally:
                box.close()
        put(root/'STATUS.json', state)
        return state


def launch(spec, path):
    """Freeze, replicate and arm one GPU fit and its durable publication follower."""
    if command(['git','status','--porcelain']).strip():
        raise ValueError('SOURCE_NOT_COMMITTED')
    prepared = prepare(spec)
    root = Path(spec['root'])
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    spec = dict(spec, prepared_sha256=prepared['sha256'],
                code_commit=command(['git','rev-parse','HEAD']).strip())
    spec['code_files'] = {str(p.relative_to(REPO)):file_sha256(p)
                          for folder in ('tools/i6dmo','tools/i6n_contracts','tools/i6n_swarm',
                                         'predictor_plugins/modular_temporal')
                          for p in sorted((REPO/folder).rglob('*.py'))}
    runtime = '/var/tmp/i6dmo-all-horizons-runtime-'+spec['code_commit'][:8]
    def remote(argv):
        return command(['ssh',spec['ssh_alias'],shlex.join([str(x) for x in argv])])
    archive = root/'code.bundle'
    branch = command(['git','branch','--show-current']).strip()
    command(['git','bundle','create',str(archive),BASE+'..'+branch])
    remote(['mkdir','-p',str(root),spec['gpu_root'],spec['gpu_data']])
    command(['rsync','-a','--checksum',str(archive),spec['ssh_alias']+':'+str(archive)])
    remote_repo = spec['remote_repository']
    remote(['git','-C',remote_repo,'fetch',str(archive),'refs/heads/'+branch])
    # A dedicated detached worktree keeps both existing campaigns' code untouched.
    probe = subprocess.run(['ssh',spec['ssh_alias'],shlex.join(['test','-d',runtime])],capture_output=True)
    if probe.returncode:
        remote(['git','-C',remote_repo,'worktree','add','--detach',runtime,spec['code_commit']])
    if remote(['git','-C',runtime,'rev-parse','HEAD']).strip() != spec['code_commit']:
        raise ValueError('REMOTE_CODE_PIN_CHANGED')
    command(['rsync','-a','--checksum',spec['data_root']+'/',spec['ssh_alias']+':'+spec['gpu_data']+'/'])
    spec['runtime'] = runtime
    destination = root/'RUN.private.json'
    if destination.exists() and read(destination) != spec:
        raise ValueError('DEPLOYMENT_IDENTITY_CHANGED')
    put(destination,spec)
    command(['rsync','-a','--checksum',str(destination),spec['ssh_alias']+':'+str(destination)])
    deploy = read(spec['source_deployment'])
    role = deploy['roles']['secondary']
    for target, action in [('tick','tick'),('gpu','run')]:
        run = [deploy['launcher']['crispdm_run'],'-m','1G' if target=='tick' else '4G',
               '-t','600s' if target=='tick' else '1800s','-n',NAME+'-'+target,'--',
               deploy['roles']['coordinator']['python'] if target=='tick' else role['python'],
               '-B','-m','tools.i6dmo.all_horizons',action,'--spec',str(destination)]
        workdir = str(REPO) if target=='tick' else runtime
        environment = 'EnvironmentFile='+deploy['warehouse']['environment_file']+'\n' if target=='tick' else ''
        if target=='gpu':
            environment += ''.join('Environment='+json.dumps(k+'='+v)+'\n' for k,v in role.get('environment',{}).items())
        environment += 'Environment=CUDA_VISIBLE_DEVICES='+('' if target=='tick' else role['gpus'][0]['uuid'])+'\n'
        text = ('[Unit]\nDescription=Exact short recipe with appended horizons\nX-I6DMO-Batch='+NAME+
                '\n[Service]\nType=oneshot\nWorkingDirectory='+workdir+'\n'+environment+
                'Environment=PYTHONPATH='+workdir+'\nEnvironment=PYTHONDONTWRITEBYTECODE=1\n'
                'Environment=OPENBLAS_NUM_THREADS=1\nEnvironment=OMP_NUM_THREADS=1\nExecStart='+shlex.join(run)+
                '\nMemoryMax='+('2G' if target=='tick' else '5G')+'\nMemorySwapMax=0\nTimeoutStartSec=1900\nNice=10\n')
        timer = ('[Unit]\nDescription=Automatic exact-recipe contrast\nX-I6DMO-Batch='+NAME+
                 '\n[Timer]\nOnBootSec=10s\nOnUnitInactiveSec=30s\nAccuracySec=1s\nPersistent=true\nUnit='+
                 NAME+'-'+target+'.service\n[Install]\nWantedBy=timers.target\n')
        code = "import pathlib; d=pathlib.Path.home()/'.config/systemd/user'; d.mkdir(parents=True,exist_ok=True); "
        for suffix, body in [('service',text),('timer',timer)]:
            code += 'p=d/'+repr(NAME+'-'+target+'.'+suffix)+'; assert not p.exists() or '+repr('X-I6DMO-Batch='+NAME)+' in p.read_text(); p.write_text('+repr(body)+'); '
        if target=='gpu':
            remote(['python3','-c',code])
            remote(['systemctl','--user','daemon-reload'])
            remote(['systemctl','--user','enable','--now',NAME+'-gpu.timer'])
        else:
            command(['python3','-c',code])
            command(['systemctl','--user','daemon-reload'])
            command(['systemctl','--user','enable','--now',NAME+'-tick.timer'])
    put(root/'LAUNCH.json',dict(state='ARMED',code_commit=spec['code_commit'],
                              prepared_sha256=prepared['sha256'],test_read=False))
    return dict(state='ARMED',batch_id=NAME,prepared_sha256=prepared['sha256'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare','launch','run','tick','status'])
    p.add_argument('--spec', required=True)
    a = p.parse_args()
    spec = read(a.spec)
    if a.action == 'launch':
        print(json.dumps(launch(spec,a.spec),sort_keys=True))
    elif a.action == 'status':
        path = Path(spec['root'])/'STATUS.json'
        value = read(path) if path.exists() else dict(state='NOT_STARTED')
        if 'result' in value:
            value['metrics'] = value.pop('result').get('metrics',{}).get('validation')
        print(json.dumps(value, sort_keys=True))
    else:
        value = globals()[a.action](spec)
        if value:
            print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
