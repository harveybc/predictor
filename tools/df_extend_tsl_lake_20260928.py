"""Bounded operator extension of the existing TSL lake; never rewrites ECL."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import df_sota_lake_adopt as S

HOME = Path.home()
HOST = HOME / '.local/state/crispdm-data-foundation/satoshi-sota-lake-adoption-20260922T024837Z/public-panels.host.json'
STATE = HOME / '.local/state/crispdm-data-foundation/tsl-extension-20260928'
DOWNLOADS = HOME / 'Downloads/TSL_benchmarks_20260928'
SCRIPT = Path(__file__).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save(path, doc):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(doc, indent=2) + '\n')
    os.replace(temp, path)


def extension(before, records):
    after = copy.deepcopy(before)
    settings = after['backend']['settings']
    for name in ['weather', 'traffic']:
        resource = f'thuml_tsl_{name}/{name}.csv'
        assert resource not in settings['include_globs']
        settings['include_globs'].append(resource)
        settings['untimed'].append(resource)
        settings['resource_contracts'][resource] = {
            'event_time_column': 'date', 'available_time_column': 'date',
            'timezone': 'NAIVE_WALL_CLOCK', 'time_unit': None,
            'frequency': '600s' if name == 'weather' else '3600s'}
    # Only these additions are allowed; the existing provider and resource stay intact.
    reduced = copy.deepcopy(after)
    for name in ['weather', 'traffic']:
        resource = f'thuml_tsl_{name}/{name}.csv'
        for key in ['include_globs', 'untimed']:
            reduced['backend']['settings'][key].remove(resource)
        del reduced['backend']['settings']['resource_contracts'][resource]
    assert reduced == before
    return after


def prepare():
    STATE.mkdir(exist_ok=False)
    records = json.loads((DOWNLOADS / 'INVENTORY.json').read_text())
    assert {r['dataset'] for r in records} == {'electricity', 'weather', 'traffic'}
    for r in records:
        source = Path(r['path'])
        assert sha(source) == r['sha256'] and source.stat().st_size == r['bytes']
        target = S.STORE_ROOT / f"thuml_tsl_{r['dataset']}" / source.name
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            assert sha(target) == r['sha256']
        else:
            shutil.copy2(source, target)
        assert sha(target) == r['sha256']
        r['lake_resource'] = str(target.relative_to(S.STORE_ROOT))
    shutil.copy2(HOST, STATE / 'host.before.json')
    candidate = extension(json.loads(HOST.read_text()), records)
    save(STATE / 'host.candidate.json', candidate)
    save(STATE / 'RESOURCES.json', records)
    save(STATE / 'PLAN.json', {
        'scope': 'two additive resources; existing ECL unchanged; only lake service restart',
        'availability': 'UNDECLARED; historical benchmark; whole-resource AS_IS only',
        'acceptance': ['disposable route per resource', 'matching delivered SHA256',
                       'reconciled terminals', 'warehouse content match',
                       'date-range and unauthenticated-campaign refusal', 'ECL regression'],
        'source_revision': records[0]['revision'], 'host_before_sha256': sha(HOST),
        'candidate_sha256': sha(STATE/'host.candidate.json'), 'script_sha256': sha(SCRIPT)})
    print('PREPARED', flush=True)


def binding():
    return {'candidate_sha256': sha(STATE/'host.candidate.json'),
            'script_sha256': sha(SCRIPT), 'resources_sha256': sha(STATE/'RESOURCES.json')}


def adopter():
    A = S.bind()
    resources = json.loads((STATE / 'RESOURCES.json').read_text())
    original = S.lake_entry()
    candidate = json.loads((STATE/'host.candidate.json').read_text())
    A.RESOURCES = {r['lake_resource']: f"thuml_tsl_{r['dataset']}" for r in resources}
    built = copy.deepcopy(original)
    for r in resources:
        built['declared'][r['lake_resource']] = r
    built['entry'].update(candidate['backend']['settings'])
    A.lake_entry = lambda: built

    def host_config(*, port, state_dir, token_file=None):
        result = copy.deepcopy(candidate)
        result['web_port'] = port
        result['operator_config_path'] = str(Path(state_dir)/'pending.json')
        return result

    A.lake_host_config = host_config
    return A, resources


def rehearse():
    A, _ = adopter()
    report = A.rehearse(STATE/'REHEARSAL.json')
    assert report['route_ok'], 'Disposable route failed; production untouched'
    save(STATE/'REHEARSED_BINDING.json', binding())
    print('REHEARSAL_PASSED', flush=True)


def adopt():
    A, resources = adopter()
    assert not (STATE/'ADOPTION.json').exists()
    assert json.loads((STATE/'REHEARSED_BINDING.json').read_text()) == binding()
    plan = json.loads((STATE/'PLAN.json').read_text())
    assert sha(HOST) == plan['host_before_sha256']
    gov_sha = sha(A.RUNTIME_CONFIG)
    for r in resources:
        assert sha(S.STORE_ROOT/r['lake_resource']) == r['sha256']
    candidate = json.loads((STATE/'host.candidate.json').read_text())
    before = json.loads((STATE/'host.before.json').read_text())
    assert extension(before, resources) == candidate
    report = {'started': A.now_iso(), 'routes': {}, 'adopted': False}
    mutated = False
    try:
        mutated = True
        save(HOST, candidate)
        subprocess.run(['systemctl','--user','restart',S.LAKE_HOST_UNIT],check=True,timeout=120)
        assert A._service_healthy('http://127.0.0.1:5060')
        token = A.API_KEY_FILE.read_text().strip()
        for r in resources:
            result = A.route_checks(A.GOV_URL, token, cache_dir=STATE/'cache',
                run_id=f'tsl-extension-{int(time.time())}', resource=r['lake_resource'],
                expect_sha=r['sha256'], outbox_dir=STATE/'outbox',
                cube_url='http://127.0.0.1:5057', cube_token=A._cube_token())
            report['routes'][r['dataset']] = result
            save(STATE/'ADOPTION.progress.json', report)
            assert result['route_complete'], r['dataset']
        assert sha(A.RUNTIME_CONFIG) == gov_sha
        report['data_gov_config_unchanged'] = True
        report['adopted'] = True
        report['service'] = A.service_state(S.LAKE_HOST_UNIT)
        save(S.STORE_ROOT/'WEATHER_TRAFFIC_EXTENSION_20260928.json', {
            'resources': resources, 'receipt': str(STATE/'ADOPTION.json'),
            'availability': 'UNDECLARED', 'use_class': 'BENCHMARK/PUBLIC'})
    except BaseException as exc:
        report['failure'] = type(exc).__name__ + ': ' + str(exc)
        if mutated:
            try:
                shutil.copy2(STATE/'host.before.json', HOST)
                subprocess.run(['systemctl','--user','restart',S.LAKE_HOST_UNIT],check=True,timeout=120)
                report['rollback_verified'] = sha(HOST) == sha(STATE/'host.before.json') and A._service_healthy('http://127.0.0.1:5060')
                # JSON formatting can differ; verify content as well.
                report['rollback_content_equal'] = json.loads(HOST.read_text()) == before
            except BaseException as rollback:
                report['rollback_failure'] = str(rollback)
        raise
    finally:
        report['finished'] = A.now_iso()
        save(STATE/'ADOPTION.json', report)
    print('ADOPTED: electricity, weather, traffic; all governed routes verified', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action',choices=['prepare','rehearse','adopt'])
    args = ap.parse_args()
    globals()[args.action]()
