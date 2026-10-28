#!/usr/bin/env python3
"""Authenticate the complete v5 result set before invoking the frozen native decision."""
import argparse
from pathlib import Path
import shutil
import subprocess

import expert_package_worker as w
from expert_package_t3 import jobs
from prepare_expert_package_t3_v5 import verify, SCIENCE, PINS
from collect_expert_package import validate_mirror
from finalize_t3_combination import arm


def paired(rows):
    if set(rows) != {f't3_unit_{i:02d}' for i in range(45)}:
        raise ValueError('V5_INCOMPLETE_UNITS')
    ref = rows['t3_unit_00']
    for name, row in rows.items():
        if any(row[k] != ref[k] for k in ('support_hash', 'consumed_features', 'seeds')):
            raise ValueError('V5_PAIRING_MISMATCH:' + name)
    return ref


def verify_source(root, reference):
    root = Path(root)
    expected_code = set()
    for filename, expected in reference['pins'].items():
        relative = Path(filename).relative_to(reference['root'])
        if relative.parts[0] not in ('code', 'data', 'in', 't3'):
            continue
        if w.digest(root / relative) != expected:
            raise ValueError('V5_SOURCE_CHANGED:' + str(relative))
        if relative.parts[0] == 'code':
            expected_code.add(str(relative))
    actual = {str(p.relative_to(root)) for area in ('app', 'tools')
              for p in (root / 'code' / area).rglob('*.py')}
    if not expected_code or actual != expected_code:
        raise ValueError('V5_SOURCE_CODE_SET')


def close(cfg):
    reference_root, peer, out = map(Path, (cfg['reference'], cfg['peer'], cfg['out']))
    manifest, assignment = verify(reference_root)
    reference = w.read(reference_root / 'ops/CONFIG.private.json')
    w.verify_pins(reference)
    verify_source(peer, reference)
    peer_cfg = w.read(peer / 'CONFIG.json')
    if peer_cfg['science_commit'] != SCIENCE:
        raise ValueError('V5_PEER_CODE_IDENTITY')
    expected_code = {k.replace('<root>/', ''): v for k, v in peer_cfg['pins'].items()}
    trusted_code = {str(Path(k).relative_to(reference_root)): v for k, v in reference['pins'].items()
                    if Path(k).is_relative_to(reference_root / 'code')}
    if expected_code != trusted_code:
        raise ValueError('V5_PEER_PRODUCER_CODE')
    peer_assignment = w.read(peer / 't3/T3_UNITS_satoshi.json')
    canonical = {j['name']: j for j in jobs(manifest, assignment) + jobs(manifest, peer_assignment, worker='satoshi')}
    sources = {}
    for shard in cfg['shards']:
        root = Path(shard)
        state = validate_mirror(root)
        producer = w.read(root / 'ops/CONFIG.private.json')
        if producer['t3_manifest'] != manifest:
            raise ValueError('V5_WRONG_SHARD_MANIFEST')
        for name in state['verified_jobs']:
            if name in sources:
                raise ValueError('V5_DUPLICATE_UNIT')
            sources[name] = (root, producer, False)
    for job in jobs(manifest, peer_assignment, worker='satoshi'):
        sources[job['name']] = (peer, peer_cfg, True)
    rows = {}
    for name, (root, producer, is_peer) in sources.items():
        job = canonical[name]
        receipt_path = root / 'ops' / (name + '.RESULT.json')
        receipt = w.read(receipt_path)
        if receipt['config_sha256'] != w.identity(producer):
            raise ValueError('V5_RECEIPT_CONFIG')
        for key in ('name', 'peer_command', 'peer_command_identity', 'groups_sha256', 'dropout', 'wd'):
            if receipt['job'][key] != job[key]:
                raise ValueError('V5_RECEIPT_JOB:' + name + ':' + key)
        for filename, expected in receipt['hashes'].items():
            if Path(filename).name != filename or w.digest(root / 'out' / name / filename) != expected:
                raise ValueError('V5_OUTPUT_CHANGED:' + name)
        checked = w.validate_result({'root': str(root), 'campaign': 'v4_t3', 'seeds': reference['seeds']}, job)
        if any(receipt['hashes'].get(k) != v for k, v in checked['hashes'].items()):
            raise ValueError('V5_NATIVE_RECEIPT_HASHES')
        result = w.read(root / 'out' / name / 'E3M.json')
        arm(result)
        rows[name] = dict(support_hash=checked['support_hash'], consumed_features=checked['consumed_features'],
                          seeds=sorted(result['trials']['FULL']), result_sha256=checked['hashes']['E3M.json'],
                          receipt_sha256=w.digest(receipt_path), source_root=str(root))
    ref = paired(rows)
    out.mkdir(parents=True, exist_ok=True)
    for name, (root, _, _) in sources.items():
        destination = out / 'results' / name
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / 'out' / name / 'E3M.json', destination / 'E3M.json')
    shutil.copytree(reference_root / 't3', out / 't3', dirs_exist_ok=True)
    ready = dict(schema='t3_joined_artifact_readiness.v1', state='READY_FOR_PER_GROUP_DECISION_AUDIT',
                 count=45, units=rows, support_hash=ref['support_hash'], columns=len(ref['consumed_features']),
                 manifest_sha256=PINS['t3/T3_MANIFEST.json'], scientific_release=False)
    w.atomic(out / 'T3_READY.json', ready)
    native = reference_root / 'code/tools/expert_guided_t3_manifest.py'
    subprocess.run(['python3', str(native), 'decide', '--manifest', str(out / 't3/T3_MANIFEST.json'),
                    '--results', str(out / 'results'), '--out', str(out / 'T3_DECISION.json')], check=True)
    subprocess.run(['python3', str(native), 'combine', '--manifest', str(out / 't3/T3_MANIFEST.json'),
                    '--decision', str(out / 'T3_DECISION.json')], check=True)
    for old, new in (('E3M.json', 'T3_CONTROL.json'), ('SCALER.json', 'T3_CONTROL_SCALER.json')):
        shutil.copyfile(peer / 'out/t3_unit_00' / old, out / new)
    print({'verified_units': len(rows), 'columns': ready['columns'], 'decision_sha256': w.digest(out / 'T3_DECISION.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    close(w.read(a.config))
