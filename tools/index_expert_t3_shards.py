#!/usr/bin/env python3
"""Index authenticated disjoint mirrors without changing scientific result identities."""
import argparse
from pathlib import Path
import subprocess

from collect_expert_package import validate_mirror
from expert_package_worker import read, atomic, digest


def collect(cfg):
    expected = {f't3_unit_{i:02d}' for i in range(1, 45, 2)}
    assigned = [n for shard in cfg['shards'] for n in shard['units']]
    if len(assigned) != len(set(assigned)) or set(assigned) != expected:
        raise ValueError('INVALID_SHARD_COVERAGE')
    rows, completed_shards = {}, []
    for shard in cfg['shards']:
        root = Path(shard['mirror'])
        if not (root / 'ops/CONFIG.private.json').exists():
            continue
        checked = validate_mirror(root)
        native = read(root / 'ops/CONFIG.private.json')
        if native['t3_manifest']['identity'] != cfg['manifest']:
            raise ValueError('MIXED_MANIFEST')
        if not set(checked['verified_jobs']) <= set(shard['units']):
            raise ValueError('UNASSIGNED_RESULT')
        replica = read(root / 'REMOTE_REPLICA_STATUS.json', {})
        if not replica.get('verified') or replica.get('count') != checked['count']:
            continue
        for name in checked['verified_jobs']:
            receipt = root / 'ops' / (name + '.RESULT.json')
            rows[name] = dict(shard=shard['role'], receipt_sha256=digest(receipt),
                              replica_root=shard['replica'], config_sha256=checked['config_sha256'])
        if set(checked['verified_jobs']) == set(shard['units']):
            completed_shards.append(shard['role'])
    return dict(done=len(rows), pending=len(expected) - len(rows), total=len(expected),
                units=rows, completed_shards=completed_shards, scientific_release=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    cfg = read(a.config)
    state = collect(cfg)
    target = Path(a.config).parent / 'COVERAGE.json'
    atomic(target, state)
    subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', str(target), cfg['delivery']],
                   check=True, timeout=60)
    if 'retained' in state['completed_shards']:
        subprocess.run(['systemctl', '--user', 'disable', '--now', cfg['retained_collector']], check=True)
    if not state['pending']:
        subprocess.run(['systemctl', '--user', 'disable', '--now', cfg['timer']], check=True)
    print({k: state[k] for k in ('done', 'pending', 'total')})
