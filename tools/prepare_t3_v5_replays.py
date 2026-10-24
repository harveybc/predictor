#!/usr/bin/env python3
"""Derive read-only replay plans from authenticated joined receipts."""
import argparse
from pathlib import Path
import expert_package_worker as w


def prepare(cfg):
    ready = w.read(cfg['ready'])
    if ready['count'] != 45 or ready['state'] != 'READY_FOR_PER_GROUP_DECISION_AUDIT':
        raise ValueError('REPLAY_REQUIRES_V5_JOIN')
    plans = {role: {'source_root': data['source'], 'units': {}, 'trading_replay': False}
             for role, data in cfg['roles'].items()}
    for role, data in cfg['roles'].items():
        if data.get('reference'):
            plans[role]['source_reference_config'] = data['reference']
    for name, row in ready['units'].items():
        root = Path(row['source_root'])
        receipt_path = root / 'ops' / (name + '.RESULT.json')
        if w.digest(receipt_path) != row['receipt_sha256']:
            raise ValueError('REPLAY_RECEIPT_CHANGED')
        receipt = w.read(receipt_path)
        n = int(name.rsplit('_', 1)[1])
        role = 'peer' if n % 2 == 0 else ('slow' if n in (1, 3, 5, 37, 39, 41, 43) else 'fast')
        source = (cfg['roles'][role]['source'] if role == 'peer'
                  else w.read(root / 'ops/CONFIG.private.json')['root'])
        plans[role]['units'][name] = {'run': str(Path(source) / 'out' / name), 'hashes': receipt['hashes']}
    plans['control'] = dict(plans['peer'], units={'t3_unit_00': plans['peer']['units']['t3_unit_00']}, trading_replay=True)
    for role, plan in plans.items():
        w.atomic(Path(cfg['out']) / (role + '.json'), plan)
    print({role: len(plan['units']) for role, plan in plans.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    prepare(w.read(p.parse_args().config))
