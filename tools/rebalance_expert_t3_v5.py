#!/usr/bin/env python3
"""Hand off only unstarted work; preserve native fit and original receipt identities."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

import expert_package_worker as w
from t3_parallel_handoff import argv_of, owns_manager, launcher_live


def partition_pending(names, states, current, count):
    if len(names) != len(set(names)) or current not in names or states.get(current, {}).get('state') != 'RUNNING':
        raise ValueError('INVALID_SOURCE_CURSOR')
    retained, pending = [], []
    for name in names:
        rec = states.get(name)
        if name == current or (rec and rec.get('state') == 'DONE'):
            retained.append(name)
        elif rec:
            raise ValueError('TAIL_ALREADY_ENTERED:' + name)
        else:
            pending.append(name)
    if not 0 < count < len(pending):
        raise ValueError('INSUFFICIENT_UNSTARTED_TAIL')
    return dict(retained=retained, fast=pending[:-count], slow=pending[-count:])


def freeze(cfg):
    root, control = Path(cfg['source']), Path(cfg['control'])
    if (control / 'FROZEN.json').exists():
        raise ValueError('ALREADY_FROZEN')
    original = w.read(root / 'ops/CONFIG.private.json')
    w.verify_pins(original)
    status = w.read(root / 'ops/STATUS.json')
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', cfg['source_service'], '-p', 'MainPID', '--value'], text=True))
    if not owns_manager(argv_of(pid), root):
        raise ValueError('MANAGER_OWNERSHIP')
    context = dict(manager_pid=pid, manager_identity=w.process_identity(pid),
                   launcher_pid=status['launcher_pid'], current=status['current_job'],
                   source_config_sha256=w.identity(original), frozen_at=time.time())
    context['launcher_identity'] = w.process_identity(context['launcher_pid'])
    os.kill(pid, signal.SIGSTOP)
    try:
        states = w.read(root / 'ops/JOBS.json')
        parts = partition_pending(original['only_jobs'], states, context['current'], cfg['slow_count'])
        if parts != cfg['partition']:
            raise ValueError('PARTITION_CHANGED')
        for name in parts['fast'] + parts['slow']:
            if (root / 'out' / name).exists() or (root / 'ops' / (name + '.RESULT.json')).exists():
                raise ValueError('TAIL_HAS_OUTPUT')
        # A durable service condition also prevents resurrection after a reboot.
        guard = Path.home() / '.config/systemd/user' / (cfg['source_service'] + '.d')
        guard.mkdir(parents=True, exist_ok=True)
        (guard / 'handoff.conf').write_text('[Unit]\nConditionPathExists=!' + str(control / 'FROZEN.json') + '\n')
        subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
        subprocess.run(['systemctl', '--user', 'disable', '--now', original['timer_unit']], check=True)
        w.atomic(control / 'FROZEN.json', context)
    except BaseException:
        os.kill(pid, signal.SIGCONT)
        raise
    print({'frozen': True, 'preserved_native_children': w.live_children(root), 'partition': parts})


def tick(cfg):
    root, control = Path(cfg['source']), Path(cfg['control'])
    context = w.read(control / 'FROZEN.json')
    if not context:
        raise ValueError('NOT_FROZEN')
    original = w.read(root / 'ops/CONFIG.private.json')
    if w.identity(original) != context['source_config_sha256']:
        raise ValueError('SOURCE_CHANGED')
    w.verify_pins(original)
    pid = context['manager_pid']
    if not (control / 'HARVESTED.json').exists():
        if w.live_children(root) or launcher_live(context):
            print({'state': 'WAIT_NATIVE_FIT', 'unit': context['current']})
            return
        job = next(j for j in w.planned_jobs(original) if j['name'] == context['current'])
        # Incomplete output is not silently accepted or rerun under a new identity.
        w.seal_result(original, job)
        states = w.read(root / 'ops/JOBS.json')
        states[job['name']].update(state='DONE', harvested_after_native_exit=True)
        w.atomic(root / 'ops/JOBS.json', states)
        if Path(f'/proc/{pid}').exists() and w.process_identity(pid) == context['manager_identity']:
            os.kill(pid, signal.SIGTERM)
            os.kill(pid, signal.SIGCONT)
        subprocess.run(['systemctl', '--user', 'stop', cfg['source_service']], check=True)
        w.publish_status(original, states, 'RELOCATED_TAIL', partition=cfg['partition'])
        w.atomic(control / 'HARVESTED.json', {'unit': job['name'], 'partition': cfg['partition'], 'time': time.time()})
    target = w.read(Path(cfg['fast_root']) / 'ops/CONFIG.private.json')
    if target['only_jobs'] != cfg['partition']['fast'] or target['t3_manifest'] != original['t3_manifest']:
        raise ValueError('TARGET_PARTITION_OR_MANIFEST_CHANGED')
    w.verify_pins(target)
    subprocess.run(['systemctl', '--user', 'enable', '--now', target['timer_unit']], check=True)
    subprocess.run(['systemctl', '--user', 'start', cfg['fast_service']], check=True)
    w.atomic(control / 'HANDOFF.json', {'state': 'COMPLETE', 'partition': cfg['partition'], 'time': time.time()})
    subprocess.run(['systemctl', '--user', 'disable', '--now', cfg['timer']], check=True)
    print({'state': 'HANDOFF_COMPLETE'})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['freeze', 'tick'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    (freeze if a.mode == 'freeze' else tick)(w.read(a.config))
