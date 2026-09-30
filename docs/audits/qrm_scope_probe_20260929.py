"""Independent CPU-only review probe; no launcher, service or GPU is invoked."""
import json
import subprocess
import tempfile
import types
import hashlib
import urllib.request
from pathlib import Path
from unittest.mock import patch


def main():
    source = subprocess.check_output(
        ['git', 'show', 'c1033dc6:tools/df_cell_scope.py'], text=True)
    module = types.ModuleType('review_scope')
    module.__file__ = '/tmp/review_scope.py'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)

    class Watcher:
        identity = None
        last_peak = None
        samples = 0
        def __init__(self, *args):
            self.stop = types.SimpleNamespace(set=lambda: None)
        def start(self):
            pass
        def join(self, **kwargs):
            pass

    class Child:
        def __init__(self, *args, **kwargs):
            pass
        def wait(self):
            return 0

    good = {'cell_id': 'requested', 'stage': 'TRAIN', 'recorded_at': 1,
            'scope': {'inode': 123, 'cgroup': 'old.scope'},
            'host_ram': {'cgroup_peak': {'status': 'MEASURED', 'bytes': 123456}},
            'usable_for_costing': True}
    cases = {'production_nested_record': {'cell_scope': good},
             'stale_foreign_record': dict(good, cell_id='another-cell'),
             'negative_peak_record': dict(good, host_ram={
                 'cgroup_peak': {'status': 'MEASURED', 'bytes': -1}})}
    results = {}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, record in cases.items():
            path = root / (name + '.json')
            path.write_text(json.dumps(record))
            with patch.object(module, '_ScopeWatcher', Watcher), \
                 patch.object(module, '_slice_cgroup', return_value=None), \
                 patch.object(module, '_find_lease', return_value=(None, 'not observed')), \
                 patch.object(module.subprocess, 'Popen', Child):
                result = module.supervise(cell_id='requested', argv=['unused'],
                    launcher='/bin/true', cap_bytes=1048576, wall_seconds=5,
                    supervisor_dir=root / 'supervisor', log_path=root / 'log',
                    record_path=path, stage='TRAIN')
            results[name] = {k: result[k] for k in
                ('usable_for_costing', 'lease_confirmed', 'host_ram')}
    compact = {name: {'usable_for_costing': row['usable_for_costing'],
                     'lease_confirmed': row['lease_confirmed'],
                     'peak': row['host_ram']['cgroup_peak']}
               for name, row in results.items()}
    raw = subprocess.check_output(['git', 'show',
        '19a37baf:docs/audits/evidence/cb03_20260929/native_published400.json'])
    native = json.loads(raw)
    rows = native['per_row']
    assert len(rows) == 400 and len({r['i'] for r in rows}) == 400
    confusion = [[sum(r['gold'] == i and r['predicted'] == j for r in rows)
                  for j in range(4)] for i in range(4)]
    correct = sum(confusion[i][i] for i in range(4))
    f1 = sum(2 * confusion[i][i] /
             (sum(confusion[i]) + sum(row[i] for row in confusion))
             for i in range(4)) / 4
    url = ('https://raw.githubusercontent.com/NandhaKishorM/laya/'
           '010bacef/research/results/app_benchmark_results.json')
    with urllib.request.urlopen(url, timeout=20) as response:
        published_bytes = response.read()
    published = json.loads(published_bytes)['suites']['jev.ag_news']['typed-decisions']
    compact['independent_agnews_recount'] = {
        'native_sha256': hashlib.sha256(raw).hexdigest(),
        'published_source': url,
        'published_sha256': hashlib.sha256(published_bytes).hexdigest(),
        'n': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
        'macro_f1': f1, 'confusion': confusion,
        'published_accuracy': published['accuracy'],
        'published_in_training': published['in_training'],
        'accuracy_equal': correct / len(rows) == published['accuracy'],
        'macro_f1_equal_at_published_precision': round(f1, 4) == published['macro_f1'],
        'scope': 'Recount of retained outputs and source lookup, not fresh inference'}
    output = Path(__file__).with_suffix('.results.json')
    output.write_text(json.dumps(compact, indent=2) + '\n')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
