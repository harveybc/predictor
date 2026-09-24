import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('custody_audit', Path(__file__).with_name('check_live_custody.py'))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


@pytest.mark.parametrize('damage', [None, 'predictions', 'checkpoint', 'record', 'design', 'missing', 'duplicate'])
def test_live_content_not_receipt_alone(tmp_path, damage):
    unit = 'cell'
    cell = {'cell_id': unit, 'horizon': 96, 'seed': 1, 'seq_len': 96, 'arm': 'reference'}
    folder = tmp_path / 'attempts' / unit
    folder.mkdir(parents=True)
    design = {'design_sha256': 'd' * 64, 'cells': [cell]}
    record = {'design_sha256': design['design_sha256'], 'cell': cell}
    contents = {'DESIGN.json': design, 'BENCH_DATA.json': {}, 'TERMINAL_RECEIPTS.json': {'units': {
        name: {'campaign_sha256': digest * 64, 'terminal_sha256': digest * 64}
        for name, digest in [('prepare', 'a'), ('cell', 'b')]}},
        'REPORT.json': {'design_sha256': design['design_sha256'], 'verification': {'rows': [{'unit': unit, 'replay': {
            'finite': True, 'shape_equal': True, 'allclose_rule': True, 'max_abs_prediction_difference': 0.0}}]}}}
    for name, obj in contents.items():
        (tmp_path / name).write_text(json.dumps(obj))
    (folder / 'cell.json').write_text(json.dumps(record))
    for path in [tmp_path / 'BENCH_DATA.npz', folder / 'arrays.npz', folder / 'checkpoint.pth']:
        path.write_bytes(b'fixture bytes')
    artifacts = {}
    for digest, files in [('a', {'data': tmp_path / 'BENCH_DATA.npz', 'record': tmp_path / 'BENCH_DATA.json'}),
                          ('b', {'predictions': folder / 'arrays.npz', 'checkpoint': folder / 'checkpoint.pth', 'record': folder / 'cell.json'})]:
        artifacts[digest] = [{'role': role, 'sha256': A.sha(path), 'bytes': path.stat().st_size} for role, path in files.items()]
    if damage in ('predictions', 'checkpoint', 'record'):
        next(x for x in artifacts['b'] if x['role'] == damage)['sha256'] = 'f' * 64
    if damage == 'duplicate':
        artifacts['b'].append(dict(artifacts['b'][0]))

    def query(sql):
        digest = 'a' if 'a' * 64 in sql else 'b'
        if 'gov_terminal_artifact' in sql:
            return artifacts[digest]
        if damage == 'missing' and digest == 'b':
            return []
        return [{'unit_id': 'prepare' if digest == 'a' else unit, 'terminal_sha256': digest * 64,
                 'generation': 1, 'status': 'COMPLETED', 'config_sha256': 'f' * 64 if damage == 'design' else design['design_sha256']}]

    if damage == 'missing':
        with pytest.raises(ValueError, match='Missing accepted terminal'):
            A.audit(tmp_path, unit, query)
    else:
        assert A.audit(tmp_path, unit, query)['live_custody_pass'] is (damage is None)
