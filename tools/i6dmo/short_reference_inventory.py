"""Read-only census of retained six-head short scores; never imports ML code."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SHORT = (2, 3, 4, 5, 6, 12)
SPLITS = {'fit', 'train', 'inner', 'validation', 'development'}
FILES = {'RECEIPT.json', 'RESULT.json', 'REPORT.json', 'REPORTS.json', 'SUMMARY.json'}
SKIP = {'.git', 'data', 'inputs', 'features', 'source_bundle', 'population',
        'population-tf2210', 'data-verified', 'model', 'models', '__pycache__'}


def extract(doc, horizons):
    found = []
    def walk(value, split=None):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, key.lower() if key.lower() in SPLITS else split)
        elif isinstance(value, list):
            if split is not None and len(value) == 6 and all(isinstance(r, dict) for r in value):
                hs = [r.get('horizon', r.get('horizon_hours')) for r in value]
                if hs == [None]*6:
                    hs = list(horizons)
                if len(hs) == 6 and sorted(hs) == list(SHORT):
                    rows = []
                    for h, r in zip(hs, value):
                        if 'mae' not in r or not ('naive_mae' in r or 'naive' in r):
                            return
                        mae, naive = r['mae'], r.get('naive_mae', r.get('naive'))
                        n = r.get('n', r.get('count'))
                        if n is None or n <= 0 or not all(isinstance(v, (int, float)) and math.isfinite(v)
                                                          for v in (mae, naive)) or mae < 0 or naive <= 0:
                            raise ValueError('INVALID_SCORE')
                        relative = (mae-naive)/naive
                        if 'relative_error' in r and abs(relative-r['relative_error']) > 1e-12:
                            raise ValueError('RELATIVE_ERROR_MISMATCH')
                        rows.append(dict(horizon=h, n=n, mae=mae, naive=naive, relative_error=relative))
                    rows.sort(key=lambda r:r['horizon'])
                    found.append(dict(split=split, wins=sum(r['relative_error'] < 0 for r in rows), rows=rows))
                    return
            for child in value:
                walk(child, split)
    walk(doc)
    return found


def census(roots):
    vectors, errors, files = {}, [], 0
    for root_id, root in enumerate(roots):
        root = Path(root)
        if not root.is_dir():
            errors.append(dict(source=f'root{root_id}', reason='ROOT_ABSENT'))
            continue
        for parent, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for name in names:
                if name not in FILES:
                    continue
                path = Path(parent)/name
                source = f'root{root_id}/'+str(path.relative_to(root))
                files += 1
                try:
                    if path.stat().st_size > 10_000_000:
                        raise ValueError('OVERSIZED_METADATA_NOT_READ')
                    raw = path.read_bytes()
                    doc = json.loads(raw)
                    if not isinstance(doc, dict):
                        continue
                    horizons = []
                    model_path = path.parent/'model/MODEL.json'
                    if model_path.is_file():
                        horizons = json.loads(model_path.read_text()).get('config', {}).get('horizons_hours', [])
                    cell = doc.get('cell') if isinstance(doc.get('cell'), dict) else {}
                    identity = dict(source=source, file_sha256=hashlib.sha256(raw).hexdigest(),
                                    experiment=doc.get('experiment_id', doc.get('batch_id')),
                                    cell_id=cell.get('id'), period=cell.get('period'),
                                    scientific=doc.get('scientific'))
                    for row in extract(doc, horizons):
                        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
                        if key not in vectors:
                            vectors[key] = dict(row, references=[])
                        vectors[key]['references'].append(identity)
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    errors.append(dict(source=source, reason=type(exc).__name__+':'+str(exc)[:100]))
    values = list(vectors.values())
    return dict(schema='i6dmo.short_score_inventory.v1', metadata_files=files,
                distinct_vectors=len(values), score_vectors=values, errors=errors,
                all_six_validation_vectors=[v for v in values if v['wins']==6 and v['split'] in ('validation','development')],
                all_six_fit_vectors=sum(v['wins']==6 and v['split'] in ('fit','train') for v in values),
                fit_executed=False, test_read=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', action='append', required=True)
    p.add_argument('--out', required=True)
    a=p.parse_args()
    result=census(a.root)
    out=Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, name=tempfile.mkstemp(dir=out.parent, prefix=out.name+'.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, out)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    print(json.dumps({k:v for k,v in result.items() if k!='score_vectors'}, indent=2))


if __name__=='__main__':
    main()
