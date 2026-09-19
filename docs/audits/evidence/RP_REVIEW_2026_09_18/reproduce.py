"""Read-only review of originals; counterexamples use disposable copies, no training.

Run with the reviewed repository and its retained MOD-E0-DEV run root.
This is a diagnostic, not a scientific campaign or a governing verifier.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / 'tools'))
    import df_mod_e0_verify as V
    E = V.E
    before = inventory(args.run)
    honest = V.verify(args.run, None, None)
    source_checks = []
    for attempt in sorted((args.run / 'attempts').iterdir()):
        if not (attempt / 'cell.json').is_file():
            continue
        rec = json.loads((attempt / 'cell.json').read_text())
        gen = E.generate(rec['level'], rec['r'], rec['seed'])
        b = E.boundaries(len(gen['x']), rec['window'], rec['horizon'])
        lo, hi = b['train']
        den = []
        for k, group in enumerate(gen['params']['latent_groups']):
            period = round(gen['params']['groups'][group]['period'])
            train = gen['x'][lo:hi, k]
            den.append(np.mean(np.abs(train[period:] - train[:-period])))
        issues = []
        with np.load(attempt / 'arrays.npz') as arr:
            if not np.array_equal(den, arr['denominator']):
                issues.append('denominator')
            for part in ('train', 'validation', 'test'):
                if f'{part}_y' not in arr.files:
                    continue
                rows = np.arange(*b[part])
                for key, value in (('rows', rows), ('y', gen['x'][rows + rec['horizon']]),
                                   ('naive', gen['x'][rows]), ('oracle', gen['oracle'][rows])):
                    if not np.array_equal(arr[f'{part}_{key}'], value):
                        issues.append(f'{part}.{key}')
                for model, key in (('model', 'pred'), ('naive', 'naive'), ('oracle', 'oracle'), ('linear_window', 'linear')):
                    mae = np.abs(arr[f'{part}_{key}'] - arr[f'{part}_y']).mean(axis=0)
                    got = rec['scores'][part][model]
                    if not np.isclose(mae.mean(), got['mae_mean'], rtol=0, atol=1e-12):
                        issues.append(f'{part}.{model}.mae')
                    if not np.isclose((mae / den).mean(), got['mase_mean'], rtol=0, atol=1e-12):
                        issues.append(f'{part}.{model}.mase')
        if digest(attempt / 'weights.weights.h5') != rec['weights_sha256']:
            issues.append('weights_digest')
        source_checks.append({'cell_id': rec['cell_id'], 'issues': issues})
    result = {'scope': 'LOCAL_READ_ONLY_NO_WAREHOUSE_QUERY_NO_TRAINING',
              'honest': {k: honest[k] for k in ('all_verified', 'parent_equal', 'effects', 'bootstrap')},
              'independent_source_checks': source_checks,
              'counterexamples': {}}
    cases = result['counterexamples']
    source = args.run / 'attempts' / 'H3__r1__s1__sequence'
    with tempfile.TemporaryDirectory(prefix='rp-review-') as tmp:
        tmp = Path(tmp)
        empty = tmp / 'empty'
        (empty / 'attempts').mkdir(parents=True)
        shutil.copy2(args.run / 'REPORT.json', empty / 'REPORT.json')
        v = V.verify(empty, None, None)
        cases['empty_population'] = {k: v[k] for k in ('all_verified', 'parent_equal', 'effects')}

        for name in ('labels', 'denominator', 'mae_and_linear', 'weights', 'row_ids'):
            target = tmp / name
            shutil.copytree(source, target)
            rec = json.loads((target / 'cell.json').read_text())
            with np.load(target / 'arrays.npz') as z:
                arr = {k: z[k].copy() for k in z.files}
            if name == 'labels':
                for part in ('train', 'validation', 'test'):
                    arr[f'{part}_y'] = arr[f'{part}_pred'].copy()
            elif name == 'denominator':
                arr['denominator'] *= 100
                rec['mase_denominator'] = arr['denominator'].tolist()
            elif name == 'mae_and_linear':
                rec['scores']['validation']['model']['mae_mean'] = 999999.0
                rec['scores']['validation']['linear_window']['mase_mean'] = 999999.0
            elif name == 'weights':
                (target / 'weights.weights.h5').write_bytes(b'not model weights')
            elif name == 'row_ids':
                for part in ('train', 'validation', 'test'):
                    arr[f'{part}_rows'] += 100000
            if name in ('labels', 'denominator'):
                for part in ('train', 'validation', 'test'):
                    for model, src in (('model', 'pred'), ('naive', 'naive'), ('oracle', 'oracle'), ('linear_window', 'linear')):
                        rec['scores'][part][model] = E.mase(arr[f'{part}_{src}'], arr[f'{part}_y'], arr['denominator'])
            np.savez(target / 'arrays.npz', **arr)
            rec['arrays_sha256'] = digest(target / 'arrays.npz')
            (target / 'cell.json').write_text(json.dumps(rec))
            # Local receipts remain byte-consistent; independent accounting is untouched.
            for filename in ('result.json', 'outcome.json'):
                doc = json.loads((target / filename).read_text())
                dst = doc if filename == 'result.json' else doc['verified']
                dst['output_sha256'] = digest(target / 'cell.json')
                (target / filename).write_text(json.dumps(doc))
            v = V.verify_cell(target)
            cases[name] = {'problems': v['problems'], 'accepted': v.get('record') is not None and not v['problems'],
                           'validation_model_score': rec['scores']['validation']['model']['mase_mean']}

        partial = tmp / 'partial'
        (partial / 'attempts').mkdir(parents=True)
        shutil.copy2(args.run / 'REPORT.json', partial / 'REPORT.json')
        for seed in (1, 2, 3):
            for arm in ('sequence', 'summary'):
                name = f'H3__r1__s{seed}__{arm}'
                shutil.copytree(args.run / 'attempts' / name, partial / 'attempts' / name)
        v = V.verify(partial, None, None)
        cases['six_of_69'] = {k: v[k] for k in ('all_verified', 'parent_equal', 'effects')}
        doc = json.loads((partial / 'REPORT.json').read_text())
        key = next(iter(v['cells']))
        doc['cells'][key]['mase_validation'] = 999999.0
        (partial / 'REPORT.json').write_text(json.dumps(doc))
        v = V.verify(partial, None, None)
        cases['parent_disagreement'] = {k: v[k] for k in ('all_verified', 'parent_equal', 'effects')}

    nan_out = E.mase(np.array([[np.nan]]), np.array([[1.0]]), [1.0])
    cases['nonfinite_metric'] = {'returned_nan': bool(np.isnan(nan_out['mase_mean'])),
                               'status': nan_out['per_variable'][0]['status']}
    g = E.generate(3, 1, 1)
    result['geometry'] = {'periods': g['params']['groups'], 'window': E.WINDOW,
                          'span_cycles_B': (E.WINDOW - 1) / g['params']['groups']['B']['period']}
    # Independent formula check of the descriptor, using the same decomposition.
    x = np.sin(np.arange(240) * 2 * np.pi / 24)
    trend = np.convolve(x, np.ones(24) / 24, mode='same')
    detr = x - trend
    seasonal = np.array([detr[i::24].mean() for i in range(24)])
    rem = detr - np.tile(seasonal, 11)[:len(x)]
    result['trend_strength'] = {'implemented': E.trend_seasonal_strength(x)[0],
                                'one_minus_var_R_over_var_T_plus_R': max(0., 1 - np.var(rem) / np.var(trend + rem))}
    original_descriptor = E.trend_seasonal_strength

    def corrected_descriptor(x, period=E.P_A):
        trend = np.convolve(x, np.ones(period) / period, mode='same')
        detr = x - trend
        seasonal = np.array([detr[i::period].mean() for i in range(period)])
        rem = detr - np.tile(seasonal, len(x) // period + 1)[:len(x)]
        denominator = np.var(trend + rem)
        return [max(0., 1 - np.var(rem) / denominator) if denominator > 0 else 0.,
                original_descriptor(x, period)[1]]

    profile_changes = []
    keys = {(e['record']['level'], e['record']['r'], e['record']['seed'])
            for e in honest['cells'].values() if e.get('record')}
    for level, r, seed in sorted(keys):
        gen = E.generate(level, r, seed)
        lo, hi = E.boundaries()['train']
        x = gen['x'][lo:hi]
        old = E.average_linkage(E.profiles(x)['scaled'], 2)
        E.trend_seasonal_strength = corrected_descriptor
        try:
            new = E.average_linkage(E.profiles(x)['scaled'], 2)
        finally:
            E.trend_seasonal_strength = original_descriptor
        profile_changes.append({'level': level, 'r': r, 'seed': seed, 'old': old, 'corrected': new,
                                'same_partition': E.same_partition(old, new)})
    result['descriptor_reanalysis_no_training'] = profile_changes
    after = inventory(args.run)
    result['original_files'] = len(before)
    result['originals_unchanged'] = before == after
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
