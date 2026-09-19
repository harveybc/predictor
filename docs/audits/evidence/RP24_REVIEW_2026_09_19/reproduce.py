"""Read-only scientific audit, plus disposable fixtures. No fitting or live writes."""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rl', action='store_true')
    parser.add_argument('--panels-root', type=Path)
    parser.add_argument('--run-root', type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / 'tools'))
    import df_mod_e0_arch_verify as A
    import df_e1_tasks as T

    base = repo / 'docs/audits/evidence/d3_k5_20260917'
    names = ['RP14_ARCH_STAGE_DESIGN.json', 'RP22_ARCH_STAGE_CLOSE_V3.json',
             'RP22_ARCH_READOUT_COMPLETION_DESIGN.json', 'RP22_ARCH_RC_CLOSE.json']
    paths = [base / name for name in names]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    d, c, sd, sc = [json.loads(p.read_text()) for p in paths]
    c, sc = c.get('local', c), sc.get('local', sc)

    def combine(parent=c, child=sc, child_design=sd):
        mc, md = A.merge_successor(parent, d, child, child_design)
        return A.effects(mc, md, n_boot=0)

    good = combine()
    gamma = {a: p['FACT']['gamma_factorial']['value'] for a, p in good['per_arch'].items()}
    manual = {}
    all_units = {**c['units'], **{k: v for k, v in sc['units'].items() if v.get('role') == 'CELL'}}
    for a in d['archs']:
        ds = {}
        for r in (0, 1):
            pairs = []
            for s in d['replicates']:
                def loss(arm):
                    return all_units[f'H3__r{r}__s{s}__{a}__{arm}']['record']['mase']['validation']
                pairs.append((loss('sequence') + loss('sequence_gap') - loss('summary') - loss('summary_last')) / 2)
            ds[r] = float(np.mean(pairs))
        manual[a] = {'d0': ds[0], 'd1': ds[1], 'gamma': ds[1] - ds[0],
                     'agrees': abs(gamma[a] - (ds[1] - ds[0])) < 1e-12}

    cases = {}
    for label in ('foreign_parent', 'empty_parent_population', 'empty_successor_population', 'failed_inherited_donors', 'changed_successor_window'):
        cc, ss, dd = copy.deepcopy(c), copy.deepcopy(sc), copy.deepcopy(sd)
        if label == 'foreign_parent':
            cc['design_sha256'] = '0' * 64
        elif label == 'empty_parent_population':
            cc['population']['members'] = []
        elif label == 'empty_successor_population':
            ss['population']['members'] = []
        elif label == 'failed_inherited_donors':
            for u in ss['units'].values():
                if u.get('role') == 'INHERITED':
                    u['status'] = 'PROBLEMS'
                    u['problems'] = ['audit: donor replay failed']
        else:
            dd['window'] = 999
        try:
            eff = combine(cc, ss, dd)
            cases[label] = {'accepted': True, 'gamma_states': {a: p['FACT']['gamma_factorial']['state'] for a, p in eff['per_arch'].items()}}
        except Exception as exc:
            cases[label] = {'accepted': False, 'reason': type(exc).__name__ + ': ' + str(exc)}

    n, W, h = 5000, 60, 1
    times = pd.Series(pd.date_range('2010-01-01', periods=n, freq='min'))
    t = np.arange(n)
    columns = T.DEFAULT_ROLES['uci_235']['targets'] + T.DEFAULT_ROLES['uci_235']['features']
    df = pd.DataFrame({k: 3 + np.sin(t / (8 + i)) for i, k in enumerate(columns)})
    df['timestamp_label'] = times.dt.strftime('%d/%m/%Y %H:%M:%S')
    def contract(frame, ts=times):
        with patch.object(T, 'load_panel', return_value=(frame, ts, {'fixture': True})):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return T.family_contract('uci_235', [W], [h])
    original = contract(df)
    changed = df.copy()
    changed.loc[800, 'Global_active_power'] = np.nan
    target_missing = contract(changed)
    extra = df.copy()
    extra['unused_numeric_metadata'] = 1.0
    extra.loc[800, 'unused_numeric_metadata'] = np.nan
    extra_meta = contract(extra)
    gap_time = times.copy()
    gap_time.iloc[1000:] += pd.Timedelta(minutes=1)
    gap_frame = df.copy()
    gap_frame['timestamp_label'] = gap_time.dt.strftime('%d/%m/%Y %H:%M:%S')
    gap = contract(gap_frame, gap_time)
    def counts(doc):
        return doc['windows']['W60_h1']['usable_windows_all_targets_valid']
    spec = {'normal': counts(original), 'target_missing': counts(target_missing), 'unused_numeric_metadata_missing': counts(extra_meta),
            'gap': counts(gap), 'gap_irregular_steps': gap['time']['irregular_steps'],
            'same_declared_roles': original['roles'] == extra_meta['roles'] == target_missing['roles'],
            'one_missing_target_should_drop_one_origin_with_declared_six_features': 1,
            'actual_train_origins_lost': counts(original)['train'] - counts(target_missing)['train'],
            'gap_should_invalidate_crossing_windows_or_refuse': True,
            'welch_max_resolvable_period_days_household': min(32768, int(2075259 * .7)) * 60 / 86400,
            'reported_slower_than_days_threshold': 35}

    output = {'revision': 'de8ae34', 'factorial_recalculation': manual, 'merge_counterexamples': cases,
              'e1_counterexamples': spec, 'scope': 'saved closures and disposable fixtures; no training, warehouse or live data writes',
              'evidence_unchanged': hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if args.panels_root:
        panel = args.panels_root / T.FAMILIES['uci_321']['panel'] / 'panel.parquet'
        digest = hashlib.sha256()
        with panel.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != T.FAMILIES['uci_321']['panel_sha256']:
            raise ValueError('panel differs from reviewed contract')
        days = {}
        for day in ('2011-03-27', '2012-03-25', '2013-03-31', '2014-03-30'):
            part = pd.read_parquet(panel, filters=[('timestamp_label', '>=', day + ' 00:00:00'),
                                                  ('timestamp_label', '<=', day + ' 06:00:00')])
            ts = pd.to_datetime(part['timestamp_label'])
            zero = (part.select_dtypes(include=[np.number]) == 0).all(axis=1)
            days[day] = {'all_clients_zero_labels': ts[zero].dt.strftime('%H:%M:%S').tolist(),
                         'nominal_hour_bins_fully_zero': [hr for hr in range(6)
                             if (ts.dt.hour == hr).any() and bool(zero[ts.dt.hour == hr].all())]}
        output['physical_dst_scan'] = {'panel_digest_verified': True, 'days': days,
                                      'scope': 'existing canonical panel, bounded six-hour ranges only'}
    if args.run_root:
        count, diffs = 0, []
        for cid, unit in sc['units'].items():
            if unit.get('role') != 'CELL' or unit.get('status') != 'VERIFIED':
                continue
            with np.load(args.run_root / 'attempts' / unit['attempt'] / 'arrays.npz', allow_pickle=False) as arr:
                for split, value in unit['record']['mase'].items():
                    measured = float(np.mean(np.abs(arr[f'{split}_pred'] - arr[f'{split}_y']).mean(axis=0) / arr['denominator']))
                    diffs.append(abs(measured - value))
            count += 1
        output['physical_successor_arrays'] = {'attempts': count, 'max_mase_difference': max(diffs),
                                               'all_finite': bool(np.isfinite(diffs).all()),
                                               'scope': 'saved predictions/labels/denominators, not fresh weight replay'}
    if args.rl:
        sys.path.insert(0, str(repo / 'tests'))
        import test_e3_weekly_env as R
        with tempfile.TemporaryDirectory(prefix='rp24-rl-review-') as temp:
            env, cfg = R._env(Path(temp), R._frame(80, slope=0.0), cash=1.0)
            trace = R._run(env, [0] * 10 + [1] + [0] * 10 + [2] + [0] * 30)
            output['rl_action_2_after_long'] = {'position_states': sorted({int(x['position']) for x in trace}),
                                                'minimum_position_units': min(float(x['position_units']) for x in trace),
                                                'action_space_n': int(env.action_space.n),
                                                'scope': 'same environment/fixture/config as RP23, no training'}
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
