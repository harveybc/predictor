"""Offline ML audit; no training, no production writes, no unblinding of reserves."""
import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-root', type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo / 'tools'))
    import df_mod_e0_arch_verify as A
    import df_mod_e0_close as C
    import df_mod_e0 as E

    evidence = args.repo / 'docs/audits/evidence/d3_k5_20260917'
    design_path = evidence / 'RP14_ARCH_STAGE_DESIGN.json'
    close_path = evidence / 'RP14_ARCH_STAGE_CLOSE_LOCAL.json'
    design, close = json.loads(design_path.read_text()), json.loads(close_path.read_text())
    local = close.get('local', close)
    before = {p.name: sha(p) for p in (design_path, close_path)}
    reported = A.effects(local, design)
    comparisons = {}
    for arch in design['archs']:
        ds = {}
        branches = {}
        for r in (0, 1):
            differences = []
            for seed in design['replicates']:
                prefix = f'H3__r{r}__s{seed}__{arch}__'
                def value(arm):
                    return local['units'][prefix + arm]['record']['mase']['validation']
                differences.append(value('sequence') - value('summary'))
            ds[r] = float(np.mean(differences))
            branches[r] = sorted({A._parse(k)['arm'] for k in local['units']
                                  if k.startswith(f'H3__r{r}__') and A._parse(k)['arch'] == arch
                                  and not A._parse(k)['dsum']})
        comparisons[arch] = {'reported_gamma': reported['per_arch'][arch]['H3']['gamma'],
                             'common_pair_d': ds, 'common_pair_gamma': ds[1] - ds[0],
                             'arms_by_regime': branches}

    fabricated = copy.deepcopy(local)
    for cid, unit in fabricated['units'].items():
        m = A._parse(cid)
        if m['hypothesis'] == 'H3' and unit.get('record'):
            for split in unit['record']['mase']:
                unit['record']['mase'][split] = 0.5 if m['arm'] in A.LAST_READOUTS else 1.0
    null_effect = A.effects(fabricated, design)
    no_fusion = {a: null_effect['per_arch'][a]['H3']['gamma'] for a in design['archs']}

    missing = copy.deepcopy(local)
    key = 'H3__r1__s2__A__summary'
    missing['units'][key]['status'] = 'PROBLEMS'
    missing['units'][key]['problems'] = ['review deliberately omitted arm']
    partial = A.effects(missing, design)
    mismatched = copy.deepcopy(design)
    mismatched['design_sha256'] = '0' * 64
    foreign = A.effects(local, mismatched)

    # A cached replay is returned without examining the scientific code/environment.
    # The trap subprocess proves no fresh process was even attempted.
    with tempfile.TemporaryDirectory(prefix='rp16-review-') as td:
        td = Path(td)
        attempt = td / 'attempt'
        attempt.mkdir()
        for name in ('cell.json', 'arrays.npz', 'weights.weights.h5'):
            (attempt / name).write_bytes(b'diagnostic fixture')
        (attempt / 'job.json').write_text('{}')
        inputs = {n: sha(attempt / f) for n, f in
                  (('cell', 'cell.json'), ('arrays', 'arrays.npz'), ('weights', 'weights.weights.h5'), ('job', 'job.json'))}
        out = td / 'replays'
        out.mkdir()
        (out / 'attempt.json').write_text(json.dumps({'inputs': inputs, 'producer_code_identity': 'DIFFERENT_CODE',
                                                     'schema': 'df_mod_e0_replay.v1', 'problems': []}))
        with patch.object(C.subprocess, 'run', side_effect=AssertionError('fresh replay attempted')):
            reuse = C.run_replays([attempt], out, workers=1)
        cache = {'notes': reuse['notes'], 'cpu_seconds_children': reuse['cpu_seconds_children'],
                 'returned_code': reuse['docs']['attempt']['producer_code_identity']}

    adequacy = {}
    for arch, v in reported['adequacy'].items():
        adequacy[arch] = [{'cell': c['cell'], 'gap_to_linear': c['model'] - c['linear'],
                           'within_previous_0_03_bar': c['model'] <= c['linear'] + .03}
                          for c in v['cells']]
    import df_mod_e0_metrics as M
    g = E.generate(2, 1, 1, diagnostic='trend_event')
    # Measurement's claimed clean component omits the diagnostic trend/event term.
    x_without = g['s'] + g['periodic'] + g['cross'] + g['noise']
    composition_gap = float(np.max(np.abs(g['x'] - x_without)))
    output = {'reviewed_revision': '6f6c1a0', 'scope': 'saved evidence and disposable diagnostic fixtures only',
              'h3': comparisons, 'pure_readout_no_fusion_true_gamma_zero': no_fusion,
              'incomplete_pair_H3': partial['per_arch']['A']['H3'],
              'foreign_design_identity_accepted': foreign['per_arch'] == reported['per_arch'],
              'replay_cache': cache, 'adequacy': adequacy,
              'DX_component_reconstruction_max_error': composition_gap,
              'DX_metric_signal_definition': M.DECLARATIONS['snr_planted'],
              'evidence_unchanged': before == {p.name: sha(p) for p in (design_path, close_path)}}
    if args.run_root:
        checked, mismatches, max_error = 0, [], 0.0
        for cid, unit in local['units'].items():
            if unit['status'] != 'VERIFIED':
                continue
            attempt = args.run_root / 'attempts' / unit['attempt']
            with np.load(attempt / 'arrays.npz', allow_pickle=False) as arrays:
                for split in unit['record']['mase']:
                    error = np.abs(arrays[f'{split}_pred'] - arrays[f'{split}_y']).mean(axis=0)
                    mase = float(np.mean(error / arrays['denominator']))
                    diff = abs(mase - unit['record']['mase'][split])
                    max_error = max(max_error, diff)
                    if not np.isfinite(mase) or diff > 1e-10:
                        mismatches.append([cid, split])
            checked += 1
        cid = 'H2__h3__s1__A__profiles'
        attempt = args.run_root / 'attempts' / local['units'][cid]['attempt']
        job = json.loads((attempt / 'job.json').read_text())
        replay = json.loads((args.run_root / 'closure_v2d/replays' / f'{attempt.name}.json').read_text())
        base = C.verify_attempt(attempt, job, replay, C.DEFAULT_TOLERANCE)
        bad = copy.deepcopy(replay)
        bad['restore_abs_diff'] = float('nan')
        bad['recorded_best_validation_loss'] = float('nan')
        mutated = C.verify_attempt(attempt, job, bad, C.DEFAULT_TOLERANCE)
        output['physical_arrays'] = {'verified_attempts_checked': checked,
                                     'mismatches': mismatches, 'max_mase_difference': max_error,
                                     'scope': 'model MASE from saved arrays; not a fresh weight replay or live warehouse audit'}
        output['nonfinite_restore_replay'] = {'original_status': base['status'],
                                              'mutated_status': mutated['status'],
                                              'mutated_problems': mutated['problems'],
                                              'scope': 'in-memory replay copy; original files unchanged'}
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
