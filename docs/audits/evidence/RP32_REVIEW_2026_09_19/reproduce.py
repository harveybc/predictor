"""ML review: no training, no original writes, no live service operations."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--run-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve() / 'tools'))
    import df_e1_pilot as P
    import e3_weekly_controller as C
    import df_mod_e0_arch_verify as A
    root = a.run_root
    source = [p for p in root.rglob('*') if p.is_file()]
    before = {str(p.relative_to(root)): sha(p) for p in source}
    out = {'reviewed_revision': '60daac9', 'training_executed': False}
    d = dict(np.load(root / 'DATA.npz'))
    design = json.loads((root / 'DESIGN.json').read_text())
    W, h, j = (int(d[k][0]) for k in ('window', 'horizon', 'target_channel'))
    records, checks = {}, {}
    for ad in sorted((root / 'attempts').iterdir()):
        if not (ad / 'cell.json').exists():
            continue
        rec = json.loads((ad / 'cell.json').read_text())
        records[ad.name] = rec
        z = dict(np.load(ad / 'arrays.npz'))
        if rec['kind'] == 'fit':
            err = z['validation_pred'].astype(float) - z['validation_y']
            mae = float(np.mean(np.abs(err)))
            expected = rec['scores']['validation']['model']
            checks[ad.name] = {
                'finite': bool(np.isfinite(err).all()),
                'mae': mae, 'mase': mae / float(z['denominator'][0]),
                'mae_difference': abs(mae - expected['mae_mean']),
                'mase_difference': abs(mae / float(z['denominator'][0]) - expected['mase_mean']),
                'origins_equal_prepared': bool(np.array_equal(z['eval_origins'], d['eval_origins'])),
                'labels_equal_prepared': bool(np.array_equal(z['validation_y'][:, 0], d['Y'][d['eval_origins'] + h])),
                'restore_verified_record': rec['training']['restore_verified'],
                'budget_hit': rec['training']['truncation']['hit_update_ceiling']}
    out['saved_array_checks'] = checks

    # Valid record bytes alone must not make an unrelated/invalid scientific record acceptable.
    original = records['R0_s1']
    z0 = dict(np.load(root / 'attempts/R0_s1/arrays.npz'))
    cases = {}
    with tempfile.TemporaryDirectory(prefix='rp32-verifier-') as td:
        dest = Path(td)
        for case in ('intact_without_weights_or_job', 'mae_999', 'mase_nan', 'foreign_design_and_data', 'shifted_origin_ids'):
            rec, z = copy.deepcopy(original), {k: v.copy() for k, v in z0.items()}
            if case == 'mae_999':
                rec['scores']['validation']['model']['mae_mean'] = 999.0
            elif case == 'mase_nan':
                rec['scores']['validation']['model']['mase_mean'] = float('nan')
            elif case == 'foreign_design_and_data':
                rec['design_sha256'], rec['data_sha256'], rec['cell_id'] = '0' * 64, '1' * 64, 'R2_s999'
            elif case == 'shifted_origin_ids':
                z['eval_origins'] += 123
            np.savez(dest / 'arrays.npz', **z)
            rec['arrays_sha256'] = sha(dest / 'arrays.npz')
            (dest / 'cell.json').write_text(json.dumps(rec))
            result = {'output_file': 'cell.json', 'output_sha256': sha(dest / 'cell.json')}
            val, refusal = P.verified_unit(dest, result, result)
            cases[case] = {'accepted': val is not None, 'refusal': refusal}
    out['pilot_verifier_cases'] = cases

    with tempfile.TemporaryDirectory(prefix='rp32-close-') as td:
        dest = Path(td)
        for f in ('DESIGN.json', 'REPORT.json'):
            shutil.copyfile(root / f, dest / f)
        for ad in (root / 'attempts').iterdir():
            if not ad.is_dir():
                continue
            target = dest / 'attempts' / ad.name
            target.mkdir(parents=True)
            for f in ('cell.json', 'arrays.npz', 'outcome.json', 'result.json'):
                if (ad / f).is_file():
                    shutil.copyfile(ad / f, target / f)
        base_close = P.close(dest)
        altered = copy.deepcopy(design)
        altered['design_sha256'] = '0' * 64
        altered['task']['horizon_steps'] = 120
        (dest / 'DESIGN.json').write_text(json.dumps(altered))
        bad_close = P.close(dest)
        out['full_closure_foreign_design'] = {
            'emitted_design': bad_close['design_sha256'],
            'emitted_horizon_steps': bad_close['task']['horizon_steps'],
            'all_present_units_verified': all(bad_close['verified_units'].values()),
            'means_unchanged': base_close['means'] == bad_close['means']}

    c = P.L.household_contract(window=8, horizon=0)
    n = 200
    frame = pd.DataFrame({k: np.arange(n, dtype=float) + i for i, k in enumerate(c.features + c.targets)})
    frame['timestamp_label'] = pd.date_range('2010-01-01', periods=n, freq='min').strftime('%d/%m/%Y %H:%M:%S')
    resolved = P.L.resolve(frame, c)
    en = P.L.enumerate_windows(resolved, c)
    ten = P.L.build_tensors(resolved, en, 'train', c, None)
    out['zero_horizon'] = {'accepted': True, 'origins': len(ten['origins']),
        'target_is_in_input_window': bool(np.array_equal(ten['y'][:, 0], ten['X'][:, -1, -1]))}
    c.horizon = -1
    try:
        P.L.enumerate_windows(resolved, c)
    except Exception as exc:
        out['negative_horizon'] = {'exception_type': type(exc).__name__, 'reason': str(exc)}

    WB = P._dataset_class()
    origins = d['eval_origins'][:16]
    ds = WB(d['Xs'], d['Y'], origins, W, h, j, 16, scaler_mean=d['scaler_mean'], scaler_sd=d['scaler_sd'], shuffle=False, seed=0, masked=0.3)
    ae, _ = P.RG.build_autoencoder(design['graph']['assignment'], W, d['Xs'].shape[1], arch='A', seed=1, mask_ratio=0.3)
    for filename in ('detector_pretrained.npz', 'decoder.npz'):
        P.RG.load_detector(ae, root / 'attempts/ae_s1' / filename)
    def masked_loss(batch):
        x, y = batch
        p = d['Xs'].shape[1]
        pred = np.asarray(ae(x, training=False))
        return float(np.sum(y[..., p:] * np.square(y[..., :p] - pred)) / np.sum(y[..., p:]))
    b0 = ds[0]
    loss0 = masked_loss(b0)
    ds.on_epoch_end()
    b1 = ds[0]
    out['masked_validation'] = {'same_clean_values': bool(np.array_equal(b0[1][..., :d['Xs'].shape[1]], b1[1][..., :d['Xs'].shape[1]])),
        'changed_mask_positions': int(np.count_nonzero(b0[1][..., d['Xs'].shape[1]:] != b1[1][..., d['Xs'].shape[1]:])),
        'fixed_weights_loss_epoch0': loss0, 'fixed_weights_loss_epoch1': masked_loss(b1)}

    model = P._model_for_target(design['graph']['assignment'], W, d['Xs'].shape[1], j, 1)
    model.load_weights(str(root / 'attempts/R0_s1/weights.weights.h5'))
    x = P._gather(d['Xs'], d['eval_origins'][:64], W)
    y0 = np.asarray(model(x, training=False))
    xp = x.copy()
    xp[:, :-7, :] += 100
    yp = np.asarray(model(xp, training=False))
    xl = x.copy()
    xl[:, -7:, :] += 1
    yl = np.asarray(model(xl, training=False))
    z = z0['validation_pred'][:64, 0]
    native = y0[:, 0] * d['scaler_sd'][j] + d['scaler_mean'][j]
    out['actual_model_reach'] = {'old_53_rows_perturbation_max_difference': float(np.max(np.abs(y0 - yp))),
        'last_7_rows_perturbation_max_difference': float(np.max(np.abs(y0 - yl))),
        'saved_weights_prediction_check_rows': 64, 'max_difference_to_saved_predictions': float(np.max(np.abs(native - z))),
        'sample_count': 7, 'first_to_last_span_seconds': 6 * 60,
        'linear_control_samples': W, 'seasonal_control_lookup_lag_steps': 1440 - h}

    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    step = timedelta(hours=1)
    release = C.ModelRelease(t, t - 3 * step, t - 3 * step, t - 2 * step, t - step, t, 'old')
    ctrl = C.WeeklyLongFlatController([release], latency_bars=0)
    rec = ctrl.decide(t, step, C.LONG, 1000.0, 100.0, 0.0)
    out['weekly_controller'] = {'zero_latency_accepted': rec['expected_fill_time'] == rec['decision_time'],
        'external_proposal_not_model_bound': {'record_model': rec['model'], 'action': rec['action'],
             'note': 'API accepts an unlabelled proposal; it does not call or bind the selected model'},
        'nan_price_action': ctrl.decide(t, step, C.LONG, 1000.0, float('nan'), 0.0)['action']}

    base = a.repo / 'docs/audits/evidence/d3_k5_20260917'
    pdsg, pc, sdsg, sc = [json.loads((base / name).read_text()) for name in (
        'RP14_ARCH_STAGE_DESIGN.json', 'RP22_ARCH_STAGE_CLOSE_V3.json',
        'RP22_ARCH_READOUT_COMPLETION_DESIGN.json', 'RP22_ARCH_RC_CLOSE.json')]
    pc, sc = pc.get('local', pc), sc.get('local', sc)
    donor = next(k for k, u in sc['units'].items() if u.get('role') == 'INHERITED')
    original_donor = copy.deepcopy(sc['units'][donor])
    changed = copy.deepcopy(sc)
    changed['units'][donor]['record']['updates'] = 0
    try:
        mc, md = A.merge_successor(pc, pdsg, changed, sdsg)
        out['e0_donor_update_change'] = {'accepted': True, 'contradictions': mc['contradictions'],
            'original_updates': original_donor['record']['updates'], 'changed_updates': 0}
    except Exception as exc:
        out['e0_donor_update_change'] = {'accepted': False, 'reason': str(exc)}

    out['original_files_count'] = len(before)
    out['originals_unchanged'] = all(sha(root / p) == hsh for p, hsh in before.items())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
