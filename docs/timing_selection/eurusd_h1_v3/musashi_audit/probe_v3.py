"""Read-only independent v3 evidence checks. No fitting, services or TEST use.

Run with CUDA_VISIBLE_DEVICES='' and single-thread BLAS. All output paths are
chosen by the caller; nothing is written to the sealed study.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'predictor', 'strategy', 'admitted', 'lake', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--replay-labels', action='store_true')
    p.add_argument('--replay-strategy', action='store_true')
    a = p.parse_args()
    if a.output.resolve().is_relative_to(a.study.resolve()):
        raise ValueError('OUTPUT_MUST_NOT_MODIFY_STUDY')
    sys.path.insert(0, str(a.strategy))
    from tools import short_horizon_timing_eval as E
    ev = a.study / 'eval_input'
    sets, feats, targets, ts = E.load_inputs(ev, include_2024=True)
    common, period, y, _, _ = E.common_rows(feats, targets, ts, sets, sets['k_bars'])
    ctx = E.Ctx(a.lake)
    assert ts.equals(ctx.bars.index)
    seg = ctx.cal.assign(ts)
    H = pd.Timedelta(hours=1)
    operational = np.array([(t + H >= ctx.cal.segments[s].operational_start
                             and t + H < ctx.cal.segments[s].force_flat)
                            for t, s in zip(ts, seg)])
    # A tradable next-open origin does not imply every future target is usable.
    k = int(sets['k_bars'])
    endpoint_same_segment = np.zeros(len(ts), bool)
    endpoint_before_flat = np.zeros(len(ts), bool)
    endpoint_same_segment[:-k] = seg[:-k] == seg[k:]
    endpoint_before_flat[:-k] = np.array([t < ctx.cal.segments[s].force_flat
                                        for t, s in zip(ts[k:], seg[:-k])])
    business = {}
    for role in E.EVAL_PERIODS:
        m = common & (period == role)
        business[role] = dict(rows=int(m.sum()), outside_operational_origin=int((m & ~operational).sum()),
            horizon_crosses_segment=int((m & ~endpoint_same_segment).sum()),
            horizon_endpoint_not_before_flat=int((m & ~endpoint_before_flat).sum()))

    # Baseline probabilities are fitted on TRAIN only, never on evaluated labels.
    train_y = y[common & (period == 'TRAIN_2020_2022')].astype(int)
    prior = np.bincount(train_y, minlength=3) / len(train_y)
    classification = []
    frozen = load(a.study / 'eval/FROZEN.json')
    for key, fz in frozen['shards'].items():
        shard = a.study / 'eval/shards' / key
        dev = load(shard / 'DEVELOP.json')
        assert sha(shard / 'develop_predictions.npz') == dev['predictions_sha256']
        for model in fz['models'].values():
            assert sha(shard / 'models' / model['file']) == model['sha256']
        for role in ('INNER_B_2023H2', 'REOBSERVED_2024'):
            probabilities = []
            for seed in fz['models']:
                if role == 'INNER_B_2023H2':
                    z = np.load(shard / 'frozen_calibrated_develop.npz', allow_pickle=True)
                    mask = z['period'].astype(str) == role
                    rows, yy, P = z['rows'][mask], z['y'][mask], z['s' + seed][mask]
                else:
                    z = np.load(shard / ('reobserved_2024_s' + seed + '.npz'))
                    rows, yy, P = z['rows'], z['y'], z['P']
                assert np.array_equal(yy, y[rows].astype(int))
                assert np.array_equal(rows, np.flatnonzero(common & (period == role)))
                probabilities.append(P.astype(float))
            P = np.mean(probabilities, axis=0)
            ce = float(-np.log(np.clip(P[np.arange(len(yy)), yy], 1e-9, 1)).mean())
            prior_ce = float(-np.log(prior[yy]).mean())
            classification.append(dict(set=fz['set'], arm=fz['arm'], period=role,
                n=len(yy), ensemble_log_loss=ce, train_prior_log_loss=prior_ce,
                delta_vs_train_prior=ce - prior_ce))

    # The NY localization is a tested candidate mapping, not an assumed clock certificate.
    admitted = pd.read_parquet(a.admitted, filters=[('available_at', '<', pd.Timestamp('2025-01-01', tz='UTC'))])
    assert admitted.available_at.max() < pd.Timestamp('2025-01-01', tz='UTC')
    admitted = admitted.set_index(pd.DatetimeIndex(admitted.timestamp).tz_convert('UTC'))
    maps = {'identity_utc': ts, 'fixed_plus4': ts + 4 * H, 'fixed_plus5': ts + 5 * H,
            'candidate_NY_to_UTC': ts.tz_localize(None).tz_localize('America/New_York').tz_convert('UTC')}
    clock = {}
    for name, index in maps.items():
        actual = admitted.BidClose.reindex(index).to_numpy()
        d = np.abs(actual - ctx.bars.CLOSE.to_numpy())
        ok = np.isfinite(d)
        clock[name] = dict(matched=int(ok.sum()), exact_close_matches=int((d[ok] == 0).sum()),
                           p50_abs=float(np.median(d[ok])), p95_abs=float(np.quantile(d[ok], .95)))
    mapped = maps['candidate_NY_to_UTC']
    aidx = admitted.index
    same_bounds = (aidx >= mapped[0]) & (aidx <= mapped[-1])
    missing = aidx[same_bounds & ~aidx.isin(mapped)]
    missing_counts = {str(year): int((missing.year == year).sum()) for year in range(2020, 2025)}
    elapsed = feats.RAW_ELAPSED_HOURS.to_numpy()
    gaps = []
    for i in np.flatnonzero(elapsed > 1):
        between = (aidx > mapped[i - 1]) & (aidx < mapped[i])
        n = int(between.sum())
        if n:
            gaps.append(dict(origin_label=str(ts[i]), elapsed_feature=float(elapsed[i]),
                             admitted_bars_skipped=n, year=int(ts[i].year)))

    snapshot = load(a.study / 'SNAPSHOT.json')
    covered = set(snapshot['sha256sums'])
    essential = ['data/pool_all.parquet', 'data/features_train.parquet', 'data/targets_train.parquet',
                 'eval_input/features_all.parquet', 'eval_input/targets_all.parquet', 'metrics.duckdb']
    snapshot_scope = dict(listed_files=len(covered), omitted_replay_inputs=[x for x in essential if x not in covered],
        model_files_listed=sum('/models/' in x for x in covered),
        prediction_files_listed=sum(x.endswith('.npz') for x in covered),
        manifested_selected_models=sum(len(x['models']) for x in frozen['shards'].values()))

    # Direct counterexample: daily_returns currently drops first-day PnL.
    daily = pd.Series([110., 121.], index=pd.date_range('2024-01-01', periods=2, tz='UTC'))
    daily_probe = dict(initial_equity=100, daily_closes=daily.tolist(),
        adapter_definition_returns=[.1, .1], evaluator_returns=E.daily_returns(SimpleNamespace(daily_equity=daily)).tolist())
    label_replay = None
    if a.replay_labels:
        spec = importlib.util.spec_from_file_location('audit_timing_targets', a.predictor / 'tools/timing_pivot_targets.py')
        tp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tp)
        params = E.Z.Params(**sets['expert']['params'])
        cuts = ['2021-01-01', '2022-01-01', '2023-01-01', '2023-07-01', '2024-01-01']
        cut_positions = tp._cut_positions(ctx.bars.index, cuts)
        label_replay = tp.finality_check(ctx.bars, E.Z, params, k_bars=k,
            cut_positions=cut_positions, confirmation=sets['expert']['confirmation'])
    strategy_replay = []
    if a.replay_strategy:
        _, pivots, ktp = E.setup_expert(sets, ctx, None)
        scores = pd.read_csv(a.study / 'eval/scores_reobserved_2024.csv')
        for key, fz in frozen['shards'].items():
            shard = a.study / 'eval/shards' / key
            for seed in fz['models']:
                z = np.load(shard / ('reobserved_2024_s' + seed + '.npz'))
                met, res, _ = E.strategy(ctx, pivots, z['P'].astype(float), z['rows'],
                    fz['threshold'], k, sets['strategy']['k_tp'], 'REOBSERVED_2024')
                original = scores[(scores['set'] == fz['set']) & (scores['arm'] == fz['arm']) & (scores.seed == int(seed))].iloc[0]
                strategy_replay.append(dict(set=fz['set'], arm=fz['arm'], seed=int(seed),
                    net_delta=float(met['net_profit_usd'] - original.net_profit_usd),
                    sharpe_delta=float(met['sharpe_daily_sqrt365'] - original.sharpe_daily_sqrt365),
                    entries_delta=int(met['entries_filled'] - original.entries_filled),
                    initial_capital=ctx.prof.initial_capital if hasattr(ctx.prof, 'initial_capital') else E.STUDY.CAPITAL,
                    first_daily_close=float(res.daily_equity.iloc[0])))
    # A changed strategy must be rejected against FROZEN before model loading.
    # Retain real data, model bytes and freeze; change only cap4 -> cap3 in a
    # temporary FEATURE_SETS. The sentinel prevents inference or any score writes.
    with tempfile.TemporaryDirectory(prefix='timing-v3-binding-audit-') as tmp:
        temp = Path(tmp)
        tev, tout = temp / 'eval_input', temp / 'eval'
        tev.mkdir()
        tout.mkdir()
        changed = load(ev / 'FEATURE_SETS.json')
        changed['strategy']['max_orders_per_day'] = 3
        (tev / 'FEATURE_SETS.json').write_text(json.dumps(changed))
        for filename in ('features_all.parquet', 'targets_all.parquet'):
            (tev / filename).symlink_to(ev / filename)
        shutil.copyfile(a.study / 'eval/FROZEN.json', tout / 'FROZEN.json')
        first_key = next(iter(frozen['shards']))
        first_model = next(iter(frozen['shards'][first_key]['models'].values()))['file']
        model_dest = tout / 'shards' / first_key / 'models' / first_model
        model_dest.parent.mkdir(parents=True)
        model_dest.symlink_to(a.study / 'eval/shards' / first_key / 'models' / first_model)
        old_loader, old_threads = E.load_model, E._set_threads
        class ReachedModelLoad(RuntimeError):
            pass
        def sentinel(*args, **kwargs):
            raise ReachedModelLoad()
        E.load_model, E._set_threads = sentinel, lambda: None
        try:
            E.reconfirm(SimpleNamespace(eval_input=tev, out=tout, data=a.lake, zigzag_run=None))
            binding = 'UNEXPECTED_COMPLETION'
        except ReachedModelLoad:
            binding = 'CHANGED_CAP_REACHED_MODEL_LOAD_WITHOUT_FREEZE_REFUSAL'
        except SystemExit as exc:
            binding = str(exc)
        finally:
            E.load_model, E._set_threads = old_loader, old_threads
    result = dict(audited_predictor_commit='53d2bfb79e5363b77cc04e654320f5b0ca7e5c3b',
        pins=dict(lake=sha(a.lake), admitted=sha(a.admitted), expert=sha(a.strategy / 'app/short_horizon/zigzag.py'),
                  features=sha(ev / 'features_all.parquet'), targets=sha(ev / 'targets_all.parquet')),
        business_support=business, classification=classification, train_class_prior=prior.tolist(),
        clock_mapping=clock, admitted_missing_from_lake_by_year_candidate_clock=missing_counts,
        elapsed_feature_skipped_admitted_bars=gaps, snapshot_scope=snapshot_scope,
        daily_return_counterexample=daily_probe,
        all_declared_cut_label_replay=label_replay, strategy_replay_2024=strategy_replay,
        frozen_strategy_binding_probe=binding,
        limits=['No fitting or GPU use', 'NY mapping remains a candidate until verified by data owner',
                'Per-seed trained models authenticated but not re-inferred',
                'Business counts use the sealed adapter calendar, not a broker-certified calendar'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:result[k] for k in ('business_support','clock_mapping','snapshot_scope','daily_return_counterexample')}))


if __name__ == '__main__':
    main()
