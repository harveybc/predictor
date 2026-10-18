"""Explicit coverage for the weekly TEST closure (part B of ORDEN_SATOSHI_OPERACION.md).

Adds, without touching any metric: declared / scored / no-pair / failed / pending weeks per family
with the exact invariant, the ids and bounds of the no-pair weeks (year boundaries flagged), and
per-horizon support counts. Aggregation stays sums/counts with each head's own paired naive.
"""
from datetime import datetime

GLOSSARY = {
    'SIN_PARES': 'Descriptive name of the persisted week disposition SKIPPED_NO_EVAL_ROWS: the week has no '
                 'valid (origin, head) pair in the common window; nothing is predicted or scored, the donor '
                 'checkpoint is carried forward unchanged. It is NOT a zero MAE and NOT a successful fit.',
    'SKIPPED_NO_EVAL_ROWS': 'Persisted RESULT.json state of a SIN_PARES week (receipts are never renamed).',
    'NO_VALID_TARGET_IN_PERIOD': 'Per-head disposition inside a scored week: that horizon had no valid pair, '
                                 'so it contributes no error and no count to the annual pool.',
    'FAILED': 'A terminal failure (permanent, unknown or transient budget exhausted); its prediction, if '
              'sealed, is retained but the week is not scored and its successor is blocked.',
    'PENDING': 'Declared week without a terminal yet.',
}


def _year(iso):
    return datetime.fromisoformat(iso.replace('Z', '+00:00')).year


def _disposition(result):
    if result is None:
        return 'pending'
    state = result['state']
    if state == 'COMPLETE':
        return 'scored'
    if state == 'SKIPPED_NO_EVAL_ROWS':
        return 'no_pair'
    if state.startswith('FAILED'):
        return 'failed'
    raise ValueError('UNKNOWN_TERMINAL_STATE:' + state)


def _week(cell, result):
    period = cell['period']
    start, end = period['train_end'], period['validation_end']
    return dict(id=cell['id'], period_id=period.get('id'), week_ids=period.get('week_ids'),
                market_open=start, market_close=end, year_boundary=_year(start) != _year(end),
                disposition=_disposition(result), state=None if result is None else result['state'])


def coverage(spec, results, heads_by_family):
    """{families: {family: {...}}, glossary, invariant_holds}; raises on a broken invariant."""
    families = {}
    for family, heads in heads_by_family.items():
        cells = [c for c in spec['cells'] if c['family'] == family]
        weeks = [_week(c, results.get(c['id'])) for c in cells]
        counts = {k: sum(w['disposition'] == k for w in weeks) for k in ('scored', 'no_pair', 'failed', 'pending')}
        if sum(counts.values()) != len(cells):
            raise ValueError('COVERAGE_INVARIANT:' + family)
        per_head = {}
        for j, h in enumerate(heads):
            with_pairs, without, pairs = [], [], 0
            for c, w in zip(cells, weeks, strict=True):
                if w['disposition'] == 'no_pair':
                    without.append(w['id'])
                    continue
                if w['disposition'] != 'scored':
                    continue
                row = next((m for m in results[c['id']]['metrics']['TEST'] if m['horizon'] == h), None)
                if row is None or not row['n']:
                    without.append(w['id'])
                else:
                    with_pairs.append(w['id'])
                    pairs += int(row['n'])
            per_head[str(h)] = dict(horizon=h, weeks_with_pairs=len(with_pairs), weeks_without_support=len(without),
                                    weeks_without_support_ids=without, pairs_scored=pairs,
                                    aggregation='sum_abs_error/sum_n with the paired own naive of this horizon')
        families[family] = dict(
            declared_weeks=len(cells), **{k + '_weeks': v for k, v in counts.items()},
            invariant='scored_weeks + no_pair_weeks + failed_weeks + pending_weeks == declared_weeks',
            invariant_holds=True,
            prediction_coverage=dict(predicted_and_scored=counts['scored'], declared=len(cells),
                                     fraction=counts['scored'] / len(cells) if cells else None,
                                     note='no-pair, failed and pending weeks are NOT predictions; this is never '
                                          '100% while any such week exists'),
            no_pair_weeks_detail=[w for w in weeks if w['disposition'] == 'no_pair'],
            failed_weeks_detail=[dict(w, cause=results[w['id']].get('reason')) for w in weeks if w['disposition'] == 'failed'],
            pending_week_ids=[w['id'] for w in weeks if w['disposition'] == 'pending'],
            year_boundary_week_ids=[w['id'] for w in weeks if w['year_boundary']],
            weeks=weeks, per_horizon=per_head,
            complete=counts['pending'] == 0 and counts['failed'] == 0)
    return dict(schema='i6dmo.weekly_test_coverage.v1', families=families, glossary=GLOSSARY,
                invariant_holds=True, test_read=True)


def markdown(cov):
    lines = ['## Coverage', '',
             '| Family | Declared | Scored | SIN_PARES (no pair) | Failed | Pending | Scored / declared |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for family, f in cov['families'].items():
        frac = f['prediction_coverage']['fraction']
        lines.append(f"| {family} | {f['declared_weeks']} | {f['scored_weeks']} | {f['no_pair_weeks']} | "
                     f"{f['failed_weeks']} | {f['pending_weeks']} | {'n/a' if frac is None else f'{100 * frac:.2f}%'} |")
    lines += ['', 'Invariant: scored + SIN_PARES + failed + pending = declared, per family. A SIN_PARES week is '
                  'not a prediction, not a zero MAE and not a fit; no-pair, failed and pending weeks are '
                  'excluded from the prediction coverage above.', '']
    for family, f in cov['families'].items():
        if f['no_pair_weeks_detail']:
            lines += [f'### {family}: SIN_PARES weeks (SKIPPED_NO_EVAL_ROWS)', '',
                      '| Cell | Market open | Market close | Year boundary |', '|---|---|---|---|']
            lines += [f"| {w['id']} | {w['market_open']} | {w['market_close']} | {'yes' if w['year_boundary'] else 'no'} |"
                      for w in f['no_pair_weeks_detail']]
            lines.append('')
        if f['failed_weeks_detail']:
            lines += [f'### {family}: failed weeks', '', '| Cell | Cause |', '|---|---|']
            lines += [f"| {w['id']} | {w['cause']} |" for w in f['failed_weeks_detail']]
            lines.append('')
        if f['pending_week_ids']:
            lines += [f"### {family}: pending weeks: {', '.join(f['pending_week_ids'])} (closure INCOMPLETE)", '']
        lines += [f'### {family}: support per horizon', '',
                  '| Horizon | Weeks with pairs | Weeks without support | Pairs scored (N) |', '|---:|---:|---:|---:|']
        lines += [f"| {h['horizon']} h | {h['weeks_with_pairs']} | {h['weeks_without_support']} | {h['pairs_scored']} |"
                  for h in f['per_horizon'].values()]
        lines.append('')
    lines += ['### Glossary', ''] + [f'- `{k}`: {v}' for k, v in cov['glossary'].items()] + ['']
    return '\n'.join(lines)
