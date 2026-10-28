"""Publish audit measurements via the existing governed outbox; no learning."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--connection-contract', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    v = json.loads(a.probe.read_text())
    source_hash = hashlib.sha256(a.probe.read_bytes()).hexdigest()
    publication = json.loads(a.connection_contract.read_text())['publication']
    from tools import rl_warehouse
    a.out.mkdir(parents=True, exist_ok=True)
    rl_warehouse.prepare(publication, a.out)
    metrics = []
    def add(name, value, unit='count', split=None):
        metrics.append(dict(metric=name, value=value, unit=unit, split=split, horizon=None))
    for role, fields in v['business_support'].items():
        for key, value in fields.items():
            add('timing_audit.business.' + key, value, split=role)
    for item in v['classification']:
        for key in ('ensemble_log_loss', 'train_prior_log_loss', 'delta_vs_train_prior'):
            add('timing_audit.' + item['set'] + '.' + item['arm'] + '.' + key,
                item[key], 'nats', item['period'])
    for name, fields in v['clock_mapping'].items():
        for key, value in fields.items():
            add('timing_audit.clock.' + name + '.' + key, value,
                'price' if key.endswith('abs') else 'count')
    for year, value in v['admitted_missing_from_lake_by_year_candidate_clock'].items():
        add('timing_audit.candidate_clock.missing_admitted_bars', value, split=year)
    add('timing_audit.snapshot.model_files_listed', v['snapshot_scope']['model_files_listed'])
    add('timing_audit.snapshot.prediction_files_listed', v['snapshot_scope']['prediction_files_listed'])
    add('timing_audit.strategy.replayed_runs', len(v['strategy_replay_2024']))
    add('timing_audit.strategy.max_net_delta', max(abs(x['net_delta']) for x in v['strategy_replay_2024']), 'USD')
    add('timing_audit.strategy.max_sharpe_delta', max(abs(x['sharpe_delta']) for x in v['strategy_replay_2024']), 'ratio')
    add('timing_audit.labels.finality_passed', int(v['all_declared_cut_label_replay']['all_final_rows_unchanged']), 'flag')
    add('timing_audit.freeze.changed_cap_reached_model', int(v['frozen_strategy_binding_probe'].startswith('CHANGED_CAP_')), 'flag')
    key = 'timing-v3-audit:' + source_hash
    report = dict(experiment_key=key, experiment_set_key='timing-v3-independent-audit-20261019',
        actor=publication['actor'], lake=publication['lake'], project='predictor',
        phase='TIMING_SELECTION_V3_INDEPENDENT_AUDIT', code_commit=v['audited_predictor_commit'],
        config_sha256='b4da1012c53fd21003ec36f1d0c9f0e79354db1661444013168f86592fc2cced',
        lineage='UNVERIFIED', datasets=[], metrics=metrics,
        tags=dict(classification='RETROSPECTIVE_NONADMITTED_LAKE_PROXY_AUDIT',
                  audit_probe_sha256=source_hash, source_lake_sha256=v['pins']['lake'],
                  admitted_witness_sha256=v['pins']['admitted'],
                  label='AUDIT_DIAGNOSTIC_NOT_A_NEW_MODEL_OR_ADMISSION'))
    request = a.out / 'request.json'
    request.write_text(json.dumps(dict(report=report, source_unit=key), allow_nan=False))
    rl_warehouse.invoke(publication, a.out, 'enqueue', request)
    pump = a.out / 'pump.json'
    pump.write_text(json.dumps({k:publication[k] for k in ('warehouse_url','token_env')}))
    status = rl_warehouse.invoke(publication, a.out, 'pump', pump)
    (a.out / 'WAREHOUSE_STATUS.json').write_text(json.dumps(status, indent=2))
    w = status['warehouse']
    if w['pending'] or w['mismatch'] or w['delivered'] != 1:
        raise ValueError('AUDIT_PUBLICATION_NOT_COMPLETE')
    with sqlite3.connect(a.out / 'warehouse-outbox.sqlite') as source:
        with sqlite3.connect(a.out / 'warehouse-backup.sqlite') as backup:
            source.backup(backup)
    print(json.dumps(dict(metrics=len(metrics), warehouse={k:w[k] for k in ('expected','delivered','pending','mismatch')})))


if __name__ == '__main__':
    main()
