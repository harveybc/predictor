"""Export verified historical C2 groups with exact readback; never refit a model."""
import argparse
import json
from pathlib import Path
import urllib.parse

from tools.i6n_contracts import paired_metrics as c2
from tools.i6n_historical_cell import verified
from tools.i6m_data import file_hash
from tools.i6m_queue import atomic_json
from tools.i6a_publish_olap import request_json, token_from_file, report_sha256
from tools.i6e_publish_olap import verify_readback


def reports(result, terminal_sha256, pin):
    """Keep S/U, role, calendar, protocol, original units and horizon distinct."""
    if result.get('scope') not in ('HISTORICAL_SINGLE_WEEK_DEVELOPMENT', 'TRAIN_COST_PILOT') or result.get('state') != 'COMPLETED':
        raise ValueError('only completed historical development terminals admitted')
    rows = c2.olap_records(result['metrics'])
    cfg = result['identity']['config']
    output = []
    for row in rows:
        if row['record_type'] != 'I6N_PAIRED_METRIC':
            continue
        kind, role, h = row['population_kind'], row['role'], row['horizon_hours']
        items = []
        for name, unit in [('mae', row['unit']), ('naive_mae', row['unit']),
                           ('mse', row['unit']+'_squared'), ('naive_mse', row['unit']+'_squared'),
                           ('skill_mae', 'ratio'), ('skill_mse', 'ratio'), ('n', 'count'),
                           ('n_declared', 'count'), ('n_excluded', 'count'), ('coverage', 'ratio')]:
            if row.get(name) is not None:
                items.append(dict(metric=name, value=row[name], unit=unit, horizon=h, split='validation'))
        for exclusion in rows:
            if (exclusion['record_type'] == 'I6N_POPULATION_COUNT'
                    and (exclusion['population_kind'], exclusion['role'], exclusion['horizon_hours']) == (kind, role, h)):
                items.append(dict(metric=exclusion['category']+':'+exclusion['reason'],
                    value=exclusion['count'], unit='count', horizon=h, split='validation'))
        report = dict(experiment_key=f'i6n:{terminal_sha256}:{kind}:{role}:h{h}',
            experiment_set_key='i6n:historical-development:'+row['config_id'], actor='predictor',
            lake='LOCAL_RETAINED', lineage='UNVERIFIED', config_sha256=row['config_id'], code_commit=pin,
            project='predictor', phase='I6N_HISTORICAL_DEVELOPMENT', metrics=items,
            tags=dict(population_kind=kind, role=role, horizon_hours=str(h),
                calendar_id=row['calendar_id'], protocol_id=row['protocol_id'],
                week=cfg['evaluation_week_id'], seed=str(cfg['training']['seed']),
                selected_using_development='true', broker_calendar_verified='false',
                point_in_time_vintages_certified='false', production_eligible='false',
                terminal_sha256=terminal_sha256, paired_rows_sha256=row['row_keys_sha256'],
                clock_manifest_sha256=cfg['source_clock']['sha256'], test_read='false',
                scope='ONE_WEEK_NOT_ANNUAL', skill_formula='(error_h-naive_h)/naive_h'))
        report['report_sha256'] = report_sha256(report)
        output.append(report)
    return output


def publish(root, pin, url=None, token_file=None):
    """Verify retained artifacts and recompute C2 before preparing or submitting."""
    terminal = root/'RESULT.json'
    retained = json.loads(terminal.read_text())
    # Digest custody is unkeyed; caller must supply a trusted retained directory.
    result = verified(root, retained['identity'])
    bodies = reports(result, file_hash(terminal), pin)
    out = root/'olap'
    out.mkdir(exist_ok=True)
    atomic_json(out/'REPORTS.json', bodies)
    if url is None:
        return dict(state='PREPARED_NOT_PUBLISHED', reports=len(bodies))
    token = token_from_file(token_file)
    for body in bodies:
        receipt = request_json(url.rstrip('/')+'/api/v1/metrics', token, 'POST', body)
        sha = body['report_sha256']
        if receipt.get('report_sha256') != sha:
            raise ValueError('warehouse receipt digest mismatch')
        sql = "SELECT report_sha256,metric,value,unit,horizon,split FROM gov_metric WHERE report_sha256='"+sha+"' LIMIT 1000"
        rows = request_json(url.rstrip('/')+'/api/v1/query?'+urllib.parse.urlencode(dict(sql=sql)), token, 'GET')['rows']
        verify_readback([body], rows)
        atomic_json(out/(sha+'.json'), dict(state='PUBLISHED_READBACK_VERIFIED', report_sha256=sha,
                    terminal_sha256=file_hash(terminal), rows=len(rows)))
    state = dict(state='PUBLISHED_READBACK_VERIFIED', reports=len(bodies), terminal_sha256=file_hash(terminal),
                 destination=url.rstrip('/'), code_commit=pin)
    atomic_json(out/'STATUS.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--url')
    parser.add_argument('--token-file', type=Path)
    args = parser.parse_args()
    if args.url and not args.token_file:
        parser.error('--token-file required with --url')
    print(json.dumps(publish(args.root, args.pin, args.url, args.token_file)))


if __name__ == '__main__':
    main()
