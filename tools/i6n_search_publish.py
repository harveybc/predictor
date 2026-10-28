"""Verify historical search terminals and publish TRAIN/inner metrics, never fit.

Run once per completed candidate on the coordinator. Full directory rewrites
are outside unkeyed hash custody; the caller must trust the retained transport.
"""
import argparse
import json
from pathlib import Path
import urllib.parse

from tools import i6n_publish as base


def verified_search(root):
    """Recompute paired scores from source observations without importing TF."""
    from tools import i6n_historical_evaluator as evaluator
    state = json.loads((root/'STATUS.json').read_text())
    if state['state'] != 'COMPLETED':
        raise ValueError('only completed candidates can publish')
    result = json.loads((root/'RESULT.json').read_text())
    identity = result['identity']
    request = dict(config=identity['effective_config'], generation=0,
                   context=identity['context'], support=identity['support'])
    cfg, frozen = evaluator.load_request(request)
    if cfg != identity['cell']['config']:
        raise ValueError('retained config differs from original binding')
    data = evaluator.cell.prepare(cfg, trajectory=request['context']['schema'] == evaluator.SCHEMA)
    if evaluator.support(data, cfg) != frozen['support']:
        raise ValueError('retained support differs from source')
    evaluator.retained(root, identity, data, cfg, frozen)
    return result


def search_reports(root, pin):
    """Reuse the C2 projection while explicitly separating scoring splits."""
    result = verified_search(root)
    sha = base.file_hash(root/'RESULT.json')
    output = []
    for split in ('train', 'inner'):
        projected = dict(scope='HISTORICAL_SINGLE_WEEK_DEVELOPMENT', state='COMPLETED',
                         identity=result['identity']['cell'],
                         metrics=json.loads((root/f'{split}.c2.json').read_text()))
        for report in base.reports(projected, sha, pin):
            report['experiment_key'] += ':'+split
            report['phase'] = 'I6N_SEARCH_TRAIN_INNER'
            report['tags'].update(scope='SEARCH_TRAIN_INNER_NOT_OUTER_VALIDATION',
                                  scoring_split=split, search_space_scope='ENDPOINT_ONLY')
            for metric in report['metrics']:
                metric['split'] = split
            report['report_sha256'] = base.report_sha256(report)
            output.append(report)
    return output


def publish(root, pin, url, token_file):
    """Idempotent submission with exact readback, independently of GPU work."""
    bodies = search_reports(root, pin)
    output = root/'olap'
    output.mkdir(exist_ok=True)
    base.atomic_json(output/'REPORTS.json', bodies)
    token = base.token_from_file(token_file)
    count = 0
    for body in bodies:
        receipt = base.request_json(url.rstrip('/')+'/api/v1/metrics', token, 'POST', body)
        sha = body['report_sha256']
        if receipt.get('report_sha256') != sha:
            raise ValueError('warehouse receipt digest mismatch')
        sql = "SELECT report_sha256,metric,value,unit,horizon,split FROM gov_metric WHERE report_sha256='"+sha+"' LIMIT 1000"
        rows = base.request_json(url.rstrip('/')+'/api/v1/query?'+urllib.parse.urlencode({'sql': sql}), token, 'GET')['rows']
        base.verify_readback([body], rows)
        count += len(rows)
        base.atomic_json(output/(sha+'.json'), dict(state='PUBLISHED_READBACK_VERIFIED', rows=len(rows)))
    status = dict(state='PUBLISHED_READBACK_VERIFIED', reports=len(bodies), rows=count,
                  terminal_sha256=base.file_hash(root/'RESULT.json'),
                  destination=url.rstrip('/'), code_commit=pin)
    base.atomic_json(output/'STATUS.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--token-file', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.pin, args.url, args.token_file)))


if __name__ == '__main__':
    main()
