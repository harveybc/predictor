"""Read-only artifact custody audit; not a closure/deletion certificate."""
import argparse
import hashlib
import json
import re
import shlex
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(root, unit, query):
    design = json.loads((root / 'DESIGN.json').read_text())
    receipts = json.loads((root / 'TERMINAL_RECEIPTS.json').read_text())['units']
    cell = next(c for c in design['cells'] if c['cell_id'] == unit)
    report_path = root / 'REPORT.json'
    report = json.loads(report_path.read_text())
    checks, terminals = {}, {}
    for name, files in {
        'prepare': {'data': root / 'BENCH_DATA.npz', 'record': root / 'BENCH_DATA.json'},
        unit: {role: root / 'attempts' / unit / file for role, file in
               [('predictions', 'arrays.npz'), ('checkpoint', 'checkpoint.pth'), ('record', 'cell.json')]},
    }.items():
        receipt = receipts[name]
        campaign = receipt['campaign_sha256']
        if not re.fullmatch('[a-f0-9]{64}', campaign):
            raise ValueError('Invalid campaign digest')
        rows = query('SELECT unit_id, terminal_sha256, generation, status, config_sha256 '
                     f"FROM main.gov_terminal WHERE campaign_sha256 = '{campaign}' LIMIT 5000")
        rows = [r for r in rows if r['unit_id'] == name]
        if not rows:
            raise ValueError('Missing accepted terminal')
        row = max(rows, key=lambda r: int(r['generation']))
        digest = row['terminal_sha256']
        if not re.fullmatch('[a-f0-9]{64}', digest):
            raise ValueError('Invalid terminal digest')
        artifacts = query('SELECT role, sha256, bytes FROM main.gov_terminal_artifact '
                          f"WHERE terminal_sha256 = '{digest}' LIMIT 1000")
        checks[name + ':terminal'] = digest == receipt['terminal_sha256'] and row['status'] == 'COMPLETED'
        checks[name + ':design'] = row['config_sha256'] == design['design_sha256']
        local = {}
        for role, path in files.items():
            local[role] = {'sha256': sha(path), 'bytes': path.stat().st_size}
            matches = [r for r in artifacts if r['role'] == role]
            checks[name + ':' + role] = (len(matches) == 1 and matches[0]['sha256'] == local[role]['sha256']
                                        and int(matches[0]['bytes']) == local[role]['bytes'])
        terminals[name] = {'terminal_sha256': digest, 'generation': row['generation'], 'artifacts': local}
    record = json.loads((root / 'attempts' / unit / 'cell.json').read_text())
    checks['record_design'] = record['design_sha256'] == design['design_sha256']
    checks['record_cell'] = all(record['cell'][k] == cell[k] for k in ('cell_id', 'horizon', 'seed', 'seq_len', 'arm'))
    rr = [r for r in report['verification']['rows'] if r['unit'] == unit]
    checks['report_design'] = report['design_sha256'] == design['design_sha256']
    checks['report_population'] = len(rr) == 1
    rep = rr[0].get('replay') or {} if len(rr) == 1 else {}
    checks['recorded_replay_exact'] = (rep.get('finite') is True and rep.get('shape_equal') is True
                                       and rep.get('allclose_rule') is True
                                       and type(rep.get('max_abs_prediction_difference')) in (int, float)
                                       and rep['max_abs_prediction_difference'] == 0)
    return {'schema': 'musashi.live_custody_followup.v1', 'at': datetime.now(timezone.utc).isoformat(),
            'unit': unit, 'checks': checks, 'live_custody_pass': all(v for k, v in checks.items() if k != 'recorded_replay_exact'),
            'previous_report_sha256': sha(report_path), 'previous_report_replay': rep,
            'accepted_terminals': terminals,
            'scope': 'Fresh authenticated custody and recorded replay observation only. No fresh replay, no full closure, no deletion permission.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--unit', required=True)
    ap.add_argument('--service-env', type=Path, required=True)
    ap.add_argument('--warehouse-url', default='http://127.0.0.1:5057')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    values = [line.split('=', 1)[1] for line in args.service_env.read_text().splitlines()
              if line.startswith('DATA_GOV_LAKE_TOKEN=')]
    if len(values) != 1:
        raise ValueError('Expected one configured warehouse token')
    token = shlex.split(values[0])[0]

    def query(sql):
        req = urllib.request.Request(args.warehouse_url + '/api/v1/query?' + urllib.parse.urlencode({'sql': sql}),
                                     headers={'Authorization': 'Bearer ' + token})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)['rows']

    result = audit(args.root, args.unit, query)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'unit': result['unit'], 'checks': result['checks'], 'scope': result['scope']}))
    return 0 if all(result['checks'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
