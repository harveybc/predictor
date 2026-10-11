"""One finite coordinator copy/publication cycle, intended for a systemd timer.

Private config contains entries[{host, root}], pin, url and token_file. Remote
and local absolute roots match the evaluator's retained bindings. No fits,
restarts, admission changes or credentials copied to workers.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from tools.i6m_data import file_hash
from tools.i6m_queue import atomic_json
from tools.i6n_search_publish import publish
from tools.i6n_publish import publish as publish_pilot


def sync(entry):
    """Copy without deleting local receipts or touching the remote runner."""
    root = Path(entry['root'])
    if not root.is_absolute() or any(c in str(root) for c in '\n\r'):
        raise ValueError('absolute artifact root required')
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '--protect-args', '--timeout=30',
                    '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                    '--exclude=*.tmp*', '--exclude=RUN.lock', '--exclude=olap/',
                    entry['host']+':'+str(root)+'/', str(root)+'/'],
                   capture_output=True, check=True, timeout=60)


def once(config):
    """Publish each terminal once; failures remain independently retryable."""
    state = dict(state='DELIVERED_CURRENT_TERMINALS', published=0, already_published=0,
                 errors=[], observed_at=datetime.now(timezone.utc).isoformat())
    for entry in config['entries']:
        root = Path(entry['root'])
        try:
            sync(entry)
        except Exception as exc:
            state['errors'].append(dict(root=str(root), error_type=type(exc).__name__, stage='transport'))
            continue
        kind = entry.get('kind', 'search')
        if kind not in ('pilot', 'search'):
            raise ValueError('unknown publication kind')
        paths = [root/'STATUS.json'] if kind == 'pilot' else sorted((root/'candidates').glob('*/STATUS.json'))
        for path in paths:
            candidate = path.parent
            try:
                if json.loads(path.read_text())['state'] != 'COMPLETED':
                    continue
                retained = candidate/'olap'/'STATUS.json'
                if retained.exists():
                    receipt = json.loads(retained.read_text())
                    if (receipt.get('state') == 'PUBLISHED_READBACK_VERIFIED'
                            and receipt.get('terminal_sha256') == file_hash(candidate/'RESULT.json')
                            and receipt.get('destination') == config['url'].rstrip('/')
                            and receipt.get('code_commit') == entry.get('pin', config['pin'])):
                        state['already_published'] += 1
                        continue
                publisher = publish_pilot if kind == 'pilot' else publish
                publisher(candidate, entry.get('pin', config['pin']), config['url'], Path(config['token_file']))
                state['published'] += 1
            except Exception as exc:
                state['errors'].append(dict(candidate=candidate.name, error_type=type(exc).__name__, stage='publish'))
    if state['errors']:
        state['state'] = 'PENDING_DELIVERY'
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    status = args.config.with_suffix('.status.json')
    if args.status:
        print(status.read_text() if status.exists() else '{"state":"NOT_RUN"}')
        return
    with args.config.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = once(json.loads(args.config.read_text()))
        atomic_json(status, result)
        print(json.dumps(result))


if __name__ == '__main__':
    main()
